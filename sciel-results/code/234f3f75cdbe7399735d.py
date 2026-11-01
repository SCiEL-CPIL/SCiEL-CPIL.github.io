import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 142  # safety margin before 150s kill

    def now():
        return time.time()

    # ---- Precomputation ----
    t_subsets = list(itertools.combinations(range(v), t))
    t_subsets_set = {tuple(s): i for i, s in enumerate(t_subsets)}
    num_tsubs = len(t_subsets)

    k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(k_subsets)

    # For each block: which t-subsets it covers (list) + bitmask
    block_to_tsubs = []
    block_bitmasks = [0] * num_blocks
    for i, block in enumerate(k_subsets):
        covered = []
        mask = 0
        for tsub in itertools.combinations(block, t):
            idx = t_subsets_set[tsub]
            mask |= (1 << idx)
            covered.append(idx)
        block_bitmasks[i] = mask
        block_to_tsubs.append(covered)

    # For each t-subset: which blocks cover it
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for i, tsubs in enumerate(block_to_tsubs):
        for s in tsubs:
            tsub_to_blocks[s].append(i)
    tsub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_to_blocks]
    tsub_to_blocks_set = [set(lst) for lst in tsub_to_blocks]

    # ---- Densest-subgraph greedy core (dynamic unique-coverage weights) ----
    # score(block) = number of t-subsets it covers that are still uncovered.
    # Equivalent to a bipartite block/tsub graph where we repeatedly pick the
    # block of max "unique" edge count, then mark those edges as covered
    # (penalizing redundant coverage). This selects a high-leverage core.
    cover_count = np.zeros(num_tsubs, dtype=np.int64)
    uncovered_mask = (1 << num_tsubs) - 1
    selected = []
    selected_set = set()
    selected_block_mask = 0  # union of blocks chosen (for redundancy checks)

    # Maintain per-block current unique-gain; update incrementally.
    gain = np.zeros(num_blocks, dtype=np.int32)
    for i in range(num_blocks):
        gain[i] = block_bitmasks[i].bit_count()  # all uncovered initially

    while uncovered_mask != 0 and now() < deadline - 30:
        # Pick block with max unique gain (tie-break by higher total degree / rarity)
        best_b = int(np.argmax(gain))
        best_gain = gain[best_b]
        if best_gain <= 0:
            # Fallback exhaustive scan
            best_gain = 0
            for b in range(num_blocks):
                g = (block_bitmasks[b] & uncovered_mask).bit_count()
                if g > best_gain:
                    best_gain = g
                    best_b = b
        if best_b < 0 or best_gain <= 0:
            break

        selected.append(best_b)
        selected_set.add(best_b)
        selected_block_mask |= block_bitmasks[best_b]

        # Update: this block's tsubs become covered. Recompute gains for
        # blocks that share any of these tsubs.
        newly = block_to_tsubs[best_b]
        for s in newly:
            cover_count[s] += 1
            if cover_count[s] == 1:
                # s just became covered: decrement gain of all other blocks covering s
                for b in tsub_to_blocks[s]:
                    if b != best_b:
                        gain[b] -= 1
        # best_b's own gain is irrelevant now
        gain[best_b] = 0

    # ---- Repair: cover any remaining uncovered tsubs (should be none) ----
    while uncovered_mask != 0 and now() < deadline - 8:
        # pick a random uncovered tsub, add a block covering it
        # find an uncovered tsub
        # (uncovered_mask not updated above; recompute from cover_count)
        # recompute uncovered list
        unc = np.where(cover_count == 0)[0]
        if len(unc) == 0:
            break
        s = int(unc[random.randint(0, len(unc) - 1)])
        cands = tsub_to_blocks_np[s]
        # choose candidate with max unique gain
        best_b = -1
        best_g = -1
        for b in cands:
            g = (block_bitmasks[b] & ((1 << num_tsubs) - 1)).bit_count()
            # unique gain:
            ug = 0
            for ss in block_to_tsubs[b]:
                if cover_count[ss] == 0:
                    ug += 1
            if ug > best_g:
                best_g = ug
                best_b = int(b)
        if best_b < 0:
            break
        selected.append(best_b)
        selected_set.add(best_b)
        for ss in block_to_tsubs[best_b]:
            cover_count[ss] += 1

    # ---- Local-search minimization ----
    def is_redundant(b_idx):
        for s in block_to_tsubs[b_idx]:
            if cover_count[s] == 1:
                return False
        return True

    def find_covering_block(tsubs_to_cover, exclude):
        if not tsubs_to_cover:
            return None
        first_s = tsubs_to_cover[0]
        best = None
        best_score = -1
        for c in tsub_to_blocks_np[first_s]:
            if c in exclude:
                continue
            ok = True
            for s in tsubs_to_cover[1:]:
                if c not in tsub_to_blocks_set[s]:
                    ok = False
                    break
            if ok:
                # score: prefer blocks with lower total degree (less redundancy added)
                score = -len(block_to_tsubs[c])
                if score > best_score:
                    best_score = score
                    best = int(c)
        return best

    # Initial full redundancy pass
    for b in list(selected):
        if is_redundant(b):
            selected.remove(b)
            selected_set.discard(b)
            for s in block_to_tsubs[b]:
                cover_count[s] -= 1

    improved = True
    while improved and now() < deadline - 6:
        improved = False

        # 2a: redundancy removal
        for b in list(selected):
            if is_redundant(b):
                selected.remove(b)
                selected_set.discard(b)
                for s in block_to_tsubs[b]:
                    cover_count[s] -= 1
                improved = True

        # 2b: 1-for-1 swaps
        unique_counts = {}
        unique_tsubs = {}
        for b in selected:
            u = [s for s in block_to_tsubs[b] if cover_count[s] == 1]
            unique_counts[b] = len(u)
            unique_tsubs[b] = u
        if unique_counts:
            min_u = min(unique_counts.values())
            cands_out = [b for b, c in unique_counts.items() if c == min_u]
            random.shuffle(cands_out)
            for b_out in cands_out:
                s_uniq = unique_tsubs[b_out]
                if not s_uniq:
                    selected.remove(b_out)
                    selected_set.discard(b_out)
                    for s in block_to_tsubs[b_out]:
                        cover_count[s] -= 1
                    improved = True
                    break
                b_in = find_covering_block(s_uniq, selected_set)
                if b_in is not None:
                    selected.remove(b_out)
                    selected_set.discard(b_out)
                    selected.append(b_in)
                    selected_set.add(b_in)
                    for s in block_to_tsubs[b_out]:
                        cover_count[s] -= 1
                    for s in block_to_tsubs[b_in]:
                        cover_count[s] += 1
                    improved = True
                    break

        # 2c: 2-for-1 moves
        n_sel = len(selected)
        if n_sel >= 2 and now() < deadline - 6:
            max_pairs = min(300, n_sel * (n_sel - 1) // 2)
            tried = 0
            while tried < max_pairs and now() < deadline - 6:
                i = random.randint(0, n_sel - 1)
                j = random.randint(0, n_sel - 1)
                if i == j:
                    continue
                b1 = selected[i]
                b2 = selected[j]
                tried += 1
                u1 = [s for s in block_to_tsubs[b1] if cover_count[s] == 1]
                u2 = [s for s in block_to_tsubs[b2] if cover_count[s] == 1]
                allu = set(u1) | set(u2)
                if not allu:
                    continue
                first_s = next(iter(allu))
                found = None
                for c in tsub_to_blocks_np[first_s]:
                    if c == b1 or c == b2:
                        continue
                    ok = True
                    for s in allu:
                        if c not in tsub_to_blocks_set[s]:
                            ok = False
                            break
                    if ok:
                        found = int(c)
                        break
                if found is not None:
                    selected.remove(b1)
                    selected.remove(b2)
                    selected_set.discard(b1)
                    selected_set.discard(b2)
                    selected.append(found)
                    selected_set.add(found)
                    for s in block_to_tsubs[b1]:
                        cover_count[s] -= 1
                    for s in block_to_tsubs[b2]:
                        cover_count[s] -= 1
                    for s in block_to_tsubs[found]:
                        cover_count[s] += 1
                    improved = True
                    break

        # 2d: add-then-minimize (escape local minima)
        if now() < deadline - 12:
            min_cov = int(cover_count.min())
            if min_cov <= 2:
                low = np.where(cover_count == min_cov)[0]
                if len(low) > 0:
                    s_rand = int(low[random.randint(0, len(low) - 1)])
                    cands = tsub_to_blocks_np[s_rand]
                    if len(cands) > 0:
                        # pick candidate maximizing unique gain
                        best_b = -1
                        best_g = -1
                        for b in cands:
                            ug = 0
                            for ss in block_to_tsubs[b]:
                                if cover_count[ss] == 0:
                                    ug += 1
                            if ug > best_g:
                                best_g = ug
                                best_b = int(b)
                        if best_b is not None and best_b not in selected_set:
                            selected.append(best_b)
                            selected_set.add(best_b)
                            for s in block_to_tsubs[best_b]:
                                cover_count[s] += 1
                            removed = 0
                            for b in list(selected):
                                if is_redundant(b):
                                    selected.remove(b)
                                    selected_set.discard(b)
                                    for s in block_to_tsubs[b]:
                                        cover_count[s] -= 1
                                    removed += 1
                            if removed >= 1:
                                improved = True
                            else:
                                # revert
                                selected.remove(best_b)
                                selected_set.discard(best_b)
                                for s in block_to_tsubs[best_b]:
                                    cover_count[s] -= 1

    # ---- Final: ensure valid (cover all tsubs) ----
    # Recompute coverage from selected to be safe
    final_cover = np.zeros(num_tsubs, dtype=np.int64)
    for b in selected:
        for s in block_to_tsubs[b]:
            final_cover[s] += 1
    unc = np.where(final_cover == 0)[0]
    for s in unc:
        cands = tsub_to_blocks_np[s]
        if len(cands) > 0:
            b_add = int(cands[random.randint(0, len(cands) - 1)])
            if b_add not in selected_set:
                selected.append(b_add)
                selected_set.add(b_add)

    # ---- Build result ----
    result = [list(k_subsets[i]) for i in selected]
    seen = set()
    dedup = []
    for blk in result:
        key = tuple(blk)
        if key not in seen:
            seen.add(key)
            dedup.append(blk)
    return dedup
