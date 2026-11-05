"""Combined program: Global vectorized greedy set-cover + Local Search (removal, 2-for-1, 2-for-2, reconstruction) + re-optimization."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 142  # safety margin for 150s limit

    # ---- Precomputation -------------------------------------------------
    all_tsets = list(itertools.combinations(range(v), t))
    n_tsets = len(all_tsets)
    tset_to_id = {ts: i for i, ts in enumerate(all_tsets)}

    # For each (t-1)-subset, the list of t-set IDs it extends to
    t1_to_tsets = {}
    for tid, ts in enumerate(all_tsets):
        for sub in itertools.combinations(ts, t - 1):
            t1_to_tsets.setdefault(sub, []).append(tid)

    # For each (t-1)-subset, the list of points that complete it to a t-set
    # (used to vectorize the gain of adding a point to a block)
    t1_to_points = {}
    for sub, ids in t1_to_tsets.items():
        pts = []
        for tid in ids:
            for p in all_tsets[tid]:
                if p not in sub:
                    pts.append(p)
        t1_to_points[sub] = np.array(pts, dtype=np.int32)

    def tset_id(ts_tuple):
        return tset_to_id[ts_tuple]

    def block_tids(block):
        return [tset_to_id[ts] for ts in itertools.combinations(block, t)]

    # ---- Global vectorized greedy set-cover -----------------------------
    def greedy_global(seed=0):
        rng = random.Random(seed)
        uncovered = np.ones(n_tsets, dtype=np.int32)
        blocks = []

        while uncovered.sum() > 0:
            # Start block with a random uncovered t-set
            start_id = int(rng.choice(np.flatnonzero(uncovered)))
            block = list(all_tsets[start_id])
            block_set = set(block)

            while len(block) < k:
                # gain[p] = number of uncovered t-sets that would be completed
                # by adding point p to the current block
                gain = np.zeros(v, dtype=np.int32)
                for sub in itertools.combinations(block, t - 1):
                    pts = t1_to_points[sub]
                    gain[np.bincount(pts, minlength=v)] += 1
                # mask points already in the block
                for p in block:
                    gain[p] = -1
                cands = np.flatnonzero(gain > 0)
                if cands.size == 0:
                    best_p = int(rng.choice([p for p in range(v) if p not in block_set]))
                else:
                    gmax = gain[cands].max()
                    top = cands[gain[cands] == gmax]
                    best_p = int(rng.choice(top))
                block.append(best_p)
                block_set.add(best_p)

            block.sort()
            blocks.append(block)
            uncovered[np.array(block_tids(block), dtype=np.int64)] = 0

        return blocks

    # ---- Phase 1: multiple greedy restarts ------------------------------
    best_blocks = None
    best_count = float('inf')
    for seed in range(40):
        if time.time() > deadline - 30:
            break
        blocks = greedy_global(seed)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
    if best_blocks is None:
        best_blocks = greedy_global(0)

    # ---- Phase 2: local search ------------------------------------------
    count = np.zeros(n_tsets, dtype=np.int32)

    def update_counts(blocks, delta=1):
        for block in blocks:
            count[np.array(block_tids(block), dtype=np.int64)] += delta

    update_counts(best_blocks, 1)
    current_blocks = list(best_blocks)

    def is_redundant(block):
        for tid in block_tids(block):
            if count[tid] == 1:
                return False
        return True

    def try_remove_redundant(blocks):
        removed = False
        new_blocks = []
        for b in blocks:
            if is_redundant(b):
                update_counts([b], -1)
                removed = True
            else:
                new_blocks.append(b)
        return new_blocks, removed

    def try_swap_2_for_1(blocks):
        n = len(blocks)
        if n < 2:
            return blocks, False
        block_ids = [set(block_tids(b)) for b in blocks]
        pair_scores = []
        for i in range(n):
            if time.time() > deadline - 12:
                break
            for j in range(i + 1, n):
                if time.time() > deadline - 12:
                    break
                overlap = len(block_ids[i] & block_ids[j])
                total = len(block_ids[i] | block_ids[j])
                if total > 0:
                    score = overlap / total
                    if (len(block_ids[i] - block_ids[j]) < 5 or
                            len(block_ids[j] - block_ids[i]) < 5):
                        score += 0.5
                    pair_scores.append((score, i, j))
        pair_scores.sort(reverse=True)
        for _score, i, j in pair_scores[:300]:
            if time.time() > deadline - 12:
                break
            b1, b2 = blocks[i], blocks[j]
            ids_b1, ids_b2 = block_ids[i], block_ids[j]
            u_need = [tid for tid in (ids_b1 | ids_b2)
                      if count[tid] == (1 if tid in ids_b1 else 0) + (1 if tid in ids_b2 else 0)]
            if not u_need:
                update_counts([b1, b2], -1)
                return [b for idx, b in enumerate(blocks) if idx != i and idx != j], True
            p_need = set()
            for tid in u_need:
                p_need.update(all_tsets[tid])
            if len(p_need) > k:
                continue
            new_block = list(p_need)
            new_block_set = set(new_block)
            rng = random.Random(i * 1000 + j)
            candidates = [p for p in range(v) if p not in new_block_set]
            for _ in range(k - len(new_block)):
                if not candidates:
                    break
                best_p, best_gain = -1, -1
                for p in candidates:
                    gain = 0
                    for sub in itertools.combinations(new_block, t - 1):
                        tid = tset_id(tuple(sorted(sub + (p,))))
                        if tid not in ids_b1 and tid not in ids_b2:
                            gain += 1
                    if gain > best_gain:
                        best_gain, best_p = gain, p
                if best_p == -1 or best_gain == 0:
                    best_p = rng.choice(candidates)
                new_block.append(best_p)
                new_block_set.add(best_p)
                candidates.remove(best_p)
            new_block.sort()
            new_ids = set(block_tids(new_block))
            if all(tid in new_ids for tid in u_need):
                update_counts([b1, b2], -1)
                update_counts([new_block], 1)
                nb = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                nb.append(new_block)
                return nb, True
        return blocks, False

    def try_swap_2_for_2(blocks):
        """Replace two blocks with two new blocks that cover their unique t-sets."""
        n = len(blocks)
        if n < 2:
            return blocks, False
        rng = random.Random(int(time.time() * 1000) % 100000)
        pairs = list(itertools.combinations(range(n), 2))
        rng.shuffle(pairs)
        for i, j in pairs[:40]:
            if time.time() > deadline - 12:
                break
            b1, b2 = blocks[i], blocks[j]
            ids_b1 = set(block_tids(b1))
            ids_b2 = set(block_tids(b2))
            u1 = [tid for tid in ids_b1 if count[tid] == 1]
            u2 = [tid for tid in ids_b2 if count[tid] == 1]
            if not u1 and not u2:
                continue
            p1 = set()
            for tid in u1:
                p1.update(all_tsets[tid])
            p2 = set()
            for tid in u2:
                p2.update(all_tsets[tid])
            if len(p1) > k or len(p2) > k:
                continue
            nb1 = _fill_block(list(p1), k, v, rng, ids_b1 | ids_b2)
            nb2 = _fill_block(list(p2), k, v, rng, ids_b1 | ids_b2)
            if nb1 is None or nb2 is None:
                continue
            ids_n1 = set(block_tids(nb1))
            ids_n2 = set(block_tids(nb2))
            if all(tid in ids_n1 for tid in u1) and all(tid in ids_n2 for tid in u2):
                update_counts([b1, b2], -1)
                update_counts([nb1, nb2], 1)
                nb = [b for idx, b in enumerate(blocks) if idx != i and idx != j]
                nb.append(nb1)
                nb.append(nb2)
                return nb, True
        return blocks, False

    def _fill_block(start, k, v, rng, avoid_ids):
        block = list(start)
        block_set = set(block)
        candidates = [p for p in range(v) if p not in block_set]
        for _ in range(k - len(block)):
            if not candidates:
                return None
            best_p, best_gain = -1, -1
            for p in candidates:
                gain = 0
                for sub in itertools.combinations(block, t - 1):
                    tid = tset_id(tuple(sorted(sub + (p,))))
                    if tid not in avoid_ids:
                        gain += 1
                if gain > best_gain:
                    best_gain, best_p = gain, p
            if best_p == -1:
                best_p = rng.choice(candidates)
            block.append(best_p)
            block_set.add(best_p)
            candidates.remove(best_p)
        block.sort()
        return block

    def try_reconstruct_block(blocks, i):
        b = blocks[i]
        ids_b = set(block_tids(b))
        u_need = [tid for tid in ids_b if count[tid] == 1]
        if not u_need:
            return blocks, False
        p_need = set()
        for tid in u_need:
            p_need.update(all_tsets[tid])
        if len(p_need) > k:
            return blocks, False
        rng = random.Random(i * 7919 + int(time.time() % 1000))
        new_block = _fill_block(list(p_need), k, v, rng, ids_b)
        if new_block is None or new_block == b:
            return blocks, False
        new_ids = set(block_tids(new_block))
        if all(tid in new_ids for tid in u_need):
            update_counts([b], -1)
            update_counts([new_block], 1)
            nb = [x for idx, x in enumerate(blocks) if idx != i]
            nb.append(new_block)
            return nb, True
        return blocks, False

    # ---- Local search loop ----------------------------------------------
    improved = True
    stall_count = 0
    for _ in range(120):
        if time.time() > deadline - 18:
            break
        if not improved:
            stall_count += 1
            if stall_count >= 3:
                if time.time() < deadline - 22 and current_blocks:
                    idx = random.Random(int(time.time() * 1000) % 10000).randrange(len(current_blocks))
                    new_blocks, pert = try_reconstruct_block(current_blocks, idx)
                    if pert:
                        current_blocks = new_blocks
                        improved = True
                        stall_count = 0
                        for _pass in range(3):
                            if time.time() > deadline - 18:
                                break
                            new_blocks, rem = try_remove_redundant(current_blocks)
                            if rem:
                                current_blocks = new_blocks
                                improved = True
                            else:
                                break
                if not improved:
                    break
                continue
            else:
                break
        else:
            stall_count = 0
        improved = False

        for _pass in range(3):
            if time.time() > deadline - 18:
                break
            new_blocks, rem = try_remove_redundant(current_blocks)
            if rem:
                current_blocks = new_blocks
                improved = True
            else:
                break

        new_blocks, swp = try_swap_2_for_1(current_blocks)
        if swp:
            current_blocks = new_blocks
            improved = True
            for _pass in range(2):
                if time.time() > deadline - 18:
                    break
                new_blocks, rem = try_remove_redundant(current_blocks)
                if rem:
                    current_blocks = new_blocks
                    improved = True
                else:
                    break

        new_blocks, swp2 = try_swap_2_for_2(current_blocks)
        if swp2:
            current_blocks = new_blocks
            improved = True
            for _pass in range(2):
                if time.time() > deadline - 18:
                    break
                new_blocks, rem = try_remove_redundant(current_blocks)
                if rem:
                    current_blocks = new_blocks
                    improved = True
                else:
                    break

        if current_blocks and time.time() < deadline - 18:
            for _ in range(3):
                if time.time() > deadline - 18:
                    break
                idx = random.Random(int(time.time() * 10000) % 100000).randrange(len(current_blocks))
                new_blocks, swp = try_reconstruct_block(current_blocks, idx)
                if swp:
                    current_blocks = new_blocks
                    improved = True
                    break

    # ---- Final verification + polish ------------------------------------
    covered = set()
    for block in current_blocks:
        covered.update(block_tids(block))
    if len(covered) < n_tsets:
        return best_blocks

    if time.time() < deadline - 8:
        for _pass in range(5):
            if time.time() > deadline - 6:
                break
            new_blocks, rem = try_remove_redundant(current_blocks)
            if rem:
                current_blocks = new_blocks
            else:
                break

    # ---- Re-optimization pass (second local search on the best solution) ----
    if time.time() < deadline - 25 and len(current_blocks) < best_count:
        count = np.zeros(n_tsets, dtype=np.int32)
        update_counts(current_blocks, 1)
        rb = list(current_blocks)
        r_improved = True
        r_stall = 0
        for _ in range(60):
            if time.time() > deadline - 8:
                break
            if not r_improved:
                r_stall += 1
                if r_stall >= 3:
                    break
                continue
            r_stall = 0
            r_improved = False
            for _pass in range(2):
                if time.time() > deadline - 8:
                    break
                nb, rem = try_remove_redundant(rb)
                if rem:
                    rb = nb
                    r_improved = True
                else:
                    break
            nb, swp = try_swap_2_for_1(rb)
            if swp:
                rb = nb
                r_improved = True
            nb, swp2 = try_swap_2_for_2(rb)
            if swp2:
                rb = nb
                r_improved = True
        cov = set()
        for block in rb:
            cov.update(block_tids(block))
        if len(cov) == n_tsets and len(rb) < len(current_blocks):
            current_blocks = rb

    return current_blocks
# EVOLVE-BLOCK-END
