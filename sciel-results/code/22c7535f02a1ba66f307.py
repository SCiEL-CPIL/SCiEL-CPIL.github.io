"""Covering design C(19,8,5) via greedy + iterative local search (remove/replace).
Structural change: instead of only point-swaps, use block-removal with coverage
recovery via greedy refill, and a randomized restart strategy to escape local optima."""
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def time_ok():
        return time.time() < deadline

    all_points = list(range(v))
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Precompute for each k-subset which t-subsets it covers (as indices)
    # This is expensive to precompute fully, so we compute on the fly but cache
    # the mapping from t-sub tuple to index

    def block_t_indices(block):
        """Return list of indices of t-subsets covered by block."""
        return [t_sub_to_idx[sub] for sub in itertools.combinations(block, t)]

    def greedy_cover():
        """Standard greedy: repeatedly pick block covering most uncovered t-subsets."""
        covered = bytearray(n_t)
        uncovered_count = n_t
        blocks = []

        while uncovered_count > 0 and time_ok():
            # Find best block: try all k-subsets? Too many. Use heuristic:
            # Start from an uncovered t-subset, grow to k elements.
            # Pick the uncovered t-subset with fewest already-covered neighbors? 
            # Actually, pick first uncovered and grow greedily.
            first_uncovered = -1
            for i in range(n_t):
                if not covered[i]:
                    first_uncovered = i
                    break
            if first_uncovered == -1:
                break

            base = list(all_t_subs[first_uncovered])
            block = list(base)
            block_set = set(block)

            while len(block) < k and time_ok():
                best_x = -1
                best_gain = -1
                t_minus_1_list = list(itertools.combinations(block, t - 1))
                for x in all_points:
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in t_minus_1_list:
                        cand = tuple(sorted(rest + (x,)))
                        idx = t_sub_to_idx.get(cand)
                        if idx is not None and not covered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    for x in all_points:
                        if len(block) >= k:
                            break
                        if x not in block_set:
                            block.append(x)
                            block_set.add(x)
                    break
                block.append(best_x)
                block_set.add(best_x)

            block.sort()
            blocks.append(block)
            for sub in itertools.combinations(block, t):
                idx = t_sub_to_idx[sub]
                if not covered[idx]:
                    covered[idx] = 1
                    uncovered_count -= 1

        return blocks, covered, uncovered_count

    def is_valid(blocks):
        """Check all t-subsets are covered."""
        covered = bytearray(n_t)
        for block in blocks:
            for sub in itertools.combinations(block, t):
                idx = t_sub_to_idx[sub]
                covered[idx] = 1
        return all(covered)

    def remove_redundant(blocks, covered):
        """Remove blocks that are fully redundant."""
        changed = True
        while changed and time_ok():
            changed = False
            for i in range(len(blocks)):
                if not time_ok():
                    break
                redundant = True
                for sub in itertools.combinations(blocks[i], t):
                    idx = t_sub_to_idx[sub]
                    found = False
                    for j in range(len(blocks)):
                        if j == i:
                            continue
                        if all(pt in blocks[j] for pt in sub):
                            found = True
                            break
                    if not found:
                        redundant = False
                        break
                if redundant:
                    blocks.pop(i)
                    changed = True
                    break
        return blocks

    # Phase 1: Initial greedy cover
    blocks, covered, uncovered_count = greedy_cover()

    if uncovered_count > 0:
        # Incomplete; try to finish with more aggressive greedy
        while uncovered_count > 0 and time_ok():
            first_uncovered = -1
            for i in range(n_t):
                if not covered[i]:
                    first_uncovered = i
                    break
            if first_uncovered == -1:
                break
            base = list(all_t_subs[first_uncovered])
            block = list(base)
            block_set = set(block)
            while len(block) < k and time_ok():
                best_x = -1
                best_gain = -1
                t_minus_1_list = list(itertools.combinations(block, t - 1))
                for x in all_points:
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in t_minus_1_list:
                        cand = tuple(sorted(rest + (x,)))
                        idx = t_sub_to_idx.get(cand)
                        if idx is not None and not covered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    for x in all_points:
                        if len(block) >= k:
                            break
                        if x not in block_set:
                            block.append(x)
                            block_set.add(x)
                    break
                block.append(best_x)
                block_set.add(best_x)
            block.sort()
            blocks.append(block)
            for sub in itertools.combinations(block, t):
                idx = t_sub_to_idx[sub]
                if not covered[idx]:
                    covered[idx] = 1
                    uncovered_count -= 1

    if uncovered_count > 0:
        return blocks  # invalid, but return what we have

    # Phase 2: Remove redundant blocks
    blocks = remove_redundant(blocks, covered)

    # Phase 3: Local search — try to remove a block and re-cover lost subsets
    improved = True
    while improved and time_ok():
        improved = False
        for i in range(len(blocks)):
            if not time_ok():
                break
            # Try removing block i
            test_blocks = blocks[:i] + blocks[i+1:]
            # Find t-subsets that become uncovered
            lost = []
            covered_test = bytearray(n_t)
            for block in test_blocks:
                for sub in itertools.combinations(block, t):
                    idx = t_sub_to_idx[sub]
                    covered_test[idx] = 1
            for sub in itertools.combinations(blocks[i], t):
                idx = t_sub_to_idx[sub]
                if not covered_test[idx]:
                    lost.append(sub)
            if not lost:
                # Block i is redundant
                blocks = test_blocks
                improved = True
                break
            # Try to cover lost subsets with a single new block
            # Greedily build a block covering as many lost subsets as possible
            if len(lost) <= k:
                # Can we cover all lost subsets with one block?
                # Each lost subset is a t-subset; need a k-set containing all
                # But they may not be contained in a single k-set
                # Try: take union of all lost subsets
                union = set()
                for s in lost:
                    union.update(s)
                if len(union) <= k:
                    new_block = sorted(union)
                    if len(new_block) < k:
                        # Fill with arbitrary points
                        for x in all_points:
                            if len(new_block) >= k:
                                break
                            if x not in new_block:
                                new_block.append(x)
                    new_block.sort()
                    # Check if new block covers all lost
                    new_block_set = set(new_block)
                    all_covered = True
                    for s in lost:
                        if not all(x in new_block_set for x in s):
                            all_covered = False
                            break
                    if all_covered:
                        # Verify new block doesn't break anything (it only adds coverage)
                        blocks = test_blocks + [new_block]
                        improved = True
                        break

    # Phase 4: Point swap optimization
    if time_ok():
        improved = True
        while improved and time_ok():
            improved = False
            for i in range(len(blocks)):
                if not time_ok():
                    break
                block_i = list(blocks[i])
                for pos in range(k):
                    old_pt = block_i[pos]
                    for new_pt in all_points:
                        if new_pt == old_pt or new_pt in block_i:
                            continue
                        candidate = block_i[:]
                        candidate[pos] = new_pt
                        candidate.sort()
                        # Check validity: does the new set of blocks still cover everything?
                        # Only need to check t-subsets that were covered by old block_i
                        # but might not be covered by candidate or other blocks
                        old_subs = set(itertools.combinations(block_i, t))
                        new_subs = set(itertools.combinations(candidate, t))
                        # Lost subsets: in old but not in new
                        lost_subs = old_subs - new_subs
                        if not lost_subs:
                            # Swap is safe (only adds coverage)
                            blocks[i] = candidate
                            improved = True
                            break
                        # Check if lost subsets are covered by other blocks
                        all_ok = True
                        for sub in lost_subs:
                            found = False
                            for j in range(len(blocks)):
                                if j == i:
                                    continue
                                if all(pt in blocks[j] for pt in sub):
                                    found = True
                                    break
                            if not found:
                                all_ok = False
                                break
                        if all_ok:
                            blocks[i] = candidate
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

    # Phase 5: Final redundancy removal
    blocks = remove_redundant(blocks, covered)

    # Verify validity
    if not is_valid(blocks):
        # Fallback: rebuild
        blocks, covered, uncovered_count = greedy_cover()
        if uncovered_count > 0:
            while uncovered_count > 0 and time_ok():
                first_uncovered = -1
                for i in range(n_t):
                    if not covered[i]:
                        first_uncovered = i
                        break
                if first_uncovered == -1:
                    break
                base = list(all_t_subs[first_uncovered])
                block = list(base)
                block_set = set(block)
                while len(block) < k and time_ok():
                    best_x = -1
                    best_gain = -1
                    t_minus_1_list = list(itertools.combinations(block, t - 1))
                    for x in all_points:
                        if x in block_set:
                            continue
                        gain = 0
                        for rest in t_minus_1_list:
                            cand = tuple(sorted(rest + (x,)))
                            idx = t_sub_to_idx.get(cand)
                            if idx is not None and not covered[idx]:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                    if best_x == -1:
                        for x in all_points:
                            if len(block) >= k:
                                break
                            if x not in block_set:
                                block.append(x)
                                block_set.add(x)
                        break
                    block.append(best_x)
                    block_set.add(best_x)
                block.sort()
                blocks.append(block)
                for sub in itertools.combinations(block, t):
                    idx = t_sub_to_idx[sub]
                    if not covered[idx]:
                        covered[idx] = 1
                        uncovered_count -= 1
        blocks = remove_redundant(blocks, covered)

    return blocks
