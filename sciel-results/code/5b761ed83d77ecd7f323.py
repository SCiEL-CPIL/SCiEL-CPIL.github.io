import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    # Precompute all t-subsets and assign indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_index = {s: i for i, s in enumerate(all_t_subsets)}

    # Precompute all k-blocks and the t-subsets each covers
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k = len(all_k_blocks)

    # block_covers[b] = list of t-subset indices covered by block b
    block_covers = []
    for b in all_k_blocks:
        covers = [t_index[sub] for sub in itertools.combinations(b, t)]
        block_covers.append(covers)

    # t_covers[s] = list of block indices that cover t-subset s
    t_covers = [[] for _ in range(n_t)]
    for bi, covers in enumerate(block_covers):
        for si in covers:
            t_covers[si].append(bi)

    # Convert block_covers to numpy arrays for fast ops
    # We'll track uncovered as a boolean numpy array of length n_t
    # gain computation: for candidate point x, count uncovered t-subsets S where S\{x} subset of current_block and x not in block

    # Precompute: for each point x and each (t-1)-subset, the t-subset formed
    # Actually, for a current block B (size m < k), adding x covers t-subsets S where
    # S = (t-1 subset of B) + x, and x not in B.
    # We can precompute for each x, a mapping from frozenset of (t-1) elements to the t-subset index.
    # But that's large. Instead, compute on the fly but vectorize.

    # Better: precompute for each point x, all t-subsets containing x, as a dict from (t-1)-subset to t-index.
    # For v=19, t=5: each x is in C(18,4)=3060 t-subsets. For each such t-subset S, S\{x} is a 4-subset.
    # So for point x, we have a dict: 4-subset (of the other 18 points) -> t-index.
    # When block B has size m, the (t-1)-subsets of B are C(m, t-1). For each such 4-subset Q, if Q is a subset of (all points except x), look up t_index of Q+{x}.
    # This is fast: C(m,4) lookups per candidate x.

    # Precompute: for each x, a dict from 4-subset (tuple) to t-subset index
    x_to_tsub = []
    for x in range(v):
        d = {}
        for s in range(n_t):
            if x in all_t_subsets[s]:
                # remove x from s to get the (t-1)-subset
                others = tuple(y for y in all_t_subsets[s] if y != x)
                d[others] = s
        x_to_tsub.append(d)

    best_blocks = None
    best_count = float('inf')

    # Min-cut / dual price analysis: use a greedy dual to estimate "hardness" of t-subsets.
    # We'll run a simple primal-dual: start with all dual prices 0, repeatedly pick the block with best (coverage / sum of dual prices of its t-subsets), 
    # then increase dual prices of newly-covered t-subsets. The final dual prices indicate which t-subsets were "expensive."
    # Actually, simpler: run the greedy set cover (pick block covering most uncovered), record the order in which t-subsets get covered.
    # T-subsets covered later are "harder." Use that order to bias restarts.

    def compute_hardness_order():
        """Run a standard greedy set cover and return t-subsets in order of being covered (later = harder)."""
        uncovered = np.ones(n_t, dtype=bool)
        remaining = set(range(n_t))
        order = []
        while remaining:
            # Pick block covering most uncovered
            best_b = -1
            best_cnt = -1
            for bi in range(n_k):
                cnt = 0
                for si in block_covers[bi]:
                    if si in remaining:
                        cnt += 1
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_b = bi
            if best_cnt <= 0:
                break
            # Cover all t-subsets in block best_b
            for si in block_covers[best_b]:
                if si in remaining:
                    order.append(si)
                    remaining.discard(si)
        return order

    # Precompute hardness order once
    hardness_order = compute_hardness_order()
    # hardness_rank[s] = position in order (higher = harder)
    hardness_rank = np.zeros(n_t, dtype=int)
    for i, s in enumerate(hardness_order):
        hardness_rank[s] = i

    def greedy_with_softmax(seed, temp=0.0, hard_bias=0.0):
        """
        Greedy construction.
        temp: temperature for softmax over gains (0 = hard max, >0 = soft).
        hard_bias: weight to add to gain of t-subsets that are "hard" (from min-cut analysis).
        Returns list of blocks (tuples).
        """
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        blocks = []

        # Precompute hardness gain bonus: for each t-subset, a bonus if hard
        if hard_bias > 0:
            hard_bonus = np.zeros(n_t)
            hard_bonus[hardness_order] = np.linspace(0, 1, len(hardness_order))
        else:
            hard_bonus = None

        while uncovered.any():
            if time.time() > deadline:
                break

            # Pick a starting t-subset: if hard_bias, prefer hard ones; else random
            uncovered_idx = np.where(uncovered)[0]
            if hard_bias > 0 and len(uncovered_idx) > 1:
                # Weight by hardness
                weights = hardness_rank[uncovered_idx] + 1.0
                weights /= weights.sum()
                start_s = int(rng.choices(uncovered_idx.tolist(), weights=weights.tolist())[0])
            else:
                start_s = int(rng.choice(uncovered_idx.tolist()))

            block = list(all_t_subsets[start_s])
            block_set = set(block)

            while len(block) < k:
                if time.time() > deadline:
                    break

                # Compute gain for each candidate point
                block_arr = np.array(block, dtype=int)
                # (t-1)-subsets of block
                t_minus_1_subs = list(itertools.combinations(block, t - 1))

                gains = np.zeros(v, dtype=float)
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0.0
                    d = x_to_tsub[x]
                    for sub in t_minus_1_subs:
                        si = d.get(sub, -1)
                        if si >= 0 and uncovered[si]:
                            gain += 1.0
                    if hard_bonus is not None:
                        # Add bonus for hard t-subsets that would be covered
                        # (approximate: scale gain by hardness)
                        pass
                    gains[x] = gain

                # Softmax selection
                max_gain = gains.max()
                if max_gain <= 0:
                    # No gain, pick random
                    candidates = [x for x in range(v) if x not in block_set]
                    x = rng.choice(candidates)
                else:
                    if temp > 0:
                        # Softmax
                        exp_gains = np.exp((gains - max_gain) / temp)
                        probs = exp_gains / exp_gains.sum()
                        # Only consider non-zero gain points
                        valid = gains > 0
                        if not valid.any():
                            candidates = [x for x in range(v) if x not in block_set]
                            x = rng.choice(candidates)
                        else:
                            probs = probs * valid
                            probs /= probs.sum()
                            x = int(rng.choices(range(v), weights=probs.tolist())[0])
                    else:
                        # Hard max with random tie-breaking
                        max_val = gains.max()
                        candidates = np.where(gains == max_val)[0]
                        x = int(rng.choice(candidates.tolist()))

                block.append(x)
                block_set.add(x)

            block.sort()
            block_tuple = tuple(block)
            blocks.append(block_tuple)

            # Update uncovered
            for sub in itertools.combinations(block_tuple, t):
                uncovered[t_index[sub]] = False

        return blocks

    def local_search(blocks, seed):
        """Local search: remove redundant, replace blocks."""
        rng = random.Random(seed)
        current = list(blocks)

        def compute_counts(blk_list):
            counts = np.zeros(n_t, dtype=int)
            for b in blk_list:
                for sub in itertools.combinations(b, t):
                    counts[t_index[sub]] += 1
            return counts

        improved = True
        while improved and time.time() < deadline:
            improved = False
            counts = compute_counts(current)

            # 1. Remove redundant blocks
            for i in range(len(current)):
                b = current[i]
                unique = 0
                for sub in itertools.combinations(b, t):
                    if counts[t_index[sub]] == 1:
                        unique += 1
                        break
                if unique == 0:
                    current.pop(i)
                    improved = True
                    break

            if improved:
                continue

            # 2. Replace blocks: for each block, find its unique t-subsets.
            # If the unique t-subsets can all be covered by a single different block, try replacing.
            counts = compute_counts(current)
            block_uniques = []
            for i, b in enumerate(current):
                unique_subs = []
                for sub in itertools.combinations(b, t):
                    if counts[t_index[sub]] == 1:
                        unique_subs.append(sub)
                if unique_subs:
                    block_uniques.append((i, b, unique_subs))

            # Sort by number of unique subsets (fewer = easier to replace)
            block_uniques.sort(key=lambda x: len(x[2]))

            for i, b, unique_subs in block_uniques:
                if time.time() > deadline:
                    break

                # Find a replacement block that covers all unique_subs
                # A block B' covers S if S subset B'
                # For each unique sub S, B' must contain all elements of S.
                # So B' must be a superset of union(unique_subs).
                union_set = set()
                for S in unique_subs:
                    union_set.update(S)

                if len(union_set) > k:
                    continue

                # Find candidate blocks that contain union_set
                # Precompute: for each k-block, its set of elements
                # We'll search among all_k_blocks
                found = False
                # Try random candidates first
                remaining = [x for x in range(v) if x not in union_set]
                need = k - len(union_set)
                if need < 0 or len(remaining) < need:
                    continue

                # Try a few random replacements
                for _ in range(20):
                    extra = rng.sample(remaining, need)
                    new_block = tuple(sorted(list(union_set) + extra))
                    if new_block == b:
                        continue
                    if new_block in current:
                        continue
                    # Check it covers all unique_subs (it does by construction since union_set covers them)
                    # But also check it doesn't break coverage of other t-subsets? No, we're replacing, so we lose b's coverage.
                    # We need to ensure that after replacement, all t-subsets still covered.
                    # The unique_subs of b are covered by new_block. Other t-subsets covered by b were covered by other blocks too (since they're not unique).
                    # So replacement is safe.
                    current[i] = new_block
                    improved = True
                    found = True
                    break

                if found:
                    break

        return current

    # Run multiple restarts with varying parameters
    for seed in range(200):
        if time.time() > deadline:
            break

        # Vary temperature and hard_bias
        temp = 0.0 if seed % 3 == 0 else (0.5 if seed % 3 == 1 else 1.0)
        hard_bias = 0.5 if seed % 4 == 0 else 0.0

        blocks = greedy_with_softmax(seed, temp=temp, hard_bias=hard_bias)

        if len(blocks) == 0:
            continue

        # Check validity
        counts = np.zeros(n_t, dtype=int)
        for b in blocks:
            for sub in itertools.combinations(b, t):
                counts[t_index[sub]] += 1
        if not counts.all():
            continue

        # Local search
        refined = local_search(blocks, seed)

        if len(refined) < best_count:
            best_count = len(refined)
            best_blocks = refined
            # Print progress (optional)
            # print(f"seed={seed}, count={best_count}")

    if best_blocks is None:
        # Fallback: return a valid covering from a simple greedy
        blocks = greedy_with_softmax(0, temp=0.0, hard_bias=0.0)
        if blocks:
            return [list(b) for b in blocks]
        return []

    return [list(b) for b in best_blocks]
