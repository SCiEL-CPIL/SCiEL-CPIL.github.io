"""Global greedy set-cover for C(v,k,t) with numpy-vectorised gain tracking and a deadline guard."""
import itertools
import time
import numpy as np


def construct(v, k, t):
    start = time.time()
    deadline = start + 105.0  # safety margin against 120/150s kills

    # Enumerate all t-subsets and assign each an integer id.
    all_combos = list(itertools.combinations(range(v), t))
    n_t = len(all_combos)
    combo_to_id = {c: i for i, c in enumerate(all_combos)}

    uncovered = np.ones(n_t, dtype=np.int32)

    # Precompute, for each (v-1)-subset S (the "rest" when adding x), the list of
    # t-subset-ids that contain S and the point x (i.e., c = S + (x,)).
    # For t = 3: (v-1)-subsets are 2-subsets, so v choose 2 of them.
    # rest_to_points: for each (v-1)-subset, a list of (x, t_id) pairs.
    # We also need, for each point x, the list of (v-1)-subsets that do NOT contain x.
    # But we'll handle that differently: for each x, precompute the list of t-ids
    # that would be covered if x is added to a given block.

    # Actually, the gain of adding point x to a current block B is:
    #   sum over all (t-1)-subsets R of B: number of uncovered t-subsets that contain R and x.
    # So for each (t-1)-subset R and point x not in R, precompute the list of t-ids
    # that contain R and x. For t=3, R is a 2-subset.

    # For general t, we need for each (t-1)-subset R and each x not in R:
    #   t_ids_for[R][x] = list of t-subset-ids c such that c contains R and c contains x.
    # Since c must be a t-subset containing R and x, and |R|=t-1, |x|=1, we need |R ∪ {x}| = t.
    # So c = R ∪ {x} is the unique t-subset? No! c is a t-subset containing R and x, but R has t-1 elements,
    # so R ∪ {x} has t elements (if x not in R), so c = R ∪ {x} is unique.

    # Wait, that's only for the specific case. Let me re-think.
    # For a block B, gain(x) = number of uncovered t-subsets c such that c ⊆ B ∪ {x} and c contains x.
    # c contains x and |c| = t, and c ⊆ B ∪ {x}. Since x ∉ B, c = R ∪ {x} where R ⊆ B and |R| = t-1.
    # So gain(x) = sum over (t-1)-subsets R of B: [uncovered[R ∪ {x}]]

    # So for each (t-1)-subset R and each x not in R, we need the id of R ∪ {x} as a t-subset.
    # That's just one t-subset! So t_id_for[R][x] = combo_to_id[frozenset(R) | {x}]

    # For t=3: (t-1)=2 subsets, so for each pair (a,b) and each x not in {a,b}, 
    # the t-subset is {a,b,x}. That's C(v,3) total, which matches.

    # Precompute: for each (t-1)-subset R (as a tuple), a dict or array mapping x -> t_id.
    # For t=3, there are C(v,2) pairs. For each pair, v-2 possible x values.

    if t == 3:
        # pairs: list of (pair_tuple, {x: t_id})
        # Use numpy for the uncovered array.
        # For each pair (a,b), we have v-2 x values. Store as a 2D array?
        # pair_to_x_ids: for each pair index, an array of t_ids indexed by x.
        # Let's use a dict: pair -> array of length v with -1 for x in pair.
        
        pairs = list(itertools.combinations(range(v), 2))
        pair_idx = {p: i for i, p in enumerate(pairs)}
        n_pairs = len(pairs)
        # pair_t_ids[i][x] = t_id of {pairs[i][0], pairs[i][1], x} if x not in pairs[i], else -1
        pair_t_ids = np.full((n_pairs, v), -1, dtype=np.int32)
        for i, (a, b) in enumerate(pairs):
            for x in range(v):
                if x != a and x != b:
                    pair_t_ids[i, x] = combo_to_id[(min(a,b,x), min(max(a,b),x), max(a,b,x))]
        # Wait, combo_to_id keys are sorted tuples. Let's fix that.
        for i, (a, b) in enumerate(pairs):
            for x in range(v):
                if x != a and x != b:
                    triple = tuple(sorted((a, b, x)))
                    pair_t_ids[i, x] = combo_to_id[triple]

        blocks = []
        while np.any(uncovered) and time.time() < deadline:
            # Pick the first uncovered t-subset to start a new block
            first_uncovered = np.argmax(uncovered)
            # Get the points of that t-subset
            start_combo = all_combos[first_uncovered]
            block = list(start_combo)
            
            while len(block) < k:
                if time.time() >= deadline:
                    break
                # Compute gain for each candidate x not in block
                best_x = -1
                best_gain = -1
                block_set = set(block)
                # For each x not in block, compute gain
                # gain(x) = sum over (t-1)-subsets R of block: uncovered[R ∪ {x}]
                # For t=3, (t-1)-subsets are pairs. For each pair (a,b) in block,
                # gain contribution = uncovered[pair_t_ids[pair_idx[(a,b)]][x]]
                
                # Precompute list of pair indices for pairs in block
                block_pairs = list(itertools.combinations(block, 2))
                block_pair_indices = [pair_idx[p] for p in block_pairs]
                
                # Vectorized: for each candidate x, gain = sum of uncovered[pair_t_ids[pi][x]] for pi in block_pair_indices
                # candidates = [x for x in range(v) if x not in block_set]
                # This is O(v * |block_pairs|) per iteration, which is fine for v=24.
                
                candidates = [x for x in range(v) if x not in block_set]
                if not candidates:
                    break
                
                best_gain = -1
                best_x = candidates[0]
                for x in candidates:
                    gain = 0
                    for pi in block_pair_indices:
                        tid = pair_t_ids[pi, x]
                        if tid >= 0:
                            gain += uncovered[tid]
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                if best_gain <= 0:
                    # No more gain, break
                    break
                block.append(best_x)
            
            if len(block) >= k:
                block.sort()
                blocks.append(tuple(block))
                # Mark all t-subsets of this block as covered
                for c in itertools.combinations(block, t):
                    uncovered[combo_to_id[c]] = 0
            elif time.time() >= deadline:
                break

        # Final check: if we didn't cover everything within deadline, we need to return something valid.
        # The task requires valid output. Let's check if we have a valid covering.
        if not np.all(uncovered == 0):
            # Fallback: use the original greedy to finish, but this may be slow.
            # Actually, we should have covered everything if we had time. Let's do a quick finish.
            while np.any(uncovered) and time.time() < deadline:
                first_uncovered = np.argmax(uncovered)
                start_combo = all_combos[first_uncovered]
                block = list(start_combo)
                block_set = set(block)
                while len(block) < k:
                    best_x = -1
                    best_gain = -1
                    block_pairs = list(itertools.combinations(block, 2))
                    block_pair_indices = [pair_idx[p] for p in block_pairs]
                    candidates = [x for x in range(v) if x not in block_set]
                    if not candidates:
                        break
                    for x in candidates:
                        gain = 0
                        for pi in block_pair_indices:
                            tid = pair_t_ids[pi, x]
                            if tid >= 0:
                                gain += uncovered[tid]
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                    if best_gain <= 0:
                        break
                    block.append(best_x)
                if len(block) >= k:
                    block.sort()
                    blocks.append(tuple(block))
                    for c in itertools.combinations(block, t):
                        uncovered[combo_to_id[c]] = 0

        # Convert to list of lists
        return [list(b) for b in blocks]
    else:
        # Fallback for general t (not expected for this task, but safe)
        uncovered_set = set(all_combos)
        blocks = []
        while uncovered_set and time.time() < deadline:
            block = list(min(uncovered_set))
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered_set)
                block.append(max((x for x in range(v) if x not in block), key=gain))
            block.sort()
            blocks.append(block)
            uncovered_set.difference_update(itertools.combinations(block, t))
        return blocks
