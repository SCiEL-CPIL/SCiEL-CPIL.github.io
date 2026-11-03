"""Optimized greedy covering for C(v,k,t) with precomputation and deadline guard."""
# EVOLVE-BLOCK-START
import itertools
import time
import sys


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    # Precompute all t-subsets as frozensets for fast membership
    all_t_subsets = [frozenset(comb) for comb in itertools.combinations(range(v), t)]
    n_total = len(all_t_subsets)

    # For each t-subset, precompute the set of (v - t) "complement positions"
    # Actually, we need: for a block B, which t-subsets does it cover?
    # A block covers a t-subset S iff S ⊆ B.
    # So for a block B, it covers all t-subsets that are subsets of B.

    # Strategy: Greedy set cover with precomputed incidence.
    # uncovered[i] = True if t-subset i is not yet covered.
    uncovered = bytearray(b'\x01') * n_total  # all uncovered initially
    uncovered_count = n_total

    # Precompute for each point x, and for each (t-1)-subset of other points,
    # which t-subsets would be covered if x is added.
    # gain(x, block) = number of uncovered t-subsets S such that S ⊆ (block ∪ {x})
    #                 = number of uncovered S where S ⊆ block ∪ {x} and x ∈ S
    #                 = number of uncovered S where (S \ {x}) ⊆ block and x ∈ S

    # For efficiency: for a given block B and candidate x not in B,
    # gain = count of uncovered t-subsets S with x ∈ S and (S \\ {x}) ⊆ B
    #       = count of uncovered (t-1)-subsets R ⊆ B such that R ∪ {x} is uncovered.

    # Precompute: for each x in 0..v-1, and each (t-1)-subset R of {0..v-1} \ {x},
    # map R to the index of R ∪ {x} in all_t_subsets.
    # This is large: v * C(v-1, t-1). For v=19, t=4: 19 * C(18,3) = 19 * 816 = 15504. Fine.

    # Build a mapping: for each x, a dict from frozenset (t-1)-subset R (not containing x)
    # to the index of R ∪ {x} in all_t_subsets.
    subset_index = {}
    for idx, s in enumerate(all_t_subsets):
        subset_index[s] = idx

    # For each x, precompute a list of (R_frozenset, t_subset_index) pairs
    # where R is a (t-1)-subset not containing x.
    # Actually, we'll compute gain on the fly but with a precomputed lookup.

    # Alternative: precompute for each x, a dict mapping frozenset R (size t-1, x not in R)
    # to the index of R ∪ {x}.
    x_to_tsubset_idx = [dict() for _ in range(v)]
    for x in range(v):
        for R in itertools.combinations(range(v), t - 1):
            if x in R:
                continue
            Rf = frozenset(R)
            S = Rf | {x}
            idx = subset_index.get(S)
            if idx is not None:
                x_to_tsubset_idx[x][Rf] = idx

    # Now greedy:
    blocks = []
    while uncovered_count > 0:
        if time.time() > deadline:
            break  # Return best partial (but must be valid... we need to ensure validity)
        
        # Find the first uncovered t-subset to seed a new block
        # For speed, scan uncovered
        seed_idx = -1
        for i in range(n_total):
            if uncovered[i]:
                seed_idx = i
                break
        if seed_idx == -1:
            break
        
        seed = list(all_t_subsets[seed_idx])
        block = list(seed)
        block_set = set(block)

        # Greedily add points to maximize gain
        while len(block) < k:
            if time.time() > deadline:
                break
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                # Compute gain: number of uncovered t-subsets S with x in S and (S\{x}) ⊆ block
                # S\{x} is a (t-1)-subset of block
                gain = 0
                x_dict = x_to_tsubset_idx[x]
                for R in itertools.combinations(block, t - 1):
                    Rf = frozenset(R)
                    if Rf in x_dict:
                        idx = x_dict[Rf]
                        if uncovered[idx]:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x == -1:
                break
            block.append(best_x)
            block_set.add(best_x)

        # Sort and add block
        block.sort()
        blocks.append(block)
        block_set = set(block)

        # Update uncovered
        for S in itertools.combinations(block, t):
            idx = subset_index[frozenset(S)]
            if uncovered[idx]:
                uncovered[idx] = 0
                uncovered_count -= 1

    # If we timed out, the covering may be incomplete. 
    # We need to return a valid covering. If incomplete, we must finish it.
    # But finishing might take too long. Let's check if complete.
    if uncovered_count > 0:
        # We have an incomplete covering. We need to complete it to be valid.
        # Continue without deadline check to ensure validity, but this may timeout.
        # Better: if we can't finish in time, we risk scoring 0.
        # Let's try to finish with a simple greedy without the complex gain computation.
        while uncovered_count > 0:
            # Find first uncovered
            seed_idx = -1
            for i in range(n_total):
                if uncovered[i]:
                    seed_idx = i
                    break
            if seed_idx == -1:
                break
            seed = list(all_t_subsets[seed_idx])
            block = list(seed)
            block_set = set(block)
            while len(block) < k:
                # Simple: pick random or first available
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    x_dict = x_to_tsubset_idx[x]
                    for R in itertools.combinations(block, t - 1):
                        Rf = frozenset(R)
                        if Rf in x_dict:
                            idx = x_dict[Rf]
                            if uncovered[idx]:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    break
                block.append(best_x)
                block_set.add(best_x)
            block.sort()
            blocks.append(block)
            block_set = set(block)
            for S in itertools.combinations(block, t):
                idx = subset_index[frozenset(S)]
                if uncovered[idx]:
                    uncovered[idx] = 0
                    uncovered_count -= 1

    # Remove duplicates
    seen = set()
    unique_blocks = []
    for b in blocks:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(b)

    return unique_blocks
# EVOLVE-BLOCK-END
