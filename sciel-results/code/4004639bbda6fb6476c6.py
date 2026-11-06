import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n_points = 16

    # Precompute all possible triples
    all_triples = list(itertools.combinations(range(n_points), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    n_triples_total = len(all_triples)  # 560

    # Precompute conflict matrix: conflict[i][j] is True if triple i and j share > 1 point
    conflicts = [set() for _ in range(n_triples_total)]
    for i in range(n_triples_total):
        t1 = set(all_triples[i])
        for j in range(i + 1, n_triples_total):
            t2 = set(all_triples[j])
            if len(t1 & t2) > 1:
                conflicts[i].add(j)
                conflicts[j].add(i)

    # Convert to numpy arrays for faster checking
    conflict_arrays = [np.array(list(c), dtype=np.int32) if c else np.array([], dtype=np.int32) for c in conflicts]

    # Precompute subset masks for independence check
    subset_masks = {}
    for k in [7, 6, 5, 4, 3, 2, 1]:
        combos = list(itertools.combinations(range(n_points), k))
        masks = np.array([sum(1 << p for p in c) for c in combos], dtype=np.int64)
        subset_masks[k] = masks

    def compute_alpha(triples):
        """
        triples: list of tuples (a,b,c)
        Returns the independence number.
        """
        if not triples:
            return n_points

        triple_masks = np.array([sum(1 << p for p in t) for t in triples], dtype=np.int64)

        # Check from high k down to low k
        for k in range(n_points, 0, -1):
            if k > 7:
                # For k > 7, generate masks on the fly
                combos = list(itertools.combinations(range(n_points), k))
                masks = np.array([sum(1 << p for p in c) for c in combos], dtype=np.int64)
            else:
                masks = subset_masks[k]

            if len(triple_masks) == 0:
                return k

            # Vectorized check
            # (masks[:, None] & triple_masks[None, :]) == triple_masks[None, :]
            # This checks if triple is subset of mask.
            contains = (masks[:, None] & triple_masks[None, :]) == triple_masks[None, :]
            any_contains = np.any(contains, axis=1)
            is_independent = ~any_contains

            if np.any(is_independent):
                return k

        return 1

    # Initial construction: STS(15) on 0..14
    vectors = [
        0b0001, 0b0010, 0b0011, 0b0100, 0b0101,
        0b0110, 0b0111, 0b1000, 0b1001, 0b1010,
        0b1011, 0b1100, 0b1101, 0b1110, 0b1111
    ]
    vec_to_idx = {v: i for i, v in enumerate(vectors)}

    initial_triples = set()
    n_sts = 15
    for i in range(n_sts):
        for j in range(i + 1, n_sts):
            v_k = vectors[i] ^ vectors[j]
            k = vec_to_idx[v_k]
            t = tuple(sorted([i, j, k]))
            initial_triples.add(t)

    current_triples = list(initial_triples)
    current_alpha = compute_alpha(current_triples)

    best_triples = current_triples[:]
    best_alpha = current_alpha

    random.seed(42)

    # Local Search
    # We represent the current set of triples as a set of indices into all_triples for fast lookup.
    current_set = set(triple_to_idx[t] for t in current_triples)

    def is_linear(idx_new, current_indices):
        """Check if triple idx_new is linear with all triples in current_indices."""
        for idx_old in current_indices:
            if idx_new in conflicts[idx_old]:
                return False
        return True

    # Main loop
    while time.time() < deadline - 5:
        # Try a move
        # 1. Swap: Remove one, add one
        # 2. Add: Add one (if space)
        # 3. Remove: Remove one (usually bad for alpha, but maybe for linearity? No, linearity is preserved by removal)

        # Prioritize moves that involve point 15
        # Pick a random triple to remove
        if current_set:
            idx_remove = random.choice(list(current_set))
            t_remove = all_triples[idx_remove]

            # Pick a candidate to add
            # Bias towards triples containing 15
            if random.random() < 0.5:
                # Generate random triple with 15
                a, b = random.sample(range(15), 2)
                t_add = tuple(sorted([a, b, 15]))
            else:
                t_add = random.choice(all_triples)

            idx_add = triple_to_idx[t_add]

            # Check if t_add is already in set
            if idx_add in current_set:
                continue

            # Check linearity
            # t_add must be linear with all triples in current_set except t_remove
            # We can check against current_set - {idx_remove}
            # To speed up, we can just check against all, and if it conflicts with t_remove, that's fine since t_remove is gone.
            # But if it conflicts with t_remove, and t_remove is gone, it's okay.
            # So we need to check conflicts with (current_set - {idx_remove}).

            valid = True
            for idx_old in current_set:
                if idx_old == idx_remove:
                    continue
                if idx_add in conflicts[idx_old]:
                    valid = False
                    break

            if not valid:
                continue

            # Apply move
            new_set = set(current_set)
            new_set.remove(idx_remove)
            new_set.add(idx_add)

            new_triples = [all_triples[i] for i in new_set]
            new_alpha = compute_alpha(new_triples)

            if new_alpha < best_alpha:
                best_alpha = new_alpha
                best_triples = new_triples
                current_set = new_set
                current_alpha = new_alpha
            elif new_alpha <= current_alpha and random.random() < 0.5:
                # Accept to explore
                current_set = new_set
                current_alpha = new_alpha
        else:
            break

    # Return best found
    return [list(t) for t in best_triples]
