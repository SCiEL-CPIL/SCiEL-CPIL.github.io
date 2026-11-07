import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit

    n = 12
    m = 40
    total_masks = 1 << n  # 4096

    # Precompute popcount for all masks (useful for tie-breaking / heuristics)
    popcount = np.zeros(total_masks, dtype=np.int64)
    for i in range(1, total_masks):
        popcount[i] = popcount[i >> 1] + (i & 1)

    all_masks = np.arange(total_masks, dtype=np.int64)

    def count_pairs_np(masks_np):
        """Vectorized count of comparable (strict subset) pairs among distinct sets."""
        inter = masks_np & masks_np.T
        sub1 = (inter == masks_np)
        sub2 = (inter == masks_np.T)
        comp = sub1 | sub2
        # Exclude diagonal (self) and count upper triangle
        np.fill_diagonal(comp, False)
        indices = np.triu_indices(m, k=1)
        return int(np.sum(comp[indices]))

    def get_initial_selection():
        """
        Greedy construction: start with a full chain (maximally comparable base),
        then fill the rest by picking masks that maximize comparability with the
        current family.
        """
        selected = []
        selected_set = set()

        # Build a full chain: empty set, then add one element at a time in random order
        perm = list(range(n))
        random.shuffle(perm)
        cur = 0
        chain = [0]
        for p in perm:
            cur |= (1 << p)
            chain.append(cur)

        for mask in chain:
            if mask not in selected_set:
                selected.append(mask)
                selected_set.add(mask)

        # Fill the rest greedily
        while len(selected) < m:
            selected_np = np.array(selected, dtype=np.int64).reshape(-1, 1)
            candidates_np = all_masks.reshape(-1, 1)

            # Vectorized comparability: candidates (4096,1) vs selected (k,1)
            inter = candidates_np & selected_np.T  # (4096, k)
            sub1 = (inter == candidates_np)
            sub2 = (inter == selected_np.T)
            comp = sub1 | sub2
            degrees = np.sum(comp, axis=1)  # (4096,)

            # Exclude already selected
            for s in selected:
                degrees[s] = -1

            best_idx = int(np.argmax(degrees))
            if degrees[best_idx] < 0:
                break
            selected.append(int(best_idx))
            selected_set.add(int(best_idx))

        # Fallback if we didn't reach m (shouldn't happen)
        while len(selected) < m:
            cand = random.randint(0, total_masks - 1)
            if cand not in selected_set:
                selected.append(cand)
                selected_set.add(cand)

        return selected

    def local_search(family, max_passes=100):
        """
        Best-improvement local search: for each position, find the single best
        replacement from all 4096 candidates using vectorized scoring.
        """
        current_masks = list(family)
        current_score = count_pairs_np(np.array(current_masks, dtype=np.int64).reshape(-1, 1))

        improved = True
        passes = 0

        while improved and passes < max_passes and time.time() < deadline - 3:
            improved = False
            passes += 1

            for i in range(m):
                if time.time() > deadline - 3:
                    break

                old_mask = current_masks[i]
                others = [current_masks[j] for j in range(m) if j != i]
                others_np = np.array(others, dtype=np.int64).reshape(-1, 1)
                candidates_np = all_masks.reshape(-1, 1)

                # Vectorized comparability: candidates (4096,1) vs others (39,1)
                inter = candidates_np & others_np.T  # (4096, 39)
                sub1 = (inter == candidates_np)
                sub2 = (inter == others_np.T)
                comp = sub1 | sub2
                degrees = np.sum(comp, axis=1)  # (4096,)

                # Exclude already-used masks (except the one at position i)
                for j in range(m):
                    if j != i:
                        degrees[current_masks[j]] = -1

                best_idx = int(np.argmax(degrees))
                if degrees[best_idx] < 0:
                    continue

                new_mask = int(best_idx)
                if new_mask == old_mask:
                    continue

                # Compute score change: new_degree - old_degree
                old_degree = int(np.sum(comp[old_mask]))
                new_degree = int(degrees[best_idx])
                delta = new_degree - old_degree

                if delta > 0:
                    current_masks[i] = new_mask
                    current_score += delta
                    improved = True

        return current_masks, current_score

    # Initial construction
    family = get_initial_selection()
    best_family = list(family)
    best_score = count_pairs_np(np.array(best_family, dtype=np.int64).reshape(-1, 1))

    # Local search on the initial family
    result_family, result_score = local_search(family, max_passes=50)
    if result_score > best_score:
        best_score = result_score
        best_family = result_family

    # Multiple restarts with perturbations
    while time.time() < deadline - 5:
        perturbed = list(best_family)
        n_perturb = random.randint(1, 4)
        for _ in range(n_perturb):
            idx = random.randint(0, m - 1)
            new_mask = random.randint(0, total_masks - 1)
            attempts = 0
            while new_mask in perturbed and attempts < 50:
                new_mask = random.randint(0, total_masks - 1)
                attempts += 1
            if new_mask not in perturbed:
                perturbed[idx] = new_mask

        result_family, result_score = local_search(perturbed, max_passes=30)

        if result_score > best_score:
            best_score = result_score
            best_family = result_family

    # Final polish: a few more passes with minimal perturbation
    if time.time() < deadline - 2:
        polish_family, polish_score = local_search(list(best_family), max_passes=20)
        if polish_score > best_score:
            best_score = polish_score
            best_family = polish_family

    # Convert masks to lists of Python ints
    result = []
    for mask in best_family:
        s = []
        for b in range(n):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)

    return result
