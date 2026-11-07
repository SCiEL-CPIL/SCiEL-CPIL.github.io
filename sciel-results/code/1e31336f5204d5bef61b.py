# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n = 12
    m = 40
    total_masks = 1 << n

    # Deadline guard: 90s limit, use 85s for safety
    start_time = time.time()
    max_time = 85.0

    # Precompute masks as numpy array for vectorization
    all_masks = np.arange(total_masks, dtype=np.int64)

    def get_initial_selection():
        """
        Greedily construct a good initial set of masks.
        Strategy: Start with a long chain, then add sets that maximize comparability.
        """
        selected = []
        selected_set = set()

        # Start with a deterministic long chain (all subsets of first 12 elements in order)
        cur = 0
        chain = [0]
        for p in range(n):
            cur |= (1 << p)
            chain.append(cur)

        # Add chain elements first
        for mask in chain:
            if mask not in selected_set:
                selected.append(mask)
                selected_set.add(mask)

        # Fill the rest greedily with full scan (4096 is small)
        while len(selected) < m:
            best_mask = -1
            best_score = -1

            # Full scan of all 4096 masks - this is fast enough for initialization
            for mask in range(total_masks):
                if mask in selected_set:
                    continue
                score = 0
                for s in selected:
                    if (mask & s) == mask or (mask & s) == s:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_mask = mask

            if best_mask != -1:
                selected.append(best_mask)
                selected_set.add(best_mask)
            else:
                break

        return selected

    def count_pairs_np(masks_np):
        """
        Vectorized count of comparable pairs.
        """
        inter = masks_np & masks_np.T
        sub1 = (inter == masks_np)
        sub2 = (inter == masks_np.T)
        comp = sub1 | sub2

        indices = np.triu_indices(m, k=1)
        return int(np.sum(comp[indices]))

    def compute_comparability_matrix(masks_list):
        """
        Compute the full comparability matrix for a list of masks.
        Returns a boolean matrix C where C[i,j] is True if masks[i] and masks[j] are comparable.
        """
        masks_np = np.array(masks_list, dtype=np.int64).reshape(-1, 1)
        inter = masks_np & masks_np.T
        sub1 = (inter == masks_np)
        sub2 = (inter == masks_np.T)
        comp = sub1 | sub2
        # Zero out diagonal
        np.fill_diagonal(comp, False)
        return comp

    def score_from_matrix(comp):
        """Compute score from comparability matrix."""
        indices = np.triu_indices(m, k=1)
        return int(np.sum(comp[indices]))

    def optimize(masks_list):
        """
        Greedy refinement using NumPy vectorization for candidate evaluation.
        Uses both 1-swap and 2-swap moves.
        """
        current_masks = list(masks_list)
        comp = compute_comparability_matrix(current_masks)
        current_score = score_from_matrix(comp)

        max_passes = 300
        improved = True

        while improved and time.time() - start_time < max_time - 5.0:
            improved = False
            order = list(range(m))
            random.shuffle(order)

            # 1-swap moves
            for i in order:
                old_mask = current_masks[i]
                others = [current_masks[k] for k in range(m) if k != i]

                if not others:
                    continue

                others_np = np.array(others, dtype=np.int64).reshape(-1, 1)

                # Create valid mask
                valid_mask = np.ones(total_masks, dtype=bool)
                for cm in current_masks:
                    valid_mask[cm] = False

                # Compute degrees for all valid candidates
                candidates_np = all_masks.reshape(-1, 1)
                inter = candidates_np & others_np.T
                sub1 = (inter == candidates_np)
                sub2 = (inter == others_np.T)
                comp_cand = sub1 | sub2
                degrees = np.sum(comp_cand, axis=1)

                degrees[~valid_mask] = -1

                best_idx = np.argmax(degrees)
                best_gain = int(degrees[best_idx])

                # Get current degree of old_mask
                old_degree = int(np.sum(comp[i, :]) + np.sum(comp[:, i]) - (1 if comp[i, i] else 0))
                # Since diagonal is False, old_degree = sum(comp[i,:])
                old_degree = int(np.sum(comp[i, :]))

                if best_gain > old_degree:
                    new_mask = int(all_masks[best_idx])
                    # Update matrix
                    comp[i, :] = False
                    comp[:, i] = False
                    current_masks[i] = new_mask
                    # Recompute row/col i
                    others_np2 = np.array(current_masks, dtype=np.int64).reshape(-1, 1)
                    inter_new = others_np2[i] & others_np2.T
                    sub1_new = (inter_new == others_np2[i])
                    sub2_new = (inter_new == others_np2.T[i])
                    comp[i, :] = sub1_new | sub2_new
                    comp[:, i] = comp[i, :]
                    np.fill_diagonal(comp, False)
                    current_score = score_from_matrix(comp)
                    improved = True

            if time.time() - start_time > max_time - 5.0:
                break

            # 2-swap moves (try to improve further)
            if not improved and time.time() - start_time < max_time - 10.0:
                for i in range(m - 1):
                    for j in range(i + 1, m):
                        if time.time() - start_time > max_time - 5.0:
                            break
                        old_mask_i = current_masks[i]
                        old_mask_j = current_masks[j]
                        others = [current_masks[k] for k in range(m) if k != i and k != j]

                        if not others:
                            continue

                        others_np = np.array(others, dtype=np.int64).reshape(-1, 1)

                        # Create valid mask
                        valid_mask = np.ones(total_masks, dtype=bool)
                        for cm in current_masks:
                            valid_mask[cm] = False

                        # Compute degrees for all valid candidates
                        candidates_np = all_masks.reshape(-1, 1)
                        inter = candidates_np & others_np.T
                        sub1 = (inter == candidates_np)
                        sub2 = (inter == others_np.T)
                        comp_cand = sub1 | sub2
                        degrees = np.sum(comp_cand, axis=1)

                        degrees[~valid_mask] = -1

                        # For 2-swap, we need to find two new masks that maximize total comparability
                        # with others AND with each other
                        # This is more complex, so let's just try replacing one at a time
                        # and see if we can find a better pair

                        # Get current degrees
                        old_degree_i = int(np.sum(comp[i, :]) - (1 if comp[i, j] else 0))
                        old_degree_j = int(np.sum(comp[j, :]) - (1 if comp[i, j] else 0))
                        old_pair = 1 if comp[i, j] else 0
                        old_total = old_degree_i + old_degree_j + old_pair

                        # Try to find best single replacement for i
                        best_idx_i = np.argmax(degrees)
                        best_gain_i = int(degrees[best_idx_i])

                        # Try to find best single replacement for j (excluding the one we'd pick for i)
                        degrees_j = degrees.copy()
                        degrees_j[best_idx_i] = -1
                        best_idx_j = np.argmax(degrees_j)
                        best_gain_j = int(degrees[best_idx_j])

                        # Check if new masks are comparable to each other
                        new_mask_i = int(all_masks[best_idx_i])
                        new_mask_j = int(all_masks[best_idx_j])
                        new_pair = 1 if (new_mask_i & new_mask_j) == new_mask_i or (new_mask_i & new_mask_j) == new_mask_j else 0
                        new_total = best_gain_i + best_gain_j + new_pair

                        if new_total > old_total:
                            # Update
                            comp[i, :] = False
                            comp[j, :] = False
                            comp[:, i] = False
                            comp[:, j] = False
                            current_masks[i] = new_mask_i
                            current_masks[j] = new_mask_j
                            # Recompute rows i and j
                            others_np2 = np.array(current_masks, dtype=np.int64).reshape(-1, 1)
                            for idx in [i, j]:
                                inter_new = others_np2[idx] & others_np2.T
                                sub1_new = (inter_new == others_np2[idx])
                                sub2_new = (inter_new == others_np2.T[idx])
                                comp[idx, :] = sub1_new | sub2_new
                                comp[:, idx] = comp[idx, :]
                            np.fill_diagonal(comp, False)
                            current_score = score_from_matrix(comp)
                            improved = True

                    if time.time() - start_time > max_time - 5.0:
                        break

        return current_masks, current_score

    def final_polish(masks_list):
        """
        Final high-precision polish pass with more iterations.
        """
        current_masks = list(masks_list)
        comp = compute_comparability_matrix(current_masks)
        current_score = score_from_matrix(comp)

        # Do several more passes of 1-swap
        for _ in range(50):
            if time.time() - start_time > max_time - 3.0:
                break
            improved = False
            order = list(range(m))
            random.shuffle(order)

            for i in order:
                old_mask = current_masks[i]
                others = [current_masks[k] for k in range(m) if k != i]

                if not others:
                    continue

                others_np = np.array(others, dtype=np.int64).reshape(-1, 1)

                valid_mask = np.ones(total_masks, dtype=bool)
                for cm in current_masks:
                    valid_mask[cm] = False

                candidates_np = all_masks.reshape(-1, 1)
                inter = candidates_np & others_np.T
                sub1 = (inter == candidates_np)
                sub2 = (inter == others_np.T)
                comp_cand = sub1 | sub2
                degrees = np.sum(comp_cand, axis=1)

                degrees[~valid_mask] = -1

                best_idx = np.argmax(degrees)
                best_gain = int(degrees[best_idx])
                old_degree = int(np.sum(comp[i, :]))

                if best_gain > old_degree:
                    new_mask = int(all_masks[best_idx])
                    comp[i, :] = False
                    comp[:, i] = False
                    current_masks[i] = new_mask
                    others_np2 = np.array(current_masks, dtype=np.int64).reshape(-1, 1)
                    inter_new = others_np2[i] & others_np2.T
                    sub1_new = (inter_new == others_np2[i])
                    sub2_new = (inter_new == others_np2.T[i])
                    comp[i, :] = sub1_new | sub2_new
                    comp[:, i] = comp[i, :]
                    np.fill_diagonal(comp, False)
                    current_score = score_from_matrix(comp)
                    improved = True

            if not improved:
                break

        return current_masks, current_score

    # Main execution
    best_masks = None
    best_score = -1

    # Run a few restarts with different initializations
    num_restarts = 8

    for r in range(num_restarts):
        if time.time() - start_time > max_time - 10.0:
            break

        if r == 0:
            init_masks = get_initial_selection()
        else:
            # Perturb the best solution found so far
            if best_masks is not None:
                init_masks = list(best_masks)
                # Randomly swap some elements
                for _ in range(r * 2):
                    i = random.randint(0, m - 1)
                    j = random.randint(0, m - 1)
                    init_masks[i], init_masks[j] = init_masks[j], init_masks[i]
                # Ensure distinctness
                seen = set()
                unique_masks = []
                for mask in init_masks:
                    if mask not in seen:
                        seen.add(mask)
                        unique_masks.append(mask)
                while len(unique_masks) < m:
                    # Add a random mask not in seen
                    rand_mask = random.randint(0, total_masks - 1)
                    if rand_mask not in seen:
                        seen.add(rand_mask)
                        unique_masks.append(rand_mask)
                init_masks = unique_masks[:m]
            else:
                init_masks = get_initial_selection()

        # Optimization
        opt_masks, opt_score = optimize(init_masks)

        # Final polish
        polished_masks, polished_score = final_polish(opt_masks)

        if polished_score > best_score:
            best_score = polished_score
            best_masks = polished_masks

        # Check time
        if time.time() - start_time > max_time - 2.0:
            break

    if best_masks is None:
        # Fallback
        best_masks = list(range(m))

    # Convert best_masks to list of lists
    result = []
    for mask in best_masks:
        s = []
        for b in range(n):
            if mask & (1 << b):
                s.append(b)
        result.append(s)

    return result

# EVOLVE-BLOCK-END
