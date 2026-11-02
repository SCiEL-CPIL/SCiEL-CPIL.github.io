<<<<<<< SEARCH
import numpy as np
import itertools
import time
import random
from collections import defaultdict
=======
import numpy as np
import itertools
import time
import random
from collections import defaultdict
from scipy.optimize import milp, LinearConstraint, Bounds
>>>>>>> REPLACE
<<<<<<< SEARCH
    def greedy(seed):
        """Greedy set cover with rarity-weighted tie-breaking."""
        rng = random.Random(seed)
        covered = np.zeros(n_ts, dtype=bool)
        blocks = []
        cover_freq = np.zeros(n_ts, dtype=np.int32)
        remaining = n_ts

        while remaining > 0:
            if time.time() - start_time > time_limit - 5.0:
                break

            # Compute gain: number of new t-subsets covered
            gains = (~covered[block_t_indices]).sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break

            candidates = np.where(gains == max_gain)[0]

            if len(candidates) > 1:
                # Rarity-weighted scoring: prefer blocks covering rare triples
                best_score = -float('inf')
                best_idx = candidates[0]
                for c_idx in candidates:
                    subs = block_t_indices[c_idx]
                    newly = ~covered[subs]
                    if np.any(newly):
                        # Sum of inverse (1 + cover_freq) for newly covered
                        bonus = np.sum(1.0 / (1.0 + cover_freq[subs[newly]]))
                    else:
                        bonus = 0.0
                    score = max_gain + 0.1 * bonus + rng.uniform(0, 0.1)
                    if score > best_score:
                        best_score = score
                        best_idx = c_idx
            else:
                best_idx = candidates[0]

            blocks.append(best_idx)
            subs = block_t_indices[best_idx]
            newly = ~covered[subs]
            remaining -= newly.sum()
            cover_freq[subs] += 1
            covered[subs] = True

        return blocks
=======
    def greedy(seed, stop_threshold=None):
        """Greedy set cover with rarity-weighted tie-breaking.
        If stop_threshold is set, stop when remaining <= stop_threshold."""
        rng = random.Random(seed)
        covered = np.zeros(n_ts, dtype=bool)
        blocks = []
        cover_freq = np.zeros(n_ts, dtype=np.int32)
        remaining = n_ts

        while remaining > 0:
            if time.time() - start_time > time_limit - 5.0:
                break
            if stop_threshold is not None and remaining <= stop_threshold:
                break

            # Compute gain: number of new t-subsets covered
            gains = (~covered[block_t_indices]).sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break

            candidates = np.where(gains == max_gain)[0]

            if len(candidates) > 1:
                # Rarity-weighted scoring: prefer blocks covering rare triples
                best_score = -float('inf')
                best_idx = candidates[0]
                for c_idx in candidates:
                    subs = block_t_indices[c_idx]
                    newly = ~covered[subs]
                    if np.any(newly):
                        # Sum of inverse (1 + cover_freq) for newly covered
                        bonus = np.sum(1.0 / (1.0 + cover_freq[subs[newly]]))
                    else:
                        bonus = 0.0
                    score = max_gain + 0.1 * bonus + rng.uniform(0, 0.1)
                    if score > best_score:
                        best_score = score
                        best_idx = c_idx
            else:
                best_idx = candidates[0]

            blocks.append(best_idx)
            subs = block_t_indices[best_idx]
            newly = ~covered[subs]
            remaining -= newly.sum()
            cover_freq[subs] += 1
            covered[subs] = True

        return blocks, covered
>>>>>>> REPLACE
<<<<<<< SEARCH
    def local_search_remove(blocks_list):
        """Remove redundant blocks iteratively."""
        current = list(blocks_list)
        coverage_count = build_coverage(current)
        return aggressive_deletion(current, coverage_count, 5.0)
=======
    def local_search_remove(blocks_list):
        """Remove redundant blocks iteratively."""
        current = list(blocks_list)
        coverage_count = build_coverage(current)
        return aggressive_deletion(current, coverage_count, 5.0)

    def ilp_complete(blocks_list, time_budget=30.0):
        """Solve the residual covering problem as an ILP using scipy.optimize.milp.
        Returns the completed block list."""
        deadline = time.time() + time_budget

        # Compute current coverage
        coverage = build_coverage(blocks_list)
        uncovered_mask = (coverage == 0)
        n_uncovered = np.sum(uncovered_mask)

        if n_uncovered == 0:
            return list(blocks_list)

        # If too many uncovered, ILP might be too large; use greedy instead
        if n_uncovered > 200:
            # Fall back to greedy for the remainder
            extra_blocks, _ = greedy(random.randint(0, 10000))
            # Only add blocks that help
            result = list(blocks_list)
            existing = set(result)
            for b in extra_blocks:
                if b not in existing:
                    result.append(b)
                    existing.add(b)
            return result

        # Get indices of uncovered triples
        uncovered_indices = np.where(uncovered_mask)[0]

        # For the ILP, we need to find blocks that cover uncovered triples.
        # We'll use all blocks (n_ks) as potential additions, but only those
        # that cover at least one uncovered triple are useful.
        # Build constraint matrix: rows = uncovered triples, cols = all blocks
        # A[i,j] = 1 if block j covers triple i

        # Use sparse representation for efficiency
        from scipy.sparse import csr_matrix

        # For each uncovered triple, find which blocks cover it
        # triple_to_blocks already maps triple index -> list of block indices
        # Build the constraint matrix
        n_blocks_total = n_ks
        A_rows = []
        A_cols = []
        A_vals = []

        for i, tri_idx in enumerate(uncovered_indices):
            for b_idx in triple_to_blocks[tri_idx]:
                A_rows.append(i)
                A_cols.append(b_idx)
                A_vals.append(1)

        A = csr_matrix((A_vals, (A_rows, A_cols)), shape=(len(uncovered_indices), n_blocks_total))

        # Objective: minimize sum of x_j (number of blocks added)
        c = np.ones(n_blocks_total)

        # Constraints: A @ x >= 1 for each uncovered triple
        # Convert to standard form: A @ x >= 1  =>  -A @ x <= -1
        # scipy milp uses: lb <= A_ub @ x <= ub
        constraint = LinearConstraint(A, lb=np.ones(len(uncovered_indices)), ub=np.inf)

        # Bounds: x_j in {0, 1}
        bounds = Bounds(lb=0, ub=1)
        integrality = np.ones(n_blocks_total)

        try:
            result = milp(c=c, constraints=[constraint], integrality=integrality, bounds=bounds,
                          options={'time_limit': max(1.0, time_budget - 2.0)})
        except Exception:
            # Fallback to greedy
            extra_blocks, _ = greedy(random.randint(0, 10000))
            result_list = list(blocks_list)
            existing = set(result_list)
            for b in extra_blocks:
                if b not in existing:
                    result_list.append(b)
                    existing.add(b)
            return result_list

        if result.status == 0:  # Optimal
            x = np.round(result.x)
            added = np.where(x > 0.5)[0]
        elif result.status == 1:  # Feasible
            x = np.round(result.x)
            added = np.where(x > 0.5)[0]
        else:
            # Fallback to greedy
            extra_blocks, _ = greedy(random.randint(0, 10000))
            result_list = list(blocks_list)
            existing = set(result_list)
            for b in extra_blocks:
                if b not in existing:
                    result_list.append(b)
                    existing.add(b)
            return result_list

        new_blocks = list(blocks_list)
        existing = set(new_blocks)
        for b in added:
            if b not in existing:
                new_blocks.append(b)
                existing.add(b)

        return new_blocks
>>>>>>> REPLACE
<<<<<<< SEARCH
    # Main loop
    best_blocks = None
    best_count = float('inf')
    restart_deadline = time_limit * 0.45

    seed = 42
    while time.time() - start_time < restart_deadline:
        greedy_blocks = greedy(seed)
        if not greedy_blocks:
            seed += 1
            continue

        # Local search pipeline
        refined = local_search_remove(greedy_blocks)
        if time.time() - start_time < time_limit - 15:
            refined = local_search_replace(refined, min(1.0, max(0.1, (time_limit - (time.time() - start_time)) * 0.03)))
        if time.time() - start_time < time_limit - 12:
            refined = local_search_remove(refined)

        count = len(refined)
        if count < best_count:
            best_count = count
            best_blocks = refined

        seed += 1
        if best_count <= 70:
            break

    # Final polish
    if best_blocks is not None and time.time() - start_time < time_limit - 5:
        polished = local_search_remove(best_blocks)
        if time.time() - start_time < time_limit - 4:
            polished = local_search_replace(polished, min(4.0, max(0.5, time_limit - (time.time() - start_time)) - 4.0)))
        if time.time() - start_time < time_limit - 3:
            polished = local_search_remove(polished)

        # Simulated annealing
        if time.time() - start_time < time_limit - 2:
            sa_time = min(6.0, max(0.5, time_limit - (time.time() - start_time) - 2.0))
            sa_rng = random.Random(12345)
            sa_result = simulated_annealing(polished, sa_time, sa_rng)
            if len(sa_result) < len(polished):
                polished = sa_result

        if time.time() - start_time < time_limit - 1:
            polished = local_search_remove(polished)
        if time.time() - start_time < time_limit - 0.5:
            polished = local_search_single(polished, min(2.0, max(0.2, time_limit - (time.time() - start_time) - 0.5)))
        best_blocks = polished

    final_blocks = [list(all_ks[i]) for i in best_blocks]
    return final_blocks
=======
    # Main loop
    best_blocks = None
    best_count = float('inf')
    restart_deadline = time_limit * 0.40

    seed = 42
    while time.time() - start_time < restart_deadline:
        greedy_blocks, covered_arr = greedy(seed)
        if not greedy_blocks:
            seed += 1
            continue

        # Local search pipeline
        refined = local_search_remove(greedy_blocks)
        if time.time() - start_time < time_limit - 20:
            refined = local_search_replace(refined, min(1.0, max(0.1, (time_limit - (time.time() - start_time)) * 0.02)))
        if time.time() - start_time < time_limit - 15:
            refined = local_search_remove(refined)

        count = len(refined)
        if count < best_count:
            best_count = count
            best_blocks = list(refined)

        seed += 1
        if best_count <= 70:
            break

    # Final polish with ILP completion
    if best_blocks is not None:
        # First pass: local search
        if time.time() - start_time < time_limit - 30:
            polished = local_search_remove(best_blocks)
            if time.time() - start_time < time_limit - 25:
                polished = local_search_replace(polished, min(3.0, max(0.5, time_limit - (time.time() - start_time)) - 25.0)))
            if time.time() - start_time < time_limit - 20:
                polished = local_search_remove(polished)
            best_blocks = polished
            best_count = len(best_blocks)

        # ILP completion: try to optimally complete the residual
        if time.time() - start_time < time_limit - 10:
            ilp_time = min(20.0, max(5.0, time_limit - (time.time() - start_time)) - 10.0))
            ilp_result = ilp_complete(best_blocks, ilp_time)
            if len(ilp_result) < best_count:
                best_blocks = ilp_result
                best_count = len(best_blocks)

        # Final local search after ILP
        if time.time() - start_time < time_limit - 5:
            polished = local_search_remove(best_blocks)
            if time.time() - start_time < time_limit - 3:
                polished = local_search_replace(polished, min(2.0, max(0.5, time_limit - (time.time() - start_time)) - 3.0)))
            if time.time() - start_time < time_limit - 2:
                polished = local_search_remove(polished)
            best_blocks = polished

        # Simulated annealing if time remains
        if time.time() - start_time < time_limit - 2:
            sa_time = min(5.0, max(0.5, time_limit - (time.time() - start_time)) - 2.0))
            sa_rng = random.Random(12345)
            sa_result = simulated_annealing(best_blocks, sa_time, sa_rng)
            if len(sa_result) < len(best_blocks):
                best_blocks = sa_result

        # Final cleanup
        if time.time() - start_time < time_limit - 1:
            best_blocks = local_search_remove(best_blocks)
        if time.time() - start_time < time_limit - 0.5:
            best_blocks = local_search_single(best_blocks, min(1.5, max(0.2, time_limit - (time.time() - start_time)) - 0.5)))

    # Ensure validity: if greedy didn't cover everything, use full greedy
    if best_blocks is None:
        greedy_blocks, _ = greedy(42)
        best_blocks = greedy_blocks

    # Verify coverage and fix if needed
    coverage_check = build_coverage(best_blocks)
    if np.any(coverage_check == 0):
        # Add blocks to cover remaining
        existing = set(best_blocks)
        covered_arr = coverage_check > 0
        while np.any(~covered_arr) and time.time() - start_time < time_limit - 1:
            gains = (~covered_arr[block_t_indices]).sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[0]
            if best_idx not in existing:
                best_blocks.append(best_idx)
                existing.add(best_idx)
            subs = block_t_indices[best_idx]
            newly = ~covered_arr[subs]
            covered_arr[subs] = True

    # Deduplicate
    seen = set()
    final_idx = []
    for b in best_blocks:
        if b not in seen:
            seen.add(b)
            final_idx.append(b)

    final_blocks = [list(all_ks[i]) for i in final_idx]
    return final_blocks
>>>>>>> REPLACE

Wait, I notice there's a syntax error with extra parentheses. Let me fix that:

