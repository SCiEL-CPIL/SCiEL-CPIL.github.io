import itertools
import numpy as np
import time
import random
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin

    # 1. Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_index = {s: i for i, s in enumerate(t_subsets)}
    n_t = len(t_subsets)

    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)

    # Build data structures for fast access
    # subsets_of_block[j]: indices of t-subsets contained in k-subset j
    # blocks_of_subset[i]: indices of k-subsets containing t-subset i
    subsets_of_block = [[] for _ in range(n_k)]
    blocks_of_subset = [[] for _ in range(n_t)]
    
    rows = []
    cols = []
    data = []

    for j, B in enumerate(k_subsets):
        for S in itertools.combinations(B, t):
            i = t_index[S]
            subsets_of_block[j].append(i)
            blocks_of_subset[i].append(j)
            rows.append(i)
            cols.append(j)
            data.append(1.0)

    # Convert to numpy arrays for vectorized operations where possible
    # Note: subsets_of_block is a list of lists of different lengths, so we keep it as list of lists
    # but we can use numpy for the 'gains' and 'uncovered' arrays.
    
    M = csr_matrix((data, (rows, cols)), shape=(n_t, n_k))

    best_solution = None
    best_size = float('inf')

    # 2. LP Relaxation to get duals (weights)
    # This provides a strong initial heuristic for the greedy search.
    duals = None
    try:
        if time.time() > deadline - 20:
            raise TimeoutError()

        c_lp = np.ones(n_k)
        A_ub = -M.T.tocsr()
        b_ub = -np.ones(n_t)
        bounds_lp = [(0, 1)] * n_k

        res = linprog(c_lp, A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method='highs', 
                      options={"presolve": True, "time_limit": 30})

        if res.success:
            # Dual variables are associated with the constraints M.T @ x >= 1
            # In linprog, for A_ub @ x <= b_ub, the marginals are for the constraints.
            # Our constraint is -M.T @ x <= -1.
            duals = res.ineqlin.marginals
            duals = np.abs(duals)
            if np.sum(duals) == 0:
                duals = None
    except Exception:
        duals = None

    if duals is None:
        duals = np.ones(n_t)

    # Pre-compute base weights for blocks based on duals
    # Weight of a block is the sum of duals of the subsets it covers.
    # We will use this as the initial 'gain' in the greedy step.
    base_block_weights = np.zeros(n_k)
    for j in range(n_k):
        if subsets_of_block[j]:
            base_block_weights[j] = np.sum(duals[subsets_of_block[j]])

    # 3. Multi-start Vectorized Greedy Search
    # We use the LP duals to guide the greedy selection.
    # As we pick blocks, we reduce the weight of subsets they cover,
    # effectively lowering the gain of blocks that share those subsets.
    
    # Precompute M as dense for fast row/column extraction (n_t=3876, n_k=50388)
    M_dense = M.toarray()  # shape (n_t, n_k), ~1.5M elements, ~12MB
    
    num_restarts = 200  # Reduced to allow time for local search
    rng = np.random.default_rng(42)

    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break

        # Initialize gains with base weights + noise
        # The noise helps escape symmetry and find different local optima
        if restart == 0:
            gains = base_block_weights.copy()
        else:
            # Add noise proportional to the scale of weights
            noise_scale = np.max(base_block_weights) * 0.05
            gains = base_block_weights + rng.uniform(0, noise_scale, n_k)

        uncovered_mask = np.ones(n_t, dtype=bool)
        solution_blocks = []
        
        # Track current coverage count to allow for "deletion" logic during greedy?
        # No, standard greedy is pick-and-go. We'll do deletion at the end.
        
        while True:
            if time.time() > deadline - 2:
                break

            # Find the block with the highest gain
            # Gain is the sum of duals of *uncovered* subsets in the block
            # Wait, the standard greedy for set cover uses "number of uncovered elements covered".
            # Here, we are using a weighted version.
            # We need to calculate the current gain for each block:
            # gain[j] = sum(duals[s] for s in subsets_of_block[j] if uncovered[s])
            
            # Doing this from scratch every step is O(N_k * C(k,t)) which is too slow.
            # We must update gains incrementally.
            
            # Initial gain is base_block_weights (sum of all duals).
            # When we pick a block j, we mark its subsets as covered.
            # For every other block j' that shares a subset s with j, 
            # we should subtract duals[s] from gains[j'].
            
            # Let's implement this incremental update.
            
            best_idx = np.argmax(gains)
            
            if gains[best_idx] <= 1e-9:
                # Fallback: find any block covering an uncovered subset
                if np.any(uncovered_mask):
                    # Compute gain for all blocks using vectorized dot product
                    # gain_j = sum of duals[s] for uncovered s in block j
                    # = sum(duals[uncovered_mask] * M_dense[uncovered_mask, j])
                    unc_duals = duals[uncovered_mask]
                    if len(unc_duals) > 0:
                        all_gains = M_dense[uncovered_mask, :].T @ unc_duals
                        best_idx = np.argmax(all_gains)
                        if all_gains[best_idx] <= 1e-9:
                            break
                    else:
                        break
                else:
                    break

            # Pick best_idx
            solution_blocks.append(best_idx)
            
            # Update uncovered_mask
            subs_b = subsets_of_block[best_idx]
            newly_covered_indices = [s for s in subs_b if uncovered_mask[s]]
            
            if not newly_covered_indices:
                continue

            uncovered_mask[newly_covered_indices] = False
            
            # Vectorized gain update:
            # For each newly covered subset s, subtract duals[s] from all blocks containing s
            # gains -= M_dense[newly_covered_indices, :].T @ duals[newly_covered_indices]
            if len(newly_covered_indices) > 0:
                delta = M_dense[newly_covered_indices, :].T @ duals[newly_covered_indices]
                gains -= delta

            # Optimization: If all covered, break
            if not np.any(uncovered_mask):
                break

        # Check if solution is valid and better
        if len(solution_blocks) < best_size:
            # Verify coverage just in case
            # We can check if uncovered_mask is all False
            if not np.any(uncovered_mask):
                best_size = len(solution_blocks)
                best_solution = solution_blocks.copy()
                
                # Early exit if we hit a very good score?
                # 151 is the target. 150 would be a breakthrough.
                # If we find 151, we might continue to try for 150.
                # If we find 145, we can stop.
                if best_size <= 151:
                    # Keep searching for a while
                    pass

    # 4. Fallback if no solution found (should not happen)
    if best_solution is None:
        # Simple greedy without weights
        uncovered_mask = np.ones(n_t, dtype=bool)
        solution_blocks = []
        while np.any(uncovered_mask):
            best_idx = -1
            max_cov = -1
            for j in range(n_k):
                cnt = np.sum(uncovered_mask[subsets_of_block[j]])
                if cnt > max_cov:
                    max_cov = cnt
                    best_idx = j
            if best_idx == -1:
                break
            solution_blocks.append(best_idx)
            uncovered_mask[subsets_of_block[best_idx]] = False
        best_solution = solution_blocks
        best_size = len(best_solution)

    # 5. Post-processing: Deletion Pass
    # Try to remove blocks that are redundant.
    if best_solution is not None and time.time() < deadline - 1:
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in best_solution:
            cover_count[subsets_of_block[b_idx]] += 1

        changed = True
        while changed and time.time() < deadline - 1:
            changed = False
            new_solution = []
            for b_idx in best_solution:
                subs_b = subsets_of_block[b_idx]
                can_remove = True
                # If any subset in this block is covered ONLY by this block, we can't remove it
                for s_idx in subs_b:
                    if cover_count[s_idx] == 1:
                        can_remove = False
                        break

                if can_remove:
                    cover_count[subs_b] -= 1
                    changed = True
                else:
                    new_solution.append(b_idx)

            best_solution = new_solution
            best_size = len(best_solution)

    # 6. Local Search: Try swapping blocks to improve coverage
    if best_solution is not None and time.time() < deadline - 2:
        # Compute current coverage counts
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in best_solution:
            cover_count[subsets_of_block[b_idx]] += 1
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            new_solution = list(best_solution)
            
            for i in range(len(new_solution)):
                if time.time() > deadline - 2:
                    break
                
                b_old = new_solution[i]
                subs_old = subsets_of_block[b_old]
                
                # Count how many subsets are uniquely covered by b_old
                unique_count = 0
                for s_idx in subs_old:
                    if cover_count[s_idx] == 1:
                        unique_count += 1
                
                # If b_old uniquely covers subsets, we need to find a replacement
                # that covers those plus potentially more
                if unique_count == 0:
                    # Can just remove b_old
                    cover_count[subs_old] -= 1
                    new_solution.pop(i)
                    improved = True
                    break
                else:
                    # Try to find a block that covers all uniquely-covered subsets of b_old
                    # and has additional uncovered-or-shared coverage
                    unique_subs = [s for s in subs_old if cover_count[s] == 1]
                    
                    # Find candidate blocks that cover all unique_subs
                    # Use intersection of blocks_of_subset for each unique sub
                    if len(unique_subs) <= 3:  # Limit search for speed
                        candidates = set(blocks_of_subset[unique_subs[0]])
                        for s in unique_subs[1:]:
                            candidates &= set(blocks_of_subset[s])
                        candidates.discard(b_old)
                        
                        best_candidate = None
                        best_score = -1
                        for cand in candidates:
                            if cand == b_old:
                                continue
                            # Score: number of subsets in cand that are currently
                            # covered only once (by b_old) or not covered at all
                            subs_cand = subsets_of_block[cand]
                            score = 0
                            for s_idx in subs_cand:
                                if cover_count[s_idx] <= 1:
                                    score += 1
                            # Prefer candidates that cover more low-coverage subsets
                            if score > best_score:
                                best_score = score
                                best_candidate = cand
                        
                        if best_candidate is not None and best_score > unique_count:
                            # Perform swap
                            # Remove b_old coverage
                            cover_count[subs_old] -= 1
                            # Add best_candidate coverage
                            subs_new = subsets_of_block[best_candidate]
                            cover_count[subs_new] += 1
                            new_solution[i] = best_candidate
                            improved = True
                            break
            
            if improved:
                best_solution = new_solution
                best_size = len(best_solution)

    return [list(k_subsets[idx]) for idx in best_solution]
