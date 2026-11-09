"""Strategy: Fast greedy with multi-start local search on vertex ordering.
The key insight is that the greedy decoder is very sensitive to the ordering.
We use a fast baseline (Min-Degree) and then refine the order using a
local search that swaps adjacent or random vertices to improve the count.
We also try to directly optimize the independent set via a "repair" step
that attempts to swap vertices in/out to increase the set size.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def _greedy_decode(adjacency, order):
    """
    Run the fixed greedy decoder given a specific vertex order (permutation).
    Returns: (count, selected_mask)
    """
    n = adjacency.shape[0]
    selected = np.zeros(n, dtype=bool)
    available = np.ones(n, dtype=bool)
    count = 0
    
    # Precompute neighbors as lists of indices for faster access?
    # For n=200, boolean indexing is fast enough.
    
    for v in order:
        if not available[v]:
            continue
        selected[v] = True
        count += 1
        # Mark neighbors as unavailable
        # adjacency[v] is a boolean row
        available[adjacency[v]] = False
        
    return count, selected

def _evaluate_scores(adjacency, scores):
    """
    Evaluate a set of scores by sorting them and running the greedy decoder.
    Returns: count
    """
    order = np.argsort(-scores, kind='stable')
    count, _ = _greedy_decode(adjacency, order)
    return count

def _compute_core_numbers(adjacency):
    """
    Compute the core number for each vertex in the graph.
    """
    n = adjacency.shape[0]
    degrees = adjacency.sum(axis=1).astype(int)
    
    deg = degrees.copy()
    core = np.zeros(n, dtype=int)
    
    remaining = np.ones(n, dtype=bool)
    
    for _ in range(n):
        if not np.any(remaining):
            break
        # Find vertex with min degree among remaining
        masked_degrees = np.where(remaining, deg, np.inf)
        v = np.argmin(masked_degrees)
        core[v] = deg[v]
        
        remaining[v] = False
        neighbors = np.where(adjacency[v])[0]
        active_neighbors = neighbors[remaining[neighbors]]
        if len(active_neighbors) > 0:
            deg[active_neighbors] -= 1
            
    return core

def _compute_spectral_features(adjacency, n):
    """
    Compute spectral features using the graph Laplacian.
    Returns: (fiedler_vector, lambda_2)
    The Fiedler vector (second-smallest eigenvector) identifies the sparsest cut.
    Vertices with small absolute values are near the cut boundary;
    vertices with large absolute values are in the "core" of one partition.
    """
    try:
        # Build Laplacian L = D - A
        degrees = adjacency.sum(axis=1)
        D = np.diag(degrees)
        L = D - adjacency.astype(float)
        
        # Convert to sparse for efficiency
        L_sparse = sp.csr_matrix(L)
        
        # Compute smallest eigenvalues using shift-invert mode
        # We want lambda_1 ≈ 0, lambda_2 (Fiedler value), and corresponding eigenvector
        k = min(3, n - 1)  # Get up to 3 smallest eigenvalues
        if k < 2:
            return np.zeros(n), 0.0
            
        eigvals, eigvecs = spla.eigsh(L_sparse, k=k, sigma=-1e-6, which='LM')
        
        # Sort by eigenvalue
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Fiedler vector is the eigenvector for lambda_2 (index 1, since lambda_1 ≈ 0)
        if len(eigvals) >= 2:
            fiedler = eigvecs[:, 1]
            lambda_2 = eigvals[1]
        else:
            fiedler = eigvecs[:, -1]
            lambda_2 = eigvals[-1]
            
        # Normalize Fiedler vector to have unit norm
        norm_f = np.linalg.norm(fiedler)
        if norm_f > 1e-10:
            fiedler = fiedler / norm_f
            
        return fiedler, lambda_2
        
    except Exception:
        # Fallback if spectral computation fails
        return np.zeros(n), 0.0

def _baseline_scores(adjacency):
    """
    Generate several baseline score vectors.
    """
    n = adjacency.shape[0]
    degrees = adjacency.sum(axis=1).astype(float)
    neighbor_deg_sum = adjacency @ degrees
    coreness = _compute_core_numbers(adjacency).astype(float)
    
    # Compute spectral features
    fiedler, lambda_2 = _compute_spectral_features(adjacency, n)
    
    scores_list = []
    
    # 1. Min Degree (classic)
    scores_list.append(-degrees)
    
    # 2. Min Neighbor Degree Sum
    scores_list.append(-neighbor_deg_sum)
    
    # 3. Min Coreness
    scores_list.append(-coreness)
    
    # 4. Spectral: prioritize vertices far from the cut boundary (large |fiedler|)
    #    These are in the "core" of a partition and less likely to conflict
    spectral_score = -np.abs(fiedler)
    scores_list.append(spectral_score)
    
    # 5. Combination
    # Normalize
    def norm(x):
        return x / (x.max() + 1e-9)
        
    d_n = norm(degrees)
    nds_n = norm(neighbor_deg_sum)
    c_n = norm(coreness)
    s_n = norm(np.abs(fiedler))
    
    # Heuristic: low degree and low coreness are good
    scores_list.append(-(0.5 * d_n + 0.5 * c_n))
    scores_list.append(-(0.7 * d_n + 0.3 * nds_n))
    scores_list.append(-(0.3 * d_n + 0.7 * nds_n))
    scores_list.append(-(d_n + nds_n + c_n))
    
    # Spectral combinations
    scores_list.append(-(0.5 * d_n + 0.5 * s_n))
    scores_list.append(-(0.3 * d_n + 0.3 * c_n + 0.4 * s_n))
    scores_list.append(-(0.2 * d_n + 0.2 * nds_n + 0.2 * c_n + 0.4 * s_n))
    
    # 6. Random
    scores_list.append(np.random.randn(n))
    
    return scores_list

def _local_search_order(adjacency, initial_order, deadline):
    """
    Improve the order via local search (swaps).
    We focus on swaps that might unblock a better vertex or block a worse one.
    """
    n = adjacency.shape[0]
    current_order = initial_order.copy()
    current_count, _ = _greedy_decode(adjacency, current_order)
    
    best_order = current_order.copy()
    best_count = current_count
    
    # Precompute adjacency rows for speed
    adj_rows = [adjacency[i] for i in range(n)]
    
    step = 0
    no_improve_count = 0
    max_no_improve = 500
    T = 1.0
    cooling_rate = 0.995
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            break
            
        # Choose a move: swap two random positions
        # Or swap a vertex with a later vertex to bring it forward
        i, j = np.random.choice(n, 2, replace=False)
        
        # Swap
        current_order[i], current_order[j] = current_order[j], current_order[i]
        
        new_count, _ = _greedy_decode(adjacency, current_order)
        
        delta = new_count - current_count
        
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current_count = new_count
            if new_count > best_count:
                best_count = new_count
                best_order = current_order.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Revert
            current_order[i], current_order[j] = current_order[j], current_order[i]
            no_improve_count += 1
            
        T *= cooling_rate
        if T < 0.01:
            T = 0.01
            
        step += 1
        
    return best_order, best_count

def _local_search_set(adjacency, initial_set, deadline):
    """
    Directly improve the independent set.
    """
    n = adjacency.shape[0]
    current_set = initial_set.copy()
    current_size = np.sum(current_set)
    
    best_set = current_set.copy()
    best_size = current_size
    
    # Helper: is vertex v independent from current_set?
    # v is independent if no neighbor of v is in current_set
    # i.e., adjacency[v] @ current_set == 0
    
    # Precompute neighbors
    neighbors = [np.where(adjacency[i])[0] for i in range(n)]
    
    while time.time() < deadline:
        # Try to add a vertex
        # Candidates: vertices not in set and not neighbors of any vertex in set
        # Efficient way:
        # blocked = set of vertices in set OR neighbors of set
        # We can maintain a "blocked" array.
        
        # Recompute blocked for simplicity (n=200 is small)
        blocked = np.zeros(n, dtype=bool)
        blocked[current_set] = True
        # For each v in set, mark neighbors
        for v in np.where(current_set)[0]:
            blocked[neighbors[v]] = True
            
        candidates = np.where(~blocked)[0]
        
        if len(candidates) > 0:
            # Pick a random candidate
            # Or pick the one with min degree among candidates?
            # Min degree is usually a good heuristic for adding
            cand_degrees = [len(neighbors[c]) for c in candidates]
            min_idx = np.argmin(cand_degrees)
            v = candidates[min_idx]
            
            # Add v
            current_set[v] = True
            current_size += 1
            
            if current_size > best_size:
                best_size = current_size
                best_set = current_set.copy()
                
        else:
            # No candidates to add. Try a swap?
            # Remove a vertex u from set, see if we can add 2 new vertices?
            # This is complex. Let's just try removing a random vertex and see if we can add more.
            # This is a "kick".
            if np.random.random() < 0.1:
                set_vertices = np.where(current_set)[0]
                if len(set_vertices) > 0:
                    u = set_vertices[np.random.randint(len(set_vertices))]
                    current_set[u] = False
                    current_size -= 1
                    # Now try to add as many as possible greedily
                    # This is essentially re-running a local greedy on the remaining
                    # But we only have time for a few steps.
                    # Let's just try to add one best candidate.
                    blocked = np.zeros(n, dtype=bool)
                    blocked[current_set] = True
                    for v in np.where(current_set)[0]:
                        blocked[neighbors[v]] = True
                    candidates = np.where(~blocked)[0]
                    if len(candidates) > 0:
                        cand_degrees = [len(neighbors[c]) for c in candidates]
                        min_idx = np.argmin(cand_degrees)
                        v = candidates[min_idx]
                        current_set[v] = True
                        current_size += 1
                        
                        # Try to add another
                        blocked = np.zeros(n, dtype=bool)
                        blocked[current_set] = True
                        for v in np.where(current_set)[0]:
                            blocked[neighbors[v]] = True
                        candidates = np.where(~blocked)[0]
                        if len(candidates) > 0:
                            cand_degrees = [len(neighbors[c]) for c in candidates]
                            min_idx = np.argmin(cand_degrees)
                            v = candidates[min_idx]
                            current_set[v] = True
                            current_size += 1
                            
                    if current_size > best_size:
                        best_size = current_size
                        best_set = current_set.copy()
                    else:
                        # Revert if it didn't help? 
                        # Actually, we modified current_set.
                        # It's safer to just continue from this state, 
                        # as it might be a better starting point for future kicks.
                        pass
            else:
                # Stuck, break out of this loop to save time?
                # Or just break.
                break
                
    return best_set, best_size

def priority(adjacency):
    """
    Computes priority scores for vertices to maximize the size of the 
    independent set found by the greedy decoder.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
    
    adjacency = np.asarray(adjacency, dtype=bool)
    
    # 1. Generate baselines
    baseline_scores = _baseline_scores(adjacency)
    
    best_count = -1
    best_order = None
    
    # Evaluate baselines
    for scores in baseline_scores:
        order = np.argsort(-scores, kind='stable')
        count, _ = _greedy_decode(adjacency, order)
        if count > best_count:
            best_count = count
            best_order = order.copy()
            
    # 2. Local Search on Order
    # Run multiple restarts if time permits
    restarts = 5
    for r in range(restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Start from best_order or a perturbed version
        if r == 0:
            init_order = best_order.copy()
        else:
            # Perturb best_order
            init_order = best_order.copy()
            for _ in range(10):
                i, j = np.random.choice(n, 2, replace=False)
                init_order[i], init_order[j] = init_order[j], init_order[i]
                
        # Run local search with a time slice
        slice_deadline = min(deadline - 10.0, time.time() + 20.0)
        ls_order, ls_count = _local_search_order(adjacency, init_order, slice_deadline)
        
        if ls_count > best_count:
            best_count = ls_count
            best_order = ls_order.copy()
            
    # 3. Local Search on Set (Polishing)
    if time.time() < deadline - 5.0:
        _, init_set = _greedy_decode(adjacency, best_order)
        ls_set, ls_count = _local_search_set(adjacency, init_set, deadline - 1.0)
        
        if ls_count > best_count:
            # Convert set to scores
            # We want the greedy decoder to pick exactly this set (or a superset).
            # If we assign high score to set vertices and 0 to others,
            # the decoder will pick the set vertices first.
            # Then it will pick any other available vertex with score 0.
            # This might add more vertices, which is good.
            scores = np.zeros(n)
            scores[ls_set] = 1.0
            return scores
            
    # 4. Convert best_order to scores
    scores = np.zeros(n)
    for idx, v in enumerate(best_order):
        scores[v] = n - idx
        
    return scores

# EVOLVE-BLOCK-END
