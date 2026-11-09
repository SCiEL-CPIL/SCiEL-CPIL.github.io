"""Strategy: Spectral-partition guided greedy with fast local search.
Uses the Fiedler vector to identify the graph's sparsest cut, then prioritizes
vertices in the smaller partition to minimize cross-cut conflicts.
Combined with fast local search on vertex ordering and set-based polishing.
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
    
    for v in order:
        if not available[v]:
            continue
        selected[v] = True
        count += 1
        # Mark neighbors as unavailable
        available[adjacency[v]] = False
        
    return count, selected

def _greedy_decode_fast(neighbors, order):
    """
    Fast greedy decode using precomputed neighbor lists.
    neighbors: list of arrays, neighbors[i] = array of neighbor indices of i
    order: array of vertex indices in selection order
    Returns: (count, selected_mask)
    """
    n = len(neighbors)
    selected = np.zeros(n, dtype=bool)
    available = np.ones(n, dtype=bool)
    count = 0
    
    for v in order:
        if not available[v]:
            continue
        selected[v] = True
        count += 1
        for nb in neighbors[v]:
            available[nb] = False
            
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
    Returns: (fiedler_vector, lambda_2, partition_mask)
    The Fiedler vector identifies the sparsest cut.
    partition_mask: True for vertices in the smaller partition.
    """
    try:
        # Build Laplacian L = D - A
        degrees = adjacency.sum(axis=1)
        D = np.diag(degrees)
        L = D - adjacency.astype(float)
        
        # Convert to sparse for efficiency
        L_sparse = sp.csr_matrix(L)
        
        # Compute smallest eigenvalues using shift-invert mode
        k = min(4, n - 1)
        if k < 2:
            return np.zeros(n), 0.0, np.ones(n, dtype=bool)
            
        eigvals, eigvecs = spla.eigsh(L_sparse, k=k, sigma=-1e-6, which='LM')
        
        # Sort by eigenvalue
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Fiedler vector is the eigenvector for lambda_2
        if len(eigvals) >= 2:
            fiedler = eigvecs[:, 1]
            lambda_2 = eigvals[1]
        else:
            fiedler = eigvecs[:, -1]
            lambda_2 = eigvals[-1]
            
        # Normalize
        norm_f = np.linalg.norm(fiedler)
        if norm_f > 1e-10:
            fiedler = fiedler / norm_f
        
        # Determine partition: split at median of Fiedler values
        median_f = np.median(fiedler)
        # Smaller partition is the one with fewer vertices
        below = fiedler < median_f
        above = fiedler >= median_f
        if np.sum(below) <= np.sum(above):
            partition_mask = below
        else:
            partition_mask = above
            
        return fiedler, lambda_2, partition_mask
        
    except Exception:
        return np.zeros(n), 0.0, np.ones(n, dtype=bool)

def _baseline_scores(adjacency):
    """
    Generate several baseline score vectors.
    """
    n = adjacency.shape[0]
    degrees = adjacency.sum(axis=1).astype(float)
    neighbor_deg_sum = adjacency @ degrees
    coreness = _compute_core_numbers(adjacency).astype(float)
    
    # Compute spectral features
    fiedler, lambda_2, partition_mask = _compute_spectral_features(adjacency, n)
    
    scores_list = []
    
    # 1. Min Degree (classic)
    scores_list.append(-degrees)
    
    # 2. Min Neighbor Degree Sum
    scores_list.append(-neighbor_deg_sum)
    
    # 3. Min Coreness
    scores_list.append(-coreness)
    
    # 4. Spectral: prioritize vertices in the smaller partition
    #    Strongly favor smaller partition vertices, then by |fiedler| within partition
    part_bonus = np.where(partition_mask, 1.0, 0.0)
    spectral_score = part_bonus * 2.0 - np.abs(fiedler)
    scores_list.append(spectral_score)
    
    # 5. Normalize
    def norm(x):
        mx = x.max()
        if mx < 1e-9:
            return np.zeros_like(x)
        return x / mx
        
    d_n = norm(degrees)
    nds_n = norm(neighbor_deg_sum)
    c_n = norm(coreness)
    s_n = norm(np.abs(fiedler))
    p_n = part_bonus  # already 0/1
    
    # Heuristic: low degree and low coreness are good
    scores_list.append(-(0.5 * d_n + 0.5 * c_n))
    scores_list.append(-(0.7 * d_n + 0.3 * nds_n))
    scores_list.append(-(0.3 * d_n + 0.7 * nds_n))
    scores_list.append(-(d_n + nds_n + c_n))
    
    # Spectral combinations with partition bonus
    scores_list.append(p_n * 2.0 - (0.5 * d_n + 0.5 * s_n))
    scores_list.append(p_n * 3.0 - (0.3 * d_n + 0.3 * c_n + 0.4 * s_n))
    scores_list.append(p_n * 2.5 - (0.2 * d_n + 0.2 * nds_n + 0.2 * c_n + 0.4 * s_n))
    scores_list.append(p_n * 4.0 - s_n)  # Strong partition preference
    scores_list.append(p_n * 1.5 - (0.6 * d_n + 0.4 * s_n))
    
    # 6. Random
    scores_list.append(np.random.randn(n))
    
    return scores_list

def _local_search_order(adjacency, neighbors, initial_order, deadline):
    """
    Improve the order via local search with targeted swaps.
    """
    n = adjacency.shape[0]
    current_order = initial_order.copy()
    current_count, _ = _greedy_decode_fast(neighbors, current_order)
    
    best_order = current_order.copy()
    best_count = current_count
    
    step = 0
    no_improve_count = 0
    max_no_improve = 300
    T = 1.0
    cooling_rate = 0.99
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            break
            
        # Targeted move: pick a vertex that was skipped (blocked) and swap it
        # with a vertex that was selected, to try to unblock it
        # Or just do random swap for simplicity
        i, j = np.random.choice(n, 2, replace=False)
        
        vi = current_order[i]
        vj = current_order[j]
        
        # Swap
        current_order[i] = vj
        current_order[j] = vi
        
        new_count, _ = _greedy_decode_fast(neighbors, current_order)
        
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
            current_order[i] = vi
            current_order[j] = vj
            no_improve_count += 1
            
        T *= cooling_rate
        if T < 0.01:
            T = 0.01
            
        step += 1
        
    return best_order, best_count

def _local_search_set(adjacency, neighbors, initial_set, deadline):
    """
    Directly improve the independent set using incremental blocked set.
    """
    n = adjacency.shape[0]
    current_set = initial_set.copy()
    current_size = int(np.sum(current_set))
    
    best_set = current_set.copy()
    best_size = current_size
    
    # Maintain blocked array incrementally
    blocked = np.zeros(n, dtype=bool)
    blocked[current_set] = True
    for v in np.where(current_set)[0]:
        for nb in neighbors[v]:
            blocked[nb] = True
    
    stuck_count = 0
    max_stuck = 100
    
    while time.time() < deadline:
        if stuck_count > max_stuck:
            break
            
        candidates = np.where(~blocked)[0]
        
        if len(candidates) > 0:
            # Pick min degree candidate
            cand_degrees = np.array([len(neighbors[c]) for c in candidates])
            min_idx = np.argmin(cand_degrees)
            v = candidates[min_idx]
            
            # Add v
            current_set[v] = True
            current_size += 1
            blocked[v] = True
            for nb in neighbors[v]:
                blocked[nb] = True
            
            if current_size > best_size:
                best_size = current_size
                best_set = current_set.copy()
            stuck_count = 0
        else:
            # No candidates. Try a kick: remove a vertex and re-add greedily
            set_vertices = np.where(current_set)[0]
            if len(set_vertices) == 0:
                break
                
            # Remove a random vertex
            u = set_vertices[np.random.randint(len(set_vertices))]
            current_set[u] = False
            current_size -= 1
            
            # Unblock u and its neighbors that are now unblocked
            blocked[u] = False
            for nb in neighbors[u]:
                if not current_set[nb]:
                    # Check if nb is still blocked by other set vertices
                    is_blocked = False
                    for nb2 in neighbors[nb]:
                        if current_set[nb2]:
                            is_blocked = True
                            break
                    if not is_blocked:
                        blocked[nb] = False
            
            # Greedily add as many as possible
            for _ in range(5):
                cands = np.where(~blocked)[0]
                if len(cands) == 0:
                    break
                cand_degrees = np.array([len(neighbors[c]) for c in cands])
                min_idx = np.argmin(cand_degrees)
                v = cands[min_idx]
                current_set[v] = True
                current_size += 1
                blocked[v] = True
                for nb in neighbors[v]:
                    blocked[nb] = True
            
            if current_size > best_size:
                best_size = current_size
                best_set = current_set.copy()
                stuck_count = 0
            else:
                stuck_count += 1
                
    return best_set, best_size

def priority(adjacency):
    """
    Computes priority scores for vertices to maximize the size of the 
    independent set found by the greedy decoder.
    """
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin (evaluator kills at 150s)
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
    
    adjacency = np.asarray(adjacency, dtype=bool)
    
    # Precompute neighbor lists for fast decoding
    neighbors = [np.where(adjacency[i])[0] for i in range(n)]
    
    # 1. Generate baselines
    baseline_scores = _baseline_scores(adjacency)
    
    best_count = -1
    best_order = None
    
    # Evaluate baselines
    for scores in baseline_scores:
        order = np.argsort(-scores, kind='stable')
        count, _ = _greedy_decode_fast(neighbors, order)
        if count > best_count:
            best_count = count
            best_order = order.copy()
            
    # 2. Local Search on Order
    # Run multiple restarts if time permits
    restarts = 8
    for r in range(restarts):
        if time.time() > deadline - 15.0:
            break
            
        # Start from best_order or a perturbed version
        if r == 0:
            init_order = best_order.copy()
        else:
            # Perturb best_order
            init_order = best_order.copy()
            num_swaps = min(15, n // 10)
            for _ in range(num_swaps):
                i, j = np.random.choice(n, 2, replace=False)
                init_order[i], init_order[j] = init_order[j], init_order[i]
                
        # Run local search with a time slice
        slice_deadline = min(deadline - 15.0, time.time() + 12.0)
        ls_order, ls_count = _local_search_order(adjacency, neighbors, init_order, slice_deadline)
        
        if ls_count > best_count:
            best_count = ls_count
            best_order = ls_order.copy()
            
    # 3. Local Search on Set (Polishing)
    if time.time() < deadline - 5.0:
        _, init_set = _greedy_decode_fast(neighbors, best_order)
        ls_set, ls_count = _local_search_set(adjacency, neighbors, init_set, deadline - 2.0)
        
        if ls_count > best_count:
            # Convert set to scores: high score for set vertices, 0 for others
            # The decoder will pick set vertices first, then any remaining available
            scores = np.zeros(n)
            scores[ls_set] = 1.0
            return scores
            
    # 4. Convert best_order to scores
    scores = np.zeros(n)
    for idx, v in enumerate(best_order):
        scores[v] = n - idx
        
    return scores

# EVOLVE-BLOCK-END
