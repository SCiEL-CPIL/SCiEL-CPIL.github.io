import numpy as np
import itertools
import time

D = 11

def _generate_vectors_by_norm_sq(max_norm_sq):
    """Generate all nonzero integer vectors in R^D with squared norm <= max_norm_sq."""
    vecs = []
    
    def gen(idx, remaining_sq, current):
        if idx == D:
            if remaining_sq == 0 and any(x != 0 for x in current):
                vecs.append(tuple(current))
            return
        # Option: current[idx] = 0
        gen(idx + 1, remaining_sq, current)
        # Option: current[idx] = +/- val
        max_val = int(np.sqrt(remaining_sq))
        for val in range(1, max_val + 1):
            current[idx] = val
            gen(idx + 1, remaining_sq - val * val, current)
            current[idx] = -val
            gen(idx + 1, remaining_sq - val * val, current)
        current[idx] = 0
    
    gen(0, max_norm_sq, [0] * D)
    return np.array(list(set(vecs)), dtype=np.int64)

def _build_conflict_matrix(candidates, min_dist_sq):
    """
    Build a conflict matrix where conflict[i,j] = True if vectors i and j
    cannot coexist (distance < min_dist_sq).
    """
    n = len(candidates)
    # Compute pairwise squared distances
    # dist_sq[i,j] = ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j
    norms_sq = np.sum(candidates ** 2, axis=1)
    dot = candidates @ candidates.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dot
    
    # Conflict if dist_sq < min_dist_sq (and i != j)
    conflict = dist_sq < min_dist_sq
    np.fill_diagonal(conflict, False)
    return conflict

def _greedy_min_degree(conflict, weights, rng):
    """
    Minimum-degree greedy: repeatedly pick the vertex with the fewest
    conflicts among remaining candidates, weighted by (degree / weight)
    to prefer high-weight, low-degree vertices.
    """
    n = conflict.shape[0]
    remaining = set(range(n))
    selected = []
    
    while remaining:
        # For each remaining vertex, count conflicts with other remaining vertices
        # and compute a score = degree / weight (lower is better)
        best_v = None
        best_score = float('inf')
        
        for v in remaining:
            deg = sum(1 for u in remaining if u != v and conflict[v, u])
            w = weights[v]
            score = deg / max(w, 1e-9)
            # Add small random tie-breaker
            score += rng.random() * 0.01
            if score < best_score:
                best_score = score
                best_v = v
        
        if best_v is None:
            break
        
        selected.append(best_v)
        remaining.discard(best_v)
    
    return selected

def _swap_local_search(conflict, weights, selected, rng, max_swaps=500):
    """
    Local search: try to swap one selected vertex with one unselected vertex
    if it improves the objective (sum of weights).
    """
    n = conflict.shape[0]
    selected_set = set(selected)
    current_weight = sum(weights[v] for v in selected)
    improved = True
    swaps_done = 0
    
    while improved and swaps_done < max_swaps:
        improved = False
        selected_list = list(selected)
        
        # Shuffle order for diversity
        rng.shuffle(selected_list)
        
        for i, v_out in enumerate(selected_list):
            if v_out not in selected_set:
                continue
            # Try swapping v_out with each unselected vertex v_in
            # v_in must not conflict with any other selected vertex (except v_out)
            # and the new set must be valid
            
            # Find candidates: unselected vertices that don't conflict with
            # selected_set - {v_out}
            others = [u for u in selected_set if u != v_out]
            for v_in in range(n):
                if v_in in selected_set:
                    continue
                # Check v_in doesn't conflict with others
                if any(conflict[v_in, u] for u in others):
                    continue
                # Also need: removing v_out and adding v_in is valid
                # Since v_in doesn't conflict with others, and we remove v_out,
                # the new set is valid as long as v_in isn't in selected_set (already checked)
                new_weight = current_weight - weights[v_out] + weights[v_in]
                if new_weight > current_weight:
                    selected_set.discard(v_out)
                    selected_set.add(v_in)
                    current_weight = new_weight
                    improved = True
                    swaps_done += 1
                    break
        
    return list(selected_set), current_weight

def _spectral_seed(conflict, weights, rng):
    """
    Use spectral relaxation to get an initial seed set.
    Compute the adjacency matrix of the compatibility graph (non-conflict),
    find the principal eigenvector, and take the top-k components.
    """
    n = conflict.shape[0]
    # Compatibility matrix: 1 if no conflict, 0 if conflict (excluding diagonal)
    compat = (~conflict).astype(float)
    np.fill_diagonal(compat, 0)
    
    try:
        # Use power iteration for the top eigenvector
        x = np.ones(n) / np.sqrt(n)
        for _ in range(50):
            x_new = compat @ x
            norm = np.linalg.norm(x_new)
            if norm < 1e-12:
                break
            x = x_new / norm
        # Take top k vertices by eigenvector component, weighted
        scores = x * weights
        indices = np.argsort(scores)[::-1]
        return [int(i) for i in indices[:min(50, n)]]
    except:
        return []

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    # Generate all vectors with norm squared up to 4 (weight 2, 3, 4 vectors)
    # Weight-2: norm_sq=2, weight=2 (can coexist with more)
    # Weight-3: norm_sq=3, weight=1
    # Weight-4: norm_sq=4, weight=1
    try:
        all_candidates = _generate_vectors_by_norm_sq(4)
    except Exception:
        all_candidates = None
    
    if all_candidates is None or len(all_candidates) == 0:
        # Fallback: just weight-2 vectors
        try:
            all_candidates = _generate_vectors_by_norm_sq(2)
        except Exception:
            return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
    
    # Minimum distance squared: use the maximum norm squared among selected
    # For a mixed set, the constraint is min pairwise dist >= max norm
    # This is complex; instead, use a fixed min_dist_sq = 4 (norm_sq of weight-2 vectors)
    # Actually, the task says: min pairwise distance >= max vector norm
    # So if we have weight-2 vectors (norm=sqrt(2)) and weight-4 vectors (norm=2),
    # max norm = 2, so min pairwise dist must be >= 2, i.e. dist_sq >= 4.
    # Let's use min_dist_sq = 4 for all vectors.
    
    min_dist_sq = 4
    
    try:
        conflict = _build_conflict_matrix(all_candidates, min_dist_sq)
    except Exception:
        # Fallback to simple weight-2 only
        k2 = _generate_vectors_by_norm_sq(2)
        return k2
    
    # Assign weights: weight-2 vectors get weight 2, others get weight 1
    weights = np.ones(len(all_candidates), dtype=float)
    norms_sq = np.sum(all_candidates ** 2, axis=1)
    weights[norms_sq <= 2] = 2.0
    
    rng = np.random.RandomState(42)
    
    best_selected = []
    best_weight = 0
    best_vectors = None
    
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Try different initialization strategies
        if restart % 3 == 0:
            # Spectral seed
            seed = _spectral_seed(conflict, weights, rng)
        elif restart % 3 == 1:
            # Random seed
            seed = list(rng.choice(len(all_candidates), size=min(20, len(all_candidates)), replace=False))
        else:
            # Weight-2 priority seed
            w2_indices = [i for i in range(len(all_candidates)) if norms_sq[i] <= 2]
            seed = list(rng.choice(w2_indices, size=min(20, len(w2_indices)), replace=False)) if w2_indices else []
        
        # Greedy min-degree from seed
        try:
            # Start with seed, then extend
            selected_set = set(seed)
            # Remove conflicts within seed
            valid_seed = []
            for v in seed:
                if all(not conflict[v, u] for u in valid_seed):
                    valid_seed.append(v)
            selected_set = set(valid_seed)
            
            # Extend greedily
            remaining = set(range(len(all_candidates))) - selected_set
            while remaining:
                if time.time() > deadline:
                    break
                best_v = None
                best_score = float('inf')
                for v in remaining:
                    deg = sum(1 for u in remaining if u != v and conflict[v, u])
                    # Count conflicts with selected too
                    sel_conf = sum(1 for u in selected_set if conflict[v, u])
                    if sel_conf > 0:
                        continue
                    score = (deg + sel_conf) / max(weights[v], 1e-9)
                    score += rng.random() * 0.01
                    if score < best_score:
                        best_score = score
                        best_v = v
                if best_v is None:
                    break
                selected_set.add(best_v)
                remaining.discard(best_v)
            
            selected = list(selected_set)
            
            # Local search
            if time.time() < deadline - 5:
                selected, cur_weight = _swap_local_search(
                    conflict, weights, selected, rng, 
                    max_swaps=200 if restart < 5 else 100
                )
            else:
                cur_weight = sum(weights[v] for v in selected)
            
            if cur_weight > best_weight:
                best_weight = cur_weight
                best_selected = selected
                best_vectors = all_candidates[selected]
                
        except Exception:
            continue
    
    if best_vectors is not None and len(best_vectors) > 0:
        return best_vectors.astype(np.int64)
    
    # Fallback
    k2 = _generate_vectors_by_norm_sq(2)
    if k2 is not None and len(k2) > 0:
        return k2
    return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
