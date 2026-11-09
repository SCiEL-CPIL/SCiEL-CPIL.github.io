"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations, product

def _norm_sq(v):
    return sum(x*x for x in v)

def _dist_sq(u, v):
    return sum((a-b)*(a-b) for a, b in zip(u, v))

def _make_candidates():
    """Generate candidate integer vectors in R^D with small norms."""
    candidates = []
    
    # Norm^2 = 2: pairs of +-1 (these are the minimal vectors of D_11)
    for i, j in combinations(range(D), 2):
        for si, sj in product((1,-1), (1,-1)):
            v = [0]*D
            v[i], v[j] = si, sj
            candidates.append(tuple(v))
    
    # Norm^2 = 4: vectors with one coordinate +-2
    for i in range(D):
        for s in (2, -2):
            v = [0]*D
            v[i] = s
            candidates.append(tuple(v))
    
    # Norm^2 = 4: vectors with four coordinates +-1
    for i, j, k, l in combinations(range(D), 4):
        for si, sj, sk, sl in product((1,-1),(1,-1),(1,-1),(1,-1)):
            v = [0]*D
            v[i], v[j], v[k], v[l] = si, sj, sk, sl
            candidates.append(tuple(v))
    
    return candidates

def _build_conflict_graph(candidates, deadline=130.0):
    """
    Build conflict graph: edge (i,j) exists if adding both violates the constraint.
    Constraint: for selected set S, min_{u,v in S, u!=v} ||u-v||^2 >= max_{w in S} ||w||^2.
    
    We'll use a conservative approach: assume max norm is the max norm in the entire candidate set,
    or better, build the graph incrementally. Actually, let's precompute using a fixed threshold.
    
    Strategy: Use the maximum possible norm in our candidate set as the threshold.
    This is conservative but gives a valid conflict graph.
    """
    start_time = time.time()
    n = len(candidates)
    norms_sq = [_norm_sq(v) for v in candidates]
    max_norm_sq = max(norms_sq)
    
    # Conflict: ||u - v||^2 < max_norm_sq means they can't both be in a set where
    # the max norm is max_norm_sq. But if we don't include the vector with max_norm_sq,
    # we could have a smaller threshold. This is complex, so let's use the full max.
    
    # Actually, a better approach: for each pair (i,j), they conflict if 
    # ||u_i - u_j||^2 < max(norms_sq[i], norms_sq[j])
    # This is the necessary and sufficient condition for a pair to be incompatible
    # in any set containing both (since the max norm of the set is at least max of the two).
    
    # Wait, that's not quite right either. If we have vectors with norms 2, 2, 4, and two
    # norm-2 vectors have distance^2 = 3, they're incompatible because max norm is 4.
    # But if we only select norm-2 vectors, they'd be fine.
    
    # For a safe conflict graph: i and j conflict if ||u_i - u_j||^2 < max(norms_sq[i], norms_sq[j])
    # This ensures that in ANY set containing both i and j, the constraint is violated.
    # This might be too conservative (missing valid sets) but guarantees validity.
    
    # Actually no: if ||u_i - u_j||^2 >= max(norms_sq[i], norms_sq[j]), then for any set S 
    # containing both i and j, we have min distance in S >= ||u_i - u_j||^2 >= max(norms_sq[i], norms_sq[j]).
    # But the max norm of S could be larger than max(norms_sq[i], norms_sq[j]), so the constraint 
    # requires min distance >= max_norm(S) which could be larger. So this condition is necessary 
    # but not sufficient for compatibility in a larger set.
    
    # The correct approach: we need to solve this as a constraint satisfaction where the 
    # threshold depends on the selected set. This is hard.
    
    # Alternative: fix the threshold to be the maximum norm in our candidate pool.
    # Then any valid set under this threshold is valid for the actual problem.
    # This is conservative but correct.
    
    conflict = np.zeros((n, n), dtype=bool)
    cand_arr = np.array(candidates, dtype=np.int64)
    
    for i in range(n):
        if time.time() > deadline:
            break
        # Vectorized: compute distances from candidate i to all others
        diffs = cand_arr - cand_arr[i]
        dist_sq = np.sum(diffs * diffs, axis=1)
        # Conflict with j if dist_sq[j] < max_norm_sq (and i != j)
        conflict[i, :] = dist_sq < max_norm_sq
        conflict[i, i] = False  # no self-conflict
    
    return conflict, max_norm_sq

def _greedy_mis(conflict, deadline=130.0, n_trials=50):
    """
    Greedy Maximum Independent Set with degree-based ordering and multiple restarts.
    """
    start_time = time.time()
    n = len(conflict)
    best_size = 0
    best_set = []
    
    # Precompute degrees
    degrees = np.sum(conflict, axis=1)
    
    for trial in range(n_trials):
        if time.time() > deadline - 3:
            break
        
        rng = random.Random(1000 + trial)
        
        # Create a priority: lower degree first, with random tie-breaking
        # Use a list of (random_key, degree, index) for ordering
        order = list(range(n))
        rng.shuffle(order)
        
        # Greedy: process in order, add if no conflict with already selected
        selected = []
        selected_set = set()
        
        # Sort by degree (with some randomization)
        # Use a weighted key: degree + random noise
        keys = [(degrees[i] + rng.random() * 0.5, i) for i in order]
        keys.sort(key=lambda x: x[0])
        sorted_order = [k[1] for k in keys]
        
        for idx in sorted_order:
            if time.time() > deadline - 1:
                break
            
            # Check if idx conflicts with any selected
            compatible = True
            for sel in selected:
                if conflict[idx, sel]:
                    compatible = False
                    break
            
            if compatible:
                selected.append(idx)
                selected_set.add(idx)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
    
    return best_set

def _local_search(conflict, initial_set, deadline=130.0, n_iterations=1000):
    """
    Local search: try to add vectors or swap to improve the set.
    """
    start_time = time.time()
    selected = set(initial_set)
    selected_list = list(initial_set)
    
    # Precompute for each unselected node, which selected nodes it conflicts with
    n = len(conflict)
    
    for iteration in range(n_iterations):
        if time.time() > deadline - 2:
            break
        
        # Try to add a random unselected node
        unselected = [i for i in range(n) if i not in selected]
        if not unselected:
            break
        
        # Find nodes that can be added (no conflict with selected)
        addable = []
        for i in unselected:
            if time.time() > deadline - 2:
                break
            if not any(conflict[i, s] for s in selected):
                addable.append(i)
        
        if addable:
            # Add one at random
            new_node = random.choice(addable)
            selected.add(new_node)
            selected_list.append(new_node)
        else:
            # Try a swap: remove one selected, add one or two unselected
            if len(selected) < 2:
                break
            
            # Pick a random selected node to potentially remove
            remove_candidates = list(selected)
            rng = random.Random(iteration)
            rng.shuffle(remove_candidates)
            
            improved = False
            for rem in remove_candidates[:10]:  # try up to 10 removals
                if time.time() > deadline - 2:
                    break
                
                # If we remove rem, which nodes become addable?
                new_selected = selected - {rem}
                addable_after_remove = []
                for i in unselected:
                    if i == rem:
                        continue
                    if not any(conflict[i, s] for s in new_selected):
                        addable_after_remove.append(i)
                
                if len(addable_after_remove) > 1:
                    # We can add more than we removed
                    # Add the first few
                    new_selected_list = list(new_selected)
                    for add in addable_after_remove[:2]:
                        new_selected.add(add)
                        new_selected_list.append(add)
                    
                    if len(new_selected) > len(selected):
                        selected = new_selected
                        selected_list = new_selected_list
                        improved = True
                        break
            
            if not improved:
                # No improvement, continue
                pass
    
    return selected_list

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm. Return as int64 array of shape (k, d).
    """
    deadline = 135.0  # 15s safety margin before 150s kill
    start_time = time.time()
    
    # Generate candidates
    candidates = _make_candidates()
    n_cand = len(candidates)
    
    # Build conflict graph
    conflict, max_norm_sq = _build_conflict_graph(candidates, deadline=deadline)
    
    # Greedy MIS with multiple trials
    best_set = _greedy_mis(conflict, deadline=deadline, n_trials=30)
    
    # Local search to improve
    if time.time() < deadline - 5:
        best_set = _local_search(conflict, best_set, deadline=deadline, n_iterations=500)
    
    # Convert to vectors
    best_vectors = [candidates[i] for i in best_set]
    
    # Verify the result satisfies the constraint
    if len(best_vectors) >= 2:
        actual_max_norm_sq = max(_norm_sq(v) for v in best_vectors)
        valid = True
        for i, j in combinations(range(len(best_vectors)), 2):
            if _dist_sq(best_vectors[i], best_vectors[j]) < actual_max_norm_sq:
                valid = False
                break
        if not valid:
            # Fallback: use only norm^2=2 vectors
            best_vectors = []
            for i, j in combinations(range(D), 2):
                for si, sj in product((1,-1), (1,-1)):
                    v = [0]*D
                    v[i], v[j] = si, sj
                    best_vectors.append(v)
    
    return np.array(best_vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
