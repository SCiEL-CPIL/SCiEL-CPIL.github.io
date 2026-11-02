"""Kissing number in dimension 11 - find max independent set in conflict graph."""
import numpy as np
import time
import random
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

D = 11
DEADLINE = 145.0  # safety margin before 150s kill

def construct():
    start_time = time.time()
    
    # Generate candidate vectors: all nonzero integer vectors with coords in {-2,-1,0,1,2}
    # but limit to keep computation tractable. Use coords in {-1,0,1} first, then add {-2,2}.
    # Actually, let's use a structured approach:
    # Vectors with exactly 1 nonzero coord: ±e_i (11*2 = 22)
    # Vectors with exactly 2 nonzero coords: ±e_i ± e_j (C(11,2)*4 = 180)
    # Vectors with exactly 3 nonzero coords: ±e_i ± e_j ± e_k (C(11,3)*8 = 616)
    # Vectors with exactly 4 nonzero coords: C(11,4)*16 = 330*16 = 5280
    # Vectors with exactly 5 nonzero coords: C(11,5)*32 = 462*32 = 14784
    # Total with 1-5 nonzero: 22+180+616+5280+14784 = 20882
    
    # But we should also consider vectors with larger coordinates like 2.
    # Let's be more selective. The kissing number in 11D is known to be 504.
    # We need vectors that can form a kissing configuration.
    
    # Strategy: Generate vectors with small integer coordinates, build conflict graph,
    # find large independent set.
    
    # Generate candidate vectors
    candidates = []
    
    # All vectors with coords in {-1, 0, 1}, nonzero, up to 5 nonzero entries
    from itertools import combinations, product
    
    for n_nonzero in range(1, 6):
        for positions in combinations(range(D), n_nonzero):
            for signs in product([1, -1], repeat=n_nonzero):
                v = [0] * D
                for pos, sign in zip(positions, signs):
                    v[pos] = sign
                candidates.append(v)
    
    # Add vectors with a coordinate of 2
    for i in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[i] = 2 * sign
            candidates.append(v)
    
    # Add vectors with one coord = 2 and one coord = 1
    for i, j in combinations(range(D), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                v = [0] * D
                v[i] = 2 * si
                v[j] = 1 * sj
                candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    # Compute norms squared
    norms_sq = np.sum(candidates ** 2, axis=1)
    max_norm_sq = np.max(norms_sq)
    
    # Compute pairwise distances squared
    # dist^2 = ||a||^2 + ||b||^2 - 2*a.b
    # We need dist >= max_norm, i.e., dist^2 >= max_norm^2
    # So conflict if dist^2 < max_norm^2
    
    # Build conflict graph using matrix operations in batches to save memory
    conflict = np.zeros((n, n), dtype=bool)
    
    batch_size = 500
    for i_start in range(0, n, batch_size):
        i_end = min(i_start + batch_size, n)
        batch = candidates[i_start:i_end]
        # Compute distances from batch to all candidates
        # dist^2 = ||batch||^2 + ||cand||^2 - 2*batch@cand^T
        batch_norms_sq = norms_sq[i_start:i_end, None]
        all_norms_sq = norms_sq[None, :]
        dots = batch @ candidates.T
        dist_sq = batch_norms_sq + all_norms_sq - 2 * dots
        conflict[i_start:i_end, :] = dist_sq < max_norm_sq
    
    # The diagonal should be False (no self-conflict)
    np.fill_diagonal(conflict, False)
    
    # Now find large independent set in this graph
    # Independent set = set of vertices with no edges between them
    # = clique in complement graph
    
    # Use greedy + local search approach
    # First: greedy by degree (pick low-degree vertices first)
    
    def greedy_independent_set(adj, n_verts, time_limit=None):
        """Greedy: repeatedly pick vertex with minimum degree among remaining."""
        selected = []
        remaining = set(range(n_verts))
        deg = np.sum(adj, axis=1)
        
        while remaining:
            if time_limit and time.time() > time_limit:
                break
            # Find vertex with min degree in remaining
            min_deg = min(remaining, key=lambda v: deg[v])
            selected.append(min_deg)
            remaining.discard(min_deg)
            # Remove all neighbors of min_deg from remaining
            neighbors = np.where(adj[min_deg])[0]
            remaining -= set(neighbors.tolist())
            # Update degrees (approximate: just remove from consideration)
        
        return selected
    
    # Local search: try to add vertices or swap
    def local_search_improve(selected_set, adj, n_verts, time_limit=None):
        """Try to improve independent set by adding vertices or 1-for-1 swaps."""
        selected = set(selected_set)
        selected_list = list(selected)
        
        # Precompute which vertices can be added
        def can_add(v):
            for u in selected:
                if adj[v, u]:
                    return False
            return True
        
        # Try adding vertices
        improved = True
        while improved:
            if time_limit and time.time() > time_limit:
                break
            improved = False
            for v in range(n_verts):
                if v not in selected and can_add(v):
                    selected.add(v)
                    improved = True
                    if time_limit and time.time() > time_limit:
                        break
        
        # Try 1-for-1 swaps: remove u, add v if it increases size or enables more additions
        # Actually, since we just want to maximize size, 1-for-1 doesn't help directly.
        # But 2-for-1 or perturbations can help escape local optima.
        
        return list(selected)
    
    # Multi-restart approach with different orderings
    best_set = []
    
    # Restart 1: Greedy by original degree
    best_set = greedy_independent_set(conflict, n, time_limit=start_time + DEADLINE - 30)
    best_set = local_search_improve(best_set, conflict, n, time_limit=start_time + DEADLINE - 20)
    
    # Restart 2: Greedy by random perturbation of degrees
    for restart in range(20):
        if time.time() > start_time + DEADLINE - 20:
            break
        # Perturb degrees randomly
        perturbed_deg = np.sum(conflict, axis=1).astype(float)
        perturbed_deg += np.random.randn(n) * 0.1
        # Greedy using perturbed degrees
        selected = []
        remaining = set(range(n))
        while remaining:
            if time.time() > start_time + DEADLINE - 15:
                break
            min_deg = min(remaining, key=lambda v: perturbed_deg[v])
            selected.append(min_deg)
            remaining.discard(min_deg)
            neighbors = np.where(conflict[min_deg])[0]
            remaining -= set(neighbors.tolist())
        
        selected = local_search_improve(selected, conflict, n, time_limit=start_time + DEADLINE - 10)
        if len(selected) > len(best_set):
            best_set = selected
    
    # More aggressive local search with 2-for-1 moves
    def local_search_2for1(selected_set, adj, n_verts, time_limit=None):
        """Local search with 2-for-1 swaps and perturbations."""
        selected = set(selected_set)
        best_size = len(selected)
        
        for iteration in range(500):
            if time_limit and time.time() > time_limit:
                break
            
            # Try to add any vertex
            improved = False
            for v in range(n_verts):
                if v not in selected:
                    can = True
                    for u in selected:
                        if adj[v, u]:
                            can = False
                            break
                    if can:
                        selected.add(v)
                        improved = True
                        best_size = len(selected)
            
            if not improved:
                # Try 2-for-1: remove 2 vertices, try to add 3
                # Or perturb: remove 4 random vertices, then greedily add
                if len(selected) >= 4:
                    to_remove = random.sample(list(selected), min(4, len(selected)))
                    for v in to_remove:
                        selected.discard(v)
                    # Now greedily add
                    for v in range(n_verts):
                        if v not in selected:
                            can = True
                            for u in selected:
                                if adj[v, u]:
                                    can = False
                                    break
                            if can:
                                selected.add(v)
                    if len(selected) > best_size:
                        best_size = len(selected)
        
        return list(selected)
    
    # Apply 2-for-1 local search on best set
    if time.time() < start_time + DEADLINE - 10:
        best_set = local_search_2for1(best_set, conflict, n, time_limit=start_time + DEADLINE - 5)
    
    # More restarts with 2-for-1
    for restart in range(50):
        if time.time() > start_time + DEADLINE - 10:
            break
        # Random perturbation of degree ordering
        perturbed_deg = np.sum(conflict, axis=1).astype(float)
        perturbed_deg += np.random.randn(n) * 0.5
        selected = []
        remaining = set(range(n))
        while remaining:
            if time.time() > start_time + DEADLINE - 5:
                break
            min_deg = min(remaining, key=lambda v: perturbed_deg[v])
            selected.append(min_deg)
            remaining.discard(min_deg)
            neighbors = np.where(conflict[min_deg])[0]
            remaining -= set(neighbors.tolist())
        
        selected = local_search_2for1(selected, conflict, n, time_limit=start_time + DEADLINE - 2)
        if len(selected) > len(best_set):
            best_set = selected
    
    # Convert to vector array
    result = candidates[best_set]
    return result

# Quick test
if __name__ == "__main__":
    start = time.time()
    result = construct()
    print(f"Found {len(result)} vectors in {time.time() - start:.1f}s")
    # Verify
    if len(result) > 0:
        norms = np.linalg.norm(result, axis=1)
        max_norm = np.max(norms)
        # Check all pairwise distances
        min_dist = np.inf
        for i in range(len(result)):
            for j in range(i+1, len(result)):
                d = np.linalg.norm(result[i] - result[j])
                min_dist = min(min_dist, d)
        print(f"Max norm: {max_norm:.4f}, Min pairwise dist: {min_dist:.4f}")
        print(f"Valid: {min_dist >= max_norm - 1e-9}")
