import numpy as np
import itertools
import time
import sys

D = 11

def generate_weight_4_vectors():
    """Generate all vectors of weight 4 in {-1, 0, 1}^D"""
    candidates = []
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)
    return np.array(candidates, dtype=np.int8)

def greedy_mis(conflicts, deadline, rng):
    """
    Greedy Maximum Independent Set with degree-based ordering.
    """
    N = len(conflicts)
    alive = np.ones(N, dtype=bool)
    degrees = conflicts.sum(axis=1).astype(np.int32)
    
    selected_indices = []
    
    while np.any(alive):
        if time.time() > deadline:
            break
        
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
            
        # Find min degree among alive
        min_deg = np.min(degrees[alive_indices])
        candidates = alive_indices[degrees[alive_indices] == min_deg]
        
        # Random tie-breaking
        chosen = candidates[rng.integers(len(candidates))]
        selected_indices.append(chosen)
        
        # Find neighbors to remove
        neighbors = np.where(conflicts[chosen])[0]
        to_remove = np.unique(np.concatenate(([chosen], neighbors)))
        
        # Update alive
        alive[to_remove] = False
        
        # Update degrees for remaining alive nodes
        remaining_alive = np.where(alive)[0]
        if len(remaining_alive) > 0:
            removed_conflicts = conflicts[remaining_alive][:, to_remove].sum(axis=1)
            degrees[remaining_alive] -= removed_conflicts

    return selected_indices

def tomata_mis(adjacency, deadline, rng, initial_solution):
    """
    Tomita Maximum Independent Set algorithm.
    Returns the best independent set found before deadline.
    """
    N = adjacency.shape[0]
    best_size = len(initial_solution)
    best_set = list(initial_solution)
    
    # Precompute neighbors as sets for faster intersection?
    # Or just use numpy arrays?
    # For speed in Python, sets are often better for intersection if N is moderate.
    # But 5280 sets of size ~1000? Memory might be high.
    # Let's stick to numpy arrays and boolean masks.
    
    # Order vertices by degree descending
    degrees = adjacency.sum(axis=1)
    order = np.argsort(-degrees)
    
    # We will use a recursive function.
    # To avoid recursion depth issues, we might need to be careful.
    # sys.setrecursionlimit(10000)
    
    def search(P, current_set):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
            
        if len(P) == 0:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
            return
            
        # Upper bound: Greedy Coloring
        # Simple bound: len(P)
        # Better: Greedy Coloring
        # We'll use a simple greedy coloring on the subgraph induced by P.
        # To make it fast, we'll just use len(P) if N is large?
        # No, that's weak.
        # Let's try a simple coloring.
        
        # Color P
        # P is a numpy array of indices.
        # We need to color the graph induced by P.
        # Greedy: iterate P, assign smallest color not in neighbors.
        
        # This is O(|P|^2).
        # If |P| is large, this is slow.
        # Let's just use len(P) as UB for now to see if it works.
        # If UB is weak, we prune less.
        
        if len(current_set) + len(P) <= best_size:
            return
            
        # Pick a vertex u
        # Heuristic: pick vertex with max degree in P?
        # Or just the first one?
        # Tomita picks based on the coloring order.
        
        # Let's pick the first vertex in P.
        u = P[0]
        
        # Branch 1: Include u
        # New P = P \ N[u]
        # N[u] is neighbors of u.
        # We need P elements that are NOT neighbors of u and not u.
        
        # P is an array.
        # mask = ~adjacency[u][P]
        # new_P = P[mask]
        
        # Note: adjacency[u][u] is 0, so u is not in N[u] in terms of edge, 
        # but we must remove u from P.
        # So mask = ~adjacency[u][P] & (P != u)
        
        mask = ~adjacency[u][P].astype(bool)
        # Ensure u is removed
        # If u is in P, adjacency[u][u] is 0, so ~0 is 1. 
        # So we need to explicitly remove u.
        # Actually, standard MIS: N[u] includes u? 
        # Usually N(v) is open neighborhood. N[u] is closed.
        # If we include u, we remove N[u].
        # So we remove u and its neighbors.
        
        # adjacency[u] has 1 for neighbors. 0 for non-neighbors and self.
        # So ~adjacency[u] has 1 for non-neighbors and self.
        # We want to remove u and neighbors.
        # So we keep P elements that are NOT neighbors AND NOT u.
        
        # Let's check: if v is neighbor of u, adj[u][v]=1. ~1=0. Removed.
        # If v is u, adj[u][u]=0. ~0=1. Kept. 
        # So we must remove u explicitly.
        
        mask[u == P] = False # This doesn't work element-wise like that easily with numpy arrays of different lengths?
        # P is an array. u is a scalar.
        # We can do:
        # indices_to_remove = np.where(adjacency[u][P] == 1)[0]
        # And also remove index of u in P.
        
        # Simpler:
        # new_P = P[~adjacency[u][P].astype(bool)]
        # Then remove u from new_P if present.
        
        new_P = P[~adjacency[u][P].astype(bool)]
        # Remove u
        if u in new_P:
             # Find index of u in new_P
             idx_u = np.where(new_P == u)[0][0]
             new_P = np.delete(new_P, idx_u)
             
        search(new_P, current_set + [u])
        
        # Branch 2: Exclude u
        new_P2 = P[1:] # Remove u
        search(new_P2, current_set)
        
    # Initial P is all vertices
    P = np.arange(N)
    search(P, [])
    
    return best_set

def construct():
    t0 = time.time()
    deadline = t0 + 115
    
    vectors = generate_weight_4_vectors()
    N = len(vectors)
    
    if N == 0:
        return np.empty((0, D), dtype=np.int64)
        
    dots = vectors @ vectors.T
    conflicts = (dots > 2).astype(np.int8)
    np.fill_diagonal(conflicts, 0)
    
    best_set = []
    rng = np.random.default_rng(42)
    
    # Run greedy first
    current_set = greedy_mis(conflicts, deadline - 50, rng)
    if len(current_set) > len(best_set):
        best_set = current_set
        
    # Try Tomita with remaining time
    if time.time() < deadline - 5:
        tomata_deadline = deadline - 2
        improved_set = tomata_mis(conflicts, tomata_deadline, rng, best_set)
        if len(improved_set) > len(best_set):
            best_set = improved_set
            
    if not best_set:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        
    result_vectors = vectors[best_set]
    return result_vectors.astype(np.int64)
