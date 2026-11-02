"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit
    
    # Generate Weight 3 and Weight 4 vectors
    candidates = []
    norms_sq = []
    
    # Weight 3
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            norms_sq.append(3)
            
    # Weight 4
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
            norms_sq.append(4)
            
    candidates = np.array(candidates, dtype=np.int64)
    norms_sq = np.array(norms_sq, dtype=np.int64)
    n_candidates = len(candidates)
    
    # Compute dot products
    dots = np.dot(candidates, candidates.T)
    
    # Determine conflicts
    # R^2 = 4 (since we include W4, max norm is 2)
    # dist_sq[i, j] = norms_sq[i] + norms_sq[j] - 2*dots[i, j]
    # Conflict if dist_sq[i, j] < 4
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    # We only care about i != j
    # conflict[i, j] = dist_sq[i, j] < 4
    conflicts = dist_sq < 4
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbors for each node
    # neighbors[i] is a set of j such that conflicts[i, j] is True
    neighbors = [set() for _ in range(n_candidates)]
    for i in range(n_candidates):
        # Use numpy to find indices where conflicts[i] is True
        neigh_indices = np.where(conflicts[i])[0]
        neighbors[i] = set(neigh_indices.tolist())
        
    def greedy_min_degree(order=None):
        """
        Greedy selection picking the vertex with the minimum number of 
        conflicts with the *remaining* available vertices.
        If order is provided, it's ignored in favor of dynamic min-degree, 
        but we can use order to break ties or initialize.
        Here we use dynamic min-degree.
        """
        available = set(range(n_candidates))
        selected = []
        
        while available:
            if time.time() > deadline:
                break
            
            # Find vertex with min degree in available subgraph
            best_v = None
            min_deg = float('inf')
            
            # To speed up, we can sample if available is large, 
            # but n_candidates ~ 6600 is small enough for linear scan if optimized.
            # However, checking len(neighbors[v] & available) is O(deg).
            # Sum of degrees is 2 * E. E is roughly N * (conflict_ratio).
            
            # Optimization: Just pick random? No, Min-Degree is better.
            # Let's iterate through available.
            for v in available:
                # Count neighbors in available
                # This is slow in Python if done naively.
                # Let's just use random selection for now to ensure speed, 
                # or a simpler heuristic: Min total degree (static).
                pass
            
            # Let's use Static Min-Degree as a proxy, or Random.
            # Actually, let's try a hybrid: 
            # 1. Pick a random subset of available (e.g., 100) and pick min-degree among them.
            if len(available) > 100:
                sample = random.sample(list(available), 100)
            else:
                sample = list(available)
                
            best_v = None
            min_deg = float('inf')
            for v in sample:
                deg = len(neighbors[v] & available)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            
            if best_v is None:
                break
                
            selected.append(best_v)
            available.discard(best_v)
            # Remove neighbors of best_v from available
            for n in neighbors[best_v]:
                if n in available:
                    available.discard(n)
                    
        # Fill pass
        # Try to add any remaining available (which are now all non-conflicting with selected? 
        # No, 'available' was reduced by removing neighbors. 
        # So any v in 'available' is non-conflicting with ALL selected?
        # Yes, because if v conflicted with any selected u, v would have been removed when u was added.
        # So we can add ALL remaining available?
        # Wait. If we remove neighbors, we ensure future picks don't conflict with past picks.
        # So yes, any v left in 'available' is compatible with all in 'selected'.
        # So we can just add them all?
        # No, they might conflict with *each other*.
        # But the greedy loop ended because 'available' was empty?
        # No, the loop breaks when 'available' is empty.
        # So there are no vectors left.
        # The "Fill" pass is only useful if we didn't remove neighbors? 
        # Standard MIS greedy removes neighbors.
        # So 'selected' is an independent set.
        # Are there vertices NOT in selected and NOT in available? 
        # No, we started with all, and removed selected and their neighbors.
        # So selected + available = All? No.
        # Selected are removed. Neighbors of selected are removed.
        # So 'available' contains vertices that are NOT selected and NOT neighbors of any selected.
        # So they are compatible with selected.
        # But they might conflict with each other.
        # So we should run a greedy on 'available' as well?
        # But the while loop continues until available is empty.
        # So it will pick from available, add to selected, remove its neighbors from available.
        # So it handles this.
        
        return selected

    best_selected = []
    np.random.seed(42)
    random.seed(42)
    
    # Run multiple trials with randomization in the sampling step
    num_trials = 200
    for trial in range(num_trials):
        if time.time() > deadline - 5.0:
            break
            
        # Introduce randomness by shuffling the order in which we consider vertices?
        # The current greedy uses random sampling for min-degree.
        # To diversify, we can bias the sampling.
        
        sel = greedy_min_degree()
        if len(sel) > len(best_selected):
            best_selected = sel
            
    if not best_selected:
        v = [0] * D
        v[0] = 1
        v[1] = 1
        return np.array([v], dtype=np.int64)
    
    return np.array(best_selected, dtype=np.int64)
# EVOLVE-BLOCK-END
