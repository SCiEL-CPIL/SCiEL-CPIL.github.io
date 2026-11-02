"""Rewritten program for kissing number in dimension 11."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # Generate W4 vectors (weight 4, entries in {-1,0,1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    # Generate W1 vectors (weight 1, entries in {-2,0,2}, norm^2=4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)

    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)

    N4 = len(vectors_v4)

    if N4 == 0:
        return vectors_v1

    # Compute conflict matrix for W4 vectors
    # Conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbors for fast updates
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)

    rng = np.random.default_rng(42)

    def soft_sa(init_x, max_time, lambda_pen=2.0):
        """
        Improved Soft-constraint Simulated Annealing.
        Energy E = -size + lambda * num_conflicts
        Uses adaptive temperature and block moves.
        """
        x = init_x.copy()
        # conflict_counts[i] = number of neighbors of i that are in x
        conflict_counts = np.zeros(N4, dtype=np.int32)
        for i in np.where(x)[0]:
            conflict_counts[neighbors[i]] += 1
        
        # Calculate initial energy
        size = x.sum()
        num_conflicts = np.sum(conflict_counts) // 2
        current_E = -size + lambda_pen * num_conflicts
        
        best_E = current_E
        best_x = x.copy()
        
        # Adaptive Temperature
        T = 5.0
        T_min = 0.01
        # Cooling factor per step, adjusted for N4
        cooling = 0.999
        
        sa_deadline = time.time() + max_time
        
        # Precompute list of all indices for faster random access
        all_indices = np.arange(N4)
        
        while time.time() < sa_deadline:
            # Pick random vertex to flip
            i = rng.integers(N4)
            
            # Calculate delta E
            if x[i]:
                # Flip 1 -> 0
                delta_E = 1 - lambda_pen * conflict_counts[i]
            else:
                # Flip 0 -> 1
                delta_E = -1 + lambda_pen * conflict_counts[i]
            
            # Acceptance
            if delta_E < 0 or (T > 0 and rng.random() < np.exp(-delta_E / T)):
                # Apply flip
                x[i] = not x[i]
                
                # Update conflict counts for neighbors
                nbrs = neighbors[i]
                if x[i]:
                    conflict_counts[nbrs] += 1
                else:
                    conflict_counts[nbrs] -= 1
                
                # Update energy
                current_E += delta_E
                
                if current_E < best_E:
                    best_E = current_E
                    best_x = x.copy()
            
            T *= cooling
            if T < T_min:
                T = T_min

        return best_x

    def greedy_init():
        """Improved Greedy initialization using randomization."""
        x = np.zeros(N4, dtype=bool)
        # Use a randomized greedy approach
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy()
        
        while np.any(alive):
            alive_idx = np.where(alive)[0]
            if len(alive_idx) == 0: break
            
            # Select a few candidates to increase diversity
            # Pick the one with minimum degree among a random subset
            k = min(5, len(alive_idx))
            if k < len(alive_idx):
                candidates = rng.choice(alive_idx, k, replace=False)
            else:
                candidates = alive_idx
            
            # Find min degree among candidates
            degs = degrees[candidates]
            min_deg = np.min(degs)
            cands = candidates[degs == min_deg]
            i = cands[rng.integers(len(cands))]
            
            x[i] = True
            # Remove i and its neighbors from alive
            to_remove = np.unique(np.concatenate(([i], neighbors[i])))
            alive[to_remove] = False
            
            # Update degrees for remaining alive nodes
            rem = np.where(alive)[0]
            if len(rem) > 0:
                # Only update if necessary for performance, but N4 is small enough
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return x

    def repair(x_res):
        """
        Greedy MIS repair on the subgraph induced by selected nodes.
        This is more effective than sequential removal.
        """
        selected = np.where(x_res)[0]
        if len(selected) == 0:
            return x_res
        
        # Build subgraph indices
        # We want to find MIS in the graph induced by 'selected'
        # Use a greedy heuristic: repeatedly pick node with min degree in the subgraph
        in_set = np.zeros(N4, dtype=bool)
        remaining = set(selected.tolist())
        
        # To speed up, we can just use the global neighbor list and filter
        # But for 5000 nodes, a simple loop is fine if optimized
        
        # Efficient greedy:
        # Maintain a list of active nodes
        active = list(selected)
        active_set = set(active)
        
        # Calculate degrees within the subgraph
        deg = {}
        for u in active:
            deg[u] = 0
        for u in active:
            for v in neighbors[u]:
                if v in active_set:
                    deg[u] += 1
        
        result_indices = []
        
        while active:
            # Pick node with min degree
            # Optimization: if many have same min degree, pick random
            min_deg = min(deg[u] for u in active)
            candidates = [u for u in active if deg[u] == min_deg]
            u = candidates[rng.integers(len(candidates))]
            
            result_indices.append(u)
            
            # Remove u and its neighbors from active
            to_remove = set(neighbors[u])
            to_remove.add(u)
            
            new_active = []
            for v in active:
                if v not in to_remove:
                    # Update degree of v
                    # v loses connection to u
                    deg[v] -= 1
                    new_active.append(v)
            
            # Also remove to_remove from active_set for future checks? 
            # We are rebuilding active list, so it's fine.
            active = new_active
            
        # Construct final boolean array
        x_final = np.zeros(N4, dtype=bool)
        for i in result_indices:
            x_final[i] = True
            
        return x_final

    best_size = 0
    best_x = np.zeros(N4, dtype=bool)

    n_restarts = 0
    max_restarts = 1000 # Increased, but limited by time
    
    # Adaptive time per restart
    base_sa_time = 0.5
    
    while time.time() < deadline - 2 and n_restarts < max_restarts:
        n_restarts += 1
        
        # Initialize
        x_init = greedy_init()
        
        # Run SA
        # Reduce time as we go to allow more restarts?
        # Or keep constant.
        x_res = soft_sa(x_init, base_sa_time)
        
        # Repair to ensure independence
        x_final = repair(x_res)
        
        size_final = x_final.sum()
        if size_final > best_size:
            best_size = size_final
            best_x = x_final.copy()

    final_v4_indices = np.where(best_x)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])

    return final_vectors

# EVOLVE-BLOCK-END
