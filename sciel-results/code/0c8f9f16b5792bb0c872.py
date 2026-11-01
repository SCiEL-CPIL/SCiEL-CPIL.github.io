# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import scipy.sparse

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    
    Uses norm^2=4 candidates (weight-4 in {-1,0,1}^11 and weight-1 with +/-2).
    Multi-Start Greedy with sparse conflict matrix + Kempe-chain local improvement.
    """
    D = 11
    deadline = time.time() + 110.0  # Safety margin
    
    # Generate weight-4 vectors in {-1, 0, 1}^11 (norm^2 = 4)
    v4_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            v4_list.append(v)
    
    # Generate weight-1 vectors with +/-2 (norm^2 = 4)
    v1_list = []
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int8)
            v[i] = 2 * s
            v1_list.append(v)
    
    candidates = np.vstack([np.array(v4_list), np.array(v1_list)])
    n = len(candidates)
    
    # For norm^2=4 vectors, conflict if dist^2 < 4
    # dist^2 = ||u||^2 + ||v||^2 - 2*u.v = 8 - 2*u.v
    # Conflict if 8 - 2*u.v < 4 => u.v > 2
    # So conflict iff dot product > 2
    
    dots = candidates.astype(np.int32) @ candidates.astype(np.int32).T
    conflict = (dots > 2)
    np.fill_diagonal(conflict, False)
    
    conflict_sparse = scipy.sparse.csr_matrix(conflict)
    
    best_size = 0
    best_selection = []
    
    rng = np.random.default_rng(42)
    
    def greedy_mis(conflict_sparse, rng, deadline):
        """Multi-start greedy independent set with degree-based ordering."""
        n = conflict_sparse.shape[0]
        alive = np.ones(n, dtype=bool)
        degrees = conflict_sparse.sum(axis=1).A1.astype(np.int32)
        selected = []
        
        while np.any(alive):
            if time.time() > deadline:
                break
                
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            # Find min degree among alive
            active_degrees = degrees[alive_indices]
            min_deg = np.min(active_degrees)
            min_mask = (active_degrees == min_deg)
            min_candidates = alive_indices[min_mask]
            
            # Random tie-breaking
            chosen = min_candidates[rng.integers(len(min_candidates))]
            selected.append(chosen)
            
            # Remove chosen and its neighbors
            neighbors = np.array(conflict_sparse[chosen].indices)
            to_remove = np.unique(np.concatenate(([chosen], neighbors)))
            
            alive[to_remove] = False
            
            # Update degrees for remaining alive nodes
            remaining = np.where(alive)[0]
            if len(remaining) > 0:
                # Count how many of to_remove are neighbors of each remaining node
                # Vectorized: for each remaining node, count conflicts with to_remove
                sub_conflicts = conflict_sparse[remaining][:, to_remove].toarray()
                degree_decrements = sub_conflicts.sum(axis=1)
                degrees[remaining] -= degree_decrements
        
        return selected
    
    def kempe_improve(conflict_sparse, initial_set, deadline, rng):
        """Kempe-chain local improvement: try to swap vertices to increase set size."""
        n = conflict_sparse.shape[0]
        current_set = set(initial_set)
        in_set = np.zeros(n, dtype=bool)
        for i in current_set:
            in_set[i] = True
        
        best_size = len(current_set)
        best_set = list(current_set)
        
        # Precompute conflict rows for fast access
        conflict_rows = [conflict_sparse[i] for i in range(n)]
        
        max_iters = 500
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try to add a random vertex not in set
            u = rng.integers(n)
            if not in_set[u]:
                # Check if u conflicts with any vertex in set
                neighbors_in_set = np.array(conflict_rows[u][in_set].indices)
                if len(neighbors_in_set) == 0:
                    in_set[u] = True
                    current_set.add(u)
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = list(current_set)
                    continue
            
            # Kempe-chain swap: pick u in set, try to replace with v not in set
            # such that v has fewer conflicts with (set - {u}) than u does
            if len(current_set) > 0:
                u = rng.choice(list(current_set))
                # Find vertices not in set that could replace u
                # v must not conflict with any vertex in (set - {u})
                # and v must not be u
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) == 0:
                    continue
                
                # For each candidate v in not_in_set, check if v conflicts with (set - {u})
                # This is expensive. Instead, try a few random candidates.
                for _ in range(10):
                    v = not_in_set[rng.integers(len(not_in_set))]
                    if v == u:
                        continue
                    # Check if v conflicts with any vertex in (set - {u})
                    set_minus_u = current_set - {u}
                    conflicts_with_set = False
                    for w in set_minus_u:
                        if conflict_sparse[v, w]:
                            conflicts_with_set = True
                            break
                    if not conflicts_with_set:
                        # Swap: remove u, add v
                        in_set[u] = False
                        in_set[v] = True
                        current_set.remove(u)
                        current_set.add(v)
                        break
        
        return best_set, best_size
    
    # Run multi-start greedy
    num_restarts = 200
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        selected = greedy_mis(conflict_sparse, rng, deadline)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selection = selected.copy()
    
    # Apply Kempe-chain improvement to the best set found
    if best_size > 0:
        improved_set, improved_size = kempe_improve(
            conflict_sparse, best_selection, deadline, rng
        )
        if improved_size > best_size:
            best_size = improved_size
            best_selection = improved_set
    
    result = candidates[best_selection]
    return result
# EVOLVE-BLOCK-END
