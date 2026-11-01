"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import itertools
from collections import deque

D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

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
        
    # Compute conflict matrix: conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists as arrays
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)
    
    def greedy_min_degree():
        """Greedy independent set with min-degree heuristic."""
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set
    
    def spectral_init():
        """Spectral initialization using principal eigenvector of conflict graph."""
        # Build sparse adjacency matrix
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh
        
        # Create edge list
        rows = []
        cols = []
        for i in range(N4):
            nbrs = neighbors[i]
            if len(nbrs) > 0:
                rows.extend([i] * len(nbrs))
                cols.extend(nbrs.tolist())
        
        if len(rows) == 0:
            return np.ones(N4, dtype=bool)
        
        adj = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N4, N4))
        
        # Get principal eigenvector
        try:
            # Use k=1 to get the largest eigenvalue/eigenvector
            # sigma=0.5 for shift-invert to speed up
            eigvals, eigvecs = eigsh(adj, k=1, sigma=0.5, which='LM', v0=np.ones(N4))
            principal_evec = np.abs(eigvecs[:, 0])
        except:
            # Fallback to greedy if spectral fails
            return greedy_min_degree()
        
        # Select vertices in order of decreasing principal eigenvector component
        order = np.argsort(-principal_evec)
        
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        
        for v in order:
            if alive[v]:
                in_set[v] = True
                to_remove = np.unique(np.concatenate(([v], neighbors[v])))
                alive[to_remove] = False
                if time.time() > deadline - 5:
                    break
        
        return in_set
    
    def sa_local_search(in_set_init, max_time=10.0):
        """Vectorized min-conflict simulated annealing with 2-for-1 swaps."""
        in_set = in_set_init.copy()
        current_size = int(in_set.sum())
        
        T = 3.0
        cooling = 0.9995
        min_T = 0.05
        
        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        sel_idx = np.where(in_set)[0]
        for i in sel_idx:
            count_conflicts[neighbors[i]] += 1
        
        # Precompute conflict rows as boolean arrays for fast lookup
        # conflicts is already N4xN4 boolean matrix
        
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        
        # Track selected and unselected indices for fast access
        selected = sel_idx.copy()
        
        while time.time() < sa_deadline:
            # Try single vertex addition
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break
            
            counts = count_conflicts[not_in_set]
            min_count = int(np.min(counts))
            
            min_cands = not_in_set[counts == min_count]
            v = int(min_cands[rng.integers(len(min_cands))])
            
            k = min_count
            delta = 1 - k
            
            if delta > 0:
                accept = True
            elif delta == 0:
                accept = rng.random() < 0.5
            else:
                accept = rng.random() < np.exp(delta / T)
                
            if accept:
                # Find conflicts with current set
                to_remove = np.where(conflicts[v] & in_set)[0]
                nr = len(to_remove)
                
                if nr > 0:
                    in_set[to_remove] = False
                    for u in to_remove:
                        count_conflicts[neighbors[u]] -= 1
                    # Update selected list
                    selected = np.setdiff1d(selected, to_remove)
                
                in_set[v] = True
                selected = np.concatenate([selected, [v]])
                current_size += delta
                count_conflicts[neighbors[v]] += 1
                
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
            else:
                # Try 2-for-1 swap
                if rng.random() < 0.3 and len(selected) > 0:
                    remove_v = int(selected[rng.integers(len(selected))])
                    
                    # Temporarily remove
                    in_set[remove_v] = False
                    count_conflicts[neighbors[remove_v]] -= 1
                    current_size -= 1
                    selected = np.setdiff1d(selected, [remove_v])
                    
                    # Try to add two non-selected vertices with zero conflicts
                    added = []
                    for _ in range(2):
                        not_in_set_now = np.where(~in_set)[0]
                        if len(not_in_set_now) == 0:
                            break
                        counts_now = count_conflicts[not_in_set_now]
                        zero_cands = not_in_set_now[counts_now == 0]
                        if len(zero_cands) > 0:
                            add_v = int(zero_cands[rng.integers(len(zero_cands))])
                            in_set[add_v] = True
                            count_conflicts[neighbors[add_v]] += 1
                            current_size += 1
                            selected = np.concatenate([selected, [add_v]])
                            added.append(add_v)
                    
                    # If we didn't add enough, restore
                    if len(added) < 2:
                        in_set[remove_v] = True
                        count_conflicts[neighbors[remove_v]] += 1
                        current_size += 1
                        selected = np.concatenate([selected, [remove_v]])
                        for u in added:
                            in_set[u] = False
                            count_conflicts[neighbors[u]] -= 1
                            current_size -= 1
                            selected = np.setdiff1d(selected, [u])
                    else:
                        if current_size > local_best_size:
                            local_best_size = current_size
                            local_best_in_set = in_set.copy()
                            stuck_counter = 0
                        else:
                            stuck_counter += 1
                else:
                    stuck_counter += 1
            
            if stuck_counter > 1500:
                T = max(T, 2.0)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size
    
    # Multiple restarts
    n_restarts = 0
    max_restarts = 80
    
    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1
        
        # Alternate between greedy and spectral initialization
        if n_restarts % 2 == 0:
            in_set = greedy_min_degree()
        else:
            in_set = spectral_init()
        
        current_size = int(in_set.sum())
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # SA Local Search
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break
        
        sa_time = min(remaining_time * 0.6, 12.0)
        if sa_time < 2:
            sa_time = 2.0
            
        sa_result, sa_size = sa_local_search(in_set, max_time=sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()
        
        # Second SA pass on best found so far (polish)
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 5 and sa_size > current_size:
            polish_time = min(remaining_time * 0.5, 8.0)
            if polish_time > 3:
                polish_result, polish_size = sa_local_search(sa_result, max_time=polish_time)
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result.copy()
    
    # Final greedy polish: try to add any remaining W4 vectors that have no conflicts
    final_v4_indices = np.where(best_in_set)[0]
    
    # Try to greedily add more W4 vectors
    for i in range(N4):
        if best_in_set[i]:
            continue
        if time.time() > deadline - 2:
            break
        # Check if i has any conflict with current set
        if not np.any(conflicts[i] & best_in_set):
            best_in_set[i] = True
            final_v4_indices = np.where(best_in_set)[0]
    
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
