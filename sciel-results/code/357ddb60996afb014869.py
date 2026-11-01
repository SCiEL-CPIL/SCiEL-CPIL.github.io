import numpy as np
import itertools
import time
import random

D = 11

def construct():
    t0 = time.time()
    deadline = t0 + 115  # Safety margin
    
    # 1. Generate weight-4 vectors
    supports = list(itertools.combinations(range(D), 4))
    n_supports = len(supports) # 330
    
    # Generate all vectors
    all_vectors = []
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(s, signs):
                v[pos] = sign
            all_vectors.append(v)
            
    all_vectors_np = np.array(all_vectors, dtype=np.int8)
    N = len(all_vectors_np) # 5280
    
    # Compute dots
    dots = all_vectors_np @ all_vectors_np.T
    conflicts = (dots > 2).astype(np.int8)
    np.fill_diagonal(conflicts, 0)
    
    # Convert to adjacency lists for faster traversal in greedy
    adj = [[] for _ in range(N)]
    for i in range(N):
        # Find neighbors
        neighbors = np.where(conflicts[i] == 1)[0]
        adj[i] = neighbors.tolist()
        
    rng = np.random.default_rng(42)
    
    best_set = []
    
    # Greedy MIS with sparse adj
    def greedy_mis_sparse(adj, deadline, rng):
        N = len(adj)
        alive = np.ones(N, dtype=bool)
        deg = np.array([len(adj[i]) for i in range(N)], dtype=np.int32)
        
        selected = []
        
        while np.any(alive):
            if time.time() > deadline:
                break
            
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            # Find min degree
            min_deg = np.min(deg[alive_indices])
            candidates = alive_indices[deg[alive_indices] == min_deg]
            
            chosen = candidates[rng.integers(len(candidates))]
            selected.append(chosen)
            
            # Remove chosen and neighbors
            to_remove = np.unique(np.concatenate(([chosen], adj[chosen])))
            alive[to_remove] = False
            
            # Update degrees
            remaining_alive = np.where(alive)[0]
            if len(remaining_alive) > 0:
                for r in to_remove:
                    for nb in adj[r]:
                        if alive[nb]:
                            deg[nb] -= 1
                            
        return selected

    # Local Search (Hill Climbing)
    def local_search(conflicts, adj, initial_set, deadline, rng):
        N = len(adj)
        S_mask = np.zeros(N, dtype=bool)
        for i in initial_set:
            S_mask[i] = True
            
        current_size = np.sum(S_mask)
        
        while time.time() < deadline:
            # 1. Expansion
            # blocked[i] = number of neighbors in S
            blocked = conflicts.dot(S_mask)
            
            # Candidates: not in S, and blocked == 0
            candidates = np.where((~S_mask) & (blocked == 0))[0]
            
            if len(candidates) > 0:
                S_mask[candidates] = True
                current_size += len(candidates)
                continue
            
            # 2. No expansion possible. Try Swap.
            # Pick u in S
            u_indices = np.where(S_mask)[0]
            if len(u_indices) == 0:
                break
            u = u_indices[rng.integers(len(u_indices))]
            
            # Tentatively remove u
            S_mask[u] = False
            
            # Check if any node v not in S can be added
            # v must have blocked[v] == 1 and conflicts[v, u] == 1
            # (Because removing u decreases blocked[v] by 1. If it becomes 0, we can add v)
            
            potential_adds = np.where((~S_mask) & (conflicts[:, u] == 1) & (blocked == 1))[0]
            
            if len(potential_adds) > 0:
                # Select a maximal independent subset of potential_adds
                # to add.
                # Greedy selection:
                new_adds = []
                for v in potential_adds:
                    conflict_found = False
                    for w in new_adds:
                        if conflicts[v, w]:
                            conflict_found = True
                            break
                    if not conflict_found:
                        new_adds.append(v)
                        
                if len(new_adds) > 0:
                    S_mask[new_adds] = True
                    current_size += len(new_adds)
                    # Accept move
                    continue
                else:
                    # Revert u
                    S_mask[u] = True
                    continue
            else:
                # No benefit. Revert u
                S_mask[u] = True
                continue
                
        return np.where(S_mask)[0]

    # Run multiple restarts
    mis_deadline = t0 + 30
    ls_deadline = t0 + 110
    
    # 1. Get a good initial set via Greedy
    initial_set = greedy_mis_sparse(adj, mis_deadline, rng)
    best_set = initial_set
    
    # 2. Improve with Local Search
    if time.time() < ls_deadline:
        improved_set = local_search(conflicts, adj, best_set, ls_deadline, rng)
        if len(improved_set) > len(best_set):
            best_set = improved_set
            
    # Add weight-1 vectors
    weight1_vectors = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            weight1_vectors.append(v)
            
    weight1_np = np.array(weight1_vectors, dtype=np.int64)
    
    result_vectors_w4 = all_vectors_np[best_set].astype(np.int64)
    
    final_vectors = np.vstack([result_vectors_w4, weight1_np])
    
    return final_vectors
