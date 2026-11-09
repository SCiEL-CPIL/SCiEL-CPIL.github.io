import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    # Forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Conflict matrix: conflict[i][j] is True if i and j cannot be in the set together
    # i.e., (i-j) % m in forbidden or (j-i) % m in forbidden
    # Since forbidden is symmetric (if d in F, -d in F), we just check (i-j) % m in forbidden
    conflict = np.zeros((m, m), dtype=bool)
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            if (i - j) % m in forbidden:
                conflict[i][j] = True
                conflict[j][i] = True

    # Helper to check if a candidate can be added to a set represented by a boolean array
    # S_bool: array of length m, True if element is in S
    def can_add(candidate, S_bool):
        # Check if candidate conflicts with any element in S
        # S_bool[candidate] must be False (implied by candidate not in S)
        # Check conflicts
        # conflict[candidate] is a boolean array of length m
        # We need to check if any True in S_bool is also True in conflict[candidate]
        # i.e., np.any(S_bool & conflict[candidate])
        if np.any(S_bool & conflict[candidate]):
            return False
        return True

    def greedy_construct(order):
        S = set()
        S_bool = np.zeros(m, dtype=bool)
        for i in order:
            if not S_bool[i]:
                if can_add(i, S_bool):
                    S.add(i)
                    S_bool[i] = True
        return S, S_bool

    def local_search(S, S_bool):
        # Try to improve S
        # Moves:
        # 1. Add 1 element
        # 2. Remove 1, Add 2 elements
        # 3. Remove 2, Add 3 elements (maybe too slow/complex for first pass, stick to 1 and 2)
        
        improved = True
        while improved:
            improved = False
            
            # Try Add 1
            added = False
            # Iterate candidates not in S
            # To make it deterministic but varied, we can shuffle or just iterate 0..m-1
            # For speed, iterate 0..m-1
            for v in range(m):
                if not S_bool[v]:
                    if can_add(v, S_bool):
                        S.add(v)
                        S_bool[v] = True
                        improved = True
                        added = True
                        break # Restart search after change
            
            if improved:
                continue # Restart loop to check for more adds or swaps

            # Try Remove 1, Add 2
            # Iterate over u in S to remove
            # For each u, try to find two v, z not in S such that they are compatible with S\{u} and each other
            # This is O(|S| * m^2) which is ~ 12 * 205^2 ~ 500k ops per check.
            # If we do this in a while loop, it might be slow if many iterations.
            # But we only do it if no Add 1 is possible.
            
            found_swap = False
            S_list = list(S)
            for u in S_list:
                # Tentatively remove u
                S_bool[u] = False
                S.remove(u)
                
                # Now find two elements v, z to add
                # Candidates are all v not in S
                # We need v and z to be compatible with S (which is now S_orig \ {u})
                # and with each other.
                
                # Optimization: Precompute valid candidates for current S?
                # valid_candidates = [v for v in range(m) if not S_bool[v] and can_add(v, S_bool)]
                # Then find pair v, z in valid_candidates such that not conflict[v][z]
                
                valid_candidates = []
                for v in range(m):
                    if not S_bool[v]:
                        if can_add(v, S_bool):
                            valid_candidates.append(v)
                
                # Find pair v, z in valid_candidates with no conflict
                # O(k^2) where k is len(valid_candidates)
                pair_found = False
                k = len(valid_candidates)
                for i_idx in range(k):
                    v = valid_candidates[i_idx]
                    for z_idx in range(i_idx + 1, k):
                        z = valid_candidates[z_idx]
                        if not conflict[v][z]:
                            # Add v and z
                            S.add(v)
                            S_bool[v] = True
                            S.add(z)
                            S_bool[z] = True
                            found_swap = True
                            pair_found = True
                            break
                    if pair_found:
                        break
                
                if found_swap:
                    improved = True
                    break # Restart local search with new set
                
                # If no swap found with this u, restore u
                S.add(u)
                S_bool[u] = True
            
            # If we tried all u and found no swap, improved is still False (unless we found one)
            # If improved is False, we exit the while loop
            
        return S, S_bool

    best_size = 0
    best_set = set()
    best_bool = np.zeros(m, dtype=bool)
    
    start_time = time.time()
    deadline = start_time + 500  # 500 seconds safety margin
    
    # Run multiple randomized greedy + local search cycles
    trial = 0
    while time.time() < deadline:
        # Generate random order
        order = list(range(m))
        random.shuffle(order)
        
        S, S_bool = greedy_construct(order)
        S, S_bool = local_search(S, S_bool)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            best_bool = S_bool.copy()
            
        trial += 1
        # If we found a very good solution, we might want to perturb it instead of full restart?
        # For now, full restarts with different shuffles are robust.
        # Also, we can try to perturb the best solution found so far in some iterations.
        if trial % 10 == 0 and best_size > 0:
             # Perturb best solution: remove 2 random elements, run local search
             S_pert = best_set.copy()
             S_pert_bool = best_bool.copy()
             if len(S_pert) >= 2:
                 # Remove 2 random elements
                 to_remove = random.sample(list(S_pert), 2)
                 for u in to_remove:
                     S_pert.remove(u)
                     S_pert_bool[u] = False
                 S_pert, S_pert_bool = local_search(S_pert, S_pert_bool)
                 if len(S_pert) > best_size:
                     best_size = len(S_pert)
                     best_set = S_pert.copy()
                     best_bool = S_pert_bool.copy()

    return np.array(list(best_set), dtype=np.int64)
