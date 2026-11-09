import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 256
    
    # 1. Precompute squares
    squares = set()
    i = 2
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # 2. Filter self-conflicts
    # If 2*v is a square, v cannot be in the set
    valid_nodes = [v for v in range(1, N + 1) if 2 * v not in squares]
    valid_set = set(valid_nodes)
    
    # 3. Build conflict structure
    # conflict[v] is a set of nodes u (u != v) such that u+v is a square
    # We only care about conflicts among valid_nodes
    conflict = {v: set() for v in valid_nodes}
    
    # To speed up, we can iterate pairs
    for i, v in enumerate(valid_nodes):
        for u in valid_nodes[i+1:]:
            if v + u in squares:
                conflict[v].add(u)
                conflict[u].add(v)
    
    def is_valid_set(S):
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            # Check diagonal
            if 2 * a in squares:
                return False
            # Check off-diagonal
            for j in range(i + 1, n):
                b = S_list[j]
                if a + b in squares:
                    return False
        return True

    def local_search_safe(initial_set):
        S = set(initial_set)
        current_size = len(S)
        
        while True:
            improved = False
            
            # 1. Greedy Add
            for v in valid_nodes:
                if v in S:
                    continue
                if not conflict[v].intersection(S):
                    S.add(v)
                    current_size += 1
                    improved = True
            
            if not improved:
                # 2. Try Swap (Remove one, Add many)
                swap_improved = False
                for r in list(S):
                    # Temporarily remove r
                    S.remove(r)
                    size_after_remove = current_size - 1
                    
                    # Simulate greedy add using a temporary set to avoid internal conflicts in to_add
                    S_temp = set(S)
                    to_add = []
                    for v in valid_nodes:
                        if v == r:
                            continue
                        if v in S_temp:
                            continue
                        if not conflict[v].intersection(S_temp):
                            S_temp.add(v)
                            to_add.append(v)
                    
                    new_size = len(S_temp)
                    if new_size > current_size:
                        # Apply change
                        S = S_temp
                        current_size = new_size
                        improved = True
                        swap_improved = True
                        break # Restart the while loop with the new set
                    else:
                        # Revert (r is already removed, nothing added)
                        S.add(r)
                
                if not improved:
                    break
        
        return S

    best_size = 0
    best_set = set()
    
    # Generate multiple random initial sets
    num_restarts = 100000 # High number, but we check time
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization
        # Pick a random subset of valid nodes?
        # Or greedy with random order?
        random.shuffle(valid_nodes)
        init_set = set()
        for v in valid_nodes:
            if not conflict[v].intersection(init_set):
                init_set.add(v)
        
        # Run local search
        S = local_search_safe(init_set)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Final verification
    if best_set and not is_valid_set(best_set):
        # Fallback to a simple valid set if the optimized one failed verification
        # Greedy from scratch is safe
        fallback_set = set()
        for v in valid_nodes:
            if not conflict[v].intersection(fallback_set):
                fallback_set.add(v)
        best_set = fallback_set
    
    return list(best_set)
