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

    def greedy_fill(S):
        """Try to add any valid node that doesn't conflict with S. Returns updated set and count added."""
        added = 0
        for v in valid_nodes:
            if v not in S and not conflict[v].intersection(S):
                S.add(v)
                added += 1
        return S, added

    def local_search_safe(initial_set):
        S = set(initial_set)
        current_size = len(S)
        
        while True:
            improved = False
            
            # 1. Greedy Add
            S, added = greedy_fill(S)
            if added > 0:
                current_size = len(S)
                improved = True
            
            if not improved:
                # 2. Try 1-for-k Swap (Remove one, Add many)
                swap_improved = False
                S_list = list(S)
                random.shuffle(S_list)
                for r in S_list:
                    if time.time() > deadline:
                        return S
                    
                    S.remove(r)
                    
                    # Simulate greedy add
                    S_temp = set(S)
                    _, to_add_count = greedy_fill(S_temp)
                    
                    new_size = len(S_temp)
                    if new_size > current_size:
                        # Apply change
                        S = S_temp
                        current_size = new_size
                        improved = True
                        swap_improved = True
                        break
                    else:
                        S.add(r)
                
                if not improved:
                    # 3. Try 2-for-k Swap (Remove two, Add many)
                    S_list = list(S)
                    random.shuffle(S_list)
                    for i in range(len(S_list)):
                        if time.time() > deadline:
                            return S
                        for j in range(i + 1, len(S_list)):
                            r1, r2 = S_list[i], S_list[j]
                            if r1 not in S or r2 not in S:
                                continue
                            
                            S.remove(r1)
                            S.remove(r2)
                            
                            S_temp = set(S)
                            greedy_fill(S_temp)
                            
                            new_size = len(S_temp)
                            if new_size > current_size:
                                S = S_temp
                                current_size = new_size
                                improved = True
                                break
                            else:
                                S.add(r1)
                                S.add(r2)
                        
                        if improved:
                            break
                
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
        
        # Random initialization with different strategies
        strategy = random.random()
        
        if strategy < 0.5:
            # Random order greedy
            random.shuffle(valid_nodes)
            init_set = set()
            for v in valid_nodes:
                if not conflict[v].intersection(init_set):
                    init_set.add(v)
        elif strategy < 0.8:
            # Minimum degree first greedy
            degrees = {v: len(conflict[v]) for v in valid_nodes}
            sorted_nodes = sorted(valid_nodes, key=lambda v: degrees[v])
            init_set = set()
            for v in sorted_nodes:
                if not conflict[v].intersection(init_set):
                    init_set.add(v)
        else:
            # Random subset then fill
            init_set = set(random.sample(valid_nodes, random.randint(5, 30)))
            # Remove internal conflicts
            to_remove = []
            for v in list(init_set):
                if any(u in init_set for u in conflict[v] if u != v):
                    to_remove.append(v)
            for v in to_remove:
                init_set.discard(v)
        
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
