import time
import random

def construct():
    start_time = time.time()
    time_limit = 80.0  # Safety margin before 90s kill
    
    n = 192
    
    def is_valid(S):
        """Check if no element in S divides the sum of any 3 distinct OTHER elements in S."""
        S_list = list(S)
        K = len(S_list)
        if K < 4:
            return True
        
        for a in S_list:
            others = [v for v in S_list if v != a]
            m = len(others)
            
            # Build pair-sum residue map: residue -> list of (i, j) index pairs
            pair_map = {}
            for i in range(m):
                xi = others[i]
                for j in range(i + 1, m):
                    xj = others[j]
                    s = (xi + xj) % a
                    if s not in pair_map:
                        pair_map[s] = []
                    pair_map[s].append((i, j))
            
            # For each z, check if there's a pair (x,y) with x+y+z ≡ 0 (mod a)
            # where x, y, z are all distinct
            for z_idx, z in enumerate(others):
                target = (-z) % a
                if target in pair_map:
                    for i, j in pair_map[target]:
                        if i != z_idx and j != z_idx:
                            return False
        
        return True
    
    # Phase 1: Greedy construction from largest to smallest
    selected = []
    for x in range(n, 0, -1):
        if time.time() - start_time > time_limit:
            break
        candidate = selected + [x]
        if is_valid(candidate):
            selected.append(x)
    
    S = set(selected)
    
    # Phase 2: Local search
    # Strategy: First try deterministic additions, then random swaps
    # Keep best solution found
    
    best_S = set(S)
    best_size = len(best_S)
    
    while time.time() - start_time < time_limit:
        # Try to add elements (deterministic scan first)
        added = False
        for x in range(n, 0, -1):
            if x in S:
                continue
            new_S = S | {x}
            if is_valid(new_S):
                S = new_S
                added = True
                if len(S) > best_size:
                    best_S = set(S)
                    best_size = len(best_S)
        
        if not added:
            # No more additions possible; try random swaps
            if len(S) == 0:
                break
            
            U_not_S = [x for x in range(1, n + 1) if x not in S]
            if not U_not_S:
                break
            
            # Try a few random swaps
            swap_found = False
            for _ in range(50):
                if time.time() - start_time > time_limit:
                    break
                remove_candidate = random.choice(list(S))
                add_candidate = random.choice(U_not_S)
                
                new_S = (S - {remove_candidate}) | {add_candidate}
                if is_valid(new_S):
                    S = new_S
                    # After swap, try to add more
                    for x in range(n, 0, -1):
                        if x in S:
                            continue
                        new_S2 = S | {x}
                        if is_valid(new_S2):
                            S = new_S2
                            if len(S) > best_size:
                                best_S = set(S)
                                best_size = len(best_S)
                    swap_found = True
                    break
            
            if not swap_found and time.time() - start_time > time_limit * 0.8:
                # Diminishing returns, stop
                break
    
    return list(best_S)
