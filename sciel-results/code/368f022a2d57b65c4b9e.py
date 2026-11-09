import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    n = 128
    # Precompute GCD table
    gcd_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    def is_bad_triple(a, b, c):
        """Check if three distinct elements form a bad triple."""
        return gcd_table[a][b] == gcd_table[b][c] == gcd_table[a][c]
    
    def can_add(x, selected_set):
        """Check if x can be added to selected_set without creating a bad triple."""
        for a in selected_set:
            ga = gcd_table[x][a]
            for b in selected_set:
                if b == a:
                    continue
                if ga == gcd_table[x][b] == gcd_table[a][b]:
                    return False
        return True
    
    def is_valid(selected_set):
        """Check if the set has no bad triples."""
        lst = list(selected_set)
        for i in range(len(lst)):
            a = lst[i]
            for j in range(i + 1, len(lst)):
                b = lst[j]
                ga = gcd_table[a][b]
                for k in range(j + 1, len(lst)):
                    c = lst[k]
                    if ga == gcd_table[a][c] == gcd_table[b][c]:
                        return False
        return True
    
    # Phase 1: Greedy construction with multiple orderings
    import random
    rng = random.Random(42)
    
    best_set = set()
    best_size = 0
    
    max_restarts = 100
    for r in range(max_restarts):
        if time.time() > deadline - 30:
            break
        
        ordering = list(range(1, 129))
        rng.shuffle(ordering)
        
        selected = set()
        for x in ordering:
            if can_add(x, selected):
                selected.add(x)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = set(selected)
    
    # Phase 2: Local search - try to grow by adding elements
    if time.time() < deadline - 10:
        current = set(best_set)
        current_size = len(current)
        
        # Try adding in various orders
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            remaining = [x for x in range(1, 129) if x not in current]
            rng.shuffle(remaining)
            improved = False
            for x in remaining:
                if can_add(x, current):
                    current.add(x)
                    improved = True
            if not improved:
                break
        if len(current) > best_size:
            best_size = len(current)
            best_set = set(current)
    
    # Phase 3: Local search with 1-for-2 swaps (remove 1, add 2)
    if time.time() < deadline - 5:
        current = set(best_set)
        current_size = len(current)
        
        max_swap_iters = 200
        for it in range(max_swap_iters):
            if time.time() > deadline - 5:
                break
            
            found_swap = False
            for remove in list(current):
                if time.time() > deadline - 5:
                    break
                
                temp = set(current)
                temp.discard(remove)
                
                # Try to add two elements
                added = []
                remaining = [x for x in range(1, 129) if x not in temp and x != remove]
                rng.shuffle(remaining)
                
                for x in remaining:
                    if len(added) == 2:
                        break
                    if can_add(x, temp | set(added)):
                        added.append(x)
                        temp.add(x)
                
                if len(added) == 2:
                    current = temp
                    found_swap = True
                    if len(current) > best_size:
                        best_size = len(current)
                        best_set = set(current)
                    break
            
            if not found_swap:
                break
    
    # Phase 4: Final polish - try adding any remaining elements
    if time.time() < deadline:
        current = set(best_set)
        remaining = [x for x in range(1, 129) if x not in current]
        for x in remaining:
            if time.time() > deadline:
                break
            if can_add(x, current):
                current.add(x)
        if len(current) > best_size:
            best_size = len(current)
            best_set = set(current)
    
    return sorted(best_set)
