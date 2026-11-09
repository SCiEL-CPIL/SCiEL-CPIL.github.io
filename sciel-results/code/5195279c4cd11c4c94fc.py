"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    """
    Construct a subset of [1,256] with no 4-term AP, maximizing 3-term APs.
    Uses greedy construction with lookahead, plus local search refinement.
    """
    deadline = time.time() + 80  # safety margin before 90s kill
    
    N = 256
    
    def has_4ap_with(set_s, x):
        """Check if adding x to set_s creates a 4-term AP."""
        s = set_s | {x}
        # Check all 4-term APs: a, a+d, a+2d, a+3d
        # x must be one of the four positions
        # Position 0: x = a -> need a+d, a+2d, a+3d in s
        for d in range(1, (N - x) // 3 + 1):
            if (x + d in s) and (x + 2*d in s) and (x + 3*d in s):
                return True
        # Position 1: x = a+d -> need a=x-d, a+2d=x+d, a+3d=x+2d in s
        for d in range(1, min(x-1, (N-1-x)//1) + 1):
            if d < 1:
                break
            a = x - d
            if a < 1:
                break
            if (a in s) and (x + d in s) and (x + 2*d in s):
                return True
        # Position 2: x = a+2d -> need a=x-2d, a+d=x-d, a+3d=x+d in s
        for d in range(1, min((x-1)//2, (N-1-x)//1) + 1):
            a = x - 2*d
            if a < 1:
                break
            if (a in s) and (x - d in s) and (x + d in s):
                return True
        # Position 3: x = a+3d -> need a=x-3d, a+d=x-2d, a+2d=x-d in s
        for d in range(1, (x-1)//3 + 1):
            a = x - 3*d
            if a < 1:
                break
            if (a in s) and (x - 2*d in s) and (x - d in s):
                return True
        return False
    
    def count_3aps(set_s):
        """Count distinct 3-term APs with positive difference in set_s."""
        count = 0
        s = set_s
        for a in s:
            for d in range(1, (N - a) // 2 + 1):
                if (a + d in s) and (a + 2*d in s):
                    count += 1
        return count
    
    def gain_of_adding(set_s, x):
        """Count how many new 3-APs are created by adding x to set_s."""
        s = set_s
        gain = 0
        # x as first term: x, x+d, x+2d
        for d in range(1, (N - x) // 2 + 1):
            if (x + d in s) and (x + 2*d in s):
                gain += 1
        # x as middle term: x-d, x, x+d
        for d in range(1, min(x-1, N-x) + 1):
            if (x - d in s) and (x + d in s):
                gain += 1
        # x as last term: x-2d, x-d, x
        for d in range(1, (x-1)//2 + 1):
            if (x - 2*d in s) and (x - d in s):
                gain += 1
        return gain
    
    # Greedy construction
    def greedy_construct(order_seed=0):
        rng = random.Random(order_seed)
        selected = set()
        remaining = list(range(1, N+1))
        
        # Try multiple orderings
        best_set = set()
        best_count = -1
        
        for attempt in range(20):
            if time.time() > deadline:
                break
            if attempt == 0:
                order = list(range(1, N+1))
            elif attempt == 1:
                order = list(range(N, 0, -1))
            else:
                order = list(range(1, N+1))
                rng.shuffle(order)
            
            sel = set()
            for x in order:
                if has_4ap_with(sel, x):
                    continue
                sel.add(x)
            
            cnt = count_3aps(sel)
            if cnt > best_count:
                best_count = cnt
                best_set = set(sel)
        
        return best_set, best_count
    
    # Local search: try swapping elements to improve
    def local_search(initial_set):
        if time.time() > deadline:
            return initial_set, count_3aps(initial_set)
        
        current = set(initial_set)
        current_count = count_3aps(current)
        improved = True
        max_iter = 5000
        
        while improved and max_iter > 0:
            improved = False
            max_iter -= 1
            if time.time() > deadline:
                break
            
            # Try adding an element not in current
            for x in range(1, N+1):
                if x in current:
                    continue
                if has_4ap_with(current, x):
                    continue
                new_set = current | {x}
                new_count = count_3aps(new_set)
                if new_count > current_count:
                    current = new_set
                    current_count = new_count
                    improved = True
                    break
            
            # Try removing an element and adding another
            if not improved and max_iter > 0:
                for x in list(current):
                    if max_iter <= 0:
                        break
                    new_set = current - {x}
                    for y in range(1, N+1):
                        if y in new_set:
                            continue
                        if has_4ap_with(new_set, y):
                            continue
                        candidate = new_set | {y}
                        cand_count = count_3aps(candidate)
                        if cand_count > current_count:
                            current = candidate
                            current_count = cand_count
                            improved = True
                            break
                    if improved:
                        break
        
        return current, current_count
    
    # Start with greedy
    best_set, best_count = greedy_construct(42)
    
    # Refine with local search
    best_set, best_count = local_search(best_set)
    
    # Try more greedy starts if time permits
    for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        if time.time() > deadline - 5:
            break
        s, c = greedy_construct(seed)
        if c > best_count:
            best_set, best_count = s, c
            best_set, best_count = local_search(best_set)
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
