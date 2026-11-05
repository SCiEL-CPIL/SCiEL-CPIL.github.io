"""4-term-progression-free subset of [1,512] via greedy + local search."""
import time
import random

DEADLINE = time.time() + 50  # safety margin before 60s kill

def has_4ap_with(s, x):
    """Check if adding x to set s creates a 4-term AP."""
    # Max d such that terms are in [1, 512]
    # d < 512/3 = 170.6
    for d in range(1, 171):
        # Position 0: x, x+d, x+2d, x+3d
        if x + 3*d <= 512:
            if (x + d) in s and (x + 2*d) in s and (x + 3*d) in s:
                return True
        # Position 1: x-d, x, x+d, x+2d
        if x - d >= 1 and x + 2*d <= 512:
            if (x - d) in s and (x + d) in s and (x + 2*d) in s:
                return True
        # Position 2: x-2d, x-d, x, x+d
        if x - 2*d >= 1 and x + d <= 512:
            if (x - 2*d) in s and (x - d) in s and (x + d) in s:
                return True
        # Position 3: x-3d, x-2d, x-d, x
        if x - 3*d >= 1:
            if (x - 3*d) in s and (x - 2*d) in s and (x - d) in s:
                return True
    return False

def has_4ap_set(s):
    """Check if set s contains any nonconstant 4-term AP."""
    lst = sorted(s)
    n = len(lst)
    for i in range(n):
        a = lst[i]
        for j in range(i+1, n):
            b = lst[j]
            d = b - a
            if d <= 0:
                continue
            if (a + 2*d) in s and (a + 3*d) in s:
                return True
    return False

def base3_01_set():
    """Generate set of numbers with only 0/1 digits in base 3, up to 512."""
    s = set()
    for mask in range(1, 1 << 6):  # 6 base-3 digits: positions 0-5
        x = 0
        for i in range(6):
            if mask & (1 << i):
                x += 3 ** i
        if 1 <= x <= 512:
            s.add(x)
    return s

def greedy_forward(s):
    """Try to add elements from 1 to 512 in order."""
    s = set(s)
    for x in range(1, 513):
        if x in s:
            continue
        if not has_4ap_with(s, x):
            s.add(x)
    return s

def local_search(s, max_iters=100):
    """Local search: try to add elements, or swap to improve. Optimized for speed."""
    s = set(s)
    for _ in range(max_iters):
        if time.time() > DEADLINE:
            break
        improved = False
        
        # Try to add a random element (limit candidates to keep it fast)
        candidates = [x for x in range(1, 513) if x not in s]
        if candidates:
            random.shuffle(candidates)
            for x in candidates[:10]:
                if not has_4ap_with(s, x):
                    s.add(x)
                    improved = True
                    break
        
        if not improved and len(s) >= 2:
            # Try single swap: remove one, add one different
            lst = list(s)
            random.shuffle(lst)
            for rem in lst[:5]:
                s2 = s - {rem}
                candidates = [x for x in range(1, 513) if x not in s2 and x != rem]
                if candidates:
                    random.shuffle(candidates)
                    for x in candidates[:10]:
                        if not has_4ap_with(s2, x):
                            s = s2 | {x}
                            improved = True
                            break
                if improved:
                    break
                    
    return s

def construct():
    random.seed(42)
    
    # Start with a simple greedy construction which is fast
    s = set()
    s = greedy_forward(s)
    best = s
    best_size = len(s)
    
    # Local search with multiple restarts
    for restart in range(50):
        if time.time() > DEADLINE:
            break
        s_copy = set(best)
        # Perturb: remove a few random elements
        if len(s_copy) > 10:
            lst = list(s_copy)
            random.shuffle(lst)
            for _ in range(min(3, len(lst))):
                s_copy.discard(lst.pop())
        
        s_result = local_search(s_copy, max_iters=20)
        if len(s_result) > best_size:
            best = s_result
            best_size = len(s_result)
            
    return sorted(best)
