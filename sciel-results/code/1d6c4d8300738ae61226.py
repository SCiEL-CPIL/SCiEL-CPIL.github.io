import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 256
    best_set = set()
    best_size = 0
    
    # Strategy 1: Greedy Removal from high-density starts
    # Start with sets of large numbers
    start_vals = [128, 129, 130, 120, 110, 100]
    
    for start_val in start_vals:
        if time.time() > deadline - 5:
            break
        
        S = set(range(start_val, n + 1))
        
        # Resolve conflicts by removing elements with highest conflict degree
        S = _resolve_conflicts(S, n, deadline - 5)
        
        # Try to add back any excluded elements
        if time.time() < deadline - 10:
            S = _try_add_back(S, n, deadline - 10)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)

    # Strategy 2: Original Greedy (Largest to Smallest)
    if time.time() < deadline - 5:
        S_greedy = set()
        for x in range(n, 0, -1):
            if time.time() > deadline:
                break
            if _can_add(x, S_greedy, n):
                S_greedy.add(x)
        
        if len(S_greedy) > best_size:
            best_size = len(S_greedy)
            best_set = set(S_greedy)

    if not best_set:
        return [n]
    
    return sorted(best_set)

def _can_add(x, S, n):
    """Check if x can be added to S without violating the property."""
    S_list = list(S)
    
    # Case 1: x is the smallest (a=x)
    for b in S_list:
        if b <= x:
            continue
        for c in S_list:
            if c < b:
                continue
            if (b + c) % x == 0:
                return False
    
    # Case 2: x is b or c
    for a in S_list:
        if a >= x:
            continue
        # x as b: need c in S with x <= c and a | (x+c)
        for c in S_list:
            if c < x:
                continue
            if (x + c) % a == 0:
                return False
        # x as c: need b in S with a < b <= x and a | (b+x)
        for b in S_list:
            if b <= a or b > x:
                continue
            if (b + x) % a == 0:
                return False
    
    return True

def _resolve_conflicts(S, n, deadline):
    """Remove elements from S to resolve all conflicts."""
    S = set(S)
    while True:
        if time.time() > deadline:
            break
        
        violations = []
        # Find all violations
        for a in S:
            for b in S:
                if b <= a:
                    continue
                for c in S:
                    if c < b:
                        continue
                    if (b + c) % a == 0:
                        violations.append((a, b, c))
        
        if not violations:
            break
        
        # Count degrees
        degrees = {}
        for a, b, c in violations:
            degrees[a] = degrees.get(a, 0) + 1
            degrees[b] = degrees.get(b, 0) + 1
            degrees[c] = degrees.get(c, 0) + 1
        
        # Find max degree
        max_deg = max(degrees.values())
        candidates = [x for x, d in degrees.items() if d == max_deg]
        
        # Heuristic: Remove the smallest element among the most conflicted
        # This preserves the larger numbers which are "safer"
        x_to_remove = min(candidates)
        S.remove(x_to_remove)
        
    return S

def _try_add_back(S, n, deadline):
    """Try to add any element from 1..n back into S if possible."""
    S = set(S)
    # Sort candidates by descending value? Or just iterate?
    # Larger values are easier to add.
    for x in range(n, 0, -1):
        if time.time() > deadline:
            break
        if x in S:
            continue
        if _can_add(x, S, n):
            S.add(x)
    return S
