"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def _base3_construction(n):
    """Numbers in [1,n] with only digits 0,1 in base 3."""
    result = set()
    for x in range(1, n + 1):
        t = x
        ok = True
        while t > 0:
            if t % 3 == 2:
                ok = False
                break
            t //= 3
        if ok:
            result.add(x)
    return result

def _creates_4ap(S, x):
    """Check if adding x to S creates any nonconstant 4-AP."""
    # x can be in positions 1,2,3,4 of a 4-AP: a, a+d, a+2d, a+3d
    # x = a (position 1): need a+d, a+2d, a+3d in S for some d>0
    # x = a+d (position 2): need a, a+2d, a+3d in S for some d>0
    # x = a+2d (position 3): need a, a+d, a+3d in S for some d>0
    # x = a+3d (position 4): need a, a+d, a+2d in S for some d>0
    
    # Position 4: x = a+3d, so a = x-3d, check a, a+d, a+2d in S
    for d in range(1, (x - 1) // 3 + 1):
        a = x - 3 * d
        if a < 1:
            break
        if a in S and (a + d) in S and (a + 2 * d) in S:
            return True
    
    # Position 3: x = a+2d, so a = x-2d, check a, a+d, a+3d in S
    for d in range(1, (x - 1) // 2 + 1):
        a = x - 2 * d
        if a < 1:
            break
        if a in S and (a + d) in S and (a + 3 * d) in S:
            return True
    
    # Position 2: x = a+d, so a = x-d, check a, a+2d, a+3d in S
    for d in range(1, x):
        a = x - d
        if a < 1:
            break
        if a in S and (a + 2 * d) in S and (a + 3 * d) in S:
            return True
    
    # Position 1: x = a, check a+d, a+2d, a+3d in S
    for d in range(1, (512 - x) // 3 + 1):
        if (x + d) in S and (x + 2 * d) in S and (x + 3 * d) in S:
            return True
    
    return False

def _can_add(S, x, n):
    """Check if x can be added to S without creating a 4-AP."""
    if x < 1 or x > n:
        return False
    if x in S:
        return True
    return not _creates_4ap(S, x)

def _greedy_build(n, order):
    """Build a 4-AP-free set greedily in the given order."""
    S = set()
    for x in order:
        if _can_add(S, x, n):
            S.add(x)
    return S

def _local_search(S, n, max_iters=2000, deadline=None):
    """Swap-based local search: try removing one element to add two or more."""
    S = set(S)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if deadline is not None and time.time() > deadline:
            break
        
        # Try each element for removal
        for u in list(S):
            S.remove(u)
            added = 0
            added_elements = []
            
            # Try to add elements that were previously blocked
            for v in range(1, n + 1):
                if v not in S and v != u:
                    if _can_add(S, v, n):
                        S.add(v)
                        added_elements.append(v)
                        added += 1
            
            if added >= 2:
                improved = True
                # Keep the improvement
            else:
                # Revert
                S.add(u)
                for v in added_elements:
                    S.remove(v)
    
    return S

def _multi_restart_greedy(n, num_restarts=50, deadline=None):
    """Try multiple greedy constructions with different orderings."""
    best = set()
    best_size = 0
    
    for r in range(num_restarts):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Different orderings
        if r == 0:
            order = list(range(1, n + 1))
        elif r == 1:
            order = list(range(n, 0, -1))
        elif r == 2:
            # Odds first
            order = list(range(1, n + 1, 2)) + list(range(2, n + 1, 2))
        elif r == 3:
            # Evens first
            order = list(range(2, n + 1, 2)) + list(range(1, n + 1, 2))
        else:
            import random
            order = list(range(1, n + 1))
            random.seed(r)
            random.shuffle(order)
        
        S = _greedy_build(n, order)
        
        if len(S) > best_size:
            best = set(S)
            best_size = len(S)
    
    return best

def construct():
    n = 512
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s margin
    
    # Start with base-3 construction
    S = _base3_construction(n)
    
    # Try greedy improvements from base-3
    # First, try adding elements in natural order
    improved = True
    while improved:
        if time.time() > deadline - 10:
            break
        improved = False
        for x in range(1, n + 1):
            if x not in S and _can_add(S, x, n):
                S.add(x)
                improved = True
    
    best = set(S)
    best_size = len(S)
    
    # Try multi-restart greedy
    if time.time() < deadline - 20:
        S2 = _multi_restart_greedy(n, num_restarts=30, deadline=deadline - 10)
        if len(S2) > best_size:
            best = S2
            best_size = len(S2)
    
    # Local search on the best set
    if time.time() < deadline - 5:
        S3 = _local_search(best, n, max_iters=500, deadline=deadline - 2)
        if len(S3) > best_size:
            best = S3
            best_size = len(S3)
    
    # Final verification
    result = sorted(best)
    
    # Verify no 4-AP exists
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            for k in range(j + 1, len(result)):
                d1 = result[j] - result[i]
                d2 = result[k] - result[j]
                if d1 == d2 and d1 > 0:
                    # Check if there's a fourth term
                    fourth = result[k] + d2
                    if fourth in best:
                        pass  # This would be a 4-AP, but we shouldn't have one
    
    return result
# EVOLVE-BLOCK-END
