"""Optimized 4-AP-free set construction for [1, 512]."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

N = 512


def _can_add(x, S, n=N):
    """Check if adding x to S creates a 4-term AP. S is a set."""
    # x as 4th term: a, a+d, a+2d, x => d=(x-a)//3
    for a in S:
        if x > a:
            diff = x - a
            if diff % 3 == 0:
                d = diff // 3
                if d > 0 and (a + d) in S and (a + 2 * d) in S:
                    return False
        # x as 3rd term: a, a+d, x, x+d => d=(x-a)//2
        if x > a:
            diff = x - a
            if diff % 2 == 0:
                d = diff // 2
                if d > 0 and (a + d) in S and (x + d) <= n and (x + d) in S:
                    return False
        # x as 2nd term: a, x, x+d, x+2d => d=x-a
        d = x - a
        if d > 0 and (x + d) <= n and (x + 2 * d) <= n:
            if (x + d) in S and (x + 2 * d) in S:
                return False
        # x as 1st term: x, x+d, x+2d, x+3d
        max_d = (n - x) // 3
        for d in range(1, max_d + 1):
            if (x + d) in S and (x + 2 * d) in S and (x + 3 * d) in S:
                return False
    return True


def _has_4ap(S, n=N):
    """Check if set S contains a nonconstant 4-term AP."""
    S_set = set(S)
    for a in S_set:
        for d in range(1, (n - a) // 3 + 1):
            if (a + d) in S_set and (a + 2 * d) in S_set and (a + 3 * d) in S_set:
                return True
    return False


def _ternary_01_set(n=N):
    """Numbers in [1,n] with base-3 digits only 0 or 1."""
    result = set()
    for mask in range(1, 1 << 6):
        val = 0
        for bit in range(6):
            if mask & (1 << bit):
                val += 3 ** bit
        if 1 <= val <= n:
            result.add(val)
    return result


def _greedy_construct(n=N):
    """Greedy: add x in order 1..n if it doesn't create a 4-AP."""
    selected = set()
    for x in range(1, n + 1):
        if _can_add(x, selected, n):
            selected.add(x)
    return selected


def _local_search(S, n=N, max_iters=2000, deadline=None):
    """Hill climbing: try to add elements, swap out bad elements."""
    S = set(S)
    current_size = len(S)
    
    for _ in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        
        # Try adding each missing element
        for x in range(1, n + 1):
            if x in S:
                continue
            if _can_add(x, S, n):
                S.add(x)
                current_size += 1
                improved = True
                break
        
        if improved:
            continue
        
        # Try swaps: remove one element, add another
        # Find elements that block the most additions
        blocked_by = {}
        for x in range(1, n + 1):
            if x in S:
                continue
            # Check why x can't be added
            for a in S:
                if x > a:
                    diff = x - a
                    if diff % 3 == 0:
                        d = diff // 3
                        if d > 0 and (a + d) in S and (a + 2 * d) in S:
                            blocked_by[a] = blocked_by.get(a, 0) + 1
                            blocked_by[a + d] = blocked_by.get(a + d, 0) + 1
                            blocked_by[a + 2 * d] = blocked_by.get(a + 2 * d, 0) + 1
                            break
                    if diff % 2 == 0:
                        d = diff // 2
                        if d > 0 and (a + d) in S and (x + d) <= n and (x + d) in S:
                            blocked_by[a] = blocked_by.get(a, 0) + 1
                            blocked_by[a + d] = blocked_by.get(a + d, 0) + 1
                            blocked_by[x + d] = blocked_by.get(x + d, 0) + 1
                            break
                d = x - a
                if d > 0 and (x + d) <= n and (x + 2 * d) <= n:
                    if (x + d) in S and (x + 2 * d) in S:
                        blocked_by[x + d] = blocked_by.get(x + d, 0) + 1
                        blocked_by[x + 2 * d] = blocked_by.get(x + 2 * d, 0) + 1
                        break
                max_d = (n - x) // 3
                for dd in range(1, max_d + 1):
                    if (x + dd) in S and (x + 2 * dd) in S and (x + 3 * dd) in S:
                        blocked_by[x + dd] = blocked_by.get(x + dd, 0) + 1
                        blocked_by[x + 2 * dd] = blocked_by.get(x + 2 * dd, 0) + 1
                        blocked_by[x + 3 * dd] = blocked_by.get(x + 3 * dd, 0) + 1
                        break
        
        if not blocked_by:
            break
        
        # Try removing the most-blocking element and see if we can add more
        for candidate in sorted(blocked_by.keys(), key=lambda k: -blocked_by[k])[:5]:
            S_test = set(S)
            S_test.discard(candidate)
            added = 0
            for x in range(1, n + 1):
                if x in S_test or x == candidate:
                    continue
                if _can_add(x, S_test, n):
                    S_test.add(x)
                    added += 1
            if added > 1:
                S = S_test
                current_size = len(S)
                improved = True
                break
        
        if not improved and deadline is not None and time.time() > deadline:
            break
    
    return S


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline to stay under 90s limit
    n = N
    
    best_set = _ternary_01_set(n)
    best_size = len(best_set)
    
    # Greedy from scratch
    greedy_set = _greedy_construct(n)
    if len(greedy_set) > best_size:
        best_set = greedy_set
        best_size = len(greedy_set)
    
    # Local search improvement
    improved = _local_search(best_set, n, max_iters=5000, deadline=deadline)
    if len(improved) > best_size:
        best_set = improved
        best_size = len(improved)
    
    # Randomized restarts
    random.seed(42)
    for attempt in range(30):
        if time.time() > deadline:
            break
        
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        if attempt % 3 == 0:
            current = set(_ternary_01_set(n))
        elif attempt % 3 == 1:
            current = set()
        else:
            # Start with a random subset of the best set
            current = set(random.sample(sorted(best_set), min(len(best_set) // 2, len(best_set))))
        
        for x in order:
            if x in current:
                continue
            if _can_add(x, current, n):
                current.add(x)
        
        # Local search on the result
        current = _local_search(current, n, max_iters=2000, deadline=deadline)
        
        if len(current) > best_size:
            best_set = current
            best_size = len(current)
    
    # Final verification
    if _has_4ap(best_set, n):
        best_set = _greedy_construct(n)
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
