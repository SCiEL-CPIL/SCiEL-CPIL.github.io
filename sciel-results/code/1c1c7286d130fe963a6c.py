"""Solve the packing problem: find 10 distinct positive integers <= 1000000,
each divisible by its corresponding divisor in [6,10,14,15,21,22,26,33,35,39],
minimizing max - min."""

import time
import math
from collections import deque

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
N = len(DIVISORS)
MAX_VAL = 1000000


def _bipartite_matching(divisors, L, R):
    """Check if there's a perfect matching between divisors and values in [L, R].
    
    Uses Hopcroft-Karp. Returns (True, assignment) or (False, None).
    assignment[i] = value assigned to divisor i, or -1 if not matched.
    """
    # Build adjacency: for each divisor, which values in [L, R] are multiples
    # We only need to consider values that are multiples of at least one divisor
    # Map values to indices for efficiency
    val_to_idx = {}
    values = []
    for v in range(L, R + 1):
        # Check if v is a multiple of any divisor
        for d in divisors:
            if v % d == 0:
                val_to_idx[v] = len(values)
                values.append(v)
                break
    
    if len(values) < N:
        return False, None
    
    # Build adjacency list: for each divisor i, list of value indices it can match
    adj = [[] for _ in range(N)]
    for i, d in enumerate(divisors):
        # Find all values in [L, R] that are multiples of d
        first = ((L + d - 1) // d) * d
        for v in range(first, R + 1, d):
            if v in val_to_idx:
                adj[i].append(val_to_idx[v])
    
    # Hopcroft-Karp
    INF = float('inf')
    pair_u = [-1] * N  # pair_u[i] = value index matched to divisor i
    pair_v = [-1] * len(values)  # pair_v[j] = divisor index matched to value j
    dist = [0] * N
    
    def bfs():
        queue = deque()
        for u in range(N):
            if pair_u[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = INF
        found = False
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if pair_v[v] == -1:
                    found = True
                elif dist[pair_v[v]] == INF:
                    dist[pair_v[v]] = dist[u] + 1
                    queue.append(pair_v[v])
        return found
    
    def dfs(u):
        for v in adj[u]:
            if pair_v[v] == -1 or (dist[pair_v[v]] == dist[u] + 1 and dfs(pair_v[v])):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = INF
        return False
    
    matching = 0
    while bfs():
        for u in range(N):
            if pair_u[u] == -1:
                if dfs(u):
                    matching += 1
    
    if matching < N:
        return False, None
    
    # Build assignment
    assignment = []
    for i in range(N):
        assignment.append(values[pair_u[i]])
    
    return True, assignment


def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit
    
    divisors = DIVISORS[:]
    
    # Strategy: binary search on window width W, and for each W, slide the window
    # to find a position where a perfect matching exists.
    
    # First, find a feasible solution with a reasonable window
    # Start with a moderate window and expand if needed
    
    best_result = None
    best_width = float('inf')
    
    # Try small windows first
    for W in range(1, 200):
        if time.time() > deadline:
            break
        
        # For window width W, try different starting positions L
        # L must be such that [L, L+W] contains multiples of all divisors
        # The LCM of all divisors determines the period of the pattern
        # LCM(6,10,14,15,21,22,26,33,35,39)
        # Let's compute it
        # 6=2*3, 10=2*5, 14=2*7, 15=3*5, 21=3*7, 22=2*11, 26=2*13, 33=3*11, 35=5*7, 39=3*13
        # LCM = 2 * 3 * 5 * 7 * 11 * 13 = 30030
        
        lcm = 1
        for d in divisors:
            lcm = lcm * d // math.gcd(lcm, d)
        
        # The pattern repeats every lcm values
        # We only need to check L in [1, lcm] (and wrap around if needed)
        # But values must be <= MAX_VAL, so L+W <= MAX_VAL
        
        max_L = MAX_VAL - W
        if max_L < 1:
            continue
        
        # Check all possible L in one period
        checked = set()
        found = False
        
        for L in range(1, min(lcm, max_L) + 1):
            if time.time() > deadline:
                break
            
            # Check if this L has been effectively checked (modulo lcm)
            key = L % lcm
            if key in checked:
                continue
            checked.add(key)
            
            R = L + W
            if R > MAX_VAL:
                continue
            
            ok, assignment = _bipartite_matching(divisors, L, R)
            if ok:
                width = max(assignment) - min(assignment)
                if width < best_width:
                    best_width = width
                    best_result = assignment[:]
                found = True
                # We found a solution with this W, but let's check if we can do better
                # Actually, we want the minimum W, so if we found one, we should try smaller
                # But we're iterating W from 1 upward, so the first W that works is optimal
                # However, the actual max-min might be less than W
                break
        
        if found:
            # We found the minimum W for which a solution exists
            # But we should also verify by checking if we can get an even smaller max-min
            # by looking more carefully. For now, this is our best.
            break
    
    # If we didn't find anything with W < 200, try larger windows
    if best_result is None:
        for W in range(200, 1000, 10):
            if time.time() > deadline:
                break
            
            lcm = 1
            for d in divisors:
                lcm = lcm * d // math.gcd(lcm, d)
            
            max_L = MAX_VAL - W
            if max_L < 1:
                continue
            
            checked = set()
            found = False
            
            for L in range(1, min(lcm, max_L) + 1):
                if time.time() > deadline:
                    break
                
                key = L % lcm
                if key in checked:
                    continue
                checked.add(key)
                
                R = L + W
                if R > MAX_VAL:
                    continue
                
                ok, assignment = _bipartite_matching(divisors, L, R)
                if ok:
                    width = max(assignment) - min(assignment)
                    if width < best_width:
                        best_width = width
                        best_result = assignment[:]
                    found = True
                    break
            
            if found:
                break
    
    # Fallback: if still no solution, use a greedy approach
    if best_result is None:
        # Just pick the smallest multiple of each divisor
        best_result = [d for d in divisors]
        best_width = max(best_result) - min(best_result)
    
    # Ensure types are correct
    result = [int(x) for x in best_result]
    return result


# Test
if __name__ == "__main__":
    result = construct()
    print(f"Result: {result}")
    print(f"Width: {max(result) - min(result)}")
    for i, (d, v) in enumerate(zip(DIVISORS, result)):
        print(f"  Divisor {d}: value {v}, divisible: {v % d == 0}")
    print(f"Distinct: {len(set(result)) == len(result)}")
    print(f"All <= 1000000: {all(v <= 1000000 for v in result)}")
    print(f"All positive: {all(v > 0 for v in result)}")
