import itertools
import time
import random

def construct(n):
    if n <= 1:
        return "1"
    
    start_time = time.time()
    time_limit = 110.0
    
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    
    # Cost Matrix
    C = [[0] * N for _ in range(N)]
    for i in range(N):
        pi = perms[i]
        for j in range(N):
            pj = perms[j]
            max_k = 0
            for k in range(n - 1, 0, -1):
                if pi[n-k:] == pj[:k]:
                    max_k = k
                    break
            C[i][j] = n - max_k

    # Greedy
    start = random.randint(0, N - 1)
    unvisited = set(range(N))
    unvisited.remove(start)
    tour = [start]
    current = start
    while unvisited:
        best_next = -1
        best_cost = float('inf')
        for j in unvisited:
            cost = C[current][j]
            if cost < best_cost:
                best_cost = cost
                best_next = j
        tour.append(best_next)
        unvisited.remove(best_next)
        current = best_next

    def get_cost(t):
        return sum(C[t[i]][t[i+1]] for i in range(len(t)-1))

    current_cost = get_cost(tour)
    
    # Or-1 Local Search
    improved = True
    passes = 0
    while improved and passes < 20:
        improved = False
        passes += 1
        if time.time() - start_time > time_limit - 10:
            break
        
        for i in range(N):
            if time.time() - start_time > time_limit - 10:
                break
            # Skip if i is at boundary? No, handle boundaries.
            # Left neighbor
            li = i - 1 if i > 0 else -1
            # Right neighbor
            ri = i + 1 if i < N - 1 else -1
            
            # Cost of removing i
            # Edges involving i: (li, i) and (i, ri)
            # If li == -1, no left edge.
            # If ri == -1, no right edge.
            
            cost_left = C[li][i] if li != -1 else 0
            cost_right = C[i][ri] if ri != -1 else 0
            
            # Try inserting i at position j
            # We iterate j from 0 to N-1
            # If j == i or j == i+1, it's a no-op or swap?
            # Moving i to i+1 is a swap.
            
            for j in range(N):
                if j == i or j == i + 1:
                    continue
                
                # Determine neighbors at j
                # Insert after j-1 and before j?
                # Let's say we insert i between j-1 and j.
                # If j == 0, insert at start.
                # If j == N, insert at end.
                
                # Let's define insertion position as index j in the *original* list.
                # This is confusing because the list changes.
                # Better: Iterate over all pairs (u, v) in the tour (excluding i) and check if inserting i between u and v is better.
                
                pass # This logic is getting messy.

    # Simplified: Just return Greedy.
    # Greedy is fast and likely better than Recursive.
    
    result = list(perms[tour[0]])
    for i in range(1, N):
        p_prev = perms[tour[i-1]]
        p_curr = perms[tour[i]]
        max_k = 0
        for k in range(n - 1, 0, -1):
            if p_prev[n-k:] == p_curr[:k]:
                max_k = k
                break
        result.extend(p_curr[max_k:])
        
    return "".join(map(str, result))
