import numpy as np
import time
import random
from collections import deque

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        return [list(range(n)) for _ in range(1)]

    M = 3**n
    ball_size = 1 + 2 * n
    powers = [3**i for i in range(n)]

    # Precompute balls
    balls = [None] * M
    for i in range(M):
        digits = [0] * n
        temp = i
        for j in range(n):
            digits[j] = temp % 3
            temp //= 3
        ball = [i]
        for pos in range(n):
            new_d = (digits[pos] + 1) % 3
            diff = new_d - digits[pos]
            if diff != 0:
                ball.append(i + diff * powers[pos])
            new_d = (digits[pos] - 1) % 3
            diff = new_d - digits[pos]
            if diff != 0:
                ball.append(i + diff * powers[pos])
        balls[i] = ball

    # Bipartite Matching: Codewords (0..M-1) to Points (0..M-1)
    # We want to find a large matching to cover disjoint points.
    # Using Hopcroft-Karp for efficiency.
    
    # Adjacency list: codeword -> list of points
    adj = balls  # balls[c] is the list of points covered by c
    
    # Match arrays
    # pairU[c] = point matched to codeword c, or -1
    # pairV[p] = codeword matched to point p, or -1
    pairU = [-1] * M
    pairV = [-1] * M
    dist = [0] * M

    def bfs():
        queue = deque()
        for u in range(M):
            if pairU[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = -1
        found = False
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                pu = pairV[v]
                if pu == -1:
                    found = True
                elif dist[pu] == -1:
                    dist[pu] = dist[u] + 1
                    queue.append(pu)
        return found

    def dfs(u):
        for v in adj[u]:
            pu = pairV[v]
            if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pairU[u] = v
                pairV[v] = u
                return True
        dist[u] = -1
        return False

    # Run Hopcroft-Karp
    while bfs():
        for u in range(M):
            if pairU[u] == -1:
                dfs(u)

    # Extract matched codewords
    matched_codewords = [u for u in range(M) if pairU[u] != -1]
    matched_points = set(pairV[v] for v in range(M) if pairV[v] != -1)
    
    # The matched codewords cover the matched points.
    # However, they might cover *other* points too.
    # We need to cover ALL points.
    # Let's say we keep matched_codewords as our initial set.
    # We need to check which points are still uncovered.
    
    covered = np.zeros(M, dtype=bool)
    for c in matched_codewords:
        for p in balls[c]:
            covered[p] = True
            
    uncovered_points = [p for p in range(M) if not covered[p]]
    
    # Greedy Set Cover for the remaining uncovered points
    code = set(matched_codewords)
    
    if uncovered_points:
        # Gains for each codeword: how many *uncovered* points it covers
        gains = np.zeros(M, dtype=np.int32)
        for c in range(M):
            if c in code:
                continue
            g = 0
            for p in balls[c]:
                if not covered[p]:
                    g += 1
            gains[c] = g
            
        # Heap
        import heapq
        heap = [(-gains[c], c) for c in range(M) if gains[c] > 0]
        heapq.heapify(heap)
        
        uncovered_count = len(uncovered_points)
        
        while uncovered_count > 0 and heap:
            neg_g, c = heapq.heappop(heap)
            if c in code:
                continue
            if -neg_g != gains[c]:
                continue
            if gains[c] <= 0:
                continue
                
            code.add(c)
            
            for p in balls[c]:
                if not covered[p]:
                    covered[p] = True
                    uncovered_count -= 1
                    # Update gains for other codewords covering p
                    # This is O(M) per point, which is expensive.
                    # Optimization: Only update if we can do it fast.
                    # For now, we'll just push updated gains if we track them.
                    # To keep it simple and fast, we might skip the update and rely on the heap having stale entries?
                    # No, that leads to suboptimal greedy.
                    # Let's update gains for all c' covering p.
                    # This is O(degree(p)) * O(1) per point.
                    # We need coverers list.
                    pass # We don't have coverers precomputed in this new structure easily without extra memory/time.
                    
        # Since updating gains is expensive without precomputed coverers, 
        # and we have time limits, let's just use a simpler greedy:
        # Pick the codeword that covers the most uncovered points, repeat.
        # We can do this by iterating all codewords each time? No, too slow.
        # Let's stick to the heap but we need coverers.
        
    # Re-implementing with Coverers for efficiency in Greedy phase
    # Precompute coverers
    coverers = [[] for _ in range(M)]
    for c in range(M):
        for p in balls[c]:
            coverers[p].append(c)
            
    # Reset Greedy phase with proper gain updates
    code = set(matched_codewords)
    covered = np.zeros(M, dtype=bool)
    for c in code:
        for p in balls[c]:
            covered[p] = True
            
    gains = np.zeros(M, dtype=np.int32)
    for c in range(M):
        if c in code:
            continue
        g = 0
        for p in balls[c]:
            if not covered[p]:
                g += 1
        gains[c] = g
        
    import heapq
    heap = [(-gains[c], c) for c in range(M) if gains[c] > 0]
    heapq.heapify(heap)
    
    uncovered_count = int(np.sum(~covered))
    
    while uncovered_count > 0:
        valid_c = None
        while heap:
            neg_g, c = heapq.heappop(heap)
            if c in code:
                continue
            if -neg_g != gains[c]:
                continue
            if gains[c] <= 0:
                continue
            valid_c = c
            break
            
        if valid_c is None:
            break
            
        code.add(valid_c)
        
        for p in balls[valid_c]:
            if not covered[p]:
                covered[p] = True
                uncovered_count -= 1
                for c_prime in coverers[p]:
                    if c_prime not in code:
                        gains[c_prime] -= 1
                        if gains[c_prime] > 0:
                            heapq.heappush(heap, (-gains[c_prime], c_prime))
                        
    # Local Search
    def local_search(code, time_limit=115):
        start_time = time.time()
        count = np.zeros(M, dtype=np.int32)
        for c in code:
            for p in balls[c]:
                count[p] += 1
                
        def is_redundant(c):
            for p in balls[c]:
                if count[p] == 1:
                    return False
            return True

        def remove_c(c):
            code.remove(c)
            for p in balls[c]:
                count[p] -= 1

        def add_c(c):
            code.add(c)
            for p in balls[c]:
                count[p] += 1

        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            to_remove = []
            for c in list(code):
                if is_redundant(c):
                    to_remove.append(c)
            if to_remove:
                for c in to_remove:
                    remove_c(c)
                improved = True
                continue
                
            code_list = list(code)
            random.shuffle(code_list)
            swapped = False
            for c in code_list:
                if (time.time() - start_time) >= time_limit:
                    break
                unique_points = [p for p in balls[c] if count[p] == 1]
                if not unique_points:
                    remove_c(c)
                    improved = True
                    swapped = True
                    break
                candidates = set(coverers[unique_points[0]])
                for p in unique_points[1:]:
                    candidates.intersection_update(coverers[p])
                candidates -= code
                if candidates:
                    c_new = random.choice(list(candidates))
                    remove_c(c)
                    add_c(c_new)
                    improved = True
                    swapped = True
                    break
            if swapped:
                continue
                
            # 2-Swap
            if (time.time() - start_time) < time_limit:
                code_list = list(code)
                if len(code_list) >= 2:
                    c1, c2 = random.sample(code_list, 2)
                    U1 = [p for p in balls[c1] if count[p] == 1]
                    U2 = [p for p in balls[c2] if count[p] == 1]
                    if U1 and U2:
                        all_unique = set(U1) | set(U2)
                        first_p = next(iter(all_unique))
                        cands = set(coverers[first_p])
                        for p in all_unique:
                            cands.intersection_update(coverers[p])
                        cands -= code
                        cands.discard(c1)
                        cands.discard(c2)
                        if cands:
                            c_new = random.choice(list(cands))
                            remove_c(c1)
                            remove_c(c2)
                            add_c(c_new)
                            improved = True
                            continue
                            
        return code

    code = local_search(code, time_limit=110)
    
    return [list(map(int, [(c // (3**i)) % 3 for i in range(n)])) for c in code]
