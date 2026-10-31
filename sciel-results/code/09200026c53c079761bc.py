import time
import random
import sys
from collections import deque
import numpy as np

def construct(k, g):
    deadline = time.time() + 125

    # --- LUW construction (fallback) ---
    def luw_construct(k, g):
        q = k
        # Determine d (half-girth related depth)
        # For girth g, we want cycles >= g.
        # LUW graphs have girth related to d.
        # Standard LUW for k=3 uses q=3, and d is chosen based on g.
        # If g is even, d = g/2 - 1 is often used, but let's stick to the original logic.
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
        # Rule definition for LUW
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # If not the specific hard case, just use LUW
    if k != 3 or g != 16:
        return luw_construct(k, g)

    # --- Fast Girth Verification ---
    # Uses the "non-backtracking" or "shortest path" property.
    # For a graph to have girth >= g, the ball of radius r around any vertex
    # must be a tree (no cycles) where 2*r+1 < g.
    # For g=16, r=7 means 2*7+1 = 15 < 16.
    # So we check if the BFS tree of depth 7 is indeed a tree (no cross edges within depth < 7? 
    # Actually, a cycle of length L corresponds to a path of length L-1 in the non-backtracking sense?
    # Simpler: BFS from v. If we find an edge (u, v) where dist[u] + dist[v] + 1 < g, and u != parent[v], 
    # then we have a cycle < g.
    # We only need to check up to depth g//2.
    
    def verify_girth_fast(n, adj, g):
        # adj is list of lists
        max_depth = g // 2
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        # v is already visited
                        # If v is parent of u, it's the back edge (OK)
                        # If v is child of u, it's the forward edge (OK, but we don't store children)
                        # If v is neither, it's a cross/back edge indicating a cycle.
                        if v != parent[u]:
                            # Cycle length is dist[u] + dist[v] + 1
                            # We must ensure this cycle is "short" enough to fail girth
                            # But wait: BFS finds shortest distances.
                            # Any edge (u,v) with dist[u] + dist[v] + 1 < g is a short cycle.
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True

    def build_adj(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    # --- Random 3-Regular Graph Generation ---
    def generate_random_3_regular(n):
        # Stubs method
        # n must be even for 3-regular? No, sum degrees = 3n must be even -> n must be even.
        if n % 2 != 0:
            n += 1 # Adjust to even? The task asks for n vertices. 
            # Wait, the Moore bound is 512. 512 is even.
            # If n is odd, 3-regular graph doesn't exist.
            # So we should only try even n.
        
        # Use Fisher-Yates or similar for shuffling stubs
        # Number of stubs = 3n
        # Pair them up.
        
        # To avoid self-loops and multi-edges efficiently, we can use a rejection sampling
        # or a more sophisticated algorithm like "pairing model" with rejection.
        # For n ~ 500-1000, simple rejection is fast.
        
        max_attempts = 50
        for _ in range(max_attempts):
            stubs = []
            for i in range(n):
                stubs.extend([i, i, i])
            random.shuffle(stubs)
            
            edges = []
            seen = set()
            valid = True
            for i in range(0, 3 * n, 2):
                u = stubs[i]
                v = stubs[i + 1]
                if u == v:
                    valid = False
                    break
                if (min(u, v), max(u, v)) in seen:
                    valid = False
                    break
                seen.add((min(u, v), max(u, v)))
                edges.append((u, v))
            
            if valid:
                return edges
        return None

    # --- Main Search Logic ---
    # Moore Bound for k=3, g=16:
    # n >= 1 + 3 * (1 + 2 + 4 + ... + 2^6) = 1 + 3 * (2^7 - 1) = 1 + 3 * 127 = 382?
    # Wait, for even girth g=2r, Moore bound is 1 + k * (k-1)^{r-1} + k * sum_{i=0}^{r-2} (k-1)^i ?
    # Actually:
    # For k=3, g=16 (r=8):
    # n >= 1 + 3 * ( (3-1)^7 - 1 ) / (3-2) ? No.
    # Formula: n >= 1 + k * sum_{i=0}^{r-2} (k-1)^i + k * (k-1)^{r-1}
    # r = g/2 = 8.
    # Sum_{i=0}^{6} 2^i = 2^7 - 1 = 127.
    # k * (k-1)^{r-1} = 3 * 2^7 = 384.
    # n >= 1 + 3 * 127 + 384 = 1 + 381 + 384 = 766.
    
    # So the Moore bound is 766.
    # The best known is 936.
    # We should search around 766 to 936.
    
    # Since n must be even for 3-regular:
    # Start at 766, then 768, 770...
    
    best_n = None
    best_edges = None
    
    # Try sizes from Moore bound up to 1000
    # Step size 2 (to ensure even)
    
    # To maximize chances, we try a few random graphs for each size.
    # Girth verification is O(n * (k-1)^{g/2}) roughly, which is manageable for n<1000.
    
    # Optimization: 
    # 1. Check connectivity? Girth check with BFS from every node will reveal components.
    #    If a component is small, it likely has short cycles.
    #    But if the graph is disconnected, girth is min girth of components.
    #    A random 3-regular graph is almost surely connected for n > ~20.
    
    # Let's try sizes: 766, 768, 770, ..., 1000
    # But 1000 - 766 = 234 / 2 = 117 sizes.
    # If we do 10 attempts each, that's 1170 random graphs.
    # Generating a random graph is O(n log n) or O(n). n=1000 is tiny.
    # Verification is the bottleneck.
    # Verification for n=1000, g=16.
    # BFS depth 8. Tree size ~ 3 * 2^7 = 384 nodes per start.
    # 1000 * 384 = 384,000 steps per verification.
    # 1170 * 384,000 ~ 450M steps. This might be too slow in Python (120s limit).
    
    # Strategy:
    # 1. Try a few sizes near the bound.
    # 2. If none work, fall back to LUW (which is deterministic and fast, but large).
    # 3. LUW for k=3, g=16:
    #    d = 16 - 5 = 11 (since 16-5=11 is odd? (16-5)%2=1 -> d=11).
    #    Nodes ~ 3 * 2^11 * 2 ? 
    #    LUW graph size for k=3 is roughly 2 * 3 * 2^{d-1} ? 
    #    Let's just run LUW to get a baseline n_luw.
    #    Then we know we have a valid graph of size n_luw.
    #    Then we try to beat it with random search.
    
    # Get LUW baseline
    n_luw, edges_luw = luw_construct(k, g)
    best_n = n_luw
    best_edges = edges_luw
    
    # Now try to improve
    # We know best_n. We only care about n < best_n.
    # Moore bound is 766.
    # So we try 766 to best_n - 2.
    
    # To save time, we can parallelize or just limit attempts.
    # Let's try 5 attempts per size.
    # If we run out of time, we return best_n.
    
    # Optimize verification:
    # We can pre-filter using a simpler check? 
    # Or just use the fast verifier.
    
    # Let's refine the search space.
    # If best_n is 936 (likely), we try 766 to 934.
    # That's 85 sizes.
    # 5 attempts each = 425 graphs.
    # 425 * 400k ops = 170M ops.
    # Python might do 10-50M simple ops/sec.
    # So 170M ops might take 3-15 seconds. This is feasible.
    
    # We'll add a time check inside the loop.
    
    # To be safe, we'll iterate sizes and break if time is up.
    # We should try smaller n first (closest to Moore bound) because they are rarer, 
    # but if we find one, it's the best.
    # Actually, the probability of a random 3-regular graph having girth >= 16 
    # decreases as n decreases (closer to bound) or increases?
    # Girth of random 3-regular graph is ~ log_{k-1}(n) = log_2(n).
    # For g=16, we need log_2(n) ~ 16 -> n ~ 2^16 = 65536? 
    # Wait. The girth of a random k-regular graph is asymptotically (1+o(1)) * (2 log_{k-1} n) / log log n ? 
    # No, the girth is roughly 2 log_{k-1}(n).
    # If g ~ 2 log_2(n), then n ~ 2^{g/2} = 2^8 = 256?
    # This suggests that for n=766, random graphs SHOULD have girth around 16.
    # In fact, 2 log_2(766) ~ 2 * 9.5 = 19.
    # So random 3-regular graphs of size 766 typically have girth > 16.
    # So we should find one quickly!
    
    # Therefore, searching from 766 upwards is a good strategy.
    # The first valid one we find will be very close to 766.
    
    # Let's start from 766.
    
    # Ensure n is even.
    start_n = 766
    if start_n % 2 != 0:
        start_n += 1
        
    # We will try sizes from start_n to best_n - 2
    # If we find a valid graph, we update best_n and best_edges, and continue to see if we can go lower?
    # No, we are going up. 766 is the lowest.
    # If we find one at 766, we can't go lower. So we can stop?
    # Yes, 766 is the Moore bound. We can't have n < 766.
    # So if we find a valid graph at 766, that is optimal.
    
    # So we iterate n from 766 to best_n - 2.
    # If we find a valid one, we record it and break?
    # Yes, because we are iterating upwards from the minimum possible.
    # Wait, if we start at 766, and it fails, we try 768.
    # If 768 succeeds, is it possible that 766 has a valid graph we missed?
    # We do multiple attempts. 
    # If we do 50 attempts for 766 and fail, it's unlikely there is one, but possible.
    # However, if we find one at 768, we can't be sure 766 is impossible.
    # But 766 is the bound. 
    # To be safe, we should exhaust 766 more thoroughly if we haven't found one there.
    # But time is limited.
    # Strategy:
    # Try 766 with many attempts (e.g., 100).
    # If found, return.
    # If not, try 768 with 50 attempts.
    # If found, return.
    # ...
    # If none found, return LUW.
    
    # Let's implement this.
    
    # Also, we should check time.
    
    # Pre-allocate adj list for verification?
    # Verification function builds adj list.
    
    # To speed up verification, we can use numpy?
    # No, standard library is fine for n=1000.
    
    # One more optimization: 
    # If the graph is disconnected, we can stop early?
    # Girth is min girth of components.
    # If we find a short cycle in any component, we fail.
    # Our BFS checks all vertices.
    
    # Let's write the search loop.
    
    # We'll use a fixed seed for reproducibility in debugging, but random for actual run?
    # The task says "random.seed(42)" in the current program.
    # I'll use random.seed(42) to be consistent with the "deterministic" hint, 
    # but since we are changing the algorithm, it doesn't matter much.
    # Actually, different seeds give different random graphs.
    # I'll just use the default random state.
    
    # Let's try sizes in steps of 2.
    # We'll allow more attempts for smaller n.
    
    # Budget: 120s.
    # LUW takes < 1s.
    # We have ~115s for search.
    
    # Let's try up to 200 sizes? No, 766 to 936 is only 85 sizes.
    
    # If best_n from LUW is smaller than 766? Impossible.
    # LUW n is usually large.
    
    # Let's go.
    
    current_n = start_n
    # We stop if current_n >= best_n
    while current_n < best_n and time.time() < deadline:
        # Number of attempts based on n?
        # More attempts for smaller n to be sure.
        # Or just fixed 100 attempts.
        attempts = 100
        
        found = False
        for _ in range(attempts):
            if time.time() > deadline - 5: # Safety margin
                break
            
            edges = generate_random_3_regular(current_n)
            if edges is None:
                continue
            
            # Verify
            adj = build_adj(current_n, edges)
            if verify_girth_fast(current_n, adj, g):
                best_n = current_n
                best_edges = edges
                found = True
                break
        
        if found:
            # We found a valid graph at current_n.
            # Since current_n is increasing from the Moore bound, this is the smallest possible n we found.
            # Can we go lower? No, current_n starts at Moore bound.
            # So we can break out of the main loop.
            break
        
        current_n += 2
        
    # If we didn't find one, we still have the LUW result.
    # But what if LUW result is huge and we can find a random one that is smaller but not near the bound?
    # E.g. if 766-936 all fail, maybe 1000 works?
    # Random 3-regular graphs of size 1000 have girth ~ 20. So they should work.
    # So we should extend the search if we didn't find one near the bound?
    # Yes.
    
    if best_n == n_luw: # If we didn't improve
        # Try some larger sizes
        # Start from 938? Or just best_n + 2?
        # But we want to MINIMIZE n.
        # If we failed 766-936, the next best is > 936.
        # But 938 is likely to work.
        # Let's try 938, 940, ...
        # Actually, if 766 failed 100 times, it's suspicious. 
        # Maybe my girth check is wrong?
        # Or maybe random graphs of size 766 rarely have girth 16?
        # Girth ~ 2 log_2(766) ~ 19. So they should have girth > 16.
        # So they SHOULD work.
        # If they don't, maybe there's a bug in verify or generate.
        
        # Let's try a known good size, e.g. 1000.
        for n_try in range(938, 1500, 2):
            if time.time() > deadline:
                break
            for _ in range(10):
                edges = generate_random_3_regular(n_try)
                if edges is None: continue
                adj = build_adj(n_try, edges)
                if verify_girth_fast(n_try, adj, g):
                    best_n = n_try
                    best_edges = edges
                    break
            if best_n != n_luw:
                break

    return best_n, best_edges
