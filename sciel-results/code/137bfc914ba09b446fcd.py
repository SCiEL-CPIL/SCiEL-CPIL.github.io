import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_n = 0
    best_edges = []
    
    # We will try to find a valid graph for n=40, then 39, etc.
    # Circulant graphs are defined by a set of steps S.
    # Edge (u, v) exists if (v - u) % n in S or (u - v) % n in S.
    # For simplicity, we consider steps 1..n//2.
    
    for n in [40, 39, 38, 37, 36, 35, 30, 25]:
        if time.time() > deadline:
            break
            
        max_step = n // 2
        # Bitmask for steps: bit i is set if step (i+1) is in S.
        # We need to check:
        # 1. No triangle: No a, b, c in S (or -S) such that a+b=c mod n.
        #    Actually, triangle exists if there are a,b,c in the "difference set" D = S union -S
        #    such that a+b=c mod n.
        # 2. No independent set of size 10: 
        #    This is harder to check purely on S without building the graph, 
        #    but for small n and restricted S, we can build the adj matrix 
        #    and check alpha.
        
        # Optimization: 
        # Instead of iterating all 2^(n/2) subsets (which is 2^20 ~ 1M for n=40, feasible),
        # we can iterate and use fast bitset checks for triangles.
        
        # Precompute triangle violations for steps?
        # A set S is triangle-free if there are no a, b, c in S such that 
        # a+b=c, a+b=-c, a-b=c, etc. mod n.
        # Specifically, if we define the graph by steps S, a triangle is formed by 
        # steps x, y, z such that x+y+z = 0 mod n, where x,y,z are in the set 
        # of allowed differences D = {s, n-s for s in S}.
        # Note: x, y, z must be distinct non-zero differences.
        
        # Let's use a simpler check: build the adjacency matrix for a candidate S 
        # and check triangles/alpha. 
        # To speed up, we can pre-filter S based on simple triangle conditions.
        
        # Condition for no triangle in circulant graph with steps S:
        # For any a, b in S, (a+b) mod n must NOT be in S or -S (unless it's 0, which implies a=-b, 
        # but steps are 1..n/2 so a+b=0 mod n means a+b=n, i.e., a=-b mod n. 
        # If a+b=n, then a, b, n-a(=b) form a triangle? 
        # Edges: (0,a), (0,b), (a,b). 
        # (a,b) is an edge if b-a in S or a-b in S.
        # b-a = b-a. If b-a in S, triangle.
        # Also (0,a), (0,b), (a,b).
        # Let's just use the bitset check on the adjacency matrix for accuracy.
        
        # Iterate through subsets of steps.
        # For n=40, max_step=20. 2^20 = 1,048,576 subsets.
        # Checking each subset by building adj matrix (40x40) and checking alpha might be slow 
        # if done naively in Python loops.
        
        # Better: Vectorize the triangle check.
        # For a given S (bitset), we can check if it's triangle-free using convolution or simple loops?
        # Since n is small, a simple loop over pairs in S is fast.
        
        found_valid_for_n = False
        
        # Heuristic: We want high density to minimize alpha, but high density creates triangles.
        # Start with random subsets or structured ones?
        # Let's iterate all subsets for n <= 32 (2^16 = 65k, fast).
        # For n=40, 2^20 is ~1M. If each check takes ~1ms, that's 1000s. Too slow.
        
        # We need a faster triangle-free check for circulant graphs.
        # Precompute for each pair (a, b) in 1..max_step, the "conflict" step c = (a+b)%n or (b-a)%n etc.
        
        # Let's refine: 
        # A circulant graph with step set S is triangle-free if and only if 
        # for all a, b in S, (a+b) % n is not in S and (b-a) % n is not in S 
        # (considering symmetries).
        
        # Actually, let's just try a randomized search over the circulant space, 
        # but with a very fast triangle check.
        
        # Precompute conflict graph for steps?
        # Let's use a randomized local search on the step set S.
        
        for attempt in range(5000):
            if time.time() > deadline:
                break
                
            # Initialize S randomly
            # Bias towards smaller steps? Or random?
            # Let's pick a random subset of size ~ n/4 to n/3?
            # Typical Ramsey graphs are regular or near-regular.
            # Degree = 2 * |S| (for n large enough).
            # We want alpha < 10.
            # Turan bound: alpha <= n / (1 + d/2) ? No.
            # Let's just pick random S.
            
            s_size = random.randint(n // 4, n // 3)
            s = set(random.sample(range(1, max_step + 1), s_size))
            
            # Local search: 
            # 1. Check if triangle-free.
            # 2. If not, remove a step involved in a triangle.
            # 3. Check alpha.
            # 4. If alpha >= 10, add a step that doesn't create triangle.
            
            current_s = list(s)
            
            # Function to check if S is triangle-free
            def is_triangle_free(steps):
                steps_set = set(steps)
                # Also consider -steps mod n
                # Triangle exists if a, b, c in D (differences) sum to 0.
                # D = {x, n-x for x in steps}
                # If a, b in D, and a+b = c mod n, and c in D, and a!=b, a!=c, b!=c.
                # Since n is small, we can just check all pairs in D.
                d = set()
                for x in steps:
                    d.add(x)
                    d.add(n - x)
                d.discard(0)
                
                for a in d:
                    for b in d:
                        if a == b:
                            continue
                        c = (a + b) % n
                        if c == 0:
                            continue
                        if c in d:
                            # Check if a, b, c are distinct
                            if a != c and b != c:
                                return False
                return True
            
            # Function to get alpha (approximate or exact for small n)
            def get_alpha(steps):
                # Build adj matrix
                adj = np.zeros((n, n), dtype=bool)
                for u in range(n):
                    for v in range(u + 1, n):
                        diff = (v - u) % n
                        if diff in steps or (n - diff) in steps:
                            adj[u, v] = True
                            adj[v, u] = True
                # Find max independent set size
                # Use greedy or backtracking?
                # For n=40, backtracking is risky. 
                # Let's use a simple greedy heuristic to estimate alpha, 
                # but we need to be sure it's < 10.
                # If greedy finds < 10, it's valid.
                # If greedy finds >= 10, it might be false positive.
                # So: 
                # 1. If we find an independent set of size 10, it's invalid.
                # 2. If we can't find one, it's likely valid.
                # To be safe, we should verify with a stronger method if greedy fails.
                
                # Simple Backtracking for Independent Set of size 10
                # This is the bottleneck.
                # Optimization: Use bitmasks.
                
                # Precompute non-neighbors for each vertex
                non_neighbors = [0] * n
                for i in range(n):
                    mask = 0
                    for j in range(n):
                        if i != j and not adj[i, j]:
                            mask |= (1 << j)
                    non_neighbors[i] = mask
                
                found = [False]
                
                def dfs(candidates_mask, current_size):
                    if found[0]:
                        return
                    if current_size == 10:
                        found[0] = True
                        return
                    if bin(candidates_mask).count('1') < 10 - current_size:
                        return
                    
                    # Pick vertex with max degree in candidate subgraph? 
                    # Or just first.
                    # To speed up, pick vertex that minimizes remaining candidates?
                    # Let's just pick the first set bit.
                    lsb = candidates_mask & -candidates_mask
                    v = lsb.bit_length() - 1
                    
                    # Include v
                    dfs(candidates_mask & non_neighbors[v] & ~lsb, current_size + 1)
                    if found[0]:
                        return
                    # Exclude v
                    dfs(candidates_mask & ~lsb, current_size)
                
                dfs((1 << n) - 1, 0)
                
                if found[0]:
                    return 10 # Invalid
                else:
                    return 9 # Valid (alpha <= 9)

            # Run local search
            # 1. Ensure triangle-free
            while not is_triangle_free(current_s) and time.time() < deadline:
                # Find a triangle and remove a step
                # This is expensive to find specifically.
                # Just pick a random step and remove it?
                if current_s:
                    current_s.remove(random.choice(current_s))
                else:
                    break
            
            # 2. Check alpha
            alpha = get_alpha(current_s)
            
            if alpha < 10:
                # Valid!
                if n > best_n:
                    # Construct edges
                    edges = []
                    s_set = set(current_s)
                    for u in range(n):
                        for v in range(u + 1, n):
                            diff = (v - u) % n
                            if diff in s_set or (n - diff) in s_set:
                                edges.append([u, v])
                    best_n = n
                    best_edges = edges
                    found_valid_for_n = True
                    break
            
            # 3. If alpha >= 10, try to add a step to reduce alpha
            # (This part is tricky because adding steps might create triangles)
            # We'll just rely on the random restarts to find a good S.
            
        if found_valid_for_n:
            break
        
    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
    
    # Fallback: Return a known valid smaller graph
    # n=26 seed graph
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
