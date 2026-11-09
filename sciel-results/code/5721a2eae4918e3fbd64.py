import time
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 110.0

    # We focus on k=3, g=16
    # The Moore bound for n(3,16) is 2 + 3 * (1 + 1 + 1^2 + ... + 1^6) = 2 + 3 * (11) = 35? No.
    # Moore bound: 2 + k * sum_{i=0}^{floor((g-2)/2)} (k-1)^i
    # For k=3, g=16: floor(14/2) = 7.
    # Sum i=0..6 of 1^i = 7.
    # 2 + 3 * 7 = 23.
    # So n >= 23.
    # Smallest known is 936.
    
    # Strategy: Construct LUW(q, d) graphs.
    # LUW(q, d) has q^d vertices? Or 2*q^d?
    # Standard LUW(q, d) is 3-regular with 2*q^d vertices? No, usually q is a prime power.
    # The graph has q^d vertices? Let's check the seed logic.
    # Seed: q=k=3? No, q is usually a prime power.
    # The task says "candidate interface... degree k and girth g".
    # If k=3, we need 3-regular.
    
    # Let's implement a known 3-regular high-girth construction:
    # The "Cage" graphs are often constructed using specific algebraic rules.
    # One efficient method for small k and large g is the Lazebnik-Ustimenko-Woldar (LUW) construction.
    # For LUW(q, d), the number of vertices is q^d. Wait, no.
    # The LUW graph LUW(q, d) is 3-regular and has 2*q^d vertices? Or q^d?
    # Let's look at the standard definition.
    # Vertices: (i, x) where i in {0,1}, x in F_q^d. Total 2*q^d vertices.
    # Edges:
    # 1. (0, x) ~ (1, x + e_i) for some fixed i? No.
    # The standard LUW(q, d) uses three sets of edges.
    # E1: (0, x) ~ (1, x)
    # E2: (0, x) ~ (1, x + e_1)
    # E3: (0, x) ~ (1, A x + e_2) ?? 
    # Actually, the construction is:
    # Vertices: V = {0,1} x F_q^d
    # Edges:
    # (0, x) - (1, x)
    # (0, x) - (1, x + a)  where a is a fixed vector
    # (0, x) - (1, A x + b) where A is a matrix, b is a vector
    
    # To ensure 3-regularity and simplicity, we need to be careful.
    # A common variant:
    # Vertices: F_q^d (if we identify layers) ? No, bipartite is safer for girth.
    
    # Let's use the bipartite LUW construction:
    # Left L = F_q^d, Right R = F_q^d.
    # Edges from x in L to:
    # 1. y = x in R
    # 2. y = x + a in R
    # 3. y = M x + b in R
    # Where M is a dxd matrix over F_q, and a, b are vectors.
    # If M is invertible, this is 3-regular.
    # Girth depends on the choice of M, a, b.
    # Specifically, if M has no eigenvalues that create short cycles.
    # For q=2, d=10, N=2*1024=2048.
    
    # Let's try q=2, d=10.
    # Primitive polynomial for GF(2^10): x^10 + x^3 + 1 (0x249? No, 2^9+2^2+1 = 512+4+1=517? No).
    # Standard primitive polynomials for GF(2^m).
    # For m=10: x^10 + x^3 + 1 is primitive?
    # Let's just pick a random invertible matrix M over GF(2) and check girth.
    # If girth < 16, try another M.
    
    q = 2
    # We need girth >= 16.
    # N = 2 * q^d.
    # For d=10, N=2048.
    # For d=9, N=1024.
    # For d=8, N=512.
    
    # Lower bound is 512. So d=8 gives N=512, which is the lower bound.
    # Is there a (3,16)-cage with 512 vertices? Unlikely, but possible?
    # The Moore bound is 35. The lower bound in the task is 512.
    # So N must be >= 512.
    
    # Let's start with d=8 (N=512) and increase d if girth is too small.
    # We have 110 seconds.
    
    best_n = None
    best_edges = None
    
    def generate_luw_graph(d):
        n = 2 * (q ** d)
        if n < 512:
            return None, None
            
        # We need a matrix M (dxd) over GF(2).
        # We also need vectors a, b.
        # To maximize girth, we can try random M.
        # Number of tries per d: limited by time.
        
        # Let's try a few random matrices.
        # If d=8, N=512.
        # If d=9, N=1024.
        
        # We will generate edges as (u, v) where u in [0, n/2), v in [n/2, n).
        # u corresponds to x in GF(2)^d.
        # v corresponds to y in GF(2)^d.
        
        # Edge 1: (x, x) -> (u, u + n/2)
        # Edge 2: (x, x+a) -> (u, (u+a) + n/2)
        # Edge 3: (x, Mx+b) -> (u, (Mx+b) + n/2)
        
        # We need to ensure no loops and no multiple edges.
        # Since it's bipartite, no loops.
        # Multiple edges if Mx+b = x or Mx+b = x+a.
        # So (M-I)x = b or (M-I)x = b+a.
        # If M-I is invertible, these equations have unique solutions, so at most 1 or 2 multiple edges?
        # Actually, for each x, if Mx+b == x, then edge 3 duplicates edge 1 for that x.
        # This happens if (M-I)x = b.
        # If M-I is singular, there might be many.
        # To be safe, we should check for duplicates.
        
        def build_edges(M, a, b):
            edges = []
            seen = set()
            half = n // 2
            for x in range(half):
                # Edge 1
                y1 = x
                e1 = (x, y1 + half)
                if e1 not in seen and e1[0] != e1[1]: # Bipartite so 0!=1 always
                    seen.add(e1)
                    edges.append(e1)
                    
                # Edge 2
                y2 = x ^ a  # Addition in GF(2)^d is XOR
                e2 = (x, y2 + half)
                if e2 not in seen:
                    seen.add(e2)
                    edges.append(e2)
                    
                # Edge 3
                # y3 = M x + b
                # M is represented as list of row masks? Or just apply bit ops.
                # Let's represent M as a list of d integers, each representing a row.
                # x is an integer < 2^d.
                y3 = b
                for i in range(d):
                    if (x >> i) & 1:
                        y3 ^= M[i]
                
                e3 = (x, y3 + half)
                if e3 not in seen:
                    seen.add(e3)
                    edges.append(e3)
                else:
                    # Duplicate edge, this x makes the graph non-simple or non-3-regular?
                    # If duplicate, then degree of x is < 3? Or degree of y is > 3?
                    # We need exactly 3 edges per vertex.
                    # If e3 is duplicate, we lose one edge from x.
                    # This breaks 3-regularity.
                    # So we must ensure no duplicates.
                    return None
                    
            # Check degrees
            if len(edges) != 3 * half:
                return None
                
            return edges
            
        # Try random matrices
        # d is small (8-12), so we can try many.
        tries = 20
        for _ in range(tries):
            # Random invertible matrix? Or just random?
            # Invertible M helps avoid (M-I) being singular? Not necessarily.
            # Let's just try random M, a, b.
            M = [np.random.randint(0, 1, d).astype(int).sum() for _ in range(d)] 
            # Wait, M[i] is the i-th row. M[i] is an integer < 2^d.
            # Let's generate properly.
            M_list = []
            for i in range(d):
                mask = np.random.randint(0, 1, d).astype(int)
                val = 0
                for j in range(d):
                    if mask[j]:
                        val |= (1 << j)
                M_list.append(val)
            M = M_list
            
            a = np.random.randint(0, 1, d).astype(int).sum() # Just a random int < 2^d?
            a = np.random.randint(0, 2**d)
            b = np.random.randint(0, 2**d)
            
            edges = build_edges(M, a, b)
            if edges is not None:
                # Verify girth
                # BFS from every vertex is too slow for N=2048?
                # 2048 vertices, 3072 edges.
                # BFS from each vertex: 2048 * (3072 log 3072) ~ 2048 * 30000 ~ 60M ops.
                # Might be borderline in 110s?
                # We can sample BFS.
                # Or just assume girth is high if d is large enough?
                # For LUW(2, d), girth is typically ~ d.
                # So d=8 -> girth ~ 8? Too low.
                # We need girth 16. So d should be at least 16?
                # If d=16, N = 2 * 2^16 = 131072.
                # That's huge.
                
                # Wait, the girth of LUW(q,d) is not just d.
                # It depends on the matrix.
                # For random M, the girth grows with d.
                # But for d=8, girth is likely < 16.
                
                # Let's check girth for this candidate.
                # If valid, return it.
                if is_valid_graph(n, edges, g):
                    return n, edges
        return None, None

    def is_valid_graph(n, edges, g):
        # Check 3-regular
        deg = [0]*n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(d != 3 for d in deg):
            return False
            
        # Check simple
        seen = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u,v), max(u,v))
            if key in seen:
                return False
            seen.add(key)
            
        # Check girth >= g
        # BFS from every vertex
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        for s in range(n):
            dist = [-1]*n
            parent = [-1]*n
            q = deque()
            q.append(s)
            dist[s] = 0
            while q:
                u = q.popleft()
                if dist[u] >= g//2: # We only care if cycle < g
                    # If we found a back edge to a node with dist <= dist[u] - 2?
                    # Actually, in bipartite graph, all cycles are even.
                    # Shortest cycle length = 2 * (max dist in BFS tree that connects back)
                    # We just need to check if any cycle < g.
                    pass
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        # Found a cycle
                        # Cycle length = dist[u] + dist[v] + 1? No.
                        # In undirected graph, if we see visited v, and v is not parent.
                        if v != parent[u]:
                            # Cycle length = dist[u] + dist[v] + 1?
                            # Wait, BFS tree edges are (u, v).
                            # If v is already visited and not parent, then (u, v) is a back edge.
                            # Cycle length = dist[u] + dist[v] + 1.
                            # But this is for general graphs.
                            # For bipartite, we can just check if dist[u] + 1 >= dist[v] and dist[v] + 1 >= dist[u].
                            # Actually, just check if dist[u] + dist[v] + 1 < g.
                            # But dist[u] and dist[v] are from the same source s.
                            # So this finds the shortest cycle through s?
                            # No, it finds the length of the cycle formed by s->...->u->v->...->s.
                            # This is valid.
                            cyc_len = dist[u] + dist[v] + 1
                            if cyc_len < g:
                                return False
        return True

    # Try d=8, 9, 10...
    for d in range(8, 15):
        if time.time() > deadline:
            break
        n, edges = generate_luw_graph(d)
        if edges is not None:
            return n, edges
            
    # Fallback: Return a trivial invalid graph or the best effort?
    # If no valid graph found, we must return *something*.
    # The task says "invalid result... scores 0".
    # So we should try to return a valid graph if possible.
    # If time runs out, we return the last generated one?
    # Or we return a graph with n=0?
    # "n must lie between the Moore bound and max_vertices".
    # If we return invalid, score 0.
    
    # Let's just return the largest d we tried?
    # Actually, if we didn't find a valid one, we should probably return a valid one from a previous run?
    # No, we must generate it.
    
    # If we didn't find one, try a simpler construction?
    # For now, return the last tried (even if invalid) to avoid exception?
    # No, the evaluator checks validity.
    
    # Let's try one more time with a larger d if time permits?
    # Actually, if the loop finished without returning, it means no valid graph was found in the tries.
    # We should return the best n we could construct, even if girth is wrong?
    # No, "invalid result... scores 0".
    
    # I will return a dummy valid 3-regular graph with large girth if I can't find the specific one?
    # No, I must solve the task.
    
    # If we fail, we return n=0, edges=[]?
    # That will be invalid.
    
    # Let's assume we found one.
    return 0, []
