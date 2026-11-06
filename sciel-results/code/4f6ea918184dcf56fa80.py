import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin

    def is_c4_free(adj):
        """Check C4-free: no two distinct vertices share 2+ common neighbors."""
        n = adj.shape[0]
        if n <= 1:
            return True
        A = adj.astype(np.int32)
        A2 = A @ A
        # Off-diagonal entries >= 2 means a C4
        np.fill_diagonal(A2, 0)
        return not np.any(A2 >= 2)

    def polarity_graph(n):
        """
        Construct a C4-free graph using a polarity (symmetric bilinear form)
        over GF(q). Vertices are elements of GF(q), edge (a,b) exists iff
        a != b and the bilinear form B(a,b) = 0 (or 1, depending on convention).
        
        For q = p^k, use the trace form: B(x,y) = Tr(x*y) in GF(p).
        This gives a C4-free graph (no 4-cycles) with known degree properties.
        """
        # Try to find a prime power q close to n
        # For n=46 or 47, we need q around 46-47
        # q must be a prime power. Primes near 46: 43, 47
        # Also q=49=7^2, q=45 is not a prime power, q=46 is not
        # Let's use q=47 (prime) or q=43 (prime)
        
        # Actually, for polarity graphs, the standard construction:
        # Let q be a prime power. The graph has q vertices (or q+1 for projective).
        # Use the trace polarity: B(x,y) = Tr_{GF(q)/GF(p)}(x*y)
        # Edge between x,y iff B(x,y) = 0 and x != y.
        
        # For q=47 (prime): GF(47) = Z/47Z, Tr(x*y) = x*y mod 47 (since it's already in GF(p))
        # Wait, for GF(p) the trace is just the identity. So B(x,y) = x*y mod p.
        # Edge iff x*y ≡ 0 mod p. But that only gives edges from 0 to everyone.
        # That's not right. Let me reconsider.
        
        # Actually for GF(p), the trace map Tr_{GF(p)/GF(p)} is identity.
        # So B(x,y) = x*y mod p. Edge iff x*y = 0 in GF(p), i.e., x=0 or y=0.
        # That's a star, not useful.
        
        # For GF(q) where q = p^k, k>=2, the trace is non-trivial.
        # Let's use q = p^2 for a small prime p.
        # q=25 (5^2), q=49 (7^2), q=9 (3^2), q=4 (2^2), q=16 (4^2)
        
        # For n=46, q=49 is close. For n=47, q=49 works (49>=47, take subset).
        # Actually we need exactly n vertices. Let's use q >= n and take a subset.
        
        # Better approach: use q = 49 = 7^2. Build polarity graph on 49 vertices,
        # then if we need fewer, take a random subset.
        
        # But first, let me implement GF(p^2) properly.
        
        return None  # Placeholder, will implement below

    def gf_p2_construction(p, n):
        """
        Build a C4-free graph using the trace polarity on GF(p^2).
        GF(p^2) = GF(p)[t]/(t^2 - a) where a is a non-square in GF(p).
        Elements are (x, y) representing x + y*t.
        Multiplication: (x1,y1)*(x2,y2) = (x1*x2 + a*y1*y2, x1*y2 + x2*y1)
        Trace: Tr(x+y*t) = 2x (since Tr(t)=0, Tr(1)=1 in GF(p))
        Wait: Tr_{GF(p^2)/GF(p)}(x + y*t) = (x+y*t) + (x+y*t)^p = (x+y*t) + (x + y^p * t^p)
        Since p is odd, t^p = t^(p mod 2) ... actually in GF(p^2), t^2 = a, so t^p = t * t^(p-1).
        For p odd: (x+y*t)^p = x^p + y^p*t^p = x + y*t^p (Fresnel's little theorem in finite fields).
        t^p: since t has order p+1 in GF(p^2)*, t^p = t^(p+1-1) = 1/t = t/a (since t^2=a, so 1/t = t/a).
        So (x+y*t)^p = x + y*(t/a) = x + (y/a)*t.
        Tr(x+y*t) = (x+y*t) + (x + (y/a)*t) = 2x + (y + y/a)*t.
        For this to be in GF(p), we need y + y/a = 0, i.e., y(1+1/a)=0.
        This requires a = -1, or y=0.
        
        Let me use a different approach. Use a = -1 if -1 is a non-square, else use another.
        Actually, let me just use the standard construction:
        
        For GF(q) with q = p^k, the trace polarity graph:
        Vertices: GF(q)
        Edge (x,y) iff x != y and Tr(x*y) = 0
        
        This graph is C4-free and has degree approximately (q-1)/p - 1 or similar.
        
        For our purposes, let's use a simpler well-known C4-free construction:
        
        CONSTRUCTION: Let q be a prime power. Consider the graph with vertex set GF(q) x GF(q) 
        (so q^2 vertices) where (a,b) ~ (c,d) iff (a-c)*(b-d) = 1 in GF(q).
        This is the "polarity graph of the hyperbolic quadric" or similar.
        It's C4-free and has degree q.
        For q=7: 49 vertices, degree 7. Min degree 7.
        We need min degree >= n - 39. For n=49, need degree >= 10. Not enough.
        For n=46, need degree >= 7. 49 vertices with degree 7... but we only have 49 vertices.
        
        Hmm, this doesn't give high enough degree for n=46.
        
        Let me think differently. We need a C4-free graph on n=46 vertices with min degree >= 7.
        That's actually not that hard! A random graph G(n, p) with appropriate p might work.
        Or we can use the complement: we need the complement to have no K_{1,39}, i.e., max degree of complement <= 38, i.e., min degree of G >= n-39 = 7.
        
        So we need a C4-free graph on 46 vertices with min degree at least 7.
        That's very achievable! Even a union of cliques or a random graph should work.
        
        Let me just use a smart greedy approach with better heuristics, or use a known construction.
        
        Actually, the simplest approach: take a random graph G(46, p) and check.
        For C4-freeness, we need the expected number of C4s to be small.
        E[#C4] = C(46,4) * p^4 * (1-p)^{...} roughly.
        C(46,4) = 163185. For p=0.15: 163185 * 0.0005 ≈ 82. Too many.
        For p=0.1: 163185 * 0.0001 ≈ 16. Still too many.
        For p=0.05: 163185 * 0.00000625 ≈ 1. Might work sometimes.
        
        But min degree of G(46, 0.05): expected degree = 45*0.05 = 2.25. Way too low.
        
        The tension: high min degree vs C4-free.
        
        Better approach: Use the greedy method but with a much better heuristic.
        Start with a high-degree structure and remove edges to break C4s.
        
        OR: Use the fact that for n=46, target degree is only 7. 
        A 7-regular C4-free graph on 46 vertices should be easy to construct.
        
        Let me use a structured approach: 
        1. Start with a 7-regular graph (e.g., circulant or random regular).
        2. Check for C4s.
        3. If C4s exist, swap edges to break them while maintaining regularity/degree.
        
        Actually, even simpler: use the greedy edge-addition but with a much better ordering.
        Process vertices in order of current degree (most deficient first), and for each,
        pick the neighbor that minimizes future C4 risk.
        
        Let me implement a clean version of the priority-based greedy with better tie-breaking.
        """
        return None

    def greedy_c4_free(n, target_deg):
        """
        Greedy C4-free graph construction with priority on deficient vertices.
        Uses efficient common-neighbor tracking.
        """
        adj = np.zeros((n, n), dtype=np.int32)
        degrees = np.zeros(n, dtype=int)
        
        # For efficiency, track common neighbor counts
        # common[u,v] = number of common neighbors of u and v
        common = np.zeros((n, n), dtype=np.int32)
        
        # All edges
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # Multiple restarts with different random seeds
        for attempt in range(50):
            if time.time() - start_time > deadline - 20:
                break
            
            adj = np.zeros((n, n), dtype=np.int32)
            degrees = np.zeros(n, dtype=int)
            common = np.zeros((n, n), dtype=np.int32)
            
            # Use a priority queue approach: always add an edge incident to the most deficient vertex
            # But also randomize to explore different solutions
            
            random.seed(attempt * 12345 + 67890)
            random.shuffle(all_edges)
            
            # Process edges in shuffled order, but with a twist:
            # Prefer edges that help the most deficient vertex
            # We'll do this by sorting edges by the minimum degree of their endpoints (ascending)
            # and processing in that order, with random tie-breaking
            
            # Actually, let's do a hybrid: random order but with a bias
            # toward edges that connect low-degree vertices
            
            for u, v in all_edges:
                if time.time() - start_time > deadline - 20:
                    break
                
                if np.min(degrees) >= target_deg:
                    break
                
                # Only add edge if it helps a deficient vertex
                if degrees[u] >= target_deg and degrees[v] >= target_deg:
                    continue
                
                # Check C4 condition
                if common[u, v] <= 1:
                    # Safe to add
                    adj[u, v] = 1
                    adj[v, u] = 1
                    degrees[u] += 1
                    degrees[v] += 1
                    
                    # Update common neighbor counts
                    # For all w adjacent to u (but not v): common[w,v] += 1
                    # For all w adjacent to v (but not u): common[u,w] += 1
                    for w in range(n):
                        if w != u and w != v:
                            if adj[u, w]:
                                common[w, v] += 1
                                common[v, w] += 1
                            if adj[v, w]:
                                common[u, w] += 1
                                common[w, u] += 1
            
            if np.min(degrees) >= target_deg:
                # Verify
                if is_c4_free(adj.astype(bool)):
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i, j]:
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
        
        return None

    def local_search_c4_free(n, target_deg):
        """
        Start with a random graph, then use local search to:
        1. Remove edges to eliminate C4s
        2. Add edges to increase minimum degree
        While maintaining C4-freeness.
        """
        if time.time() - start_time > deadline - 20:
            return None
        
        # Start with a moderately dense random graph
        p = 0.3
        adj = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i, j] = 1
                    adj[j, i] = 1
        
        degrees = adj.sum(axis=1).astype(int)
        
        # Phase 1: Remove edges to make C4-free
        # Find C4s and remove an edge from each
        max_iters = 1000
        for it in range(max_iters):
            if time.time() - start_time > deadline - 20:
                return None
            
            # Check if C4-free
            A = adj.astype(np.int32)
            A2 = A @ A
            np.fill_diagonal(A2, 0)
            
            if not np.any(A2 >= 2):
                break  # C4-free!
            
            # Find a pair (u,v) with A2[u,v] >= 2
            mask = A2 >= 2
            if not np.any(mask):
                break
            
            # Get all problematic pairs
            pairs = np.argwhere(mask)
            u, v = pairs[0]
            
            # Remove a random edge from the C4
            # The C4 involves u, v, and two common neighbors
            common_neighbors = np.where(adj[u] & adj[v])[0]
            # Pick two common neighbors
            if len(common_neighbors) >= 2:
                cn1, cn2 = random.sample(common_neighbors.tolist(), 2)
                # Remove one of the four edges: (u,cn1), (u,cn2), (v,cn1), (v,cn2)
                # Remove the one that causes least degree damage
                candidates = [(u, cn1), (u, cn2), (v, cn1), (v, cn2)]
                # Pick randomly for diversity
                ru, rv = random.choice(candidates)
                adj[ru, rv] = 0
                adj[rv, ru] = 0
                degrees[ru] -= 1
                degrees[rv] -= 1
        
        # Phase 2: Add edges to increase minimum degree
        # While maintaining C4-freeness
        for it in range(5000):
            if time.time() - start_time > deadline - 20:
                break
            
            if np.min(degrees) >= target_deg:
                break
            
            # Find the most deficient vertex
            min_deg = np.min(degrees)
            deficient = np.where(degrees == min_deg)[0]
            u = random.choice(deficient)
            
            # Try to add an edge from u to some v
            candidates = []
            for v in range(n):
                if v == u or adj[u, v]:
                    continue
                # Check if adding (u,v) creates a C4
                common = np.sum(adj[u] & adj[v])
                if common <= 1:
                    candidates.append((common, v))
            
            if not candidates:
                # Can't add any edge to u. Try a different strategy:
                # Remove an edge elsewhere to free up capacity
                break
            
            # Pick the candidate with fewest common neighbors
            min_common = min(c[0] for c in candidates)
            best = [v for c, v in candidates if c == min_common]
            v = random.choice(best)
            
            adj[u, v] = 1
            adj[v, u] = 1
            degrees[u] += 1
            degrees[v] += 1
        
        if np.min(degrees) >= target_deg and is_c4_free(adj.astype(bool)):
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i, j]:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
        
        return None

    def try_n(n):
        """Try to construct a valid graph on n vertices."""
        target_deg = n - 39
        if target_deg < 0:
            target_deg = 0
        
        # Method 1: Greedy with priority
        result = greedy_c4_free(n, target_deg)
        if result is not None:
            return result
        
        # Method 2: Local search
        result = local_search_c4_free(n, target_deg)
        if result is not None:
            return result
        
        # Method 3: More aggressive local search with multiple restarts
        for restart in range(10):
            if time.time() - start_time > deadline - 15:
                break
            result = local_search_c4_free(n, target_deg)
            if result is not None:
                return result
        
        return None

    # Try n=47 first, then 46, 45, ...
    for n in [47, 46, 45, 44, 43, 42, 41, 40, 39]:
        if time.time() - start_time > deadline - 10:
            break
        
        result = try_n(n)
        if result is not None:
            return result
    
    # Fallback: n=39, empty graph (min degree 0 >= 39-39=0)
    return {'n': 39, 'edges': []}
