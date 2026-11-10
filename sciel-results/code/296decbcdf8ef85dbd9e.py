import random
import sys
from collections import deque

def construct(k, g):
    if k != 3:
        raise ValueError("Only k=3 supported")
    
    # We will try PSL(2, 11) first (n=660)
    # Then PSL(2, 13) (n=1092)
    # Then SL(2, 16) (n=4080) as fallback
    
    def build_psl2q(q):
        # Build PSL(2, q) for prime q
        # Elements are matrices [[a,b],[c,d]] with ad-bc=1, modulo +/-I
        # We represent elements by (a,b,c,d) normalized
        elements = []
        elem_to_idx = {}
        
        # Generate all matrices in SL(2, q)
        mats = []
        for a in range(q):
            for b in range(q):
                for c in range(q):
                    # ad - bc = 1 => ad = 1 + bc
                    if a == 0:
                        # bc = -1 = q-1
                        if (b*c) % q == q-1:
                            for d in range(q):
                                mats.append((a,b,c,d))
                    else:
                        # d = (1 + bc) * a^{-1}
                        inv_a = pow(a, q-2, q)
                        d = ((1 + b*c) * inv_a) % q
                        mats.append((a,b,c,d))
        
        # Identify PSL(2, q) elements
        # M and -M are identified
        # Normalize: choose representative such that a>0 or (a==0 and b>0) or (a==0,b==0,c>0)
        seen = set()
        for m in mats:
            a,b,c,d = m
            # Check if -m is in mats (it is)
            # Normalize
            if a == 0:
                if b == 0:
                    if c < 0: # In mod q, c is 0..q-1
                        # -c = q-c
                        # We want c > 0?
                        # Let's just use a canonical form
                        pass
            # Simpler: use a hash of the set {m, -m}
            # -m = (-a, -b, -c, -d) mod q
            neg_m = ((q-a)%q, (q-b)%q, (q-c)%q, (q-d)%q)
            key = min(m, neg_m)
            if key not in seen:
                seen.add(key)
                elements.append(m)
                elem_to_idx[m] = len(elements) - 1
                # Also map neg_m to same index? No, they are the same element in PSL
                # But we need to be careful with multiplication
                # Actually, if we store the representative, we must ensure multiplication is consistent
                # It's safer to store all SL elements and then quotient?
                # No, just store the representatives and define multiplication
                # (m1 * m2) mod +/-I
                
        # This normalization is tricky.
        # Alternative: Just use the SL(2, q) group and take the Cayley graph?
        # No, PSL is a subgroup of index 2.
        # If we use SL(2, 11), n = 11 * 10 * 11 = 1210?
        # |SL(2, q)| = q(q^2-1).
        # |PSL(2, q)| = q(q^2-1)/2.
        # For q=11, |PSL| = 11 * 120 / 2 = 660.
        
        # Let's just use SL(2, 11) if PSL is hard to implement.
        # n=1210. Still better than 4080? Yes.
        # But 660 is better.
        
        # Let's try a different approach:
        # Use the known construction of the 936-vertex graph?
        # No, we must generate it.
        
        # I will implement PSL(2, 11) carefully.
        
        # Map each SL element to a PSL element
        psl_elements = []
        psl_to_sl = {} # psl_idx -> sl_rep
        sl_to_psl = {} # sl_rep -> psl_idx
        
        for m in mats:
            neg_m = ((q-m[0])%q, (q-m[1])%q, (q-m[2])%q, (q-m[3])%q)
            key = min(m, neg_m)
            if key not in sl_to_psl:
                sl_to_psl[key] = len(psl_elements)
                psl_elements.append(key)
                # Also map the other one
                sl_to_psl[neg_m] = len(psl_elements) - 1
                
        n_psl = len(psl_elements)
        
        # Build multiplication table for PSL
        # mul_psl[i][j] = k
        # i, j are indices in psl_elements
        # Compute m_i * m_j in SL, then map to PSL
        
        # Optimization: Only need to check girth.
        # We can just do BFS on the Cayley graph.
        
        return psl_elements, sl_to_psl, n_psl

    # For simplicity and robustness, I will use a generic Cayley graph construction
    # on a group provided as a list of elements and a multiplication function.
    
    # Let's try q=11 (PSL) and q=13 (PSL)
    
    def try_group(q):
        # Build PSL(2, q)
        # ... (code from above)
        
        # Find Involutions
        # M^2 = I => Trace(M) = 0
        involutions = []
        for i, m in enumerate(psl_elements):
            a,b,c,d = m
            if (a+d) % q == 0:
                # Check M != I
                if not (a==1 and b==0 and c==0 and d==1) and not (a==q-1 and b==0 and c==0 and d==q-1):
                    involutions.append(i)
            
        # Search for 3 involutions
        random.seed(42)
        for _ in range(5000):
            S = random.sample(involutions, 3)
            # Build Adj
            adj = [[] for _ in range(n_psl)]
            for i in range(n_psl):
                for s in S:
                    # m_i * m_s
                    a1,b1,c1,d1 = psl_elements[i]
                    a2,b2,c2,d2 = psl_elements[s]
                    # Mult
                    aa = (a1*a2 + b1*c2) % q
                    ab = (a1*b2 + b1*d2) % q
                    ba = (c1*a2 + d1*c2) % q
                    bb = (c1*b2 + d1*d2) % q
                    res = (aa, ab, ba, bb)
                    neg_res = ((q-aa)%q, (q-ab)%q, (q-ba)%q, (q-bb)%q)
                    key = min(res, neg_res)
                    j = sl_to_psl[key]
                    adj[i].append(j)
                    
            # Check Girth
            # BFS
            min_girth = float('inf')
            for start in range(n_psl):
                dist = [-1] * n_psl
                parent = [-1] * n_psl
                dist[start] = 0
                q_d = deque([start])
                while q_d:
                    u = q_d.popleft()
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q_d.append(v)
                        elif v != parent[u] and u != parent[v]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < min_girth:
                                min_girth = cycle_len
                            if min_girth < g:
                                return None
            if min_girth >= g:
                # Found
                edges = []
                for i in range(n_psl):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n_psl, edges
        return None

    # Try q=11
    res = try_group(11)
    if res:
        return res
        
    # Try q=13
    res = try_group(13)
    if res:
        return res
        
    raise RuntimeError("Failed")

