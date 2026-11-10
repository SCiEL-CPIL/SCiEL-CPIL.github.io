import random
import sys
from collections import deque

def construct(k, g):
    if k != 3:
        raise ValueError("Only k=3 supported")
    
    # We will construct the Cayley graph of PSL(2, 13)
    # Order is 1092.
    # We need to find 3 involutions that generate the group and have girth >= 16.
    
    q = 13
    
    # GF(13) arithmetic
    def mod_inv(x, m):
        return pow(x, m-2, m)
        
    # Generate PSL(2, q)
    # Elements are equivalence classes of SL(2, q) under {I, -I}
    # We represent an element by a canonical matrix (a,b,c,d)
    # such that a > 0 or (a==0 and b > 0) etc? 
    # Simpler: Just generate SL(2, q) and divide by center?
    # Or just work with SL(2, q) and note that PSL is a quotient.
    # The Cayley graph of PSL is the same as the Cayley graph of SL with generators mapped appropriately?
    # Actually, if we pick generators in PSL, we can lift them to SL.
    # Let's work directly with PSL elements.
    
    # Represent PSL(2,13) elements as tuples (a,b,c,d) in GF(13)
    # Canonical form: 
    # 1. a != 0: set a > 0? In GF(13), no natural order.
    # We can use a dictionary to map matrices to indices.
    # To canonicalize (a,b,c,d) ~ (-a,-b,-c,-d):
    # Pick the one where the first non-zero entry is "positive" (1..12)?
    # In GF(13), we can just say: if a != 0, keep (a,b,c,d) if a < 13/2? No.
    # Just pick a representative.
    # Let's say: if a != 0, we prefer the one where a is in {1, ..., 6}?
    # If a == 0, prefer b in {1, ..., 6}?
    # If a==0 and b==0, then c != 0 (since det=1). Prefer c in {1,...,6}?
    
    def canonicalize(a, b, c, d):
        # Check if (a,b,c,d) is "positive"
        # We want to pick between M and -M.
        # Let's define a key:
        # If a != 0: return (a,b,c,d) if a <= 6 else (-a,-b,-c,-d)
        # If a == 0 and b != 0: return (a,b,c,d) if b <= 6 else (-a,-b,-c,-d)
        # If a == 0 and b == 0: c != 0. return (a,b,c,d) if c <= 6 else (-a,-b,-c,-d)
        if a != 0:
            if a > 6:
                return (-a % q, -b % q, -c % q, -d % q)
            return (a, b, c, d)
        elif b != 0:
            if b > 6:
                return (-a % q, -b % q, -c % q, -d % q)
            return (a, b, c, d)
        else:
            # c must be non-zero
            if c > 6:
                return (-a % q, -b % q, -c % q, -d % q)
            return (a, b, c, d)

    elements = []
    elem_to_idx = {}
    
    for a in range(q):
        for b in range(q):
            for c in range(q):
                # ad - bc = 1
                # If a == 0: -bc = 1 => bc = -1 = 12.
                # If a != 0: d = (1 + bc) * a^(-1)
                if a == 0:
                    if b == 0:
                        continue
                    # bc = 12
                    if 12 % b == 0: # Not in field, use div
                        c_val = (12 * mod_inv(b, q)) % q
                        # d is free?
                        # Matrix [[0, b], [c, d]]. det = -bc = 1.
                        # In PSL, (0,b,c,d) ~ (0,-b,-c,-d).
                        # We just generate all valid SL elements, then canonicalize.
                        for d in range(q):
                            e = canonicalize(0, b, c_val, d)
                            if e not in elem_to_idx:
                                elem_to_idx[e] = len(elements)
                                elements.append(e)
                else:
                    # d = (1 + bc) / a
                    for c in range(q):
                        bc = (b * c) % q
                        rhs = (1 + bc) % q
                        d = (rhs * mod_inv(a, q)) % q
                        e = canonicalize(a, b, c, d)
                        if e not in elem_to_idx:
                            elem_to_idx[e] = len(elements)
                            elements.append(e)
                            
    n = len(elements)
    # n should be 1092
    
    # Multiplication of PSL elements
    # (a1,b1,c1,d1) * (a2,b2,c2,d2)
    def mul(e1, e2):
        a1, b1, c1, d1 = e1
        a2, b2, c2, d2 = e2
        aa = (a1*a2 + b1*c2) % q
        ab = (a1*b2 + b1*d2) % q
        ba = (c1*a2 + d1*c2) % q
        bb = (c1*b2 + d1*d2) % q
        return canonicalize(aa, ab, ba, bb)
        
    # Find Involutions in PSL(2,13)
    # M^2 = I => M = M^-1.
    # In PSL, M^2 = I means M is an involution or Identity.
    # We want non-identity involutions.
    involutions = []
    for e in elements:
        # Check if e * e == Identity
        # Identity in PSL is (1,0,0,1)
        # Let's find the index of Identity
        pass
        
    # Find Identity index
    id_elem = canonicalize(1, 0, 0, 1)
    id_idx = elem_to_idx[id_elem]
    
    for e in elements:
        if e == id_elem:
            continue
        e2 = mul(e, e)
        if e2 == id_elem:
            involutions.append(e)
            
    # Search for 3 involutions that generate the group and have girth >= g
    random.seed(42)
    
    def get_girth(adj, limit):
        n_nodes = len(adj)
        min_girth = float('inf')
        for start in range(n_nodes):
            dist = [-1] * n_nodes
            parent = [-1] * n_nodes
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and u != parent[v]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                        if min_girth < limit:
                            return min_girth
        return min_girth

    # To speed up, we can check connectivity first.
    # Build adj for a set S
    def build_adj(S):
        adj = [[] for _ in range(n)]
        for i, e in enumerate(elements):
            for s in S:
                f = mul(e, s)
                j = elem_to_idx[f]
                adj[i].append(j)
        return adj
        
    # Try random sets
    # Since n=1092, building adj is fast.
    # We need to ensure S generates the group.
    # We can check connectivity of the graph.
    
    for _ in range(5000):
        S = random.sample(involutions, 3)
        # Check if they are distinct (random.sample ensures this)
        adj = build_adj(S)
        
        # Check connectivity
        visited = [False]*n
        q_d = deque([0])
        visited[0] = True
        count = 1
        while q_d:
            u = q_d.popleft()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    count += 1
                    q_d.append(v)
        if count != n:
            continue
            
        girth = get_girth(adj, g)
        if girth >= g:
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return n, edges
            
    # If not found, try a specific known set?
    # Or raise error.
    raise RuntimeError("Failed to find valid graph in PSL(2,13)")
