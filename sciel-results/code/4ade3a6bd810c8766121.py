import time
import random
import numpy as np

# Deadline guard
_DEADLINE = time.time() + 100.0

def _has_c5(n, nmask):
    """
    Check if graph (given by neighbor bitmasks) contains a C5.
    For each edge (u,v), check if there exist a,b,c with
    u-a-b-c-v forming a 5-cycle.
    """
    for u in range(n):
        for v in range(u+1, n):
            if not (nmask[u] >> v) & 1:
                continue
            # a in N(u) \ {v}
            nu = nmask[u] & ~(1 << v)
            a_mask = nu
            while a_mask:
                a_bit = a_mask & (-a_mask)
                a = a_bit.bit_length() - 1
                a_mask ^= a_bit
                if a == v:
                    continue
                # b in N(a) \ {u, v}
                na = nmask[a] & ~(1 << u) & ~(1 << v)
                b_mask = na
                while b_mask:
                    b_bit = b_mask & (-b_mask)
                    b = b_bit.bit_length() - 1
                    b_mask ^= b_bit
                    # c in N(b) \ {a, u, v} and c in N(v)
                    nb = nmask[b] & ~(1 << a) & ~(1 << u) & ~(1 << v)
                    if nb & nmask[v]:
                        return True
    return False

def _max_independent(n, nmask, target=9):
    """
    Check if there's an independent set of size >= target.
    Independent set in G = clique in complement.
    Use branch-and-bound with bitmasks.
    """
    # Complement neighbor masks
    comp_mask = [0] * n
    for i in range(n):
        comp_mask[i] = ((1 << n) - 1) & ~(1 << i) & ~nmask[i]
    
    found = [False]
    
    def search(R_size, P_mask, X_mask):
        if found[0]:
            return
        if R_size >= target:
            found[0] = True
            return
        if P_mask == 0:
            return
        
        # Greedy upper bound: |P| + R_size
        if R_size + bin(P_mask).count('1') < target:
            return
        
        # Pivot: pick vertex in P∪X with max complement-degree in P
        U = P_mask | X_mask
        if U == 0:
            return
        # Pick pivot with max number of neighbors in P (within comp)
        best_pivot = -1
        best_deg = -1
        um = U
        while um:
            ub = um & (-um)
            uv = ub.bit_length() - 1
            um ^= ub
            deg = bin(comp_mask[uv] & P_mask).count('1')
            if deg > best_deg:
                best_deg = deg
                best_pivot = uv
        
        # Candidates: vertices in P NOT adjacent to pivot in comp
        candidates = P_mask & ~comp_mask[best_pivot] if best_pivot >= 0 else P_mask
        
        cm = candidates
        while cm:
            cb = cm & (-cm)
            cv = cb.bit_length() - 1
            cm ^= cb
            
            if found[0]:
                return
            
            new_P = P_mask & comp_mask[cv]
            new_X = X_mask & comp_mask[cv]
            
            search(R_size + 1, new_P, new_X)
            
            P_mask = P_mask & ~cb
            X_mask = X_mask | cb
    
    search(0, (1 << n) - 1, 0)
    return found[0]

def _validate(n, edges):
    """Validate: no C5 and alpha <= 8."""
    nmask = [0] * n
    for u, v in edges:
        nmask[u] |= (1 << v)
        nmask[v] |= (1 << u)
    if _has_c5(n, nmask):
        return False
    if _max_independent(n, nmask, 9):
        return False
    return True

def construct():
    start_time = time.time()
    
    # Start with the known valid 32-vertex graph: 8 disjoint K4s
    best_n = 32
    best_edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i+1, 4):
                best_edges.append([base+i, base+j])
    
    # Try to extend to 33, 34, 35 vertices
    for target_n in [33, 34, 35]:
        if time.time() > _DEADLINE - 20:
            break
        
        n = target_n
        # Build adjacency as bitmasks
        nmask = [0] * n
        # Initialize with 8 K4s
        for b in range(8):
            base = 4 * b
            for i in range(4):
                for j in range(i+1, 4):
                    nmask[base+i] |= (1 << (base+j))
                    nmask[base+j] |= (1 << (base+i))
        
        # Vertices 32..n-1 start isolated
        # We need to add edges to make alpha <= 8
        # Key insight: each extra vertex must be "covered" by edges that prevent
        # it from joining an independent set of 8 others
        
        found = False
        max_iters = 200000
        
        # Use a smarter strategy: for each extra vertex, try connecting it
        # to vertices in different K4s to break independence
        for it in range(max_iters):
            if time.time() > _DEADLINE - 20:
                break
            
            # Determine which extra vertices are still "independent-friendly"
            # A vertex v is dangerous if it can join 8 others in an independent set
            # Simple heuristic: prefer adding edges from extra vertices to K4 vertices
            
            if n > 32 and random.random() < 0.8:
                # Pick an extra vertex
                extra_verts = list(range(32, n))
                u = random.choice(extra_verts)
                # Pick a vertex from a K4
                v = random.randint(0, 31)
            else:
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
            
            if u == v or (nmask[u] >> v) & 1:
                continue
            
            # Try adding edge (u,v)
            nmask[u] |= (1 << v)
            nmask[v] |= (1 << u)
            
            # Check C5
            if _has_c5(n, nmask):
                nmask[u] &= ~(1 << v)
                nmask[v] &= ~(1 << u)
                continue
            
            # Check alpha
            if not _max_independent(n, nmask, 9):
                found = True
                best_n = n
                # Extract edges
                new_edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if (nmask[i] >> j) & 1:
                            new_edges.append([i, j])
                best_edges = new_edges
                break
            
            # Alpha still too large, revert
            nmask[u] &= ~(1 << v)
            nmask[v] &= ~(1 << u)
        
        if found:
            continue
        else:
            break
    
    return {'n': best_n, 'edges': best_edges}
