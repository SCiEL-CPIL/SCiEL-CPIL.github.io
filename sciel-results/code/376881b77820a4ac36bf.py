import time
import random
import numpy as np

# Deadline guard - evaluator kills at 150s, use 130s limit
_DEADLINE = time.time() + 130.0

def _has_c5(n, nmask):
    """
    Check if graph (given by neighbor bitmasks) contains a C5.
    Optimized: For each edge (u,v), check for a path of length 3 u-a-b-c-v.
    """
    for u in range(n):
        for v in range(u+1, n):
            if not (nmask[u] >> v) & 1:
                continue
            
            # A = N(u) \ {v}
            a_mask = nmask[u] & ~(1 << v)
            if a_mask == 0:
                continue
                
            # B = N(v) \ {u}
            b_mask = nmask[v] & ~(1 << u)
            if b_mask == 0:
                continue
            
            # Iterate a in A
            am = a_mask
            while am:
                ab = am & (-am)
                a = ab.bit_length() - 1
                am ^= ab
                
                # We need c in B such that N(a) and N(c) share a neighbor b
                # b must not be u, v, a, or c
                # Let's iterate c in B
                cm = b_mask
                while cm:
                    cb = cm & (-cm)
                    c = cb.bit_length() - 1
                    cm ^= cb
                    
                    if a == c:
                        continue
                        
                    # Check if N(a) and N(c) have a common neighbor b
                    # b in N(a) & N(c)
                    # b != u, v, a, c
                    common = nmask[a] & nmask[c]
                    # Exclude u, v, a, c
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    common &= ~(1 << a)
                    common &= ~(1 << c)
                    
                    if common:
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
    full_mask = (1 << n) - 1
    for i in range(n):
        comp_mask[i] = full_mask & ~(1 << i) & ~nmask[i]
    
    found = [False]
    
    def search(R_size, P_mask, X_mask):
        if found[0]:
            return
        if R_size >= target:
            found[0] = True
            return
        if P_mask == 0:
            return
        
        # Upper bound: simple count
        if R_size + bin(P_mask).count('1') < target:
            return
        
        # Pivot: pick vertex in P∪X with max complement-degree in P
        U = P_mask | X_mask
        if U == 0:
            return
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
        if best_pivot >= 0:
            candidates = P_mask & ~comp_mask[best_pivot]
        else:
            candidates = P_mask
        
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
    
    search(0, full_mask, 0)
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

def _edges_from_nmask(n, nmask):
    """Extract edge list from neighbor bitmasks."""
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (nmask[i] >> j) & 1:
                edges.append([i, j])
    return edges

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
        
        found = False
        max_iters = 500000 # Increased iterations due to faster C5 check
        
        for it in range(max_iters):
            if time.time() > _DEADLINE - 15:
                break
            
            # Try to add an edge that helps cover extra vertices
            # Priority: connect extra vertices to K4 vertices
            extra_verts = list(range(32, n))
            if extra_verts and random.random() < 0.7:
                u = random.choice(extra_verts)
                # Pick a K4 to connect to - prefer K4s where u has few connections
                k4_idx = random.randint(0, 7)
                base = 4 * k4_idx
                # Pick a vertex in this K4 that u is not yet connected to
                candidates = [base+i for i in range(4) if not (nmask[u] >> (base+i)) & 1]
                if not candidates:
                    continue
                v = random.choice(candidates)
            else:
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
            
            if u == v or (nmask[u] >> v) & 1:
                continue
            
            # Try adding edge (u,v)
            nmask[u] |= (1 << v)
            nmask[v] |= (1 << u)
            
            # Check C5 first (cheaper)
            if _has_c5(n, nmask):
                nmask[u] &= ~(1 << v)
                nmask[v] &= ~(1 << u)
                continue
            
            # Check alpha
            if not _max_independent(n, nmask, 9):
                found = True
                best_n = n
                best_edges = _edges_from_nmask(n, nmask)
                break
            
            # Alpha still too large, revert
            nmask[u] &= ~(1 << v)
            nmask[v] &= ~(1 << u)
        
        if found:
            continue
        else:
            # If simple extension fails, try a different random restart strategy
            # But given the contradiction found earlier, 33 might be hard.
            break
    
    # If we couldn't extend, try alternative constructions
    if best_n < 33:
        # Try random construction for 33 vertices with different initial edges
        for attempt in range(20):
            if time.time() > _DEADLINE - 10:
                break
            n = 33
            nmask = [0] * n
            # Start with 8 K4s
            for b in range(8):
                base = 4 * b
                for i in range(4):
                    for j in range(i+1, 4):
                        nmask[base+i] |= (1 << (base+j))
                        nmask[base+j] |= (1 << (base+i))
            
            # Add some random edges between blocks to break symmetry?
            # Or just try to connect vertex 32 to specific vertices
            # We know connecting to a whole K4 creates C5.
            # So we must connect to parts of multiple K4s.
            
            # Strategy: Connect 32 to 2 vertices in K4_0, 2 in K4_1, etc?
            # If 32 is non-adjacent to at least one in EVERY K4, alpha >= 9.
            # So 32 MUST be adjacent to all 4 in SOME K4.
            # But that creates C5.
            # Thus, 33 vertices is IMPOSSIBLE if we start with 8 K4s.
            # We need a different base graph.
            
            # Try a base graph that is not 8 K4s.
            # e.g. 7 K4s + 1 K5? No, K5 has C5.
            # e.g. 7 K4s + 1 K3 + 1 K1?
            
            # Given the time, we will stick to 32 if 33 is not found quickly.
            pass
    
    # Final validation
    if best_n >= 1:
        if not _validate(best_n, best_edges):
            # Fallback to known valid 32-vertex graph
            best_n = 32
            best_edges = []
            for b in range(8):
                base = 4 * b
                for i in range(4):
                    for j in range(i+1, 4):
                        best_edges.append([base+i, base+j])
    
    return {'n': best_n, 'edges': best_edges}
