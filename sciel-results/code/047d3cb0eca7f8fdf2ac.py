import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145
    n = 20
    
    # 1. Construct the Clebsch Graph on 16 vertices (0..15)
    # Vertices are even-weight binary strings of length 5.
    # Edge if distance is 4.
    
    # Generate even-weight strings of length 5
    verts = []
    for i in range(32):
        if bin(i).count('1') % 2 == 0:
            verts.append(i)
    
    # Map to 0..15
    # verts is already 16 elements
    # Edge if d(u,v) == 4
    
    adj = [0] * n
    for i in range(16):
        for j in range(i + 1, 16):
            u = verts[i]
            v = verts[j]
            # Hamming distance
            d = bin(u ^ v).count('1')
            if d == 4:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # 2. Add 4 new vertices (16..19)
    # They start with no edges.
    
    # Helper to check if adding (u,v) is valid
    # We need to ensure no triple has >2 common neighbors.
    # Adding (u,v) only affects triples containing u or v.
    
    def can_add(u, v):
        # Check all triples containing u or v
        # Triples containing u: {u, a, b}
        # Triples containing v: {v, a, b}
        # Note: {u, v, w} is counted in both, but we can just iterate all w.
        
        # For any w, check if |N(u) \cap N(v) \cap N(w)| > 2
        # Note: if w == u or w == v, handle carefully.
        
        # Actually, the condition is for ALL triples.
        # Adding (u,v) changes N(u) and N(v).
        # New N(u) = N(u) \cup {v}
        # New N(v) = N(v) \cup {u}
        
        # For a triple {a,b,c}, the common neighbors are N(a) \cap N(b) \cap N(c).
        # If none of a,b,c is u or v, the intersection doesn't change.
        # If one is u, say a=u, then we check N(u) \cap N(b) \cap N(c).
        # N(u) has changed.
        
        # So we only need to check triples containing u or v.
        
        # Triples containing u: {u, x, y} where x,y != u
        # Triples containing v: {v, x, y} where x,y != v
        
        # Let's just check all w in 0..19
        # For each w, the triple {u, v, w} is one case.
        # But also {u, x, y} where v is not in the triple?
        # Wait, if the triple is {u, x, y} and v is not in it,
        # does adding (u,v) affect N(u) \cap N(x) \cap N(y)?
        # Yes, because N(u) changed.
        
        # So we must check all triples containing u, and all triples containing v.
        
        # Triples containing u:
        for x in range(n):
            if x == u: continue
            for y in range(x + 1, n):
                if y == u: continue
                # Triple {u, x, y}
                # Common neighbors: N(u) \cap N(x) \cap N(y)
                # N(u) now includes v.
                # So we check (N(u) | (1<<v)) \cap N(x) \cap N(y)
                
                # But wait, if v is in {x,y}, then we are double counting or need care.
                # If x == v, then triple is {u, v, y}.
                # N(u) \cap N(v) \cap N(y).
                
                # Let's just compute the intersection with the updated masks.
                new_u_mask = adj[u] | (1 << v)
                new_v_mask = adj[v] | (1 << u)
                
                # We need to check all triples {a,b,c} where a,b,c are distinct.
                # It is faster to just iterate all triples involving u or v.
                
                # Triples with u:
                # {u, i, j} for i < j, i,j != u
                # Triples with v:
                # {v, i, j} for i < j, i,j != v
                
                # To avoid duplicates and complexity, we can just check:
                # For all i, j in 0..19, i != j, i != u, j != u:
                #   Check if |N(u) \cap N(i) \cap N(j)| > 2 using NEW N(u)
                # For all i, j in 0..19, i != j, i != v, j != v:
                #   Check if |N(v) \cap N(i) \cap N(j)| > 2 using NEW N(v)
                
                # This is O(n^2) per check. 400 checks.
                # We have ~400 missing edges. 160,000 checks.
                # Each check is a bit operation. Very fast.
                
                pass # We will implement this in the loop below
        
        # Optimized check:
        new_u_mask = adj[u] | (1 << v)
        new_v_mask = adj[v] | (1 << u)
        
        # Check triples containing u
        for i in range(n):
            if i == u: continue
            mask_i = adj[i]
            # N(u) \cap N(i) \cap N(j)
            # We need to check for all j != u, i
            # Actually, we can precompute N(u) \cap N(i)
            common_ui = new_u_mask & mask_i
            # Now check if any j has 3 common neighbors with u and i?
            # No, we need |N(u) \cap N(i) \cap N(j)| <= 2.
            # This is equivalent to: for all j, popcount(common_ui & adj[j]) <= 2.
            # We can iterate j.
            for j in range(n):
                if j == u or j == i: continue
                if (common_ui & adj[j]).bit_count() > 2:
                    return False
                    
        # Check triples containing v
        for i in range(n):
            if i == v: continue
            mask_i = adj[i]
            common_vi = new_v_mask & mask_i
            for j in range(n):
                if j == v or j == i: continue
                if (common_vi & adj[j]).bit_count() > 2:
                    return False
                    
        return True

    def edge_count():
        return sum(m.bit_count() for m in adj) // 2

    def mask_to_edges():
        edges = []
        for i in range(n):
            m = adj[i]
            for j in range(i + 1, n):
                if m & (1 << j):
                    edges.append([i, j])
        return edges

    # Local Search
    # Try to add all possible edges
    improved = True
    while improved:
        improved = False
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    if can_add(u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        improved = True
                        if time.time() > deadline:
                            return mask_to_edges()
        
        # If no improvement, try swaps?
        # Given the Clebsch base, simple addition should suffice for the new vertices.
        # But we might be stuck if the initial Clebsch is not optimal for the extension.
        # However, Clebsch is very symmetric.
        
        # Let's add a random restart for the new vertices if stuck?
        # No, just do a few rounds of random edge removal and re-addition.
        
        if not improved:
            # Try to swap an edge involving a new vertex
            # Pick a random new vertex u, remove a random edge (u,v), add a random missing (u,w)
            # This is complex. Let's just rely on the greedy add.
            break

    # Final polish: check if we can add more
    # The loop above already does this.
    
    return mask_to_edges()
