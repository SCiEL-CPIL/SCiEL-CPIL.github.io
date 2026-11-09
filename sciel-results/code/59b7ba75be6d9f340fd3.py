import time
import random
import numpy as np

def _popcount(x):
    return bin(x).count('1')

def _build_50():
    N = 50
    deadline = time.time() + 110.0
    
    # Bitmask representation of neighbors for each color
    # adj[c][u] is a bitmask of vertices v such that (u,v) has color c
    
    def try_solve(seed):
        rng = random.Random(seed)
        
        # Initialize random colors
        # adj[c][u]
        adj = [[0] * N for _ in range(4)]
        edge_colors = {} # (u,v) -> c
        
        # Assign random colors
        for i in range(N):
            for j in range(i + 1, N):
                c = rng.randint(0, 3)
                edge_colors[(i, j)] = c
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        
        # Count initial triangles
        # We don't need the total count, just a list of violating edges or triangles.
        # Let's maintain a set of "bad" edges (u,v) where (u,v) is part of a mono-triangle.
        # Or simpler: just iterate and find triangles.
        
        # To find a triangle, we can pick a random edge (u,v) and check adj[c][u] & adj[c][v].
        # If non-zero, there is a triangle.
        
        bad_edges = set()
        
        def update_bad_edges(u, v, old_c, new_c):
            # This function is complex to maintain incrementally perfectly.
            # Given N=50, we can just re-scan or do local checks.
            # Local check:
            # 1. Triangles involving (u,v) in old_c are removed.
            # 2. Triangles involving (u,v) in new_c are added.
            
            # Check if (u,v) was in a triangle in old_c
            # neighbors = adj[old_c][u] & adj[old_c][v]
            # If neighbors != 0, then (u,v) was bad.
            # Now we change to new_c.
            
            # We need to know if (u,v) is STILL bad in new_c?
            # Or if it becomes bad?
            
            # Actually, let's just maintain the global list of violations by scanning?
            # No, scanning is O(N^3).
            
            # Let's use the "pick random edge, check if bad" strategy.
            pass

        # Simpler Strategy:
        # 1. Find a monochromatic triangle.
        # 2. Change one edge.
        # 3. Repeat.
        
        # To find a triangle quickly:
        # Pick random u, v.
        # c = edge_colors[(u,v)]
        # common = adj[c][u] & adj[c][v]
        # if common:
        #    w = common.bit_length() - 1 # or any bit
        #    return (u,v,w,c)
        
        # If not, try another edge.
        
        for step in range(100000):
            if time.time() > deadline:
                return None
            
            # Find a violation
            found = False
            for _ in range(100): # Try 100 random edges
                u = rng.randint(0, N - 1)
                v = rng.randint(0, N - 1)
                if u == v: continue
                if u > v: u, v = v, u
                
                c = edge_colors[(u, v)]
                common = adj[c][u] & adj[c][v]
                if common:
                    # Found triangle (u, v, w)
                    w = (common & -common).bit_length() - 1
                    # Try to fix by changing one of the 3 edges
                    edges = [(u, v), (v, w), (u, w)]
                    rng.shuffle(edges)
                    
                    fixed = False
                    for (a, b) in edges:
                        if a > b: a, b = b, a
                        old_c = edge_colors[(a, b)]
                        
                        # Try changing to new_c
                        for new_c in range(4):
                            if new_c == old_c: continue
                            
                            # Evaluate change
                            # Removed: triangles in old_c involving (a,b)
                            # Note: (a,b) is in the triangle we found, so it is in old_c.
                            removed = _popcount(adj[old_c][a] & adj[old_c][b])
                            
                            # Added: triangles in new_c involving (a,b)
                            added = _popcount(adj[new_c][a] & adj[new_c][b])
                            
                            # Accept if added < removed, or (added == removed and random)
                            if added < removed or (added == removed and rng.random() < 0.5):
                                # Execute change
                                # Remove from old_c
                                adj[old_c][a] &= ~(1 << b)
                                adj[old_c][b] &= ~(1 << a)
                                # Add to new_c
                                adj[new_c][a] |= (1 << b)
                                adj[new_c][b] |= (1 << a)
                                edge_colors[(a, b)] = new_c
                                
                                fixed = True
                                break
                        if fixed:
                            break
                    found = True
                    break
            
            if not found:
                # No violation found in 100 random checks
                # It might be solved!
                # Verify
                if _verify_fast(adj, N):
                    # Extract colors
                    colors = []
                    for i in range(N):
                        for j in range(i + 1, N):
                            colors.append(edge_colors[(i, j)])
                    return colors
                # Else, restart or kick
                # Kick: change a random edge
                a = rng.randint(0, N - 1)
                b = rng.randint(0, N - 1)
                if a == b: continue
                if a > b: a, b = b, a
                old_c = edge_colors[(a, b)]
                new_c = rng.randint(0, 3)
                if new_c == old_c: continue
                adj[old_c][a] &= ~(1 << b)
                adj[old_c][b] &= ~(1 << a)
                adj[new_c][a] |= (1 << b)
                adj[new_c][b] |= (1 << a)
                edge_colors[(a, b)] = new_c

        return None

    for seed in range(1000):
        if time.time() > deadline:
            break
        res = try_solve(seed)
        if res:
            return 50, res
            
    return None

def _verify_fast(adj, N):
    # Check if any color has a triangle
    for c in range(4):
        for u in range(N):
            # For each neighbor v of u in color c
            v_mask = adj[c][u]
            while v_mask:
                v = (v_mask & -v_mask).bit_length() - 1
                v_mask &= v_mask - 1
                # Check if any common neighbor w of u and v in color c
                if adj[c][u] & adj[c][v]:
                    # Wait, adj[c][u] contains v. adj[c][v] contains u.
                    # We need a w != u, v.
                    # common = adj[c][u] & adj[c][v]
                    # If common has any bit other than u and v?
                    # Actually, if there is a triangle, common will have at least one bit.
                    # Since u and v are connected, u is in N(v) and v is in N(u).
                    # So common will always contain u and v?
                    # No, adj[c][u] is the set of neighbors.
                    # If (u,v) is an edge, v is in adj[c][u] and u is in adj[c][v].
                    # So adj[c][u] & adj[c][v] will contain u and v?
                    # Let's check:
                    # adj[c][u] has bit v.
                    # adj[c][v] has bit u.
                    # Intersection has bit u? No, adj[c][u] has bit v, not u (unless self loop).
                    # So intersection contains bits w such that (u,w) and (v,w) are edges.
                    # If there is a triangle (u,v,w), then w is in the intersection.
                    # Also, is u in the intersection?
                    # u is in adj[c][v]. Is u in adj[c][u]? No.
                    # So u is NOT in the intersection.
                    # Is v in the intersection?
                    # v is in adj[c][u]. Is v in adj[c][v]? No.
                    # So v is NOT in the intersection.
                    # So if intersection is non-zero, it contains a vertex w != u,v.
                    # Thus, a triangle exists.
                    return False
    return True

def _seed32():
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    c = []
    for i in range(32):
        for j in range(i + 1, 32):
            c.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return {'n': 32, 'colors': c}

def construct():
    deadline = time.time() + 140.0
    r = _build_50()
    if r is not None and time.time() < deadline:
        n, colors = r
        return {'n': n, 'colors': colors}
    return _seed32()
