"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill
    
    n = 29
    m = n * (n - 1) // 2  # 406 edges
    
    # Precompute edge index and endpoints
    edge_idx = [[-1] * n for _ in range(n)]
    edge_u = []
    edge_v = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            edge_u.append(i)
            edge_v.append(j)
            idx += 1
    
    # For each vertex v, list of neighbors and their edge indices
    # adj[v] = list of (neighbor, edge_index)
    adj = [[] for _ in range(n)]
    for e in range(m):
        u = edge_u[e]
        v = edge_v[e]
        adj[u].append((v, e))
        adj[v].append((u, e))
    
    # Bitset representation: for each color c and vertex v, 
    # color_c[v] is a bitmask of vertices connected to v by color c
    # Use Python integers as bitmasks
    
    def colors_to_bitmasks(colors):
        """Convert edge color list to per-vertex bitmasks for each color."""
        bm0 = [0] * n  # bitmask of neighbors via color 0
        bm1 = [0] * n  # bitmask of neighbors via color 1
        for e in range(m):
            u = edge_u[e]
            v = edge_v[e]
            if colors[e] == 0:
                bm0[u] |= (1 << v)
                bm0[v] |= (1 << u)
            else:
                bm1[u] |= (1 << v)
                bm1[v] |= (1 << u)
        return bm0, bm1
    
    def has_k222_from_bitmasks(bm0, bm1):
        """Check K_{2,2,2} using bitmasks. 
        For color c, K_{2,2,2} exists iff there are vertices a,b,c such that
        |N_c(a) ∩ N_c(b) ∩ N_c(c)| >= 3 where N_c(v) = neighbors of v in color c.
        Actually: K_{2,2,2} in color c means ∃ {a,b,c} and {d,e,f} with all edges 
        between them in color c. This means d,e,f ∈ N_c(a) ∩ N_c(b) ∩ N_c(c).
        So we need |N_c(a) ∩ N_c(b) ∩ N_c(c)| >= 3 for some triple {a,b,c}.
        """
        for a in range(n):
            for b in range(a + 1, n):
                common_ab = bm0[a] & bm0[b]
                # Remove a and b themselves (they shouldn't be in their own neighborhood)
                common_ab &= ~(1 << a) & ~(1 << b)
                if common_ab.bit_count() < 3:
                    continue
                for c in range(b + 1, n):
                    if c == a or c == b:
                        continue
                    common_abc = common_ab & bm0[c]
                    common_abc &= ~(1 << c)
                    if common_abc.bit_count() >= 3:
                        return True
        # Same for color 1
        for a in range(n):
            for b in range(a + 1, n):
                common_ab = bm1[a] & bm1[b]
                common_ab &= ~(1 << a) & ~(1 << b)
                if common_ab.bit_count() < 3:
                    continue
                for c in range(b + 1, n):
                    if c == a or c == b:
                        continue
                    common_abc = common_ab & bm1[c]
                    common_abc &= ~(1 << c)
                    if common_abc.bit_count() >= 3:
                        return True
        return False
    
    def count_violations_bitmasks(bm0, bm1):
        """Count number of triples {a,b,c} with |N(a)∩N(b)∩N(c)| >= 3 for either color."""
        viol = 0
        for a in range(n):
            for b in range(a + 1, n):
                common_ab_0 = bm0[a] & bm0[b] & ~(1 << a) & ~(1 << b)
                common_ab_1 = bm1[a] & bm1[b] & ~(1 << a) & ~(1 << b)
                if common_ab_0.bit_count() < 3 and common_ab_1.bit_count() < 3:
                    continue
                for c in range(b + 1, n):
                    if common_ab_0.bit_count() >= 3:
                        common_abc_0 = common_ab_0 & bm0[c] & ~(1 << c)
                        if common_abc_0.bit_count() >= 3:
                            viol += 1
                    if common_ab_1.bit_count() >= 3:
                        common_abc_1 = common_ab_1 & bm1[c] & ~(1 << c)
                        if common_abc_1.bit_count() >= 3:
                            viol += 1
        return viol
    
    # Local search with bitmasks
    rng = random.Random(12345)
    
    best_colors = None
    best_viol = float('inf')
    
    max_attempts = 100
    for attempt in range(max_attempts):
        if time.time() > deadline - 20:
            break
        
        # Initialize
        colors = [rng.randint(0, 1) for _ in range(m)]
        bm0, bm1 = colors_to_bitmasks(colors)
        viol = count_violations_bitmasks(bm0, bm1)
        
        if viol == 0:
            return {'n': n, 'colors': colors}
        
        if viol < best_viol:
            best_viol = viol
            best_colors = colors[:]
        
        # Local search: flip edges to reduce violations
        max_local_iters = 30000
        for it in range(max_local_iters):
            if time.time() > deadline - 10:
                break
            
            if viol == 0:
                break
            
            if viol < best_viol:
                best_viol = viol
                best_colors = colors[:]
            
            # Try flipping a random edge
            found = False
            for _ in range(30):
                e = rng.randint(0, m - 1)
                u = edge_u[e]
                v = edge_v[e]
                old_c = colors[e]
                new_c = 1 - old_c
                
                # Update bitmasks
                if old_c == 0:
                    bm0[u] &= ~(1 << v)
                    bm0[v] &= ~(1 << u)
                    bm1[u] |= (1 << v)
                    bm1[v] |= (1 << u)
                else:
                    bm1[u] &= ~(1 << v)
                    bm1[v] &= ~(1 << u)
                    bm0[u] |= (1 << v)
                    bm0[v] |= (1 << u)
                
                new_viol = count_violations_bitmasks(bm0, bm1)
                if new_viol < viol:
                    colors[e] = new_c
                    viol = new_viol
                    found = True
                    break
                else:
                    # Undo
                    if old_c == 0:
                        bm0[u] |= (1 << v)
                        bm0[v] |= (1 << u)
                        bm1[u] &= ~(1 << v)
                        bm1[v] &= ~(1 << u)
                    else:
                        bm1[u] |= (1 << v)
                        bm1[v] |= (1 << u)
                        bm0[u] &= ~(1 << v)
                        bm0[v] &= ~(1 << u)
            
            if not found:
                # Escape: random flip
                e = rng.randint(0, m - 1)
                u = edge_u[e]
                v = edge_v[e]
                old_c = colors[e]
                new_c = 1 - old_c
                if old_c == 0:
                    bm0[u] &= ~(1 << v)
                    bm0[v] &= ~(1 << u)
                    bm1[u] |= (1 << v)
                    bm1[v] |= (1 << u)
                else:
                    bm1[u] &= ~(1 << v)
                    bm1[v] &= ~(1 << u)
                    bm0[u] |= (1 << v)
                    bm0[v] |= (1 << u)
                colors[e] = new_c
                viol = count_violations_bitmasks(bm0, bm1)
    
    if best_colors is not None:
        return {'n': n, 'colors': best_colors}
    else:
        colors = [rng.randint(0, 1) for _ in range(m)]
        return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
