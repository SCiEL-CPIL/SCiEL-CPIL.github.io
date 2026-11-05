import time
import random
import sys

def construct():
    deadline = time.time() + 140.0  # Safety margin for 150s limit
    best_n = 32
    best_colors = None
    
    # Seed solution for n=32
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
    seed_colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            seed_colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    best_colors = seed_colors

    def count_triangles(n, colors):
        # colors is list of length n*(n-1)/2
        # Map edge (i,j) to index
        # This is slow, used only for final verification or init
        pass

    def solve_for_n(n):
        if time.time() > deadline:
            return None
        
        # Initialize random colors
        m = n * (n - 1) // 2
        colors = [random.randint(0, 3) for _ in range(m)]
        
        # Bitmasks for each vertex and color
        # masks[u][c] is a bitmask of vertices v such that edge (u,v) has color c
        masks = [[0] * 4 for _ in range(n)]
        
        def get_idx(u, v):
            if u > v:
                u, v = v, u
            return u * n - u * (u + 1) // 2 + (v - u - 1)
            # Formula: sum_{k=0}^{u-1} (n-1-k) + (v-u-1)
            # Simpler: 
            # idx = 0
            # for i in range(u):
            #   idx += n - 1 - i
            # idx += v - u - 1
            
        # Precompute indices for speed? Or just use a 2D array of indices?
        # idx[u][v]
        idx = [[0]*n for _ in range(n)]
        k = 0
        for i in range(n):
            for j in range(i+1, n):
                idx[i][j] = k
                idx[j][i] = k
                k += 1

        # Initialize masks
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx[i][j]]
                masks[i][c] |= (1 << j)
                masks[j][c] |= (1 << i)

        # Calculate initial triangle count
        tri_count = 0
        for u in range(n):
            for c in range(4):
                # Count pairs (v, w) in masks[u][c] such that (v,w) is also color c
                # This is sum over v in masks[u][c] of popcount(masks[v][c] & masks[u][c])
                # But this counts each triangle 3 times (once for each vertex)
                # And we want total triangles.
                # Actually, a triangle (u,v,w) is counted:
                # At u: v and w are in masks[u][c]. Pair (v,w) is in intersection.
                # So for a fixed u, number of triangles with vertex u is 0.5 * sum_{v in masks[u][c]} popcount(masks[v][c] & masks[u][c])
                # Let's just sum over all edges?
                # Easier: Sum over all u, c of popcount(masks[u][c] & masks[v][c]) for each edge? No.
                
                # Let's use the property:
                # Total triangles = (1/3) * Sum_{u} Sum_{c} (1/2) * Sum_{v in masks[u][c]} popcount(masks[v][c] & masks[u][c])
                # This is complex to initialize quickly.
                
                # Alternative: Just count by iterating all triples? O(N^3) is too slow for N=50? 125k triples. 4 colors.
                # 125k * 4 = 500k ops. This is fast enough for init.
                pass

        # Let's do O(N^3) init for simplicity, it's only ~100k ops
        for u in range(n):
            for v in range(u+1, n):
                for w in range(v+1, n):
                    c1 = colors[idx[u][v]]
                    if c1 == colors[idx[v][w]] and c1 == colors[idx[w][u]]:
                        tri_count += 1

        # Simulated Annealing
        T = 1.0
        cooling = 0.999
        steps = 0
        max_steps = 200000 # Adjust based on speed
        
        while T > 0.01 and time.time() < deadline - 5:
            # Pick random edge
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            i_idx = idx[u][v]
            old_c = colors[i_idx]
            new_c = random.randint(0, 3)
            if old_c == new_c:
                continue
            
            # Calculate delta in triangle count
            # Removing edge (u,v) from old_c:
            # Triangles lost: number of w such that (u,w) is old_c and (v,w) is old_c
            # Note: masks[u][old_c] contains v. masks[v][old_c] contains u.
            # Intersection contains u and v?
            # masks[u][old_c] has bit v set.
            # masks[v][old_c] has bit u set.
            # Intersection has bits u and v set?
            # No, masks[u] has neighbors of u. It does not have u itself.
            # So masks[u][old_c] & masks[v][old_c] contains w such that w is neighbor of both u and v with color old_c.
            # Does it contain u or v?
            # u is not in masks[u]. v is not in masks[v].
            # So the intersection does NOT contain u or v.
            # So the count is exactly the number of triangles (u,v,w) with color old_c.
            
            lost = (masks[u][old_c] & masks[v][old_c]).bit_count()
            
            # Adding edge (u,v) to new_c:
            gained = (masks[u][new_c] & masks[v][new_c]).bit_count()
            
            delta = gained - lost
            
            if delta < 0 or random.random() < pow(2.718, -delta / T):
                # Accept
                colors[i_idx] = new_c
                tri_count += delta
                
                # Update masks
                masks[u][old_c] &= ~(1 << v)
                masks[v][old_c] &= ~(1 << u)
                masks[u][new_c] |= (1 << v)
                masks[v][new_c] |= (1 << u)
                
                if tri_count == 0:
                    return colors
            
            T *= cooling
            steps += 1
            
            if steps % 10000 == 0:
                # Optional: restart if stuck?
                pass

        return None

    # Try to find solution for decreasing N
    for n in range(50, 32, -1):
        if time.time() > deadline - 10:
            break
        res = solve_for_n(n)
        if res is not None:
            # Verify
            # Quick check
            valid = True
            idx = [[0]*n for _ in range(n)]
            k = 0
            for i in range(n):
                for j in range(i+1, n):
                    idx[i][j] = k
                    idx[j][i] = k
                    k += 1
            for u in range(n):
                for v in range(u+1, n):
                    for w in range(v+1, n):
                        if res[idx[u][v]] == res[idx[v][w]] == res[idx[w][u]]:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            if valid:
                return {'n': n, 'colors': res}
    
    return {'n': best_n, 'colors': best_colors}
