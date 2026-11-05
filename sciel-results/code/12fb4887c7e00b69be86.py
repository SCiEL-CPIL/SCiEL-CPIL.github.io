import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    # Fallback: 32-vertex Greenwood-Gleason coloring
    def get_seed_coloring():
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
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return {'n': 32, 'colors': colors}

    best_result = get_seed_coloring()

    # Bitmask-based state: adj[k][i] is an integer bitmask of vertices connected to i via color k
    # total_triangles[k] = number of monochromatic triangles in color k
    # When flipping edge (i,j) from color a to color b:
    #   - In color a: remove triangles containing edge (i,j). 
    #     A triangle (i,j,x) in color a exists iff x is in adj_a[i] AND x is in adj_a[j].
    #     Count = popcount(adj_a[i] & adj_a[j])
    #   - In color b: add triangles containing edge (i,j).
    #     Count = popcount(adj_b[i] & adj_b[j])
    #   - Delta = new_count_b - old_count_a (for the two affected colors)
    
    def popcount(x):
        return bin(x).count('1')
    
    # Use Python's int.bit_count() for speed (Python 3.10+)
    def pc(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

    def make_paley_init(n):
        """Initialize using Paley graph structure where possible.
        For n <= 25: use quadratic residues mod n (if n is prime or prime power with -1 non-residue)
        For n > 25: use Paley(25) for first 25, then random for rest, or extend.
        """
        # adj[k][i] bitmask
        adj = [[0] * n for _ in range(4)]
        colors_flat = {}  # (i,j) -> color
        
        # For Paley graph over Z_p, edge (i,j) has color based on (j-i) mod p
        # For p=25: quadratic residues mod 25: {1,4,6,9,11,14,16,19,21,24}
        # We need 3 colors from the Paley graph. 
        # Actually, Paley graph gives a 2-coloring (residue/non-residue). 
        # For 4-coloring, we can use:
        #   color 0: (j-i) is a QR mod 25 and (j-i) mod 5 == 0 or specific subset
        #   Simpler: use the known 25-vertex 4-coloring from Paley-type construction.
        # 
        # Alternative: Use the fact that Z_25 has a 4-coloring where:
        #   color = f((j-i) mod 25) where f maps to {0,1,2,3}
        #   such that no monochromatic triangle exists.
        #
        # A simpler approach: random init is fine, but let's use a structured init
        # that's known to be close to valid.
        #
        # Actually, let's just use random init - the SA with fast scoring will work.
        # But we can bias: for the first 25 vertices, use a structured pattern.
        
        # Quadratic residues mod 25
        qr25 = set([x*x % 25 for x in range(25)])
        # qr25 = {0, 1, 4, 6, 9, 11, 14, 16, 19, 21, 24}
        # Non-zero QRs: {1, 4, 6, 9, 11, 14, 16, 19, 21, 24}
        
        # For a 4-coloring of K_25 with no mono triangle:
        # We can partition the 10 non-zero QRs and 10 non-QRs into 4 sets of ~5 each
        # such that each set is triangle-free in the circulant graph.
        #
        # This is complex. Let's just use random init - the fast SA will handle it.
        
        for i in range(n):
            for j in range(i + 1, n):
                k = random.randint(0, 3)
                adj[k][i] |= (1 << j)
                adj[k][j] |= (1 << i)
                colors_flat[(i, j)] = k
        
        return adj, colors_flat

    def count_total_triangles(adj, n):
        """Count total monochromatic triangles across all 4 colors."""
        total = 0
        for k in range(4):
            for i in range(n):
                # For each vertex i, count pairs (j,l) in adj[k][i] that are connected
                # But this double/triple counts. 
                # Better: for each edge (i,j) in color k, count common neighbors.
                # Total triangles in color k = (1/6) * sum over all edges (i,j) of popcount(adj[k][i] & adj[k][j])
                # But that's O(n^2) per color.
                # For initial count, we can do it once.
                pass
        # Actually, let's compute it properly once:
        total = 0
        for k in range(4):
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[k][i] & (1 << j):
                        total += pc(adj[k][i] & adj[k][j])
        return total // 6  # Each triangle counted 6 times (3 vertices * 2 orientations)
    
    # Wait, that's not right either. Each triangle (i,j,l) is counted once for each of its 3 edges?
    # No: for edge (i,j), we count l if l is neighbor of both i and j. 
    # Triangle (i,j,l) is counted for edge (i,j) [l is common neighbor], edge (i,l) [j is common neighbor], edge (j,l) [i is common neighbor].
    # So each triangle is counted 3 times.
    # So total_triangles = total_sum // 3
    
    def count_total_triangles_v2(adj, n):
        total = 0
        for k in range(4):
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[k][i] & (1 << j):
                        total += pc(adj[k][i] & adj[k][j])
        return total // 3

    def run_sa(n, time_limit):
        if time.time() > time_limit:
            return None
            
        # Initialize
        adj, colors_flat = make_paley_init(n)
        
        current_total = count_total_triangles_v2(adj, n)
        best_total = current_total
        best_adj = [list(row) for row in adj]
        
        if current_total == 0:
            # Extract colors
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    colors.append(colors_flat[(i, j)])
            return {'n': n, 'colors': colors}
        
        T = 2.0
        T_min = 0.05
        cooling_rate = 0.999
        steps_since_improve = 0
        max_steps_no_improve = 20000
        
        # Precompute edge list for faster random selection
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        num_edges = len(edges)
        
        while time.time() < time_limit:
            # Pick random edge
            idx = random.randint(0, num_edges - 1)
            i, j = edges[idx]
            
            # Find old color
            old_k = -1
            for k in range(4):
                if adj[k][i] & (1 << j):
                    old_k = k
                    break
            if old_k == -1:
                continue
            
            # Pick new color (different from old)
            new_k = random.randint(0, 3)
            if new_k == old_k:
                continue
            
            # Compute delta in triangle count
            # Remove edge (i,j) from color old_k:
            #   Triangles lost in old_k: popcount(adj[old_k][i] & adj[old_k][j])
            #   But this includes triangles that don't involve the edge? No, 
            #   adj[old_k][i] & adj[old_k][j] gives common neighbors x such that (i,x) and (j,x) are both color old_k.
            #   Each such x forms triangle (i,j,x) in color old_k.
            #   BUT: when we remove edge (i,j), we also need to consider that x must NOT be i or j.
            #   adj[old_k][i] doesn't include i itself, and j is in adj[old_k][i] (since edge exists).
            #   So adj[old_k][i] & adj[old_k][j] includes j? No: adj[old_k][j] has i in it, not j.
            #   Actually, adj[k][i] is the set of neighbors of i in color k. It does NOT include i.
            #   So adj[old_k][i] & adj[old_k][j] gives vertices x (x != i, x != j) such that both (i,x) and (j,x) are color old_k.
            #   Each such x forms a monochromatic triangle (i,j,x) in color old_k.
            #   So triangles_lost = pc(adj[old_k][i] & adj[old_k][j])
            
            # Add edge (i,j) to color new_k:
            #   Triangles gained in new_k: popcount(adj[new_k][i] & adj[new_k][j])
            #   BUT: after adding, adj[new_k][i] will include j and adj[new_k][j] will include i.
            #   The common neighbors x (x != i, x != j) such that (i,x) and (j,x) are both color new_k
            #   form triangles (i,j,x) in color new_k.
            #   So triangles_gained = pc(adj[new_k][i] & adj[new_k][j])
            #   Note: this is computed BEFORE adding the edge, so adj[new_k][i] doesn't include j yet.
            #   That's correct - we want x != i, x != j.
            
            triangles_lost = pc(adj[old_k][i] & adj[old_k][j])
            triangles_gained = pc(adj[new_k][i] & adj[new_k][j])
            
            delta = triangles_gained - triangles_lost
            
            # Apply move
            adj[old_k][i] ^= (1 << j)
            adj[old_k][j] ^= (1 << i)
            adj[new_k][i] |= (1 << j)
            adj[new_k][j] |= (1 << i)
            colors_flat[(i, j)] = new_k
            
            new_total = current_total + delta
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_total = new_total
                if current_total < best_total:
                    best_total = current_total
                    best_adj = [list(row) for row in adj]
                    steps_since_improve = 0
                    if best_total == 0:
                        colors = []
                        for ii in range(n):
                            for jj in range(ii + 1, n):
                                colors.append(colors_flat[(ii, jj)])
                        return {'n': n, 'colors': colors}
                else:
                    steps_since_improve += 1
            else:
                # Revert
                adj[old_k][i] |= (1 << j)
                adj[old_k][j] |= (1 << i)
                adj[new_k][i] ^= (1 << j)
                adj[new_k][j] ^= (1 << i)
                colors_flat[(i, j)] = old_k
                steps_since_improve += 1
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
            
            if steps_since_improve > max_steps_no_improve:
                # Restart from best
                adj = [list(row) for row in best_adj]
                current_total = best_total
                T = 2.0
                steps_since_improve = 0
                # Rebuild colors_flat from best_adj
                for i in range(n):
                    for j in range(i + 1, n):
                        for k in range(4):
                            if adj[k][i] & (1 << j):
                                colors_flat[(i, j)] = k
                                break
        
        # Check best
        if best_total == 0:
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(4):
                        if best_adj[k][i] & (1 << j):
                            colors.append(k)
                            break
                    else:
                        colors.append(0)  # Should not happen
            return {'n': n, 'colors': colors}
        
        return None

    # Try n values from high to low
    targets = [50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33]
    
    for n in targets:
        if time.time() > deadline:
            break
        
        time_left = deadline - time.time()
        if time_left < 3:
            break
        
        # Allocate time: more for larger n since it's harder but more valuable
        alloc = min(8.0, time_left - 1.0)
        
        result = run_sa(n, time.time() + alloc)
        
        if result:
            return result

    return best_result
