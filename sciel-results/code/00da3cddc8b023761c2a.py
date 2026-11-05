import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit

    def get_fallback():
        # Greenwood-Gleason colouring of K16 by cubic-residue classes of GF(16)
        # doubled: two copies with colour 3 on all edges between them
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

    def build_seed_coloring(n):
        """Build a seed coloring using random assignment or GF(16) base."""
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                # Random start is often sufficient for local search to find a solution
                # if we have enough time, but a structured start might be better.
                # Let's try random first as it's simple and unbiased.
                colors.append(random.randint(0, 3))
        return colors

    def count_mono_triangles(adj, n):
        """Count total monochromatic triangles using bitsets."""
        total = 0
        for c in range(4):
            for i in range(n):
                # Only count triangles where i is the smallest vertex to avoid double counting
                # Actually, summing over all i, j, k and dividing by 3 is easier with bitmasks
                # But we can just sum (adj[c][i] & adj[c][j]).bit_count() for all i<j
                # and divide by 3 at the end? No, that counts each triangle 3 times.
                # Let's just sum for i < j, and check if k > j is in the intersection.
                # Simpler: sum over all i, (adj[c][i] & (adj[c][i] >> (i+1))).bit_count() ? No.
                
                # Efficient way:
                # For each color c, the number of triangles is (1/3) * sum_{u,v} (adj[c][u] & adj[c][v]).bit_count()
                # Wait, adj[c][u] contains v if edge (u,v) is color c.
                # If we sum over all u < v, (adj[c][u] & adj[c][v]).bit_count() counts the number of common neighbors w.
                # Each triangle (u,v,w) is counted 3 times in the sum over all pairs (u,v).
                # So total triangles = sum_{c} sum_{u < v} (adj[c][u] & adj[c][v]).bit_count() // 3
                pass
        
        # Let's implement the efficient count properly
        total = 0
        for c in range(4):
            for i in range(n):
                # We need to count triangles (i, j, k) with i < j < k
                # This is equivalent to summing over i < j, the number of k > j such that (i,k) and (j,k) are color c.
                # adj[c][i] has bits for all neighbors.
                # We want bits j > i and k > j.
                # This is getting complicated to do in one shot.
                
                # Alternative: Just iterate i < j and count common neighbors k > j.
                # But we want speed.
                # Let's stick to the delta calculation for local search, and only do a full count occasionally or at the start.
                # For the initial count, we can just do a simple O(n^3) or O(n^2 * 4) check?
                # O(n^2) with bitmasks is fast enough for n=50.
                pass
        
        # Let's rewrite count_mono_triangles to be O(n^2) using bitmasks
        total = 0
        for c in range(4):
            for i in range(n):
                # Mask out bits <= i to only consider j > i
                mask_j = adj[c][i] & ~((1 << (i + 1)) - 1)
                # For each j in mask_j, we want to count k > j such that k is in adj[c][j]
                # This is hard to vectorize easily without numpy or iterating j.
                # Since n is small (<=50), iterating j is fine.
                j = i + 1
                while j < n:
                    if (adj[c][i] >> j) & 1:
                        # Count k > j in adj[c][j]
                        # adj[c][j] has bits for all neighbors. We want bits > j.
                        mask_k = adj[c][j] & ~((1 << (j + 1)) - 1)
                        total += mask_k.bit_count()
                    j += 1
        return total

    def local_search(n, colors, time_limit):
        """Local search with simulated annealing."""
        # Build adjacency bitsets
        adj = [[0] * n for _ in range(4)]
        edges = []
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        num_edges = len(edges)
        
        # Initial cost
        current_cost = count_mono_triangles(adj, n)
        best_cost = current_cost
        best_colors = colors[:]
        
        temp = 10.0
        cooling = 0.9999
        
        while time.time() < time_limit:
            # Pick random edge
            idx = random.randint(0, num_edges - 1)
            u, v = edges[idx]
            c_old = colors[idx]
            
            # Try changing to a different color
            c_new = random.choice([c for c in range(4) if c != c_old])
            
            # Calculate delta
            # Triangles removed: (adj[c_old][u] & adj[c_old][v]).bit_count()
            # Triangles added: (adj[c_new][u] & adj[c_new][v]).bit_count()
            # Note: This counts all common neighbors. 
            # Does it overcount?
            # A triangle (u,v,w) is removed if (u,v), (u,w), (v,w) are all c_old.
            # The term (adj[c_old][u] & adj[c_old][v]).bit_count() counts all w such that (u,w) and (v,w) are c_old.
            # Since (u,v) is c_old, these are exactly the monochromatic triangles containing (u,v).
            # So delta = added - removed.
            
            removed = (adj[c_old][u] & adj[c_old][v]).bit_count()
            added = (adj[c_new][u] & adj[c_new][v]).bit_count()
            delta = added - removed
            
            # Acceptance
            if delta < 0 or random.random() < math.exp(-delta / temp):
                # Update state
                colors[idx] = c_new
                adj[c_old][u] ^= (1 << v)
                adj[c_old][v] ^= (1 << u)
                adj[c_new][u] ^= (1 << v)
                adj[c_new][v] ^= (1 << u)
                
                current_cost += delta
                temp *= cooling
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]
                    if best_cost == 0:
                        return best_colors
            else:
                temp *= cooling
                
            # If we hit 0, return immediately
            if current_cost == 0:
                return colors[:]
        
        return best_colors if best_cost == 0 else None

    def solve_n(n, time_limit):
        """Try to solve for n vertices within time_limit."""
        # Try a few restarts
        for restart in range(10):
            if time.time() > start_time + time_limit:
                break
            colors = build_seed_coloring(n)
            result = local_search(n, colors, start_time + time_limit)
            if result is not None:
                return {'n': n, 'colors': result}
        return None

    # Try to find solutions for n=50 down to 33
    # Allocate time wisely
    time_per_n = 10.0
    for n in range(50, 32, -1):
        if time.time() > deadline - 5:
            break
        remaining = deadline - time.time() - 2
        if remaining < 1:
            break
        t_limit = min(time_per_n, remaining)
        res = solve_n(n, t_limit)
        if res:
            return res
            
    return get_fallback()
