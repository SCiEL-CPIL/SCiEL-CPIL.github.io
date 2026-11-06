import time
import random

def construct():
    # Fallback: 32-vertex solution (Greenwood-Gleason)
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
    
    colors_32 = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors_32.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)

    # Try to find a 50-vertex solution using local search
    n = 50
    deadline = time.time() + 110.0
    
    # Precompute edge indices
    # idx = i * n - i * (i + 1) // 2 + j - i - 1
    def get_idx(i, j):
        if i > j:
            i, j = j, i
        return i * n - i * (i + 1) // 2 + j - i - 1

    # Precompute for each edge (i,j) the list of k such that (i,j,k) is a triangle
    # Actually, for each edge (i,j), the third vertex k can be any other vertex.
    # We need to check colors of (i,k) and (j,k).
    
    def find_50():
        # List of edges (i, j)
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        num_edges = len(edges)
        
        # Map edge index to (i, j)
        # We already have edges list in order.
        
        for restart in range(100):
            if time.time() > deadline:
                return None
            
            # Random initialization
            colors = [random.randint(0, 3) for _ in range(num_edges)]
            
            # Calculate initial cost
            # Cost = number of monochromatic triangles
            # Iterate over all triples (i, j, k) with i < j < k
            cost = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c_ij = colors[get_idx(i, j)]
                    for k in range(j + 1, n):
                        c_ik = colors[get_idx(i, k)]
                        c_jk = colors[get_idx(j, k)]
                        if c_ij == c_ik == c_jk:
                            cost += 1
            
            # Local Search
            # We will try to minimize cost by changing one edge at a time.
            # For each edge e, try all 4 colors, pick the one that minimizes cost.
            
            # To speed up, we can use a heuristic:
            # Only consider edges that are part of a monochromatic triangle.
            # But finding them is also O(n^3).
            
            # Simple Hill Climbing:
            # Repeat until no improvement or time limit
            for step in range(20000):
                if time.time() > deadline:
                    return None
                
                if cost == 0:
                    return colors
                
                # Pick a random edge
                e = random.randint(0, num_edges - 1)
                i, j = edges[e]
                old_c = colors[e]
                
                best_c = old_c
                best_cost = cost
                
                # Try all 4 colors
                for new_c in range(4):
                    if new_c == old_c:
                        continue
                    
                    # Calculate delta cost
                    # Triangles involving edge (i, j)
                    # For each k != i, j:
                    # Old triangle (i, j, k) was mono if old_c == c_ik == c_jk
                    # New triangle (i, j, k) is mono if new_c == c_ik == c_jk
                    
                    delta = 0
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        
                        c_ik = colors[get_idx(i, k)]
                        c_jk = colors[get_idx(j, k)]
                        
                        # Check if (i, j, k) was monochromatic with old_c
                        if old_c == c_ik == c_jk:
                            delta += 1 # Removing a mono triangle
                        
                        # Check if (i, j, k) becomes monochromatic with new_c
                        if new_c == c_ik == c_jk:
                            delta -= 1 # Adding a mono triangle
                    
                    new_cost = cost + delta
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_c = new_c
                
                if best_c != old_c:
                    colors[e] = best_c
                    cost = best_cost
                else:
                    # No improvement for this edge
                    # To escape local minima, we could accept a worse move (Simulated Annealing)
                    # But for now, just pick another random edge.
                    pass
            
            # If we didn't find 0, restart
            # We could also try to perturb the current best solution

        return None

    res = find_50()
    if res:
        return {'n': 50, 'colors': res}
    else:
        return {'n': 32, 'colors': colors_32}
