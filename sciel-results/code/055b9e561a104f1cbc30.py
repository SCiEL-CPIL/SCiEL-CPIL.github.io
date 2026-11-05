import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 118  # safety margin

    def time_left():
        return deadline - time.time()

    def build_structure(n):
        """Build edge index and endpoint structures for n vertices."""
        num_edges = n * (n - 1) // 2
        edge_index = np.zeros((n, n), dtype=np.int32)
        endpoints = []
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_index[i, j] = idx
                edge_index[j, i] = idx
                endpoints.append((i, j))
                idx += 1
        return num_edges, edge_index, endpoints

    def build_vertex_edges(n, edge_index):
        """For each vertex, list of (other_vertex, edge_idx)."""
        vedges = []
        for v in range(n):
            lst = []
            for x in range(n):
                if x != v:
                    lst.append((x, edge_index[v, x]))
            vedges.append(lst)
        return vedges

    def compute_cn(n, colors, edge_index):
        """Compute CN[c][u,v] for all colors c and pairs (u,v)."""
        CN = [np.zeros((n, n), dtype=np.int16) for _ in range(5)]
        # For each color c, for each vertex w, find neighbors in color c
        for c in range(5):
            for w in range(n):
                nbrs = []
                for x in range(n):
                    if x != w:
                        if colors[edge_index[w, x]] == c:
                            nbrs.append(x)
                for i2 in range(len(nbrs)):
                    u = nbrs[i2]
                    for j2 in range(i2 + 1, len(nbrs)):
                        v = nbrs[j2]
                        CN[c][u, v] += 1
                        CN[c][v, u] += 1
        return CN

    def compute_cost(CN, n):
        """Total number of C4s across all colors."""
        cost = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    cn_val = int(CN[c][u, v])
                    if cn_val >= 2:
                        cost += cn_val * (cn_val - 1) // 2
        return cost

    def verify(n, colors):
        """Check if coloring is valid (no monochromatic C4)."""
        CN = compute_cn(n, colors, *build_structure(n)[1:2])
        # Actually need edge_index
        num_edges, edge_index, _ = build_structure(n)
        CN = compute_cn(n, colors, edge_index)
        cost = compute_cost(CN, n)
        return cost == 0

    def try_find_n(n, max_restarts=50, max_iter=30000, seed_colors=None):
        """Try to find a valid 5-coloring of K_n using vertex-level simulated annealing."""
        if time_left() < 3:
            return None
        
        num_edges, edge_index, endpoints = build_structure(n)
        vedges = build_vertex_edges(n, edge_index)
        
        # Precompute for each vertex v and each other vertex x, the edge index
        # This is already in vedges
        
        best_result = None
        
        for restart in range(max_restarts):
            if time_left() < 5:
                break
            
            if seed_colors is not None and restart == 0:
                colors = np.array(seed_colors, dtype=np.int8)
            else:
                colors = np.random.randint(0, 5, size=num_edges, dtype=np.int8)
            
            CN = compute_cn(n, colors, edge_index)
            current_cost = compute_cost(CN, n)
            
            if current_cost == 0:
                return colors.tolist()
            
            # Simulated annealing with vertex-level moves
            T = 50.0
            cool_rate = 0.995
            min_T = 0.1
            
            for it in range(max_iter):
                if time_left() < 5:
                    break
                if current_cost == 0:
                    return colors.tolist()
                
                # Pick a random vertex
                v = random.randint(0, n - 1)
                
                # For each new color assignment to all edges from v, compute delta
                # Edge from v to x has index edge_index[v, x]
                # The other endpoint of that edge is x
                
                # Current colors of edges from v
                old_colors = np.array([int(colors[edge_index[v, x]]) for x in range(n) if x != v], dtype=np.int8)
                old_cost = current_cost
                
                best_delta = 0
                best_assign = None
                
                # Try a few random assignments rather than all 5^(n-1)
                for trial in range(3):
                    new_colors = np.random.randint(0, 5, size=n - 1, dtype=np.int8)
                    
                    # Compute delta by checking affected C4s
                    # A C4 involving v: v-a-b-c-v where all edges color c
                    # Changing v's edges affects C4s where v is part of the C4
                    
                    # For each color c, the C4s involving v are determined by
                    # pairs of neighbors (a, b) of v in color c that also have
                    # a common neighbor in color c other than v.
                    # Actually, CN[c][a,b] includes v if both (a,v) and (b,v) are color c.
                    # So C4s through v in color c = C(common_c(a,b) excluding v, 2) for pairs (a,b)
                    
                    # Simpler: just recompute the contribution of v to the cost
                    # Cost contribution from v: for each color c, for each pair (a,b) of other vertices,
                    # if (a,v) and (b,v) are color c, then CN[c][a,b] includes v.
                    # The C4 count involving v is: for each color c, sum over pairs (a,b) where
                    # both (a,v) and (b,v) are color c, of (CN[c][a,b] - 1) choose 2 ... 
                    # No, that's not right either.
                    
                    # Let me just compute the full cost difference by updating CN incrementally
                    # For now, use a simpler approach: compute cost before and after
                    # But that's O(n^2) per trial. For n=26, that's 675 ops, times 3 trials = 2025 per iter.
                    # 30000 iters * 2025 = 60M ops. Too slow in Python.
                    
                    # Better: use the incremental update from the edge-level approach
                    # but applied to all edges from v at once.
                    
                    # For each edge (v, x) changing from old_c to new_c:
                    # delta from removing (v,x) from old_c:
                    #   for each y != v, x: if (x,y) is old_c, CN[old_c][v,y] -= 1, CN[old_c][y,v] -= 1
                    #   and the cost change: for each y where (x,y) is old_c, 
                    #     cost -= (CN[old_c][v,y] - 1)  [removing v from common neighbors of x and y... wait]
                    
                    # This is getting complex. Let me use a different strategy:
                    # Just do edge-level moves but batch them by vertex.
                    # Pick a vertex, then do n-1 edge moves for that vertex sequentially.
                    pass
                
                # Fallback: do single-edge moves but prefer edges incident to a random vertex
                # Pick an edge incident to v
                x = random.choice([u for u in range(n) if u != v])
                e_idx = edge_index[v, x]
                old_c = int(colors[e_idx])
                
                best_delta = 0
                best_new_c = old_c
                
                for new_c in range(5):
                    if new_c == old_c:
                        continue
                    delta = 0
                    # For each y != v, x
                    for y in range(n):
                        if y == v or y == x:
                            continue
                        e_vy = edge_index[v, y]
                        e_xy = edge_index[x, y]
                        c_xy = int(colors[e_xy])
                        c_vy = int(colors[e_vy])
                        
                        if c_xy == old_c:
                            delta -= (int(CN[old_c][v, y]) - 1)
                        if c_vy == old_c:
                            delta -= (int(CN[old_c][x, y]) - 1)
                        if c_xy == new_c:
                            delta += int(CN[new_c][v, y])
                        if c_vy == new_c:
                            delta += int(CN[new_c][x, y])
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_new_c = new_c
                
                if best_new_c == old_c:
                    continue
                
                # Accept with probability based on temperature
                if best_delta > 0:
                    if random.random() > np.exp(-best_delta / max(T, 0.01)):
                        continue
                
                colors[e_idx] = best_new_c
                current_cost += best_delta
                
                # Update CN
                for y in range(n):
                    if y == v or y == x:
                        continue
                    e_vy = edge_index[v, y]
                    e_xy = edge_index[x, y]
                    c_xy = int(colors[e_xy])
                    c_vy = int(colors[e_vy])
                    
                    if c_xy == old_c:
                        CN[old_c][v, y] -= 1
                        CN[old_c][y, v] -= 1
                    if c_vy == old_c:
                        CN[old_c][x, y] -= 1
                        CN[old_c][y, x] -= 1
                    if c_xy == best_new_c:
                        CN[best_new_c][v, y] += 1
                        CN[best_new_c][y, v] += 1
                    if c_vy == best_new_c:
                        CN[best_new_c][x, y] += 1
                        CN[best_new_c][y, x] += 1
                
                T *= cool_rate
                if T < min_T:
                    T = min_T
        
        return None

    def algebraic_seed(n):
        """Generate a structured initial coloring using a difference-set-like pattern."""
        # Use a cyclic pattern: color(i,j) = (j - i) mod 5 for a base, then perturb
        # This won't be valid but gives a structured start
        colors = np.zeros(n * (n - 1) // 2, dtype=np.int8)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                # Use a more sophisticated pattern
                d = j - i
                # Mix position and distance
                colors[idx] = (i * 3 + d * 7 + (i * d) % 5) % 5
                idx += 1
        return colors.tolist()

    # Try n=26 first
    for target_n in [26, 25, 24, 23]:
        if time_left() < 10:
            break
        
        # Use algebraic seed for first restart
        seed = algebraic_seed(target_n)
        result = try_find_n(target_n, max_restarts=30, max_iter=20000, seed_colors=seed)
        if result is not None:
            return {'n': target_n, 'colors': result}
        
        # Try more restarts without seed
        if time_left() > 20:
            result = try_find_n(target_n, max_restarts=50, max_iter=15000, seed_colors=None)
            if result is not None:
                return {'n': target_n, 'colors': result}
    
    # Fallback: try smaller n with more aggressive search
    for target_n in range(22, 10, -1):
        if time_left() < 5:
            break
        result = try_find_n(target_n, max_restarts=100, max_iter=10000, seed_colors=None)
        if result is not None:
            return {'n': target_n, 'colors': result}
    
    # Last resort: return a trivial valid coloring for small n
    # For n=5, any coloring works (no C4 possible in K5 with 5 colors... actually K5 has C4s)
    # For n=4, K4 has no C4, so any coloring works
    n = 4
    colors = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]  # K4 has 6 edges
    # Actually K4 has C(4,2)=6 edges
    colors = list(range(6)) % 5
    return {'n': n, 'colors': colors}
