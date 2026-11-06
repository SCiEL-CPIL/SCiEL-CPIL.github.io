import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110
    n = 18

    def edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def has_k4(adj):
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b]
                for c in range(b + 1, n):
                    if not ((common >> c) & 1):
                        continue
                    remaining = common & adj[c] & ~((1 << (c + 1)) - 1)
                    if remaining:
                        return True
        return False

    def get_triangles(adj):
        triangles = []
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b]
                for c in range(b + 1, n):
                    if (common >> c) & 1:
                        triangles.append((1 << a) | (1 << b) | (1 << c))
        return triangles

    def max_triangle_free_size_fast(adj):
        triangles = get_triangles(adj)
        if not triangles:
            return n
        
        tri_for_vertex = [[] for _ in range(n)]
        for idx, t in enumerate(triangles):
            for v in range(n):
                if (t >> v) & 1:
                    tri_for_vertex[v].append(idx)
        
        order = sorted(range(n), key=lambda v: len(tri_for_vertex[v]), reverse=True)
        
        vertex_tri_info = []
        for v in order:
            info = []
            for t_idx in tri_for_vertex[v]:
                t = triangles[t_idx]
                others = t & ~(1 << v)
                other_list = []
                for u in range(n):
                    if (others >> u) & 1:
                        other_list.append(u)
                info.append((t_idx, other_list[0], other_list[1]))
            vertex_tri_info.append(info)
        
        best_found = 0
        
        def dfs(pos, mask, size):
            nonlocal best_found
            if size + (n - pos) <= best_found:
                return
            if pos == n:
                if size > best_found:
                    best_found = size
                return
            
            v = order[pos]
            
            dfs(pos + 1, mask, size)
            
            valid = True
            for t_idx, u, w in vertex_tri_info[pos]:
                if (mask >> u) & 1 and (mask >> w) & 1:
                    valid = False
                    break
            
            if valid:
                new_mask = mask | (1 << v)
                dfs(pos + 1, new_mask, size + 1)
        
        dfs(0, 0, 0)
        return best_found

    def find_independent_set_paley(adj_paley):
        # Greedy independent set for the first 17 vertices
        n_p = 17
        remaining = list(range(n_p))
        indep = []
        while remaining:
            min_deg = float('inf')
            min_v = -1
            for v in remaining:
                deg = 0
                for u in remaining:
                    if u != v and (adj_paley[v] >> u) & 1:
                        deg += 1
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
            indep.append(min_v)
            remaining.remove(min_v)
            to_remove = [u for u in remaining if (adj_paley[min_v] >> u) & 1]
            for u in to_remove:
                remaining.remove(u)
        return indep

    # 1. Generate Paley Graph P(17)
    residues = set([i*i % 17 for i in range(17)])
    residues.discard(0)
    
    adj_paley = [0] * 17
    for u in range(17):
        for v in range(u + 1, 17):
            if (v - u) % 17 in residues:
                adj_paley[u] |= (1 << v)
                adj_paley[v] |= (1 << u)

    # 2. Initialize full graph
    random.seed(42)
    
    # Find an independent set in P(17) to connect vertex 17 to
    indep_set = find_independent_set_paley(adj_paley)
    
    # We will try a few different initial configurations for vertex 17
    # and run SA on the best one, or just run SA on one good start.
    
    best_adj = None
    best_score = 18
    
    # Try a few random independent sets or just the greedy one
    # The greedy one is size 4.
    
    for _ in range(5): # Try a few starts
        if time.time() > deadline - 50:
            break
            
        # Start with Paley
        adj = adj_paley[:] + [0]
        
        # Connect 17 to a random subset of an independent set?
        # Or just the greedy independent set.
        # Let's vary the independent set slightly by randomizing the greedy choice
        # Actually, just using the greedy one is a strong start.
        # Let's try connecting to the greedy independent set.
        
        for v in indep_set:
            adj[17] |= (1 << v)
            adj[v] |= (1 << 17)
            
        # Verify K4-free
        if has_k4(adj):
            # Should not happen if indep_set is truly independent
            # But just in case
            continue
            
        score = max_triangle_free_size_fast(adj)
        if score < best_score:
            best_score = score
            best_adj = adj[:]
            
    if best_adj is None:
        # Fallback: K_{6,6,6}
        adj1 = [0] * n
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        for i in range(3):
            for j in range(i + 1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        adj1[a] |= (1 << b)
                        adj1[b] |= (1 << a)
        best_adj = adj1
        best_score = max_triangle_free_size_fast(best_adj)

    # 3. Local Search (Simulated Annealing)
    adj = best_adj[:]
    current_score = best_score
    best_adj_sa = adj[:]
    best_score_sa = current_score
    
    temperature = 1.0
    cooling = 0.999
    
    # Precompute all possible edges for random selection
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    while time.time() < deadline - 5:
        if best_score_sa <= 5:
            break
            
        i, j = random.choice(all_edges)
        
        old_adj = adj[:]
        
        # Flip edge
        if (adj[i] >> j) & 1:
            adj[i] &= ~(1 << j)
            adj[j] &= ~(1 << i)
        else:
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            
        if has_k4(adj):
            adj = old_adj
            continue
            
        new_score = max_triangle_free_size_fast(adj)
        
        delta = new_score - current_score
        
        if delta <= 0 or random.random() < min(1.0, math.exp(-delta / max(temperature, 0.01))):
            current_score = new_score
            if new_score < best_score_sa:
                best_score_sa = new_score
                best_adj_sa = adj[:]
        else:
            adj = old_adj
            
        temperature *= cooling

    return edges_from_adj(best_adj_sa)
