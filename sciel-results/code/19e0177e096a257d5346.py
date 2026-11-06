import time
import random
import math

N = 18
DEADLINE_MARGIN = 5  # safety margin before 150s

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    
    # Represent graph as adjacency bitmask
    # adj_mask[i] is a bitmask of neighbors of vertex i
    
    # Initialize with K6,6,6
    adj_mask = [0] * N
    for i in range(N):
        for j in range(N):
            if i != j:
                part_i = i // 6
                part_j = j // 6
                if part_i != part_j:
                    adj_mask[i] |= (1 << j)
    
    def is_k4_free(adj_mask):
        for a in range(N):
            for b in range(a+1, N):
                if (adj_mask[a] >> b) & 1:
                    common = adj_mask[a] & adj_mask[b]
                    common &= ~((1 << a) | (1 << b))
                    # Check if common has any edge
                    c_list = []
                    m = common
                    while m:
                        c = (m & -m).bit_length() - 1
                        c_list.append(c)
                        m &= m - 1
                    for i in range(len(c_list)):
                        for j in range(i+1, len(c_list)):
                            if (adj_mask[c_list[i]] >> c_list[j]) & 1:
                                return False
        return True
    
    def get_triangles(adj_mask):
        triangles = []
        for a in range(N):
            for b in range(a+1, N):
                if (adj_mask[a] >> b) & 1:
                    common = adj_mask[a] & adj_mask[b]
                    common &= ~((1 << a) | (1 << b))
                    # For each pair in common that has an edge, we have a triangle
                    m = common
                    c_list = []
                    while m:
                        c = (m & -m).bit_length() - 1
                        c_list.append(c)
                        m &= m - 1
                    for i in range(len(c_list)):
                        for j in range(i+1, len(c_list)):
                            if (adj_mask[c_list[i]] >> c_list[j]) & 1:
                                triangles.append((a, b, c_list[i], c_list[j]))
        # Deduplicate and normalize
        tri_set = set()
        for t in triangles:
            tri_set.add(tuple(sorted(t)))
        return list(tri_set)
    
    # Actually, let me reconsider. A triangle is 3 vertices. Let me just find all triangles.
    def find_all_triangles(adj_mask):
        tris = []
        for a in range(N):
            for b in range(a+1, N):
                if (adj_mask[a] >> b) & 1:
                    common = adj_mask[a] & adj_mask[b]
                    common &= ~((1 << a) | (1 << b))
                    m = common
                    while m:
                        c = (m & -m).bit_length() - 1
                        if c > b:  # ensure a < b < c
                            tris.append((a, b, c))
                        m &= m - 1
        return tris
    
    def max_tri_free(adj_mask, time_limit=None):
        """Find size of largest triangle-free induced subgraph."""
        tris = find_all_triangles(adj_mask)
        if not tris:
            return N
        
        best = [0]
        # Branch and bound on vertex cover of triangles
        # State: available vertices bitmask
        # Pick a triangle, branch on removing one of its vertices
        
        def bnb(avail, depth):
            if time_limit and time.time() > time_limit:
                return
            cnt = bin(avail).count('1')
            if cnt <= best[0]:
                return
            # Find a triangle in avail
            tri = None
            for (a, b, c) in tris:
                if (avail >> a) & 1 and (avail >> b) & 1 and (avail >> c) & 1:
                    tri = (a, b, c)
                    break
            if tri is None:
                best[0] = cnt
                return
            a, b, c = tri
            bnb(avail & ~(1 << a), depth+1)
            bnb(avail & ~(1 << b), depth+1)
            bnb(avail & ~(1 << c), depth+1)
        
        bnb((1 << N) - 1, 0)
        return best[0]
    
    # Initial evaluation
    best_score = max_tri_free(adj_mask)
    best_mask = adj_mask[:]
    current_score = best_score
    
    # Simulated annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.997
    
    while time.time() < deadline and T > T_min:
        u, v = random.sample(range(N), 2)
        
        old_val = (adj_mask[u] >> v) & 1
        if old_val:
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
        else:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        if not is_k4_free(adj_mask):
            # Revert
            if old_val:
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
            else:
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
            T *= cooling
            continue
        
        new_score = max_tri_free(adj_mask)
        
        if new_score < current_score or (T > T_min and random.random() < math.exp((current_score - new_score) / T)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_mask = adj_mask[:]
        else:
            # Revert
            if old_val:
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
            else:
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
        
        T *= cooling
    
    # Return edges
    edges = []
    for i in range(N):
        for j in range(i+1, N):
            if (best_mask[i] >> j) & 1:
                edges.append([i, j])
    return edges
