import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin before 120s limit
    
    n = 18
    full_mask = (1 << n) - 1
    
    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) creates a K4."""
        if adj[u] & (1 << v):
            return True
        common = adj[u] & adj[v]
        c = common
        while c:
            lsb = c & -c
            w = lsb.bit_length() - 1
            if adj[w] & (common & ~lsb):
                return False
            c ^= lsb
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def get_triangles(adj):
        """Return list of triangle bitmasks."""
        triangles = []
        for u in range(n):
            v_mask = adj[u] & ~((1 << (u + 1)) - 1)
            while v_mask:
                lsb = v_mask & -v_mask
                v = lsb.bit_length() - 1
                v_mask ^= lsb
                common = adj[u] & adj[v] & ~((1 << (v + 1)) - 1)
                while common:
                    wlsb = common & -common
                    w = wlsb.bit_length() - 1
                    common ^= wlsb
                    triangles.append((1 << u) | (1 << v) | (1 << w))
        return triangles

    def evaluate(adj):
        """
        Find the size of the largest triangle-free induced subgraph.
        This is equivalent to n - (size of minimum triangle transversal).
        A triangle transversal is a set of vertices that hits every triangle.
        """
        triangles = get_triangles(adj)
        if not triangles:
            return n
        
        num_tris = len(triangles)
        
        # Precompute coverage mask for each vertex
        # vertex_cov[v] is a bitmask where bit i is 1 if vertex v is in triangle i
        vertex_cov = [0] * n
        for i, tri in enumerate(triangles):
            bits = tri
            while bits:
                lsb = bits & -bits
                v = lsb.bit_length() - 1
                vertex_cov[v] |= (1 << i)
                bits ^= lsb
        
        full_tri_mask = (1 << num_tris) - 1
        
        # Try transversal sizes k = 1, 2, ...
        # If a transversal of size k exists, the max triangle-free subgraph has size n - k
        for k in range(1, n + 1):
            # Check all combinations of k vertices
            for combo in combinations(range(n), k):
                cover = 0
                for v in combo:
                    cover |= vertex_cov[v]
                    if cover == full_tri_mask:
                        return n - k
            
        return 1

    # Initialize with K6,6,6 complete tripartite
    def init_k666():
        adj = [0] * n
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        for i in range(3):
            for j in range(i + 1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
        return adj

    # Generate random K4-free graph
    def generate_random_k4_free():
        adj = [0] * n
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(edges)
        for u, v in edges:
            if can_add_edge(adj, u, v):
                add_edge(adj, u, v)
        return adj

    # Generate a random regular-like graph
    def generate_random_regular_like(d):
        adj = [0] * n
        degrees = [0] * n
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(edges)
        for u, v in edges:
            if degrees[u] < d and degrees[v] < d:
                if can_add_edge(adj, u, v):
                    add_edge(adj, u, v)
                    degrees[u] += 1
                    degrees[v] += 1
        return adj

    # Generate multiple initial candidates
    best_adj = None
    best_score = n + 1
    
    candidates = []
    
    # K6,6,6
    candidates.append(init_k666())
    
    # Random K4-free
    for _ in range(5):
        candidates.append(generate_random_k4_free())
        
    # Random regular-like (degree 4, 5, 6)
    for d in [4, 5, 6]:
        candidates.append(generate_random_regular_like(d))
        
    # Evaluate all candidates
    for cand in candidates:
        score = evaluate(cand)
        if score < best_score:
            best_score = score
            best_adj = cand[:]
            
    current_adj = best_adj[:]
    current_score = best_score
    
    # Local search: focus on adding edges (reduces triangle-free subgraphs)
    # and occasional removals to escape local minima
    t = 1.5
    cooling = 0.998
    min_t = 0.01
    
    while time.time() < deadline and t > min_t:
        # Try to add a random edge (most likely to improve)
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        if current_adj[u] & (1 << v):
            continue
        if not can_add_edge(current_adj, u, v):
            continue
        
        old_u = current_adj[u]
        old_v = current_adj[v]
        add_edge(current_adj, u, v)
        new_score = evaluate(current_adj)
        
        delta = new_score - current_score
        if delta < 0 or (delta > 0 and random.random() < math.exp(-delta / t)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = current_adj[:]
        else:
            current_adj[u] = old_u
            current_adj[v] = old_v
        
        # Occasionally try removing an edge to explore
        if random.random() < 0.15:
            u = random.randint(0, n - 1)
            v_mask = current_adj[u]
            if v_mask:
                # Pick a random neighbor
                bits = []
                tmp = v_mask
                while tmp:
                    b = (tmp & -tmp).bit_length() - 1
                    bits.append(b)
                    tmp &= tmp - 1
                v = random.choice(bits)
                
                old_u = current_adj[u]
                old_v = current_adj[v]
                remove_edge(current_adj, u, v)
                new_score = evaluate(current_adj)
                
                delta = new_score - current_score
                if delta < 0 or (delta > 0 and random.random() < math.exp(-delta / t)):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = current_adj[:]
                else:
                    current_adj[u] = old_u
                    current_adj[v] = old_v
        
        t *= cooling
    
    # Convert best_adj to list of edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
    
    return edges
