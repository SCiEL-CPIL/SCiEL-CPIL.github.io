import sys
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
edge_order = P['edge_order']

# Precompute edge positions (1-based index)
edge_pos = {}
for i, (u, v) in enumerate(edge_order):
    edge_pos[(u, v)] = i + 1
    edge_pos[(v, u)] = i + 1

# Build adjacency lists for quick lookup based on threshold K
# We will rebuild or filter as K increases, but since N is small, we can just check edge_pos on the fly.
# To speed up backtracking, we pre-sort neighbors for each vertex by their edge index.
neighbors = [[] for _ in range(N)]
for u in range(N):
    for v in range(N):
        if u != v:
            idx = edge_pos[(u, v)]
            neighbors[u].append((idx, v))
    # Sort by index to explore low-index edges first (greedy bias)
    neighbors[u].sort()

def find_hamiltonian_cycle(max_k):
    """
    Finds a Hamiltonian cycle using edges with index <= max_k.
    Returns the cycle as a list of vertices, or None if not found.
    """
    # We fix vertex 0 as the start to reduce symmetry.
    # Path starts at 0.
    path = [0]
    visited = 1 << 0
    
    # Precompute allowed neighbors for this max_k
    # allowed_neighbors[u] is a list of (v) such that edge_pos[(u,v)] <= max_k
    allowed_neighbors = [[] for _ in range(N)]
    for u in range(N):
        for idx, v in neighbors[u]:
            if idx <= max_k:
                allowed_neighbors[u].append(v)
            else:
                break # Since sorted, rest are larger

    # Quick check: if any vertex has no allowed edges, return None
    for u in range(N):
        if not allowed_neighbors[u]:
            return None

    # Backtracking search
    # We need to visit all N vertices and return to 0.
    # So we need a path of length N-1 from 0, and the last vertex must connect to 0.
    
    # Use recursion with pruning
    # State: current vertex, visited mask, path list
    
    # Heuristic: At each step, try neighbors that lead to "tighter" constraints?
    # Simple heuristic: Try neighbors with fewer available unvisited neighbors first (most constrained first)
    
    def backtrack(u, visited_mask, path):
        if len(path) == N:
            # Check if last vertex connects back to 0
            if 0 in allowed_neighbors[u]:
                return path[:]
            return None
        
        # Pruning: Check if any unvisited vertex is "isolated" from the rest of the graph 
        # in a way that prevents completion.
        # Specifically, if an unvisited vertex v has no edges to (current path + other unvisited vertices), 
        # it's impossible.
        # A stronger check: For each unvisited vertex v, count how many allowed edges it has to 
        # (unvisited vertices) + (current vertex u if we are going to visit it? No, u is current).
        # Actually, standard check: If an unvisited vertex v has degree 0 in the subgraph induced by 
        # unvisited vertices + {u} (if we don't go to u? No, u is already visited).
        # Let's keep it simple: just check if any unvisited vertex has no edges to any *other* unvisited vertex 
        # AND no edge to u (current) AND no edge to 0 (start, if it's the last one).
        # This is complex. Let's rely on the low branching factor for small K.
        
        # Sort candidates by degree of unvisited neighbors to prune earlier
        candidates = []
        for v in allowed_neighbors[u]:
            if not (visited_mask & (1 << v)):
                # Count unvisited neighbors of v
                # deg = sum(1 for w in allowed_neighbors[v] if not (visited_mask & (1 << w)))
                # We can approximate or just use natural order (sorted by edge index)
                candidates.append(v)
        
        # Optional: Heuristic ordering - visit vertices with fewer options first
        # For now, use the order from allowed_neighbors (sorted by edge index)
        
        for v in candidates:
            new_mask = visited_mask | (1 << v)
            res = backtrack(v, new_mask, path + [v])
            if res is not None:
                return res
        return None

    # Start backtracking
    res = backtrack(0, 1 << 0, [0])
    return res

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s limit to be safe
    
    # Binary search or Linear search for K?
    # Linear is safer to guarantee we find the *first* K.
    # But we can start from a lower bound.
    # Lower bound: The graph must be connected.
    # Let's just iterate K from 1 to 276.
    
    best_cycle = None
    best_k = 277
    
    for k in range(1, 277):
        if time.time() > deadline:
            break
            
        cycle = find_hamiltonian_cycle(k)
        if cycle is not None:
            best_cycle = cycle
            best_k = k
            break # Found the minimal K for this specific search path. 
            # Note: Since we are iterating K from 1, the first K that yields a cycle 
            # is the optimal answer for the "existence" question. 
            # The problem asks to minimize the score, which is exactly this K.
            # So we can return immediately.
            
    if best_cycle is None:
        # Fallback: If no cycle found in time (should not happen for K=276), 
        # use a heuristic to find *any* cycle.
        # Greedy construction
        import random
        random.seed(614)
        vertices = list(range(1, N))
        random.shuffle(vertices)
        cycle = [0] + vertices
        # Try to improve with 2-opt on sum of indices? 
        # Or just return this.
        # Let's try a simple 2-opt to reduce max index
        def get_score(cyc):
            max_idx = 0
            for i in range(len(cyc)):
                u = cyc[i]
                v = cyc[(i+1) % len(cyc)]
                idx = edge_pos[(u, v)]
                if idx > max_idx:
                    max_idx = idx
            return max_idx
            
        current_score = get_score(cycle)
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for i in range(N):
                for j in range(i+1, N):
                    # Reverse segment i+1 to j
                    # Original edges: (cyc[i], cyc[i+1]) and (cyc[j], cyc[(j+1)%N])
                    # New edges: (cyc[i], cyc[j]) and (cyc[i+1], cyc[(j+1)%N])
                    u1, v1 = cyc[i], cyc[i+1]
                    u2, v2 = cyc[j], cyc[(j+1) % N]
                    
                    new_idx1 = edge_pos[(u1, v2)]
                    new_idx2 = edge_pos[(v1, u2)]
                    
                    old_max = max(edge_pos[(u1, v1)], edge_pos[(u2, v2)])
                    new_max = max(new_idx1, new_idx2)
                    
                    # We want to reduce the global max. 
                    # This local check is tricky. 
                    # Let's just calculate full score if change is promising.
                    # Or simpler: minimize sum of indices as proxy.
                    # Let's stick to the primary strategy: The linear search should find it.
                    # This fallback is just in case.
                    pass
                    
        return cycle

    return best_cycle
# EVOLVE-BLOCK-END
