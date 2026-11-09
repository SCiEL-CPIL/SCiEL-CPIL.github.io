"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s budget, 5s safety margin
    
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add(tuple(sorted(e)))
    
    # Adjacency for base
    adj_base = [set() for _ in range(n)]
    for u, v in base_edges:
        adj_base[u].add(v)
        adj_base[v].add(u)
    
    # Candidate edges: all non-base, non-self edges
    # For triangle-free, we need to be careful
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in base_edges:
                candidates.append((u, v))
    
    # Helper: check if adding edge (u,v) to current adj keeps triangle-free
    def is_triangle_free_add(adj, u, v):
        # Check no common neighbor
        common = adj[u] & adj[v]
        return len(common) == 0
    
    # Helper: compute diameter
    def compute_diameter(adj):
        # BFS from each vertex
        max_dist = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = deque([src])
            while queue:
                node = queue.popleft()
                for nbr in adj[node]:
                    if dist[nbr] == -1:
                        dist[nbr] = dist[node] + 1
                        queue.append(nbr)
            max_dist = max(max_dist, max(dist))
        return max_dist
    
    # Helper: check if a set of edges is valid (triangle-free, connected, diameter <= 4)
    def check_valid(added_edges):
        adj = [set(s) for s in adj_base]
        for u, v in added_edges:
            if not is_triangle_free_add(adj, u, v):
                return False
            adj[u].add(v)
            adj[v].add(u)
        # Check connected
        dist = [-1] * n
        dist[0] = 0
        queue = deque([0])
        while queue:
            node = queue.popleft()
            for nbr in adj[node]:
                if dist[nbr] == -1:
                    dist[nbr] = dist[node] + 1
                    queue.append(nbr)
        if any(d == -1 for d in dist):
            return False
        # Check diameter
        diam = compute_diameter(adj)
        return diam <= 4
    
    best_edges = None
    best_count = float('inf')
    
    # Approach 1: Try structured patterns - jump-based
    # Pattern: add edges (i, i+k) for various k
    for k in range(2, n):
        for offset in range(k):
            edges_to_add = []
            for i in range(0, n, k):
                j = i + k
                if j < n and (i, j) not in base_edges:
                    edges_to_add.append((i, j))
            
            if not edges_to_add:
                continue
            
            if check_valid(edges_to_add):
                if len(edges_to_add) < best_count:
                    best_count = len(edges_to_add)
                    best_edges = [list(e) for e in edges_to_add]
    
    # Approach 2: Greedy edge addition with local search
    if time.time() < deadline:
        adj_greedy = [set(s) for s in adj_base]
        added = []
        # Repeatedly add the edge that most reduces diameter while keeping triangle-free
        for iteration in range(20):  # max 20 edges
            if time.time() > deadline - 5:
                break
            
            curr_diam = compute_diameter(adj_greedy)
            if curr_diam <= 4:
                break
            
            best_edge = None
            best_new_diam = float('inf')
            
            added_set = set((min(a,b), max(a,b)) for a,b in added)
            
            for u, v in candidates:
                key = (min(u,v), max(u,v))
                if key in added_set:
                    continue
                if not is_triangle_free_add(adj_greedy, u, v):
                    continue
                
                # Test adding this edge
                adj_greedy[u].add(v)
                adj_greedy[v].add(u)
                new_diam = compute_diameter(adj_greedy)
                adj_greedy[u].remove(v)
                adj_greedy[v].remove(u)
                
                if new_diam < best_new_diam:
                    best_new_diam = new_diam
                    best_edge = (u, v)
            
            if best_edge is None:
                break
            
            u, v = best_edge
            adj_greedy[u].add(v)
            adj_greedy[v].add(u)
            added.append([u, v])
        
        curr_diam = compute_diameter(adj_greedy)
        if curr_diam <= 4 and len(added) < best_count:
            best_count = len(added)
            best_edges = added
    
    # Approach 3: Local search - try to remove edges from current best
    if best_edges is not None and time.time() < deadline - 10:
        current = [list(e) for e in best_edges]
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for i in range(len(current)):
                trial = current[:i] + current[i+1:]
                if check_valid(trial):
                    current = trial
                    improved = True
                    break
        if len(current) < best_count:
            best_count = len(current)
            best_edges = current
    
    # Approach 4: Try specific hub-based constructions
    # Hub-based: pick some hub vertices, connect them to nearby vertices
    if time.time() < deadline - 10:
        # Try different hub configurations
        for hub_count in range(2, 8):
            if time.time() > deadline - 10:
                break
            # Try different hub positions
            for hubs in combinations(range(n), hub_count):
                if time.time() > deadline - 10:
                    break
                edges_to_add = []
                for h in hubs:
                    # Connect hub to vertices within distance 2 on path
                    for d in [-2, -1, 1, 2]:
                        v = h + d
                        if 0 <= v < n and (min(h,v), max(h,v)) not in base_edges:
                            key = (min(h,v), max(h,v))
                            if key not in [(min(a,b), max(a,b)) for a,b in edges_to_add]:
                                edges_to_add.append((h, v))
                
                if check_valid(edges_to_add):
                    if len(edges_to_add) < best_count:
                        best_count = len(edges_to_add)
                        best_edges = [list(e) for e in edges_to_add]
    
    # Approach 5: Random restart greedy
    import random
    random.seed(42)
    if time.time() < deadline - 5:
        for restart in range(50):
            if time.time() > deadline - 5:
                break
            
            adj_rand = [set(s) for s in adj_base]
            added_rand = []
            shuffled = candidates[:]
            random.shuffle(shuffled)
            
            for u, v in shuffled:
                if time.time() > deadline - 5:
                    break
                if not is_triangle_free_add(adj_rand, u, v):
                    continue
                adj_rand[u].add(v)
                adj_rand[v].add(u)
                added_rand.append([u, v])
                
                # Check if diameter is now <= 4
                diam = compute_diameter(adj_rand)
                if diam <= 4:
                    # Try to remove edges
                    current = [list(e) for e in added_rand]
                    improved = True
                    while improved and time.time() < deadline - 5:
                        improved = False
                        for i in range(len(current)):
                            trial = current[:i] + current[i+1:]
                            if check_valid(trial):
                                current = trial
                                improved = True
                                break
                    if len(current) < best_count:
                        best_count = len(current)
                        best_edges = current
                    break
    
    # Fallback: return a known valid solution
    if best_edges is None:
        # Use the all-odd-difference edges as fallback
        best_edges = [[u, v] for u in range(n) for v in range(u + 1, n) if (u - v) % 2 and (u, v) not in base_edges]
    
    # Verify the solution
    adj_final = [set(s) for s in adj_base]
    for u, v in best_edges:
        adj_final[u].add(v)
        adj_final[v].add(u)
    
    # Check diameter
    diam_final = compute_diameter(adj_final)
    
    if diam_final > 4:
        # Fallback to seed solution
        best_edges = [[u, v] for u in range(n) for v in range(u + 1, n) if (u - v) % 2 and (u, v) not in base_edges]
    
    return [[int(u), int(v)] for u, v in best_edges]

# EVOLVE-BLOCK-END
