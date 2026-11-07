<<<<<<< SEARCH
# EVOLVE-BLOCK-START
import time
import math
from itertools import combinations
import random

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add((min(u,v), max(u,v)))
    
    # Build base adjacency
    base_adj = [set() for _ in range(n)]
    for u, v in base_edges:
        base_adj[u].add(v)
        base_adj[v].add(u)
    
    # All possible cross-parity non-base edges
    all_candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if (u % 2) != (v % 2):
                a, b = min(u,v), max(u,v)
                if (a,b) not in base_edges:
                    all_candidates.append((u,v))
    
    # Precompute which pairs of candidate edges are "compatible" (don't create a triangle together with base edges)
    # Two added edges (a,b) and (c,d) are incompatible if they share a vertex and that vertex's neighbors form a triangle
    # More precisely: adding edge (u,v) creates a triangle if there exists w such that (u,w) and (v,w) are both in the graph
    # For two added edges, we need to check all combinations
    
    # Faster: for each candidate edge, precompute which other candidate edges conflict with it
    # Edge e1=(u1,v1) conflicts with e2=(u2,v2) if:
    #   - They share a vertex x, and the other endpoint of e1 is adjacent to the other endpoint of e2 (via base or other added edges)
    #   - Or they form a triangle directly: e1=(u,v), e2=(v,w), and (u,w) is in base or also added
    
    # Let's precompute conflict graph
    # Two edges conflict if adding both would create a triangle
    edge_index = {}
    for i, e in enumerate(all_candidates):
        edge_index[e] = i
    
    num_cand = len(all_candidates)
    conflict = [set() for _ in range(num_cand)]
    
    for i in range(num_cand):
        u1, v1 = all_candidates[i]
        for j in range(i+1, num_cand):
            u2, v2 = all_candidates[j]
            # Check if adding both edges creates a triangle
            # Triangle exists if there's a vertex x connected to two vertices that are connected to each other
            # With edges (u1,v1) and (u2,v2), triangles can form if:
            # Case 1: share a vertex, say u1==u2, then need v1-v2 edge
            # Case 2: u1==v2, then need u2-v1 edge  
            # Case 3: v1==u2, then need u1-v2 edge
            # Case 4: v1==v2, then need u1-u2 edge
            # Case 5: no shared vertex, but u1-u2 and v1-v2 both exist (or u1-v2 and v1-u2)
            
            creates_triangle = False
            
            # Check shared vertex cases
            if u1 == u2 or u1 == v2:
                # Shared vertex u1 (or equivalent)
                other1 = v1 if u1 == u2 else u2
                other2 = u2 if u1 == u2 else v1
                # Wait, let me be more careful
                pass
            
            # Simpler: just check all 3 pairs
            # The two edges give us up to 4 vertices. A triangle needs 3 mutually connected vertices.
            # Vertices involved: u1, v1, u2, v2
            # Pairs that are edges in the new graph: (u1,v1), (u2,v2), and all base edges among these 4 vertices
            
            verts = set([u1, v1, u2, v2])
            if len(verts) < 3:
                continue  # Can't form triangle with < 3 distinct vertices
            
            # Check all triples of these vertices
            for a, b, c in combinations(sorted(verts), 3):
                # Check if a-b, b-c, a-c are all edges
                def is_edge(x, y):
                    x, y = min(x,y), max(x,y)
                    if x == y:
                        return False
                    if (x,y) in base_edges:
                        return True
                    if (x,y) == (min(u1,v1), max(u1,v1)) or (x,y) == (min(u2,v2), max(u2,v2)):
                        return True
                    return False
                
                if is_edge(a,b) and is_edge(b,c) and is_edge(a,c):
                    creates_triangle = True
                    break
            
            if creates_triangle:
                conflict[i].add(j)
                conflict[j].add(i)
    
    # Now implement validity check using precomputed conflicts
    def is_valid_fast(added_indices):
        """Check if a set of candidate edge indices is valid (triangle-free, diameter <= 4)"""
        # Check triangle-free using precomputed conflicts
        added_set = set(added_indices)
        for i in added_indices:
            for j in conflict[i]:
                if j in added_set:
                    return False
        
        # Build full adjacency
        adj = [set() for _ in range(n)]
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
        for idx in added_indices:
            u, v = all_candidates[idx]
            adj[u].add(v)
            adj[v].add(u)
        
        # Check diameter <= 4 via BFS from each vertex
        for start in range(n):
            dist = [-1]*n
            dist[start] = 0
            queue = [start]
            idx_q = 0
            while idx_q < len(queue):
                u = queue[idx_q]
                idx_q += 1
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            for d in dist:
                if d == -1 or d > 4:
                    return False
        return True
    
    # Compute diameter for a set of edges (for scoring)
    def compute_diameter(added_indices):
        adj = [set() for _ in range(n)]
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
        for idx in added_indices:
            u, v = all_candidates[idx]
            adj[u].add(v)
            adj[v].add(u)
        
        max_dist = 0
        for start in range(n):
            dist = [-1]*n
            dist[start] = 0
            queue = [start]
            idx_q = 0
            while idx_q < len(queue):
                u = queue[idx_q]
                idx_q += 1
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            for d in dist:
                if d > max_dist:
                    max_dist = d
        return max_dist
    
    # Start time and deadline
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_solution = None
    best_count = float('inf')
    
    def try_solution(added_indices):
        nonlocal best_solution, best_count
        if is_valid_fast(added_indices):
            if len(added_indices) < best_count:
                best_count = len(added_indices)
                best_solution = list(added_indices)
    
    # Constructive heuristic: try to build a solution greedily
    # Start with no edges, add edges that most reduce diameter
    def greedy_construct():
        current = []
        current_diam = 23  # Path diameter
        
        # Priority: edges that connect far-apart vertices
        # Sort candidates by potential to reduce diameter
        # For each candidate, estimate how much it reduces max distance
        
        remaining = list(range(num_cand))
        
        while time.time() < deadline:
            if not remaining:
                break
            
            # Try adding each remaining edge and see which reduces diameter most
            best_edge = None
            best_new_diam = current_diam
            
            # Sample a subset for speed
            sample_size = min(len(remaining), 50)
            sample = random.sample(remaining, sample_size)
            
            for idx in sample:
                # Check if this edge conflicts with current
                if any(j in conflict[idx] for j in current):
                    continue
                
                test_set = current + [idx]
                d = compute_diameter(test_set)
                if d < best_new_diam:
                    best_new_diam = d
                    best_edge = idx
            
            if best_edge is None:
                break
            
            current.append(best_edge)
            remaining.remove(best_edge)
            current_diam = best_new_diam
            
            if current_diam <= 4:
                break
        
        return current
    
    # Local search: start from a solution, try to remove or swap edges
    def local_search(start_indices, max_iters=1000):
        current = list(start_indices)
        current_set = set(current)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try removing edges
            for idx in current:
                if any(j in conflict[idx] for j in current if j != idx):
                    continue  # Can't remove if it would break... wait, removing always helps triangle-free
                # Actually removing an edge never creates a triangle, so always safe for that constraint
                test_set = [j for j in current if j != idx]
                if is_valid_fast(test_set):
                    current = test_set
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swapping: remove one, add one
            for remove_idx in current:
                if time.time() > deadline:
                    break
                
                base_set = [j for j in current if j != remove_idx]
                
                # Try adding each non-conflicting edge
                for add_idx in range(num_cand):
                    if add_idx in base_set:
                        continue
                    if any(j in conflict[add_idx] for j in base_set):
                        continue
                    if any(j in conflict[add_idx] for j in [remove_idx]):
                        # This is fine, we're removing remove_idx
                        pass
                    
                    test_set = base_set + [add_idx]
                    if is_valid_fast(test_set):
                        current = test_set
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                break
        
        return current
    
    # Run greedy construction
    random.seed(42)
    greedy_sol = greedy_construct()
    try_solution(greedy_sol)
    
    # Run local search from greedy solution
    if greedy_sol:
        ls_sol = local_search(greedy_sol)
        try_solution(ls_sol)
    
    # Try multiple random restarts with local search
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Random starting point: pick random non-conflicting edges
        random_set = set()
        for idx in random.sample(range(num_cand), min(num_cand, 30)):
            if not any(j in conflict[idx] for j in random_set):
                random_set.add(idx)
        
        random_list = list(random_set)
        ls_sol = local_search(random_list, max_iters=200)
        try_solution(ls_sol)
    
    # If we have a valid solution, return it
    if best_solution is not None:
        return [[u, v] for u, v in [all_candidates[i] for i in best_solution]]
    
    # Fallback: return empty list (should not happen if greedy works)
    return []

# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Base graph: path 0-1-2-...-23
    base_adj = [set() for _ in range(n)]
    for i in range(n - 1):
        base_adj[i].add(i + 1)
        base_adj[i + 1].add(i)
    
    # Key insight: In a bipartite graph (even/odd partition), there are NO triangles.
    # The path 0-1-2-...-23 is bipartite with even vertices {0,2,4,...,22} and odd {1,3,...,23}.
    # Any edge between an even and odd vertex preserves bipartiteness, hence triangle-freeness.
    # So we only need to find minimum cross-parity edges achieving diameter <= 4.
    
    even_vertices = [i for i in range(n) if i % 2 == 0]
    odd_vertices = [i for i in range(n) if i % 2 == 1]
    
    # Generate all cross-parity non-base edges
    all_edges = []
    edge_set = set()
    for u in even_vertices:
        for v in odd_vertices:
            if u != v - 1 and u != v + 1:  # Not a base edge
                a, b = min(u, v), max(u, v)
                if (a, b) not in edge_set:
                    edge_set.add((a, b))
                    all_edges.append((a, b))
    
    num_edges = len(all_edges)
    
    # Precompute: for each pair of vertices, which edges can help connect them within distance 4
    # An edge (u,v) helps if it creates a shorter path between some pair.
    
    def compute_diameter(added_edge_indices):
        """Compute diameter of graph with base + added edges."""
        adj = [set() for _ in range(n)]
        for i in range(n - 1):
            adj[i].add(i + 1)
            adj[i + 1].add(i)
        for idx in added_edge_indices:
            u, v = all_edges[idx]
            adj[u].add(v)
            adj[v].add(u)
        
        max_dist = 0
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > max_dist:
                            max_dist = dist[v]
                        q.append(v)
            if max_dist > 4:
                return max_dist  # Early exit
        return max_dist
    
    def is_valid(added_edge_indices):
        """Check if diameter <= 4 (triangle-free is guaranteed by bipartite structure)."""
        return compute_diameter(added_edge_indices) <= 4
    
    # Branch and bound: try to find minimum number of edges
    best_solution = None
    best_count = float('inf')
    
    # Sort edges by "usefulness" - edges that connect far-apart vertices first
    # An edge (u,v) where |u-v| is large is more useful for reducing diameter
    edge_order = sorted(range(num_edges), key=lambda i: -abs(all_edges[i][0] - all_edges[i][1]))
    
    # For each edge, precompute which other edges are "compatible" (all are, since bipartite)
    # No conflicts needed!
    
    # Greedy initial solution
    def greedy():
        current = []
        current_diam = 23
        
        while time.time() < deadline:
            if current_diam <= 4:
                break
            
            best_idx = None
            best_new_diam = current_diam
            
            for idx in edge_order:
                if idx in current:
                    continue
                test = current + [idx]
                d = compute_diameter(test)
                if d < best_new_diam:
                    best_new_diam = d
                    best_idx = idx
            
            if best_idx is None:
                break
            
            current.append(best_idx)
            current_diam = best_new_diam
        
        return current
    
    # Get initial solution via greedy
    greedy_sol = greedy()
    if greedy_sol and is_valid(greedy_sol):
        best_solution = greedy_sol
        best_count = len(greedy_sol)
    
    # Try to improve: local search (remove edges, swap)
    def local_search(start_sol):
        current = list(start_sol)
        current_set = set(current)
        
        for _ in range(500):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try removing each edge
            for idx in list(current):
                test = [j for j in current if j != idx]
                if is_valid(test):
                    current = test
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swapping: remove one, add one
            for remove_idx in list(current):
                if time.time() > deadline:
                    break
                base = [j for j in current if j != remove_idx]
                for add_idx in edge_order:
                    if add_idx in base:
                        continue
                    test = base + [add_idx]
                    if is_valid(test):
                        current = test
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return current
    
    if best_solution is not None:
        improved_sol = local_search(best_solution)
        if improved_sol and is_valid(improved_sol):
            if len(improved_sol) < best_count:
                best_solution = improved_sol
                best_count = len(improved_sol)
    
    # Branch and bound to prove optimality or find better
    def branch_and_bound(k, remaining_edges, current_edges):
        """Try to find a solution with exactly k edges from remaining_edges."""
        nonlocal best_solution, best_count, deadline
        
        if time.time() > deadline:
            return None
        
        if len(current_edges) == k:
            if is_valid(current_edges):
                return list(current_edges)
            return None
        
        if len(current_edges) + len(remaining_edges) < k:
            return None
        
        if k == 0:
            if is_valid(current_edges):
                return list(current_edges)
            return None
        
        # Lower bound: if even with all remaining edges we can't reach k, fail
        # Upper bound check: if current_edges already has diameter > 4 and adding
        # any subset of remaining can't fix it, fail. (Hard to check cheaply)
        
        # Try each remaining edge
        for i, idx in enumerate(remaining_edges):
            if time.time() > deadline:
                return None
            
            result = branch_and_bound(k, remaining_edges[i+1:], current_edges + [idx])
            if result is not None:
                return result
        
        return None
    
    # Try to find solutions with fewer edges than current best
    for k in range(1, best_count):
        if time.time() > deadline:
            break
        
        # Use a heuristic ordering for B&B
        result = branch_and_bound(k, edge_order[:min(num_edges, 60)], [])
        if result is not None:
            best_solution = result
            best_count = k
            break
    
    # Return the best solution found
    if best_solution is not None:
        return [[int(u), int(v)] for u, v in [all_edges[i] for i in best_solution]]
    
    # Fallback: should not happen
    return []

# EVOLVE-BLOCK-END
>>>>>>> REPLACE
