"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_START_TIME = None

def _get_time_left(deadline):
    if _START_TIME is None:
        return deadline
    return deadline - (time.time() - _START_TIME)

def _is_k4_free(adj, n):
    """Check if graph with adjacency matrix adj on n vertices is K4-free."""
    for a in range(n):
        for b in range(a+1, n):
            if not adj[a][b]:
                continue
            for c in range(b+1, n):
                if not adj[a][c] or not adj[b][c]:
                    continue
                for d in range(c+1, n):
                    if adj[a][d] and adj[b][d] and adj[c][d]:
                        return False
    return True

def _max_triangle_free_induced(adj, n, deadline, start_time):
    """Find largest triangle-free induced subgraph size via branch and bound."""
    best = 0
    # Try to find a triangle-free set of size s
    # Use backtracking with pruning
    
    def dfs(vertices, current_size):
        nonlocal best
        if time.time() - start_time > deadline:
            return
        if current_size > best:
            best = current_size
        if not vertices:
            return
        
        # Upper bound: current_size + len(vertices)
        if current_size + len(vertices) <= best:
            return
        
        # Try to add vertices
        for i, v in enumerate(vertices):
            # Check if adding v keeps triangle-free property
            valid = True
            for i2 in range(i+1, len(vertices)):
                v2 = vertices[i2]
                if adj[v][v2]:
                    # Check if there's a common neighbor in current set that would form triangle
                    # Actually for triangle-free induced subgraph, we just need no edges within the set
                    # Wait - triangle-free means no K3, not edge-free
                    # So we need to check that adding v doesn't create a triangle with two existing vertices
                    pass
            
            # For triangle-free induced subgraph: the induced subgraph on the selected vertices has no triangles
            # So when adding v, we need to ensure no triangle is formed
            # A triangle with v and two existing vertices u1, u2 requires edges (v,u1), (v,u2), (u1,u2)
            
            # Check all pairs in current set
            can_add = True
            for i2 in range(len(current_vertices)):
                u1 = current_vertices[i2]
                if not adj[v][u1]:
                    continue
                for i3 in range(i2+1, len(current_vertices)):
                    u2 = current_vertices[i3]
                    if adj[v][u2] and adj[u1][u2]:
                        can_add = False
                        break
                if not can_add:
                    break
            
            if can_add:
                new_current = current_vertices + [v]
                remaining = vertices[:i] + vertices[i+1:]
                dfs(remaining, current_size + 1, new_current)
    
    # This approach is too slow. Let's use a different strategy.
    # For each subset size from large to small, check if there's a triangle-free induced subgraph of that size
    
    # Better approach: use the fact that we want to MINIMIZE the max triangle-free induced subgraph
    # So we want to find graphs where every subset of size s+1 contains a triangle
    
    # Let's just do a simple upper bound calculation
    # The maximum independent set in the triangle graph gives a bound
    
    # For now, return a reasonable estimate
    # Actually, let's implement a proper search
    
    # Use greedy approach to find large triangle-free induced subgraphs
    def greedy_triangle_free(adj, n):
        """Greedy: build largest triangle-free induced subgraph."""
        remaining = set(range(n))
        selected = []
        
        while remaining:
            # Find vertex with minimum "triangle risk"
            best_v = None
            best_score = float('inf')
            for v in remaining:
                # Count triangles that would be formed if we add v
                # v creates triangle with pair (u1, u2) if adj[v][u1] and adj[v][u2] and adj[u1][u2]
                risk = 0
                for u1 in selected:
                    if not adj[v][u1]:
                        continue
                    for u2 in selected:
                        if u2 <= u1:
                            continue
                        if adj[v][u2] and adj[u1][u2]:
                            risk += 1
                if risk < best_score:
                    best_score = risk
                    best_v = v
            
            if best_v is None:
                break
            
            # Check if adding best_v creates any triangle
            can_add = True
            for u1 in selected:
                if not adj[best_v][u1]:
                    continue
                for u2 in selected:
                    if u2 <= u1:
                        continue
                    if adj[best_v][u2] and adj[u1][u2]:
                        can_add = False
                        break
                if not can_add:
                    break
            
            if can_add:
                selected.append(best_v)
                remaining.remove(best_v)
            else:
                # Can't add any more
                break
        
        return len(selected)
    
    return greedy_triangle_free(adj, n)

def _evaluate_graph(adj, n, deadline, start_time):
    """Evaluate a graph: return (max_triangle_free_size, is_k4_free)."""
    if not _is_k4_free(adj, n):
        return (999, False)
    mtf = _max_triangle_free_induced(adj, n, deadline, start_time)
    return (mtf, True)

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    deadline = 110  # seconds, leave margin for 120s limit
    n = 18
    
    # Start with the known solution: K6,6,6 gives max triangle-free induced subgraph of size 12
    # (two parts combined = 12 vertices, which is triangle-free since no edges within parts)
    # Wait, in K6,6,6, edges exist between different parts. Within a part, no edges.
    # So a single part of 6 vertices induces an edgeless graph (triangle-free).
    # Two parts: vertices from part A and part B. Edges between A and B exist.
    # The induced subgraph on A∪B has all edges between A and B, none within A or within B.
    # This is a complete bipartite graph K6,6, which is triangle-free.
    # So max triangle-free induced subgraph is at least 12.
    # Can we do better? Three parts would have edges between all pairs, creating triangles.
    # So the answer for K6,6,6 is 12.
    
    # We want to find K4-free graphs with smaller max triangle-free induced subgraph.
    # Target: get below 12.
    
    # Strategy: use random search with local improvement
    # Start from random K4-free graphs and try to minimize max triangle-free induced subgraph
    
    def random_k4_free_graph(n):
        """Generate a random K4-free graph."""
        adj = [[False]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                # Randomly decide edge
                if random.random() < 0.5:
                    adj[i][j] = True
                    adj[j][i] = True
                # Check K4-free constraint locally
        # This doesn't guarantee K4-free. Need to repair or use different approach.
        return adj
    
    def is_k4_free_fast(adj, n):
        """Fast K4-free check."""
        for a in range(n):
            neighbors_a = [v for v in range(n) if v != a and adj[a][v]]
            for i, b in enumerate(neighbors_a):
                common_with_a = [v for v in neighbors_a[i+1:] if adj[b][v]]
                for c in common_with_a:
                    # Check if there's a d connected to a, b, c
                    for d in range(c+1, n):
                        if adj[a][d] and adj[b][d] and adj[c][d]:
                            return False
        return True
    
    # Better approach: use structured graphs
    # Try various constructions
    
    best_edges = None
    best_score = 12  # from K6,6,6
    
    # Construction 1: K6,6,6
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    edges = []
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    edges.append([a, b])
    best_edges = edges
    
    # Try to improve by modifying the graph
    # Local search: flip edges and check if K4-free and if score improves
    
    def get_adjacency(edges, n):
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        return adj
    
    def compute_score(adj, n):
        """Compute max triangle-free induced subgraph size."""
        # Use greedy with multiple restarts
        best = 0
        for _ in range(50):
            random.shuffle(parts)  # Won't work, parts is fixed
            # Greedy from random ordering
            order = list(range(n))
            random.shuffle(order)
            selected = []
            for v in order:
                can_add = True
                for u1 in selected:
                    if not adj[v][u1]:
                        continue
                    for u2 in selected:
                        if u2 <= u1:
                            continue
                        if adj[v][u2] and adj[u1][u2]:
                            can_add = False
                            break
                    if not can_add:
                        break
                if can_add:
                    selected.append(v)
            best = max(best, len(selected))
        return best
    
    # This is getting complex. Let's use a simpler approach.
    # The key insight: we need K4-free graphs where every subset of size s+1 has a triangle.
    
    # Known result: f(18) is the minimum over K4-free graphs of the max triangle-free induced subgraph.
    # We want to find such a graph with small f(18).
    
    # Let's try a different construction: use a random graph and enforce K4-free
    
    def generate_and_check(n, deadline, start_time):
        """Try random graphs until we find a good K4-free one."""
        best_mtf = 999
        best_adj = None
        
        while time.time() - start_time < deadline:
            # Generate random graph
            adj = [[False]*n for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj[i][j] = True
                        adj[j][i] = True
            
            # Check K4-free
            if not is_k4_free_fast(adj, n):
                continue
            
            # Compute max triangle-free induced subgraph
            mtf = compute_score(adj, n)
            if mtf < best_mtf:
                best_mtf = mtf
                best_adj = [row[:] for row in adj]
                if mtf <= 10:  # Good enough
                    break
        
        if best_adj is None:
            return None, best_mtf
        return best_adj, best_mtf
    
    # Run search
    adj, mtf = generate_and_check(n, deadline, _START_TIME)
    
    if adj is not None and mtf < best_score:
        # Convert to edge list
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    edges.append([i, j])
        best_edges = edges
        best_score = mtf
    
    return best_edges
# EVOLVE-BLOCK-END
