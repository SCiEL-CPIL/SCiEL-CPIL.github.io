import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def _is_k4_free(n, edges):
    """Check if graph is K4-free using bitmasks for adjacency."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    # For every pair (u,v) that is an edge, check if they share two common neighbors
    # forming a K4
    for u in range(n):
        au = adj[u]
        for v in range(u + 1, n):
            if au & (1 << v):
                # common neighbors of u and v
                common = au & adj[v]
                # remove u and v from common
                common &= ~(1 << u)
                common &= ~(1 << v)
                # if there exist two vertices w, x in common such that w-x is an edge, then K4
                # Check all pairs in common
                w_list = [w for w in range(n) if (common >> w) & 1]
                for i in range(len(w_list)):
                    for j in range(i + 1, len(w_list)):
                        w, x = w_list[i], w_list[j]
                        if adj[w] & (1 << x):
                            return False
    return True

def _max_triangle_free_subgraph(n, edges):
    """Find the largest subset of vertices inducing a triangle-free subgraph."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # We want the largest set S such that the induced subgraph on S is triangle-free.
    # This is equivalent to: no three vertices in S form a triangle.
    # We can use a branch and bound approach.
    
    # Precompute triangles
    triangles = []
    for a in range(n):
        for b in range(a + 1, n):
            if adj[a] & (1 << b):
                for c in range(b + 1, n):
                    if (adj[b] & (1 << c)) and (adj[a] & (1 << c)):
                        triangles.append((a, b, c))
    
    # We need to find the maximum size subset S that contains no triangle.
    # This is the complement of the minimum vertex cover of the hypergraph of triangles.
    # For small n=18, we can use a simple branch and bound.
    
    best = [0]
    
    def dfs(remaining, current_size, current_set):
        if not remaining:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Upper bound: current_size + len(remaining)
        if current_size + len(remaining) <= best[0]:
            return
        
        # Pick a vertex from remaining
        v = remaining[0]
        rest = remaining[1:]
        
        # Option 1: include v
        # v can only be included if it doesn't form a triangle with two other vertices in current_set
        # Actually, we need to check: if we add v to current_set, does it create a triangle?
        # A triangle is formed if there exist a, b in current_set such that (a,b,v) is a triangle.
        can_include = True
        for tri in triangles:
            if v in tri:
                other_two = [x for x in tri if x != v]
                if other_two[0] in current_set and other_two[1] in current_set:
                    can_include = False
                    break
        
        if can_include:
            new_set = current_set | {v}
            # New remaining: we can still include others, but we need to recheck
            # Actually, we just remove v from remaining
            dfs(rest, current_size + 1, new_set)
        
        # Option 2: exclude v
        dfs(rest, current_size, current_set)
    
    all_vertices = set(range(n))
    dfs(sorted(all_vertices), 0, frozenset())
    
    return best[0]

def construct():
    n = 18
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Start with the complete tripartite graph K6,6,6
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    edges = [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
    
    # The score is the size of the largest triangle-free induced subgraph.
    # For K6,6,6, any two parts form a complete bipartite graph which is triangle-free.
    # So the largest triangle-free subgraph has size 12.
    
    best_score = _max_triangle_free_subgraph(n, edges)
    best_edges = edges[:]
    
    # Try to improve by removing edges
    # We want to reduce the largest triangle-free induced subgraph.
    
    # Random restarts with local search
    random.seed(42)
    
    for restart in range(50):
        if time.time() > deadline - 5.0:
            break
        
        # Start from best_edges or a random perturbation
        if restart == 0:
            edges = best_edges[:]
        else:
            edges = best_edges[:]
            # Remove a random edge
            if edges:
                idx = random.randint(0, len(edges) - 1)
                edges.pop(idx)
        
        current_score = _max_triangle_free_subgraph(n, edges)
        
        # Local search: try removing one edge at a time
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(len(edges)):
                if time.time() > deadline - 5.0:
                    break
                # Temporarily remove edge i
                edge_i = edges[i]
                edges[i] = edges[-1]
                edges.pop()
                
                # Check K4-free
                if _is_k4_free(n, edges):
                    new_score = _max_triangle_free_subgraph(n, edges)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_edges = edges[:]
                    else:
                        edges.append(edge_i)
                        edges[-1] = edges[i]
                        edges[i] = edge_i
                else:
                    edges.append(edge_i)
                    edges[-1] = edges[i]
                    edges[i] = edge_i
        
        if current_score < best_score:
            best_score = current_score
            best_edges = edges[:]
    
    return best_edges
# EVOLVE-BLOCK-END
