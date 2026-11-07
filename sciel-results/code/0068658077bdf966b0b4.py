"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import sys
sys.setrecursionlimit(10000)

P = {'n': 15}

def construct():
    n = P['n']
    
    # We'll search for the maximum edge count graph.
    # Key constraint: if deg(s) == deg(t), then dist(s,t) >= 4.
    # Equivalently: for any edge (u,v), and any s in N(u)\{v}, t in N(v)\{u}, s != t,
    # we need deg(s) != deg(t).
    
    # Approach: Build graph incrementally, tracking degrees and adjacency.
    # Use backtracking with pruning.
    
    # For efficiency, let's try a smarter approach:
    # 1. Decide on the degree sequence (partition of vertices by degree)
    # 2. Check if it's feasible (equal-degree vertices at distance >= 4)
    # 3. Among feasible degree sequences, find one that maximizes edges
    
    # Actually, let's do a direct search building the graph edge by edge,
    # but with strong pruning based on the constraint.
    
    # Better: Use the fact that the constraint is local.
    # Let's try to find a good graph by construction.
    
    # Insight: If we make the graph a complete multipartite graph with parts of size 1
    # (i.e., complete graph), then all vertices have degree 14, and any two vertices
    # are at distance 1, so the constraint is violated.
    
    # If we use a complete bipartite graph K_{7,8}, degrees are 8 and 7.
    # Vertices in same part have same degree. Distance between two vertices in same part
    # is 2 (via any vertex in other part). So dist = 2 < 4. Violation!
    
    # So we need equal-degree vertices to be at distance >= 4.
    
    # Let's try: partition into groups where within each group, vertices are far apart.
    
    # Strategy: Use a greedy/backtracking approach.
    # State: set of edges, degrees, adjacency lists.
    # When adding edge (u,v), check that for all s in N(u)\{v}, t in N(v)\{u}, s!=t:
    #   deg(s) != deg(t)
    # Also, after adding, the degrees of u and v increase by 1, which might create
    # new violations with other edges.
    
    # This is complex. Let's try a different approach:
    # Enumerate possible degree sequences and check feasibility.
    
    # For n=15, max degree is 14. Let's think about what degree sequences work.
    
    # If all vertices have distinct degrees, the constraint is automatically satisfied
    # (no two vertices have equal degree). But with 15 vertices and degrees 0..14,
    # we'd need degrees to be a permutation of 0..14. Sum = 105, so 52.5 edges - not integer.
    # So we can't have all distinct degrees.
    
    # We need at least one pair with equal degree. That pair must be at distance >= 4.
    
    # Let's try a construction:
    # Use a "blow-up" of a small graph where the base graph has the property that
    # equal-degree vertices are far apart.
    
    # Actually, let's just do a smart backtracking search.
    
    # State: 
    #   adj: list of sets, adj[i] = neighbors of i
    #   deg: list, deg[i] = len(adj[i])
    #   edges: set of (min(u,v), max(u,v))
    
    # We build the graph by considering each possible edge in some order.
    # When we add edge (u,v), we must verify the constraint for all paths s-u-v-t.
    # But also, adding the edge changes deg(u) and deg(v), which could invalidate
    # previously valid paths that now have s or t with changed degree.
    
    # This makes incremental building hard. Let's instead build the full graph and check.
    
    # Given the small size (15 vertices, max 105 edges), let's use a randomized/heuristic approach
    # with local search, or a systematic search over degree assignments.
    
    # Let me try a direct approach: search over graphs using backtracking on vertices.
    
    # For each vertex, decide its neighbors. But the constraint couples all vertices.
    
    # Alternative: Since n=15 is small, let's use a MILP-like approach with Python.
    # Actually, let's just do a clever construction.
    
    # Construction idea:
    # Take a cycle C15. Degrees all 2. Any two vertices at distance 2 on the cycle
    # are connected via a path of length 2, so a 3-edge path... wait, we need 3-edge paths.
    # In C15, a 3-edge path s-u-v-t has s and t at distance 3 on the cycle.
    # deg(s) = deg(t) = 2. Violation!
    
    # So we need to add chords to break up equal-degree pairs.
    
    # Let's try: start with a random graph and add edges while maintaining the constraint.
    
    import random
    random.seed(42)
    
    def is_valid(adj, deg):
        """Check if the graph satisfies the constraint."""
        n = len(adj)
        for u in range(n):
            for v in adj[u]:
                if u > v:
                    continue
                for s in adj[u]:
                    if s == v:
                        continue
                    for t in adj[v]:
                        if t == u or t == s:
                            continue
                        if deg[s] == deg[t]:
                            return False
        return True
    
    def try_build():
        n = 15
        adj = [set() for _ in range(n)]
        deg = [0] * n
        edges = []
        
        # Try to add edges in a greedy manner, prioritizing high-degree vertices
        # Order edges by some heuristic
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            # Try adding edge (u, v)
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
            
            if is_valid(adj, deg):
                edges.append((u, v))
            else:
                # Undo
                adj[u].remove(v)
                adj[v].remove(u)
                deg[u] -= 1
                deg[v] -= 1
        
        return edges, adj, deg
    
    # Try multiple random orderings
    best_edges = []
    best_count = 0
    best_adj = None
    best_deg = None
    
    for trial in range(200):
        random.seed(trial)
        edges, adj, deg = try_build()
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
            best_adj = adj
            best_deg = deg
            if best_count > 60:  # Good enough
                break
    
    # Now try to improve by adding more edges
    # Check which edges can still be added
    current_edges = set(best_edges)
    for u in range(15):
        for v in range(u+1, 15):
            if (u, v) not in current_edges:
                best_adj[u].add(v)
                best_adj[v].add(u)
                best_deg[u] += 1
                best_deg[v] += 1
                if is_valid(best_adj, best_deg):
                    best_edges.append((u, v))
                    current_edges.add((u, v))
                else:
                    best_adj[u].remove(v)
                    best_adj[v].remove(u)
                    best_deg[u] -= 1
                    best_deg[v] -= 1
    
    # Convert to list of [u, v]
    result = [[u, v] for u, v in best_edges]
    
    # Ensure types are Python int
    result = [[int(u), int(v)] for u, v in result]
    
    return result

# EVOLVE-BLOCK-END
