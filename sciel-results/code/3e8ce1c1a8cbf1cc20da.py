import random
import time
import sys

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 140  # 140s out of 150s
    
    # Target n
    n = 39
    
    # Function to check if adding edge (u, v) creates a C4
    # We maintain common neighbor counts
    # If common_neighbors[u][v] > 0, adding (u,v) creates a C4 (or multiple)
    # Wait, if common_neighbors[u][v] == k, adding (u,v) creates k triangles?
    # No, a C4 is u-w-v-z-u.
    # If u and v have a common neighbor w, and we add (u,v), we form a triangle u-v-w-u.
    # A C4 requires 4 distinct vertices.
    # If we add (u,v), and they share a neighbor w, we have a triangle.
    # Does a triangle contain a C4? No.
    # A C4 is formed if there exist w, z such that u-w, w-v, v-z, z-u are edges.
    # This is equivalent to u and v having a common neighbor w, AND v and u having a common neighbor z?
    # No.
    # Let's trace: Path u-w-v. If we add (u,v), we close a triangle.
    # A C4 is a cycle of length 4.
    # If u and v have a common neighbor w, and we add (u,v), we have a triangle.
    # To have a C4 involving the new edge (u,v), we need a path of length 3 from u to v that does not use the edge (u,v).
    # Path: u - w - x - v.
    # This requires u and v to have a common neighbor? No.
    # It requires that there is a vertex w adjacent to u, and a vertex x adjacent to v, such that w and x are adjacent?
    # No, the path is u-w-x-v.
    # So w is a neighbor of u, x is a neighbor of v, and (w,x) is an edge.
    # So adding (u,v) creates a C4 if and only if there exists an edge (w,x) such that w in N(u) and x in N(v).
    # This is NOT simply checking common neighbors of u and v.
    # Common neighbors of u and v are vertices w such that (u,w) and (v,w) are edges.
    # If w is a common neighbor, then u-w-v is a path of length 2.
    # Adding (u,v) makes it a triangle.
    # A triangle is not a C4.
    # So checking common neighbors is for **Triangles**.
    # The constraint is **No C4**.
    # So we can have triangles!
    # The seed program used triangles (disjoint), which is fine for C4-free.
    
    # So, my previous assumption that "common neighbors > 0 means C4" is WRONG.
    # Common neighbors > 0 means Triangle.
    # C4 is forbidden.
    # Condition for (u,v) to be addable:
    # There must be no path of length 3 between u and v.
    # i.e., for all w in N(u) and x in N(v), (w,x) must NOT be an edge.
    # Equivalently, N(u) and N(v) must be **independent** in the subgraph induced by N(u) U N(v)?
    # No, it means the bipartite graph between N(u) and N(v) must have no edges.
    # i.e., N(u) and N(v) must be disjoint? No, they can overlap (triangles), but if they overlap at w, then w is in N(u) and N(v).
    # If w is in N(u) and N(v), is (w,w) an edge? No.
    # So the condition is: For all w in N(u) and x in N(v), (w,x) is not an edge.
    # This implies that N(u) and N(v) cannot have any edges between them.
    # This is a strong condition.
    
    # Let's re-verify.
    # C4: u-v-w-z-u.
    # If we add (u,v), we need a path u-z-w-v of length 3.
    # So z is neighbor of u, w is neighbor of v, and (z,w) is an edge.
    # So yes, N(u) and N(v) must have no edges between them.
    
    # This makes the graph much sparser than if we only forbade triangles.
    
    # Algorithm for Greedy C4-Free Construction:
    # 1. Init empty graph.
    # 2. For each pair (u,v), check if N(u) and N(v) have any edge between them.
    #    This check is O(E) or O(n^2) depending on representation.
    #    To speed up, we can maintain a "conflict" matrix.
    #    When we add an edge (a,b), it creates conflicts for all pairs (u,v) such that:
    #    - u is neighbor of a, v is neighbor of b (and u!=b, v!=a, etc)
    #    - specifically, if we add (a,b), then for any u in N(a) and v in N(b), the pair (u,v) becomes "conflicted" (cannot be added).
    #    Also, if u in N(a) and v in N(a)? No.
    #    The conflict is: If (a,b) is added, then any pair (u,v) where u~a and v~b cannot be added (because u-a-b-v is a path of length 3).
    #    Also, if u~b and v~a, same thing.
    #    So adding (a,b) forbids all pairs (u,v) with u in N(a) \ {b} and v in N(b) \ {a}.
    
    # This is efficient enough for n=39.
    
    def is_valid_edge(u, v, adj, n):
        # Check if there is an edge between N(u) and N(v)
        for w in adj[u]:
            for x in adj[v]:
                if w == x: continue # Triangle is ok
                if w in adj[x]:
                    return False
        return True

    def construct_greedy(n, seed):
        random.seed(seed)
        adj = [set() for _ in range(n)]
        edges = []
        
        # Candidate edges
        candidates = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(candidates)
        
        # To make it "smarter", we can prioritize edges.
        # But simple random shuffle might work.
        
        for u, v in candidates:
            if u in adj[v]:
                continue
            # Check C4 constraint
            valid = True
            for w in adj[u]:
                for x in adj[v]:
                    if w == x:
                        continue
                    if x in adj[w]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                adj[u].add(v)
                adj[v].add(u)
                edges.append([u, v])
        
        return n, edges, adj

    def has_independent_set_11(n, adj):
        # Check if there is an independent set of size 11
        # Equivalent to clique of size 11 in complement
        # Use backtracking
        # adj is list of sets of neighbors in G
        # Complement neighbors: all i not in adj[u] and i != u
        
        # Precompute complement adjacency
        comp_adj = [set(range(n)) for _ in range(n)]
        for i in range(n):
            comp_adj[i].remove(i)
            for j in adj[i]:
                comp_adj[i].remove(j)
        
        # Bron-Kerbosch with pivoting for clique size 11
        # We just need to find ONE clique of size 11
        
        def expand(clique, available, excluded):
            if len(clique) == 11:
                return True
            if not available:
                return False
            # Pruning: if len(clique) + len(available) < 11, return False
            if len(clique) + len(available) < 11:
                return False
            
            # Pivoting: choose pivot p from available U excluded that maximizes |available ∩ N(p)|
            # For simplicity, just pick a random or first vertex in available as pivot?
            # Better: pick vertex in available with highest degree in induced subgraph of available
            pivot = None
            max_overlap = -1
            for p in available:
                overlap = len(available & comp_adj[p])
                if overlap > max_overlap:
                    max_overlap = overlap
                    pivot = p
            
            # Recurse on available \ N(pivot)
            # If pivot is None, recurse on all available
            candidates = available - comp_adj[pivot] if pivot is not None else available
            
            for v in list(candidates):
                if expand(clique + [v], available & comp_adj[v], excluded):
                    return True
                available.remove(v)
                excluded.add(v)
            return False
        
        return expand([], set(range(n)), set())

    best_graph = None
    best_alpha = 12 # We want < 11
    
    # Try multiple seeds
    for seed in range(50):
        if time.time() > deadline - 10:
            break
            
        n, edges, adj = construct_greedy(n, seed)
        
        # Check if valid
        if has_independent_set_11(n, adj):
            continue # Invalid, alpha >= 11
            
        # If we get here, alpha < 11
        # This is a valid graph
        return {'n': n, 'edges': edges}
        
    # If no valid graph found on 39 vertices, try 38?
    # The task says maximize n.
    # If 39 fails, we should return the best we have or a smaller n.
    # But the prompt says "best known construction has 39 vertices".
    # So we should be able to find it.
    
    # Fallback: return a smaller graph if 39 fails
    for n_target in [38, 37, 36]:
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            n, edges, adj = construct_greedy(n_target, seed)
            if not has_independent_set_11(n_target, adj):
                return {'n': n_target, 'edges': edges}
                
    # If all fails, return the seed (30 vertices)
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}
