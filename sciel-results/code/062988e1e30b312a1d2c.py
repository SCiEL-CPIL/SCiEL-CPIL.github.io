import time
import sys

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = 50.0 # seconds
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list and matrix
    adj = [set() for _ in range(n)]
    adj_matrix = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        adj_matrix[u][v] = True
        adj_matrix[v][u] = True
        
    # Greedy Coloring for bound
    # Returns list of (vertex, color) sorted by color
    def greedy_color(candidates):
        # candidates is a list of vertices
        # We want to color the subgraph induced by candidates
        # such that adjacent vertices have different colors.
        # The number of colors is an upper bound on clique size.
        # Tomita's algorithm orders vertices by decreasing degree in candidates? 
        # Or just processes them and assigns lowest possible color.
        
        # To get a tight bound, we should order vertices.
        # A common heuristic is to order by degree in the candidate set.
        
        # Let's implement a simple greedy coloring that returns the color of each vertex.
        # We need the maximum color assigned to be the bound.
        # Actually, we need to iterate in reverse order of colors for the search.
        
        # Step 1: Order candidates. Let's sort by degree in candidates descending.
        # This helps coloring be tighter? Or maybe just random? 
        # Tomita uses a specific ordering. Let's stick to a simple one first.
        
        # Actually, the standard MAXCLQ algorithm:
        # 1. Compute coloring of P.
        # 2. Sort P by color.
        # 3. Iterate P in reverse order (highest color first).
        
        # Let's compute colors for vertices in `candidates`.
        # color[v] = 1 + min(color[u] for u in neighbors of v in candidates if u already colored)
        # But we need to process in a specific order.
        
        # Let's use the "reorder" step from Tomita:
        # Reorder P by decreasing degree in P.
        # Then color greedily.
        
        # For n=24, we can just do this simply.
        
        # 1. Get degrees in candidate subgraph
        degs = {v: len(adj[v] & set(candidates)) for v in candidates}
        
        # 2. Sort candidates by decreasing degree
        sorted_candidates = sorted(candidates, key=lambda x: degs[x], reverse=True)
        
        # 3. Greedy coloring
        colors = {}
        used_colors = {} # color -> set of vertices? No, just need color of each v.
        
        # We assign colors 1, 2, ...
        # For each v in sorted_candidates:
        #    find smallest color c such that no neighbor of v in sorted_candidates[:index(v)] has color c
        #    Actually, neighbors must be already processed.
        
        processed = []
        for v in sorted_candidates:
            # Find neighbors of v that are in processed
            neigh_colors = set()
            for u in processed:
                if adj_matrix[v][u]:
                    neigh_colors.add(colors[u])
            
            # Find smallest positive integer not in neigh_colors
            c = 1
            while c in neigh_colors:
                c += 1
            colors[v] = c
            processed.append(v)
            
        # Now we have colors.
        # We want to return the list of vertices sorted by color, so we can iterate in reverse.
        # Actually, we return the list of (v, colors[v]) sorted by colors[v].
        colored = [(v, colors[v]) for v in candidates]
        colored.sort(key=lambda x: x[1])
        
        return colored

    best_clique = []
    
    def expand(clique, candidates):
        nonlocal best_clique
        
        if time.time() > deadline:
            return
            
        if len(clique) > len(best_clique):
            best_clique = clique[:]
            
        if not candidates:
            return
            
        # Get coloring
        colored = greedy_color(candidates)
        
        # Iterate in reverse order
        for i in range(len(colored) - 1, -1, -1):
            v, c = colored[i]
            
            # Bound: if len(clique) + c <= len(best_clique), we can prune.
            # Because v and any vertex with color <= c can form a clique of size at most c with v?
            # No, the bound is that the maximum clique in the subgraph induced by {colored[0..i]} has size at most c.
            # So if len(clique) + c <= len(best_clique), no better solution can be found in this branch.
            if len(clique) + c <= len(best_clique):
                break
                
            # Add v to clique
            new_clique = clique + [v]
            
            # New candidates: neighbors of v that are in candidates and have index < i in colored?
            # Actually, candidates for the next step must be adjacent to v and must be in the current `candidates`.
            # Also, to avoid duplicates, we only consider vertices that come before v in the ordering?
            # In Tomita's, P is updated to P \ {v} \cap N(v).
            # Since we are iterating in a specific order, we should only keep vertices that are in the "remaining" part of the list?
            # Actually, the standard recursion is:
            # expand(clique + [v], P \ {v} \cap N(v))
            # But we must ensure we don't re-explore permutations.
            # The coloring order helps with this if we restrict P to vertices that appear after v in the original order?
            # No, the standard algorithm removes v and its non-neighbors from P.
            # The order of P is fixed by the coloring step.
            # When we recurse, we re-color the new P.
            
            new_candidates = [u for u in candidates if u != v and adj_matrix[v][u]]
            
            expand(new_clique, new_candidates)
            
            # Remove v from candidates for the next iteration of the loop?
            # In the loop, we are effectively trying different v's as the next vertex.
            # The `candidates` list in the loop is static for this level.
            # But we must ensure that when we pick the next v, the subsequent candidates don't include v?
            # The logic `if len(clique) + c <= len(best_clique): break` handles the pruning.
            # The `new_candidates` calculation ensures disjointness.
            
    initial_candidates = list(range(n))
    expand([], initial_candidates)
    
    # If no clique found (shouldn't happen if n>=1), fallback
    if not best_clique:
         if n >= 2:
             best_clique = [0, 1] if adj_matrix[0][1] else [0, 2] # Find any edge
         elif n == 1:
             best_clique = [0]
    
    # Construct paths
    branches = list(best_clique)
    paths = []
    for i in range(len(branches)):
        for j in range(i + 1, len(branches)):
            u = branches[i]
            v = branches[j]
            # Since it's a clique, u and v are adjacent.
            # Path is [u, v]
            paths.append([u, v])
            
    return {'branches': branches, 'paths': paths}
