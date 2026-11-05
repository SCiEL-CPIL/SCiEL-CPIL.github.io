import time
import sys
from math import gcd, lcm
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Leave 10s margin
    
    N = 128
    vertices = list(range(1, N + 1))
    
    # 1. Identify all forbidden triples
    # A triple (a, b, c) is forbidden if lcm(a,b) == lcm(a,c) == lcm(b,c)
    # We store the shadow graph: u-v edge if (u,v,w) is forbidden for some w
    # We also store the hyperedges for validation
    shadow_adj = [set() for _ in range(N + 1)]
    forbidden_triples = set()
    
    # Precompute LCMs for all pairs to speed up triple check
    # lcm[a][b]
    # N=128, 128*128/2 = 8128 pairs. Fast.
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            lcm_table[i][j] = lcm_table[j][i] = i * j // gcd(i, j)
            
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                if time.time() > deadline:
                    break
                if (lcm_table[a][b] == lcm_table[a][c] == lcm_table[b][c]):
                    forbidden_triples.add((a, b, c))
                    # Add edges to shadow graph
                    shadow_adj[a].add(b)
                    shadow_adj[b].add(a)
                    shadow_adj[a].add(c)
                    shadow_adj[c].add(a)
                    shadow_adj[b].add(c)
                    shadow_adj[c].add(b)
        if time.time() > deadline:
            break
            
    # 2. Solve MIS on the shadow graph
    # Using Branch and Bound with Greedy Coloring Upper Bound
    # Vertices 1..N
    
    # Order vertices by degree (descending) for better coloring/bounding?
    # Or just use 1..N.
    
    def greedy_color(graph, vertices_subset):
        """
        Assigns colors to vertices in vertices_subset such that adjacent vertices have different colors.
        Returns a list of colors for the vertices in vertices_subset.
        The number of colors used is an upper bound on the size of the MIS in the induced subgraph.
        """
        # We use a simple greedy coloring on the induced subgraph
        # To make it effective for B&B, we should color the candidates remaining in the search.
        # Standard Tomita algorithm colors the candidates P.
        
        colors = {}
        color_map = {}
        # Sort vertices by degree in the induced subgraph? 
        # For simplicity and speed, we just iterate.
        # A better heuristic: order by degree in the remaining graph.
        
        # Let's implement a simple greedy coloring that returns the max color used.
        # We need the coloring of the *current candidate set* P.
        
        # To speed up, we can just count the number of colors needed.
        # We don't strictly need the color assignment for every node, just the bound.
        # But Tomita's algorithm uses the colors to branch.
        
        # Simplified: Just return an upper bound.
        # A very loose bound is |P|.
        # A tighter bound is the chromatic number of G[P].
        
        # Let's implement Tomita's greedy coloring which returns the color of each vertex in P.
        # This allows branching on vertices with the highest color.
        
        # We map vertices to 0..k-1
        # We'll return a dict {v: color}
        
        # Sort P by degree in G[P] descending to improve coloring quality?
        # Actually, for the bound, we just need a valid coloring.
        # The number of colors is the bound.
        
        p_list = list(vertices_subset)
        # Heuristic: sort by degree in the induced subgraph
        degs = {}
        for u in p_list:
            degs[u] = len(shadow_adj[u] & vertices_subset)
        p_list.sort(key=lambda x: degs[x], reverse=True)
        
        colors = {}
        for v in p_list:
            # Find smallest color not used by neighbors in P
            used_colors = set()
            for u in shadow_adj[v]:
                if u in colors:
                    used_colors.add(colors[u])
            c = 0
            while c in used_colors:
                c += 1
            colors[v] = c
            
        max_color = max(colors.values()) + 1 if colors else 0
        return colors, max_color

    def branch_and_bound(P, current_set):
        """
        P: set of candidate vertices
        current_set: set of vertices already in the MIS
        """
        nonlocal best_mis, best_size
        
        if not P:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_mis = set(current_set)
            return
            
        # Upper bound: |current_set| + |P|
        if len(current_set) + len(P) <= best_size:
            return
            
        # Greedy coloring of G[P]
        colors, max_color = greedy_color(shadow_adj, P)
        
        # If |current| + max_color <= best, prune
        if len(current_set) + max_color <= best_size:
            return
            
        # Branch
        # Tomita's algorithm branches on vertices with the highest color
        # We group vertices by color
        # We iterate colors from max_color down to 1
        # For each color c, we branch on vertices v in P with color c
        
        # To avoid deep recursion stack issues, we use a stack or careful recursion.
        # N=128, depth 128 is fine.
        
        # Get vertices with max_color
        # We can just pick one vertex from the highest color class to branch on?
        # No, Tomita's algorithm:
        # Let P_k be vertices with color k.
        # For k from max_color down to 1:
        #   For each v in P_k:
        #      branch_and_bound(P \ (N(v) U {v}), current_set U {v})
        #      P = P \ {v}
        #   P = P \ P_k  (remove all vertices of color k? No, the loop structure handles it)
        
        # Standard implementation:
        # We maintain P as a set.
        # We color P.
        # We find the max color K.
        # We consider vertices v in P with color K.
        # For each such v:
        #    New_P = P \ (N(v) U {v})
        #    branch_and_bound(New_P, current_set U {v})
        #    P.remove(v)
        # After processing all v with color K, we remove all remaining vertices with color K from P?
        # Actually, the standard Tomita loop is:
        # while P is not empty:
        #   Color P
        #   Let K be max color
        #   Let P_K be vertices in P with color K
        #   For v in P_K:
        #       branch_and_bound(P \ (N(v) U {v}), S U {v})
        #       P.remove(v)
        #   # Note: P_K is fixed at the start of the loop iteration?
        #   # No, P changes inside.
        
        # Let's use a simpler recursive B&B that picks a vertex with high degree.
        # It might be slower but easier to implement correctly.
        # However, the coloring bound is the key to speed.
        
        # Let's stick to the Tomita structure but carefully.
        
        # We need to iterate over vertices in P that have the maximum color.
        # We'll get the list of such vertices.
        max_c = max_color - 1 # 0-indexed
        candidates = [v for v in P if colors[v] == max_c]
        
        # We must be careful: if we remove v from P, the coloring of the remaining P might change.
        # But Tomita's algorithm re-colors P in each recursive call.
        # So we just pass the new P.
        
        for v in candidates:
            if time.time() > deadline:
                return
            # New candidate set: P minus v and its neighbors
            new_P = set(P)
            new_P.discard(v)
            for u in shadow_adj[v]:
                new_P.discard(u)
                
            branch_and_bound(new_P, current_set | {v})
            
            # Remove v from P for the next iteration in this level?
            # In Tomita's algorithm, we remove v from P.
            P.discard(v)
            
            # Pruning: if len(current_set) + len(P) <= best_size, we can break?
            # Yes, because any solution from further iterations will be <= len(current_set) + len(P).
            if len(current_set) + len(P) <= best_size:
                break

    best_mis = set()
    best_size = 0
    
    initial_P = set(vertices)
    branch_and_bound(initial_P, set())
    
    # 3. Post-processing: Try to add vertices not in best_mis
    # The MIS of the shadow graph is a valid solution for the hypergraph problem.
    # Can we add more vertices?
    # If we add v, we must ensure no forbidden triple is formed with existing vertices.
    # A forbidden triple involving v would be {v, u, w} where u,w are in the set.
    # This implies {v,u,w} is a forbidden triple.
    # In the shadow graph, v is connected to u and w.
    # So if we add v, we must remove any u,w that form a forbidden triple with v.
    # This is a local search / repair strategy.
    
    current_solution = set(best_mis)
    remaining = set(vertices) - current_solution
    
    # Sort remaining by "safety" or degree?
    # Try adding vertices one by one.
    # If adding v creates a conflict, we might need to remove others.
    # Simple greedy add:
    # Check if v can be added to current_solution.
    # v can be added if there is no {v, u, w} in forbidden_triples with u,w in current_solution.
    
    # Precompute for each v, the set of pairs (u,w) in forbidden triples involving v.
    # conflict_pairs[v] = set of frozensets({u,w})
    
    conflict_pairs = {v: set() for v in vertices}
    for (a, b, c) in forbidden_triples:
        conflict_pairs[a].add(frozenset((b, c)))
        conflict_pairs[b].add(frozenset((a, c)))
        conflict_pairs[c].add(frozenset((a, b)))
        
    # Heuristic: try adding vertices with fewest conflicts first?
    # Or just iterate.
    
    # We can try multiple passes.
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Sort remaining by number of potential conflicts with current_solution
        # For v in remaining:
        #   count = number of pairs in conflict_pairs[v] that are subsets of current_solution
        #   if count == 0, v can be added safely.
        
        safe_vertices = []
        for v in remaining:
            can_add = True
            for pair in conflict_pairs[v]:
                if pair <= current_solution: # pair is subset of current_solution
                    can_add = False
                    break
            if can_add:
                safe_vertices.append(v)
                
        if safe_vertices:
            # Add the one with smallest degree in remaining? Or just the first one?
            # Let's add the one that minimizes future conflicts.
            # For simplicity, add the first one found or the one with min degree in shadow graph.
            # Actually, adding ANY safe vertex increases the size.
            # We should add as many as possible.
            # Greedy: pick v that allows the most other vertices to be added?
            # Too complex. Just pick one.
            
            # Pick v with min degree in the subgraph induced by remaining?
            # Or just pick v with min len(conflict_pairs[v])
            v = min(safe_vertices, key=lambda x: len(conflict_pairs[x]))
            
            current_solution.add(v)
            remaining.remove(v)
            changed = True
        else:
            # No single vertex can be added.
            # Try swapping?
            # If we remove u from current_solution, can we add v?
            # This is complex.
            # Let's just break if no safe vertex is found.
            break
            
    return list(current_solution)

# Test
if __name__ == "__main__":
    res = construct()
    print(len(res))
