"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    """
    Returns three vertex lists, one per instance, representing a minimal
    transversal of maximal cliques for the specified random graphs.
    """
    start_time = time.time()
    timeout = 115.0  # Leave 5s margin for 120s limit

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adjacency(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def max_clique_branch_and_bound(n, adj):
        """
        Finds the maximum clique in a graph using Bron-Kerbosch with pivot 
        and greedy coloring upper bound.
        Returns the list of vertices in the maximum clique found.
        """
        best_clique = []
        
        # Precompute degree for initial ordering if needed, but greedy coloring 
        # inside the recursion is standard for speed on small graphs.
        
        def expand(R, P, X):
            nonlocal best_clique
            
            if not P and not X:
                if len(R) > len(best_clique):
                    best_clique = R.copy()
                return
            
            if not P:
                return

            # Greedy coloring to get an upper bound on the size of the clique
            # that can be formed from P.
            # We order P by color class.
            # colors[i] is the color assigned to P[i]
            # We process P in reverse order of colors to prune early.
            
            # Simple greedy coloring
            colors = [0] * len(P)
            color_class = []
            
            # We need to be careful with the ordering. 
            # Standard approach:
            # 1. Color P greedily.
            # 2. Order P such that vertices with higher colors come first? 
            #    Actually, we want to prune. If we process vertices in order 
            #    of decreasing color, we can stop when the potential clique size
            #    (current R size + color of current vertex) is <= best_clique size.
            
            # Let's implement a standard greedy coloring for the upper bound.
            # We assign colors 1..k.
            # The number of colors k is an upper bound on the clique size in P.
            
            # We'll create a list of (vertex, color) and sort by color descending.
            # However, the standard Bron-Kerbosch with coloring usually modifies P 
            # to be ordered by color.
            
            # Simplified coloring for small N:
            # We just need a valid coloring to get an upper bound.
            # Let's use a simple greedy algorithm on the induced subgraph of P.
            
            # To make it fast, we can just use the degree or a simple greedy pass.
            # For N<=40, even O(N^2) coloring is fast.
            
            p_list = list(P)
            # Sort by degree in P? Or just arbitrary? 
            # Arbitrary is fine for correctness, sorting by degree might improve bound.
            # Let's sort by degree in the whole graph (static) or induced.
            # Induced degree:
            # deg_in_p = {v: len(adj[v] & P) for v in p_list}
            # p_list.sort(key=lambda v: deg_in_p[v], reverse=True)
            
            # Greedy coloring
            # color_map[v] = color
            color_map = {}
            next_color = 1
            # We want to assign minimum colors? No, we just need a valid coloring.
            # The number of colors used is the bound.
            # To get a *tight* bound, we should try to use few colors.
            # Greedy: for each v, assign the smallest color not used by neighbors in P.
            
            for v in p_list:
                # Find used colors by neighbors in P
                used_colors = set()
                for u in adj[v]:
                    if u in P and u in color_map:
                        used_colors.add(color_map[u])
                
                c = 1
                while c in used_colors:
                    c += 1
                color_map[v] = c
                if c > next_color:
                    next_color = c
            
            # Now, we process vertices.
            # We can prune if len(R) + color_map[v] <= len(best_clique)
            # But we must be careful: color_map[v] is the color of v.
            # The clique formed by v and its successors in the ordering must be 
            # within the color classes.
            
            # Let's order P by color descending to maximize pruning?
            # If we pick a vertex with high color, the remaining potential 
            # is bounded by that color?
            # Actually, the standard trick is:
            # Order P by color.
            # For each vertex v in that order:
            #   if len(R) + color(v) <= best_size: break (since subsequent vertices 
            #   have <= color(v), they can't form a larger clique)
            #   expand(R + [v], P' , X')
            
            # Let's create the ordered list
            # We need to map the colors back to the list P.
            # The coloring was done on p_list.
            # We sort p_list by color_map descending.
            p_list.sort(key=lambda v: color_map[v], reverse=True)
            
            # Now iterate
            for i, v in enumerate(p_list):
                # Upper bound check
                # The max clique size we can form starting with v is 
                # len(R) + 1 + (max clique in P' where P' are vertices in p_list[i+1:] 
                # that are neighbors of v).
                # But a simpler bound is: len(R) + color_map[v]
                # Why? Because all vertices in p_list[i+1:] have color <= color_map[v].
                # And they must be independent from each other? No.
                # But they are colored such that neighbors have different colors.
                # The clique containing v can include at most one vertex of each color 
                # from the remaining set? No, that's not right.
                # The bound is: The size of the clique in P is at most the number of colors.
                # If we fix v, the rest of the clique must be in N(v) ∩ P.
                # The coloring of P restricts the clique size of P to next_color.
                # So the total clique size is at most len(R) + next_color.
                # But we can do better with the specific order.
                
                # Standard Bron-Kerbosch with coloring:
                # If len(R) + color_map[v] <= len(best_clique):
                #     break
                # This works if p_list is sorted by color descending.
                
                if len(R) + color_map[v] <= len(best_clique):
                    break
                
                # New P and X
                # P' = P ∩ N(v)
                # X' = X ∩ N(v)
                # We must use the *remaining* vertices in p_list for P' to maintain order?
                # Yes, to avoid duplicates and respect the ordering for the bound.
                
                new_P = set()
                for u in p_list[i+1:]:
                    if u in adj[v]:
                        new_P.add(u)
                
                new_X = set()
                for u in X:
                    if u in adj[v]:
                        new_X.add(u)
                
                expand(R + [v], new_P, new_X)
        
        # Initial call
        # P = all vertices, X = empty
        # We can order initial P by degree to help?
        # The expand function handles coloring internally.
        expand([], set(range(n)), set())
        
        return best_clique

    results = []
    
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() - start_time > timeout:
            # Fallback: return all vertices (valid but not minimal)
            # Or break and return what we have?
            # The task requires 3 lists. If we timeout, we should return valid lists.
            # Let's just generate a trivial valid solution: all vertices.
            # But we need to process the remaining instances too.
            pass

        edges = instance(n, p, seed)
        adj = build_adjacency(n, edges)
        
        # Find Max Clique
        max_clique = max_clique_branch_and_bound(n, adj)
        
        # Transversal is V \ MaxClique
        clique_set = set(max_clique)
        transversal = sorted([v for v in range(n) if v not in clique_set])
        
        results.append(transversal)
        
        # Check time
        if time.time() - start_time > timeout:
             # If we ran out of time, fill remaining with trivial solutions
             # We need to know which instances are left
             # The loop is sequential. If we break, we need to handle the rest.
             # Let's just let the loop finish if it's fast, but this B&B is very fast for N=40.
             pass

    return results

# EVOLVE-BLOCK-END
