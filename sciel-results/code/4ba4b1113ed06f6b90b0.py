import math
import time

def construct():
    N = 255
    candidates = [x for x in range(1, N + 1) if math.gcd(x, N) > 1]
    
    # Remove 255 from candidates to find max clique in the rest
    # Then add 255 back
    if 255 in candidates:
        candidates.remove(255)
        
    n = len(candidates)
    
    # Build adjacency list
    # adj[i] is a set of neighbors of i
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
                
    # Tomita Max Clique
    # We will use a recursive function with coloring
    
    # To speed up coloring, we can order vertices by degree
    # But Tomita works on the current candidate set P
    
    best_clique = []
    
    def color_sort(P):
        # Returns (ordered_P, colors)
        # Greedy coloring
        uncolored = list(P)
        colors = {}
        color_order = []
        # Simple greedy: assign smallest available color
        # This is O(|P|^2) which is fine for 126 nodes
        
        # Actually, for Tomita, we need to process vertices in reverse color order.
        # We can just sort by degree or use a specific heuristic.
        # A common approach is to color the subgraph induced by P.
        
        # Let's use a simple greedy coloring
        # Sort P by degree in P descending?
        # For now, just list(P)
        
        # We need to assign colors such that adjacent nodes have different colors.
        # The max color is an upper bound on clique size.
        
        # Implementation:
        # 1. Iterate through vertices in P.
        # 2. Assign the smallest color not used by its neighbors in P.
        
        # To make it efficient, we can process in a specific order.
        # Let's just use a basic greedy.
        
        # Map node -> color
        # We also need to group nodes by color for the expansion loop.
        
        # Let's return a list of lists: color_classes[k] contains nodes with color k+1
        # And the max color used.
        
        # Note: The bound is the max color.
        
        # However, Tomita's algorithm typically uses a "color bound" array.
        # Let's stick to a simpler recursive branch and bound if coloring is tricky to implement correctly quickly.
        # But 126 nodes might be too slow for simple branch and bound without coloring.
        
        # Let's try a simple recursive search with pruning based on degree?
        # No, coloring is key.
        
        # Let's implement a simple coloring:
        # 1. Order P by degree (descending).
        # 2. Assign colors greedily.
        
        ordered_P = sorted(P, key=lambda x: len(adj[x] & P), reverse=True)
        
        node_color = {}
        color_classes = {}
        max_color = 0
        
        for node in ordered_P:
            # Find available colors
            used_colors = set()
            for neighbor in adj[node]:
                if neighbor in P:
                    if neighbor in node_color:
                        used_colors.add(node_color[neighbor])
                        
            # Find smallest positive integer not in used_colors
            c = 1
            while c in used_colors:
                c += 1
                
            node_color[node] = c
            if c > max_color:
                max_color = c
                
            if c not in color_classes:
                color_classes[c] = []
            color_classes[c].append(node)
            
        # Return the list of nodes in reverse color order
        # Nodes with higher color are processed first?
        # In Tomita, we process nodes in reverse order of the color bound.
        # The bound for a node is its color.
        # We want to explore nodes with higher potential first.
        
        # The standard Tomita expansion:
        # 1. Color P.
        # 2. For each color class from max_color down to 1:
        #    For each node v in class:
        #       if |C| + color(v) < best_size: break (prune)
        #       expand(C + [v], P \cap N(v))
        #       P.remove(v)
        
        # Wait, the bound is usually the max color of the remaining nodes?
        # Or the color of the node itself?
        # In the standard algorithm, the color of a node v in the coloring of P is an upper bound on the size of a clique in P \cap N(v) plus 1?
        # Actually, the color of v is an upper bound on the size of the clique that can be formed by v and its neighbors in P.
        # So if |C| + color(v) <= best_size, we can prune.
        
        return ordered_P, node_color, color_classes, max_color

    def expand(C, P):
        nonlocal best_clique
        
        if not P:
            if len(C) > len(best_clique):
                best_clique = C[:]
            return
            
        # If |C| + |P| <= best_size, prune
        if len(C) + len(P) <= len(best_clique):
            return
            
        # Color P
        ordered_P, node_color, color_classes, max_color = color_sort(P)
        
        # Process in reverse color order
        # We need to iterate through nodes in the order they were added to color classes, reversed?
        # The bound for a node v is node_color[v].
        
        # We iterate through colors from max down to 1
        # And within each color, we iterate through nodes.
        # But we must remove nodes from P as we go.
        
        # To make this efficient, we can just iterate through the ordered_P list (which is sorted by degree, not color).
        # Actually, the coloring order matters.
        
        # Let's use the color_classes.
        # We iterate c from max_color down to 1.
        # For each v in color_classes[c]:
        #   if v not in P: continue (it might have been removed)
        #   if len(C) + node_color[v] <= len(best_clique): break (since node_color is non-increasing in this order? No, color classes are groups)
        #   Actually, if we process by color class, all nodes in class c have bound c.
        #   So if len(C) + c <= len(best_clique), we can stop processing this and all lower color classes.
            
        for c in range(max_color, 0, -1):
            if len(C) + c <= len(best_clique):
                break
                
            # Get nodes in this color class that are still in P
            # We need to iterate in a specific order.
            # Let's just iterate through the list in color_classes[c]
            
            # To ensure we don't process removed nodes, we check membership in P.
            # P is a set.
            
            # We need to remove v from P before recursing?
            # Yes, to avoid finding the same clique multiple times.
            
            # We iterate through a copy of the list because P changes
            for v in list(color_classes[c]):
                if v not in P:
                    continue
                    
                # Bound check
                if len(C) + node_color[v] <= len(best_clique):
                    # Since we are processing in descending color order, and within a class,
                    # do we need to break?
                    # If node_color[v] is the bound, and it's too small, we can prune v.
                    # But other nodes in the same class have the same color.
                    # So we can break the inner loop?
                    # Yes, if the bound is insufficient for this node, it's insufficient for all subsequent nodes in this class?
                    # Not necessarily, but the bound is the same.
                    # So we can break.
                    break
                    
                # Expand
                C.append(v)
                new_P = P & adj[v]
                expand(C, new_P)
                C.pop()
                
                # Remove v from P
                P.remove(v)
                
                # Update bound?
                # The coloring was done on the original P.
                # As P shrinks, the bound might be loose, but it's still valid.
                
    # Initial call
    # P is all nodes
    initial_P = set(range(n))
    expand([], initial_P)
    
    # best_clique contains indices
    clique_indices = best_clique
    clique_vals = [candidates[i] for i in clique_indices]
    
    # Add 255
    clique_vals.append(255)
    
    return clique_vals
