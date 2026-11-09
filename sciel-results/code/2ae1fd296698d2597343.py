"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit to be safe within 60s total
    
    # 1. Define the universe and weights
    # Vertices are 2..256
    # Weight of vertex v is 1/v
    max_val = 256
    vertices = list(range(2, max_val + 1))
    n = len(vertices)
    weights = {v: 1.0 / v for v in vertices}
    
    # Map vertex values to indices for array-based operations
    idx = {v: i for i, v in enumerate(vertices)}
    
    # 2. Build the graph for Maximum Weight Clique
    # We want a subset S such that lcm(a,b) > 256 for all distinct a,b in S.
    # This means in the "clique graph", an edge exists between a and b if lcm(a,b) > 256.
    # If lcm(a,b) <= 256, they conflict (no edge in clique graph).
    
    # Adjacency matrix or list of neighbors
    # Using a list of sets for neighbors in the clique graph
    neighbors = [set() for _ in range(n)]
    
    # Precompute lcm relationships
    # To speed up, we can iterate i < j
    for i in range(n):
        vi = vertices[i]
        for j in range(i + 1, n):
            vj = vertices[j]
            # Check if they are compatible (lcm > 256)
            # lcm(vi, vj) > 256
            # Note: lcm(a,b) = a*b // gcd(a,b)
            # If a*b is large, lcm is likely large.
            # Optimization: if vi * vj > 256 * gcd(vi, vj), then lcm > 256.
            # Actually, just computing lcm is fast enough for n=255.
            l = lcm(vi, vj)
            if l > 256:
                neighbors[i].add(j)
                neighbors[j].add(i)
                
    # 3. Order vertices by weight descending
    # This helps the greedy coloring bound to be tighter earlier in the search
    # We will remap indices or just sort the vertex list and rebuild neighbors?
    # Rebuilding is safer but O(N^2). N=255 is small.
    
    sorted_vertices = sorted(vertices, key=lambda x: -weights[x])
    new_idx = {v: i for i, v in enumerate(sorted_vertices)}
    
    # Rebuild neighbors based on sorted order
    new_neighbors = [set() for _ in range(n)]
    for i in range(n):
        vi = sorted_vertices[i]
        for j in range(i + 1, n):
            vj = sorted_vertices[j]
            if j in neighbors[idx[vi]]: # Check if original indices were connected
                # Wait, neighbors is built on original indices.
                # Let's just use the condition directly or map back.
                # Simpler: just check condition again or map.
                pass
            
    # Let's just rebuild from scratch with sorted order to avoid confusion
    new_neighbors = [set() for _ in range(n)]
    for i in range(n):
        vi = sorted_vertices[i]
        for j in range(i + 1, n):
            vj = sorted_vertices[j]
            if lcm(vi, vj) > 256:
                new_neighbors[i].add(j)
                new_neighbors[j].add(i)
                
    # 4. Bron-Kerbosch with Pivot and Greedy Coloring
    # State:
    # R: set of vertices in current clique
    # P: set of candidates that can be added to R
    # X: set of vertices excluded from consideration (for correctness, though for MWIS/Clique 
    #    with weights we usually just want the max weight, so X is less critical if we 
    #    don't need to enumerate all, but standard BK uses it. 
    #    Actually, for Max Weight Clique, we just need to find one.
    #    We track best_weight and best_clique.
    
    best_weight = 0.0
    best_clique = []
    
    # Convert neighbors to lists for faster iteration, sorted by weight?
    # Greedy coloring needs to order P.
    
    def greedy_coloring(P):
        """
        Assigns colors to vertices in P such that adjacent vertices have different colors.
        Returns a list of colors for vertices in P (in the order of P).
        The maximum color is an upper bound on the clique size/weight potential.
        """
        # We want to order P by weight descending to get tight bounds?
        # Standard coloring for clique:
        # Order vertices in P.
        # Assign smallest possible color.
        # The color of a vertex is an upper bound on the size of the clique it can form with R.
        # For weights, the sum of weights of vertices with color <= k is an upper bound.
        
        # To make the bound tight for weights, we should sort P by weight descending
        # before coloring, so high weight vertices get lower colors (smaller numbers)
        # and the bound "sum of weights of colors <= current" is tighter?
        # Actually, the bound is: if we are at vertex v with color c, 
        # the max weight we can add is sum(weights of u in P such that color(u) <= c).
        # If high weight vertices have small colors, this sum is large.
        # We want the bound to be small to prune.
        # So we want high weight vertices to have LARGE colors?
        # No, the bound is usually "max clique size <= max color".
        # For weights: UpperBound = sum(weights[u] for u in P if color[u] <= color[v]).
        # If we color such that high weight vertices get high colors, 
        # then when we process a low weight vertex with low color, the bound is small.
        # When we process a high weight vertex with high color, the bound is large.
        # This doesn't seem to help pruning early.
        
        # Alternative: Sort P by weight descending.
        # Then assign colors.
        # The bound for a vertex at position i (with color c_i) is sum(weights[j] for j in P if color[j] <= c_i).
        # If we process vertices in order of decreasing weight, 
        # and if the coloring respects the order (which it doesn't necessarily),
        # we can use the sum of remaining weights as a simple bound.
        
        # Let's stick to standard coloring but sort P by weight descending.
        # The color assigned is the "level".
        # The bound for a node v is the sum of weights of all nodes u in P with color(u) <= color(v).
        
        P_list = list(P)
        # Sort by weight descending
        P_list.sort(key=lambda x: -weights[sorted_vertices[x]])
        
        colors = {}
        # Color classes: list of sets of vertices
        color_classes = []
        
        for v in P_list:
            # Find smallest color c such that no neighbor of v has color c
            # Neighbors of v in P are neighbors[v] & P
            used_colors = set()
            for u in neighbors[v]:
                if u in P:
                    if u in colors:
                        used_colors.add(colors[u])
                        
            c = 1
            while c in used_colors:
                c += 1
                
            colors[v] = c
            if c > len(color_classes):
                color_classes.append(set())
            color_classes[c].add(v)
            
        return colors, color_classes, P_list

    def bron_kerbosch(R, P, X):
        nonlocal best_weight, best_clique
        
        if time.time() > deadline:
            return False # Timeout signal
            
        if not P and not X:
            # Found a maximal clique
            w = sum(weights[sorted_vertices[v]] for v in R)
            if w > best_weight:
                best_weight = w
                best_clique = list(R)
            return True
            
        # Greedy coloring to get upper bounds
        colors, color_classes, P_sorted = greedy_coloring(P)
        
        # Process vertices in reverse order of color (or weight?)
        # Standard BK processes in reverse color order to prune early?
        # Actually, we iterate through P.
        # If current weight + upper_bound < best_weight, prune.
        
        # Upper bound for a vertex v is sum(weights[u] for u in P if colors[u] <= colors[v])
        # We can precompute prefix sums of color classes?
        # Let's compute the bound on the fly or precompute for P_sorted.
        
        # Since P_sorted is sorted by weight, and colors are assigned based on that order,
        # the bound for the i-th vertex in P_sorted is sum(weights[P_sorted[k]] for k in range(i) if colors[P_sorted[k]] <= colors[P_sorted[i]])?
        # No, the standard bound is: 
        # Let P be ordered v1, ..., vk.
        # If we pick vi, the max weight we can get is weight(vi) + max_clique_weight(P \ N[vi]).
        # The coloring bound says max_clique_size(P) <= max_color.
        # For weights: max_clique_weight(P) <= sum(weights[u] for u in P if color(u) <= max_color_in_subgraph)?
        # A simpler bound: 
        # Iterate P in order of decreasing color.
        # If we are considering vertex v, the upper bound on the weight of any clique containing v 
        # is weight(v) + sum(weights[u] for u in P \ {v} if u is compatible with v and color(u) <= color(v))?
        # This is getting complicated.
        
        # Let's use a simpler bound: 
        # Sum of weights of all vertices in P.
        # If current_weight + sum(weights[P]) <= best_weight, prune.
        # This is weak.
        
        # Better bound using colors:
        # The vertices in P are colored.
        # Any clique in P can contain at most one vertex from each color class? No, that's independent set.
        # For clique, vertices in the SAME color class are NOT adjacent? 
        # Wait, greedy coloring for clique finding:
        # We color P such that adjacent vertices have DIFFERENT colors.
        # Therefore, a clique can contain at most one vertex from each color class? NO.
        # A clique can contain multiple vertices from different color classes.
        # A clique CANNOT contain two vertices of the SAME color.
        # So, the size of a clique is bounded by the number of color classes.
        # For weights: The max weight of a clique is bounded by the sum of the max weight in each color class?
        # Or simply, if we order vertices by color, the bound for vertex v_i (with color c_i) 
        # is sum(weights[v_j] for j <= i such that color(v_j) <= color(v_i))?
        
        # Let's use the standard optimization:
        # Order P by color ascending.
        # For each vertex v in P (in that order):
        #   if weight(R) + weight(v) + sum(weights[u] for u in P_remaining_after_v) < best_weight: break?
        
        # Let's implement the bound as:
        # Precompute suffix sums of weights in P_sorted?
        # If we process P in order of decreasing weight, 
        # and we have a coloring, 
        # the bound for vertex v is: weight(v) + sum(weights[u] for u in P \ {v} if u is neighbor of v).
        # This is too expensive.
        
        # Let's use the color bound:
        # Max weight clique in P <= sum(weights[u] for u in P if color(u) <= max_color_used_in_P)? No.
        
        # Actually, a very effective bound for MWIS/Clique is:
        # Order P by weight descending.
        # For each v in P:
        #   if weight(R) + weight(v) + sum(weights[u] for u in P \ {v} if u is compatible with v) < best: skip
        # But we don't have that sum precomputed.
        
        # Let's stick to the simplest effective bound:
        # If weight(R) + sum(weights[u] for u in P) <= best_weight: return
        
        sum_P_weights = sum(weights[sorted_vertices[u]] for u in P)
        current_weight = sum(weights[sorted_vertices[v]] for v in R)
        
        if current_weight + sum_P_weights <= best_weight:
            return True
            
        # Pivot selection
        # Choose u from P U X that maximizes |N(u) ∩ P|
        if not P:
            return True
            
        # For simplicity and speed with small N, we might skip pivot or use simple one
        # Pivot reduces branching.
        # Let's pick u from P that has most neighbors in P.
        max_degree = -1
        pivot = None
        for u in P:
            deg = len(neighbors[u] & P)
            if deg > max_degree:
                max_degree = deg
                pivot = u
                
        # Branch on vertices in P \ N(pivot)
        # If P is empty, pivot is None.
        candidates = P - neighbors[pivot] if pivot is not None else P
        
        # Sort candidates by weight descending for better early finds
        candidates_sorted = sorted(candidates, key=lambda x: -weights[sorted_vertices[x]])
        
        for v in candidates_sorted:
            # Check bound for this specific branch
            # Upper bound: weight(R) + weight(v) + sum(weights[u] for u in P \ {v} if u in neighbors[v])
            # Let's just use the global sum bound for now, it's O(N) per check.
            # Or we can use the color bound if implemented correctly.
            
            # Recurse
            R.add(v)
            P_new = P & neighbors[v]
            X_new = X & neighbors[v]
            
            # Optional: Prune if bound is low
            # sum_P_new = sum(weights[sorted_vertices[u]] for u in P_new)
            # if current_weight + weights[sorted_vertices[v]] + sum_P_new <= best_weight:
            #    R.remove(v)
            #    continue
            
            if not bron_kerbosch(R, P_new, X_new):
                R.remove(v)
                return False # Timeout propagated
                
            P.remove(v)
            X.add(v)
            R.remove(v)
            
    # Initialize
    P_initial = set(range(n))
    X_initial = set()
    R_initial = set()
    
    bron_kerbosch(R_initial, P_initial, X_initial)
    
    # Extract result
    if best_clique:
        result = [sorted_vertices[i] for i in best_clique]
    else:
        # Fallback to single best element
        result = [max(vertices, key=lambda x: weights[x])]
        
    return result

# EVOLVE-BLOCK-END
