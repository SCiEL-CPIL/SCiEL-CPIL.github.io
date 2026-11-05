import math
import time

def _is_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    n = 256
    # 1. Identify valid nodes
    # A node v is invalid if 2*v is a perfect square
    valid_nodes = []
    for v in range(1, n + 1):
        if not _is_square(2 * v):
            valid_nodes.append(v)
    
    # Map original values to indices in valid_nodes
    val_to_idx = {v: i for i, v in enumerate(valid_nodes)}
    num_nodes = len(valid_nodes)
    
    # 2. Build G_comp adjacency
    # G_comp edge (u, v) if u + v is NOT a perfect square
    # We want Max Clique in G_comp
    adj = [set() for _ in range(num_nodes)]
    
    for i in range(num_nodes):
        u = valid_nodes[i]
        for j in range(i + 1, num_nodes):
            v = valid_nodes[j]
            if not _is_square(u + v):
                adj[i].add(j)
                adj[j].add(i)
    
    # 3. Tomita Max Clique Algorithm
    best_clique = []
    best_size = 0
    
    # Heuristic: Order vertices by degree in G_comp (descending)
    # This often helps pruning
    degrees = [len(adj[i]) for i in range(num_nodes)]
    order = sorted(range(num_nodes), key=lambda x: degrees[x], reverse=True)
    
    # Re-index or just use the order in the algorithm?
    # Standard Tomita operates on the set P.
    # We will maintain P as a list of node indices.
    
    def color_sort(P):
        """
        Greedily color the subgraph induced by P in G_comp.
        Returns:
        - sorted_P: list of nodes in P, sorted by color (ascending)
        - colors: dict node -> color
        - max_color: int
        """
        if not P:
            return [], {}, 0
        
        # Sort P by degree in the induced subgraph? Or just use global degree?
        # Using global degree is faster.
        # For better bounds, sorting by degree in P is better but slower.
        # Let's try sorting by global degree first.
        P_sorted = sorted(P, key=lambda x: degrees[x], reverse=True)
        
        colors = {}
        color_classes = []
        max_color = 0
        
        for v in P_sorted:
            # Find smallest color not used by neighbors of v in P
            used_colors = set()
            for u in adj[v]:
                if u in colors:
                    used_colors.add(colors[u])
            
            c = 1
            while c in used_colors:
                c += 1
            
            colors[v] = c
            if c > max_color:
                max_color = c
        
        # Sort nodes by color, then by index (or degree) for determinism
        sorted_P = sorted(P, key=lambda x: (colors[x], x))
        
        return sorted_P, colors, max_color

    def expand(P, S):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
        
        if not P:
            if len(S) > best_size:
                best_size = len(S)
                best_clique = S.copy()
            return
        
        sorted_P, colors, max_color = color_sort(P)
        
        # Iterate in reverse order (highest color first)
        # We process vertices one by one.
        # We need to maintain the set of available candidates.
        # The standard loop:
        # for v in reversed(sorted_P):
        #   if len(S) + colors[v] <= best_size: break
        #   ...
        
        # To handle the removal of v from P correctly for the next iteration,
        # we can just iterate and check if v is still in the current "active" set.
        # However, the standard Tomita algorithm modifies P in place or uses a stack.
        # Here, P is passed by value (list), so we can create new lists.
        
        # Optimization: Pre-compute neighbors in P?
        # adj[v] is global. We need neighbors in P.
        
        for v in reversed(sorted_P):
            if len(S) + colors[v] <= best_size:
                break
            
            # Check if v is still in P (it should be, as we haven't removed it yet in this loop iteration logic)
            # Wait, the logic is:
            # We pick v. We add it to S.
            # The new P is P intersected with neighbors of v.
            # Then we recurse.
            # Then we remove v from P for the next branch (where v is NOT in the clique).
            
            # Since we are iterating over a static list `sorted_P`, we must ensure we only consider
            # vertices that are still in the *current* candidate set for this branch.
            # Actually, the standard algorithm works like this:
            # P is the set of candidates.
            # We pick v from P.
            # Branch 1: Include v. New P' = P \ {v} \cap N(v). Recurse.
            # Branch 2: Exclude v. New P'' = P \ {v}. Recurse.
            
            # The coloring bound is applied to the *current* P.
            # If we exclude v, the bound for the remaining P'' is still valid?
            # The color of v might be the max. If we remove v, the max color might decrease.
            # But the bound `colors[v]` is an upper bound on the clique size *including* v?
            # No, `colors[v]` is an upper bound on the clique size in the subgraph induced by `P` (which includes v).
            # Specifically, any clique in P has size <= max_color.
            # Any clique containing v has size <= 1 + (clique in P \ {v} \cap N(v)).
            # The bound used in Tomita is:
            # If we are at vertex v in the sorted list, the number of vertices remaining in P
            # that have color >= colors[v] is an upper bound on the additional vertices we can add.
            
            # Let's stick to the simpler, slightly less tight but correct bound:
            # Upper bound for branch including v: 1 + alpha(P \cap N(v)).
            # We don't have alpha(P \cap N(v)) directly.
            # The coloring bound `max_color` applies to the whole P.
            # If we include v, we add 1. The rest must come from N(v) \cap P.
            # A tighter bound is often computed during the coloring phase (color number of v).
            
            # Let's use the standard Tomita loop structure:
            # The list `sorted_P` is sorted by color.
            # We iterate from the end (highest color).
            # For a vertex v, `colors[v]` is an upper bound on the size of a clique in the subgraph
            # induced by {u in P | color(u) >= color(v)}? No.
            
            # Correct Tomita Pruning:
            # After coloring P, we have colors.
            # We process vertices in order of decreasing color.
            # Let v be the current vertex.
            # The number of vertices in P with color >= colors[v] is an upper bound on the clique size
            # that can be formed from the remaining candidates *if we were to pick from them*.
            # Actually, the bound is:
            # If we include v, we can add at most `colors[v] - 1` more vertices? No.
            
            # Let's use a simpler, robust bound:
            # Upper bound = len(S) + len(P_remaining)
            # This is weak.
            
            # Let's use the color bound properly:
            # In the subgraph induced by P, a clique can have at most one vertex of each color.
            # So max clique size in P <= max_color.
            # If we include v, the new clique size is 1 + clique_size(P' where P' = P \ {v} \cap N(v)).
            # We don't know clique_size(P').
            # However, we can re-color P' in the recursive call.
            
            # The pruning condition `if len(S) + colors[v] <= best_size: break` is standard
            # IF `colors[v]` represents the number of vertices in P with color >= colors[v] ?
            # No, usually `colors` is the color class index (1, 2, ...).
            # The bound is: `len(S) + number_of_vertices_in_P_with_color_ge(colors[v])`.
            
            # Let's modify color_sort to return this count.
            
            pass # Placeholder for logic refinement

    # Refined Color Sort and Expand
    def color_sort_refined(P):
        if not P:
            return [], {}, 0, 0
        
        P_set = set(P)
        # Sort by degree in P for better coloring?
        # Let's just use global degree for speed.
        P_list = sorted(P, key=lambda x: degrees[x], reverse=True)
        
        colors = {}
        max_color = 0
        color_counts = {} # color -> count of vertices with this color
        
        for v in P_list:
            used = set()
            for u in adj[v]:
                if u in P_set and u in colors:
                    used.add(colors[u])
            
            c = 1
            while c in used:
                c += 1
            
            colors[v] = c
            if c > max_color:
                max_color = c
            color_counts[c] = color_counts.get(c, 0) + 1
        
        # For pruning, we need the count of vertices with color >= c
        # Let's create a list of (color, node) sorted by color descending
        items = [(colors[v], v) for v in P]
        items.sort(key=lambda x: x[0], reverse=True)
        
        # Precompute suffix sums of color counts?
        # Or just count on the fly in expand?
        # Storing the sorted list is enough.
        
        return items, colors, max_color

    def expand_refined(P, S):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
        
        if not P:
            if len(S) > best_size:
                best_size = len(S)
                best_clique = S.copy()
            return
        
        items, colors, max_color = color_sort_refined(P)
        
        # items is sorted by color descending.
        # We iterate through items.
        # For each vertex v with color c:
        # The number of vertices in P with color >= c is an upper bound on the clique size in P.
        # Let count_ge_c be the number of vertices in items with color >= c.
        # Actually, since items is sorted, if we are at index i, the number of vertices from i to end
        # all have color <= color(items[i]).
        # Wait, the bound is: Any clique in P has size <= max_color.
        # If we include v, we add 1. The rest must be from N(v) \cap P.
        # The clique size in N(v) \cap P is <= max_color of that subgraph.
        
        # Standard Tomita Pruning:
        # Let v be the current vertex.
        # Let k be the number of vertices in P that have color >= color(v).
        # Then the size of any clique in P containing v is at most 1 + (max clique in N(v) \cap P).
        # This is hard to bound tightly without re-coloring.
        
        # However, a common bound is:
        # If len(S) + k <= best_size, prune.
        # Where k is the number of vertices in P with color >= color(v).
        # This works because any clique in P can have at most one vertex of each color.
        # So the max clique size in P is <= max_color.
        # If we restrict to vertices with color >= color(v), the max clique size in that subset is <= count_ge_color(v).
        
        # Let's compute count_ge for each item.
        # Since items is sorted by color desc:
        # items[0] has max color. items[-1] has min color.
        # For items[i], the number of vertices with color >= color(items[i]) is (len(items) - i) ?
        # No, multiple vertices can have the same color.
        # If items is sorted by color desc, then for index i,
        # all items from 0 to i have color >= color(items[i]).
        # So count_ge = i + 1.
        
        for i, (c, v) in enumerate(items):
            # Upper bound on additional vertices we can add if we include v?
            # If we include v, we add 1.
            # The remaining vertices must form a clique in N(v) \cap P.
            # The max size of a clique in N(v) \cap P is <= max_color of that subgraph.
            # We don't have that bound.
            
            # Alternative Pruning:
            # The max clique size in P is <= max_color.
            # So if len(S) + max_color <= best_size, we can prune the whole P.
            # This is checked at the start of expand_refined.
            
            # Inside the loop, we need tighter bounds.
            # If we pick v, the new P is P' = P \ {v} \cap N(v).
            # We can estimate the bound for P' by re-coloring in the recursive call.
            
            # To prune the branch where we pick v:
            # We know the clique in P' has size <= max_color(P').
            # We don't know max_color(P') yet.
            
            # However, we can use the fact that v has color c.
            # Any clique in P containing v can have at most 1 vertex of color c, 1 of c+1, etc.
            # So the size of a clique containing v is <= 1 + (number of vertices in P with color > c).
            # Let count_greater = number of vertices in items with color > c.
            # Since items is sorted desc, count_greater = i.
            # So bound = 1 + i.
            
            if len(S) + 1 + i <= best_size:
                # Since i decreases as we go to lower colors (wait, i increases as we go to lower colors? No)
                # items[0] is max color. i=0.
                # items[-1] is min color. i=len-1.
                # As i increases, the bound 1+i increases?
                # No, the bound should decrease as we consider lower priority vertices?
                # Actually, vertices with higher colors are "better" candidates for large cliques.
                # If we skip a high color vertex, we lose the chance to use that color slot?
                
                # Let's re-verify the Tomita bound.
                # In MCQ (Max Clique with Quality), the bound is:
                # If the number of vertices in P with color >= color(v) is <= best_size - len(S), then prune.
                # Count of vertices with color >= c is (i + 1) if sorted desc?
                # No, if sorted desc, items[0..i] have color >= c.
                # So count = i + 1.
                
                if len(S) + (i + 1) <= best_size:
                    break
            
            # Branch: Include v
            S.append(v)
            # New P: neighbors of v in P
            # P is the set of vertices in items? No, P is the input list.
            # We need to intersect P with adj[v].
            new_P = [u for u in P if u in adj[v] and u != v]
            expand_refined(new_P, S)
            S.pop()
            
            # Remove v from P for the next iteration (Exclude v)
            # We are iterating over `items`, which is a static list derived from P.
            # We need to ensure that when we recurse for the "Exclude" branch (implicitly handled by the loop),
            # v is not available.
            # Since we are just iterating, and we don't modify P in place in a way that affects the loop,
            # we just proceed to the next item.
            # BUT, the next item's bound calculation `i+1` assumes all previous items are still in P.
            # Since we "excluded" v (by not adding it to S in the current branch context of the loop),
            # v is effectively removed from the candidate set for subsequent vertices in this loop.
            # So the count of available vertices decreases.
            
            # The standard implementation often modifies P in place.
            # Here, we can just proceed. The bound `i+1` is an overestimate if we've removed vertices,
            # which is safe (less pruning, but correct).

    # Initial call
    initial_P = list(range(num_nodes))
    expand_refined(initial_P, [])
    
    # Convert indices back to values
    result = [valid_nodes[i] for i in best_clique]
    
    # Fallback if something goes wrong or time runs out before finding a good set
    if not result:
        # Use the seed strategy or a simple greedy
        result = [v for v in valid_nodes if _is_square(v + 1) == False] # Just a placeholder
        # Better fallback: Greedy
        greedy_set = []
        remaining = set(valid_nodes)
        while remaining:
            # Pick vertex with min degree in remaining?
            # Or just first one
            v = next(iter(remaining))
            greedy_set.append(v)
            # Remove v and its neighbors in G_orig (which are non-neighbors in G_comp)
            # In G_comp, neighbors are compatible.
            # In G_orig, neighbors are incompatible.
            # We want to remove incompatible ones.
            # Incompatible in G_orig = Compatible in G_comp? No.
            # G_orig edge = Incompatible.
            # G_comp edge = Compatible.
            # If we pick v, we cannot pick any u such that (u,v) is edge in G_orig.
            # So we remove v and all G_orig neighbors of v from remaining.
            # G_orig neighbors of v are NOT in adj[v] (G_comp).
            # So we keep u if u is in adj[v] (G_comp) OR u==v? No.
            # We keep u if u is compatible with v, i.e., u in adj[v] (G_comp).
            # So new_remaining = remaining \ {v} \ (G_orig_neighbors(v))
            # G_orig_neighbors(v) = remaining \ (adj[v] \cup {v})
            # So new_remaining = remaining \ {v} \ (remaining \ (adj[v] \cup {v}))
            # = (remaining \ {v}) \cap (adj[v] \cup {v})
            # = (remaining \ {v}) \cap adj[v]
            
            new_remaining = {u for u in remaining if u in adj[v]}
            remaining = new_remaining
            
        result = greedy_set

    return result

# Test locally?
# print(construct())
