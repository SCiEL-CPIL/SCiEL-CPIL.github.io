import math
import time
import sys

def construct():
    N = 511
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # 1. Precompute weights and coprime relationships
    weights = [0.0] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = 1.0 / (512 - i)
        
    # neighbors[i] is a bitmask of j such that gcd(i, j) == 1
    neighbors = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if i == j:
                continue
            if math.gcd(i, j) == 1:
                mask |= (1 << j)
        neighbors[i] = mask

    # 2. Greedy Initialization
    # Sort by weight descending
    nodes = list(range(1, N + 1))
    nodes.sort(key=lambda x: -weights[x])
    
    init_set = []
    for v in nodes:
        # Check if v is coprime with all in init_set
        ok = True
        for u in init_set:
            if math.gcd(v, u) > 1:
                ok = False
                break
        if ok:
            init_set.append(v)
            
    best_weight = sum(weights[v] for v in init_set)
    best_set = init_set
    
    # 3. Exact Branch and Bound (Maximum Weight Clique in Coprime Graph)
    # State: P (bitmask of candidates), current_weight
    # We use a global best_weight for pruning
    
    # Helper to get list of set bits from a mask, sorted by weight descending
    # This is expensive if done repeatedly. 
    # Optimization: Pre-sort nodes by weight. 
    # When we have a mask P, we can iterate through the pre-sorted list and pick those in P.
    
    ordered_nodes = nodes # Already sorted by weight desc
    
    def get_candidates_from_mask(mask):
        return [v for v in ordered_nodes if (mask >> v) & 1]
        
    # Greedy Coloring of the Coprime Graph induced by candidates
    # Returns a list of max weights per color class
    def color_bound(candidates):
        # candidates is a list of node indices
        # We assign colors to these nodes based on edges in Coprime Graph
        # color[node] = color_id
        # We want to minimize the sum of max weights per color? 
        # Standard greedy coloring: assign smallest available color.
        # The bound is sum(max_weight[color])
        
        if not candidates:
            return 0.0
            
        color_of = {}
        # max_weight_for_color[c]
        max_weight_for_color = []
        
        # To make the bound tight, we should process nodes in an order that helps coloring.
        # Usually, processing high degree or high weight first helps.
        # We already have candidates sorted by weight desc.
        
        for v in candidates:
            # Find smallest color c such that v is NOT adjacent to any node with color c
            # Wait, in Clique problem, we color the Complement?
            # Recap:
            # We want Clique in G_coprime.
            # Bound: Partition P into Independent Sets in G_coprime (which are Cliques in G_conflict).
            # Each Independent Set in G_coprime can contribute at most 1 node to the Clique.
            # So we need to cover P with Independent Sets in G_coprime.
            # This is exactly Vertex Coloring of G_coprime.
            # If we color G_coprime with k colors, each color class is an Independent Set in G_coprime.
            # The max weight we can pick from a color class is the max weight of a node in that class.
            # So Bound = sum(max_weight[color_class]).
            
            # Greedy Coloring:
            # For node v, check its neighbors in G_coprime that are already colored.
            # If neighbor u has color c, then v cannot have color c.
            
            used_colors = set()
            # Iterate through already colored nodes?
            # This is O(N^2).
            # Optimization: Maintain for each color c, a bitmask of nodes with color c?
            # Or just check against the list of colored nodes.
            
            # Since N is small, O(N^2) is fine.
            for u in candidates:
                if u == v:
                    continue
                if u in color_of:
                    # Check if v and u are adjacent in G_coprime
                    # i.e., gcd(v, u) == 1
                    # We have neighbors[v] bitmask.
                    if (neighbors[v] >> u) & 1:
                        used_colors.add(color_of[u])
                        
            # Find smallest color not in used_colors
            c = 0
            while c in used_colors:
                c += 1
                
            color_of[v] = c
            
            # Update max weight for color c
            if c >= len(max_weight_for_color):
                max_weight_for_color.append(weights[v])
            else:
                if weights[v] > max_weight_for_color[c]:
                    max_weight_for_color[c] = weights[v]
                    
        return sum(max_weight_for_color)

    # Recursive B&B
    # To avoid recursion limit issues and overhead, we can use a stack or just standard recursion with sys.setrecursionlimit
    sys.setrecursionlimit(10000)
    
    def expand(P_mask, current_weight):
        nonlocal best_weight, best_set, deadline
        
        if time.time() > deadline:
            return
            
        if P_mask == 0:
            if current_weight > best_weight:
                best_weight = current_weight
                # We need to recover the set. 
                # This is tricky with just weight. 
                # We should pass the current set or store it.
                # Passing the set is O(N) copy. 
                # Alternatively, we can store the path.
                # Let's pass the current set as a list.
            return

        # Get candidates
        candidates = get_candidates_from_mask(P_mask)
        
        if not candidates:
             return

        # Calculate Bound
        bound = current_weight + color_bound(candidates)
        if bound <= best_weight:
            return
            
        # Iterate candidates
        # We iterate in the order of `candidates` (weight desc)
        for v in candidates:
            # Check if adding v can beat best
            # Simple prune: current_weight + weights[v] + (sum of weights of remaining? No, bound handles it)
            # Just check if current_weight + weights[v] <= best_weight? 
            # If yes, and all other weights are positive, we can't beat it? 
            # No, because we might add more. 
            # But if current_weight + weights[v] + color_bound(P_mask & neighbors[v]) <= best_weight, prune.
            
            new_P_mask = P_mask & neighbors[v]
            
            # Optional: Pre-check bound for new state
            # This might be expensive. Let's rely on the next call's bound.
            
            # Recurse
            # We need to track the set to update best_set.
            # To save time, we only update best_set if we find a better weight.
            # We can pass the current set.
            
            # To avoid passing large lists, we can use a global stack or just accept the copy cost for N=511.
            # Copying a list of ~10-20 items is cheap.
            
            # However, `expand` doesn't know the current set in this signature.
            # Let's modify `expand` to take `current_set`.
            
            expand_helper(new_P_mask, current_weight + weights[v], current_set + [v])
            
            # Remove v from P_mask for next iteration
            P_mask &= ~(1 << v)
            
            # Update bound? 
            # The bound calculated at the start of the loop was for the full P.
            # As we remove v, the bound decreases.
            # We can recalculate the bound for the remaining P?
            # Or just let the next recursive call handle it?
            # The next recursive call is for a specific branch.
            # The loop continues with the remaining P.
            # We should check if the remaining P can still beat best.
            # bound_remaining = current_weight + color_bound(get_candidates_from_mask(P_mask))
            # if bound_remaining <= best_weight: break
            
            if time.time() > deadline:
                return

    # Wrapper to manage state
    def expand_helper(P_mask, current_weight, current_set):
        nonlocal best_weight, best_set
        
        if time.time() > deadline:
            return
            
        if P_mask == 0:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = list(current_set)
            return
            
        candidates = get_candidates_from_mask(P_mask)
        if not candidates:
            return
            
        bound = current_weight + color_bound(candidates)
        if bound <= best_weight:
            return
            
        for v in candidates:
            new_P_mask = P_mask & neighbors[v]
            
            # Prune if even adding v and max possible from new_P is not enough?
            # The bound in the next call will do this.
            
            expand_helper(new_P_mask, current_weight + weights[v], current_set + [v])
            
            P_mask &= ~(1 << v)
            
            # Check if remaining candidates can beat best
            # Recalculating color_bound every iteration is expensive.
            # Heuristic: If current_weight + sum(weights of remaining) <= best_weight, break.
            # This is a weak bound but cheap.
            # Let's skip this optimization for now and rely on the recursive bound.
            
            if time.time() > deadline:
                return

    # Initial call
    # P_mask is all nodes
    full_mask = 0
    for i in range(1, N + 1):
        full_mask |= (1 << i)
        
    # We already have a good best_weight from greedy.
    # Start B&B
    expand_helper(full_mask, 0.0, [])

    return best_set
