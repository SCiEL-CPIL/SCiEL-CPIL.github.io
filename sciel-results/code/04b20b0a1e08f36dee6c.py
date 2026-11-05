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
    nodes = list(range(1, N + 1))
    nodes.sort(key=lambda x: -weights[x])
    
    init_set = []
    for v in nodes:
        ok = True
        for u in init_set:
            if math.gcd(v, u) > 1:
                ok = False
                break
        if ok:
            init_set.append(v)
            
    # 3. Local Search Improvement
    # Try to swap elements to improve weight
    improved_set = list(init_set)
    improved_weight = sum(weights[v] for v in improved_set)
    
    # Simple 1-swap local search
    # For each u in set, for each v not in set, if v is coprime with set\{u}, and w(v) > w(u), swap
    # To make it faster, we can limit iterations or just do one pass.
    # Since N is small, one pass is cheap.
    changed = True
    while changed:
        changed = False
        for i, u in enumerate(improved_set):
            best_v = -1
            best_gain = 0
            # Check all v not in improved_set
            # We can iterate through all nodes 1..N
            for v in range(1, N + 1):
                if v in improved_set:
                    continue
                if weights[v] <= weights[u] + 1e-9:
                    continue
                # Check if v is coprime with all in improved_set except u
                ok = True
                for k, w_node in enumerate(improved_set):
                    if k == i:
                        continue
                    if math.gcd(v, w_node) > 1:
                        ok = False
                        break
                if ok:
                    gain = weights[v] - weights[u]
                    if gain > best_gain:
                        best_gain = gain
                        best_v = v
            if best_v != -1:
                improved_set[i] = best_v
                improved_weight += best_gain
                changed = True
    
    best_weight = improved_weight
    best_set = improved_set
    
    # 4. Exact Branch and Bound
    sys.setrecursionlimit(10000)
    
    ordered_nodes = nodes # Sorted by weight desc

    def get_candidates_from_mask(mask):
        # Optimization: iterate only through ordered_nodes
        return [v for v in ordered_nodes if (mask >> v) & 1]

    def color_bound(candidates):
        if not candidates:
            return 0.0
            
        color_of = {}
        max_weight_for_color = []
        colored_list = [] # List of (node, color) to speed up neighbor check
        
        for v in candidates:
            used_colors = set()
            # Check against already colored nodes
            for u, c in colored_list:
                # If v and u are adjacent (coprime), they must have different colors
                if (neighbors[v] >> u) & 1:
                    used_colors.add(c)
                    
            c = 0
            while c in used_colors:
                c += 1
                
            color_of[v] = c
            colored_list.append((v, c))
            
            if c >= len(max_weight_for_color):
                max_weight_for_color.append(weights[v])
            else:
                if weights[v] > max_weight_for_color[c]:
                    max_weight_for_color[c] = weights[v]
                    
        return sum(max_weight_for_color)

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
            
        # Simple bound: sum of weights of all candidates
        # This is O(K) and very fast.
        # If this bound is not enough, use color bound.
        # But color bound is tighter. Let's stick to color bound but it's optimized now.
        
        bound = current_weight + color_bound(candidates)
        if bound <= best_weight:
            return
            
        for v in candidates:
            new_P_mask = P_mask & neighbors[v]
            
            # Prune if even adding v and max possible from new_P is not enough?
            # The bound in the next call will do this.
            
            expand_helper(new_P_mask, current_weight + weights[v], current_set + [v])
            
            P_mask &= ~(1 << v)
            
            if time.time() > deadline:
                return

    full_mask = 0
    for i in range(1, N + 1):
        full_mask |= (1 << i)
        
    expand_helper(full_mask, 0.0, [])

    return best_set
