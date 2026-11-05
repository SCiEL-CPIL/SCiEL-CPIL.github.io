import math
import time

def construct():
    N = 512
    start_time = time.time()
    time_limit = 85.0  # Leave 5s margin for 90s limit

    # 1. Precompute LCM conflict graph
    # adj[i] is a bitmask of j such that lcm(i, j) > N (conflict)
    adj = [0] * (N + 1)
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = math.gcd(i, j)
            l = i * j // g
            if l > N:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # comp_adj[i] = bitmask of vertices compatible with i (excluding i itself)
    full_mask = (1 << (N + 1)) - 2
    comp_adj = [0] * (N + 1)
    for i in range(1, N + 1):
        comp_adj[i] = full_mask & ~adj[i] & ~(1 << i)

    def get_divisors(n):
        divs = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        return divs

    def greedy_color_bound(vertices, color_map):
        """
        Greedy coloring upper bound for max independent set.
        vertices: list of vertex indices
        color_map: dict vertex -> color number
        Returns number of colors used (upper bound on independent set size in subgraph).
        """
        if not vertices:
            return 0
        # Order by degree in subgraph (descending) for tighter bound
        sub_adj = {v: 0 for v in vertices}
        vset = set(vertices)
        for v in vertices:
            neighbors_in_sub = 0
            for u in vertices:
                if u != v and (adj[v] >> u) & 1:
                    neighbors_in_sub += 1
            sub_adj[v] = neighbors_in_sub
        
        ordered = sorted(vertices, key=lambda v: -sub_adj[v])
        color_map.clear()
        next_color = 0
        for v in ordered:
            # Find smallest color not used by colored neighbors
            used_colors = set()
            for u in vertices:
                if u != v and u in color_map and (adj[v] >> u) & 1:
                    used_colors.add(color_map[u])
            c = 0
            while c in used_colors:
                c += 1
            color_map[v] = c
            next_color = max(next_color, c + 1)
        return next_color

    def branch_and_bound(candidates_mask, current_size, best_size, best_set, deadline):
        """
        Branch and bound for maximum independent set.
        candidates_mask: bitmask of available vertices
        current_size: size of current independent set
        best_size: best size found so far
        best_set: list of vertices in best set
        deadline: time limit
        Returns (best_size, best_set)
        """
        if (time.time() - start_time) > deadline:
            return best_size, best_set
        
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
            return best_size, best_set
        
        # Upper bound: current_size + greedy_color_bound of candidates
        candidates_list = [i for i in range(1, N + 1) if (candidates_mask >> i) & 1]
        if not candidates_list:
            if current_size > best_size:
                best_size = current_size
            return best_size, best_set
        
        color_map = {}
        ub = greedy_color_bound(candidates_list, color_map)
        if current_size + ub <= best_size:
            return best_size, best_set
        
        # Branch on a vertex with highest color
        # Pick vertex with max color (most constraining)
        max_color = -1
        pivot = None
        for v in candidates_list:
            if color_map.get(v, 0) > max_color:
                max_color = color_map[v]
                pivot = v
        
        if pivot is None:
            pivot = candidates_list[0]
        
        # Branch 1: Include pivot
        # New candidates: those compatible with pivot
        new_mask = candidates_mask & comp_adj[pivot]
        new_mask &= ~(1 << pivot)
        best_size, best_set = branch_and_bound(
            new_mask, current_size + 1, best_size, best_set, deadline
        )
        
        # Branch 2: Exclude pivot
        new_mask2 = candidates_mask & ~(1 << pivot)
        best_size, best_set = branch_and_bound(
            new_mask2, current_size, best_size, best_set, deadline
        )
        
        return best_size, best_set

    # 3. Generate Initial Solution
    # Use divisors of 360 as seed (size 24)
    candidates_list = [360, 480, 420, 336, 300, 240, 210, 180]
    best_initial = []
    for m in candidates_list:
        divs = get_divisors(m)
        if len(divs) > len(best_initial):
            best_initial = divs
    
    # Verify initial set is valid
    init_set = set(best_initial)
    valid = True
    for i in range(len(best_initial)):
        for j in range(i+1, len(best_initial)):
            a, b = best_initial[i], best_initial[j]
            l = a * b // math.gcd(a, b)
            if l > 512:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to {1, 2, ..., 20}
        best_initial = list(range(1, 21))

    # 4. Run Branch and Bound
    # Start with empty set, let B&B find maximum
    init_mask = 0
    for v in best_initial:
        init_mask |= (1 << v)
    
    # Full candidate universe
    full_candidates = 0
    for i in range(1, N + 1):
        full_candidates |= (1 << i)
    
    best_size = len(best_initial)
    best_set = list(best_initial)
    
    # Run B&B from full universe
    best_size, best_set = branch_and_bound(
        full_candidates, 0, best_size, best_set, time_limit
    )
    
    # 5. Final Validation
    result = best_set
    result_set = set(result)
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            a, b = result[i], result[j]
            l = a * b // math.gcd(a, b)
            if l > 512:
                # Fallback to known valid set
                return best_initial
                
    return sorted(result)

if __name__ == "__main__":
    # Test
    res = construct()
    print(f"Size: {len(res)}")
    # Verify
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            a, b = res[i], res[j]
            l = a * b // math.gcd(a, b)
            if l > 512:
                print(f"Invalid: {a}, {b}, lcm={l}")
                break
