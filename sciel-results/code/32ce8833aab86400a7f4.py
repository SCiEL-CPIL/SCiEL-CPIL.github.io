import math
import time

def construct():
    n = 256
    deadline = time.time() + 85  # safety margin before 90s kill

    # Sieve of Eratosthenes
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.sqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False

    primes = [i for i in range(2, n + 1) if is_prime[i]]

    # Precompute prime factors for each number 1..n
    prime_factors = [[] for _ in range(n + 1)]
    for i in range(2, n + 1):
        if is_prime[i]:
            for j in range(i, n + 1, i):
                prime_factors[j].append(i)

    # Build conflict graph: vertices 1..n, edge if gcd > 1
    # Use bitmasks for efficient conflict checking
    # n+1 vertices (index 0 unused, or use 1..n)
    # We'll use indices 1..n for numbers 1..n
    
    # For each vertex, store a bitmask of its neighbors (conflicting vertices)
    # Since n=256, we can use Python integers as bitmasks
    neighbors = [0] * (n + 1)
    for i in range(1, n + 1):
        fi = set(prime_factors[i])
        for j in range(i + 1, n + 1):
            fj = set(prime_factors[j])
            if fi & fj:
                neighbors[i] |= (1 << j)
                neighbors[j] |= (1 << i)

    # Weight of each vertex
    weight = [0] * (n + 1)
    for i in range(1, n + 1):
        weight[i] = i

    # Greedy coloring for upper bound
    # Colors assigned such that adjacent vertices have different colors
    # The number of colors is an upper bound on the size of any independent set
    # We use a simple greedy coloring: process vertices in order, assign smallest available color
    
    # Precompute a fixed ordering for coloring (by descending weight for tighter bounds)
    order = list(range(1, n + 1))
    order.sort(key=lambda x: -weight[x])
    
    # Map from vertex to position in order
    pos_in_order = [0] * (n + 1)
    for idx, v in enumerate(order):
        pos_in_order[v] = idx

    # Greedy coloring: assign colors to all vertices
    # color[v] = color assigned to vertex v
    # color_count[v] = number of colors used up to and including v's color
    # Actually, we need: for a subset S of vertices, the min number of colors needed
    # Standard approach: order vertices, assign smallest available color
    # Then for any subset, the max color index + 1 is an upper bound on independence number

    color = [0] * (n + 1)
    color_class = [[] for _ in range(n + 1)]  # color_class[c] = list of vertices with color c
    
    # Process in order (descending weight)
    for v in order:
        # Find smallest color not used by neighbors
        used_colors = set()
        nb = neighbors[v]
        # Check neighbors that come before v in order (or all neighbors)
        # For proper coloring, check all neighbors
        for u in range(1, n + 1):
            if nb & (1 << u):
                if color[u] > 0:
                    used_colors.add(color[u])
        
        # Assign smallest available color
        c = 1
        while c in used_colors:
            c += 1
        color[v] = c
        if c > len(color_class) - 1:
            color_class.append([])
        color_class[c].append(v)
    
    max_color = max(color[1:]) if n >= 1 else 0

    # For branch and bound, we need a dynamic coloring bound
    # When we have a candidate set P (vertices to consider), we compute
    # a coloring of the induced subgraph on P, and the number of colors
    # is an upper bound on the independence number of P

    def compute_coloring_bound(vertex_list):
        """
        Compute greedy coloring upper bound for the induced subgraph on vertex_list.
        Returns (upper_bound, colored_vertices) where colored_vertices maps each vertex
        to its color in this local coloring.
        """
        if not vertex_list:
            return 0, {}
        
        # Order by descending weight for tighter bounds
        ordered = sorted(vertex_list, key=lambda x: -weight[x])
        
        local_color = {}
        for v in ordered:
            # Find colors used by neighbors of v that are in vertex_list
            used = set()
            nb = neighbors[v]
            for u in vertex_list:
                if u != v and (nb & (1 << u)):
                    if u in local_color:
                        used.add(local_color[u])
            
            c = 1
            while c in used:
                c += 1
            local_color[v] = c
        
        max_c = max(local_color.values()) if local_color else 0
        return max_c, local_color

    # Branch and bound for maximum weight independent set
    best_sum = [0]
    best_set = [set()]
    
    def branch_and_bound(candidates, current_set, current_sum):
        """
        candidates: list of vertices still to consider
        current_set: set of vertices in current independent set
        current_sum: sum of weights in current_set
        """
        if time.time() > deadline:
            return
        
        if not candidates:
            if current_sum > best_sum[0]:
                best_sum[0] = current_sum
                best_set[0] = set(current_set)
            return
        
        # Upper bound: current_sum + sum of all remaining weights
        # But we can do better with coloring
        # Compute coloring bound on candidates
        ub, _ = compute_coloring_bound(candidates)
        
        # The upper bound on total weight is current_sum + sum of top ub weights
        # Actually, a simpler bound: current_sum + sum of all candidate weights
        # But coloring gives: the independence number of candidates <= ub
        # So the max additional weight <= sum of the ub largest weights in candidates
        
        # For a tighter bound, sort candidates by weight descending
        # and take the sum of the top min(ub, len(candidates)) weights
        sorted_cands = sorted(candidates, key=lambda x: -weight[x])
        additional_bound = sum(weight[v] for v in sorted_cands[:min(ub, len(sorted_cands))])
        
        if current_sum + additional_bound <= best_sum[0]:
            return
        
        # Pick a vertex to branch on
        # Choose the vertex with highest weight among candidates
        v = max(candidates, key=lambda x: weight[x])
        
        # Branch 1: include v
        new_set = current_set | {v}
        # New candidates: those in candidates that are not v and not neighbors of v
        nb_v = neighbors[v]
        new_candidates = [u for u in candidates if u != v and not (nb_v & (1 << u))]
        
        branch_and_bound(new_candidates, new_set, current_sum + weight[v])
        
        # Branch 2: exclude v
        new_candidates2 = [u for u in candidates if u != v]
        branch_and_bound(new_candidates2, current_set, current_sum)

    # Initialize with all vertices
    all_vertices = list(range(1, n + 1))
    
    # Start branch and bound
    branch_and_bound(all_vertices, set(), 0)
    
    # Convert best set to list of Python ints
    result = sorted(best_set[0])
    return [int(x) for x in result]
