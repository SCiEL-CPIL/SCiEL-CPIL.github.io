"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    N = 256
    # Precompute which sums are perfect squares
    max_sum = 2 * N
    is_square = [False] * (max_sum + 1)
    for k in range(1, int(math.isqrt(max_sum)) + 1):
        is_square[k * k] = True
    
    # Identify forbidden vertices (where 2*a is a perfect square)
    forbidden = [False] * N
    for i in range(N):
        val = i + 1
        if is_square[2 * val]:
            forbidden[i] = True
    
    # Build adjacency list for the conflict graph among allowed vertices
    allowed = [i for i in range(N) if not forbidden[i]]
    m = len(allowed)
    
    # Build adjacency list
    adj = [[] for _ in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            a = allowed[i] + 1
            b = allowed[j] + 1
            if is_square[a + b]:
                adj[i].append(j)
                adj[j].append(i)
    
    # Convert to bitsets for fast operations
    adj_bits = [0] * m
    for i in range(m):
        bits = 0
        for j in adj[i]:
            bits |= (1 << j)
        adj_bits[i] = bits
    
    best_set = []
    best_size = 0
    
    # Greedy initial solution using min-degree heuristic
    def greedy_min_degree():
        nonlocal best_set, best_size
        in_set = [False] * m
        remaining = set(range(m))
        deg = [len(adj[i]) for i in range(m)]
        while remaining:
            min_idx = min(remaining, key=lambda x: deg[x])
            in_set[min_idx] = True
            to_remove = {min_idx} | set(adj[min_idx])
            for v in to_remove:
                if v in remaining:
                    for u in adj[v]:
                        if u in remaining:
                            deg[u] -= 1
                    remaining.discard(v)
        sol = [i for i in range(m) if in_set[i]]
        if len(sol) > best_size:
            best_size = len(sol)
            best_set = sol[:]
        return sol
    
    greedy_min_degree()
    
    # Local search improvement
    def local_search_improve(sol):
        nonlocal best_set, best_size
        sol_set = set(sol)
        improved = True
        while improved:
            improved = False
            for v in range(m):
                if v in sol_set:
                    continue
                can_add = True
                for u in adj[v]:
                    if u in sol_set:
                        can_add = False
                        break
                if can_add:
                    sol_set.add(v)
                    improved = True
        if len(sol_set) > best_size:
            best_size = len(sol_set)
            best_set = list(sol_set)
        return sol_set
    
    local_search_improve(set(best_set))
    
    # Specialized B&B exploiting matching structure
    # The conflict graph is a union of matchings (one per square sum)
    # Use fractional packing bound for tighter pruning
    
    def fractional_packing_bound(candidates):
        """Compute a tight upper bound using the matching structure.
        Each square sum defines a matching. We can compute a fractional vertex cover
        to get a lower bound on vertex cover, hence upper bound on independent set."""
        if not candidates:
            return 0
        
        cand_set = set(candidates)
        n_cand = len(cand_set)
        
        # Simple bound: for each matching, at most floor(|V_match|/2) vertices can be excluded
        # Actually, let's use a simpler but effective bound:
        # The maximum independent set in a graph with maximum degree Δ is at most n * (Δ+1)/(2Δ)
        # But we can do better using the matching decomposition
        
        # For each square sum s, the matching M_s has edges (a,b) where a+b=s
        # In any independent set, for each edge in M_s, at most one endpoint is included
        # So |IS ∩ V(M_s)| ≤ |V(M_s)| - |M_s| (since each edge removes at least 1 vertex)
        
        # Actually, a cleaner bound: 
        # For each matching M_s, the number of vertices in IS that are incident to M_s ≤ |V(M_s)| - |M_s|
        # But vertices may be in multiple matchings...
        
        # Let's use a different approach: greedy coloring for upper bound
        # Since the graph is sparse (union of few matchings), greedy coloring should be tight
        
        color_of = {}
        max_color = 0
        # Order vertices by degree in candidate subgraph for better coloring
        deg_in_cand = {}
        for v in cand_set:
            deg_in_cand[v] = sum(1 for u in adj[v] if u in cand_set)
        
        ordered = sorted(cand_set, key=lambda v: -deg_in_cand[v])
        for v in ordered:
            used_colors = set()
            for u in adj[v]:
                if u in color_of:
                    used_colors.add(color_of[u])
            c = 0
            while c in used_colors:
                c += 1
            color_of[v] = c
            max_color = max(max_color, c + 1)
        
        return max_color
    
    # B&B with the matching-aware bound
    def branch_and_bound(chosen, candidates):
        nonlocal best_set, best_size, deadline
        
        if time.time() > deadline:
            return
        
        # Upper bound
        ub = len(chosen) + fractional_packing_bound(candidates)
        if ub <= best_size:
            return
        
        if not candidates:
            if len(chosen) > best_size:
                best_size = len(chosen)
                best_set = chosen[:]
            return
        
        # Pick vertex with highest degree in candidate subgraph
        cand_list = list(candidates)
        cand_set = set(candidates)
        best_v = max(cand_list, key=lambda v: sum(1 for u in adj[v] if u in cand_set))
        
        # Branch 1: include best_v
        new_chosen = chosen + [best_v]
        neighbors = set(adj[best_v])
        new_candidates = cand_set - {best_v} - neighbors
        branch_and_bound(new_chosen, new_candidates)
        
        # Branch 2: exclude best_v
        new_candidates2 = cand_set - {best_v}
        branch_and_bound(chosen, new_candidates2)
    
    # Run B&B
    initial_candidates = set(range(m))
    branch_and_bound([], initial_candidates)
    
    # Check time
    if time.time() > deadline - 5:
        result = [allowed[i] + 1 for i in best_set]
        if not result:
            result = list(range(1, 257, 3))
        return result
    
    # Multiple random restarts with local search
    random.seed(42)
    for restart in range(200):
        if time.time() > deadline - 5:
            break
        
        random_order = list(range(m))
        random.shuffle(random_order)
        sol_set = set()
        for v in random_order:
            can_add = True
            for u in adj[v]:
                if u in sol_set:
                    can_add = False
                    break
            if can_add:
                sol_set.add(v)
        
        sol_set = local_search_improve(sol_set)
        
        if time.time() > deadline - 5:
            break
    
    result = [allowed[i] + 1 for i in best_set]
    if not result:
        result = list(range(1, 257, 3))
    return result
# EVOLVE-BLOCK-END
