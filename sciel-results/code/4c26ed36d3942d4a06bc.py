import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 256
    
    # Perfect squares up to 512
    squares = set()
    k = 1
    while k * k <= 512:
        squares.add(k * k)
        k += 1
        
    # Diagonal constraint: 2a is a perfect square
    # a = 2 * m^2
    diagonal_bad = set()
    m = 1
    while 2 * m * m <= n:
        diagonal_bad.add(2 * m * m)
        m += 1
        
    # Available vertices
    available_vertices = [i for i in range(1, n + 1) if i not in diagonal_bad]
    available_set = set(available_vertices)
    
    # Build adjacency lists
    conflicts = {v: set() for v in available_vertices}
    
    for a in available_vertices:
        for b in available_vertices:
            if a < b and (a + b) in squares:
                conflicts[a].add(b)
                conflicts[b].add(a)
                
    # Greedy Initialization using Minimum Degree Heuristic
    def greedy_init():
        current_set = set()
        remaining = set(available_vertices)
        
        while remaining:
            min_deg = float('inf')
            best_v = None
            for v in remaining:
                deg = len(conflicts[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            
            if best_v is None:
                break
                
            current_set.add(best_v)
            to_remove = {best_v} | (conflicts[best_v] & remaining)
            remaining -= to_remove
            
        return current_set

    def local_search(s, time_limit):
        current = set(s)
        in_set = set(current)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1. Try adding any vertex that is not in set and has no conflicts with set
            for v in available_vertices:
                if v in in_set:
                    continue
                if not (conflicts[v] & in_set):
                    current.add(v)
                    in_set.add(v)
                    improved = True
                    break
            
            if not improved:
                # 2. Try 1-for-2 swap
                swap_done = False
                for u in list(current):
                    candidates = []
                    for v in available_vertices:
                        if v in in_set:
                            continue
                        if u not in conflicts[v]:
                            continue
                        common = conflicts[v] & in_set
                        if len(common) == 1 and u in common:
                            candidates.append(v)
                    
                    if len(candidates) >= 2:
                        found_swap = False
                        for i in range(len(candidates)):
                            if found_swap: break
                            for j in range(i + 1, len(candidates)):
                                if candidates[i] not in conflicts[candidates[j]]:
                                    current.remove(u)
                                    in_set.remove(u)
                                    current.add(candidates[i])
                                    in_set.add(candidates[i])
                                    current.add(candidates[j])
                                    in_set.add(candidates[j])
                                    improved = True
                                    swap_done = True
                                    found_swap = True
                                    break
                        if swap_done:
                            break
                
                # 3. Try 2-for-3 swap (remove 2, add 3)
                if not swap_done and not improved:
                    for i_u in range(len(list(current))):
                        if swap_done:
                            break
                        u1 = list(current)[i_u]
                        for j_u in range(i_u + 1, len(list(current))):
                            if swap_done:
                                break
                            u2 = list(current)[j_u]
                            # Find vertices that conflict with u1 or u2 but not both
                            cand1 = []
                            cand2 = []
                            for v in available_vertices:
                                if v in in_set:
                                    continue
                                conflicts_with_u1 = u1 in conflicts[v]
                                conflicts_with_u2 = u2 in conflicts[v]
                                if conflicts_with_u1 and not conflicts_with_u2:
                                    cand1.append(v)
                                elif conflicts_with_u2 and not conflicts_with_u1:
                                    cand2.append(v)
                            
                            # Try to find 3 non-conflicting vertices from cand1+cand2
                            # that together don't conflict with (current - {u1, u2})
                            remaining_set = in_set - {u1, u2}
                            all_cands = list(set(cand1 + cand2))
                            # Filter: must not conflict with remaining_set
                            valid_cands = [v for v in all_cands if not (conflicts[v] & remaining_set)]
                            
                            if len(valid_cands) >= 3:
                                # Find 3 mutually non-conflicting
                                found = False
                                for a_i in range(len(valid_cands)):
                                    if found: break
                                    for b_i in range(a_i + 1, len(valid_cands)):
                                        if found: break
                                        if valid_cands[a_i] in conflicts[valid_cands[b_i]]:
                                            continue
                                        for c_i in range(b_i + 1, len(valid_cands)):
                                            if found: break
                                            if valid_cands[a_i] in conflicts[valid_cands[c_i]] or valid_cands[b_i] in conflicts[valid_cands[c_i]]:
                                                continue
                                            current.discard(u1)
                                            in_set.discard(u1)
                                            current.discard(u2)
                                            in_set.discard(u2)
                                            for vv in [valid_cands[a_i], valid_cands[b_i], valid_cands[c_i]]:
                                                current.add(vv)
                                                in_set.add(vv)
                                            improved = True
                                            swap_done = True
                                            found = True
                                            break
                
                if not swap_done:
                    # 4. Perturbation
                    if time.time() > time_limit - 2:
                        break
                        
                    k_remove = random.randint(1, 3)
                    to_remove = random.sample(list(current), min(k_remove, len(current)))
                    for v in to_remove:
                        current.remove(v)
                        in_set.remove(v)
                    
                    refill_improved = True
                    while refill_improved:
                        refill_improved = False
                        for v in available_vertices:
                            if v in in_set:
                                continue
                            if not (conflicts[v] & in_set):
                                current.add(v)
                                in_set.add(v)
                                refill_improved = True
                                break
                                
        return current

    def branch_and_bound_mis(vertices, time_limit, current_best):
        """
        Branch and bound for maximum independent set using Tomita-style
        greedy coloring on the conflict graph to get tight upper bounds.
        """
        best_set = set()
        best_size = 0
        node_count = [0]
        
        def greedy_color_order(P):
            """
            Order vertices of P and assign color numbers (1..k) such that
            vertices with color i form an independent set in the complement
            (i.e., a clique in the conflict graph). Returns (ordered_list, color_list)
            where ordered_list[i] has color color_list[i].
            The max color is an upper bound on the independence number of P.
            """
            if not P:
                return [], []
            
            P_list = list(P)
            # Sort by degree in P (descending) for better coloring
            P_list.sort(key=lambda v: len(conflicts[v] & P), reverse=True)
            
            color = [0] * len(P_list)
            max_color = 0
            
            for i in range(len(P_list)):
                v = P_list[i]
                # Find the smallest color not used by any neighbor of v in P_list[0..i-1]
                used_colors = set()
                for j in range(i):
                    if P_list[j] in conflicts[v]:
                        used_colors.add(color[j])
                
                c = 1
                while c in used_colors:
                    c += 1
                color[i] = c
                if c > max_color:
                    max_color = c
            
            # Order by color (descending) - Tomita's ordering
            order = sorted(range(len(P_list)), key=lambda i: color[i], reverse=True)
            ordered = [P_list[i] for i in order]
            colors = [color[i] for i in order]
            
            return ordered, colors
        
        def expand(P, current_set, current_size):
            """
            Tomita-style maximum independent set search.
            P: set of candidate vertices
            current_set: set of vertices already in the independent set
            current_size: len(current_set)
            Returns True if we should stop (time limit or complete search done)
            """
            nonlocal best_set, best_size
            
            if time.time() > time_limit:
                return True
            
            node_count[0] += 1
            
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
            
            if not P:
                return False
            
            # Greedy coloring to get upper bound and ordering
            ordered, colors = greedy_color_order(P)
            max_color = colors[0] if colors else 0
            
            # Upper bound: current_size + max_color
            if current_size + max_color <= best_size:
                return False  # Pruned
            
            # Process vertices in color order (descending)
            for idx in range(len(ordered)):
                v = ordered[idx]
                c = colors[idx]
                
                if time.time() > time_limit:
                    return True
                
                # Upper bound check: even if we include all remaining vertices
                # with color >= c, we can't beat best
                if current_size + c <= best_size:
                    return False  # All remaining vertices have color <= c
                
                # Include v
                P_new = P - {v} - conflicts[v]
                current_set.add(v)
                
                if expand(P_new, current_set, current_size + 1):
                    current_set.discard(v)
                    return True
                
                current_set.discard(v)
                P = P - {v}
            
            return False
        
        # Start
        expand(set(vertices), set(), 0)
        
        return best_set

    best_set = set()
    best_size = 0
    
    # Initial greedy
    s = greedy_init()
    best_set = set(s)
    best_size = len(s)
    
    # Local search on initial
    s = local_search(s, deadline - 10)
    if len(s) > best_size:
        best_size = len(s)
        best_set = set(s)
    
    # Multiple restarts
    random.seed(42)
    
    while time.time() < deadline - 3:
        # Randomized Greedy Init
        order = available_vertices[:]
        random.shuffle(order)
        
        current = set()
        in_set = set()
        for v in order:
            if not (conflicts[v] & in_set):
                current.add(v)
                in_set.add(v)
                
        # Local search
        current = local_search(current, deadline - 5)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = set(current)
        
        # Try branch-and-bound on a focused subset
        # Only if we have time
        if time.time() < deadline - 10:
            # Focus B&B on vertices that are "close" to the current best solution
            # i.e., vertices not in best_set that have limited conflicts with best_set
            # This creates a smaller, more tractable subproblem
            if len(available_vertices) <= 120:
                # Use all available vertices
                bb_candidates = set(available_vertices)
            else:
                # Focus on vertices with fewest conflicts with current best
                scored = []
                for v in available_vertices:
                    if v in best_set:
                        continue
                    n_conf = len(conflicts[v] & best_set)
                    scored.append((n_conf, v))
                scored.sort()
                # Take top 60 non-best vertices
                bb_candidates = set(best_set) | set(v for _, v in scored[:60])
            
            bb_result = branch_and_bound_mis(bb_candidates, deadline - 5, best_size)
            if len(bb_result) > best_size:
                best_size = len(bb_result)
                best_set = set(bb_result)
            
    return list(best_set)
