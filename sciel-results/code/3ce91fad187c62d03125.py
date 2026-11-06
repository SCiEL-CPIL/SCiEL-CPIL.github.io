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
                
                if not swap_done:
                    # 3. Perturbation
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
        Branch and bound for maximum independent set.
        vertices: set of candidate vertices to consider
        time_limit: absolute time limit
        current_best: current best solution size
        Returns: best independent set found within time limit
        """
        best_set = set()
        best_size = current_best  # Start with current best as baseline
        best_set_from_init = set()
        
        # Precompute: for quick degree checks
        def get_neighbors(v, candidate_set):
            return conflicts[v] & candidate_set
        
        def max_clique_lower_bound(candidate_set):
            """
            Compute a lower bound on the maximum clique size in the conflict graph
            restricted to candidate_set. This provides a tighter upper bound on
            the independence number: alpha <= |candidate_set| - omega + 1.
            Uses a greedy heuristic to find a large clique.
            """
            if not candidate_set:
                return 0
            
            # Greedy clique finding: repeatedly pick vertex with max degree in remaining
            remaining = set(candidate_set)
            clique_size = 0
            while remaining:
                # Pick vertex with maximum degree in remaining
                max_deg = -1
                best_v = None
                for v in remaining:
                    deg = len(conflicts[v] & remaining)
                    if deg > max_deg:
                        max_deg = deg
                        best_v = v
                
                if best_v is None:
                    break
                
                clique_size += 1
                # Keep only neighbors of best_v in remaining
                remaining = remaining & conflicts[best_v]
            
            return clique_size
        
        def independence_upper_bound(candidate_set):
            """
            Upper bound on the independence number of the conflict graph
            restricted to candidate_set.
            Uses: alpha <= |V| - omega + 1 where omega is a lower bound on max clique.
            Also uses greedy coloring of the conflict graph as an alternative bound.
            Takes the minimum of both bounds.
            """
            if not candidate_set:
                return 0
            
            n_cand = len(candidate_set)
            
            # Bound 1: n - omega + 1
            omega_lb = max_clique_lower_bound(candidate_set)
            bound1 = n_cand - omega_lb + 1
            
            # Bound 2: Greedy coloring of conflict graph (chromatic number upper bounds independence)
            # Actually, chromatic number >= clique number, and independence number <= n - clique + 1
            # Let's use a different approach: greedy coloring gives upper bound on independence
            uncolored = set(candidate_set)
            color_count = 0
            while uncolored:
                color_count += 1
                min_deg = float('inf')
                best_v = None
                for v in uncolored:
                    deg = len(conflicts[v] & uncolored)
                    if deg < min_deg:
                        min_deg = deg
                        best_v = v
                
                if best_v is None:
                    break
                    
                uncolored.remove(best_v)
                for v in list(uncolored):
                    if v not in conflicts[best_v]:
                        uncolored.remove(v)
            
            # The number of colors is an upper bound on the chromatic number,
            # and chromatic number >= n / alpha, so alpha >= n / chi
            # This gives a lower bound, not upper. Let me reconsider.
            # Actually for independent set: we want upper bound on alpha.
            # A better bound: use the fact that in any independent set, no two vertices
            # from the same color class can be chosen... wait, that's for coloring.
            # Let me just use bound1 as the primary bound.
            
            return min(bound1, n_cand)
        
        def dfs(candidate_set, current_set, current_size):
            nonlocal best_set, best_size
            
            if time.time() > time_limit:
                return False
            
            # Update best if improved
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
            
            if not candidate_set:
                return True
            
            # Upper bound: current_size + independence_upper_bound(candidate_set)
            # If this doesn't beat best_size, prune
            ub = current_size + independence_upper_bound(candidate_set)
            if ub <= best_size:
                return True  # Pruned
            
            # Branch on a vertex
            # Choose vertex with maximum degree in candidate_set (more pruning)
            max_deg = -1
            branch_v = None
            for v in candidate_set:
                deg = len(conflicts[v] & candidate_set)
                if deg > max_deg:
                    max_deg = deg
                    branch_v = v
            
            if branch_v is None:
                return True
            
            # Option 1: Include branch_v
            new_candidate = candidate_set - {branch_v} - conflicts[branch_v]
            new_current = current_set | {branch_v}
            
            if dfs(new_candidate, new_current, current_size + 1):
                return True
            
            # Option 2: Exclude branch_v
            new_candidate = candidate_set - {branch_v}
            
            return dfs(new_candidate, current_set, current_size)
        
        # Start DFS
        dfs(set(vertices), set(), 0)
        
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
        
        # Try branch-and-bound on a subset around the current best
        # Only if we have time
        if time.time() < deadline - 10:
            # Use B&B on the full problem
            bb_result = branch_and_bound_mis(set(available_vertices), deadline - 5, best_size)
            if len(bb_result) > best_size:
                best_size = len(bb_result)
                best_set = set(bb_result)
            
    return list(best_set)
