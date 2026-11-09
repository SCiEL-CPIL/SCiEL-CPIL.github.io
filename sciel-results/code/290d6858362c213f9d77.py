import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (limit is 150s)
    
    # Precompute K4s and edge mappings
    def get_idx(i, j):
        return i * 15 - i * (i - 1) // 2 + (j - i - 1)
        
    edges = []
    for i in range(16):
        for j in range(i + 1, 16):
            edges.append((i, j))
            
    edge_idx_map = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx_map[(i, j)] = idx
        edge_idx_map[(j, i)] = idx

    k4s_all = []
    edge_k4s = [[] for _ in range(120)]
    
    for quad in combinations(range(16), 4):
        a, b, c, d = quad
        e_indices = [
            edge_idx_map[(a, b)],
            edge_idx_map[(a, c)],
            edge_idx_map[(a, d)],
            edge_idx_map[(b, c)],
            edge_idx_map[(b, d)],
            edge_idx_map[(c, d)]
        ]
        k4_idx = len(k4s_all)
        k4s_all.append(e_indices)
        for ei in e_indices:
            edge_k4s[ei].append(k4_idx)
    
    def count_invalid_k4s(colors):
        invalid_count = 0
        for e_indices in k4s_all:
            c0 = colors[e_indices[0]]
            c1 = colors[e_indices[1]]
            c2 = colors[e_indices[2]]
            c3 = colors[e_indices[3]]
            c4 = colors[e_indices[4]]
            c5 = colors[e_indices[5]]
            if len({c0, c1, c2, c3, c4, c5}) < 5:
                invalid_count += 1
        return invalid_count

    def is_valid_coloring(colors, K):
        # Check if all colors are in range and all K4s are valid
        for c in colors:
            if c < 0 or c >= K:
                return False
        return count_invalid_k4s(colors) == 0

    def try_csp_backtracking(K, time_limit):
        """
        CSP with backtracking, forward checking, and MRV.
        Variables: edges (0..119)
        Domain: 0..K-1
        Constraint: every K4 must have at least 5 distinct colors
        """
        # We'll use a heuristic ordering: assign edges in an order that
        # creates K4s early so we can prune early.
        
        # Precompute for each edge, which K4s it belongs to
        # We already have edge_k4s
        
        # For forward checking, we need to track for each K4:
        # - the set of colors used so far
        # - the number of unassigned edges in that K4
        # - whether the K4 is still satisfiable
        
        # State:
        # colors[120] = -1 if unassigned, else color value
        colors = [-1] * 120
        
        # For each K4, track:
        # k4_colors[k4_idx] = set of colors assigned so far
        # k4_unassigned[k4_idx] = number of unassigned edges in this K4
        # k4_satisfied[k4_idx] = whether this K4 already has >= 5 distinct colors
        
        k4_colors = [set() for _ in range(len(k4s_all))]
        k4_unassigned = [len(k4s_all[0]) for _ in range(len(k4s_all))]  # initially 6 each
        k4_satisfied = [False] * len(k4s_all)
        
        # Count of assigned edges
        assigned_count = 0
        
        # For MRV: we want to pick the unassigned edge that is in the most
        # "critical" K4s. A simple heuristic: pick the edge that appears in
        # the most K4s that are not yet satisfied and have fewest unassigned edges.
        
        # Precompute edge degree (number of K4s containing each edge)
        edge_degree = [len(edge_k4s[i]) for i in range(120)]
        
        # For forward checking: when we assign color c to edge e, for each K4 containing e:
        # - add c to k4_colors
        # - decrement k4_unassigned
        # - check if K4 is now impossible: if k4_unassigned > 0 and 
        #   len(k4_colors) + k4_unassigned < 5, then it's impossible
        #   (we can't get to 5 distinct colors)
        
        def is_k4_impossible(k4_idx):
            # A K4 is impossible if even assigning all remaining edges distinct
            # colors from the already-used set, we can't reach 5 distinct colors.
            # Max distinct colors we can have = len(k4_colors) + k4_unassigned
            # (if all unassigned edges get new colors not in k4_colors)
            # But actually, unassigned edges can also reuse colors, so the max
            # distinct is min(6, len(k4_colors) + k4_unassigned)
            # We need at least 5 distinct colors.
            if len(k4_colors) + k4_unassigned < 5:
                return True
            return False
        
        def check_all_k4s_possible():
            for k4_idx in range(len(k4s_all)):
                if not k4_satisfied[k4_idx] and is_k4_impossible(k4_idx):
                    return False
            return True
        
        # MRV heuristic: pick the unassigned edge that minimizes the maximum
        # "tightness" of K4s it's in. Or simpler: pick the unassigned edge
        # with the highest edge_degree (most constrained).
        
        # Actually, let's use a simpler approach: random restarts with 
        # randomized edge ordering, and forward checking.
        
        best_colors = None
        best_assigned = 0
        
        # We'll do a depth-first search with backtracking
        # To avoid recursion depth issues, we'll use an explicit stack
        
        # Actually, for 120 variables, backtracking with forward checking
        # should work if we have good heuristics. Let's try iterative deepening
        # or just straight backtracking with time limit.
        
        # Let's use a recursive approach with sys.setrecursionlimit
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(10000)
        
        def backtrack():
            nonlocal best_colors, best_assigned
            
            if time.time() > time_limit:
                return False
            
            if assigned_count == 120:
                # All edges assigned, check if valid
                if is_valid_coloring(colors, K):
                    best_colors = colors[:]
                    best_assigned = 120
                    return True
                return False
            
            # Find the next unassigned edge using MRV
            # MRV: pick the unassigned edge that is in the most unsatisfied K4s
            # with the fewest remaining unassigned edges
            
            best_edge = -1
            best_score = -1
            
            for e in range(120):
                if colors[e] != -1:
                    continue
                # Score: sum over K4s containing e of (1 if not satisfied) * (1/k4_unassigned)
                score = 0
                for k4_idx in edge_k4s[e]:
                    if not k4_satisfied[k4_idx]:
                        if k4_unassigned[k4_idx] > 0:
                            score += 1.0 / k4_unassigned[k4_idx]
                        else:
                            # This K4 has no unassigned edges but isn't satisfied
                            # That means it's already invalid
                            return False
                if score > best_score:
                    best_score = score
                    best_edge = e
            
            if best_edge == -1:
                return False
            
            e = best_edge
            
            # Try colors in random order
            color_order = list(range(K))
            random.shuffle(color_order)
            
            for c in color_order:
                if time.time() > time_limit:
                    return False
                
                # Assign color c to edge e
                colors[e] = c
                assigned_count += 1
                
                # Update K4s
                affected_k4s = []
                for k4_idx in edge_k4s[e]:
                    k4_colors[k4_idx].add(c)
                    k4_unassigned[k4_idx] -= 1
                    affected_k4s.append(k4_idx)
                    
                    if len(k4_colors[k4_idx]) >= 5:
                        k4_satisfied[k4_idx] = True
                
                # Check if any K4 is now impossible
                possible = True
                for k4_idx in affected_k4s:
                    if not k4_satisfied[k4_idx] and is_k4_impossible(k4_idx):
                        possible = False
                        break
                
                if possible:
                    if backtrack():
                        return True
                
                # Undo assignment
                for k4_idx in affected_k4s:
                    k4_unassigned[k4_idx] += 1
                    # Note: we can't easily remove c from k4_colors because
                    # other edges in the K4 might also have color c.
                    # This is a problem with the forward checking approach.
                    # We need a more careful undo.
                
                # For simplicity, let's just reset the K4 state for affected K4s
                # by recomputing from the current colors array
                for k4_idx in affected_k4s:
                    k4_colors[k4_idx] = set()
                    k4_unassigned[k4_idx] = 0
                    k4_satisfied[k4_idx] = False
                    for ei in k4s_all[k4_idx]:
                        if colors[ei] == -1:
                            k4_unassigned[k4_idx] += 1
                        else:
                            k4_colors[k4_idx].add(colors[ei])
                    if len(k4_colors[k4_idx]) >= 5:
                        k4_satisfied[k4_idx] = True
                
                colors[e] = -1
                assigned_count -= 1
            
            return False
        
        # Try multiple random restarts
        num_restarts = 0
        while time.time() < time_limit:
            num_restarts += 1
            
            # Reset state
            colors = [-1] * 120
            k4_colors = [set() for _ in range(len(k4s_all))]
            k4_unassigned = [6] * len(k4s_all)
            k4_satisfied = [False] * len(k4s_all)
            assigned_count = 0
            
            if backtrack():
                if best_colors is not None:
                    return best_colors
        
        return best_colors

    def try_simulated_annealing(K, time_limit):
        """
        Simulated annealing with better moves.
        """
        colors = [random.randint(0, K - 1) for _ in range(120)]
        
        # Calculate initial invalid count
        total_invalid = count_invalid_k4s(colors)
        
        best_colors = colors[:]
        best_invalid = total_invalid
        
        if best_invalid == 0:
            return best_colors
        
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.9995
        
        # Precompute for each edge, the list of K4 indices
        # We already have edge_k4s
        
        while time.time() < time_limit:
            # Pick a random edge
            e_idx = random.randint(0, 119)
            old_color = colors[e_idx]
            new_color = random.randint(0, K - 1)
            
            if new_color == old_color:
                continue
            
            # Calculate delta in invalid count
            delta_invalid = 0
            
            for k4_idx in edge_k4s[e_idx]:
                e_indices = k4s_all[k4_idx]
                
                # Get colors before
                c_before = [colors[ei] for ei in e_indices]
                valid_before = len(set(c_before)) >= 5
                
                # Temporarily change
                colors[e_idx] = new_color
                
                # Get colors after
                c_after = [colors[ei] for ei in e_indices]
                valid_after = len(set(c_after)) >= 5
                
                if valid_before and not valid_after:
                    delta_invalid += 1
                elif not valid_before and valid_after:
                    delta_invalid -= 1
            
            new_total_invalid = total_invalid + delta_invalid
            
            # Acceptance criteria
            accept = False
            if new_total_invalid < total_invalid:
                accept = True
            elif new_total_invalid == total_invalid:
                accept = random.random() < 0.5
            else:
                delta = new_total_invalid - total_invalid
                if delta > 0:
                    prob = pow(2.71828, -delta / T)
                    if random.random() < prob:
                        accept = True
            
            if accept:
                total_invalid = new_total_invalid
                
                if total_invalid < best_invalid:
                    best_invalid = total_invalid
                    best_colors = colors[:]
                    
                    if best_invalid == 0:
                        return best_colors
            else:
                colors[e_idx] = old_color
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
        
        return best_colors

    # Try different K values
    # The theoretical lower bound is 13, upper bound is 15
    
    # Strategy:
    # 1. Try CSP with K=13 for a short time
    # 2. Try simulated annealing with K=13 for a short time
    # 3. Try CSP with K=14 for a short time
    # 4. Try simulated annealing with K=14 for a short time
    # 5. Try CSP with K=15 for a short time
    # 6. Try simulated annealing with K=15 for remaining time
    
    best_result = None
    best_K = 999
    
    # Time allocation
    t_now = time.time()
    remaining = deadline - t_now
    
    # Try K=13 with CSP
    if remaining > 5:
        t_limit = t_now + min(10, remaining * 0.15)
        result = try_csp_backtracking(13, t_limit)
        if result is not None and is_valid_coloring(result, 13):
            best_result = result
            best_K = 13
            return [c for c in result]
    
    # Try K=13 with simulated annealing
    t_now = time.time()
    remaining = deadline - t_now
    if remaining > 5:
        t_limit = t_now + min(15, remaining * 0.15)
        result = try_simulated_annealing(13, t_limit)
        if result is not None and is_valid_coloring(result, 13):
            best_result = result
            best_K = 13
            return [c for c in result]
    
    # Try K=14 with CSP
    t_now = time.time()
    remaining = deadline - t_now
    if remaining > 5:
        t_limit = t_now + min(15, remaining * 0.15)
        result = try_csp_backtracking(14, t_limit)
        if result is not None and is_valid_coloring(result, 14):
            best_result = result
            best_K = 14
            return [c for c in result]
    
    # Try K=14 with simulated annealing
    t_now = time.time()
    remaining = deadline - t_now
    if remaining > 5:
        t_limit = t_now + min(20, remaining * 0.2)
        result = try_simulated_annealing(14, t_limit)
        if result is not None and is_valid_coloring(result, 14):
            best_result = result
            best_K = 14
            return [c for c in result]
    
    # Try K=15 with CSP
    t_now = time.time()
    remaining = deadline - t_now
    if remaining > 5:
        t_limit = t_now + min(20, remaining * 0.2)
        result = try_csp_backtracking(15, t_limit)
        if result is not None and is_valid_coloring(result, 15):
            best_result = result
            best_K = 15
            return [c for c in result]
    
    # Try K=15 with simulated annealing (use remaining time)
    t_now = time.time()
    remaining = deadline - t_now
    if remaining > 5:
        t_limit = t_now + remaining * 0.95
        result = try_simulated_annealing(15, t_limit)
        if result is not None and is_valid_coloring(result, 15):
            best_result = result
            best_K = 15
            return [c for c in result]
    
    # Fallback: try K=16 with simulated annealing
    t_now = time.time()
    remaining = deadline - t_now
    if remaining > 2:
        t_limit = t_now + remaining * 0.95
        result = try_simulated_annealing(16, t_limit)
        if result is not None and is_valid_coloring(result, 16):
            best_result = result
            best_K = 16
            return [c for c in result]
    
    # If we couldn't find a valid coloring with 13-16, use a known construction
    # with 17 colors as fallback
    colors_17 = [0] * 120
    for idx, (i, j) in enumerate(edges):
        colors_17[idx] = (i + j) % 17
    
    if count_invalid_k4s(colors_17) == 0:
        unique_colors = sorted(set(colors_17))
        mapping = {c: i for i, c in enumerate(unique_colors)}
        return [mapping[c] for c in colors_17]
    
    # Final fallback: random coloring with 17 colors
    random.seed(42)
    colors = [random.randint(0, 16) for _ in range(120)]
    return colors
