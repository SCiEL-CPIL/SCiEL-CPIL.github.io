import numpy as np
import time
from itertools import combinations
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def build_structure(n):
        """Build all the structural data needed for constraint propagation."""
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # For each triple, which K4s does it belong to?
        # K4: 4 vertices, 4 triples
        k4s = []
        triple_to_k4s = defaultdict(list)
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            t1 = triple_to_idx[(a,b,c)]
            t2 = triple_to_idx[(a,b,d)]
            t3 = triple_to_idx[(a,c,d)]
            t4 = triple_to_idx[(b,c,d)]
            k4_idx = len(k4s)
            k4s.append((t1, t2, t3, t4))
            for t in (t1, t2, t3, t4):
                triple_to_k4s[t].append(k4_idx)
        
        # For each triple, which K5s does it belong to?
        # K5: 5 vertices, 10 triples
        k5s = []
        triple_to_k5s = defaultdict(list)
        for quint in combinations(range(n), 5):
            idxs = []
            for triple in combinations(quint, 3):
                idxs.append(triple_to_idx[triple])
            k5_idx = len(k5s)
            k5s.append(tuple(idxs))
            for t in idxs:
                triple_to_k5s[t].append(k5_idx)
        
        return triples, num_triples, triple_to_idx, k4s, k5s, triple_to_k4s, triple_to_k5s
    
    def backtrack_search(n, time_limit):
        """
        Backtracking search with constraint propagation.
        Colors triples one by one, pruning when a constraint would be violated.
        """
        triples, num_triples, triple_to_idx, k4s, k5s, triple_to_k4s, triple_to_k5s = build_structure(n)
        
        # colors[i] = -1 (unassigned), 0 (red), 1 (blue)
        colors = [-1] * num_triples
        
        # For constraint propagation, track for each K4 how many red edges and how many unassigned
        k4_red_count = [0] * len(k4s)
        k4_blue_count = [0] * len(k4s)
        k4_unassigned = [4] * len(k4s)
        
        k5_red_count = [0] * len(k5s)
        k5_blue_count = [0] * len(k5s)
        k5_unassigned = [10] * len(k5s)
        
        # For efficient backtracking, we need to track assignments
        assignment_stack = []
        
        def is_feasible_k4(k4_idx):
            """Check if K4 can still avoid being all-red."""
            # If all 4 are red, infeasible
            if k4_red_count[k4_idx] == 4:
                return False
            # If 3 are red and 1 unassigned, the last must be blue - that's ok, it's forced
            # But if 3 are red and the 4th is already blue, that's fine
            # The constraint is: NOT (all 4 red)
            # So infeasible only if all 4 are red
            return True
        
        def is_feasible_k5(k5_idx):
            """Check if K5 can still avoid being all-blue."""
            # If all 10 are blue, infeasible
            if k5_blue_count[k5_idx] == 10:
                return False
            return True
        
        def assign(triple_idx, color):
            """Assign a color to a triple and update counts."""
            colors[triple_idx] = color
            assignment_stack.append((triple_idx, color))
            
            # Update K4 counts
            for k4_idx in triple_to_k4s[triple_idx]:
                k4_unassigned[k4_idx] -= 1
                if color == 0:
                    k4_red_count[k4_idx] += 1
                else:
                    k4_blue_count[k4_idx] += 1
                if not is_feasible_k4(k4_idx):
                    return False
            
            # Update K5 counts
            for k5_idx in triple_to_k5s[triple_idx]:
                k5_unassigned[k5_idx] -= 1
                if color == 0:
                    k5_red_count[k5_idx] += 1
                else:
                    k5_blue_count[k5_idx] += 1
                if not is_feasible_k5(k5_idx):
                    return False
            
            return True
        
        def unassign(triple_idx):
            """Unassign a triple and roll back counts."""
            _, color = assignment_stack.pop()
            colors[triple_idx] = -1
            
            # Roll back K4 counts
            for k4_idx in triple_to_k4s[triple_idx]:
                k4_unassigned[k4_idx] += 1
                if color == 0:
                    k4_red_count[k4_idx] -= 1
                else:
                    k4_blue_count[k4_idx] -= 1
            
            # Roll back K5 counts
            for k5_idx in triple_to_k5s[triple_idx]:
                k5_unassigned[k5_idx] += 1
                if color == 0:
                    k5_red_count[k5_idx] -= 1
                else:
                    k5_blue_count[k5_idx] -= 1
        
        def get_forced_assignments():
            """
            Find triples that are forced by constraints.
            A triple is forced if:
            - It's in a K4 where 3 other edges are red -> must be blue
            - It's in a K5 where 9 other edges are blue -> must be red
            """
            forced = {}
            
            for t in range(num_triples):
                if colors[t] != -1:
                    continue
                
                # Check K4 constraints
                for k4_idx in triple_to_k4s[t]:
                    if k4_unassigned[k4_idx] == 1 and k4_red_count[k4_idx] == 3:
                        # Must be blue (1)
                        forced[t] = 1
                        break
                
                if t in forced:
                    continue
                
                # Check K5 constraints
                for k5_idx in triple_to_k5s[t]:
                    if k5_unassigned[k5_idx] == 1 and k5_blue_count[k5_idx] == 9:
                        # Must be red (0)
                        forced[t] = 0
                        break
            
            return forced
        
        def search(depth, time_limit):
            """Recursive backtracking search."""
            if time.time() > time_limit:
                return None
            
            # Check for contradictions
            for k4_idx in range(len(k4s)):
                if k4_red_count[k4_idx] == 4:
                    return None
            for k5_idx in range(len(k5s)):
                if k5_blue_count[k5_idx] == 10:
                    return None
            
            # Find forced assignments
            forced = get_forced_assignments()
            
            # Apply forced assignments
            while forced:
                t, color = forced.popitem()
                if colors[t] != -1:
                    continue
                if not assign(t, color):
                    # Contradiction
                    # Need to backtrack
                    # This is complex - for now, if we hit a contradiction from forced,
                    # we need to undo all forced assignments in this round
                    while assignment_stack:
                        unassign(assignment_stack[-1][0])
                    return None
                # Re-check for new forced assignments
                new_forced = get_forced_assignments()
                forced.update(new_forced)
            
            # Find the next unassigned triple
            # Use most-constrained-first: pick the triple with the most constraints
            unassigned = [i for i in range(num_triples) if colors[i] == -1]
            if not unassigned:
                # All assigned, check if valid
                return colors[:]
            
            # Pick the most constrained unassigned triple
            # Count how many K4s and K5s each unassigned triple is in
            best_t = unassigned[0]
            best_constraint_count = 0
            for t in unassigned:
                constraint_count = len(triple_to_k4s[t]) + len(triple_to_k5s[t])
                if constraint_count > best_constraint_count:
                    best_constraint_count = constraint_count
                    best_t = t
            
            # Try both colors
            for color in [0, 1]:
                if time.time() > time_limit:
                    return None
                
                if assign(best_t, color):
                    result = search(depth + 1, time_limit)
                    if result is not None:
                        return result
                    # Roll back
                    unassign(best_t)
            
            return None
        
        # Start search
        result = search(0, time_limit)
        if result is not None:
            return result
        return None
    
    def simulate_annealing(n, time_limit, init_colors=None):
        """Simulated annealing fallback for larger n."""
        triples, num_triples, triple_to_idx, k4s, k5s, triple_to_k4s, triple_to_k5s = build_structure(n)
        
        if init_colors is None:
            # Heuristic initialization
            best_init = None
            best_init_cost = float('inf')
            
            patterns = [
                lambda a,b,c: 0 if (a+b+c) % 3 == 0 else 1,
                lambda a,b,c: 0 if (a+b+c) % 2 == 0 else 1,
                lambda a,b,c: 0 if (a*b+c) % 3 == 0 else 1,
                lambda a,b,c: 0 if (a**2+b**2+c**2) % 2 == 0 else 1,
                lambda a,b,c: 0 if (a*b*c) % 2 == 0 else 1,
            ]
            
            for pat in patterns:
                c = np.array([pat(a,b,c) for a,b,c in triples], dtype=np.int8)
                cost = count_violations(c, k4s, k5s)
                if cost < best_init_cost:
                    best_init_cost = cost
                    best_init = c
            if best_init is None:
                best_init = np.random.randint(0, 2, num_triples, dtype=np.int8)
                best_init_cost = count_violations(best_init, k4s, k5s)
            colors = best_init.copy()
        else:
            colors = init_colors.copy()
        
        current_cost = count_violations(colors, k4s, k5s)
        best_cost = current_cost
        best_colors = colors.copy()
        
        temp = 1.0
        cool_rate = 0.9995
        min_temp = 0.05
        
        while time.time() < time_limit:
            t = np.random.randint(num_triples)
            old_color = colors[t]
            new_color = 1 - old_color
            
            delta_cost = compute_delta(t, new_color, colors, k4s, k5s, triple_to_k4s, triple_to_k5s)
            new_cost = current_cost + delta_cost
            
            if new_cost < current_cost or (new_cost == current_cost and np.random.rand() < 0.5):
                accept = True
            elif temp > min_temp:
                accept = np.random.rand() < np.exp(-new_cost / max(temp, 1e-10))
            else:
                accept = False
            
            if accept:
                colors[t] = new_color
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()
                    if best_cost == 0:
                        return best_colors.tolist()
            
            temp *= cool_rate
        
        return best_colors.tolist()
    
    def count_violations(colors, k4s, k5s):
        """Count total violations."""
        cost = 0
        for idxs in k4s:
            if all(colors[i] == 0 for i in idxs):
                cost += 1
        for idxs in k5s:
            if all(colors[i] == 1 for i in idxs):
                cost += 1
        return cost
    
    def compute_delta(t, new_color, colors, k4s, k5s, triple_to_k4s, triple_to_k5s):
        """Compute change in cost when flipping triple t to new_color."""
        delta = 0
        
        # K4s containing t
        for k4_idx in triple_to_k4s[t]:
            idxs = k4s[k4_idx]
            was_all_red = all(colors[i] == 0 for i in idxs)
            if was_all_red:
                if new_color != 0:
                    delta -= 1
            else:
                if new_color == 0:
                    others_all_red = all(colors[i] == 0 for i in idxs if i != t)
                    if others_all_red:
                        delta += 1
        
        # K5s containing t
        for k5_idx in triple_to_k5s[t]:
            idxs = k5s[k5_idx]
            was_all_blue = all(colors[i] == 1 for i in idxs)
            if was_all_blue:
                if new_color != 1:
                    delta -= 1
            else:
                if new_color == 1:
                    others_all_blue = all(colors[i] == 1 for i in idxs if i != t)
                    if others_all_blue:
                        delta += 1
        
        return delta
    
    # Try backtracking for smaller n first (where it's feasible)
    # Then use SA for larger n
    best_n = 9
    best_colors = [0 if (a+b+c)%3==0 else 1 for a,b,c in combinations(range(9),3)]
    
    # Try n from 10 up to 34
    for n in range(10, 35):
        if time.time() > deadline - 15:
            break
        
        # Allocate time: more time for larger n
        time_budget = min(10.0, (deadline - time.time()) * 0.3)
        
        # Try backtracking first
        try:
            result = backtrack_search(n, time.time() + time_budget * 0.5)
            if result is not None:
                # Verify
                triples, num_triples, _, k4s, k5s, _, _ = build_structure(n)
                colors_np = np.array(result, dtype=np.int8)
                if count_violations(colors_np, k4s, k5s) == 0:
                    best_n = n
                    best_colors = result
                    continue
        except Exception:
            pass
        
        # Try SA if backtracking failed
        remaining = deadline - time.time() - 5
        if remaining > 2:
            try:
                sa_result = simulate_annealing(n, time.time() + remaining * 0.8)
                triples, num_triples, _, k4s, k5s, _, _ = build_structure(n)
                colors_np = np.array(sa_result, dtype=np.int8)
                if count_violations(colors_np, k4s, k5s) == 0:
                    best_n = n
                    best_colors = sa_result
                    continue
            except Exception:
                pass
        
        # If we couldn't find a valid coloring for n, try smaller
        # But we want to maximize n, so keep going if we have time
        if time.time() > deadline - 10:
            break
    
    return {'n': best_n, 'colors': best_colors}
