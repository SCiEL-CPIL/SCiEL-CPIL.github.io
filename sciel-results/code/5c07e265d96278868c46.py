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
    
    def is_prime(x):
        if x < 2:
            return False
        for i in range(2, int(np.sqrt(x)) + 1):
            if x % i == 0:
                return False
        return True
    
    def is_prime(x):
        if x < 2:
            return False
        for i in range(2, int(np.sqrt(x)) + 1):
            if x % i == 0:
                return False
        return True
    
    def quadratic_residue_coloring(n, mod=None):
        """
        Deterministic cyclic construction using quadratic residues.
        For vertices 0..n-1, color triple (a,b,c) based on whether
        the differences (b-a) and (c-b) mod p are quadratic residues.
        """
        if mod is None:
            mod = n
            while not is_prime(mod):
                mod += 1
        
        # Precompute quadratic residues mod p
        qr = set()
        for i in range(1, mod):
            qr.add((i * i) % mod)
        
        triples = list(combinations(range(n), 3))
        colors = []
        for (a, b, c) in triples:
            d1 = (b - a) % mod
            d2 = (c - b) % mod
            # Color red (0) if both differences are quadratic residues
            # Color blue (1) otherwise
            if d1 in qr and d2 in qr:
                colors.append(0)
            else:
                colors.append(1)
        return colors
    
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
        """Simulated annealing with quadratic residue initialization."""
        triples, num_triples, triple_to_idx, k4s, k5s, triple_to_k4s, triple_to_k5s = build_structure(n)
        
        if init_colors is None:
            # Try multiple initialization strategies
            candidates = []
            
            # Quadratic residue constructions with different primes
            for mod_offset in range(0, 30):
                mod = n + mod_offset
                while not is_prime(mod):
                    mod += 1
                qr = set()
                for i in range(1, mod):
                    qr.add((i * i) % mod)
                
                c = np.zeros(num_triples, dtype=np.int8)
                for idx, (a, b, cc) in enumerate(triples):
                    d1 = (b - a) % mod
                    d2 = (cc - b) % mod
                    if d1 in qr and d2 in qr:
                        c[idx] = 0
                    else:
                        c[idx] = 1
                candidates.append(c)
            
            # Simple patterns
            patterns = [
                lambda a,b,c: 0 if (a+b+c) % 3 == 0 else 1,
                lambda a,b,c: 0 if (a+b+c) % 2 == 0 else 1,
                lambda a,b,c: 0 if (a*b+c) % 3 == 0 else 1,
                lambda a,b,c: 0 if (a*b+c) % 2 == 0 else 1,
                lambda a,b,c: 0 if (a+b) % 3 == 0 else 1,
                lambda a,b,c: 0 if (a*c) % 3 == 0 else 1,
            ]
            for pat in patterns:
                c = np.array([pat(a,b,c) for a,b,c in triples], dtype=np.int8)
                candidates.append(c)
            
            # Random initialization
            candidates.append(np.random.randint(0, 2, num_triples, dtype=np.int8))
            
            # Pick best candidate
            best_init = None
            best_init_cost = float('inf')
            for c in candidates:
                cost = count_violations(c, k4s, k5s)
                if cost < best_init_cost:
                    best_init_cost = cost
                    best_init = c
            
            colors = best_init.copy()
        else:
            colors = init_colors.copy()
        
        current_cost = count_violations(colors, k4s, k5s)
        best_cost = current_cost
        best_colors = colors.copy()
        
        # Adaptive simulated annealing with restarts
        temp = 1.0
        cool_rate = 0.9995
        min_temp = 0.01
        no_improve_steps = 0
        max_no_improve = 10000
        restart_count = 0
        max_restarts = 5
        
        while time.time() < time_limit and restart_count < max_restarts:
            # Run SA until no improvement for max_no_improve steps or time runs out
            while time.time() < time_limit:
                if no_improve_steps > max_no_improve:
                    break
                    
                # Pick a random triple to flip
                t = np.random.randint(num_triples)
                old_color = colors[t]
                new_color = 1 - old_color
                
                delta_cost = compute_delta(t, new_color, colors, k4s, k5s, triple_to_k4s, triple_to_k5s)
                new_cost = current_cost + delta_cost
                
                if new_cost <= current_cost:
                    accept = True
                elif temp > min_temp:
                    # Accept with probability exp(-delta/temp)
                    accept = np.random.rand() < np.exp(-delta_cost / max(temp, 1e-10))
                else:
                    accept = False
                
                if accept:
                    colors[t] = new_color
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colors = colors.copy()
                        no_improve_steps = 0
                        if best_cost == 0:
                            return best_colors.tolist()
                    else:
                        no_improve_steps += 1
                else:
                    no_improve_steps += 1
                
                # Adaptive cooling
                if no_improve_steps > max_no_improve:
                    break
                else:
                    temp *= cool_rate
                
                if temp < min_temp:
                    temp = min_temp
            
            if best_cost == 0:
                return best_colors.tolist()
            
            # Restart: try a different initialization
            restart_count += 1
            if restart_count < max_restarts:
                # Try a random restart
                colors = np.random.randint(0, 2, num_triples, dtype=np.int8)
                current_cost = count_violations(colors, k4s, k5s)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()
                temp = 1.0
                no_improve_steps = 0
        
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
        old_color = colors[t]
        
        # K4s containing t
        for k4_idx in triple_to_k4s[t]:
            idxs = k4s[k4_idx]
            # Count red edges in this K4
            red_count = 0
            for i in idxs:
                if i == t:
                    if old_color == 0:
                        red_count += 1
                else:
                    if colors[i] == 0:
                        red_count += 1
            
            was_violation = (red_count == 4)
            
            # After flip
            if new_color == 0:
                new_red_count = red_count + (1 if old_color != 0 else 0)
            else:
                new_red_count = red_count - (1 if old_color == 0 else 0)
            
            new_violation = (new_red_count == 4)
            
            if new_violation and not was_violation:
                delta += 1
            elif was_violation and not new_violation:
                delta -= 1
        
        # K5s containing t
        for k5_idx in triple_to_k5s[t]:
            idxs = k5s[k5_idx]
            # Count blue edges in this K5
            blue_count = 0
            for i in idxs:
                if i == t:
                    if old_color == 1:
                        blue_count += 1
                else:
                    if colors[i] == 1:
                        blue_count += 1
            
            was_violation = (blue_count == 10)
            
            # After flip
            if new_color == 1:
                new_blue_count = blue_count + (1 if old_color != 1 else 0)
            else:
                new_blue_count = blue_count - (1 if old_color == 1 else 0)
            
            new_violation = (new_blue_count == 10)
            
            if new_violation and not was_violation:
                delta += 1
            elif was_violation and not new_violation:
                delta -= 1
        
        return delta
    
    # Use quadratic residue construction as baseline
    best_n = 9
    best_colors = quadratic_residue_coloring(9)
    
    # Try to find valid colorings for increasing n
    # Start with quadratic residue, then use SA to refine
    for n in range(10, 35):
        if time.time() > deadline - 30:
            break
        
        # Try quadratic residue first (fast, deterministic)
        qr_colors = quadratic_residue_coloring(n)
        triples, num_triples, _, k4s, k5s, _, _ = build_structure(n)
        colors_np = np.array(qr_colors, dtype=np.int8)
        cost = count_violations(colors_np, k4s, k5s)
        
        if cost == 0:
            best_n = n
            best_colors = qr_colors
            continue
        
        # Use SA to try to fix violations
        remaining = deadline - time.time() - 15
        if remaining > 10:
            try:
                sa_result = simulate_annealing(n, time.time() + remaining * 0.9, init_colors=qr_colors)
                colors_np = np.array(sa_result, dtype=np.int8)
                if count_violations(colors_np, k4s, k5s) == 0:
                    best_n = n
                    best_colors = sa_result
                    continue
            except Exception:
                pass
        
        # If SA didn't find valid coloring, try without init
        remaining = deadline - time.time() - 15
        if remaining > 10:
            try:
                sa_result = simulate_annealing(n, time.time() + remaining * 0.9)
                colors_np = np.array(sa_result, dtype=np.int8)
                if count_violations(colors_np, k4s, k5s) == 0:
                    best_n = n
                    best_colors = sa_result
                    continue
            except Exception:
                pass
        
        # If we can't find valid coloring for this n, stop trying larger n
        # (since if n fails, larger n will likely also fail)
        if cost > 100:  # Too many violations, unlikely to fix
            break
    
    return {'n': best_n, 'colors': best_colors}
