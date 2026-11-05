import numpy as np
import time
from itertools import combinations
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def is_prime(x):
        if x < 2:
            return False
        if x < 4:
            return True
        if x % 2 == 0 or x % 3 == 0:
            return False
        i = 5
        while i * i <= x:
            if x % i == 0 or x % (i + 2) == 0:
                return False
            i += 6
        return True

    def build_structure(n):
        """Build all the structural data needed for constraint propagation."""
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # For each triple, which K4s does it belong to?
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

    def quadratic_residue_coloring(n, mod=None):
        """
        Deterministic cyclic construction using quadratic residues.
        """
        if mod is None:
            mod = n
            while not is_prime(mod):
                mod += 1
        
        qr = set()
        for i in range(1, mod):
            qr.add((i * i) % mod)
        
        triples = list(combinations(range(n), 3))
        colors = []
        for (a, b, c) in triples:
            d1 = (b - a) % mod
            d2 = (c - b) % mod
            if d1 in qr and d2 in qr:
                colors.append(0)
            else:
                colors.append(1)
        return colors

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
            red_count = 0
            for i in idxs:
                if i == t:
                    if old_color == 0:
                        red_count += 1
                else:
                    if colors[i] == 0:
                        red_count += 1
            
            was_violation = (red_count == 4)
            
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
            blue_count = 0
            for i in idxs:
                if i == t:
                    if old_color == 1:
                        blue_count += 1
                else:
                    if colors[i] == 1:
                        blue_count += 1
            
            was_violation = (blue_count == 10)
            
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

    def simulate_annealing(n, time_limit, init_colors=None):
        """Simulated annealing with quadratic residue initialization."""
        triples, num_triples, triple_to_idx, k4s, k5s, triple_to_k4s, triple_to_k5s = build_structure(n)
        
        if init_colors is None:
            candidates = []
            for mod_offset in range(0, 20):
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
            
            patterns = [
                lambda a,b,c: 0 if (a+b+c) % 3 == 0 else 1,
                lambda a,b,c: 0 if (a+b+c) % 2 == 0 else 1,
                lambda a,b,c: 0 if (a*b+c) % 3 == 0 else 1,
            ]
            for pat in patterns:
                c = np.array([pat(a,b,c) for a,b,c in triples], dtype=np.int8)
                candidates.append(c)
            
            candidates.append(np.random.randint(0, 2, num_triples, dtype=np.int8))
            
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
        
        temp = 2.0
        cool_rate = 0.999
        min_temp = 0.01
        no_improve_steps = 0
        max_no_improve = 50000
        
        while time.time() < time_limit:
            t = np.random.randint(num_triples)
            old_color = colors[t]
            new_color = 1 - old_color
            
            delta_cost = compute_delta(t, new_color, colors, k4s, k5s, triple_to_k4s, triple_to_k5s)
            new_cost = current_cost + delta_cost
            
            if new_cost <= current_cost:
                accept = True
            elif temp > min_temp:
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
            
            if no_improve_steps > max_no_improve:
                temp *= 0.9
                no_improve_steps = 0
            else:
                temp *= cool_rate
            
            if temp < min_temp:
                temp = min_temp
        
        return best_colors.tolist()

    def backtrack_search(n, time_limit):
        """
        Backtracking search with constraint propagation.
        """
        triples, num_triples, triple_to_idx, k4s, k5s, triple_to_k4s, triple_to_k5s = build_structure(n)
        
        colors = [-1] * num_triples
        
        k4_red_count = [0] * len(k4s)
        k4_blue_count = [0] * len(k4s)
        k4_unassigned = [4] * len(k4s)
        
        k5_red_count = [0] * len(k5s)
        k5_blue_count = [0] * len(k5s)
        k5_unassigned = [10] * len(k5s)
        
        assignment_stack = []
        
        def is_feasible_k4(k4_idx):
            if k4_red_count[k4_idx] == 4:
                return False
            return True
        
        def is_feasible_k5(k5_idx):
            if k5_blue_count[k5_idx] == 10:
                return False
            return True
        
        def assign(triple_idx, color):
            colors[triple_idx] = color
            assignment_stack.append((triple_idx, color))
            
            for k4_idx in triple_to_k4s[triple_idx]:
                k4_unassigned[k4_idx] -= 1
                if color == 0:
                    k4_red_count[k4_idx] += 1
                else:
                    k4_blue_count[k4_idx] += 1
                if not is_feasible_k4(k4_idx):
                    return False
            
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
            _, color = assignment_stack.pop()
            colors[triple_idx] = -1
            
            for k4_idx in triple_to_k4s[triple_idx]:
                k4_unassigned[k4_idx] += 1
                if color == 0:
                    k4_red_count[k4_idx] -= 1
                else:
                    k4_blue_count[k4_idx] -= 1
            
            for k5_idx in triple_to_k5s[triple_idx]:
                k5_unassigned[k5_idx] += 1
                if color == 0:
                    k5_red_count[k5_idx] -= 1
                else:
                    k5_blue_count[k5_idx] -= 1
        
        def get_forced_assignments():
            forced = {}
            
            for t in range(num_triples):
                if colors[t] != -1:
                    continue
                
                for k4_idx in triple_to_k4s[t]:
                    if k4_unassigned[k4_idx] == 1 and k4_red_count[k4_idx] == 3:
                        forced[t] = 1
                        break
                
                if t in forced:
                    continue
                
                for k5_idx in triple_to_k5s[t]:
                    if k5_unassigned[k5_idx] == 1 and k5_blue_count[k5_idx] == 9:
                        forced[t] = 0
                        break
            
            return forced
        
        def search(depth, time_limit):
            if time.time() > time_limit:
                return None
            
            for k4_idx in range(len(k4s)):
                if k4_red_count[k4_idx] == 4:
                    return None
            for k5_idx in range(len(k5s)):
                if k5_blue_count[k5_idx] == 10:
                    return None
            
            forced = get_forced_assignments()
            
            while forced:
                t, color = forced.popitem()
                if colors[t] != -1:
                    continue
                if not assign(t, color):
                    while assignment_stack:
                        unassign(assignment_stack[-1][0])
                    return None
                new_forced = get_forced_assignments()
                forced.update(new_forced)
            
            unassigned = [i for i in range(num_triples) if colors[i] == -1]
            if not unassigned:
                return colors[:]
            
            best_t = unassigned[0]
            best_constraint_count = 0
            for t in unassigned:
                constraint_count = len(triple_to_k4s[t]) + len(triple_to_k5s[t])
                if constraint_count > best_constraint_count:
                    best_constraint_count = constraint_count
                    best_t = t
            
            for color in [0, 1]:
                if time.time() > time_limit:
                    return None
                
                if assign(best_t, color):
                    result = search(depth + 1, time_limit)
                    if result is not None:
                        return result
                    unassign(best_t)
            
            return None
        
        result = search(0, time_limit)
        if result is not None:
            return result
        return None

    # Main search logic
    best_n = 9
    best_colors = quadratic_residue_coloring(9)
    
    # Try to find valid colorings for increasing n
    for n in range(10, 35):
        if time.time() > deadline - 20:
            break
        
        qr_colors = quadratic_residue_coloring(n)
        triples, num_triples, _, k4s, k5s, _, _ = build_structure(n)
        colors_np = np.array(qr_colors, dtype=np.int8)
        cost = count_violations(colors_np, k4s, k5s)
        
        if cost == 0:
            best_n = n
            best_colors = qr_colors
            continue
        
        remaining = deadline - time.time() - 10
        if remaining > 5:
            try:
                sa_result = simulate_annealing(n, time.time() + remaining * 0.85, init_colors=qr_colors)
                colors_np = np.array(sa_result, dtype=np.int8)
                if count_violations(colors_np, k4s, k5s) == 0:
                    best_n = n
                    best_colors = sa_result
                    continue
            except Exception:
                pass
        
        remaining = deadline - time.time() - 10
        if remaining > 5:
            try:
                sa_result = simulate_annealing(n, time.time() + remaining * 0.85)
                colors_np = np.array(sa_result, dtype=np.int8)
                if count_violations(colors_np, k4s, k5s) == 0:
                    best_n = n
                    best_colors = sa_result
                    continue
            except Exception:
                pass
        
        if cost > 50:
            break
    
    return {'n': best_n, 'colors': best_colors}
