"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def solve(instance):
    start_time = time.time()
    time_limit = 295.0  # Safety margin before 330s kill
    
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Preprocessing: pure literal elimination and unit propagation
    active_clauses = list(clauses)
    fixed = {}
    
    def preprocess(clause_list, fixed_assign):
        changed = True
        while changed:
            changed = False
            var_polarity = {}
            for c in clause_list:
                for l in c:
                    v = abs(l) - 1
                    p = l > 0
                    if v not in fixed_assign:
                        if v not in var_polarity:
                            var_polarity[v] = set()
                        var_polarity[v].add(p)
            
            for v, pols in var_polarity.items():
                if len(pols) == 1 and v not in fixed_assign:
                    p = pols.pop()
                    fixed_assign[v] = p
                    changed = True
            
            new_clauses = []
            for c in clause_list:
                remaining = []
                satisfied = False
                for l in c:
                    v = abs(l) - 1
                    p = l > 0
                    if v in fixed_assign:
                        if fixed_assign[v] == p:
                            satisfied = True
                            break
                        else:
                            continue
                    remaining.append(l)
                if satisfied:
                    continue
                new_clauses.append(remaining)
            
            for c in new_clauses:
                if len(c) == 0:
                    return None
                if len(c) == 1:
                    l = c[0]
                    v = abs(l) - 1
                    p = l > 0
                    if v not in fixed_assign:
                        fixed_assign[v] = p
                        changed = True
            
            clause_list = new_clauses
        
        return clause_list
    
    fixed_assign = {}
    remaining_clauses = preprocess(active_clauses, fixed_assign)
    
    if remaining_clauses is None:
        return [fixed_assign.get(i, random.random() < 0.5) for i in range(n)]
    
    all_sat = True
    for c in remaining_clauses:
        if len(c) == 0:
            all_sat = False
            break
        sat = False
        for l in c:
            v = abs(l) - 1
            p = l > 0
            if v in fixed_assign and fixed_assign[v] == p:
                sat = True
                break
        if not sat:
            all_sat = False
            break
    
    if all_sat:
        return [fixed_assign.get(i, random.random() < 0.5) for i in range(n)]
    
    unfixed_vars = [i for i in range(n) if i not in fixed_assign]
    var_map = {v: i for i, v in enumerate(unfixed_vars)}
    num_unfixed = len(unfixed_vars)
    
    clause_vars = []
    var_to_clauses = [[] for _ in range(num_unfixed)]
    
    for i, c in enumerate(remaining_clauses):
        cv = []
        for l in c:
            v = abs(l) - 1
            p = l > 0
            if v in var_map:
                local_idx = var_map[v]
                cv.append((local_idx, p))
                var_to_clauses[local_idx].append(i)
        clause_vars.append(cv)
    
    m2 = len(clause_vars)
    
    if num_unfixed == 0:
        return [fixed_assign.get(i, random.random() < 0.5) for i in range(n)]
    
    # Precompute clause polarity lookup: clause_polarity[c_idx][var_idx] = bool (True if positive)
    # This eliminates the inner O(3) scan in damage calculation
    clause_polarity = [{} for _ in range(m2)]
    for c_idx, cv in enumerate(clause_vars):
        for var_idx, is_pos in cv:
            clause_polarity[c_idx][var_idx] = is_pos
    
    def is_satisfied(clause_idx, assignment):
        cv = clause_vars[clause_idx]
        for var_idx, is_pos in cv:
            if assignment[var_idx] == is_pos:
                return True
        return False
    
    def build_full(assignment):
        full_assign = [fixed_assign.get(i, False) for i in range(n)]
        for j, v in enumerate(unfixed_vars):
            full_assign[v] = assignment[j]
        return full_assign
    
    def compute_damage(var_idx, current_clause_idx, assignment):
        """Compute damage of flipping var_idx, using precomputed polarity map."""
        damage = 0
        for c_idx in var_to_clauses[var_idx]:
            if c_idx == current_clause_idx:
                continue
            # Get polarity from precomputed map (O(1) dict lookup)
            p_c = clause_polarity[c_idx].get(var_idx)
            if p_c is None:
                continue
            was_sat = (assignment[var_idx] == p_c)
            if was_sat:
                # Check if any other literal in c_idx is satisfied
                cv = clause_vars[c_idx]
                for v, p in cv:
                    if v != var_idx and assignment[v] == p:
                        break
                else:
                    damage += 1
        return damage
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Strategy 1: WalkSAT with adaptive noise and precomputed polarity
    def walksat_phase(time_budget):
        nonlocal best_assignment, best_unsat_count
        max_steps_per_restart = 800
        max_restarts = 15000
        base_noise = 0.25
        
        while time.time() - start_time < time_budget:
            assignment = [random.random() < 0.5 for _ in range(num_unfixed)]
            
            unsat_set = set()
            for i in range(m2):
                if not is_satisfied(i, assignment):
                    unsat_set.add(i)
            
            if not unsat_set:
                return build_full(assignment)
            
            current_unsat = len(unsat_set)
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]
            
            steps = 0
            no_improve_steps = 0
            prev_unsat = current_unsat
            
            while steps < max_steps_per_restart and time.time() - start_time < time_budget:
                if not unsat_set:
                    return build_full(assignment)
                
                idx = random.choice(list(unsat_set))
                clause = clause_vars[idx]
                
                current_noise = base_noise * (1.0 + no_improve_steps / 150.0)
                current_noise = min(current_noise, 0.85)
                
                if random.random() < current_noise:
                    var_idx, is_pos = random.choice(clause)
                else:
                    best_var_idx = -1
                    best_damage = float('inf')
                    
                    for var_idx, is_pos in clause:
                        damage = compute_damage(var_idx, idx, assignment)
                        if damage < best_damage:
                            best_damage = damage
                            best_var_idx = var_idx
                            best_is_pos = is_pos
                    
                    if best_var_idx == -1:
                        var_idx, is_pos = random.choice(clause)
                    else:
                        var_idx = best_var_idx
                        is_pos = best_is_pos
                
                assignment[var_idx] = not assignment[var_idx]
                
                for c_idx in var_to_clauses[var_idx]:
                    was_unsat = c_idx in unsat_set
                    is_now_sat = is_satisfied(c_idx, assignment)
                    is_now_unsat = not is_now_sat
                    if was_unsat and not is_now_unsat:
                        unsat_set.remove(c_idx)
                    elif not was_unsat and is_now_unsat:
                        unsat_set.add(c_idx)
                
                steps += 1
                current_unsat = len(unsat_set)
                
                if current_unsat < prev_unsat:
                    no_improve_steps = 0
                    prev_unsat = current_unsat
                else:
                    no_improve_steps += 1
                
                if current_unsat < best_unsat_count:
                    best_unsat_count = current_unsat
                    best_assignment = assignment[:]
                
                if no_improve_steps > 400:
                    break
        
        return None
    
    # Strategy 2: Simulated Annealing
    def simulated_annealing_phase(time_budget):
        nonlocal best_assignment, best_unsat_count
        assignment = [random.random() < 0.5 for _ in range(num_unfixed)]
        
        unsat_set = set()
        for i in range(m2):
            if not is_satisfied(i, assignment):
                unsat_set.add(i)
        
        current_unsat = len(unsat_set)
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
        
        T = 5.0
        cooling = 0.9995
        min_T = 0.05
        steps = 0
        max_steps = 50000
        
        while steps < max_steps and T > min_T and time.time() - start_time < time_budget:
            if not unsat_set:
                return build_full(assignment)
            
            # Pick random variable (not restricted to unsat clauses)
            var_idx = random.randint(0, num_unfixed - 1)
            
            old_val = assignment[var_idx]
            new_val = not old_val
            assignment[var_idx] = new_val
            
            # Compute delta
            delta = 0
            for c_idx in var_to_clauses[var_idx]:
                was_unsat = c_idx in unsat_set
                is_now_sat = is_satisfied(c_idx, assignment)
                is_now_unsat = not is_now_sat
                if was_unsat and not is_now_unsat:
                    delta -= 1
                    unsat_set.remove(c_idx)
                elif not was_unsat and is_now_unsat:
                    delta += 1
                    unsat_set.add(c_idx)
            
            current_unsat = len(unsat_set)
            
            if delta <= 0:
                # Accept improvement
                if current_unsat < best_unsat_count:
                    best_unsat_count = current_unsat
                    best_assignment = assignment[:]
            else:
                # Accept with probability exp(-delta/T)
                if random.random() < math.exp(-delta / T):
                    if current_unsat < best_unsat_count:
                        best_unsat_count = current_unsat
                        best_assignment = assignment[:]
                else:
                    # Reject: revert
                    assignment[var_idx] = old_val
                    for c_idx in var_to_clauses[var_idx]:
                        was_unsat = c_idx in unsat_set
                        is_now_sat = is_satisfied(c_idx, assignment)
                        is_now_unsat = not is_now_sat
                        if was_unsat and not is_now_unsat:
                            unsat_set.remove(c_idx)
                        elif not was_unsat and is_now_unsat:
                            unsat_set.add(c_idx)
            
            T *= cooling
            steps += 1
        
        return None
    
    # Strategy 3: Tabu Search
    def tabu_search_phase(time_budget):
        nonlocal best_assignment, best_unsat_count
        assignment = [random.random() < 0.5 for _ in range(num_unfixed)]
        
        unsat_set = set()
        for i in range(m2):
            if not is_satisfied(i, assignment):
                unsat_set.add(i)
        
        current_unsat = len(unsat_set)
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
        
        tabu_tenure = 7
        tabu_until = [0] * num_unfixed
        steps = 0
        max_steps = 30000
        no_improve = 0
        
        while steps < max_steps and time.time() - start_time < time_budget:
            if not unsat_set:
                return build_full(assignment)
            
            # Pick random unsat clause
            idx = random.choice(list(unsat_set))
            clause = clause_vars[idx]
            
            best_var_idx = -1
            best_damage = float('inf')
            best_new_unsat = float('inf')
            
            for var_idx, is_pos in clause:
                if steps < tabu_until[var_idx]:
                    continue
                damage = compute_damage(var_idx, idx, assignment)
                # Compute resulting unsat count
                new_unsat = current_unsat - (1 if idx in unsat_set else 0) + damage
                # Actually idx is in unsat_set, so flipping may satisfy it
                # Let's compute more carefully
                if damage < best_damage:
                    best_damage = damage
                    best_var_idx = var_idx
            
            if best_var_idx == -1:
                # All tabu, pick best anyway
                for var_idx, is_pos in clause:
                    damage = compute_damage(var_idx, idx, assignment)
                    if damage < best_damage:
                        best_damage = damage
                        best_var_idx = var_idx
            
            if best_var_idx == -1:
                var_idx, is_pos = random.choice(clause)
            else:
                var_idx = best_var_idx
            
            assignment[var_idx] = not assignment[var_idx]
            tabu_until[var_idx] = steps + tabu_tenure
            
            for c_idx in var_to_clauses[var_idx]:
                was_unsat = c_idx in unsat_set
                is_now_sat = is_satisfied(c_idx, assignment)
                is_now_unsat = not is_now_sat
                if was_unsat and not is_now_unsat:
                    unsat_set.remove(c_idx)
                elif not was_unsat and is_now_unsat:
                    unsat_set.add(c_idx)
            
            current_unsat = len(unsat_set)
            steps += 1
            
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]
                no_improve = 0
            else:
                no_improve += 1
            
            # Adapt tabu tenure
            if no_improve > 100:
                tabu_tenure = max(3, tabu_tenure - 1)
                no_improve = 0
        
        return None
    
    # Run multi-strategy portfolio with time division
    # Phase 1: WalkSAT (50% of time)
    phase1_time = start_time + (time_limit - (time.time() - start_time)) * 0.45
    result = walksat_phase(phase1_time)
    if result is not None:
        return result
    
    # Phase 2: Simulated Annealing (25% of time)
    phase2_time = start_time + (time_limit - (time.time() - start_time)) * 0.25
    result = simulated_annealing_phase(phase2_time)
    if result is not None:
        return result
    
    # Phase 3: Tabu Search (25% of time)
    phase3_time = start_time + (time_limit - (time.time() - start_time)) * 0.25
    result = tabu_search_phase(phase3_time)
    if result is not None:
        return result
    
    # Final polish on best found
    if best_assignment is not None and best_unsat_count > 0 and best_unsat_count < m2 * 0.15:
        assignment = best_assignment[:]
        unsat_set = set()
        for i in range(m2):
            if not is_satisfied(i, assignment):
                unsat_set.add(i)
        
        polish_steps = 0
        max_polish_steps = 8000
        while polish_steps < max_polish_steps and time.time() - start_time < time_limit - 2.0:
            if not unsat_set:
                return build_full(assignment)
            
            idx = random.choice(list(unsat_set))
            clause = clause_vars[idx]
            
            best_var_idx = -1
            best_damage = float('inf')
            
            for var_idx, is_pos in clause:
                damage = compute_damage(var_idx, idx, assignment)
                if damage < best_damage:
                    best_damage = damage
                    best_var_idx = var_idx
                    best_is_pos = is_pos
            
            if best_var_idx == -1:
                var_idx, is_pos = random.choice(clause)
            else:
                var_idx = best_var_idx
            
            assignment[var_idx] = not assignment[var_idx]
            
            for c_idx in var_to_clauses[var_idx]:
                was_unsat = c_idx in unsat_set
                is_now_sat = is_satisfied(c_idx, assignment)
                is_now_unsat = not is_now_sat
                if was_unsat and not is_now_unsat:
                    unsat_set.remove(c_idx)
                elif not was_unsat and is_now_unsat:
                    unsat_set.add(c_idx)
            
            polish_steps += 1
    
    if best_assignment:
        return build_full(best_assignment)
    
    return [fixed_assign.get(i, random.random() < 0.5) for i in range(n)]

# EVOLVE-BLOCK-END
