"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import sys

def solve(instance):
    start_time = time.time()
    time_limit = 290.0  # Safety margin (300s kill, leave buffer)
    
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Preprocess: build clause variables and variable-to-clause index
    clause_vars = []  # list of lists of (var_idx, is_pos)
    var_to_clauses = [[] for _ in range(n)]
    
    for i, c in enumerate(clauses):
        cv = []
        for l in c:
            var_idx = abs(l) - 1
            is_pos = l > 0
            cv.append((var_idx, is_pos))
            var_to_clauses[var_idx].append(i)
        clause_vars.append(cv)
    
    # === PREPROCESSING: Pure Literal Elimination + Unit Propagation ===
    # This simplifies the problem before search
    fixed = [None] * n  # None = unassigned, True/False = fixed
    active_clauses = list(range(m))
    pure_eliminated = [False] * m
    unit_propagated = [False] * m
    
    def propagate_units():
        """Apply unit propagation repeatedly until fixpoint."""
        changed = True
        while changed:
            changed = False
            for ci in active_clauses:
                if pure_eliminated[ci] or unit_propagated[ci]:
                    continue
                cv = clause_vars[ci]
                # Count unassigned literals and check if any are already satisfied
                unsat_count = 0
                unassigned_vars = []
                for var_idx, is_pos in cv:
                    if fixed[var_idx] is not None:
                        if fixed[var_idx] == is_pos:
                            # Clause is satisfied
                            unit_propagated[ci] = True
                            break
                        else:
                            # Literal is false
                            unsat_count += 1
                    else:
                        unassigned_vars.append((var_idx, is_pos))
                else:
                    # No break, so clause not satisfied
                    if unsat_count + len(unassigned_vars) == 0:
                        # All literals false - UNSAT (shouldn't happen for planted SAT)
                        return False
                    if len(unassigned_vars) == 1 and unsat_count == 0:
                        # Unit clause: force the remaining literal
                        var_idx, is_pos = unassigned_vars[0]
                        if fixed[var_idx] is None:
                            fixed[var_idx] = is_pos
                            changed = True
                            unit_propagated[ci] = True
                    elif len(unassigned_vars) == 0 and unsat_count > 0:
                        # All literals false - UNSAT
                        return False
        return True
    
    def eliminate_pure_literals():
        """Eliminate pure literals: if a variable appears only in one polarity, fix it."""
        changed = True
        while changed:
            changed = False
            # Count polarities for each variable in active clauses
            pos_count = [0] * n
            neg_count = [0] * n
            for ci in active_clauses:
                if pure_eliminated[ci] or unit_propagated[ci]:
                    continue
                for var_idx, is_pos in clause_vars[ci]:
                    if fixed[var_idx] is not None:
                        continue
                    if is_pos:
                        pos_count[var_idx] += 1
                    else:
                        neg_count[var_idx] += 1
            
            for var_idx in range(n):
                if fixed[var_idx] is not None:
                    continue
                if pos_count[var_idx] > 0 and neg_count[var_idx] == 0:
                    fixed[var_idx] = True
                    changed = True
                elif neg_count[var_idx] > 0 and pos_count[var_idx] == 0:
                    fixed[var_idx] = False
                    changed = True
    
    # Run preprocessing
    eliminate_pure_literals()
    if not propagate_units():
        # Shouldn't happen for planted SAT, but fallback
        return [fixed[i] if fixed[i] is not None else random.random() < 0.5 for i in range(n)]
    eliminate_pure_literals()
    if not propagate_units():
        return [fixed[i] if fixed[i] is not None else random.random() < 0.5 for i in range(n)]
    
    # Check if all clauses are satisfied by preprocessing
    all_satisfied = True
    for ci in range(m):
        cv = clause_vars[ci]
        sat = False
        for var_idx, is_pos in cv:
            if fixed[var_idx] is not None:
                if fixed[var_idx] == is_pos:
                    sat = True
                    break
            # If unassigned, can't be satisfied yet
        if not sat:
            # Check if all assigned literals are false and there are unassigned ones
            all_false = True
            has_unassigned = False
            for var_idx, is_pos in cv:
                if fixed[var_idx] is not None:
                    if fixed[var_idx] == is_pos:
                        all_false = False
                        break
                else:
                    has_unassigned = True
            if all_false and not has_unassigned:
                all_satisfied = False
                break
            elif all_false and has_unassigned:
                all_satisfied = False
                break
            else:
                # Some literal might be satisfied by unassigned vars, or clause is trivially ok
                pass
    
    # Build reduced problem: only unassigned variables and unsatisfied clauses
    var_map = {}  # old var_idx -> new var_idx
    new_n = 0
    for i in range(n):
        if fixed[i] is None:
            var_map[i] = new_n
            new_n += 1
    
    if new_n == 0:
        # All variables fixed
        result = [fixed[i] if fixed[i] is not None else False for i in range(n)]
        return result
    
    # Build reduced clauses
    reduced_clauses = []
    reduced_clause_vars = []
    reduced_var_to_clauses = [[] for _ in range(new_n)]
    
    for ci in range(m):
        cv = clause_vars[ci]
        new_cv = []
        for var_idx, is_pos in cv:
            if fixed[var_idx] is not None:
                if fixed[var_idx] == is_pos:
                    # Clause is satisfied, skip
                    new_cv = None
                    break
                else:
                    # Literal is false, skip
                    continue
            else:
                new_var_idx = var_map[var_idx]
                new_cv.append((new_var_idx, is_pos))
        
        if new_cv is None:
            continue  # Clause satisfied
        if len(new_cv) == 0:
            # All literals false - UNSAT (shouldn't happen)
            continue
        if len(new_cv) > 3:
            # Shouldn't happen for 3-SAT, but keep as is
            pass
        
        if len(new_cv) >= 1:
            reduced_clauses.append(ci)
            reduced_clause_vars.append(new_cv)
            for nv, _ in new_cv:
                reduced_var_to_clauses[nv].append(len(reduced_clause_vars) - 1)
    
    new_m = len(reduced_clauses)
    
    # If no reduced clauses, all satisfied
    if new_m == 0:
        result = [fixed[i] if fixed[i] is not None else False for i in range(n)]
        return result
    
    # Now run WalkSAT on the reduced problem
    rng = random.Random()
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    def walk_sat():
        nonlocal best_assignment, best_unsat_count
        
        # Random initial assignment for reduced variables
        assignment = [rng.random() < 0.5 for _ in range(new_n)]
        
        # Build support counts
        support = [0] * new_m
        for i in range(new_m):
            cv = reduced_clause_vars[i]
            for var_idx, is_pos in cv:
                if assignment[var_idx] == is_pos:
                    support[i] += 1
        
        # Build initial unsat set
        unsat_set = set()
        for i in range(new_m):
            if support[i] == 0:
                unsat_set.add(i)
        
        if not unsat_set:
            return assignment
        
        current_unsat = len(unsat_set)
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
        
        # Adaptive walk parameters
        max_steps = 80000
        noise = 0.45
        
        for step in range(max_steps):
            if time.time() - start_time > time_limit:
                break
            
            if not unsat_set:
                return assignment
            
            # Pick a random unsat clause
            idx = rng.choice(list(unsat_set))
            clause = reduced_clause_vars[idx]
            
            # Try to find a flip with minimum damage
            best_var_idx = -1
            best_damage = float('inf')
            best_is_pos = False
            
            for var_idx, is_pos in clause:
                # Compute damage
                damage = 0
                for c_idx in reduced_var_to_clauses[var_idx]:
                    if c_idx == idx:
                        continue
                    if support[c_idx] == 1:
                        cv = reduced_clause_vars[c_idx]
                        for v, p in cv:
                            if assignment[v] == p:
                                if v == var_idx:
                                    damage += 1
                                break
                
                if damage < best_damage:
                    best_damage = damage
                    best_var_idx = var_idx
                    best_is_pos = is_pos
            
            # Decide which variable to flip
            if best_var_idx == -1:
                var_idx, is_pos = rng.choice(clause)
            elif rng.random() < noise or best_damage > 10:
                var_idx, is_pos = rng.choice(clause)
            else:
                var_idx = best_var_idx
                is_pos = best_is_pos
            
            # Perform flip
            old_val = assignment[var_idx]
            assignment[var_idx] = not old_val
            
            # Update support counts and unsat_set incrementally
            for c_idx in reduced_var_to_clauses[var_idx]:
                cv = reduced_clause_vars[c_idx]
                p_c = None
                for v, p in cv:
                    if v == var_idx:
                        p_c = p
                        break
                
                if p_c is None:
                    continue
                
                old_support = support[c_idx]
                if old_val == p_c:
                    support[c_idx] = old_support - 1
                    if old_support == 1:
                        unsat_set.add(c_idx)
                else:
                    support[c_idx] = old_support + 1
                    if old_support == 0:
                        unsat_set.discard(c_idx)
            
            if not unsat_set:
                return assignment
            
            current_unsat = len(unsat_set)
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]
            
            # Adaptive noise
            if step % 500 == 0 and step > 0:
                if current_unsat >= best_unsat_count:
                    noise = min(0.65, noise + 0.02)
                else:
                    noise = max(0.3, noise - 0.01)
        
        return None
    
    # Main loop with restarts
    restarts = 0
    max_restarts = 300
    
    while time.time() - start_time < time_limit and restarts < max_restarts:
        result = walk_sat()
        if result is not None:
            # Map back to original assignment
            full_assignment = [fixed[i] if fixed[i] is not None else False for i in range(n)]
            for old_var, new_var in var_map.items():
                full_assignment[old_var] = result[new_var]
            return full_assignment
        restarts += 1
    
    # Final polish: try to improve best_assignment
    if best_assignment is not None and best_unsat_count > 0:
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 5.0:
            assignment = best_assignment[:]
            
            support = [0] * new_m
            unsat_set = set()
            for i in range(new_m):
                cv = reduced_clause_vars[i]
                for var_idx, is_pos in cv:
                    if assignment[var_idx] == is_pos:
                        support[i] += 1
                if support[i] == 0:
                    unsat_set.add(i)
            
            current_unsat = len(unsat_set)
            
            max_polish_steps = 15000
            for step in range(max_polish_steps):
                if time.time() - start_time > time_limit:
                    break
                
                if not unsat_set:
                    full_assignment = [fixed[i] if fixed[i] is not None else False for i in range(n)]
                    for old_var, new_var in var_map.items():
                        full_assignment[old_var] = assignment[new_var]
                    return full_assignment
                
                idx = rng.choice(list(unsat_set))
                clause = reduced_clause_vars[idx]
                
                best_var_idx = -1
                best_damage = float('inf')
                best_is_pos = False
                
                for var_idx, is_pos in clause:
                    damage = 0
                    for c_idx in reduced_var_to_clauses[var_idx]:
                        if c_idx == idx:
                            continue
                        if support[c_idx] == 1:
                            cv = reduced_clause_vars[c_idx]
                            for v, p in cv:
                                if assignment[v] == p:
                                    if v == var_idx:
                                        damage += 1
                                    break
                    
                    if damage < best_damage:
                        best_damage = damage
                        best_var_idx = var_idx
                        best_is_pos = is_pos
                
                if best_var_idx == -1:
                    var_idx, is_pos = rng.choice(clause)
                elif rng.random() < 0.3:
                    var_idx, is_pos = rng.choice(clause)
                else:
                    var_idx = best_var_idx
                    is_pos = best_is_pos
                
                old_val = assignment[var_idx]
                assignment[var_idx] = not old_val
                
                for c_idx in reduced_var_to_clauses[var_idx]:
                    cv = reduced_clause_vars[c_idx]
                    p_c = None
                    for v, p in cv:
                        if v == var_idx:
                            p_c = p
                            break
                    
                    if p_c is None:
                        continue
                    
                    old_support = support[c_idx]
                    if old_val == p_c:
                        support[c_idx] = old_support - 1
                        if old_support == 1:
                            unsat_set.add(c_idx)
                    else:
                        support[c_idx] = old_support + 1
                        if old_support == 0:
                            unsat_set.discard(c_idx)
                
                current_unsat = len(unsat_set)
                if current_unsat < best_unsat_count:
                    best_unsat_count = current_unsat
                    best_assignment = assignment[:]
            
            # Return best found
            full_assignment = [fixed[i] if fixed[i] is not None else False for i in range(n)]
            for old_var, new_var in var_map.items():
                full_assignment[old_var] = best_assignment[new_var]
            return full_assignment
    
    # Fallback: return fixed + random for unassigned
    return [fixed[i] if fixed[i] is not None else random.random() < 0.5 for i in range(n)]

# EVOLVE-BLOCK-END
