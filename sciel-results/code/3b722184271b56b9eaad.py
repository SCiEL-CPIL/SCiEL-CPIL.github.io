"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def solve(instance):
    start_time = time.time()
    time_limit = 290.0  # Safety margin (330s kill, leave buffer)
    
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
    
    # Build literal polarity lookup: for each clause, store polarity of each var in it
    # clause_polarity[i] = dict mapping var_idx -> is_pos
    clause_polarity = [{} for _ in range(m)]
    for i in range(m):
        for var_idx, is_pos in clause_vars[i]:
            clause_polarity[i][var_idx] = is_pos
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    rng = random.Random()
    
    def stochastic_ac3_propagate(assignment, support, unsat_set, max_rounds=20):
        """
        Stochastic Local Consistency: randomly assign variables to enforce arc consistency.
        For each variable, check if fixing it to a value causes any clause to become
        permanently unsatisfiable (all other literals false). If so, fix the variable
        to the other value. Repeat for multiple rounds.
        Returns (assignment, support, unsat_set, changed)
        """
        nonlocal best_assignment, best_unsat_count
        
        changed = False
        for round_num in range(max_rounds):
            if time.time() - start_time > time_limit:
                break
            
            # Shuffle variable order for stochasticity
            var_order = list(range(n))
            rng.shuffle(var_order)
            
            round_changed = False
            for var_idx in var_order:
                if time.time() - start_time > time_limit:
                    break
                
                # Check if setting var_idx = True causes any clause to become unsat
                # A clause becomes unsat if:
                # 1. It contains var_idx with positive polarity (so setting True satisfies it)
                #    OR it contains var_idx with negative polarity (setting True doesn't help)
                # 2. All other literals in the clause are currently false
                #    AND var_idx's literal in this clause is also false (or not helping)
                
                # Actually, let's think more carefully:
                # For a clause c containing var_idx:
                # If polarity is positive (var_idx in clause as +var_idx):
                #   Setting var_idx = True satisfies the clause
                #   Setting var_idx = False: clause is sat iff some other literal is true
                # If polarity is negative (var_idx in clause as -var_idx):
                #   Setting var_idx = False satisfies the clause
                #   Setting var_idx = True: clause is sat iff some other literal is true
                
                # We want to check: if we fix var_idx to value v, does any clause become
                # permanently unsatisfiable?
                # A clause c is permanently unsatisfiable if all its literals are false
                # given the current assignment and the fix.
                
                # For each clause containing var_idx:
                # Let pol = polarity of var_idx in clause c
                # If we set var_idx = True:
                #   The literal (var_idx, pol) is true iff pol == True
                #   Other literals: check against current assignment
                #   Clause is unsat iff (pol != True) AND all other literals are false
                # If we set var_idx = False:
                #   The literal (var_idx, pol) is true iff pol == False
                #   Other literals: check against current assignment
                #   Clause is unsat iff (pol != False) AND all other literals are false
                
                # Check for value = True
                true_ok = True
                for c_idx in var_to_clauses[var_idx]:
                    pol = clause_polarity[c_idx][var_idx]
                    if pol == True:
                        continue  # This clause is satisfied by var_idx = True
                    # pol == False, so var_idx = True doesn't satisfy this clause
                    # Check if all other literals are false
                    cv = clause_vars[c_idx]
                    all_other_false = True
                    for v, p in cv:
                        if v == var_idx:
                            continue
                        if assignment[v] == p:
                            all_other_false = False
                            break
                    if all_other_false:
                        true_ok = False
                        break
                
                # Check for value = False
                false_ok = True
                for c_idx in var_to_clauses[var_idx]:
                    pol = clause_polarity[c_idx][var_idx]
                    if pol == False:
                        continue  # This clause is satisfied by var_idx = False
                    # pol == True, so var_idx = False doesn't satisfy this clause
                    # Check if all other literals are false
                    cv = clause_vars[c_idx]
                    all_other_false = True
                    for v, p in cv:
                        if v == var_idx:
                            continue
                        if assignment[v] == p:
                            all_other_false = False
                            break
                    if all_other_false:
                        false_ok = False
                        break
                
                if not true_ok and not false_ok:
                    # Conflict - shouldn't happen in SAT, but skip
                    continue
                elif true_ok and not false_ok:
                    # Must set var_idx = True
                    if not assignment[var_idx]:
                        old_val = assignment[var_idx]
                        assignment[var_idx] = True
                        # Update support counts
                        for c_idx in var_to_clauses[var_idx]:
                            pol = clause_polarity[c_idx][var_idx]
                            old_support = support[c_idx]
                            if old_val == pol:
                                support[c_idx] = old_support - 1
                                if old_support == 1:
                                    unsat_set.add(c_idx)
                            else:
                                support[c_idx] = old_support + 1
                                if old_support == 0:
                                    unsat_set.discard(c_idx)
                        round_changed = True
                elif not true_ok and false_ok:
                    # Must set var_idx = False
                    if assignment[var_idx]:
                        old_val = assignment[var_idx]
                        assignment[var_idx] = False
                        # Update support counts
                        for c_idx in var_to_clauses[var_idx]:
                            pol = clause_polarity[c_idx][var_idx]
                            old_support = support[c_idx]
                            if old_val == pol:
                                support[c_idx] = old_support - 1
                                if old_support == 1:
                                    unsat_set.add(c_idx)
                            else:
                                support[c_idx] = old_support + 1
                                if old_support == 0:
                                    unsat_set.discard(c_idx)
                        round_changed = True
                # If both ok, leave as is
            
            if round_changed:
                changed = True
                # Check if solved
                if not unsat_set:
                    return assignment, support, unsat_set, True
            
            if not round_changed:
                break
        
        # Update best
        current_unsat = len(unsat_set)
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
        
        return assignment, support, unsat_set, changed
    
    def walk_sat():
        nonlocal best_assignment, best_unsat_count
        
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Build support counts
        support = [0] * m
        for i in range(m):
            cv = clause_vars[i]
            for var_idx, is_pos in cv:
                if assignment[var_idx] == is_pos:
                    support[i] += 1
        
        # Build initial unsat set
        unsat_set = set()
        for i in range(m):
            if support[i] == 0:
                unsat_set.add(i)
        
        if not unsat_set:
            return assignment
        
        # Apply stochastic AC-3 propagation
        assignment, support, unsat_set, _ = stochastic_ac3_propagate(
            assignment, support, unsat_set, max_rounds=10
        )
        
        if not unsat_set:
            return assignment
        
        current_unsat = len(unsat_set)
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
        
        # Adaptive walk parameters
        max_steps = 50000
        noise = 0.45
        
        for step in range(max_steps):
            if time.time() - start_time > time_limit:
                break
            
            if not unsat_set:
                return assignment
            
            # Every 2000 steps, apply a light AC-3 propagation
            if step > 0 and step % 2000 == 0:
                assignment, support, unsat_set, _ = stochastic_ac3_propagate(
                    assignment, support, unsat_set, max_rounds=3
                )
                if not unsat_set:
                    return assignment
                current_unsat = len(unsat_set)
                if current_unsat < best_unsat_count:
                    best_unsat_count = current_unsat
                    best_assignment = assignment[:]
            
            # Pick a random unsat clause
            idx = rng.choice(list(unsat_set))
            clause = clause_vars[idx]
            
            # Try to find a flip with minimum damage
            best_var_idx = -1
            best_damage = float('inf')
            best_is_pos = False
            
            for var_idx, is_pos in clause:
                damage = 0
                for c_idx in var_to_clauses[var_idx]:
                    if c_idx == idx:
                        continue
                    if support[c_idx] == 1:
                        pol = clause_polarity[c_idx][var_idx]
                        if assignment[var_idx] == pol:
                            damage += 1
                
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
            for c_idx in var_to_clauses[var_idx]:
                pol = clause_polarity[c_idx][var_idx]
                old_support = support[c_idx]
                if old_val == pol:
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
            
            if step % 500 == 0 and step > 0:
                if current_unsat >= best_unsat_count:
                    noise = min(0.6, noise + 0.02)
                else:
                    noise = max(0.3, noise - 0.01)
        
        return None
    
    # Main loop with restarts
    restarts = 0
    max_restarts = 200
    
    while time.time() - start_time < time_limit and restarts < max_restarts:
        result = walk_sat()
        if result is not None:
            return result
        restarts += 1
    
    # Final polish
    if best_assignment is not None and best_unsat_count > 0:
        polish_time_limit = time_limit - (time.time() - start_time)
        if polish_time_limit > 5.0:
            assignment = best_assignment[:]
            
            support = [0] * m
            unsat_set = set()
            for i in range(m):
                cv = clause_vars[i]
                for var_idx, is_pos in cv:
                    if assignment[var_idx] == is_pos:
                        support[i] += 1
                if support[i] == 0:
                    unsat_set.add(i)
            
            # Apply AC-3
            assignment, support, unsat_set, _ = stochastic_ac3_propagate(
                assignment, support, unsat_set, max_rounds=20
            )
            
            if not unsat_set:
                return assignment
            
            current_unsat = len(unsat_set)
            
            max_polish_steps = 10000
            for step in range(max_polish_steps):
                if time.time() - start_time > time_limit:
                    break
                
                if not unsat_set:
                    return assignment
                
                idx = rng.choice(list(unsat_set))
                clause = clause_vars[idx]
                
                best_var_idx = -1
                best_damage = float('inf')
                best_is_pos = False
                
                for var_idx, is_pos in clause:
                    damage = 0
                    for c_idx in var_to_clauses[var_idx]:
                        if c_idx == idx:
                            continue
                        if support[c_idx] == 1:
                            pol = clause_polarity[c_idx][var_idx]
                            if assignment[var_idx] == pol:
                                damage += 1
                    
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
                
                for c_idx in var_to_clauses[var_idx]:
                    pol = clause_polarity[c_idx][var_idx]
                    old_support = support[c_idx]
                    if old_val == pol:
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
            
            # Final AC-3 pass
            assignment, support, unsat_set, _ = stochastic_ac3_propagate(
                assignment, support, unsat_set, max_rounds=10
            )
            if not unsat_set:
                return assignment
            current_unsat = len(unsat_set)
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]
        
        return best_assignment
    
    return [random.random() < 0.5 for _ in range(n)]

# EVOLVE-BLOCK-END
