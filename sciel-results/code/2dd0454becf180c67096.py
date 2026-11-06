# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 290  # safety margin before 300s limit
    
    # Preprocess: convert clauses to 0-based signed literals
    clauses_internal = []
    for c in clauses:
        clauses_internal.append(tuple((abs(l) - 1, l > 0) for l in c))
    
    m = len(clauses_internal)
    
    # Build index: for each variable, which clauses contain it
    var_to_clauses = [[] for _ in range(n)]
    for i, c in enumerate(clauses_internal):
        for var, _ in c:
            var_to_clauses[var].append(i)
    
    rng = random.Random(42)
    
    def is_satisfied(assignment, c):
        for var, is_pos in c:
            if assignment[var] == is_pos:
                return True
        return False

    def initial_unsat_state(assignment):
        """Calculate initial unsat list, count, and boolean array."""
        unsat_list = []
        is_unsat = [False] * m
        for i, c in enumerate(clauses_internal):
            if not is_satisfied(assignment, c):
                unsat_list.append(i)
                is_unsat[i] = True
        return unsat_list, len(unsat_list), is_unsat

    # Main search loop
    best_unsat_count = m + 1
    best_assignment = None
    
    max_restarts = 200
    base_steps = 2000
    
    restart = 0
    while restart < max_restarts and time.time() < deadline:
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Calculate initial unsat state
        unsat_list, unsat_count, is_unsat = initial_unsat_state(assignment)
        
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = list(assignment)
        
        # Maintain var_unsat_count: number of unsat clauses containing var i
        var_unsat_count = [0] * n
        for ci in unsat_list:
            c = clauses_internal[ci]
            for var, _ in c:
                var_unsat_count[var] += 1
        
        steps = int(base_steps * (1.1 ** restart))
        steps = min(steps, 50000)
        
        # Clean up unsat_list periodically to prevent it from growing too large with stale entries
        clean_counter = 0
        max_clean_interval = 1000
        
        for step in range(steps):
            if time.time() >= deadline:
                break
            
            if unsat_count == 0:
                return assignment
            
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = list(assignment)
            
            # Pick a random unsatisfied clause
            # Handle stale entries in unsat_list
            chosen_ci = None
            attempts = 0
            while attempts < 10:
                if not unsat_list:
                    break
                idx = rng.randrange(len(unsat_list))
                ci_candidate = unsat_list[idx]
                if is_unsat[ci_candidate]:
                    chosen_ci = ci_candidate
                    break
                attempts += 1
            
            if chosen_ci is None:
                # Fallback: clean the list
                unsat_list = [i for i in unsat_list if is_unsat[i]]
                if not unsat_list:
                    break # Should be 0 unsat
                chosen_ci = rng.choice(unsat_list)
            
            c = clauses_internal[chosen_ci]
            
            # Find false literals in this clause
            false_vars = []
            for var, is_pos in c:
                if assignment[var] != is_pos:
                    false_vars.append((var, is_pos))
            
            if not false_vars:
                continue # Should not happen
            
            # Select variable to flip
            if len(false_vars) == 1:
                var, is_pos = false_vars[0]
            else:
                # Weighted random selection
                weights = [var_unsat_count[v] for v, _ in false_vars]
                total_w = sum(weights)
                if total_w == 0:
                    var, is_pos = rng.choice(false_vars)
                else:
                    r = rng.random() * total_w
                    cum = 0
                    chosen_idx = 0
                    for i, w in enumerate(weights):
                        cum += w
                        if r <= cum:
                            chosen_idx = i
                            break
                    var, is_pos = false_vars[chosen_idx]
            
            # Flip the variable
            old_val = assignment[var]
            new_val = is_pos
            assignment[var] = new_val
            
            # Update unsat state
            # Iterate through clauses containing 'var'
            for ci2 in var_to_clauses[var]:
                c2 = clauses_internal[ci2]
                
                # Determine if c2 is satisfied in the NEW assignment
                is_sat_now = False
                for v, p in c2:
                    if assignment[v] == p:
                        is_sat_now = True
                        break
                
                was_unsat = is_unsat[ci2]
                is_unsat_now = not is_sat_now
                
                if was_unsat == is_unsat_now:
                    continue
                
                # Status changed
                if was_unsat:
                    # Was unsat, now sat
                    is_unsat[ci2] = False
                    unsat_count -= 1
                    # Update var_unsat_count for all vars in c2
                    for v, _ in c2:
                        var_unsat_count[v] -= 1
                    # Do NOT remove from unsat_list (lazy deletion)
                else:
                    # Was sat, now unsat
                    is_unsat[ci2] = True
                    unsat_count += 1
                    unsat_list.append(ci2)
                    # Update var_unsat_count for all vars in c2
                    for v, _ in c2:
                        var_unsat_count[v] += 1
            
            # Periodic cleanup
            clean_counter += 1
            if clean_counter >= max_clean_interval:
                if len(unsat_list) > 2 * unsat_count + 10:
                    unsat_list = [i for i in unsat_list if is_unsat[i]]
                clean_counter = 0

        restart += 1
        
    if best_assignment:
        return best_assignment
    return assignment
# EVOLVE-BLOCK-END
