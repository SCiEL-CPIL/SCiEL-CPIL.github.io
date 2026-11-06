"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Deadline: 330s wall-clock, use 320s for safety
    deadline = time.time() + 320.0
    
    # Preprocess: convert clauses to 0-based indices with sign
    # clause_vars[i] = list of (var_idx, is_positive)
    clause_vars = []
    for c in clauses:
        cv = []
        for l in c:
            var_idx = abs(l) - 1
            is_pos = l > 0
            cv.append((var_idx, is_pos))
        clause_vars.append(cv)
    
    # Build variable-to-clause index: for each var, list of (clause_idx, is_positive)
    var_clauses = [[] for _ in range(n)]
    for i, cv in enumerate(clause_vars):
        for var_idx, is_pos in cv:
            var_clauses[var_idx].append((i, is_pos))
    
    # Precompute clause sizes for weighting
    clause_sizes = [len(cv) for cv in clause_vars]
    
    def init_clause_state(assignment):
        """Compute sat_count for each clause and unsat set."""
        sat_counts = [0] * m
        unsat_clauses = []
        for i, cv in enumerate(clause_vars):
            cnt = 0
            for var_idx, is_pos in cv:
                if assignment[var_idx] == is_pos:
                    cnt += 1
            sat_counts[i] = cnt
            if cnt == 0:
                unsat_clauses.append(i)
        return sat_counts, unsat_clauses
    
    def walk_sat(assignment, rng, max_flips, deadline, use_weighted_clause=False):
        """Incremental WalkSAT with clause state tracking and set-based unsat tracking."""
        sat_counts, unsat_clauses_list = init_clause_state(assignment)
        unsat_clauses_set = set(unsat_clauses_list)
        if len(unsat_clauses_list) == 0:
            return True
        
        # Precompute variable weights based on clause frequency
        var_weights = [0.0] * n
        for i in range(m):
            for var_idx, is_pos in clause_vars[i]:
                var_weights[var_idx] += 1.0 / max(1, len(var_clauses[var_idx]))
        
        flip_count = 0
        best_unsat_seen = len(unsat_clauses_list)
        stagnation_counter = 0
        last_improvement_flip = 0
        
        while flip_count < max_flips:
            if time.time() > deadline:
                break
            
            if not unsat_clauses_list:
                return True
            
            # Stagnation detection: if no improvement for many flips, restart
            current_unsat = len(unsat_clauses_list)
            if current_unsat < best_unsat_seen:
                best_unsat_seen = current_unsat
                last_improvement_flip = flip_count
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # If stagnating badly, break early to allow restart
            if stagnation_counter > 2000 and flip_count > 500:
                break
            
            # Pick a random unsatisfied clause (weighted by size if enabled)
            if use_weighted_clause and len(unsat_clauses_list) > 1:
                # Weight by inverse clause size (harder clauses get picked more)
                weights = [1.0 / max(1, clause_sizes[ci]) for ci in unsat_clauses_list]
                total_w = sum(weights)
                r = rng.random() * total_w
                cumsum = 0
                ci = unsat_clauses_list[0]
                for idx, w in enumerate(weights):
                    cumsum += w
                    if r <= cumsum:
                        ci = unsat_clauses_list[idx]
                        break
            else:
                ci = rng.choice(unsat_clauses_list)
            
            cv = clause_vars[ci]
            
            # Adaptive noise: decrease over time
            noise = 0.25 * max(0, 1 - flip_count / max_flips)
            
            # For each variable in the unsatisfied clause, compute delta
            best_var = -1
            best_delta = float('inf')
            
            for var_idx, is_pos in cv:
                delta = 0
                for ci2, is_pos2 in var_clauses[var_idx]:
                    old_sat = sat_counts[ci2] > 0
                    # After flip: does this clause remain satisfied?
                    old_contrib = (assignment[var_idx] == is_pos2)
                    new_contrib = (not assignment[var_idx]) == is_pos2
                    
                    if old_sat:
                        # Was satisfied. If var was the only satisfying literal, becomes unsat
                        if old_contrib and not new_contrib:
                            # Check if any other literal satisfies
                            other_sat = False
                            for v2, p2 in clause_vars[ci2]:
                                if v2 != var_idx and assignment[v2] == p2:
                                    other_sat = True
                                    break
                            if not other_sat:
                                delta += 1
                    else:
                        # Was unsatisfied. If flip makes this literal satisfy, becomes sat
                        if not old_contrib and new_contrib:
                            delta -= 1
                
                # Add noise for exploration
                if noise > 0:
                    delta += rng.gauss(0, noise)
                
                if delta < best_delta:
                    best_delta = delta
                    best_var = var_idx
            
            if best_var < 0:
                best_var = cv[0][0]
            
            # Apply flip and update state incrementally
            old_val = assignment[best_var]
            assignment[best_var] = not old_val
            
            for ci2, is_pos2 in var_clauses[best_var]:
                old_contrib = (old_val == is_pos2)
                new_contrib = (not old_val) == is_pos2
                old_sat = sat_counts[ci2] > 0
                
                if old_contrib and not new_contrib:
                    # This literal was satisfying, now it's not
                    sat_counts[ci2] -= 1
                    if sat_counts[ci2] == 0:
                        # Became unsatisfied
                        if ci2 not in unsat_clauses_set:
                            unsat_clauses_set.add(ci2)
                            unsat_clauses_list.append(ci2)
                elif not old_contrib and new_contrib:
                    # This literal was not satisfying, now it is
                    sat_counts[ci2] += 1
                    if sat_counts[ci2] == 1:
                        # Became satisfied (was unsat before since sat_count was 0)
                        if ci2 in unsat_clauses_set:
                            unsat_clauses_set.discard(ci2)
                            # Remove from list efficiently: swap with last
                            try:
                                pos = unsat_clauses_list.index(ci2)
                                unsat_clauses_list[pos] = unsat_clauses_list[-1]
                                unsat_clauses_list.pop()
                            except ValueError:
                                pass
            
            flip_count += 1
            
            # Periodic check for solution
            if flip_count % 32 == 0 and not unsat_clauses_list:
                return True
        
        return len(unsat_clauses_list) == 0
    
    def count_unsat(assignment):
        count = 0
        for cv in clause_vars:
            sat = False
            for var_idx, is_pos in cv:
                if assignment[var_idx] == is_pos:
                    sat = True
                    break
            if not sat:
                count += 1
        return count
    
    def polish(assignment, deadline):
        """Final polish pass: try to fix remaining unsatisfied clauses with targeted flips."""
        sat_counts, unsat_clauses_list = init_clause_state(assignment)
        unsat_clauses_set = set(unsat_clauses_list)
        
        if not unsat_clauses_list:
            return True
        
        max_polish_flips = 5000
        flip_count = 0
        
        while flip_count < max_polish_flips and time.time() < deadline:
            if not unsat_clauses_list:
                return True
            
            # Pick a random unsatisfied clause
            ci = unsat_clauses_list[rng_polish.randint(0, len(unsat_clauses_list) - 1)]
            cv = clause_vars[ci]
            
            # Try each variable in the clause: find one that minimizes delta
            best_var = -1
            best_delta = float('inf')
            
            for var_idx, is_pos in cv:
                delta = 0
                for ci2, is_pos2 in var_clauses[var_idx]:
                    old_sat = sat_counts[ci2] > 0
                    old_contrib = (assignment[var_idx] == is_pos2)
                    new_contrib = (not assignment[var_idx]) == is_pos2
                    
                    if old_sat:
                        if old_contrib and not new_contrib:
                            other_sat = False
                            for v2, p2 in clause_vars[ci2]:
                                if v2 != var_idx and assignment[v2] == p2:
                                    other_sat = True
                                    break
                            if not other_sat:
                                delta += 1
                    else:
                        if not old_contrib and new_contrib:
                            delta -= 1
                
                if delta < best_delta:
                    best_delta = delta
                    best_var = var_idx
            
            if best_var < 0:
                best_var = cv[0][0]
            
            # Only flip if it doesn't make things worse
            if best_delta <= 0:
                old_val = assignment[best_var]
                assignment[best_var] = not old_val
                
                for ci2, is_pos2 in var_clauses[best_var]:
                    old_contrib = (old_val == is_pos2)
                    new_contrib = (not old_val) == is_pos2
                    old_sat = sat_counts[ci2] > 0
                    
                    if old_contrib and not new_contrib:
                        sat_counts[ci2] -= 1
                        if sat_counts[ci2] == 0:
                            if ci2 not in unsat_clauses_set:
                                unsat_clauses_set.add(ci2)
                                unsat_clauses_list.append(ci2)
                    elif not old_contrib and new_contrib:
                        sat_counts[ci2] += 1
                        if sat_counts[ci2] == 1:
                            if ci2 in unsat_clauses_set:
                                unsat_clauses_set.discard(ci2)
                                try:
                                    pos = unsat_clauses_list.index(ci2)
                                    unsat_clauses_list[pos] = unsat_clauses_list[-1]
                                    unsat_clauses_list.pop()
                                except ValueError:
                                    pass
                
                flip_count += 1
            else:
                # No improving move; try a random flip within the clause to escape
                var_idx, is_pos = cv[rng_polish.randint(0, len(cv) - 1)]
                old_val = assignment[var_idx]
                assignment[var_idx] = not old_val
                
                for ci2, is_pos2 in var_clauses[var_idx]:
                    old_contrib = (old_val == is_pos2)
                    new_contrib = (not old_val) == is_pos2
                    old_sat = sat_counts[ci2] > 0
                    
                    if old_contrib and not new_contrib:
                        sat_counts[ci2] -= 1
                        if sat_counts[ci2] == 0:
                            if ci2 not in unsat_clauses_set:
                                unsat_clauses_set.add(ci2)
                                unsat_clauses_list.append(ci2)
                    elif not old_contrib and new_contrib:
                        sat_counts[ci2] += 1
                        if sat_counts[ci2] == 1:
                            if ci2 in unsat_clauses_set:
                                unsat_clauses_set.discard(ci2)
                                try:
                                    pos = unsat_clauses_list.index(ci2)
                                    unsat_clauses_list[pos] = unsat_clauses_list[-1]
                                    unsat_clauses_list.pop()
                                except ValueError:
                                    pass
                
                flip_count += 1
        
        return len(unsat_clauses_list) == 0
    
    # Multiple restarts with different seeds
    best_assignment = None
    best_unsat = float('inf')
    
    restart = 0
    while time.time() < deadline:
        # Each restart: random initial assignment
        rng = random.Random(42 + restart * 7919)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Max flips per restart: start moderate, increase over time
        max_flips = 15000 + restart * 3000
        if restart > 15:
            max_flips = 60000
        
        # Use weighted clause selection for later restarts
        use_weighted = restart % 3 == 2
        
        if walk_sat(assignment, rng, max_flips, deadline, use_weighted):
            return assignment
        
        # Try polish pass
        global rng_polish
        rng_polish = random.Random(1000 + restart * 13)
        if polish(assignment, deadline):
            return assignment
        
        unsat = count_unsat(assignment)
        if unsat < best_unsat:
            best_unsat = unsat
            best_assignment = assignment[:]
        
        restart += 1
    
    # Final attempt: polish the best assignment found
    if best_assignment is not None:
        rng_polish = random.Random(9999)
        if polish(best_assignment, deadline):
            return best_assignment
        return best_assignment
    
    # Fallback: all true
    return [True] * n
# EVOLVE-BLOCK-END
