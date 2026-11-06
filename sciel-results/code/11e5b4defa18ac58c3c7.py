"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def solve(instance):
    start_time = time.time()
    time_limit = 285.0  # Safety margin before 300s kill
    
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Precompute clause structure
    clause_vars = []
    clause_signs = []
    for c in clauses:
        vars_list = []
        signs_list = []
        for lit in c:
            v = abs(lit) - 1
            s = lit > 0
            vars_list.append(v)
            signs_list.append(s)
        clause_vars.append(vars_list)
        clause_signs.append(signs_list)
    
    # Build variable-to-clauses mapping
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for v in clause_vars[i]:
            var_to_clauses[v].append(i)
    
    # Variable occurrence counts
    var_counts = np.zeros(n, dtype=np.int32)
    for v in range(n):
        var_counts[v] = len(var_to_clauses[v])
    
    def is_satisfied(assignment, ci):
        """Check if clause ci is satisfied under assignment."""
        for j in range(len(clause_vars[ci])):
            v = clause_vars[ci][j]
            s = clause_signs[ci][j]
            if assignment[v] == s:
                return True
        return False
    
    def count_unsat(assignment):
        """Count unsatisfied clauses."""
        cnt = 0
        for i in range(m):
            if not is_satisfied(assignment, i):
                cnt += 1
        return cnt
    
    def get_unsat_clauses(assignment):
        """Return list of unsatisfied clause indices."""
        unsat = []
        for i in range(m):
            if not is_satisfied(assignment, i):
                unsat.append(i)
        return unsat
    
    # WalkSAT with restarts and damage-minimizing literal selection
    rng = random.Random(42)
    
    best_assignment = None
    best_unsat_count = m + 1
    
    # Parameters
    p_random = 0.5  # probability of random flip vs damage-minimizing
    base_steps = 5000
    
    max_restarts = 200
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Random initialization
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        unsat_list = get_unsat_clauses(assignment)
        unsat_count = len(unsat_list)
        
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment.copy()
        
        # Track which clauses are unsatisfied (as a set for O(1) lookup)
        unsat_set = set(unsat_list)
        
        max_steps = base_steps + restart * 1000
        steps = 0
        
        while steps < max_steps:
            if time.time() - start_time > time_limit:
                break
            
            if len(unsat_set) == 0:
                return assignment
            
            if len(unsat_set) < best_unsat_count:
                best_unsat_count = len(unsat_set)
                best_assignment = assignment.copy()
            
            # Pick an unsatisfied clause
            if len(unsat_set) == 1:
                ci = next(iter(unsat_set))
            else:
                # Weighted pick: prefer clauses with fewer satisfied literals
                # (all are unsat, so all have 0 satisfied literals)
                # Just pick randomly
                ci = rng.choice(list(unsat_set))
            
            # Select literal to flip using damage minimization
            cl_vars = clause_vars[ci]
            cl_signs = clause_signs[ci]
            
            # For each literal in the clause, compute the damage
            # Damage = number of currently-satisfied clauses that would become unsatisfied
            #         minus number of currently-unsatisfied clauses (other than ci) that would become satisfied
            
            best_var = cl_vars[0]
            best_damage = float('inf')
            
            for idx in range(len(cl_vars)):
                v = cl_vars[idx]
                s = cl_signs[idx]
                
                # Current value of variable v
                cur_val = assignment[v]
                
                # After flip:
                new_val = not cur_val
                
                # This literal's satisfaction changes:
                # Currently: cur_val == s
                # After: new_val == s
                # Since the clause is unsatisfied, all literals are currently false.
                # So cur_val != s for all literals in this clause.
                # After flip, new_val == s might be true (if we flip to satisfy)
                
                # Compute damage: for each clause j containing v (j != ci),
                # check if flipping v changes its satisfaction status.
                
                damage = 0
                gain = 0
                
                for j in var_to_clauses[v]:
                    if j == ci:
                        continue
                    
                    # Check if clause j is currently satisfied
                    # We need to know if clause j is in unsat_set
                    j_unsat = j in unsat_set
                    
                    # Determine if flipping v changes clause j's status
                    # Find the literal of v in clause j
                    # We need to know the sign of v in clause j
                    # Let's find it
                    j_sat_before = not j_unsat
                    
                    # After flipping v, is clause j still satisfied?
                    # We need to check all literals in clause j
                    # But we only need to check if the literal involving v changes
                    # and whether that changes the overall satisfaction
                    
                    # For clause j, the literal (v, s_j) where s_j is the sign of v in clause j
                    # Currently: if s_j matches cur_val, this literal is satisfied
                    # After flip: if s_j matches new_val, this literal is satisfied
                    
                    # Find s_j
                    s_j = None
                    for k in range(len(clause_vars[j])):
                        if clause_vars[j][k] == v:
                            s_j = clause_signs[j][k]
                            break
                    
                    if s_j is None:
                        continue  # shouldn't happen
                    
                    lit_sat_before = (cur_val == s_j)
                    lit_sat_after = (new_val == s_j)
                    
                    if j_sat_before:
                        # Clause j is currently satisfied
                        # After flip, is it still satisfied?
                        # If the literal (v, s_j) was satisfied and becomes unsatisfied,
                        # we need to check if another literal in clause j is satisfied
                        if lit_sat_before and not lit_sat_after:
                            # Check if clause j is still satisfied without this literal
                            still_sat = False
                            for k in range(len(clause_vars[j])):
                                if clause_vars[j][k] == v:
                                    continue
                                if assignment[clause_vars[j][k]] == clause_signs[j][k]:
                                    still_sat = True
                                    break
                            if not still_sat:
                                damage += 1
                    else:
                        # Clause j is currently unsatisfied
                        # After flip, does it become satisfied?
                        if not lit_sat_before and lit_sat_after:
                            # This literal becomes satisfied, so clause j becomes satisfied
                            gain += 1
                
                net_damage = damage - gain
                
                if net_damage < best_damage:
                    best_damage = net_damage
                    best_var = v
            
            # With probability p_random, flip a random literal from the clause
            if rng.random() < p_random:
                idx = rng.randint(0, len(cl_vars) - 1)
                v_to_flip = cl_vars[idx]
            else:
                v_to_flip = best_var
            
            # Flip the variable
            old_val = assignment[v_to_flip]
            assignment[v_to_flip] = not assignment[v_to_flip]
            new_val = assignment[v_to_flip]
            
            # Update unsat_set for all clauses containing v_to_flip
            for j in var_to_clauses[v_to_flip]:
                # Find the sign of v_to_flip in clause j
                s_j = None
                for k in range(len(clause_vars[j])):
                    if clause_vars[j][k] == v_to_flip:
                        s_j = clause_signs[j][k]
                        break
                
                if s_j is None:
                    continue
                
                # Determine if clause j's satisfaction status changed
                # Before flip: was it satisfied?
                # After flip: is it satisfied?
                
                # We can check directly
                j_sat_after = is_satisfied(assignment, j)
                
                if j_sat_after:
                    unsat_set.discard(j)
                else:
                    unsat_set.add(j)
            
            steps += 1
    
    # Return best assignment found
    if best_assignment is not None:
        return best_assignment
    else:
        return [rng.random() < 0.5 for _ in range(n)]

# EVOLVE-BLOCK-END
