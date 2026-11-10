"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Pre-compute variable to clause indices mapping
    var_to_clauses = [[] for _ in range(n)]
    for i, c in enumerate(clauses):
        for l in c:
            v = abs(l) - 1
            var_to_clauses[v].append(i)
            
    # Pre-compute clause variables for faster access
    clause_vars = []
    for c in clauses:
        clause_vars.append([abs(l) - 1 for l in c])
        
    # Pre-compute literal signs: 1 for positive, 0 for negative
    # Actually, we can just use the original literals for checking satisfaction
    # assignment[v] == (l > 0)
    
    def is_satisfied_clause(i, assignment):
        c = clauses[i]
        for l in c:
            if assignment[abs(l) - 1] == (l > 0):
                return True
        return False

    def initial_state(assignment):
        """
        Returns (clause_sat, unsat_count)
        clause_sat is a list of booleans indicating if clause i is satisfied
        """
        clause_sat = [False] * m
        unsat_count = 0
        for i in range(m):
            if is_satisfied_clause(i, assignment):
                clause_sat[i] = True
            else:
                unsat_count += 1
        return clause_sat, unsat_count

    def get_unsat_clauses(clause_sat):
        return [i for i, s in enumerate(clause_sat) if not s]

    def walksat(assignment, max_steps, initial_p=0.5):
        clause_sat, unsat_count = initial_state(assignment)
        
        if unsat_count == 0:
            return True, assignment

        p = initial_p
        stagnant = 0
        
        for step in range(max_steps):
            if unsat_count == 0:
                return True, assignment
            
            # Find an unsatisfied clause
            # We can maintain a list of unsat indices, but updating it on every flip is expensive.
            # Since n=50 and m is small (likely < 200 for ratio ~4), scanning is fast.
            # Alternatively, pick a random clause and check if it's unsat.
            # Let's pick a random clause index and check.
            while True:
                ci = random.randint(0, m - 1)
                if not clause_sat[ci]:
                    break
            
            # Get variables in this clause
            vars_in_clause = clause_vars[ci]
            
            # Evaluate best flip
            min_delta = float('inf')
            best_vars = []
            
            for v in vars_in_clause:
                # Simulate flip
                old_val = assignment[v]
                assignment[v] = not old_val
                
                # Update clause_sat for clauses containing v
                affected_clauses = var_to_clauses[v]
                local_unsat_delta = 0
                
                for ci2 in affected_clauses:
                    was_sat = clause_sat[ci2]
                    # Check new satisfaction
                    c = clauses[ci2]
                    is_now_sat = False
                    for l in c:
                        if assignment[abs(l) - 1] == (l > 0):
                            is_now_sat = True
                            break
                    
                    if was_sat and not is_now_sat:
                        local_unsat_delta += 1
                        clause_sat[ci2] = False
                    elif not was_sat and is_now_sat:
                        local_unsat_delta -= 1
                        clause_sat[ci2] = True
                    # If state didn't change, no delta
                
                new_unsat_count = unsat_count + local_unsat_delta
                delta = local_unsat_delta
                
                if delta < min_delta:
                    min_delta = delta
                    best_vars = [v]
                elif delta == min_delta:
                    best_vars.append(v)
                
                # Revert flip
                assignment[v] = old_val
                # Revert clause_sat changes
                for ci2 in affected_clauses:
                    # We need to restore the previous state.
                    # Since we only changed it if it flipped, we can just re-evaluate or store old states.
                    # Storing old states is safer.
                    pass # We'll handle revert differently below for efficiency/correctness

            # The above simulation is a bit messy with reverts. 
            # Let's implement a cleaner 'evaluate_flip' that doesn't modify global state, 
            # or modifies and reverts carefully.
            
            # Actually, for n=50, we can just compute the delta analytically without modifying global state.
            # Delta for flipping v in clause ci (which is unsat):
            # 1. Clause ci becomes sat if v is the only false literal in ci? No, we are flipping v.
            #    If v was false, it becomes true -> ci becomes sat (delta -1 for ci).
            #    If v was true, it becomes false -> ci remains unsat (delta 0 for ci).
            #    Since ci is unsat, all literals are false. So v is false. Flipping makes it true.
            #    So ci definitely becomes satisfied. Delta for ci is -1.
            
            # 2. Other clauses containing v:
            #    For each other clause cj containing v:
            #    - If cj is currently satisfied:
            #       - If v is the ONLY satisfied literal in cj, flipping v makes cj unsat. (Delta +1)
            #       - Else, cj remains satisfied. (Delta 0)
            #    - If cj is currently unsatisfied:
            #       - v is false (since cj is unsat). Flipping v makes v true.
            #       - cj becomes satisfied. (Delta -1)
            
            # This analytical approach is O(degree(v)) and doesn't require state modification/reversion.
            
            min_delta = float('inf')
            best_vars = []
            
            for v in vars_in_clause:
                delta = -1 # The selected clause ci becomes satisfied
                
                for ci2 in var_to_clauses[v]:
                    if ci2 == ci:
                        continue
                    
                    if clause_sat[ci2]:
                        # Clause is satisfied. Check if v is the only satisfied literal.
                        # If so, flipping v breaks it.
                        c = clauses[ci2]
                        sat_count = 0
                        for l in c:
                            if assignment[abs(l) - 1] == (l > 0):
                                sat_count += 1
                                if sat_count > 1:
                                    break
                        if sat_count == 1:
                            # v must be the one? We need to check if v is satisfied in cj.
                            # We know v is in cj.
                            # Check if v is currently satisfied in cj.
                            l_v = None
                            for l in c:
                                if abs(l) - 1 == v:
                                    l_v = l
                                    break
                            if l_v is not None and assignment[v] == (l_v > 0):
                                # v is satisfied in cj. And it's the only one.
                                # Flipping v will make cj unsat.
                                delta += 1
                    else:
                        # Clause is unsatisfied. v is false in cj.
                        # Flipping v makes v true, so cj becomes satisfied.
                        delta -= 1
                
                if delta < min_delta:
                    min_delta = delta
                    best_vars = [v]
                elif delta == min_delta:
                    best_vars.append(v)
            
            # Choose variable to flip
            if random.random() < p:
                # Random walk
                v_to_flip = random.choice(vars_in_clause)
            else:
                # Greedy
                v_to_flip = random.choice(best_vars)
            
            # Perform the flip and update state
            assignment[v_to_flip] = not assignment[v_to_flip]
            
            # Update clause_sat and unsat_count for affected clauses
            for ci2 in var_to_clauses[v_to_flip]:
                was_sat = clause_sat[ci2]
                c = clauses[ci2]
                is_now_sat = False
                for l in c:
                    if assignment[abs(l) - 1] == (l > 0):
                        is_now_sat = True
                        break
                
                if was_sat and not is_now_sat:
                    unsat_count += 1
                    clause_sat[ci2] = False
                elif not was_sat and is_now_sat:
                    unsat_count -= 1
                    clause_sat[ci2] = True
            
            # Adapt probability p
            if min_delta >= 0:
                stagnant += 1
                # Increase random probability to escape local minima
                p = min(0.8, p + 0.05)
            else:
                stagnant = 0
                # Decrease random probability to exploit
                p = max(0.1, p - 0.02)

        return False, assignment

    # Main loop
    start_time = time.time()
    max_time = 290.0  # Safety margin
    
    best_assignment = None
    best_unsat = float('inf')
    
    while time.time() - start_time < max_time:
        # Random initial assignment
        assignment = [random.random() < 0.5 for _ in range(n)]
        
        # Increase steps per restart slightly
        # Base steps 5000, increase by 1000 each restart, cap at 20000
        # Or just fixed 10000 steps?
        # For 50 vars, 10000 steps is fast.
        steps = 10000
        
        solved, final_assignment = walksat(assignment, steps, initial_p=0.5)
        
        if solved:
            return final_assignment
        
        # Track best unsat count found
        clause_sat, unsat_count = initial_state(final_assignment)
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = final_assignment[:]
            
        # If we are close to solution, maybe extend?
        # For now, just restart.
        
    if best_assignment:
        return best_assignment
    return assignment

# EVOLVE-BLOCK-END
