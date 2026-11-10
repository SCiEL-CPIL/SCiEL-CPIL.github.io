"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Deadline with safety margin
    deadline = time.time() + 290.0 

    # Precompute variable to clauses mapping
    # var_to_clauses[v] = list of (clause_idx, sign) where sign is True for +v, False for -v
    var_to_clauses = [[] for _ in range(n)]
    for ci, c in enumerate(clauses):
        for l in c:
            v = abs(l) - 1
            sign = l > 0
            var_to_clauses[v].append((ci, sign))

    def run_walksat(rng, max_steps, initial_assignment=None):
        if initial_assignment is None:
            assignment = [rng.random() < 0.5 for _ in range(n)]
        else:
            assignment = initial_assignment[:]
            
        # Initialize clause satisfaction counts
        # clause_sat_count[ci] = number of satisfied literals in clause ci
        clause_sat_count = [0] * m
        unsat_clauses = set()
        
        for ci, c in enumerate(clauses):
            sat_cnt = 0
            for l in c:
                v = abs(l) - 1
                sign = l > 0
                if assignment[v] == sign:
                    sat_cnt += 1
            clause_sat_count[ci] = sat_cnt
            if sat_cnt == 0:
                unsat_clauses.add(ci)
                
        if not unsat_clauses:
            return assignment, True

        # Adaptive noise parameters
        base_noise = 0.5
        min_noise = 0.1
        max_noise = 0.9
        noise = base_noise
        
        # Activity tracking for variables (optional refinement, but let's keep it simple first: 
        # standard impact-based WalkSAT is usually sufficient with good restarts)
        # We will add a simple "stuck" detection to increase noise.
        last_improvement_step = 0
        min_unsat_so_far = len(unsat_clauses)
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if not unsat_clauses:
                return assignment, True

            # Pick a random unsatisfied clause
            clause_idx = rng.choice(list(unsat_clauses))
            clause = clauses[clause_idx]

            # Evaluate candidates (the 3 literals in the clause)
            best_var = -1
            best_score = -float('inf')
            
            for l in clause:
                v = abs(l) - 1
                target_sign = l > 0 # The sign we want v to have to satisfy this clause
                
                # Calculate Gain: How many unsat clauses would become sat if we flip v to target_sign?
                # Note: We are flipping v. Currently v has some value.
                # If we set v = target_sign, does that satisfy other unsat clauses?
                # An unsat clause C' is satisfied by setting v=target_sign if:
                # 1. C' is currently unsat.
                # 2. C' contains v.
                # 3. The literal of v in C' has sign target_sign.
                #    (Because if C' is unsat, all literals are false. If we make v true for that literal, 
                #     and no other literal in C' becomes true... wait.
                #     WalkSAT flips v. This might break other clauses.
                #     The "Gain" is typically: number of unsat clauses that contain v with the sign 
                #     that would be satisfied by the flip.
                #     However, simply satisfying that literal in an unsat clause makes it sat ONLY IF 
                #     no other literal in that clause was already sat (impossible, it's unsat) 
                #     AND the flip doesn't break it (it can't break it, it was false, now true).
                #     So yes, if an unsat clause contains v with sign S, and we flip v to satisfy S, 
                #     that clause becomes satisfied.
                
                gain = 0
                # Calculate Loss: How many sat clauses would become unsat?
                # A sat clause C' becomes unsat if:
                # 1. C' is currently sat.
                # 2. C' contains v.
                # 3. The literal of v in C' is the ONLY satisfied literal in C'.
                #    (i.e., clause_sat_count[C'] == 1 and the satisfied literal is v with the current sign of v).
                
                loss = 0
                
                # Iterate clauses containing v
                for (ci2, sign2) in var_to_clauses[v]:
                    if ci2 in unsat_clauses:
                        # If we flip v to target_sign, does it satisfy this clause?
                        # It satisfies it if sign2 == target_sign.
                        if sign2 == target_sign:
                            gain += 1
                    else:
                        # Clause is currently satisfied.
                        # Will it become unsatisfied?
                        # It becomes unsat if the literal v in this clause was the ONLY satisfied one.
                        # And the flip makes it false.
                        # Current value of v is assignment[v].
                        # The literal in clause is sign2.
                        # It is currently satisfied if assignment[v] == sign2.
                        # If we flip v, it becomes not assignment[v].
                        # If assignment[v] == sign2, it was satisfied. After flip, it is not.
                        # So it loses 1 sat literal.
                        # If clause_sat_count[ci2] == 1, it becomes unsat.
                        
                        if assignment[v] == sign2:
                            if clause_sat_count[ci2] == 1:
                                loss += 1

                score = gain - noise * loss
                if score > best_score:
                    best_score = score
                    best_var = v

            if best_var < 0:
                break

            # Flip best_var
            old_val = assignment[best_var]
            assignment[best_var] = not old_val
            
            # Update clause_sat_count and unsat_clauses
            for (ci2, sign2) in var_to_clauses[best_var]:
                # Determine change in sat count
                # Old state: assignment[best_var] was old_val.
                # Literal sign is sign2.
                # Was it satisfied before?
                was_sat = (old_val == sign2)
                # Is it satisfied now?
                is_sat_now = (assignment[best_var] == sign2)
                
                if was_sat and not is_sat_now:
                    clause_sat_count[ci2] -= 1
                    if clause_sat_count[ci2] == 0:
                        unsat_clauses.add(ci2)
                elif not was_sat and is_sat_now:
                    clause_sat_count[ci2] += 1
                    if clause_sat_count[ci2] > 0:
                        unsat_clauses.discard(ci2)
                        
            # Adaptive Noise
            # If we are stuck (no reduction in unsat count for a while), increase noise
            current_unsat = len(unsat_clauses)
            if current_unsat < min_unsat_so_far:
                min_unsat_so_far = current_unsat
                last_improvement_step = step
                # Decrease noise slightly if we are improving
                noise = max(min_noise, noise * 0.98)
            else:
                # If no improvement for 100 steps, increase noise to escape
                if step - last_improvement_step > 100:
                    noise = min(max_noise, noise * 1.05)
                    last_improvement_step = step # Reset timer to prevent continuous explosion

        return assignment, (len(unsat_clauses) == 0)

    rng = random.Random(0)
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Restart strategy
    # Start with small budgets, increase if not solved
    restarts = 0
    base_budget = 1000
    
    while time.time() < deadline:
        if restarts > 500:
            break
            
        budget = base_budget + restarts * 50
        assignment, solved = run_walksat(rng, budget)
        
        if solved:
            return assignment
            
        # Track best
        # Recalculate unsat count for best tracking (or pass it out)
        # For efficiency, let's just return the count from run_walksat? 
        # The function returns bool. Let's modify it to return count or just recalculate.
        # Recalculating is O(M) which is fine.
        unsat_count = 0
        for c in clauses:
            ok = False
            for l in c:
                if assignment[abs(l)-1] == (l>0):
                    ok = True
                    break
            if not ok:
                unsat_count += 1
                
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
            
        # If close to solution, do a polish pass with higher noise or different strategy?
        # Or just continue restarting.
        
        restarts += 1
        # Increase base budget slowly
        if restarts % 10 == 0:
            base_budget += 100

    if best_assignment is not None:
        return best_assignment
    return [rng.random() < 0.5 for _ in range(n)]

# EVOLVE-BLOCK-END
