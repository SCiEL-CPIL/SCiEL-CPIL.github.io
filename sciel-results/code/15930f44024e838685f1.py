"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 290.0
    
    # Preprocessing
    m = len(clauses)
    # clause_data[i] = list of (var_idx, sign) where sign=1 means +var, -1 means -var
    # var_clauses[v] = list of clause indices containing variable v
    clause_data = []
    var_clauses = [[] for _ in range(n)]
    
    for ci, clause in enumerate(clauses):
        cd = []
        for lit in clause:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            cd.append((v, s))
            var_clauses[v].append(ci)
        clause_data.append(cd)
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Helper to check if a clause is satisfied given current assignment
    def is_clause_sat(c_idx, assignment):
        for v, s in clause_data[c_idx]:
            if assignment[v] == (s == 1):
                return True
        return False

    def init_state(assignment):
        """Initialize clause_sat and num_unsat for a given assignment."""
        clause_sat = [False] * m
        num_unsat = 0
        for ci in range(m):
            if is_clause_sat(ci, assignment):
                clause_sat[ci] = True
            else:
                num_unsat += 1
        return clause_sat, num_unsat

    # Main search loop
    # We use multiple restarts. Each restart uses a fixed budget of steps.
    # If we find a solution, return immediately.
    # If we time out, return the best assignment found so far.
    
    max_restarts = 1000 # Will be limited by time
    steps_per_restart = 2000
    
    for restart in range(max_restarts):
        if time.time() >= deadline:
            break
            
        rng = random.Random(restart * 12345 + 6789)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        clause_sat, num_unsat = init_state(assignment)
        
        if num_unsat == 0:
            return assignment
        
        if num_unsat < best_unsat_count:
            best_unsat_count = num_unsat
            best_assignment = assignment[:]
            
        # WalkSAT loop
        for step in range(steps_per_restart):
            if time.time() >= deadline:
                break
                
            if num_unsat == 0:
                return assignment
                
            # Find unsatisfied clauses
            # Optimization: Keep a list of unsat clauses? 
            # For now, scanning m clauses is O(m) = ~200. 
            # With O(K) flip cost, total step cost is ~O(m + K).
            # This is acceptable for 50 vars.
            
            unsat_indices = []
            for ci in range(m):
                if not clause_sat[ci]:
                    unsat_indices.append(ci)
            
            if not unsat_indices:
                return assignment
                
            # Pick a random unsatisfied clause
            ci = unsat_indices[rng.randrange(len(unsat_indices))]
            
            # Compute delta for each literal in this clause
            # Delta = (clauses that become unsat) - (clauses that become sat)
            # We want to minimize delta (maximize gain).
            
            best_var = -1
            best_delta = float('inf')
            best_gain = -1 # Number of clauses satisfied - broken
            
            for v, s in clause_data[ci]:
                # Calculate gain if we flip v
                gain = 0
                for c_idx in var_clauses[v]:
                    # Check if c_idx is currently sat
                    c_sat_now = clause_sat[c_idx]
                    
                    # Check if c_idx will be sat after flipping v
                    # We only need to check if the other literals in c_idx are satisfied
                    # If they are, the clause is sat iff the v-literal is sat.
                    # If they aren't, the clause is unsat regardless of v (unless v satisfies it? No, 3-SAT).
                    # Actually, if other literals are all false, the clause is sat ONLY if v is set to satisfy it.
                    # If other literals contain a true one, the clause is always sat.
                    
                    # Efficient check:
                    # Is the clause satisfied by literals other than v?
                    satisfied_by_others = False
                    for v2, s2 in clause_data[c_idx]:
                        if v2 != v:
                            if assignment[v2] == (s2 == 1):
                                satisfied_by_others = True
                                break
                    
                    if satisfied_by_others:
                        # Clause is always sat, regardless of v
                        continue
                    else:
                        # Clause is sat iff v is set to satisfy it
                        # Current state:
                        if c_sat_now:
                            # It is sat now, so v must be satisfying it.
                            # If we flip v, it becomes unsat.
                            gain -= 1
                        else:
                            # It is unsat now, so v must be NOT satisfying it.
                            # If we flip v, it becomes sat.
                            gain += 1
                
                # Add noise
                noise = rng.random() * 0.5
                score = -gain + noise # Minimize score -> Maximize gain
                
                if score < best_delta:
                    best_delta = score
                    best_var = v
            
            if best_var == -1:
                # Fallback: random flip
                v = rng.randrange(n)
            else:
                v = best_var
                
            # Flip v
            assignment[v] = not assignment[v]
            
            # Update state
            for c_idx in var_clauses[v]:
                # Determine new sat status
                new_sat = False
                for v2, s2 in clause_data[c_idx]:
                    if assignment[v2] == (s2 == 1):
                        new_sat = True
                        break
                
                old_sat = clause_sat[c_idx]
                
                if old_sat and not new_sat:
                    num_unsat += 1
                elif not old_sat and new_sat:
                    num_unsat -= 1
                
                clause_sat[c_idx] = new_sat
            
            if num_unsat == 0:
                return assignment
                
            if num_unsat < best_unsat_count:
                best_unsat_count = num_unsat
                best_assignment = assignment[:]
                
        # End of steps for this restart
        if time.time() >= deadline:
            break

    if best_assignment is not None:
        return best_assignment
        
    # Fallback
    rng = random.Random(0)
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
