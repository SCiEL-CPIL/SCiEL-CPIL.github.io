"""WalkSAT with time-budgeted search, clause-weighted flips, and restarts."""
# EVOLVE-BLOCK-START
import time
import random
import math

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    # Deadline: 300s wall clock, keep 10s margin
    deadline = time.time() + 290.0
    
    # Precompute clause data for speed
    # For each clause, store (clause, list of variable indices, list of signs)
    # signs: +1 if literal is positive, -1 if negative
    clause_data = []
    for c in clauses:
        vars_in_c = [abs(l) - 1 for l in c]
        signs = [1 if l > 0 else -1 for l in c]
        clause_data.append((c, vars_in_c, signs))
    
    # Also build variable-to-clause index for faster unsatisfied clause finding
    # var_to_clauses[v] = list of indices of clauses containing variable v
    var_to_clauses = [[] for _ in range(n)]
    for ci, (c, vars_in_c, signs) in enumerate(clause_data):
        for v in vars_in_c:
            var_to_clauses[v].append(ci)
    
    def check_all(assignment):
        """Check if all clauses are satisfied. Returns True if SAT."""
        for ci, (c, vars_in_c, signs) in enumerate(clause_data):
            sat = False
            for i in range(len(vars_in_c)):
                v = vars_in_c[i]
                s = signs[i]
                if assignment[v] == (s > 0):
                    sat = True
                    break
            if not sat:
                return False
        return True
    
    def count_unsat(assignment):
        """Count unsatisfied clauses."""
        cnt = 0
        for ci, (c, vars_in_c, signs) in enumerate(clause_data):
            sat = False
            for i in range(len(vars_in_c)):
                v = vars_in_c[i]
                s = signs[i]
                if assignment[v] == (s > 0):
                    sat = True
                    break
            if not sat:
                cnt += 1
        return cnt
    
    best_assignment = None
    best_unsat_count = m + 1
    
    # WalkSAT with restarts
    max_restarts = 500
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Initialize random assignment
        assignment = [random.random() < 0.5 for _ in range(n)]
        
        # Track unsatisfied clause count
        unsat_count = count_unsat(assignment)
        if unsat_count == 0:
            return assignment
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
        
        # WalkSAT parameters
        # Use a mix of greedy and random flips
        # The probability of random flip increases as we go (or stays moderate)
        p_random = 0.55  # probability of choosing a random variable from unsat clause
        
        max_steps = 50000  # steps per restart
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Find an unsatisfied clause
            # Pick a random clause and check if it's unsatisfied; if so, use it
            # Otherwise, scan from random start
            found = False
            for _ in range(10):  # try up to 10 random clauses
                ci = random.randint(0, m - 1)
                c, vars_in_c, signs = clause_data[ci]
                sat = False
                for i in range(len(vars_in_c)):
                    v = vars_in_c[i]
                    s = signs[i]
                    if assignment[v] == (s > 0):
                        sat = True
                        break
                if not sat:
                    found = True
                    break
            
            if not found:
                # Scan all clauses to find one unsatisfied
                for ci in range(m):
                    c, vars_in_c, signs = clause_data[ci]
                    sat = False
                    for i in range(len(vars_in_c)):
                        v = vars_in_c[i]
                        s = signs[i]
                        if assignment[v] == (s > 0):
                            sat = True
                            break
                    if not sat:
                        break
                else:
                    # All satisfied
                    return assignment
            
            if found:
                pass  # ci is already set
            # else ci is set from scan
            
            # Choose a variable to flip from this clause
            if random.random() < p_random:
                # Random choice
                idx = random.randint(0, len(vars_in_c) - 1)
            else:
                # Greedy: pick variable that maximizes satisfied clauses
                best_gain = -1
                best_idx = 0
                for i in range(len(vars_in_c)):
                    v = vars_in_c[i]
                    s = signs[i]
                    # If we flip v, what's the gain?
                    # Currently v has value assignment[v]
                    # After flip, assignment[v] becomes not assignment[v]
                    # For clauses containing v, check how many become satisfied/unsatisfied
                    gain = 0
                    for cj in var_to_clauses[v]:
                        c2, vars_in_c2, signs2 = clause_data[cj]
                        # Check if clause cj is currently satisfied
                        cur_sat = False
                        for j in range(len(vars_in_c2)):
                            v2 = vars_in_c2[j]
                            s2 = signs2[j]
                            if assignment[v2] == (s2 > 0):
                                cur_sat = True
                                break
                        # Check if clause cj would be satisfied after flipping v
                        new_sat = False
                        for j in range(len(vars_in_c2)):
                            v2 = vars_in_c2[j]
                            s2 = signs2[j]
                            if v2 == v:
                                new_val = not assignment[v]
                                if new_val == (s2 > 0):
                                    new_sat = True
                                    break
                            else:
                                if assignment[v2] == (s2 > 0):
                                    new_sat = True
                                    break
                        if not cur_sat and new_sat:
                            gain += 1
                        elif cur_sat and not new_sat:
                            gain -= 1
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
                idx = best_idx
            
            v = vars_in_c[idx]
            assignment[v] = not assignment[v]
            
            # Update unsat count: recompute is expensive, so just check periodically
            # For efficiency, we'll check every few steps
            if step % 10 == 0:
                unsat_count = count_unsat(assignment)
                if unsat_count == 0:
                    return assignment
                if unsat_count < best_unsat_count:
                    best_unsat_count = unsat_count
                    best_assignment = assignment[:]
        
        # After max_steps, if not solved, continue to next restart
        # Check final state
        unsat_count = count_unsat(assignment)
        if unsat_count == 0:
            return assignment
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
    
    # Return best assignment found
    if best_assignment is not None:
        return best_assignment
    return assignment
# EVOLVE-BLOCK-END
