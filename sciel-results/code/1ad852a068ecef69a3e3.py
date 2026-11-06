"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math


def solve(instance):
    """
    Solve a planted satisfiable 3-SAT instance using a hybrid approach:
    1. Try a fast randomized restart WalkSAT with phase saving and dynamic parameters.
    2. If that fails, fall back to a more systematic search (DPLL with unit propagation and pure literal elimination).
    
    The instance is expected to be a dict with keys 'n' and 'clauses'.
    'n' is the number of variables (1-based indexing in clauses).
    'clauses' is a list of clauses, each a list of 3 signed integers (literals).
    """
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Helper: check if assignment satisfies all clauses
    def is_satisfied(assignment, clauses):
        for clause in clauses:
            sat = False
            for lit in clause:
                val = assignment[abs(lit) - 1]
                if (lit > 0 and val) or (lit < 0 and not val):
                    sat = True
                    break
            if not sat:
                return False
        return True
    
    # --- Phase 1: Advanced WalkSAT with restarts ---
    # We'll try multiple restarts with increasing effort.
    rng = random.Random(12345)
    
    # Phase saving: keep track of the most recent value each variable took
    phase = [False] * n
    
    # Dynamic WalkSAT parameters
    # p: probability of picking a variable from the clause
    # We can vary this. Let's start with a standard approach but with restarts.
    
    max_restarts = 50
    max_flips_per_restart = 10000
    
    for restart in range(max_restarts):
        # Initialize assignment
        # Use phase saving: start with the phase values, but randomize if no phase info
        assignment = [phase[i] if rng.random() < 0.8 else rng.random() < 0.5 for i in range(n)]
        
        # Count unsatisfied clauses
        def count_unsat(assignment):
            count = 0
            for clause in clauses:
                sat = False
                for lit in clause:
                    val = assignment[abs(lit) - 1]
                    if (lit > 0 and val) or (lit < 0 and not val):
                        sat = True
                        break
                if not sat:
                    count += 1
            return count
        
        # Track best assignment found (lowest unsat count)
        best_unsat = count_unsat(assignment)
        best_assignment = assignment[:]
        
        for flip in range(max_flips_per_restart):
            if best_unsat == 0:
                return best_assignment
            
            # Find an unsatisfied clause
            unsat_clauses = []
            for clause in clauses:
                sat = False
                for lit in clause:
                    val = assignment[abs(lit) - 1]
                    if (lit > 0 and val) or (lit < 0 and not val):
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(clause)
            
            if not unsat_clauses:
                return assignment
            
            clause = rng.choice(unsat_clauses)
            
            # Choose a variable to flip
            # With probability p, pick a random variable from the clause
            # Otherwise, pick the variable that maximizes the number of satisfied clauses (gain)
            p = 0.5  # Can be tuned
            
            if rng.random() < p:
                lit = rng.choice(clause)
            else:
                # Compute gain for each variable in the clause
                best_gain = -1
                best_lit = clause[0]
                for lit in clause:
                    var = abs(lit) - 1
                    # Count how many clauses would become satisfied if we flip this variable
                    gain = 0
                    for c in clauses:
                        # Check if this clause is currently unsatisfied
                        c_sat = False
                        for l2 in c:
                            val2 = assignment[abs(l2) - 1]
                            if (l2 > 0 and val2) or (l2 < 0 and not val2):
                                c_sat = True
                                break
                        if c_sat:
                            continue
                        # Check if flipping var would satisfy this clause
                        # The clause contains lit if any l2 has abs(l2) == abs(lit)
                        for l2 in c:
                            if abs(l2) == abs(lit):
                                # If the literal in the clause matches the value we'd set
                                # After flip, assignment[var] = not assignment[var]
                                new_val = not assignment[var]
                                if (l2 > 0 and new_val) or (l2 < 0 and not new_val):
                                    gain += 1
                                break
                    if gain > best_gain:
                        best_gain = gain
                        best_lit = lit
                lit = best_lit
            
            var = abs(lit) - 1
            old_val = assignment[var]
            assignment[var] = not old_val
            
            # Update phase
            phase[var] = assignment[var]
            
            # Count unsatisfied clauses
            current_unsat = count_unsat(assignment)
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assignment = assignment[:]
    
    # --- Phase 2: DPLL with unit propagation and pure literal elimination ---
    if best_unsat == 0:
        return best_assignment
    
    # If WalkSAT didn't fully solve it, try DPLL
    # This is more expensive but more systematic
    
    def dpll(clauses, assignment):
        # assignment is a dict var -> bool, or None for unassigned
        # clauses is a list of clauses (lists of literals)
        
        # Simplify: unit propagation and pure literal elimination
        # First, check if all clauses are satisfied
        for clause in clauses:
            sat = False
            for lit in clause:
                var = abs(lit)
                if var in assignment:
                    val = assignment[var]
                    if (lit > 0 and val) or (lit < 0 and not val):
                        sat = True
                        break
            if not sat:
                # Check if clause is falsified
                falsified = True
                for lit in clause:
                    var = abs(lit)
                    if var not in assignment:
                        falsified = False
                        break
                if falsified:
                    return None
                # Clause is not satisfied and not falsified, continue
                break
        
        # Unit propagation
        changed = True
        while changed:
            changed = False
            new_clauses = []
            for clause in clauses:
                # Simplify clause by removing falsified literals
                simplified = []
                for lit in clause:
                    var = abs(lit)
                    if var in assignment:
                        val = assignment[var]
                        if (lit > 0 and val) or (lit < 0 and not val):
                            # This literal is satisfied, clause is satisfied
                            simplified = None
                            break
                        # Literal is falsified, skip it
                        continue
                    else:
                        simplified.append(lit)
                if simplified is None:
                    continue  # Clause is satisfied
                if not simplified:
                    # Clause is empty -> conflict
                    return None
                if len(simplified) == 1:
                    # Unit clause
                    lit = simplified[0]
                    var = abs(lit)
                    val = lit > 0
                    if var in assignment:
                        if assignment[var] != val:
                            return None
                    else:
                        assignment[var] = val
                        changed = True
                        # Don't add to new_clauses, it's satisfied
                        continue
                new_clauses.append(simplified)
            clauses = new_clauses
        
        if not clauses:
            return assignment
        
        # Pure literal elimination
        # Find literals that appear only with one sign
        from collections import defaultdict
        lit_count = defaultdict(int)  # lit -> count of occurrences
        for clause in clauses:
            for lit in clause:
                lit_count[lit] += 1
        
        pure_lits = []
        for lit, count in lit_count.items():
            neg_lit = -lit
            if neg_lit not in lit_count or lit_count[neg_lit] == 0:
                pure_lits.append(lit)
        
        if pure_lits:
            for lit in pure_lits:
                var = abs(lit)
                if var not in assignment:
                    assignment[var] = lit > 0
            # Re-simplify
            return dpll(clauses, assignment)
        
        # Choose a variable to branch on
        # Pick a variable that appears in the most clauses
        var_count = defaultdict(int)
        for clause in clauses:
            for lit in clause:
                var_count[abs(lit)] += 1
        
        if not var_count:
            return assignment
        
        # Pick variable with highest count
        var = max(var_count, key=lambda v: var_count[v])
        
        # Try True
        assignment[var] = True
        result = dpll(clauses, assignment)
        if result is not None:
            return result
        
        # Try False
        assignment[var] = False
        result = dpll(clauses, assignment)
        if result is not None:
            return result
        
        # Backtrack
        del assignment[var]
        return None
    
    # Convert clauses to use 1-based variables (already are)
    # DPLL uses a dict for assignment
    assignment_dict = {}
    
    # Limit DPLL to avoid timeout
    # We'll use a simple recursion limit by tracking depth
    import sys
    sys.setrecursionlimit(10000)
    
    result = dpll(clauses, assignment_dict)
    
    if result is not None:
        # Convert dict to list
        final_assignment = [result.get(i + 1, False) for i in range(n)]
        return final_assignment
    
    # If DPLL fails, return the best assignment from WalkSAT
    return best_assignment
# EVOLVE-BLOCK-END
