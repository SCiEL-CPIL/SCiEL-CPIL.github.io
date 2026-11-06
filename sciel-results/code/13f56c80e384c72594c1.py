"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # 10s safety margin
    
    # Pre-process: convert clauses to lists of ints if they are tuples
    clauses = [list(c) for c in clauses]
    
    # DPLL Solver
    # assignment: list of bools, None means unassigned
    # We use 0/1 for False/True internally for speed, or just bools.
    # Let's use a dict or list for assignment status.
    # assignment[i] = True/False/None
    
    assignment = [None] * n
    
    def check_satisfied(clause):
        for lit in clause:
            var = abs(lit) - 1
            val = assignment[var]
            if val is None:
                continue
            is_true = (lit > 0)
            if val == is_true:
                return True
        return False

    def check_empty(clause):
        # A clause is empty (unsatisfiable) if all its literals are assigned False
        for lit in clause:
            var = abs(lit) - 1
            val = assignment[var]
            if val is None:
                return False
            is_true = (lit > 0)
            if val == is_true:
                return False
        return True

    def unit_propagate():
        # Returns False if contradiction found, True otherwise
        # Modifies assignment in place
        changed = True
        while changed:
            changed = False
            for clause in clauses:
                # Count unassigned and check if unit
                unassigned_lits = []
                for lit in clause:
                    var = abs(lit) - 1
                    if assignment[var] is None:
                        unassigned_lits.append(lit)
                
                if len(unassigned_lits) == 0:
                    return False # Contradiction: clause is unsatisfied
                elif len(unassigned_lits) == 1:
                    # Unit clause
                    lit = unassigned_lits[0]
                    var = abs(lit) - 1
                    # If already assigned, check consistency (should be handled by previous steps, but safe)
                    if assignment[var] is not None:
                        if assignment[var] != (lit > 0):
                            return False
                    else:
                        assignment[var] = (lit > 0)
                        changed = True
        return True

    def pure_literal_eliminate():
        # Returns True if any pure literals were eliminated
        # Modifies assignment in place
        # Count occurrences of positive and negative literals
        pos_count = [0] * n
        neg_count = [0] * n
        
        for clause in clauses:
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] is None:
                    if lit > 0:
                        pos_count[var] += 1
                    else:
                        neg_count[var] += 1
        
        eliminated = False
        for var in range(n):
            if assignment[var] is not None:
                continue
            if pos_count[var] > 0 and neg_count[var] == 0:
                assignment[var] = True
                eliminated = True
            elif neg_count[var] > 0 and pos_count[var] == 0:
                assignment[var] = False
                eliminated = True
        return eliminated

    def dpll():
        # Check for timeout
        if time.time() > deadline:
            return False
        
        # 1. Check if all clauses satisfied
        for clause in clauses:
            if not check_satisfied(clause):
                if check_empty(clause):
                    return False
                break
        else:
            # All clauses satisfied
            return True

        # 2. Unit Propagation
        if not unit_propagate():
            return False
        
        # Re-check after unit propagation
        for clause in clauses:
            if not check_satisfied(clause):
                if check_empty(clause):
                    return False
                break
        else:
            return True

        # 3. Pure Literal Elimination
        pure_literal_eliminate()
        
        # Re-check after pure literal elimination
        for clause in clauses:
            if not check_satisfied(clause):
                if check_empty(clause):
                    return False
                break
        else:
            return True

        # 4. Branching
        # Heuristic: Pick a variable from an unsatisfied clause with the fewest unassigned literals
        best_var = -1
        min_unassigned = float('inf')
        
        for clause in clauses:
            if check_satisfied(clause):
                continue
            unassigned_lits = []
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] is None:
                    unassigned_lits.append(lit)
            
            if len(unassigned_lits) < min_unassigned:
                min_unassigned = len(unassigned_lits)
                if min_unassigned == 1:
                    # Should have been caught by unit propagation, but just in case
                    best_var = abs(unassigned_lits[0]) - 1
                    break
                else:
                    best_var = abs(unassigned_lits[0]) - 1
        
        if best_var == -1:
            # No unassigned variables in unsatisfied clauses? 
            # This implies all unsatisfied clauses are empty or something went wrong.
            # But we checked for empty clauses.
            # If no unassigned vars, all are assigned. If not satisfied, it's UNSAT.
            return False

        # Try True
        assignment[best_var] = True
        if dpll():
            return True
        
        # Try False
        assignment[best_var] = False
        if dpll():
            return True
        
        # Backtrack
        assignment[best_var] = None
        return False

    # Run DPLL
    if dpll():
        # Ensure all variables are assigned (DPLL should have done this, but just in case)
        for i in range(n):
            if assignment[i] is None:
                assignment[i] = False # Arbitrary default, shouldn't happen if SAT
        
        # Convert to list of booleans
        return [bool(a) for a in assignment]
    else:
        # If DPLL fails (shouldn't happen for satisfiable instances), 
        # or if we hit the deadline, return the current assignment.
        # The problem states instances are satisfiable, so DPLL should succeed.
        # If we hit deadline, we return whatever we have.
        for i in range(n):
            if assignment[i] is None:
                assignment[i] = False
        return [bool(a) for a in assignment]

# EVOLVE-BLOCK-END
