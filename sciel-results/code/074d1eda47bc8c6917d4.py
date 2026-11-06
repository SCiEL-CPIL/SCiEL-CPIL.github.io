import time
import random
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 280.0
    
    # Convert to 0-based literals for internal processing? 
    # DPLL works well with sets of literals. 
    # Let's keep 1-based literals as they are, just need to be careful with sign.
    # Actually, standard DPLL uses integers where sign indicates polarity.
    
    # Preprocessing: Remove tautologies? 3-SAT clauses don't have tautologies usually.
    # Remove duplicates?
    
    # We will implement DPLL with Unit Propagation and Pure Literal Elimination.
    
    # State:
    # assignment: dict var -> bool (0 or 1)
    # clauses: list of sets of literals
    
    # To make it efficient, we might use arrays, but for n=50, sets are fine.
    
    def dpll(clauses, assignment, deadline):
        """
        clauses: list of sets of literals (integers, 1-based abs value, sign indicates polarity)
        assignment: dict {var: bool}
        Returns: (success, assignment)
        """
        # Check for timeout
        if time.time() > deadline:
            return False, assignment
            
        # 1. Check for satisfied/unsatisfied clauses
        # If any clause is empty, it's unsat
        # If any clause is satisfied, remove it
        
        new_clauses = []
        for clause in clauses:
            # Check if clause is satisfied
            sat = False
            for lit in clause:
                v = abs(lit)
                if v in assignment:
                    if (lit > 0 and assignment[v]) or (lit < 0 and not assignment[v]):
                        sat = True
                        break
            if sat:
                continue # Clause satisfied, remove
            
            # Check if clause is empty (unsat)
            if len(clause) == 0:
                return False, assignment
                
            new_clauses.append(clause)
            
        clauses = new_clauses
        
        # 2. Unit Propagation
        # Find unit clauses (clauses with 1 literal)
        changed = True
        while changed:
            changed = False
            for i, clause in enumerate(clauses):
                if len(clause) == 1:
                    lit = list(clause)[0]
                    v = abs(lit)
                    # Assign v
                    val = (lit > 0)
                    if v in assignment:
                        # Check consistency
                        if assignment[v] != val:
                            return False, assignment
                    else:
                        assignment[v] = val
                        changed = True
                        # Remove this clause and any clause containing the negation of lit
                        # Actually, the loop above will handle removal in next pass, 
                        # but we can be more aggressive.
                        # Let's just break and restart the loop to handle all units cleanly.
                        # Or better, update clauses in place.
                        # For simplicity and correctness with n=50, restarting is fine.
                        # But let's try to be efficient.
                        # We'll just mark changed and break to re-evaluate all clauses.
                        break
            if changed:
                # Re-filter clauses based on new assignment
                new_clauses = []
                for clause in clauses:
                    # Remove assigned literals
                    new_clause = set()
                    for lit in clause:
                        v = abs(lit)
                        if v not in assignment:
                            new_clause.add(lit)
                        else:
                            # Check if satisfied
                            if (lit > 0 and assignment[v]) or (lit < 0 and not assignment[v]):
                                new_clause = None # Satisfied
                                break
                    if new_clause is not None:
                        if len(new_clause) == 0:
                            return False, assignment
                        new_clauses.append(new_clause)
                clauses = new_clauses
                if time.time() > deadline:
                    return False, assignment
        
        # 3. Pure Literal Elimination
        # Find variables that appear with only one sign
        # Build a map of var -> set of signs
        var_signs = {}
        for clause in clauses:
            for lit in clause:
                v = abs(lit)
                if v not in var_signs:
                    var_signs[v] = set()
                var_signs[v].add(1 if lit > 0 else -1)
                
        pure_vars = []
        for v, signs in var_signs.items():
            if v not in assignment:
                if len(signs) == 1:
                    pure_vars.append(v)
                    
        if pure_vars:
            # Assign all pure literals
            for v in pure_vars:
                # Determine value
                # Find a literal in any clause containing v
                for clause in clauses:
                    for lit in clause:
                        if abs(lit) == v:
                            assignment[v] = (lit > 0)
                            break
                    if v in assignment:
                        break
            # Recurse with updated assignment
            # We don't need to modify clauses explicitly for pure literals 
            # because the assignment will satisfy them, and the next call will remove them.
            return dpll(clauses, assignment, deadline)
            
        # 4. If no clauses left, all satisfied
        if len(clauses) == 0:
            # Ensure all variables are assigned? 
            # The problem asks for a list of booleans of length n.
            # Unassigned variables can be arbitrary.
            return True, assignment
            
        # 5. Branching
        # Pick a variable
        # Heuristic: Pick variable from the clause with the fewest literals? 
        # Or most frequent variable?
        # Let's pick the first variable in the first clause.
        # Better: Pick variable that appears most often.
        
        var_count = {}
        for clause in clauses:
            for lit in clause:
                v = abs(lit)
                var_count[v] = var_count.get(v, 0) + 1
                
        if not var_count:
            return True, assignment
            
        # Pick variable with max count
        var_to_branch = max(var_count, key=var_count.get)
        
        # Try True
        assignment[var_to_branch] = True
        success, final_assign = dpll(clauses, assignment, deadline)
        if success:
            return True, final_assign
            
        # Try False
        assignment[var_to_branch] = False
        success, final_assign = dpll(clauses, assignment, deadline)
        if success:
            return True, final_assign
            
        # Backtrack
        del assignment[var_to_branch]
        return False, assignment

    # Initial clauses as sets
    initial_clauses = [set(clause) for clause in clauses]
    
    # Run DPLL
    success, assignment_dict = dpll(initial_clauses, {}, deadline)
    
    # Construct final assignment list
    final_assignment = [False] * n
    for v in range(1, n + 1):
        if v in assignment_dict:
            final_assignment[v - 1] = assignment_dict[v]
        else:
            # Unassigned, set to False (or random)
            final_assignment[v - 1] = False
            
    # Verify solution
    # If DPLL failed or timed out, we might need a fallback.
    # But DPLL is complete for SAT. If it returns False, it's UNSAT.
    # The problem states instances are satisfiable.
    # If it timed out, success is False, but we might have a partial assignment.
    # Let's check if the current assignment satisfies all clauses.
    
    is_sat = True
    for clause in clauses:
        sat = False
        for lit in clause:
            v = abs(lit) - 1
            if (lit > 0 and final_assignment[v]) or (lit < 0 and not final_assignment[v]):
                sat = True
                break
        if not sat:
            is_sat = False
            break
            
    if is_sat:
        return final_assignment
        
    # Fallback: WalkSAT if DPLL failed/timed out
    # (This is unlikely for planted 3-SAT, but good for safety)
    
    # Re-implementing a quick WalkSAT fallback
    deadline_walk = deadline
    if time.time() >= deadline_walk:
         return final_assignment # Return best effort
        
    rng = random.Random(42)
    assignment = [rng.random() < 0.5 for _ in range(n)]
    
    # Simple WalkSAT
    m = len(clauses)
    for restart in range(1000):
        if time.time() >= deadline_walk:
            break
        assignment = [rng.random() < 0.5 for _ in range(n)]
        for step in range(10000):
            if time.time() >= deadline_walk:
                break
            # Check if sat
            is_sat = True
            unsat_clause_idx = -1
            for ci, clause in enumerate(clauses):
                sat = False
                for lit in clause:
                    v = abs(lit) - 1
                    if (lit > 0 and assignment[v]) or (lit < 0 and not assignment[v]):
                        sat = True
                        break
                if not sat:
                    is_sat = False
                    unsat_clause_idx = ci
                    break
            if is_sat:
                return assignment
                
            if unsat_clause_idx == -1:
                break
                
            # Pick random var from unsat clause
            clause = clauses[unsat_clause_idx]
            v = abs(rng.choice(list(clause))) - 1
            assignment[v] = not assignment[v]
            
    return final_assignment

