"""DPLL solver with Unit Propagation and Pure Literal Elimination."""
# EVOLVE-BLOCK-START
import sys
import time

# Increase recursion limit for safety, though n=150 is small
sys.setrecursionlimit(10000)

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    start_time = time.monotonic()
    time_limit = 290.0

    # DPLL implementation
    # State:
    # assignment: list of -1 (unassigned), 0 (False), 1 (True)
    # clauses: list of lists of literals (integers). 
    #          We will filter out assigned literals dynamically during propagation.
    
    def dpll(assignment, clauses):
        # 1. Check for empty clause (contradiction)
        if any(len(c) == 0 for c in clauses):
            return False

        # 2. Check if all clauses are satisfied
        if all(len(c) == 0 or True for c in clauses): # This check is weak, we rely on propagation
             pass # We handle satisfied clauses by removing them? 
             # Actually, standard DPLL: if a clause is satisfied, we can remove it from consideration.
             # Let's refine: 
             # A clause is satisfied if any literal is assigned True.
             # A clause is falsified if all literals are assigned False.
             # We will maintain `clauses` as the set of *unsatisfied* clauses with *unassigned* literals.
             
        # Let's restructure the state to be cleaner.
        # assignment: dict or list. List is faster.
        # active_clauses: clauses that are not yet satisfied.
        # We remove literals that are assigned False from active clauses.
        # We remove clauses that have a literal assigned True.
        
        # This function `dpll` takes the current state.
        # To make it efficient, we should do Unit Propagation in a loop.
        
        # Step 1: Unit Propagation
        changed = True
        while changed:
            changed = False
            new_clauses = []
            unit_vars = []
            
            for clause in clauses:
                if not clause:
                    return False # Contradiction
                # Filter out literals that are assigned False
                filtered_clause = []
                unit_lit = None
                satisfied = False
                for lit in clause:
                    v = abs(lit) - 1
                    val = assignment[v]
                    if val == 1:
                        if (lit > 0 and val == 1) or (lit < 0 and val == 0):
                            # Wait, val is 0 or 1. lit > 0 means var must be True (1).
                            # So if val == 1 and lit > 0, satisfied.
                            # If val == 0 and lit < 0, satisfied.
                            pass
                    # Let's just check satisfaction directly
                    if val == 1:
                        if lit > 0:
                            satisfied = True
                            break
                    elif val == 0:
                        if lit < 0:
                            satisfied = True
                            break
                    else: # unassigned
                        filtered_clause.append(lit)
                
                if satisfied:
                    continue # Clause is satisfied, remove from active set
                    
                if len(filtered_clause) == 1:
                    unit_lit = filtered_clause[0]
                    unit_vars.append(unit_lit)
                    # We don't add this to new_clauses yet, we force the assignment
                else:
                    new_clauses.append(filtered_clause)
            
            # If any unit clauses found, assign them
            if unit_vars:
                changed = True
                for lit in unit_vars:
                    v = abs(lit) - 1
                    val_to_assign = 1 if lit > 0 else 0
                    if assignment[v] == -1:
                        assignment[v] = val_to_assign
                    else:
                        # Conflict
                        if assignment[v] != val_to_assign:
                            return False
            
            clauses = new_clauses
            
        # After propagation, check for empty clause again
        if any(len(c) == 0 for c in clauses):
            return False
            
        # If no clauses left, all satisfied
        if not clauses:
            return True
            
        # Step 2: Pure Literal Elimination
        # Count occurrences of each literal
        # This is O(m * clause_len). For n=150, this is fine.
        # If a variable appears only as positive, set to True.
        # If only as negative, set to False.
        
        pos_count = [0] * n
        neg_count = [0] * n
        for clause in clauses:
            for lit in clause:
                v = abs(lit) - 1
                if lit > 0:
                    pos_count[v] += 1
                else:
                    neg_count[v] += 1
        for v in range(n):
            if assignment[v] == -1:
                if pos_count[v] > 0 and neg_count[v] == 0:
                    assignment[v] = 1
                    changed = True
                elif neg_count[v] > 0 and pos_count[v] == 0:
                    assignment[v] = 0
                    changed = True
        if changed:
            return dpll(assignment, clauses) # Restart loop to propagate
            
        # Step 3: Branching
        # Pick a variable to branch on.
        # Heuristic: Pick a variable from the shortest clause.
        # If multiple, pick the one with highest frequency?
        # Let's pick the variable in the first shortest clause.
        
        min_len = min(len(c) for c in clauses)
        # Find a clause of that length
        branch_clause = None
        for c in clauses:
            if len(c) == min_len:
                branch_clause = c
                break
                
        if not branch_clause:
             return True # Should not happen if clauses not empty
            
        # Pick a literal from this clause to branch on
        # Random or first? First is deterministic.
        lit = branch_clause[0]
        v = abs(lit) - 1
        
        # Try setting to True
        assignment[v] = 1
        # Create a copy of clauses? Or just pass the same list and rely on propagation to filter?
        # Propagation will filter. But we need to restore state on backtrack.
        # Simplest: copy assignment and clauses.
        # n=150, m is likely a few hundred. Copying is cheap.
        
        import copy
        assign_copy = assignment[:]
        clauses_copy = [c[:] for c in clauses]
        
        if dpll(assign_copy, clauses_copy):
            # Restore assignment? No, we return True up the stack.
            # But the caller might need the final assignment.
            # We need to propagate the successful assignment back up.
            # Since we are modifying `assignment` in place in the recursive call,
            # and if it returns True, the `assignment` in the parent scope (which is the same list object 
            # if we passed it, but we passed a copy `assign_copy`) ...
            # Wait, I passed `assign_copy`. So the parent's `assignment` is NOT updated.
            # I need to update the parent's assignment from the successful child.
            
            # Actually, let's just pass the original assignment and clauses, and make copies inside the recursive call?
            # Or better: pass references, and if successful, copy back.
            # Let's stick to copying for simplicity and correctness.
            
            # If dpll(assign_copy, clauses_copy) returns True, we need to update the current `assignment`
            # to match `assign_copy` so that the final result is available.
            assignment[:] = assign_copy
            return True
            
        # Try setting to False
        assignment[v] = 0
        assign_copy2 = assignment[:]
        clauses_copy2 = [c[:] for c in clauses]
        
        if dpll(assign_copy2, clauses_copy2):
            assignment[:] = assign_copy2
            return True
            
        # Backtrack
        assignment[v] = -1
        return False

    # Initialize
    assignment = [-1] * n
    # Copy clauses to avoid modifying input
    active_clauses = [c[:] for c in clauses]
    
    # Check time limit periodically? 
    # DPLL is recursive. It's hard to interrupt cleanly without exceptions or global flags.
    # Given n=150, it should be fast. If it's slow, we might timeout.
    # We can add a simple global timer check in the recursion.
    
    def timed_dpll(assignment, clauses):
        if time.monotonic() - start_time > time_limit:
            return False, False # Timeout
            
        # 1. Unit Propagation
        changed = True
        while changed:
            changed = False
            new_clauses = []
            unit_vars = []
            
            for clause in clauses:
                if not clause:
                    return False, False # Contradiction
                filtered_clause = []
                satisfied = False
                for lit in clause:
                    v = abs(lit) - 1
                    val = assignment[v]
                    if val == 1:
                        if lit > 0:
                            satisfied = True
                            break
                    elif val == 0:
                        if lit < 0:
                            satisfied = True
                            break
                    else:
                        filtered_clause.append(lit)
                
                if satisfied:
                    continue
                    
                if len(filtered_clause) == 1:
                    unit_vars.append(filtered_clause[0])
                else:
                    new_clauses.append(filtered_clause)
            
            if unit_vars:
                changed = True
                for lit in unit_vars:
                    v = abs(lit) - 1
                    val_to_assign = 1 if lit > 0 else 0
                    if assignment[v] == -1:
                        assignment[v] = val_to_assign
                    else:
                        if assignment[v] != val_to_assign:
                            return False, False
            
            clauses = new_clauses
            
        if any(len(c) == 0 for c in clauses):
            return False, False
            
        if not clauses:
            return True, True # Satisfied
            
        # 2. Pure Literal Elimination
        pos_count = [0] * n
        neg_count = [0] * n
        for clause in clauses:
            for lit in clause:
                v = abs(lit) - 1
                if lit > 0:
                    pos_count[v] += 1
                else:
                    neg_count[v] += 1
        pure_changed = False
        for v in range(n):
            if assignment[v] == -1:
                if pos_count[v] > 0 and neg_count[v] == 0:
                    assignment[v] = 1
                    pure_changed = True
                elif neg_count[v] > 0 and pos_count[v] == 0:
                    assignment[v] = 0
                    pure_changed = True
        if pure_changed:
            return timed_dpll(assignment, clauses)
            
        # 3. Branching
        min_len = min(len(c) for c in clauses)
        branch_clause = None
        for c in clauses:
            if len(c) == min_len:
                branch_clause = c
                break
                
        if not branch_clause:
            return True, True
            
        lit = branch_clause[0]
        v = abs(lit) - 1
        
        # Branch True
        assignment[v] = 1
        assign_copy = assignment[:]
        clauses_copy = [c[:] for c in clauses]
        
        sat, valid = timed_dpll(assign_copy, clauses_copy)
        if sat:
            assignment[:] = assign_copy
            return True, valid
            
        # Branch False
        assignment[v] = 0
        assign_copy2 = assignment[:]
        clauses_copy2 = [c[:] for c in clauses]
        
        sat, valid = timed_dpll(assign_copy2, clauses_copy2)
        if sat:
            assignment[:] = assign_copy2
            return True, valid
            
        assignment[v] = -1
        return False, valid

    sat, valid = timed_dpll(assignment, active_clauses)
    
    if sat:
        # Convert 0/1 to False/True
        return [bool(x) for x in assignment]
    else:
        # If not solved or timeout, return a random assignment? 
        # The task says "Return a satisfying assignment". 
        # If we fail, we return an invalid one, which scores 0.
        # But we should try to return *something* if we timed out? 
        # No, if it's not satisfying, it's 0.
        # We can try to return the best partial assignment? No, it must be a list of booleans.
        # If we don't have a solution, we just return a random guess to avoid exceptions.
        import random
        return [random.choice([True, False]) for _ in range(n)]

# EVOLVE-BLOCK-END
