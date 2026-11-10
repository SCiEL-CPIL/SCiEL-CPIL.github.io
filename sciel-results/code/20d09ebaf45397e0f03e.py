"""Improved 3-SAT solver using Simulated Annealing with incremental updates."""
import random
import time
import math
import sys

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    # Deadline: 290 seconds to be safe
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Pre-compute variable to clauses mapping
    # var_to_clauses[i] is a list of indices of clauses containing variable i+1
    var_to_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(clauses):
        for lit in clause:
            var_idx = abs(lit) - 1
            var_to_clauses[var_idx].append(i)
    
    # Helper to check if a clause is satisfied given the assignment
    # We will maintain clause_unsat array to avoid full scans
    
    def initialize():
        """Initialize random assignment and clause states."""
        assignment = [random.random() < 0.5 for _ in range(n)]
        clause_unsat = [False] * m
        unsat_count = 0
        for i, clause in enumerate(clauses):
            sat = False
            for lit in clause:
                var_idx = abs(lit) - 1
                val = assignment[var_idx]
                if (lit > 0 and val) or (lit < 0 and not val):
                    sat = True
                    break
            if not sat:
                clause_unsat[i] = True
                unsat_count += 1
        return assignment, clause_unsat, unsat_count

    def flip_var(var_idx, assignment, clause_unsat, unsat_count):
        """
        Flip variable var_idx and update clause states incrementally.
        Returns new unsat_count.
        """
        # Determine new value
        new_val = not assignment[var_idx]
        
        # Update clauses containing this variable
        # We need to know for each clause if this variable was the 'critical' one
        # Critical means:
        # 1. Clause was unsat, and this var was the only false one (flipping makes it sat)
        # 2. Clause was sat, and this var was the only true one (flipping makes it unsat)
        
        # To do this efficiently, we just re-evaluate the specific clauses.
        # Since degree is low (~12), this is fast.
        
        delta_unsat = 0
        
        for c_idx in var_to_clauses[var_idx]:
            clause = clauses[c_idx]
            was_unsat = clause_unsat[c_idx]
            
            # Check if flipping var_idx changes the satisfaction of this clause
            # We check the other two literals in the clause
            
            # Find the other literals
            other_lits = []
            for lit in clause:
                if abs(lit) - 1 != var_idx:
                    other_lits.append(lit)
            
            # Evaluate the clause with the new value of var_idx
            # Lit for var_idx in this clause:
            lit_for_var = 0
            for lit in clause:
                if abs(lit) - 1 == var_idx:
                    lit_for_var = lit
                    break
            
            # Check if the clause is satisfied with new_val
            is_sat_now = False
            # Check if the flipped variable satisfies the clause
            if (lit_for_var > 0 and new_val) or (lit_for_var < 0 and not new_val):
                is_sat_now = True
            else:
                # Check if any other literal satisfies it
                for ol in other_lits:
                    o_var_idx = abs(ol) - 1
                    o_val = assignment[o_var_idx]
                    if (ol > 0 and o_val) or (ol < 0 and not o_val):
                        is_sat_now = True
                        break
            
            is_sat_after = is_sat_now
            
            if was_unsat and is_sat_after:
                delta_unsat -= 1
                clause_unsat[c_idx] = False
            elif not was_unsat and not is_sat_after:
                delta_unsat += 1
                clause_unsat[c_idx] = True
            # else: state unchanged
            
        assignment[var_idx] = new_val
        return unsat_count + delta_unsat

    best_assignment = None
    best_unsat = float('inf')
    
    # Simulated Annealing Parameters
    # We will run multiple restarts if time permits
    while time.time() < deadline:
        assignment, clause_unsat, unsat_count = initialize()
        
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]
            if unsat_count == 0:
                return assignment
        
        # SA Loop
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.999 # Tune this
        max_steps = 50000 # Max steps per restart before forcing new random start
        
        steps_in_restart = 0
        last_improvement_step = 0
        current_unsat = unsat_count
        
        for step in range(max_steps):
            if time.time() > deadline - 10.0: # Safety margin
                break
            
            if current_unsat == 0:
                return assignment
            
            # Select a variable to flip
            # Strategy: Pick a random unsatisfied clause, then a random var in it
            # If all sat, we are done (handled above)
            
            # Find an unsatisfied clause
            # To avoid O(M) scan, we can keep a list of unsat clause indices
            # But maintaining that list is complex.
            # Alternative: Just pick a random variable?
            # Or pick a random clause, if it's unsat, pick a var from it.
            # Since M is ~600, picking a random clause is O(1).
            
            c_idx = random.randrange(m)
            if not clause_unsat[c_idx]:
                # Clause is satisfied, pick a random variable from it?
                # Or just pick a random variable overall?
                # Let's pick a random variable from this clause to keep local focus
                lit = random.choice(clauses[c_idx])
                var_idx = abs(lit) - 1
            else:
                # Clause is unsatisfied, pick a variable from it
                lit = random.choice(clauses[c_idx])
                var_idx = abs(lit) - 1
            
            # Calculate delta
            # We need to know if flipping this var changes the unsat count
            # We can just do the flip and see, or calculate delta first.
            # Calculating delta first allows us to decide acceptance without modifying state if rejected.
            
            # Let's calculate delta
            new_val = not assignment[var_idx]
            delta = 0
            for c_idx_2 in var_to_clauses[var_idx]:
                clause = clauses[c_idx_2]
                was_unsat = clause_unsat[c_idx_2]
                
                # Determine if this var is the critical one
                # Find lit for var_idx in this clause
                lit_for_var = 0
                other_lits = []
                for lit in clause:
                    if abs(lit) - 1 == var_idx:
                        lit_for_var = lit
                    else:
                        other_lits.append(lit)
                
                # Check if clause is satisfied by new_val
                sat_by_new = (lit_for_var > 0 and new_val) or (lit_for_var < 0 and not new_val)
                
                if sat_by_new:
                    is_sat_after = True
                else:
                    # Check if other literals satisfy it
                    is_sat_after = False
                    for ol in other_lits:
                        o_var_idx = abs(ol) - 1
                        o_val = assignment[o_var_idx]
                        if (ol > 0 and o_val) or (ol < 0 and not o_val):
                            is_sat_after = True
                            break
                
                if was_unsat and is_sat_after:
                    delta -= 1
                elif not was_unsat and not is_sat_after:
                    delta += 1
            
            # Acceptance
            if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                # Accept
                unsat_count = flip_var(var_idx, assignment, clause_unsat, unsat_count)
                if unsat_count < current_unsat:
                    current_unsat = unsat_count
                    last_improvement_step = step
                    if current_unsat < best_unsat:
                        best_unsat = current_unsat
                        best_assignment = assignment[:]
                        if current_unsat == 0:
                            return assignment
                else:
                    current_unsat = unsat_count
            # else reject, do nothing
            
            steps_in_restart += 1
            T *= cooling_rate
            if T < T_min:
                T = T_min
            
            # Restart condition: no improvement for a while
            if step - last_improvement_step > 1000:
                break
        
        # If we broke out due to stagnation, we loop back to initialize()
        # If we broke out due to time, the while loop will exit
        
    return best_assignment if best_assignment is not None else [False] * n
