"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math


def solve(instance):
    """
    Solve a planted 3-SAT instance using Simulated Annealing.
    This approach uses a temperature schedule to escape local minima,
    distinct from the previous WalkSAT/DPLL hybrid.
    """
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)
    
    if num_clauses == 0:
        return [False] * n
    
    # Preprocess: build variable-to-clause index for efficiency
    var_to_clauses = [[] for _ in range(n)]
    for idx, clause in enumerate(clauses):
        for lit in clause:
            var_to_clauses[abs(lit) - 1].append(idx)
    
    def get_unsat_count(assignment):
        """Calculate initial unsatisfied clause count."""
        count = 0
        for clause in clauses:
            sat = False
            for lit in clause:
                var = abs(lit) - 1
                if assignment[var] == (lit > 0):
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    def is_clause_satisfied(clause, assignment):
        for lit in clause:
            var = abs(lit) - 1
            if assignment[var] == (lit > 0):
                return True
        return False

    def simulated_annealing(start_time, time_limit):
        """
        Run Simulated Annealing.
        Returns a satisfying assignment or None if timeout.
        """
        rng = random.Random()
        
        # Initial random assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        current_energy = get_unsat_count(assignment)
        
        if current_energy == 0:
            return assignment
        
        # Annealing parameters
        # T_start should be high enough to accept most moves initially
        T_start = 10.0 
        T_min = 0.001
        # Cooling rate: T_new = T_old * cooling_rate
        # We want to reach T_min in about max_steps
        max_steps_per_run = 50000
        cooling_rate = (T_min / T_start) ** (1.0 / max_steps_per_run)
        
        best_assignment = assignment[:]
        best_energy = current_energy
        current_T = T_start
        
        no_improve_steps = 0
        max_no_improve = 20000
        
        while time.time() - start_time < time_limit:
            if current_T < T_min:
                # Restart with new random assignment and higher temperature
                assignment = [rng.random() < 0.5 for _ in range(n)]
                current_energy = get_unsat_count(assignment)
                if current_energy == 0:
                    return assignment
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_assignment = assignment[:]
                current_T = T_start
                no_improve_steps = 0
                continue
            
            # Choose a random variable to flip
            var = rng.randint(0, n - 1)
            
            # Calculate delta energy
            # Delta = new_energy - old_energy
            # We flip var. This affects all clauses containing var.
            delta = 0
            for c_idx in var_to_clauses[var]:
                clause = clauses[c_idx]
                # Check if clause is currently satisfied
                currently_sat = is_clause_satisfied(clause, assignment)
                
                # Check if clause would be satisfied after flip
                # We simulate the flip by checking if any other literal in the clause
                # satisfies it, or if the flipped literal would satisfy it.
                # Actually, simpler:
                # If currently_sat:
                #   If the flipped literal was the ONLY one satisfying the clause, 
                #   then after flip it becomes unsat. Delta += 1
                #   Else, it remains sat. Delta += 0
                # If not currently_sat:
                #   If the flipped literal becomes True (satisfying) after flip, 
                #   then it becomes sat. Delta -= 1
                #   Else, it remains unsat. Delta += 0
                
                # Optimization: 
                # Let's just evaluate the state before and after for the affected clauses.
                # Since n=150, clauses are small, this is fast enough.
                
                # Current state of clause
                sat_count_before = 0
                for lit in clause:
                    v = abs(lit) - 1
                    val = assignment[v]
                    if val == (lit > 0):
                        sat_count_before += 1
                        
                # State after flip
                # We only need to check if the status changes.
                # The flip only affects the literal corresponding to `var` in this clause.
                # Let's find that literal.
                # Actually, we can just check:
                # 1. Is the clause satisfied by any literal OTHER than the one for `var`?
                # 2. Is the literal for `var` currently satisfied?
                
                lit_for_var = None
                for lit in clause:
                    if abs(lit) - 1 == var:
                        lit_for_var = lit
                        break
                
                if lit_for_var is None:
                    continue # Should not happen if var_to_clauses is correct
                
                # Check if any OTHER literal satisfies the clause
                other_sat = False
                for lit in clause:
                    if abs(lit) - 1 == var:
                        continue
                    v = abs(lit) - 1
                    if assignment[v] == (lit > 0):
                        other_sat = True
                        break
                
                # Current satisfaction of the clause
                curr_sat = other_sat or (assignment[var] == (lit_for_var > 0))
                
                # New satisfaction of the clause (var is flipped)
                new_val = not assignment[var]
                new_sat = other_sat or (new_val == (lit_for_var > 0))
                
                if curr_sat and not new_sat:
                    delta += 1
                elif not curr_sat and new_sat:
                    delta -= 1
            
            # Acceptance probability
            if delta <= 0:
                # Always accept improving or neutral moves
                assignment[var] = not assignment[var]
                current_energy += delta
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_assignment = assignment[:]
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
            else:
                # Accept with probability exp(-delta / T)
                p = math.exp(-delta / current_T)
                if rng.random() < p:
                    assignment[var] = not assignment[var]
                    current_energy += delta
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_assignment = assignment[:]
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
                else:
                    no_improve_steps += 1
            
            # Cool down
            current_T *= cooling_rate
            
            if no_improve_steps > max_no_improve:
                # Restart
                assignment = [rng.random() < 0.5 for _ in range(n)]
                current_energy = get_unsat_count(assignment)
                if current_energy == 0:
                    return assignment
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_assignment = assignment[:]
                current_T = T_start
                no_improve_steps = 0
        
        if best_energy == 0:
            return best_assignment
        return None

    start_time = time.time()
    time_limit = 290
    
    result = simulated_annealing(start_time, time_limit)
    
    if result is not None:
        return result
        
    # Fallback: Return the best found so far? 
    # The SA function returns None if it didn't find a solution, but it tracks best.
    # Let's modify SA to return the best assignment found even if not satisfied, 
    # or just return a random one? 
    # The task asks for a satisfying assignment. If we fail, we return a dummy.
    # However, the score is "fraction solved". So we need to return a valid list of bools.
    # If we don't find a solution, we return a random assignment (which will be scored 0 for that instance).
    
    return [False] * n

# EVOLVE-BLOCK-END
