"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    if not clauses:
        return [False] * n
        
    # Precompute variable to clauses mapping for fast delta calculation
    var_to_clauses = [[] for _ in range(n)]
    for idx, clause in enumerate(clauses):
        for lit in clause:
            var_to_clauses[abs(lit) - 1].append(idx)
            
    def count_unsat(assignment):
        count = 0
        for clause in clauses:
            sat = False
            for lit in clause:
                v = abs(lit) - 1
                if assignment[v] == (lit > 0):
                    sat = True
                    break
            if not sat:
                count += 1
        return count
        
    def delta_energy(assignment, var):
        """
        Calculate change in energy if we flip 'var'.
        """
        current_val = assignment[var]
        new_val = not current_val
        
        delta = 0
        for c_idx in var_to_clauses[var]:
            clause = clauses[c_idx]
            
            # Check if clause is currently satisfied
            currently_sat = False
            for lit in clause:
                v = abs(lit) - 1
                if v == var:
                    continue
                if assignment[v] == (lit > 0):
                    currently_sat = True
                    break
            
            # Check if clause would be satisfied after flip
            would_be_sat = False
            for lit in clause:
                v = abs(lit) - 1
                if v == var:
                    if new_val == (lit > 0):
                        would_be_sat = True
                        break
                else:
                    if assignment[v] == (lit > 0):
                        would_be_sat = True
                        break
            
            if currently_sat and not would_be_sat:
                delta += 1
            elif not currently_sat and would_be_sat:
                delta -= 1
                
        return delta

    # Simulated Annealing Parameters
    start_time = time.time()
    time_limit = 290
    
    # Initial temperature
    T = 10.0
    cooling_rate = 0.999
    
    # Multiple restarts
    for restart in range(20):
        if time.time() - start_time > time_limit:
            break
            
        assignment = [random.random() < 0.5 for _ in range(n)]
        current_energy = count_unsat(assignment)
        
        if current_energy == 0:
            return assignment
            
        T = 10.0
        
        while time.time() - start_time < time_limit:
            # Pick random variable
            var = random.randint(0, n - 1)
            
            delta = delta_energy(assignment, var)
            
            if delta < 0:
                # Always accept improvements
                assignment[var] = not assignment[var]
                current_energy += delta
            else:
                # Accept with probability exp(-delta / T)
                if T > 0:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        assignment[var] = not assignment[var]
                        current_energy += delta
                
            # Cool down
            T *= cooling_rate
            
            if current_energy == 0:
                return assignment
                
            # If T is too low, restart with higher T
            if T < 0.01:
                break
                
    # Fallback: Random search
    return [False] * n

# EVOLVE-BLOCK-END
