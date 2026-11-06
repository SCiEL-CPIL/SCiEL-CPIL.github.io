# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 290  # Safety margin for 300s limit
    
    m = len(clauses)
    
    # Preprocessing: Convert to 0-based and separate vars/signs
    # vars_matrix: (m, 3) int32
    # signs_matrix: (m, 3) int32 (0 for False/neg, 1 for True/pos)
    vars_matrix = np.zeros((m, 3), dtype=np.int32)
    signs_matrix = np.zeros((m, 3), dtype=np.int32)
    
    for i, c in enumerate(clauses):
        for j, l in enumerate(c):
            vars_matrix[i, j] = abs(l) - 1
            signs_matrix[i, j] = 1 if l > 0 else 0

    # Build variable-to-clause index for efficient flip scoring
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            var_to_clauses[vars_matrix[i, j]].append(i)

    rng = random.Random(42)
    
    def count_unsat(assignment):
        """
        Returns the number of unsatisfied clauses.
        assignment: numpy array of int32 (0/1)
        """
        a = assignment
        matches = (a[vars_matrix] == signs_matrix)
        sat_counts = matches.sum(axis=1)
        return int(np.sum(sat_counts == 0))

    def get_unsat_clauses(assignment):
        """
        Returns list of indices of unsatisfied clauses.
        """
        a = assignment
        matches = (a[vars_matrix] == signs_matrix)
        sat_counts = matches.sum(axis=1)
        unsat_indices = np.where(sat_counts == 0)[0]
        return unsat_indices.tolist()

    def score_flip(assignment, var_to_flip, current_unsat_indices):
        """
        Calculate the change in unsatisfied clause count if we flip var_to_flip.
        Returns the delta (new_unsat - old_unsat).
        """
        delta = 0
        v = var_to_flip
        
        # Iterate over clauses containing this variable
        for ci in var_to_clauses[v]:
            # Check current state of this clause
            # We need to know if it's satisfied and if this var is the unique satisfying literal
            # Or if it's unsatisfied and this var is the one that would satisfy it.
            
            c_vars = vars_matrix[ci]
            c_signs = signs_matrix[ci]
            
            # Count how many literals in this clause are currently satisfied
            # And check if 'v' is among them
            sat_count = 0
            v_is_sat = False
            
            for k in range(3):
                vv = c_vars[k]
                ss = c_signs[k]
                if assignment[vv] == ss:
                    sat_count += 1
                    if vv == v:
                        v_is_sat = True
            
            if sat_count > 0:
                # Clause is currently satisfied
                if v_is_sat and sat_count == 1:
                    # This var is the ONLY satisfying literal. Flipping it breaks the clause.
                    delta += 1
            else:
                # Clause is currently unsatisfied
                # Flipping v makes the literal (v, sign_in_clause) true.
                # So the clause becomes satisfied.
                delta -= 1
                
        return delta

    best_assignment = None
    best_unsat_count = m + 1
    
    max_restarts = 1000
    base_steps = 2000
    step_growth = 1.05
    
    restart = 0
    while restart < max_restarts and time.time() < deadline:
        # Random initial assignment
        assignment = np.random.randint(0, 2, n, dtype=np.int32)
        
        unsat_count = count_unsat(assignment)
        
        if unsat_count == 0:
            return assignment.tolist()
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment.copy()
        
        steps = int(base_steps * (step_growth ** restart))
        steps = min(steps, 50000)
        
        # Adaptive WalkSAT / Simulated Annealing hybrid
        # We use a temperature-like mechanism to accept worse moves to escape local minima
        # T starts high, cools down.
        T = 5.0
        cooling = 0.999
        
        for step in range(steps):
            if time.time() >= deadline:
                break
            
            unsat_indices = get_unsat_clauses(assignment)
            
            if not unsat_indices:
                return assignment.tolist()
            
            current_unsat_count = len(unsat_indices)
            if current_unsat_count < best_unsat_count:
                best_unsat_count = current_unsat_count
                best_assignment = assignment.copy()
            
            # Pick a random unsatisfied clause
            ci = rng.choice(unsat_indices)
            
            # Get the literals in this clause
            c_vars = vars_matrix[ci]
            c_signs = signs_matrix[ci]
            
            # Identify false literals in this clause
            # A literal is false if assignment[var] != sign
            false_vars = []
            for k in range(3):
                v = c_vars[k]
                s = c_signs[k]
                if assignment[v] != s:
                    false_vars.append(v)
            
            if not false_vars:
                continue
            
            # Choose which variable to flip
            # Strategy: 
            # 1. Calculate score (delta unsat) for each false var in the clause.
            # 2. With probability p_random, pick randomly.
            # 3. Else, pick the one with minimum delta (best improvement).
            #    If delta > 0 (worsening), we might still pick it based on temperature (Simulated Annealing logic).
            
            p_random = 0.3
            best_var = -1
            min_delta = float('inf')
            deltas = []
            
            for v in false_vars:
                d = score_flip(assignment, v, unsat_indices)
                deltas.append(d)
                if d < min_delta:
                    min_delta = d
                    best_var = v
            
            if rng.random() < p_random:
                var_to_flip = rng.choice(false_vars)
            else:
                # If the best move worsens the solution, use Simulated Annealing acceptance
                if min_delta > 0:
                    # Accept with probability exp(-delta / T)
                    # delta is the increase in unsat
                    prob = np.exp(-min_delta / max(T, 0.1))
                    if rng.random() < prob:
                        var_to_flip = best_var
                    else:
                        # Pick a random var from the clause (or best one?)
                        # Let's stick to random to explore more
                        var_to_flip = rng.choice(false_vars)
                else:
                    var_to_flip = best_var
            
            # Perform the flip
            assignment[var_to_flip] = 1 - assignment[var_to_flip]
            
            # Cool down
            T *= cooling
            
        restart += 1

    if best_assignment is not None:
        return best_assignment.tolist()
    return assignment.tolist()
# EVOLVE-BLOCK-END
