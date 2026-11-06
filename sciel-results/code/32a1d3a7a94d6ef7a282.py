"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(42) # Fixed seed for reproducibility, or maybe 0
    
    # Preprocessing
    m = len(clauses)
    clause_vars = []
    var_clauses = [[] for _ in range(n)]
    
    for i, c in enumerate(clauses):
        cv = []
        for l in c:
            var_idx = abs(l) - 1
            is_pos = l > 0
            cv.append((var_idx, is_pos))
            var_clauses[var_idx].append((i, is_pos))
        clause_vars.append(cv)
    
    # Time management
    start_time = time.time()
    time_limit = 295.0 # Safety margin for 330s kill
    
    def time_up():
        return time.time() - start_time > time_limit

    def init_assignment():
        return [rng.random() < 0.5 for _ in range(n)]
    
    def calc_sat_counts(assignment):
        """Calculate initial satisfaction counts for all clauses."""
        counts = [0] * m
        for i in range(m):
            cnt = 0
            for var_idx, is_pos in clause_vars[i]:
                if assignment[var_idx] == is_pos:
                    cnt += 1
            counts[i] = cnt
        return counts

    def get_unsat_clauses(counts):
        return [i for i in range(m) if counts[i] == 0]

    def walk_sat(assignment, max_steps):
        """
        Run WalkSAT with damage-minimizing heuristic.
        Returns (assignment, solved, unsat_count)
        """
        counts = calc_sat_counts(assignment)
        unsat = get_unsat_clauses(counts)
        
        if not unsat:
            return assignment, True, 0

        # Noise parameter: probability of choosing a random literal instead of best
        noise = 0.25 
        
        for step in range(max_steps):
            if time_up():
                return assignment, False, len(unsat)
            
            if not unsat:
                return assignment, True, 0
                
            # Pick a random unsatisfied clause
            clause_idx = rng.choice(unsat)
            clause = clause_vars[clause_idx]
            
            # Heuristic: Pick variable with minimum damage
            best_var = -1
            best_flip_val = False
            best_damage = float('inf')
            
            for var_idx, is_pos in clause:
                # We want to flip var_idx to is_pos to satisfy this clause.
                damage = 0
                for c_idx, c_is_pos in var_clauses[var_idx]:
                    if c_idx == clause_idx:
                        continue
                    if c_is_pos != is_pos:
                        if counts[c_idx] > 0:
                            if counts[c_idx] == 1:
                                damage += 1
                
                if damage < best_damage or (damage == best_damage and rng.random() < 0.5):
                    best_damage = damage
                    best_var = var_idx
                    best_flip_val = is_pos
            
            # With some probability, pick a random variable from the clause to escape local minima
            if rng.random() < noise:
                var_idx, is_pos = rng.choice(clause)
                best_var = var_idx
                best_flip_val = is_pos
            
            # Perform flip
            if best_var != -1:
                old_val = assignment[best_var]
                assignment[best_var] = best_flip_val
                
                # Update counts incrementally
                for c_idx, c_is_pos in var_clauses[best_var]:
                    was_sat = (old_val == c_is_pos)
                    is_sat = (best_flip_val == c_is_pos)
                    
                    if was_sat and not is_sat:
                        counts[c_idx] -= 1
                    elif not was_sat and is_sat:
                        counts[c_idx] += 1
            
            # Recalculate unsatisfied clauses list
            unsat = [i for i in range(m) if counts[i] == 0]
            
        return assignment, len(unsat) == 0, len(unsat)

    def simulated_annealing(assignment, max_steps):
        """
        Run Simulated Annealing to escape local minima.
        """
        counts = calc_sat_counts(assignment)
        unsat = get_unsat_clauses(counts)
        current_unsat = len(unsat)
        
        if current_unsat == 0:
            return assignment, True, 0

        # Temperature schedule: linear cooling
        start_temp = 1.0
        end_temp = 0.01
        cool_factor = (end_temp / start_temp) ** (1.0 / max_steps)
        temp = start_temp
        
        best_assignment = assignment[:]
        best_unsat = current_unsat
        
        for step in range(max_steps):
            if time_up():
                return best_assignment, False, best_unsat
            
            # Pick a random variable to flip
            var_idx = rng.randint(0, n - 1)
            old_val = assignment[var_idx]
            new_val = not old_val
            
            # Calculate delta in unsatisfied clauses
            delta = 0
            for c_idx, c_is_pos in var_clauses[var_idx]:
                was_sat = (old_val == c_is_pos)
                is_sat = (new_val == c_is_pos)
                
                if was_sat and not is_sat:
                    # Clause becomes unsatisfied if it was satisfied only by this literal
                    if counts[c_idx] == 1:
                        delta += 1
                elif not was_sat and is_sat:
                    # Clause becomes satisfied
                    if counts[c_idx] == 0:
                        delta -= 1
            
            # Accept or reject
            if delta < 0 or rng.random() < math.exp(-delta / temp):
                # Accept move
                assignment[var_idx] = new_val
                for c_idx, c_is_pos in var_clauses[var_idx]:
                    was_sat = (old_val == c_is_pos)
                    is_sat = (new_val == c_is_pos)
                    if was_sat and not is_sat:
                        counts[c_idx] -= 1
                    elif not was_sat and is_sat:
                        counts[c_idx] += 1
                
                current_unsat += delta
                if current_unsat < best_unsat:
                    best_unsat = current_unsat
                    best_assignment = assignment[:]
            else:
                # Reject move (no change to state)
                pass
            
            temp *= cool_factor
        
        return best_assignment, best_unsat == 0, best_unsat

    # Main loop with restarts
    best_assignment = init_assignment()
    best_unsat = float('inf')
    
    # Adaptive step limit
    steps_per_restart = 2000
    sa_steps = 1000
    
    while not time_up():
        # Generate new assignment
        if best_unsat < float('inf') and rng.random() < 0.3:
            assignment = best_assignment[:]
        else:
            assignment = init_assignment()
            
        # Phase 1: WalkSAT
        assignment, solved, cur_unsat = walk_sat(assignment, steps_per_restart)
        
        if solved:
            return assignment
        
        # Phase 2: Simulated Annealing polish
        if not solved and not time_up():
            assignment, sa_solved, sa_unsat = simulated_annealing(assignment, sa_steps)
            
            if sa_solved:
                return assignment
            
            if sa_unsat < cur_unsat:
                cur_unsat = sa_unsat
        
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]
            
    return best_assignment

# EVOLVE-BLOCK-END
