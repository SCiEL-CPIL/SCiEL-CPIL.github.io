"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 280  # safety margin before 300s kill

    # Precompute clause literals as (var_idx, polarity) for faster access
    # var_idx is 0-based, polarity is True for positive, False for negative
    clauses_opt = [[(abs(l) - 1, l > 0) for l in c] for c in clauses]
    num_clauses = len(clauses_opt)

    # Build variable-to-clause index for efficient conflict tracking
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses_opt):
        for var_idx, polarity in clause:
            var_to_clauses[var_idx].append((ci, polarity))

    best_assignment = None
    best_unsat_count = float('inf')

    def check_satisfied(assignment):
        """Check if all clauses are satisfied. Returns True if satisfied."""
        for clause in clauses_opt:
            sat = False
            for var_idx, polarity in clause:
                if assignment[var_idx] == polarity:
                    sat = True
                    break
            if not sat:
                return False
        return True

    def count_unsat(assignment):
        """Count number of unsatisfied clauses."""
        count = 0
        for clause in clauses_opt:
            sat = False
            for var_idx, polarity in clause:
                if assignment[var_idx] == polarity:
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    def min_conflicts_flip(assignment, unsat_clauses):
        """
        Min-conflicts heuristic: for each variable in an unsatisfied clause,
        compute how many clauses would be satisfied if we flipped that variable.
        Choose the variable that satisfies the most clauses (with random tie-breaking).
        """
        # For each variable in the unsat clause, compute gain if flipped
        best_var = -1
        best_gain = -1
        best_tie_break = -1
        rng_local = random.Random()
        
        # We'll evaluate all variables in the clause
        for var_idx, polarity in unsat_clauses:
            # Current state: assignment[var_idx] != polarity (since clause is unsat)
            # If we flip var_idx, it becomes assignment[var_idx] = polarity
            # Count how many clauses containing var_idx would become satisfied
            
            gain = 0
            for (ci, pol_in_clause) in var_to_clauses[var_idx]:
                # After flip, var_idx has value: not assignment[var_idx]
                new_val = not assignment[var_idx]
                # Check if this clause would be satisfied after flip
                # The clause is satisfied if any literal is true
                # We need to check if setting var_idx to new_val satisfies this clause
                # First, check if var_idx itself satisfies it
                if new_val == pol_in_clause:
                    gain += 1
                    continue
                # Otherwise, check other literals in the clause
                clause_satisfied = False
                for v, p in clauses_opt[ci]:
                    if v == var_idx:
                        continue
                    if assignment[v] == p:
                        clause_satisfied = True
                        break
                if clause_satisfied:
                    gain += 1
            
            # Add random tie-breaking
            tie_break = rng_local.random()
            if gain > best_gain or (gain == best_gain and tie_break > best_tie_break):
                best_gain = gain
                best_tie_break = tie_break
                best_var = var_idx
        
        if best_var >= 0:
            assignment[best_var] = not assignment[best_var]

    def walksat_step(assignment, unsat_clauses, rng):
        """Standard WalkSAT step: pick random unsat clause, flip random variable in it."""
        clause = rng.choice(unsat_clauses)
        literal = rng.choice(clause)
        var_idx = literal[0]
        assignment[var_idx] = not assignment[var_idx]

    def hybrid_step(assignment, unsat_clauses, rng, min_conflicts_prob=0.7):
        """Hybrid: use min-conflicts with some probability, else walksat."""
        if rng.random() < min_conflicts_prob:
            min_conflicts_flip(assignment, unsat_clauses)
        else:
            walksat_step(assignment, unsat_clauses, rng)

    # Try multiple restarts with different seeds
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        rng = random.Random(restart * 12345 + 67890)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Initial unsat count
        unsat_count = count_unsat(assignment)
        if unsat_count == 0:
            return assignment
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = list(assignment)
        
        # Adaptive parameters
        # For planted 3-SAT, the ratio of clauses to variables matters
        # Typical planted instances have m/n around 4-5
        m = num_clauses
        ratio = m / n if n > 0 else 0
        
        # Number of steps per restart
        # More steps for harder instances
        if ratio < 4.0:
            steps_per_restart = 5000
        elif ratio < 5.0:
            steps_per_restart = 10000
        else:
            steps_per_restart = 20000
        
        # Track best within this restart
        restart_best_unsat = unsat_count
        restart_best_assignment = list(assignment)
        
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
            
            # Recompute unsat clauses
            unsat_clauses = []
            for clause in clauses_opt:
                sat = False
                for var_idx, polarity in clause:
                    if assignment[var_idx] == polarity:
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(clause)
            
            if not unsat_clauses:
                return assignment
            
            current_unsat = len(unsat_clauses)
            if current_unsat < restart_best_unsat:
                restart_best_unsat = current_unsat
                restart_best_assignment = list(assignment)
            
            # Hybrid step
            hybrid_step(assignment, unsat_clauses, rng)
        
        # After restart, update global best
        if restart_best_unsat < best_unsat_count:
            best_unsat_count = restart_best_unsat
            best_assignment = list(restart_best_assignment)
        
        # If we solved it, return immediately
        if restart_best_unsat == 0:
            return restart_best_assignment

    # Return the best assignment found
    if best_assignment is not None:
        return best_assignment
    
    # Fallback: random assignment
    rng = random.Random(0)
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
