"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import defaultdict

def solve(instance):
    """
    Solve a planted satisfiable 3-SAT instance using a robust WalkSAT variant
    with:
    - Gain-based heuristic (minimizing net unsat clauses)
    - Randomness parameter tuned for planted 3-SAT
    - Multiple restarts with increasing budgets
    - Time deadline guard
    - Fast clause satisfaction tracking
    """
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline: finish well before 330s
    deadline = time.time() + 320.0
    
    # Pre-compute variable -> clause indices for fast gain computation
    var_to_clauses = defaultdict(list)
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var_to_clauses[abs(lit)].append(ci)
    
    m = len(clauses)
    
    # Best assignment found so far
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Restart strategy: increasing step budgets
    restarts = 0
    base_steps = 5000
    max_steps_per_restart = 50000
    
    while time.time() < deadline:
        restarts += 1
        # Steps for this restart: grow slightly with restarts, but cap
        steps = min(base_steps + restarts * 500, max_steps_per_restart)
        
        # Random initial assignment
        rng = random.Random(restarts * 12345 + 7)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Track unsat clauses
        unsat_clauses = set()
        for ci, clause in enumerate(clauses):
            sat = False
            for lit in clause:
                if assignment[abs(lit) - 1] == (lit > 0):
                    sat = True
                    break
            if not sat:
                unsat_clauses.add(ci)
        
        if not unsat_clauses:
            return assignment
        
        # WalkSAT loop
        # Parameters: noise (probability of random flip), weight for gain
        noise = 0.3  # fraction of random flips
        gain_weight = 1.0
        
        for step in range(steps):
            if time.time() >= deadline - 5.0:
                break
            
            if not unsat_clauses:
                return assignment
            
            # Track best in this restart
            if len(unsat_clauses) < best_unsat_count:
                best_unsat_count = len(unsat_clauses)
                best_assignment = assignment[:]
            
            # Pick a random unsat clause
            clause_idx = next(iter(unsat_clauses))
            clause = clauses[clause_idx]
            
            # Compute gain for each variable in the clause
            best_var = -1
            best_gain = float('inf')
            best_is_random = False
            
            # With some probability, do a random flip (noise)
            if rng.random() < noise:
                # Random flip among variables in the clause
                lit = rng.choice(clause)
                var = abs(lit) - 1
                best_var = var
                best_is_random = True
            else:
                # Compute gain: flipping var v changes satisfaction of clauses
                # containing v. Net gain = (# unsat clauses that become sat) 
                #              - (# sat clauses that become unsat)
                # We want to minimize unsat, so we want max positive gain.
                # Equivalently, minimize (new_unsat - old_unsat) = -gain.
                
                min_net_change = float('inf')
                for lit in clause:
                    var = abs(lit) - 1
                    # Current state of this literal
                    lit_satisfied = assignment[var] == (lit > 0)
                    
                    net_change = 0
                    # For each clause containing this variable
                    for ci in var_to_clauses[var + 1]:
                        c = clauses[ci]
                        # Determine if this clause is currently unsat
                        is_unsat = ci in unsat_clauses
                        
                        # Check if this variable's literal satisfies the clause
                        var_lit_satisfied_in_c = False
                        for l in c:
                            if abs(l) == var + 1:
                                var_lit_satisfied_in_c = (assignment[var] == (l > 0))
                                break
                        
                        # After flip, the variable's value changes
                        new_var_lit_satisfied = not var_lit_satisfied_in_c
                        
                        if is_unsat:
                            # Clause is currently unsat. After flip, it becomes sat
                            # only if the flipped literal was the only way to satisfy it
                            # and it becomes satisfied.
                            # Actually: a clause is unsat if all its literals are false.
                            # After flipping var, one literal changes.
                            # The clause becomes sat if the new value of this literal is True.
                            if new_var_lit_satisfied:
                                net_change -= 1  # one fewer unsat clause
                        else:
                            # Clause is currently sat. After flip, it becomes unsat
                            # only if this was the only satisfying literal and it becomes false.
                            if not new_var_lit_satisfied:
                                # Need to check if any other literal in the clause is sat
                                # But we know the clause is currently sat. If this literal
                                # was the only sat one, and it becomes false, clause becomes unsat.
                                # Quick check: is there another sat literal?
                                has_other_sat = False
                                for l in c:
                                    if abs(l) != var + 1:
                                        if assignment[abs(l) - 1] == (l > 0):
                                            has_other_sat = True
                                            break
                                if not has_other_sat:
                                    net_change += 1  # one more unsat clause
                    
                    if net_change < min_net_change:
                        min_net_change = net_change
                        best_var = var
                
                best_gain = min_net_change
        
        # After this restart, if we have a better assignment, keep it
        # But we already track best_assignment in the loop above.
        
        # Check if we solved it
        if not unsat_clauses:
            return assignment
        
        # If we've been restarting many times, check best
        if restarts >= 20 and best_assignment is not None:
            # Verify best_assignment
            all_sat = True
            for clause in clauses:
                sat = False
                for lit in clause:
                    if best_assignment[abs(lit) - 1] == (lit > 0):
                        sat = True
                        break
                if not sat:
                    all_sat = False
                    break
            if all_sat:
                return best_assignment
    
    # Return best assignment found
    if best_assignment is not None:
        return best_assignment
    
    # Fallback: return a random assignment
    rng = random.Random(42)
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
