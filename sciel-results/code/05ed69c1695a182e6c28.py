"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 300s limit

    # Preprocess: convert clauses to 0-based signed literals
    # Each clause is a list of signed 1-based literals
    # Convert to 0-based: literal l -> (l-1) if l>0, -(l+1) if l<0... 
    # Actually let's keep as (var_index, is_positive) tuples for clarity
    # var_index = abs(l) - 1, is_positive = (l > 0)
    
    parsed_clauses = []
    for c in clauses:
        parsed = []
        for l in c:
            vi = abs(l) - 1
            pos = (l > 0)
            parsed.append((vi, pos))
        parsed_clauses.append(parsed)
    
    m = len(parsed_clauses)
    
    # Track clause satisfaction status efficiently
    # For each clause, we track: 
    #   sat_count: number of literals currently satisfied
    #   For each literal in the clause, whether it's satisfied
    
    # State: assignment as list of booleans
    # We'll maintain:
    #   clause_sat[i] = number of satisfied literals in clause i
    #   For flipping var v: for each clause containing v, we need to know
    #   how many of the literals involving v are currently satisfied
    
    # Build variable-to-clause index
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(parsed_clauses):
        for (vi, pos) in clause:
            var_to_clauses[vi].append((ci, pos))
    
    # Initialize assignment
    rng = random.Random(42)
    assignment = [rng.random() < 0.5 for _ in range(n)]
    
    # Initialize clause satisfaction counts
    clause_sat = [0] * m
    for ci, clause in enumerate(parsed_clauses):
        for (vi, pos) in clause:
            if assignment[vi] == pos:
                clause_sat[ci] += 1
    
    def is_satisfied():
        return all(cs >= 1 for cs in clause_sat)
    
    def get_unsat_clauses():
        return [i for i in range(m) if clause_sat[i] == 0]
    
    def flip_var(v):
        """Flip variable v and update clause_sat. Returns (newly_satisfied, newly_unsatisfied)"""
        old_val = assignment[v]
        new_val = not old_val
        assignment[v] = new_val
        
        newly_sat = 0
        newly_unsat = 0
        for (ci, pos) in var_to_clauses[v]:
            if new_val == pos:
                # This literal is now satisfied
                if clause_sat[ci] == 0:
                    newly_sat += 1
                clause_sat[ci] += 1
            else:
                # This literal is now not satisfied
                if clause_sat[ci] == 1:
                    newly_unsat += 1
                clause_sat[ci] -= 1
        return newly_sat, newly_unsat
    
    def unflip_var(v):
        """Undo a flip of variable v (flip it back)."""
        flip_var(v)  # flipping again undoes it
    
    # WalkSAT with clause weighting and restarts
    best_assignment = assignment[:]
    best_unsat_count = sum(1 for cs in clause_sat if cs == 0)
    
    if best_unsat_count == 0:
        return assignment
    
    # Weighted WalkSAT
    # Parameters
    P = 0.55  # probability of choosing a random literal from an unsat clause
    W = 1.0   # weight for clause frequency
    max_steps_per_restart = 10000
    max_restarts = 200
    
    # Clause weights: frequency of appearance
    clause_weight = [1.0] * m
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Reinitialize assignment with phase saving from previous best
        if restart == 0:
            # Already initialized
            pass
        else:
            # Use phase saving: keep the best assignment as starting point
            # with some randomization
            assignment = best_assignment[:]
            # Randomly flip some variables
            for v in range(n):
                if rng.random() < 0.3:
                    assignment[v] = not assignment[v]
            
            # Rebuild clause_sat
            clause_sat = [0] * m
            for ci, clause in enumerate(parsed_clauses):
                for (vi, pos) in clause:
                    if assignment[vi] == pos:
                        clause_sat[ci] += 1
        
        unsat_count = sum(1 for cs in clause_sat if cs == 0)
        if unsat_count == 0:
            return assignment[:]
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
        
        # WalkSAT loop
        for step in range(max_steps_per_restart):
            if time.time() > deadline - 5.0:
                break
            
            unsat_clauses = get_unsat_clauses()
            if not unsat_clauses:
                return assignment[:]
            
            unsat_count = len(unsat_clauses)
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment[:]
            
            # Choose an unsatisfied clause (weighted by clause_weight)
            # Use weighted random selection
            total_weight = sum(clause_weight[ci] for ci in unsat_clauses)
            if total_weight <= 0:
                ci = rng.choice(unsat_clauses)
            else:
                r = rng.random() * total_weight
                cum = 0
                ci = unsat_clauses[0]
                for uci in unsat_clauses:
                    cum += clause_weight[uci]
                    if r <= cum:
                        ci = uci
                        break
                # Fallback
                if cum < r:
                    ci = unsat_clauses[-1]
            
            clause = parsed_clauses[ci]
            
            # Choose a literal to flip
            if rng.random() < P:
                # Random literal from the clause
                (vi, pos) = rng.choice(clause)
            else:
                # Greedy: choose the literal that maximizes (newly_satisfied - newly_unsatisfied)
                best_gain = -1
                best_vi = clause[0][0]
                for (vi, pos) in clause:
                    # Compute gain if we flip vi
                    gain = 0
                    for (uci, upos) in var_to_clauses[vi]:
                        if assignment[vi] == upos:
                            # Currently satisfied, flipping makes it unsatisfied
                            if clause_sat[uci] == 1:
                                gain -= 1
                        else:
                            # Currently unsatisfied, flipping makes it satisfied
                            if clause_sat[uci] == 0:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_vi = vi
                vi = best_vi
            
            # Flip the variable
            newly_sat, newly_unsat = flip_var(vi)
            
            # Update clause weights
            if newly_sat > newly_unsat:
                # Good move, increase weight of clauses that became satisfied
                pass
            else:
                # Bad move, increase weight of the chosen clause
                clause_weight[ci] *= 1.5
                # Decay other weights slightly
                # (skip for speed)
            
            # Check if all satisfied
            if newly_sat > 0 and all(cs >= 1 for cs in clause_sat):
                return assignment[:]
            
            # Check unsat count
            current_unsat = sum(1 for cs in clause_sat if cs == 0)
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]
        
        # After each restart, decay clause weights
        for i in range(m):
            clause_weight[i] *= 0.8
            if clause_weight[i] < 0.1:
                clause_weight[i] = 1.0
    
    return best_assignment
# EVOLVE-BLOCK-END
