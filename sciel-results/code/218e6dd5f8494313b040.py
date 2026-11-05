"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    if m == 0:
        return [False] * n
    
    # Deadline guard: 325s safety margin out of 330s limit
    start_time = time.time()
    deadline = start_time + 325.0
    
    # Pre-compute clause structure for fast updates
    # clause_vars[ci] = list of (var_idx, is_positive)
    clause_vars = []
    for c in clauses:
        cv = [(abs(l) - 1, l > 0) for l in c]
        clause_vars.append(cv)
    
    # var_clauses[vi] = list of (clause_idx, is_positive)
    var_clauses = [[] for _ in range(n)]
    for ci, cv in enumerate(clause_vars):
        for vi, pos in cv:
            var_clauses[vi].append((ci, pos))
    
    def init_state(seed):
        rng = random.Random(seed)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # clause_sat[ci] = number of literals in clause ci that are satisfied
        clause_sat = [0] * m
        for ci, cv in enumerate(clause_vars):
            for vi, pos in cv:
                if assignment[vi] == pos:
                    clause_sat[ci] += 1
        
        # unsat_count: number of unsatisfied clauses
        unsat_count = sum(1 for s in clause_sat if s == 0)
        
        return assignment, clause_sat, unsat_count, rng
    
    def update_flip(assignment, clause_sat, unsat_count, vi):
        """Flip variable vi, update clause_sat and unsat_count. Returns new unsat_count."""
        new_val = not assignment[vi]
        for ci, pos in var_clauses[vi]:
            old_sat = 1 if assignment[vi] == pos else 0
            new_sat = 1 if new_val == pos else 0
            delta = new_sat - old_sat
            clause_sat[ci] += delta
            if clause_sat[ci] == 0 and delta < 0:
                unsat_count += 1
            elif clause_sat[ci] > 0 and delta > 0:
                unsat_count -= 1
        assignment[vi] = new_val
        return unsat_count
    
    def find_unsat_clause(clause_sat):
        """Find an unsatisfied clause. Returns clause index or -1."""
        for ci in range(m):
            if clause_sat[ci] == 0:
                return ci
        return -1
    
    def walksat_step(assignment, clause_sat, unsat_count, rng, p_flip=0.5):
        """Perform one WalkSAT step. Returns new unsat_count."""
        if unsat_count == 0:
            return 0
        
        # Find an unsatisfied clause
        ci = -1
        while ci == -1:
            ci = rng.randrange(m)
            if clause_sat[ci] == 0:
                break
        
        # Choose a literal to flip
        # With probability p_flip, flip a random literal in the clause
        # Otherwise, use min-conflicts heuristic: flip the variable that satisfies
        # the most other unsatisfied clauses (damage-minimizing)
        cv = clause_vars[ci]
        
        if rng.random() < p_flip:
            # Random flip within the clause
            vi, pos = cv[rng.randrange(len(cv))]
        else:
            # Min-conflicts: for each variable in the clause, count how many
            # other unsatisfied clauses would be satisfied by flipping it
            best_vi = None
            best_score = -1
            for vi, pos in cv:
                score = 0
                for ci2, pos2 in var_clauses[vi]:
                    if ci2 != ci and clause_sat[ci2] == 0:
                        # If we flip vi to satisfy pos, does it satisfy clause ci2?
                        if not assignment[vi] == pos2:
                            # Currently not satisfying ci2 via vi
                            # After flip, vi will be (not assignment[vi])
                            # Check if that satisfies pos2
                            if (not assignment[vi]) == pos2:
                                score += 1
                if score > best_score:
                    best_score = score
                    best_vi = vi
            if best_vi is not None:
                vi = best_vi
            else:
                vi, pos = cv[rng.randrange(len(cv))]
        
        unsat_count = update_flip(assignment, clause_sat, unsat_count, vi)
        return unsat_count
    
    # Multiple restarts with adaptive step budgets
    best_assignment = None
    best_unsat = m + 1
    
    attempt = 0
    while time.time() < deadline:
        seed = 42 + attempt
        assignment, clause_sat, unsat_count, rng = init_state(seed)
        
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]
        
        # Adaptive step budget: more steps for harder instances
        # Base steps scale with m and n
        base_steps = max(5000, 50 * m)
        # Increase steps for later attempts (exploration)
        step_budget = int(base_steps * (1 + 0.1 * attempt))
        
        # Adjust p_flip: start with more random, then more heuristic
        p_flip = max(0.2, 0.5 - 0.05 * attempt)
        
        for step in range(step_budget):
            if time.time() > deadline - 5:
                break
            
            unsat_count = walksat_step(assignment, clause_sat, unsat_count, rng, p_flip)
            
            if unsat_count == 0:
                return assignment
            
            # Track best
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]
            
            # If stuck (no improvement in last 1000 steps), do a random restart of the walk
            # by flipping a few random variables
            if step > 0 and step % 1000 == 0:
                for _ in range(3):
                    vi = rng.randrange(n)
                    unsat_count = update_flip(assignment, clause_sat, unsat_count, vi)
                if unsat_count == 0:
                    return assignment
        
        attempt += 1
    
    # Return best found
    if best_assignment is not None:
        return best_assignment
    return [False] * n
# EVOLVE-BLOCK-END
