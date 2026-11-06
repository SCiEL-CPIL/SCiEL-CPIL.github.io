"""Optimized WalkSAT with fast incremental state updates."""
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 285.0
    m = len(clauses)
    
    if m == 0:
        return [False] * n
    
    # Precompute structures
    var_clauses = [[] for _ in range(n)]
    clause_lits = []
    for ci, clause in enumerate(clauses):
        lits = []
        for lit in clause:
            vi = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            lits.append((vi, sign))
            var_clauses[vi].append((ci, sign))
        clause_lits.append(lits)

    def init_state(assignment):
        """Compute clause_sat_count and unsat_count."""
        clause_sat_count = [0] * m
        unsat_count = 0
        for ci, lits in enumerate(clause_lits):
            cnt = 0
            for vi, sign in lits:
                if assignment[vi] == (sign == 1):
                    cnt += 1
            clause_sat_count[ci] = cnt
            if cnt == 0:
                unsat_count += 1
        return clause_sat_count, unsat_count

    def flip_var(assignment, vi, clause_sat_count, unsat_count):
        """Flip variable vi and update state incrementally."""
        old_val = assignment[vi]
        new_val = not old_val
        assignment[vi] = new_val
        
        delta_unsat = 0
        for ci, sign in var_clauses[vi]:
            old_lit_satisfied = (old_val == (sign == 1))
            new_lit_satisfied = (new_val == (sign == 1))
            
            old_sat = clause_sat_count[ci]
            
            if old_lit_satisfied and not new_lit_satisfied:
                clause_sat_count[ci] = old_sat - 1
                if old_sat == 1:
                    delta_unsat += 1
            elif not old_lit_satisfied and new_lit_satisfied:
                clause_sat_count[ci] = old_sat + 1
                if old_sat == 0:
                    delta_unsat -= 1
        
        return unsat_count + delta_unsat

    def evaluate(assignment, clause_sat_count, unsat_count):
        """Run local search."""
        rng = random.Random()
        max_steps = 10000
        best_local = unsat_count
        best_local_assign = assignment[:]
        
        for step in range(max_steps):
            if time.time() >= deadline:
                return best_local_assign, best_local
            
            if unsat_count == 0:
                return assignment, 0
            
            if unsat_count < best_local:
                best_local = unsat_count
                best_local_assign = assignment[:]
            
            # Pick a random unsatisfied clause
            # Optimization: maintain a list of unsat clauses? 
            # For now, random sampling is fine if m is not huge.
            # If m is large, we might need a better strategy, but 50 vars usually implies m < 1000.
            ci = rng.randrange(m)
            # Retry a few times if we hit a sat clause
            for _ in range(5):
                if clause_sat_count[ci] == 0:
                    break
                ci = rng.randrange(m)
            
            if clause_sat_count[ci] != 0:
                # Fallback: find one
                unsat_indices = [i for i in range(m) if clause_sat_count[i] == 0]
                if not unsat_indices:
                    return assignment, 0
                ci = rng.choice(unsat_indices)
            
            # Choose variable to flip
            # Heuristic: Min-conflicts
            # Count how many unsat clauses are affected by flipping each var in the clause
            # If we flip var v in clause C, C becomes satisfied (since it was unsat, and we flip a lit in it).
            # The cost is the number of OTHER unsat clauses containing v that might become unsat?
            # Actually, standard WalkSAT: 
            # For each var v in C, let cost(v) be the number of unsatisfied clauses containing v (excluding C, since C will be satisfied).
            # If v is flipped, C becomes satisfied.
            # Other clauses containing v:
            #   If they were unsat:
            #     If the lit in that clause for v was the only sat lit (impossible, it was unsat), or if flipping v makes it sat?
            #     If flipping v makes an unsat clause sat, cost decreases.
            #     If flipping v makes a sat clause unsat, cost increases.
            
            # Simpler heuristic: 
            # Pick the variable that minimizes the number of unsatisfied clauses it is part of.
            # This is a proxy for "least disruptive".
            
            min_cost = float('inf')
            best_var = -1
            for vi, sign in clause_lits[ci]:
                # Count unsat clauses containing vi
                # We don't have a direct list of unsat clauses per var.
                # We can compute it on the fly:
                cost = 0
                for c_idx, s in var_clauses[vi]:
                    if clause_sat_count[c_idx] == 0:
                        cost += 1
                if cost < min_cost:
                    min_cost = cost
                    best_var = vi
            
            # Noise: with prob 0.5, pick random var from clause
            if rng.random() < 0.5:
                best_var = clause_lits[ci][rng.randrange(3)][0]
            
            if best_var >= 0:
                unsat_count = flip_var(assignment, best_var, clause_sat_count, unsat_count)
        
        return best_local_assign, best_local

    # Main search loop
    best_unsat = float('inf')
    best_assignment = None
    attempt = 0
    
    while time.time() < deadline:
        attempt += 1
        rng_init = random.Random(attempt * 7919 + 42)
        assignment = [rng_init.random() < 0.5 for _ in range(n)]
        
        clause_sat_count, unsat_count = init_state(assignment)
        
        if unsat_count == 0:
            return assignment
        
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]
        
        local_assign, local_best = evaluate(assignment, clause_sat_count, unsat_count)
        
        if local_best == 0:
            return local_assign
        
        if local_best < best_unsat:
            best_unsat = local_best
            best_assignment = local_assign[:]
            
        if time.time() > deadline - 5:
            break
    
    if best_assignment is not None:
        return best_assignment
    
    return [False] * n
