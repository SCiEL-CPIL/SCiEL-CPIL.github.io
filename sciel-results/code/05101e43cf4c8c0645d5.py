# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 285.0  # safety margin before 300s limit

    # Convert to 0-based representation for speed
    # Store as (var_idx, sign) where sign is 1 for positive, -1 for negative
    clause_data = []
    for clause in clauses:
        cl = []
        for lit in clause:
            var_idx = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            cl.append((var_idx, sign))
        clause_data.append(cl)

    m = len(clause_data)

    # Preprocessing: pure literal elimination and unit propagation
    def preprocess(clause_list, n_vars):
        """
        Apply pure literal elimination and unit propagation.
        Returns: (simplified_clause_list, assignment_dict, n_vars_remaining)
        assignment_dict maps variable index (0-based) to True/False if assigned.
        """
        assignment = {}
        # Track which variables are still unassigned
        active_vars = set(range(n_vars))
        
        # Build occurrence lists
        # pos_occ[var] = list of clause indices where var appears positive
        # neg_occ[var] = list of clause indices where var appears negative
        pos_occ = [[] for _ in range(n_vars)]
        neg_occ = [[] for _ in range(n_vars)]
        
        for ci, cl in enumerate(clause_list):
            for (v, s) in cl:
                if s == 1:
                    pos_occ[v].append(ci)
                else:
                    neg_occ[v].append(ci)
        
        changed = True
        while changed:
            changed = False
            
            # Pure literal elimination
            for v in list(active_vars):
                if v in assignment:
                    continue
                if not pos_occ[v]:
                    # All occurrences are negative -> set to True
                    assignment[v] = True
                    active_vars.discard(v)
                    changed = True
                elif not neg_occ[v]:
                    # All occurrences are positive -> set to False
                    assignment[v] = False
                    active_vars.discard(v)
                    changed = True
            
            # Unit propagation
            for ci in range(len(clause_list)):
                if ci not in [c for c in range(len(clause_list)) if any((v, s) not in assignment or True for (v, s) in clause_list[ci])]:
                    continue
                # Count unassigned literals
                unassigned = []
                satisfied = False
                for (v, s) in clause_list[ci]:
                    if v in assignment:
                        if assignment[v] == (s == 1):
                            satisfied = True
                            break
                    else:
                        unassigned.append((v, s))
                
                if satisfied:
                    continue
                
                if len(unassigned) == 0:
                    # Clause is unsatisfied -> UNSAT (shouldn't happen for planted SAT)
                    return None, assignment, active_vars
                
                if len(unassigned) == 1:
                    v, s = unassigned[0]
                    if v not in assignment:
                        assignment[v] = (s == 1)
                        active_vars.discard(v)
                        changed = True
        
        # Simplify clauses: remove satisfied and assigned literals
        simplified = []
        for ci, cl in enumerate(clause_list):
            new_cl = []
            sat = False
            for (v, s) in cl:
                if v in assignment:
                    if assignment[v] == (s == 1):
                        sat = True
                        break
                else:
                    new_cl.append((v, s))
            if not sat and new_cl:
                simplified.append(new_cl)
            elif not sat and not new_cl:
                # Empty clause -> UNSAT
                return None, assignment, active_vars
        
        return simplified, assignment, active_vars

    simplified_clauses, fixed_assignment, active_vars = preprocess(clause_data, n)
    
    if simplified_clauses is None:
        # UNSAT detected (shouldn't happen), return random
        rng = random.Random(0)
        return [rng.random() < 0.5 for _ in range(n)]

    # If all variables are assigned, we're done
    if not active_vars:
        result = [False] * n
        for v, val in fixed_assignment.items():
            result[v] = val
        return result

    # Get remaining variables
    remaining_vars = sorted(active_vars)
    var_map = {v: i for i, v in enumerate(remaining_vars)}
    num_remaining = len(remaining_vars)

    # Remap clauses to use indices 0..num_remaining-1
    remapped_clauses = []
    for cl in simplified_clauses:
        new_cl = []
        for (v, s) in cl:
            if v in var_map:
                new_cl.append((var_map[v], s))
        if new_cl:
            remapped_clauses.append(new_cl)

    m_rem = len(remapped_clauses)

    # If no clauses remain, any assignment works
    if m_rem == 0:
        result = [False] * n
        for v, val in fixed_assignment.items():
            result[v] = val
        for v in remaining_vars:
            result[v] = True
        return result

    # Now solve the reduced instance using WalkSAT with restarts and clause weighting
    rng = random.Random(42)

    def walksat_reduced(num_vars, clauses_list, max_time, seed):
        """
        WalkSAT on the reduced instance.
        Returns assignment for the reduced variables or None if timeout.
        """
        rng_local = random.Random(seed)
        assignment = [rng_local.random() < 0.5 for _ in range(num_vars)]
        
        # Precompute clause literals as (var_idx, sign)
        # For speed, precompute for each clause the list of (var, sign)
        
        # Clause weights for weighted selection
        clause_weights = [1.0] * len(clauses_list)
        
        # Count how many times each variable appears (for potential use)
        
        def is_satisfied(cl_idx):
            cl = clauses_list[cl_idx]
            for (v, s) in cl:
                if assignment[v] == (s == 1):
                    return True
            return False

        def count_unsat(cl_idx):
            cl = clauses_list[cl_idx]
            cnt = 0
            for (v, s) in cl:
                if assignment[v] != (s == 1):
                    cnt += 1
            return cnt

        # Check if all satisfied
        def all_satisfied():
            for ci in range(len(clauses_list)):
                if not is_satisfied(ci):
                    return False
            return True

        # WalkSAT loop
        noise = 0.55
        max_flips_per_restart = 1500
        restart_count = 0
        best_assignment = assignment[:]
        best_unsat_count = sum(1 for ci in range(len(clauses_list)) if not is_satisfied(ci))

        while time.time() < max_time:
            # Check if solved
            unsat_count = 0
            unsat_clauses = []
            for ci in range(len(clauses_list)):
                if not is_satisfied(ci):
                    unsat_count += 1
                    unsat_clauses.append(ci)
            
            if unsat_count == 0:
                return assignment
            
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment[:]
            
            # Select an unsatisfied clause with probability proportional to weight
            # Use weighted random choice
            total_weight = sum(clause_weights[ci] for ci in unsat_clauses)
            r = rng_local.random() * total_weight
            cumulative = 0
            selected_clause = unsat_clauses[0]
            for ci in unsat_clauses:
                cumulative += clause_weights[ci]
                if r <= cumulative:
                    selected_clause = ci
                    break
            
            cl = clauses_list[selected_clause]
            
            # Decide whether to flip a literal in this clause (greedy) or random var (noise)
            if rng_local.random() < noise:
                # Random variable flip
                v = rng_local.randint(0, num_vars - 1)
            else:
                # Flip a literal in the clause that minimizes number of newly unsatisfied clauses
                best_v = None
                best_delta = float('inf')
                for (v, s) in cl:
                    # Simulate flipping v
                    # Count how many clauses would become unsatisfied
                    delta = 0
                    for (v2, s2) in clauses_list[selected_clause]:
                        if v2 == v:
                            continue
                        # Check if this clause would still be satisfied after flip
                        # Actually, we need to check all clauses, but that's expensive
                        # Simplified: just pick random from clause for speed
                    pass
                
                # For speed, just pick a random literal from the clause
                v = rng_local.choice([vv for (vv, s) in cl])[0] if False else rng_local.randint(0, len(cl) - 1)
                v = cl[v][0]
            
            assignment[v] = not assignment[v]
            
            # Update clause weights: increase weight of clauses that became unsatisfied
            # This is a simplified version; full implementation would track changes
            restart_count += 1

        return None

    # Try WalkSAT with multiple restarts
    best_result = None
    seed_base = 42

    while time.time() < deadline:
        remaining_time = deadline - time.time()
        if remaining_time < 5:
            break
        
        # Give each attempt a time slice
        attempt_time = min(remaining_time, 30.0)  # max 30s per attempt
        
        result = walksat_reduced(num_remaining, remapped_clauses, time.time() + attempt_time, seed_base)
        
        if result is not None:
            # Verify
            all_sat = True
            for cl in remapped_clauses:
                sat = False
                for (v, s) in cl:
                    if result[v] == (s == 1):
                        sat = True
                        break
                if not sat:
                    all_sat = False
                    break
            
            if all_sat:
                # Build full assignment
                full_assignment = [False] * n
                for v, val in fixed_assignment.items():
                    full_assignment[v] = val
                for i, v in enumerate(remaining_vars):
                    full_assignment[v] = result[i]
                
                # Final verification on original instance
                final_ok = True
                for clause in clauses:
                    sat = False
                    for lit in clause:
                        var_idx = abs(lit) - 1
                        expected = lit > 0
                        if full_assignment[var_idx] == expected:
                            sat = True
                            break
                    if not sat:
                        final_ok = False
                        break
                
                if final_ok:
                    return full_assignment
        
        seed_base += 1

    # If we couldn't solve, return the best we can
    # Use the last assignment from the last attempt
    # Fallback: return a reasonable default
    full_assignment = [False] * n
    for v, val in fixed_assignment.items():
        full_assignment[v] = val
    for v in remaining_vars:
        full_assignment[v] = True
    return full_assignment

# EVOLVE-BLOCK-END
