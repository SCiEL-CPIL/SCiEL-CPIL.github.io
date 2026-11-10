"""
3-SAT solver with Spectral Initialization + Hybrid WalkSAT + Restarts.

This solver initializes the assignment using a spectral heuristic derived from
the clause structure (analogous to spectral bisection for Max-Cut approximations
in graph problems) to provide a strong starting point for the local search.
The core search is a fast WalkSAT implementation with incremental clause tracking.
"""
import random
import time
import math
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["instance_clauses"] if "instance_clauses" in instance else instance["clauses"]
    
    start_time = time.time()
    deadline = start_time + 290  # Safety margin

    # 1. Preprocessing
    seen = set()
    filtered_clauses = []
    for c in clauses:
        lits = set(c)
        # Remove tautologies
        if any(-l in lits for l in lits):
            continue
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            filtered_clauses.append(c)
    clauses = filtered_clauses
    clause_count = len(clauses)
    
    if clause_count == 0:
        return [True] * n

    # Build variable occurrence lists for incremental updates
    var_pos = [[] for _ in range(n)]
    var_neg = [[] for _ in range(n)]
    for i, c in enumerate(clauses):
        for l in c:
            idx = abs(l) - 1
            if l > 0:
                var_pos[idx].append(i)
            else:
                var_neg[idx].append(i)

    # 2. Spectral Initialization
    # Construct a matrix M where M[i][j] suggests similarity/dissimilarity
    # For each clause, we add "pressure" to variables to be consistent with
    # the clause being satisfied. 
    # Heuristic: If (l1, l2, l3) is a clause, it's "good" if at least one is True.
    # We can model this as: we want to avoid the state where all are False.
    # A simpler spectral proxy for initialization in random 3-SAT:
    # Build a graph where edges exist between variables in the same clause.
    # Weight +1 if they have the same sign, -1 if opposite.
    # The principal eigenvector of this matrix often correlates with the planted solution
    # due to the local structure of planted instances.
    
    M = np.zeros((n, n), dtype=np.float64)
    for c in clauses:
        # c has 3 literals
        for i in range(3):
            for j in range(i+1, 3):
                li = c[i]
                lj = c[j]
                vi = abs(li) - 1
                vj = abs(lj) - 1
                if vi == vj: continue # Should not happen in well-formed 3-SAT usually, but safe
                # If signs are same, they "want" to be same value? 
                # Actually, in 3-SAT (l1 OR l2 OR l3), if l1 is True, clause is sat.
                # The "conflict" is when all are False.
                # Let's use a standard correlation matrix approach for initialization.
                # If li and lj have same sign, a "True" assignment for both satisfies the clause strongly.
                # If opposite, one True and one False satisfies it.
                # We'll accumulate a "preference" matrix.
                if (li > 0) == (lj > 0):
                    M[vi][vj] += 1
                    M[vj][vi] += 1
                else:
                    M[vi][vj] -= 1
                    M[vj][vi] -= 1

    # Add self-loop weights based on degree to encourage stability?
    # Or just use the off-diagonal structure.
    try:
        # Compute principal eigenvector
        # For small n, full eigh is fine
        eigvals, eigvecs = np.linalg.eigh(M)
        # Principal eigenvector is the last one (largest eigenvalue)
        principal_vec = eigvecs[:, -1]
        
        # Initialize assignment based on sign of principal vector
        # Add small random noise to break symmetries
        noise = np.random.normal(0, 0.1, n)
        init_score = principal_vec + noise
        assignment = [s > 0 for s in init_score]
    except Exception:
        assignment = [random.random() < 0.5 for _ in range(n)]

    # Helper functions for incremental updates
    def count_unsat(assignment):
        count = 0
        for c in clauses:
            sat = False
            for l in c:
                idx = abs(l) - 1
                if assignment[idx] == (l > 0):
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    def get_sat_counts(assignment):
        counts = [0] * clause_count
        for i, c in enumerate(clauses):
            sc = 0
            for l in c:
                idx = abs(l) - 1
                if assignment[idx] == (l > 0):
                    sc += 1
            counts[i] = sc
        return counts

    # Check initial assignment
    current_unsat = count_unsat(assignment)
    if current_unsat == 0:
        return assignment

    best_assignment = assignment[:]
    best_unsat_count = current_unsat

    rng = random.Random(42)
    
    # 3. Local Search Loop (WalkSAT with Restarts)
    max_restarts = 100
    steps_per_restart = 10000
    noise_prob = 0.25

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Initialize state for this restart
        if restart == 0:
            # Use the spectral init
            pass
        else:
            # Perturb from best found so far
            if best_assignment is not None and rng.random() < 0.8:
                assignment = best_assignment[:]
                # Flip a few random variables
                num_flips = max(1, n // 5)
                for _ in range(num_flips):
                    v = rng.randint(0, n - 1)
                    assignment[v] = not assignment[v]
            else:
                # Random restart
                assignment = [rng.random() < 0.5 for _ in range(n)]
        
        sat_counts = get_sat_counts(assignment)
        unsat_set = set(i for i, c in enumerate(clauses) if sat_counts[i] == 0)
        
        if not unsat_set:
            return assignment

        current_unsat = len(unsat_set)
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]

        # WalkSAT inner loop
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
            
            if not unsat_set:
                return assignment

            # 1. Select a clause to satisfy
            # Randomly pick from unsatisfied clauses
            clause_idx = next(iter(unsat_set)) if len(unsat_set) == 1 else list(unsat_set)[rng.randint(0, len(unsat_set) - 1)]
            clause = clauses[clause_idx]

            # 2. Select a variable to flip
            # With probability noise, flip a random variable
            if rng.random() < noise_prob:
                var = rng.randint(0, n - 1)
            else:
                # Choose variable from the clause that maximizes "gain"
                # Gain = number of other unsat clauses that would become satisfied
                #       minus number of currently sat clauses that would become unsat
                best_var = -1
                best_gain = -float('inf')
                
                for l in clause:
                    var = abs(l) - 1
                    # If this literal is already satisfied, flipping it makes it unsat (bad)
                    # If it is unsat, flipping it makes it sat (good)
                    
                    gain = 0
                    # Calculate change in satisfied clauses
                    # We need to know which clauses are affected by flipping 'var'
                    # Affected clauses are those containing var as positive or negative literal
                    
                    # Clauses where var appears as positive literal (l > 0)
                    # If we flip var, these clauses lose a satisfied literal if var was True, gain if var was False
                    # Wait, the logic is:
                    # Current state: assignment[var]
                    # New state: not assignment[var]
                    
                    # Let's iterate over affected clauses to compute exact delta in unsat count
                    # This is O(degree(var)) per candidate var in clause. Clause has 3 vars.
                    # Total O(3 * degree(var)) per step.
                    
                    # Helper to calculate delta in unsat count for a flip
                    def calc_delta(var_idx):
                        old_val = assignment[var_idx]
                        delta_unsat = 0
                        
                        # Check positive occurrences
                        for ci in var_pos[var_idx]:
                            c = clauses[ci]
                            # Current sat count
                            sc = sat_counts[ci]
                            
                            # If old_val is True, this literal contributes 1 to sc.
                            # If we flip to False, sc decreases by 1.
                            # If sc was 1, it becomes 0 (unsat). Delta unsat +1.
                            # If sc was >1, it becomes >0 (still sat). Delta unsat 0.
                            if old_val:
                                if sc == 1:
                                    delta_unsat += 1
                                # else delta 0
                            else:
                                # old_val False, flipping to True. sc increases by 1.
                                # If sc was 0, it becomes 1 (sat). Delta unsat -1.
                                # If sc was >0, it becomes >1 (still sat). Delta unsat 0.
                                if sc == 0:
                                    delta_unsat -= 1
                            
                        # Check negative occurrences
                        for ci in var_neg[var_idx]:
                            c = clauses[ci]
                            sc = sat_counts[ci]
                            
                            # Literal is -var.
                            # If old_val is False, -var is True. Contributes 1.
                            # If old_val is True, -var is False. Contributes 0.
                            
                            if not old_val:
                                # old_val False -> -var True. Flipping to True -> -var False.
                                # sc decreases by 1.
                                if sc == 1:
                                    delta_unsat += 1
                            else:
                                # old_val True -> -var False. Flipping to False -> -var True.
                                # sc increases by 1.
                                if sc == 0:
                                    delta_unsat -= 1
                                
                        return delta_unsat

                    d = calc_delta(var)
                    if d < best_gain:
                        best_gain = d
                        best_var = var

                if best_var == -1:
                    var = rng.randint(0, n - 1)
                else:
                    var = best_var

            # Perform the flip
            old_val = assignment[var]
            assignment[var] = not old_val
            
            # Update sat_counts and unsat_set incrementally
            for ci in var_pos[var]:
                if old_val:
                    sat_counts[ci] -= 1
                    if sat_counts[ci] == 0:
                        unsat_set.add(ci)
                else:
                    if sat_counts[ci] == 0:
                        unsat_set.discard(ci)
                    sat_counts[ci] += 1
                    
            for ci in var_neg[var]:
                if not old_val:
                    sat_counts[ci] -= 1
                    if sat_counts[ci] == 0:
                        unsat_set.add(ci)
                else:
                    if sat_counts[ci] == 0:
                        unsat_set.discard(ci)
                    sat_counts[ci] += 1

            # Update best
            current_unsat = len(unsat_set)
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]
                if current_unsat == 0:
                    return assignment

    # 4. Final Polish: Try single flips from best assignment
    if best_assignment is not None:
        assignment = best_assignment[:]
        sat_counts = get_sat_counts(assignment)
        unsat_set = set(i for i, c in enumerate(clauses) if sat_counts[i] == 0)
        
        if len(unsat_set) <= 5:
            # Try flipping each variable
            for var in range(n):
                if time.time() > deadline:
                    break
                old_val = assignment[var]
                assignment[var] = not old_val
                
                # Update
                for ci in var_pos[var]:
                    if old_val:
                        sat_counts[ci] -= 1
                        if sat_counts[ci] == 0:
                            unsat_set.add(ci)
                    else:
                        if sat_counts[ci] == 0:
                            unsat_set.discard(ci)
                        sat_counts[ci] += 1
                for ci in var_neg[var]:
                    if not old_val:
                        sat_counts[ci] -= 1
                        if sat_counts[ci] == 0:
                            unsat_set.add(ci)
                    else:
                        if sat_counts[ci] == 0:
                            unsat_set.discard(ci)
                        sat_counts[ci] += 1
                
                if not unsat_set:
                    return assignment
                
                # Revert
                assignment[var] = old_val
                for ci in var_pos[var]:
                    if old_val:
                        sat_counts[ci] += 1
                        if sat_counts[ci] == 0:
                            unsat_set.add(ci)
                    else:
                        if sat_counts[ci] == 0:
                            unsat_set.discard(ci)
                        sat_counts[ci] -= 1
                for ci in var_neg[var]:
                    if not old_val:
                        sat_counts[ci] += 1
                        if sat_counts[ci] == 0:
                            unsat_set.add(ci)
                    else:
                        if sat_counts[ci] == 0:
                            unsat_set.discard(ci)
                        sat_counts[ci] -= 1

    return best_assignment if best_assignment is not None else [True] * n

