"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Deadline: 330s wall-clock, use 325s for safety
    deadline = time.time() + 325.0
    
    # Preprocess: convert clauses to 0-based indices with sign
    # clause_vars[i] = list of (var_idx, is_positive)
    clause_vars = []
    for c in clauses:
        cv = []
        for l in c:
            var_idx = abs(l) - 1
            is_pos = l > 0
            cv.append((var_idx, is_pos))
        clause_vars.append(cv)
    
    # Build variable-to-clause index
    var_clauses = [[] for _ in range(n)]
    for i, cv in enumerate(clause_vars):
        for var_idx, is_pos in cv:
            var_clauses[var_idx].append((i, is_pos))
    
    def init_sat_counts(assignment):
        sat_counts = [0] * m
        for i, cv in enumerate(clause_vars):
            cnt = 0
            for var_idx, is_pos in cv:
                if assignment[var_idx] == is_pos:
                    cnt += 1
            sat_counts[i] = cnt
        return sat_counts
    
    def count_unsat_from_counts(sat_counts):
        return sum(1 for c in sat_counts if c == 0)
    
    def walk_sat(assignment, rng, max_flips):
        """WalkSAT with adaptive noise and incremental clause tracking."""
        sat_counts = init_sat_counts(assignment)
        unsat_count = count_unsat_from_counts(sat_counts)
        if unsat_count == 0:
            return True
        
        unsat_clauses = [i for i, c in enumerate(sat_counts) if c == 0]
        if not unsat_clauses:
            return True
        
        flip_count = 0
        while flip_count < max_flips:
            if not unsat_clauses:
                return True
            
            # Pick a random unsatisfied clause
            ci = rng.choice(unsat_clauses)
            cv = clause_vars[ci]
            
            # Adaptive noise: decrease over time
            noise = 0.5 * max(0, 1 - flip_count / max_flips)
            
            # For each literal in the clause, compute the delta efficiently
            best_var = -1
            best_delta = float('inf')
            
            for var_idx, is_pos in cv:
                delta = 0
                old_val = assignment[var_idx]
                new_val = not old_val
                
                for ci2, is_pos2 in var_clauses[var_idx]:
                    cv2 = clause_vars[ci2]
                    
                    # Is this clause currently satisfied?
                    cur_sat = sat_counts[ci2] > 0
                    
                    # After flipping var_idx, is it still satisfied?
                    # If the clause was already satisfied by another literal, it stays satisfied
                    # If it was unsatisfied, it becomes satisfied only if the flipped literal becomes true
                    # If it was satisfied only by this literal, it becomes unsatisfied
                    
                    if cur_sat:
                        # Check if this variable is the only one satisfying the clause
                        # Count how many literals in cv2 are satisfied
                        sat_cnt = 0
                        for v2, p2 in cv2:
                            if v2 == var_idx:
                                if old_val == p2:
                                    sat_cnt += 1
                            else:
                                if assignment[v2] == p2:
                                    sat_cnt += 1
                        
                        if sat_cnt == 1 and old_val == is_pos2:
                            # This variable was the only one satisfying the clause
                            # After flip, it becomes unsatisfied
                            delta += 1
                    else:
                        # Clause is unsatisfied. After flip, is it satisfied?
                        if new_val == is_pos2:
                            delta -= 1
                
                # Apply noise
                delta += rng.gauss(0, noise)
                
                if delta < best_delta:
                    best_delta = delta
                    best_var = var_idx
            
            if best_var < 0:
                # Fallback: random flip
                best_var = rng.choice([v for v, _ in cv])[0]
            
            # Flip the variable
            old_val = assignment[best_var]
            assignment[best_var] = not old_val
            new_val = assignment[best_var]
            
            # Update sat_counts incrementally
            for ci2, is_pos2 in var_clauses[best_var]:
                cv2 = clause_vars[ci2]
                old_contrib = (old_val == is_pos2)
                new_contrib = (new_val == is_pos2)
                
                if old_contrib and not new_contrib:
                    sat_counts[ci2] -= 1
                    if sat_counts[ci2] == 0:
                        unsat_clauses.append(ci2)
                elif not old_contrib and new_contrib:
                    sat_counts[ci2] += 1
                    if sat_counts[ci2] == 1 and ci2 in unsat_clauses:
                        unsat_clauses.remove(ci2)
            
            flip_count += 1
            
            if not unsat_clauses:
                return True
        
        return len(unsat_clauses) == 0
    
    def spectral_init(n, clause_vars, var_clauses):
        """Use spectral methods to get a good initial assignment."""
        # Build a matrix where entry (i,j) represents the interaction between variables
        # For 3-SAT, we can use a heuristic based on variable frequencies and correlations
        
        # Count variable occurrences
        var_freq = np.zeros(n)
        for cv in clause_vars:
            for var_idx, is_pos in cv:
                var_freq[var_idx] += 1
        
        # Build a simple correlation matrix based on co-occurrence in clauses
        # This is a heuristic; we use it to guide initial assignment
        corr = np.zeros((n, n))
        for cv in clause_vars:
            for i in range(len(cv)):
                for j in range(i+1, len(cv)):
                    v1, p1 = cv[i]
                    v2, p2 = cv[j]
                    # If both are positive or both negative, they tend to agree
                    # If one is positive and one negative, they tend to disagree
                    if p1 == p2:
                        corr[v1, v2] += 1
                        corr[v2, v1] += 1
                    else:
                        corr[v1, v2] -= 1
                        corr[v2, v1] -= 1
        
        # Use power iteration to find the principal eigenvector
        # This gives a direction that maximizes alignment with the correlation structure
        if n > 0:
            v = np.random.randn(n)
            v /= np.linalg.norm(v)
            
            for _ in range(20):  # 20 iterations of power method
                v_new = corr @ v
                norm = np.linalg.norm(v_new)
                if norm < 1e-10:
                    break
                v_new /= norm
                v = v_new
            
            # Convert to boolean assignment
            assignment = v > 0
        else:
            assignment = [False] * n
        
        return assignment
    
    # Multiple restarts with different strategies
    best_assignment = None
    best_unsat = float('inf')
    
    restart = 0
    while time.time() < deadline:
        # Each restart: different initialization strategy
        rng = random.Random(42 + restart)
        
        # Alternate between random init and spectral init
        if restart % 3 == 0:
            # Random initialization
            assignment = [rng.random() < 0.5 for _ in range(n)]
        elif restart % 3 == 1:
            # Spectral initialization
            assignment = spectral_init(n, clause_vars, var_clauses)
            # Add some noise to break symmetry
            for i in range(n):
                if rng.random() < 0.1:
                    assignment[i] = not assignment[i]
        else:
            # Biased random initialization based on variable frequency
            assignment = []
            for i in range(n):
                # Variables with higher frequency are more likely to be true
                bias = 0.5 + 0.2 * (var_clauses[i] and len(var_clauses[i]) / max(1, n/50) or 0)
                bias = min(0.9, max(0.1, bias))
                assignment.append(rng.random() < bias)
        
        # Max flips per restart, increasing slightly
        max_flips = 8000 + restart * 1000
        
        if walk_sat(assignment, rng, max_flips):
            return assignment
        
        # Count unsatisfied clauses
        sat_counts = init_sat_counts(assignment)
        unsat = count_unsat_from_counts(sat_counts)
        
        if unsat < best_unsat:
            best_unsat = unsat
            best_assignment = assignment[:]
        
        restart += 1
    
    if best_assignment is not None:
        return best_assignment
    
    # Fallback: all true
    return [True] * n
# EVOLVE-BLOCK-END
