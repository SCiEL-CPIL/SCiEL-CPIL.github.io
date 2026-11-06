"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(42)
    
    # Preprocess: convert clauses to 0-based indices with sign info
    # For each variable, track which clauses contain it (positive and negative)
    var_clauses_pos = [[] for _ in range(n)]
    var_clauses_neg = [[] for _ in range(n)]
    
    # clause_vars: for each clause, list of (var_index, is_positive)
    clause_vars = []
    for clause in clauses:
        cv = []
        for lit in clause:
            var_idx = abs(lit) - 1
            is_pos = lit > 0
            cv.append((var_idx, is_pos))
            if is_pos:
                var_clauses_pos[var_idx].append(len(clause_vars))
            else:
                var_clauses_neg[var_idx].append(len(clause_vars))
        clause_vars.append(cv)
    
    m = len(clauses)
    
    # Initialize random assignment
    assignment = [rng.random() < 0.5 for _ in range(n)]
    
    # Track satisfaction count for each clause (0, 1, or 2 true literals)
    # sat_count[c] = number of true literals in clause c
    sat_count = np.zeros(m, dtype=np.int32)
    
    # Compute initial satisfaction counts
    for c, cv in enumerate(clause_vars):
        count = 0
        for var_idx, is_pos in cv:
            if assignment[var_idx] == is_pos:
                count += 1
        sat_count[c] = count
    
    # unsat_clauses: set of clause indices that are unsatisfied
    unsat_clauses = set(c for c in range(m) if sat_count[c] == 0)
    
    # Precompute for each variable: list of clauses it appears in
    # For efficiency, we'll compute delta effects on the fly
    
    # Frequency of variables in unsat clauses (for bias)
    # We'll update this incrementally
    
    # WalkSAT with smart flip selection
    max_flips = 50000
    step = 0
    
    while unsat_clauses and step < max_flips:
        step += 1
        
        # Pick a random unsatisfied clause
        clause_idx = rng.choice(list(unsat_clauses))
        cv = clause_vars[clause_idx]
        
        # For each literal in this clause, compute the delta if we flip that variable
        # delta = change in number of satisfied clauses
        # If we flip variable v:
        #   - For clauses where v appears positively: if v was True, those clauses lose a true literal; if v was False, they gain one
        #   - Similarly for negative appearances
        
        var_idx = cv[0][0]  # We'll evaluate all variables in the clause
        
        best_var = None
        best_delta = -10
        best_random = rng.random()
        
        # Evaluate each variable in the unsat clause
        candidates = []
        for var_idx, is_pos in cv:
            # Compute delta of flipping var_idx
            delta = 0
            old_val = assignment[var_idx]
            
            # Positive appearances: if old_val is True, flipping makes it False -> clauses lose a true literal
            # If old_val is False, flipping makes it True -> clauses gain a true literal
            for c in var_clauses_pos[var_idx]:
                if old_val:
                    # Was contributing True, now will be False
                    if sat_count[c] == 1:
                        delta -= 1  # clause becomes unsat
                    elif sat_count[c] >= 2:
                        delta += 0  # still satisfied (at least 1 other true)
                    # if sat_count[c] == 0, can't happen since v was True and in clause
                else:
                    # Was contributing False, now will be True
                    if sat_count[c] == 0:
                        delta += 1  # clause becomes sat
                    # else still unsat or already sat
                    
            for c in var_clauses_neg[var_idx]:
                if old_val:
                    # v is True, neg literal is False. Flipping v to False makes neg literal True
                    if sat_count[c] == 0:
                        delta += 1
                else:
                    # v is False, neg literal is True. Flipping v to True makes neg literal False
                    if sat_count[c] == 1:
                        delta -= 1
            
            candidates.append((var_idx, delta))
        
        # Choose variable: prefer positive delta, with some randomness
        # Use a weighted selection based on delta
        deltas = [d for _, d in candidates]
        max_d = max(deltas)
        min_d = min(deltas)
        
        # Add small noise to break ties and explore
        scores = [d + rng.random() * 0.5 for d in deltas]
        best_i = max(range(len(candidates)), key=lambda i: scores[i])
        best_var = candidates[best_i][0]
        best_delta = candidates[best_i][1]
        
        # Flip the variable
        var_idx = best_var
        old_val = assignment[var_idx]
        assignment[var_idx] = not old_val
        
        # Update sat_counts
        for c in var_clauses_pos[var_idx]:
            if old_val:
                sat_count[c] -= 1
            else:
                sat_count[c] += 1
            
            # Update unsat set
            if sat_count[c] == 0:
                unsat_clauses.add(c)
            elif sat_count[c] == 1 and c in unsat_clauses:
                unsat_clauses.discard(c)
        
        for c in var_clauses_neg[var_idx]:
            if not old_val:
                sat_count[c] -= 1
            else:
                sat_count[c] += 1
            
            if sat_count[c] == 0:
                unsat_clauses.add(c)
            elif sat_count[c] == 1 and c in unsat_clauses:
                unsat_clauses.discard(c)
    
    return assignment
# EVOLVE-BLOCK-END
