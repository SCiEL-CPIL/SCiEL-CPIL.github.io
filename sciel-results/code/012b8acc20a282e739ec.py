"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)
    
    if m == 0:
        return [False] * n
    
    # Deadline guard: aim to finish well before 330s
    start_time = time.time()
    deadline = start_time + 320.0
    
    # Precompute literal arrays for vectorized operations
    # Each clause is a list of 3 signed literals (1-based)
    # lit_idx[i][j] = 0-based index of variable in clause i, position j
    # lit_sign[i][j] = +1 if positive literal, -1 if negative
    lit_idx = np.empty((m, 3), dtype=np.int32)
    lit_sign = np.empty((m, 3), dtype=np.int8)
    for i, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            lit_idx[i, j] = abs(lit) - 1
            lit_sign[i, j] = 1 if lit > 0 else -1
    
    # Build variable-to-clause mapping for efficient updates
    # For each variable, store list of (clause_idx, position, sign)
    var_clauses = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = lit_idx[i, j]
            var_clauses[v].append((i, j, lit_sign[i, j]))
    
    # Precompute for each variable: arrays of clause indices, positions, signs
    var_clause_idx = [np.array([c[0] for c in var_clauses[v]], dtype=np.int32) if var_clauses[v] else np.array([], dtype=np.int32) for v in range(n)]
    var_pos = [np.array([c[1] for c in var_clauses[v]], dtype=np.int8) if var_clauses[v] else np.array([], dtype=np.int8) for v in range(n)]
    var_sign = [np.array([c[2] for c in var_clauses[v]], dtype=np.int8) if var_clauses[v] else np.array([], dtype=np.int8) for v in range(n)]
    
    rng = np.random.default_rng(42)
    
    best_assignment = None
    best_unsat_count = m + 1
    
    # Multiple restarts with increasing noise
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Initialize assignment
        assignment = rng.random(n) < 0.5
        
        # Compute initial unsat counts per clause
        # A clause is satisfied if any literal is true
        # For each clause, compute how many literals are satisfied (0, 1, 2, or 3)
        # sat_count[i] = number of satisfied literals in clause i
        # Clause is unsat if sat_count[i] == 0
        
        # Vectorized: for each clause, check each literal
        # assignment[lit_idx[i,j]] == (lit_sign[i,j] > 0)
        # Let's compute: is_true[i,j] = assignment[lit_idx[i,j]] if lit_sign>0 else not assignment[lit_idx[i,j]]
        # Equivalent: is_true = (assignment[lit_idx] == (lit_sign > 0))
        
        is_true = (assignment[lit_idx] == (lit_sign > 0))
        sat_count = is_true.sum(axis=1)  # shape (m,)
        unsat_count = int((sat_count == 0).sum())
        
        if unsat_count == 0:
            return assignment.tolist()
        
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment.copy()
        
        # Adaptive noise: starts high, decreases over time within a restart
        # WalkSAT with noise parameter
        noise = 0.5 + 0.3 * (restart / max_restarts)  # noise between 0.5 and 0.8
        noise = min(noise, 0.8)
        
        # Flip budget per restart: scale with time remaining
        time_per_restart = (deadline - time.time()) / (max_restarts - restart)
        # Estimate flips per second ~ 500k (measured from vectorized approach)
        flip_budget = int(min(time_per_restart * 500000, 5000000))
        flip_budget = max(flip_budget, 10000)
        
        for step in range(flip_budget):
            if time.time() > deadline:
                break
            
            # Check if solved
            if unsat_count == 0:
                return assignment.tolist()
            
            # Track best
            if unsat_count < best_unsat_count:
                best_unsat_count = unsat_count
                best_assignment = assignment.copy()
            
            # Find unsatisfied clauses
            unsat_clauses = np.where(sat_count == 0)[0]
            
            # Select a clause: with some probability pick the one with fewest true literals (steepest descent)
            # but among unsat, all have 0 true literals, so pick randomly or use weighted
            # For unsat clauses, all have 0 satisfied literals, so we pick one at random
            clause_idx = unsat_clauses[rng.integers(len(unsat_clauses))]
            
            # Get the 3 literals in this clause
            c_idx = lit_idx[clause_idx]
            c_sign = lit_sign[clause_idx]
            c_sat = sat_count[clause_idx]  # should be 0
            
            # For each literal, compute what happens if we flip it
            # Flipping variable v changes its contribution to clauses containing v
            # For this clause, all 3 literals are currently false (since clause is unsat)
            # If we flip variable v, the literal becomes true, so this clause becomes satisfied
            # But other clauses containing v may become unsatisfied
            
            # Compute the "score" for each of the 3 variables: number of currently satisfied
            # clauses that would become unsatisfied
            # For variable v, it appears in some clauses. Currently, for each such clause,
            # the literal for v is false (since we're looking at unsat clause, but v might be
            # true in other clauses). Wait, we need to think more carefully.
            
            # Actually, let's use the standard WalkSAT approach:
            # For each variable in the clause, compute how many currently-satisfied clauses
            # would become unsatisfied if we flip it.
            # A clause is currently satisfied if sat_count > 0.
            # If we flip variable v, for each clause c containing v:
            #   - If the literal in c is currently true: it becomes false.
            #     If sat_count[c] was 1, the clause becomes unsatisfied.
            #   - If the literal in c is currently false: it becomes true.
            #     The clause stays satisfied (sat_count increases).
            
            # So for variable v, the number of clauses that would become unsatisfied is:
            #   count of clauses c containing v where the literal is currently true AND sat_count[c] == 1
            
            # Let's compute this for the 3 variables in the clause
            scores = np.zeros(3, dtype=np.float64)
            for j in range(3):
                v = c_idx[j]
                vc_idx = var_clause_idx[v]
                if len(vc_idx) == 0:
                    scores[j] = 0
                    continue
                vp = var_pos[v]
                vs = var_sign[v]
                # Current truth value of the literal for v in each clause it appears in
                # is_true = assignment[v] if vs > 0 else not assignment[v]
                # But this is the same for all clauses for a given v
                v_true = assignment[v] if c_sign[j] > 0 else not assignment[v]
                # Wait, that's not right. The sign in the clause determines the literal.
                # Let me redo: for variable v appearing in clause c at position p with sign s:
                #   literal is true iff assignment[v] == (s > 0)
                # So for all clauses where v appears, the literal truth depends on assignment[v] and the sign in that clause.
                
                # For each clause c' where v appears:
                #   literal_true[c'] = assignment[v] == (var_sign[v][k] > 0) for the k-th occurrence
                # We want to count clauses where literal is true AND sat_count[c'] == 1
                
                # Get the signs for v's occurrences
                v_s = var_sign[v]
                # literal_true for each occurrence: assignment[v] == (v_s > 0)
                # This is a boolean array
                lit_true_arr = (assignment[v] == (v_s > 0))
                # Clauses where literal is true
                true_mask = lit_true_arr
                # Among those, count where sat_count == 1
                if np.any(true_mask):
                    scores[j] = (sat_count[vc_idx[true_mask]] == 1).sum()
                else:
                    scores[j] = 0
            
            # Apply noise: with probability noise, pick randomly; otherwise pick min score
            if rng.random() < noise:
                flip_pos = rng.integers(3)
            else:
                # Pick the variable with minimum score (break ties randomly)
                min_score = scores.min()
                candidates = np.where(scores == min_score)[0]
                flip_pos = candidates[rng.integers(len(candidates))]
            
            v = c_idx[flip_pos]
            
            # Flip the variable
            assignment[v] = not assignment[v]
            
            # Update sat_count for all clauses containing v
            vc_idx = var_clause_idx[v]
            if len(vc_idx) > 0:
                vp = var_pos[v]
                vs = var_sign[v]
                # Old literal truth: assignment[v] was old value, now flipped
                # Old: lit_old = old_assignment[v] == (vs > 0)
                # New: lit_new = not old_assignment[v] == (vs > 0) = not lit_old
                # So each literal flips its truth value
                # For each clause, sat_count changes by +1 if literal was false, -1 if true
                # We need the old truth values
                old_v = not assignment[v]  # since we just flipped
                old_lit = (old_v == (vs > 0))
                # delta = +1 where old_lit is False (literal becomes true), -1 where old_lit is True
                delta = np.where(old_lit, -1, 1)
                sat_count[vc_idx] += delta
                # Clamp to [0, 3]
                sat_count = np.clip(sat_count, 0, 3)
            
            unsat_count = int((sat_count == 0).sum())
        
        # End of restart
        if unsat_count == 0:
            return assignment.tolist()
    
    # Return best assignment found
    if best_assignment is not None:
        return best_assignment.tolist()
    return [False] * n
# EVOLVE-BLOCK-END
