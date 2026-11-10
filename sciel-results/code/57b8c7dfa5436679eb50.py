import time
import random
import numpy as np

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Pre-process clauses into 0-based indices and signs
    # literal is 1-based signed. 
    # var = abs(l) - 1
    # sign = 1 if l > 0 else 0 (True for positive, False for negative)
    # Literal l is satisfied if assignment[var] == (l > 0)
    
    pre_clauses = []
    for c in clauses:
        pc = []
        for l in c:
            var = abs(l) - 1
            is_pos = 1 if l > 0 else 0
            pc.append((var, is_pos))
        pre_clauses.append(pc)
        
    m = len(pre_clauses)
    
    # Build incidence lists for fast updates
    # var_incidence[var] = list of (clause_idx, is_pos)
    var_incidence = [[] for _ in range(n)]
    for ci, c in enumerate(pre_clauses):
        for var, is_pos in c:
            var_incidence[var].append((ci, is_pos))
    
    def count_unsat(assignment):
        count = 0
        for c in pre_clauses:
            sat = False
            for var, is_pos in c:
                if assignment[var] == (is_pos == 1):
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    def compute_clause_satisfaction(assignment):
        # Returns an array of length m, where clause_sat[ci] is the number of true literals in clause ci
        clause_sat = np.zeros(m, dtype=int)
        for ci, c in enumerate(pre_clauses):
            cnt = 0
            for var, is_pos in c:
                if assignment[var] == (is_pos == 1):
                    cnt += 1
            clause_sat[ci] = cnt
        return clause_sat

    def walksat(assignment, clause_sat, max_flips, deadline, p_walk=0.5):
        """
        Vectorized-ish WalkSAT with incremental updates.
        assignment: np.array of bools
        clause_sat: np.array of int, number of satisfied literals per clause
        """
        current_unsat = np.sum(clause_sat == 0)
        
        # Pre-compute unsatisfied clause indices for quick access
        unsat_indices = np.where(clause_sat == 0)[0]
        
        for step in range(max_flips):
            if time.time() > deadline:
                break
            
            if current_unsat == 0:
                return assignment, 0, clause_sat
            
            if len(unsat_indices) == 0:
                return assignment, 0, clause_sat

            # 1. Select an unsatisfied clause
            # Randomly pick from unsatisfied ones
            c_idx = unsat_indices[random.randint(0, len(unsat_indices) - 1)]
            c = pre_clauses[c_idx]
            
            # 2. Decide whether to walk or pick best
            if random.random() < p_walk:
                # Random flip from the clause
                var, is_pos = random.choice(c)
                flip_var = var
                flip_is_pos = is_pos
            else:
                # Greedy: pick literal that maximizes gain
                # Gain = (new_unsat - old_unsat)
                # Flipping var to satisfy literal (var, is_pos)
                # 1. This clause becomes satisfied (if it was the only one? No, it was 0, becomes 1). So -1 unsat for this clause.
                # 2. Other clauses containing (var, is_pos) with 0 sat become 1 sat. -1 unsat each.
                # 3. Other clauses containing (var, 1-is_pos) with 1 sat become 0 sat. +1 unsat each.
                
                best_gain = -1000
                best_var = -1
                best_is_pos = -1
                best_candidates = []
                
                for var, is_pos in c:
                    gain = 0
                    # This clause (c_idx) was 0, becomes 1. Gain +1 (reduction in unsat)
                    gain += 1
                    
                    # Check other clauses with (var, is_pos)
                    # We need to know how many are currently unsat (sat==0)
                    # And how many clauses with (var, 1-is_pos) are currently sat==1
                    
                    # Iterate through incidence of var
                    for ci2, is_pos2 in var_incidence[var]:
                        if ci2 == c_idx:
                            continue
                        if is_pos2 == is_pos:
                            # If we set var to is_pos, this literal becomes true.
                            # If clause was unsat (0), it becomes sat (1). Gain +1.
                            if clause_sat[ci2] == 0:
                                gain += 1
                        else:
                            # is_pos2 != is_pos. This literal becomes false.
                            # If clause was sat (1) and this was the ONLY true literal, it becomes unsat (0). Loss -1.
                            if clause_sat[ci2] == 1:
                                gain -= 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_var = var
                        best_is_pos = is_pos
                        best_candidates = [(var, is_pos)]
                    elif gain == best_gain:
                        best_candidates.append((var, is_pos))
                
                if best_candidates:
                    flip_var, flip_is_pos = random.choice(best_candidates)
                else:
                    # Fallback to random if something went wrong
                    flip_var, flip_is_pos = random.choice(c)
            
            # 3. Perform the flip
            old_val = assignment[flip_var]
            new_val = not old_val
            assignment[flip_var] = new_val
            
            # 4. Update clause_sat and current_unsat
            # Only clauses containing flip_var are affected
            for ci2, is_pos2 in var_incidence[flip_var]:
                # Check if this literal is satisfied by the new assignment
                was_sat = old_val == (is_pos2 == 1)
                is_sat = new_val == (is_pos2 == 1)
                
                if was_sat and not is_sat:
                    # Lost satisfaction
                    clause_sat[ci2] -= 1
                    if clause_sat[ci2] == 0:
                        current_unsat += 1
                        # Add to unsat list? We just recompute unsat_indices occasionally or keep a set?
                        # For simplicity, we can recompute unsat_indices every K steps or just use a list and clean it.
                        # Actually, maintaining a dynamic list is tricky.
                        # Let's just recompute unsat_indices every 10 steps or if it grows too large.
                        # Or simpler: just check if current_unsat is 0.
                        # For selection, we need the list.
                        # Let's use a set for unsat_indices? No, numpy is faster for selection if we have the array.
                        # Let's just recompute unsat_indices every 50 steps.
                        pass
                elif not was_sat and is_sat:
                    # Gained satisfaction
                    clause_sat[ci2] += 1
                    if clause_sat[ci2] == 1:
                        current_unsat -= 1
                        pass
                # else: no change in satisfaction status (0->0 or 1->1)
                
            # Periodically refresh unsat_indices
            if step % 50 == 0 or current_unsat == 0:
                unsat_indices = np.where(clause_sat == 0)[0]
                
        return assignment, current_unsat, clause_sat

    def spectral_init():
        # Build graph where edges connect variables in same clause
        adj = np.zeros((n, n))
        for c in pre_clauses:
            for i in range(len(c)):
                for j in range(i+1, len(c)):
                    v1, s1 = c[i]
                    v2, s2 = c[j]
                    adj[v1, v2] += 1
                    adj[v2, v1] += 1
        
        # Compute Laplacian
        degree = adj.sum(axis=1)
        L = np.diag(degree) - adj
        
        # Find Fiedler vector (2nd smallest eigenvalue)
        try:
            from scipy.sparse.linalg import eigsh
            vals, vecs = eigsh(L, k=3, sigma=0, which='LM')
            idx = np.argsort(vals)
            fiedler_vec = vecs[:, idx[1]]
            assignment = fiedler_vec > 0
            return assignment
        except Exception:
            return np.random.rand(n) < 0.5

    # Main Loop
    deadline = time.time() + 280
    best_assignment = None
    best_unsat = float('inf')
    
    # Try spectral init first
    assignment = spectral_init()
    clause_sat = compute_clause_satisfaction(assignment)
    assignment, unsat, clause_sat = walksat(assignment, clause_sat, 10000, deadline)
    if unsat == 0:
        return assignment.tolist()
    if unsat < best_unsat:
        best_unsat = unsat
        best_assignment = assignment.copy()
        
    # Try random inits
    for i in range(100):
        if time.time() > deadline:
            break
        assignment = np.random.rand(n) < 0.5
        clause_sat = compute_clause_satisfaction(assignment)
        assignment, unsat, clause_sat = walksat(assignment, clause_sat, 10000, deadline)
        if unsat == 0:
            return assignment.tolist()
        if unsat < best_unsat:
            best_unsat = unsat
            best_assignment = assignment.copy()
            
    if best_assignment is not None:
        return best_assignment.tolist()
    return [False] * n
