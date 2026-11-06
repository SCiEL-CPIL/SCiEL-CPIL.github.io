"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import heapq
import math


def solve(instance):
    # Model-and-optimize style SAT solver:
    # 1) Deterministic preprocessing (unit propagation, pure literal elimination)
    # 2) Bitmask-accelerated Greedy Local Search with random escape
    # 3) Time-bounded multi-restart
    
    n, clauses = instance["n"], instance["clauses"]
    
    deadline = time.time() + 290  # safety margin before 330s kill
    
    # Build clause list as list of tuples of literals (signed 1-based)
    cnf = [tuple(c) for c in clauses]
    m = len(cnf)
    
    if m == 0:
        return [False] * n
        
    # Preprocessing: Unit Propagation and Pure Literal Elimination
    def preprocess(cnf_list):
        """Apply unit propagation and pure literal elimination iteratively."""
        assignment = [None] * n  # None = unassigned
        cnf_work = [list(c) for c in cnf_list]
        changed = True
        while changed:
            changed = False
            # Remove satisfied clauses
            new_cnf = []
            for c in cnf_work:
                sat = False
                for lit in c:
                    vi = abs(lit) - 1
                    if assignment[vi] is not None and assignment[vi] == (lit > 0):
                        sat = True
                        break
                if not sat:
                    new_cnf.append(c)
            cnf_work = new_cnf
            
            # Unit propagation
            for i, c in enumerate(cnf_work):
                if len(c) == 1:
                    lit = c[0]
                    vi = abs(lit) - 1
                    if assignment[vi] is None:
                        assignment[vi] = (lit > 0)
                        changed = True
                    elif assignment[vi] != (lit > 0):
                        return None, None, assignment  # contradiction
                    else:
                        cnf_work[i] = []  # satisfied
                        changed = True
                elif len(c) == 0:
                    return None, None, assignment  # contradiction
            
            # Pure literal elimination
            lit_counts = [[0, 0] for _ in range(n)]  # [pos_count, neg_count]
            for c in cnf_work:
                seen = set()
                for lit in c:
                    vi = abs(lit) - 1
                    if vi in seen:
                        continue
                    seen.add(vi)
                    if lit > 0:
                        lit_counts[vi][0] += 1
                    else:
                        lit_counts[vi][1] += 1
            
            for vi in range(n):
                if assignment[vi] is not None:
                    continue
                if lit_counts[vi][0] == 0 and lit_counts[vi][1] > 0:
                    assignment[vi] = False  # pure negative
                    changed = True
                elif lit_counts[vi][1] == 0 and lit_counts[vi][0] > 0:
                    assignment[vi] = True  # pure positive
                    changed = True
        
        # Check if all assigned
        if all(a is not None for a in assignment):
            final_assign = [bool(a) for a in assignment]
            # Verify satisfaction
            sat_all = True
            for c in cnf_list:
                sat = False
                for lit in c:
                    vi = abs(lit) - 1
                    if final_assign[vi] == (lit > 0):
                        sat = True
                        break
                if not sat:
                    sat_all = False
                    break
            if sat_all:
                return final_assign, None, final_assign
            return None, None, assignment
        
        return None, cnf_work, assignment
    
    solved, reduced_cnf, partial_assign = preprocess(cnf)
    if solved is not None:
        return solved
    
    # If preprocessing reduced the problem, use the reduced CNF for search
    # Note: The assignment from preprocessing is partial. We will fix those variables.
    # We need to map the reduced problem back to the original variable indices.
    # Actually, the preprocessing function above does not remove variables, it just assigns them.
    # The `cnf_work` contains clauses with literals that might refer to assigned variables.
    # However, the standard approach is to substitute assigned values and remove satisfied clauses.
    # The current preprocess does this implicitly by filtering `cnf_work`.
    # But the indices in `cnf_work` literals are still original 1-based.
    
    if reduced_cnf is not None:
        # Filter out clauses that became empty (satisfied) or are just literals?
        # The preprocess loop removes satisfied clauses.
        # Unit clauses are handled by assigning variables.
        # So `reduced_cnf` should contain only clauses of length >= 2?
        # Wait, if a clause becomes unit, we assign it and mark it satisfied (len 0) or remove it?
        # In the loop: `cnf_work[i] = []` for satisfied unit clauses.
        # Then next iteration removes satisfied clauses (which are empty or have a true literal).
        # So `reduced_cnf` contains only clauses that are NOT satisfied and NOT unit (unless contradiction).
        # Actually, if a clause is unit and we assign it, it becomes satisfied.
        # So `reduced_cnf` will contain clauses of size >= 2.
        
        # We will solve the reduced problem.
        # The variables in `reduced_cnf` are a subset of the original variables.
        # We need to be careful: `partial_assign` has values for some variables.
        # The remaining variables are those with `partial_assign[i] is None`.
        
        # Let's rebuild the search space for the unassigned variables.
        # This is complex because the clause indices change.
        # Alternative: Just use the original CNF and treat assigned variables as fixed.
        # This is simpler and less error-prone. The bitmask approach works on the full set of clauses.
        # We just constrain the flips to only unassigned variables.
        
        pass
    
    # Strategy: Use the original CNF.
    # Fix variables where partial_assign is not None.
    # Only flip variables where partial_assign is None.
    
    fixed_assign = [None] * n
    active_vars = []
    for i in range(n):
        if partial_assign[i] is not None:
            fixed_assign[i] = bool(partial_assign[i])
        else:
            fixed_assign[i] = False # Initial guess, will be randomized
            active_vars.append(i)
            
    # If all variables are fixed, we already checked in preprocess
    if not active_vars:
        return [bool(a) for a in fixed_assign]

    # Initialize active variables randomly
    def make_full_assignment(partial):
        rng = random.Random(42)
        assign = [False] * n
        for i in range(n):
            if partial[i] is not None:
                assign[i] = bool(partial[i])
            else:
                assign[i] = rng.random() < 0.5
        return assign

    start_assign = make_full_assignment(partial_assign)
    
    # Check if already satisfied
    def is_satisfied(assignment):
        for c in cnf:
            sat = False
            for lit in c:
                val = assignment[abs(lit) - 1]
                if val == (lit > 0):
                    sat = True
                    break
            if not sat:
                return False
        return True

    if is_satisfied(start_assign):
        return start_assign

    # Bitmask Setup
    # mask_pos[i]: bitmask of clauses satisfied if x_i = True
    # mask_neg[i]: bitmask of clauses satisfied if x_i = False
    mask_pos = [0] * n
    mask_neg = [0] * n
    
    for ci, c in enumerate(cnf):
        bit = 1 << ci
        for lit in c:
            vi = abs(lit) - 1
            if lit > 0:
                mask_pos[vi] |= bit
            else:
                mask_neg[vi] |= bit
                
    # Function to get current satisfied mask
    def get_sat_mask(assign):
        mask = 0
        for i in range(n):
            if assign[i]:
                mask |= mask_pos[i]
            else:
                mask |= mask_neg[i]
        return mask

    # Initial state
    cur_assign = start_assign[:]
    cur_sat_mask = get_sat_mask(cur_assign)
    cur_unsat = m - cur_sat_mask.bit_count()
    
    best_assign = cur_assign[:]
    best_unsat = cur_unsat
    
    # Local Search Loop
    def local_search(start_assign, seed, p_greedy, max_steps):
        """
        Greedy local search with random escape.
        p_greedy: probability of choosing the best move. If 1.0, pure greedy.
                 If lower, chooses randomly among top moves or random moves.
        """
        if time.time() > deadline:
            return False, start_assign, m - get_sat_mask(start_assign).bit_count()
            
        rng = random.Random(seed)
        cur = start_assign[:]
        cur_sat_mask = get_sat_mask(cur)
        cur_unsat = m - cur_sat_mask.bit_count()
        min_unsat = cur_unsat
        min_assign = cur[:]
        
        # Precompute active variables list for faster iteration if needed?
        # n is small (50), so iterating all is fine.
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if cur_unsat == 0:
                return True, cur, 0
            
            # Calculate gain for each active variable
            # gain[i] = (clauses newly satisfied) - (clauses newly unsatisfied)
            # If we flip i from val to not val:
            # new_mask = mask_pos[i] if not val else mask_neg[i]
            # old_mask = mask_neg[i] if not val else mask_pos[i]
            
            best_gain = -100
            best_var = -1
            gains = [0] * n # Reuse list to avoid allocation
            
            # Only consider active variables (unfixed)
            for i in active_vars:
                val = cur[i]
                if val:
                    new_mask = mask_neg[i]
                    old_mask = mask_pos[i]
                else:
                    new_mask = mask_pos[i]
                    old_mask = mask_neg[i]
                
                # Clauses satisfied by new value that are currently unsatisfied
                newly_sat = new_mask & ~cur_sat_mask
                # Clauses satisfied by old value that are currently satisfied (will become unsat)
                newly_unsat = old_mask & cur_sat_mask
                
                gain = newly_sat.bit_count() - newly_unsat.bit_count()
                gains[i] = gain
                
                if gain > best_gain:
                    best_gain = gain
                    best_var = i
            
            # Decision: Greedy or Random
            if rng.random() < p_greedy:
                # Pick best variable
                if best_var == -1:
                    # No active vars? Should not happen
                    break
                vi = best_var
            else:
                # Random walk: Pick a variable involved in an unsatisfied clause?
                # Or just any active variable?
                # Let's pick a random active variable.
                vi = rng.choice(active_vars)
                
            # Flip variable vi
            cur[vi] = not cur[vi]
            
            # Update sat_mask
            if cur[vi]:
                # Changed to True
                # Remove clauses satisfied by False (old)
                # Add clauses satisfied by True (new)
                # But we need to be careful:
                # cur_sat_mask currently includes mask_neg[vi] (since it was False)
                # We want to replace that contribution with mask_pos[vi]
                
                # Clauses that ONLY vi satisfied with False?
                # This is tricky with simple OR.
                # Easier: Recalculate? No, n=50, m~200.
                # Actually, updating bitmask incrementally is hard because of overlaps.
                # If clause C is in mask_neg[vi] and mask_pos[vj] (j != vi), and we flip vi,
                # C remains satisfied if vj is True.
                
                # Let's just recalculate the sat mask for the affected clauses?
                # Or simply:
                # old_contribution = mask_neg[vi]
                # new_contribution = mask_pos[vi]
                # The clauses that change status are those in (old_contribution XOR new_contribution)?
                # No.
                
                # Let's stick to the robust method:
                # Since n=50, recalculating the entire mask is O(n) bitwise ops.
                # 50 ops per step is very fast.
                cur_sat_mask = get_sat_mask(cur)
                cur_unsat = m - cur_sat_mask.bit_count()
                
            if cur_unsat < min_unsat:
                min_unsat = cur_unsat
                min_assign = cur[:]
                
        return min_unsat == 0, (cur if min_unsat == 0 else min_assign), min_unsat

    # Multi-restart strategy
    attempt = 0
    seeds = [42, 123, 7, 999, 5555, 12345, 67890, 24680, 13579, 98765]
    # p_greedy values: Start greedy, then more random to escape local minima
    p_greedy_values = [1.0, 0.9, 0.8, 0.7, 0.5, 0.9, 0.8, 0.6, 0.4, 0.7]
    base_steps = 50000 # Bitmask ops are fast, so we can do more steps
    
    while time.time() < deadline - 5:
        seed = seeds[attempt % len(seeds)]
        p_greedy = p_greedy_values[attempt % len(p_greedy_values)]
        
        # Use best_assign as start if it's good, otherwise random
        if attempt == 0:
            start = start_assign
        elif attempt % 3 == 0:
            # Fresh random start
            start = make_full_assignment(partial_assign)
            # Ensure we don't start with the exact same one
        else:
            start = best_assign[:]
            
        success, result, min_unsat = local_search(start, seed, p_greedy, base_steps)
        
        if success:
            return result
        
        if min_unsat < best_unsat:
            best_unsat = min_unsat
            best_assign = result[:]
            
        attempt += 1
        
    # Final check
    if is_satisfied(best_assign):
        return best_assign
    
    return best_assign

# EVOLVE-BLOCK-END
