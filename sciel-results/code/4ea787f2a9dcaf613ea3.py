import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Pre-compute variable indices and signs for each literal
    # lit_to_var[literal] = (var_index, is_positive)
    # var_to_clauses[var_index] = list of clause indices
    lit_to_var = {}
    var_to_clauses = [[] for _ in range(n)]
    
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var_idx = abs(lit) - 1
            is_pos = lit > 0
            lit_to_var[lit] = (var_idx, is_pos)
            var_to_clauses[var_idx].append(ci)
            
    # State variables
    # assignment[var_idx] = True/False
    assignment = [random.random() < 0.5 for _ in range(n)]
    
    # clause_unsat[ci] = number of unsatisfied literals in clause ci
    clause_unsat = [0] * len(clauses)
    # var_unsat_count[vi] = number of unsatisfied clauses for variable vi
    var_unsat_count = [0] * n
    
    def is_satisfied_clause(ci):
        return clause_unsat[ci] == 0

    def satisfies_lit(lit, var_idx, is_pos):
        return assignment[var_idx] == is_pos

    # Initialize clause_unsat and var_unsat_count
    for ci, clause in enumerate(clauses):
        for lit in clause:
            var_idx, is_pos = lit_to_var[lit]
            if not satisfies_lit(lit, var_idx, is_pos):
                clause_unsat[ci] += 1
                var_unsat_count[var_idx] += 1

    def flip_var(var_idx):
        assignment[var_idx] = not assignment[var_idx]
        # Update clause_unsat and var_unsat_count for all clauses containing var_idx
        for ci in var_to_clauses[var_idx]:
            clause = clauses[ci]
            # Determine if the literal in this clause was satisfied before and after
            # We need to find the literal in the clause that corresponds to var_idx
            # Since clauses are small (3), we can just scan
            lit_in_clause = None
            for lit in clause:
                if abs(lit) - 1 == var_idx:
                    lit_in_clause = lit
                    break
            
            # It's guaranteed to be there
            is_pos = lit_in_clause > 0
            
            # Before flip:
            # If assignment[var_idx] was is_pos, it was satisfied. Now it is not.
            # If assignment[var_idx] was not is_pos, it was not satisfied. Now it is.
            
            # Wait, assignment[var_idx] has ALREADY been flipped.
            # So we need to know the state BEFORE the flip to calculate the delta.
            # Let's re-structure: calculate delta based on old state.
            
            # Let's undo the logic and do it cleanly.
            # Old state: old_val = not assignment[var_idx] (since we just flipped)
            # New state: new_val = assignment[var_idx]
            
            old_val = not assignment[var_idx]
            new_val = assignment[var_idx]
            
            old_sat = (old_val == is_pos)
            new_sat = (new_val == is_pos)
            
            if old_sat and not new_sat:
                # Clause became less satisfied
                clause_unsat[ci] += 1
                var_unsat_count[var_idx] += 1
            elif not old_sat and new_sat:
                # Clause became more satisfied
                clause_unsat[ci] -= 1
                var_unsat_count[var_idx] -= 1
            # If both sat or both unsat, no change in count?
            # Wait, var_unsat_count counts unsatisfied CLAUSES.
            # If a clause was unsat (clause_unsat > 0) and becomes sat (clause_unsat == 0), var_unsat_count for vars in it should decrease?
            # No, var_unsat_count[vi] is the number of clauses where vi appears AND the clause is unsatisfied.
            
            # Let's re-evaluate var_unsat_count update.
            # var_unsat_count[vi] = sum(1 for ci in var_to_clauses[vi] if clause_unsat[ci] > 0)
            
            # So, if clause_unsat[ci] transitions from >0 to 0, we decrement var_unsat_count for all vars in clause ci?
            # That would be expensive.
            # Alternatively, var_unsat_count can be maintained incrementally.
            
            # Let's stick to a simpler update for var_unsat_count if needed, or just use clause_unsat for the heuristic.
            # The primary heuristic is picking a clause and then picking a var.
            # We don't strictly need var_unsat_count if we are picking a clause first.
            # However, for "good" flip selection, we just need to know if flipping a var in the chosen clause satisfies that clause.
            # Since the clause is unsat (clause_unsat[ci] > 0), flipping a var in it *might* satisfy it.
            # Specifically, if we flip a var `v` in clause `c` to match the literal `l` in `c`, `c` becomes satisfied IF `c` had exactly 1 unsatisfied literal (which was `l`) OR if `c` had more, it reduces the count.
            # WalkSAT standard: Pick an unsat clause. With prob p, pick a literal in it that is currently false (flipping it makes it true). With prob 1-p, pick a random literal in it.
            
            # So we don't need var_unsat_count for the basic WalkSAT heuristic.
            # We only need clause_unsat to identify unsat clauses and to check if a flip satisfies a clause.
            
            # Let's simplify: Only maintain clause_unsat.
            pass

    # Re-implementing flip_var with correct logic
    def flip_var(var_idx):
        assignment[var_idx] = not assignment[var_idx]
        for ci in var_to_clauses[var_idx]:
            clause = clauses[ci]
            # Find the literal in this clause corresponding to var_idx
            # Since clause size is 3, linear scan is fine
            for lit in clause:
                if abs(lit) - 1 == var_idx:
                    is_pos = lit > 0
                    # Check if this literal was satisfied before the flip
                    # Before flip, assignment[var_idx] was the opposite of current
                    old_val = not assignment[var_idx]
                    old_sat = (old_val == is_pos)
                    
                    if old_sat:
                        # It was satisfied, now it is not. Clause unsat count increases.
                        clause_unsat[ci] += 1
                    else:
                        # It was not satisfied, now it is. Clause unsat count decreases.
                        clause_unsat[ci] -= 1
                    break

    # Time management
    start_time = time.time()
    deadline = start_time + 290  # 300s limit, keep 10s margin
    
    # Restart strategy
    max_steps_per_restart = 5000
    current_steps = 0
    p = 0.5  # Probability of picking a "good" flip
    
    while time.time() < deadline:
        # Check if solved
        if all(cu == 0 for cu in clause_unsat):
            return assignment
            
        # If we've done too many steps in this restart, restart
        if current_steps > max_steps_per_restart:
            # Reset assignment
            assignment = [random.random() < 0.5 for _ in range(n)]
            # Recalculate clause_unsat
            clause_unsat = [0] * len(clauses)
            for ci, clause in enumerate(clauses):
                for lit in clause:
                    var_idx, is_pos = lit_to_var[lit]
                    if not (assignment[var_idx] == is_pos):
                        clause_unsat[ci] += 1
            current_steps = 0
            p = 0.5
            continue

        # Find an unsatisfied clause
        # To avoid O(N) scan every time, we can maintain a list of unsat clauses
        # But for 150 vars, N_clauses is ~225 (planted 3-SAT ratio ~1.5).
        # Scanning 225 clauses is very fast.
        unsat_clauses = [ci for ci, cu in enumerate(clause_unsat) if cu > 0]
        
        if not unsat_clauses:
            return assignment

        ci = random.choice(unsat_clauses)
        clause = clauses[ci]
        
        # Identify literals that would satisfy the clause if flipped
        # A literal is "false" if its variable's assignment does not match its sign.
        # Flipping a false literal makes it true.
        false_lits = []
        for lit in clause:
            var_idx, is_pos = lit_to_var[lit]
            if not (assignment[var_idx] == is_pos):
                false_lits.append(lit)
                
        if false_lits:
            if random.random() < p:
                # Pick a good flip
                lit = random.choice(false_lits)
            else:
                # Pick a random flip from the clause
                lit = random.choice(clause)
        else:
            # This shouldn't happen if clause is unsat, but just in case
            lit = random.choice(clause)
            
        var_idx = abs(lit) - 1
        flip_var(var_idx)
        
        current_steps += 1
        
        # Adapt p
        # If we made a good flip (satisfied the clause), maybe increase p?
        # If we made a bad flip, decrease p?
        # Standard WalkSAT often uses fixed p or p that decays.
        # Let's try: if the chosen clause is now satisfied, increase p slightly.
        # If not, decrease p slightly.
        if clause_unsat[ci] == 0:
            p = min(0.9, p + 0.05)
        else:
            p = max(0.1, p - 0.05)

    return assignment
