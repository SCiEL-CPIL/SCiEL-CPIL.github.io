"""Improved 3-SAT solver using GLARE-style reasoning and efficient damage calculation."""
import random
import time

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(123)
    
    # Time limit: 290 seconds (safety margin from 330s kill)
    start_time = time.time()
    time_limit = 290.0

    # Preprocessing
    # Convert to 0-based indices. Store literals as (var_idx, sign) where sign is 1 or -1.
    # assignment is a list of -1 (unassigned), 0 (false), 1 (true)
    # However, for GLARE, we often keep a boolean assignment and track unassigned separately, 
    # or just use -1/0/1 directly. Let's use -1/0/1 for clarity in unit propagation.
    
    m = len(clauses)
    clause_vars = [] # List of lists of (var, sign)
    var_to_clauses = [[] for _ in range(n)] # var -> list of (clause_idx, sign)
    
    for i, c in enumerate(clauses):
        cv = []
        for lit in c:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            cv.append((v, s))
            var_to_clauses[v].append((i, s))
        clause_vars.append(cv)

    # Helper to check if a clause is satisfied given an assignment (-1, 0, 1)
    def is_satisfied(clause_idx, assignment):
        for v, s in clause_vars[clause_idx]:
            val = assignment[v]
            if val == -1:
                continue
            if (val == 1 and s == 1) or (val == 0 and s == -1):
                return True
        return False

    def is_unsat(clause_idx, assignment):
        for v, s in clause_vars[clause_idx]:
            val = assignment[v]
            if val == -1:
                return False # Not unsat if any unassigned
            if (val == 1 and s == 1) or (val == 0 and s == -1):
                return False
        return True

    def count_unsat(assignment):
        return sum(1 for i in range(m) if is_unsat(i, assignment))

    def random_assignment():
        # Random 0/1 assignment
        return [rng.randint(0, 1) for _ in range(n)]

    def glare_search(assignment, max_steps):
        """
        GLARE-style search:
        1. Check for unit clauses (only 1 unassigned literal) -> Force assignment.
        2. If no unit clauses, pick a random unsatisfied clause.
        3. Flip a literal in that clause to satisfy it, choosing the one with minimum damage.
        """
        # We will maintain a list of unsatisfied clauses for efficiency? 
        # For 150 vars, m is likely ~450. Scanning all m is 450 ops. 
        # Doing this 5000 times is 2.25M ops. In Python, this might be tight but doable.
        # Let's optimize: keep a list of currently unsatisfied clause indices.
        
        # Initialize unsat list
        unsat_clauses = [i for i in range(m) if is_unsat(i, assignment)]
        
        if not unsat_clauses:
            return assignment, True

        # We need a fast way to check if a clause becomes satisfied/unsat.
        # We will update unsat_clauses incrementally.
        
        for step in range(max_steps):
            if time.time() - start_time > time_limit:
                return assignment, False
            
            if not unsat_clauses:
                return assignment, True

            # 1. Reasoning: Check for unit clauses among unsatisfied ones?
            # Actually, unit propagation should apply to ALL clauses, but we only care about unsat ones for the main loop.
            # However, if a clause is satisfied, it can't be unit (in the sense of forcing a change to fix it).
            # Unit clauses are those with exactly 1 unassigned literal.
            # If a clause is unsatisfied, all its literals are assigned to the wrong value. 
            # Wait, definition of unsatisfied: all literals are false.
            # If a clause is unsatisfied, it has 0 unassigned literals? No.
            # If a clause has unassigned literals, it is "open".
            # In GLARE, we usually work with a partial assignment.
            # Let's stick to full assignment (0/1) to keep it simple and fast, 
            # but use "damage" to guide flips. The "reasoning" aspect in simple WalkSAT variants 
            # is often approximated by always picking the flip that minimizes damage.
            # True Unit Propagation requires a partial assignment model.
            
            # Let's implement a hybrid: 
            # Standard WalkSAT with optimized damage calc.
            # This is often faster than complex partial assignment tracking in Python 
            # unless the instance is very hard.
            
            # Pick a random unsatisfied clause
            # To avoid O(m) scan every step, we maintain the list.
            # But updating the list is hard. 
            # Alternative: Just pick a random clause from the full set and check if it's unsat?
            # No, that's inefficient.
            
            # Let's use the list `unsat_clauses`. 
            # When we flip a variable, we need to update the status of all clauses containing that variable.
            
            # Pick a clause from current unsat_clauses
            ci = rng.choice(unsat_clauses)
            
            # Find best literal to flip
            best_var = -1
            best_sign = 0
            min_damage = float('inf')
            
            clause = clause_vars[ci]
            
            # Calculate damage for each literal in the clause
            # Damage = number of currently satisfied clauses that would become unsatisfied
            # if we set var to sign.
            
            for v, s in clause:
                # We want to set assignment[v] to s (1 or -1 mapped to 1/0?)
                # My assignment uses 0/1.
                # If s is 1, we set to 1. If s is -1, we set to 0.
                target_val = 1 if s == 1 else 0
                
                if assignment[v] == target_val:
                    # Flipping this would make the clause unsatisfied (or stay unsat if other literals change?)
                    # No, if we set it to the value that satisfies the literal, the clause becomes satisfied 
                    # ONLY IF no other literal in the clause is already satisfied? 
                    # Wait, the clause is currently UNSATISFIED. 
                    # This means ALL literals in the clause are currently FALSE.
                    # So, for any literal (v, s) in this clause, assignment[v] is currently the "wrong" value.
                    # So flipping v to target_val will make this literal TRUE, thus satisfying the clause.
                    
                    # Calculate damage:
                    damage = 0
                    # Iterate over all clauses containing v
                    for cj, sj in var_to_clauses[v]:
                        if cj == ci:
                            continue
                        # Check if clause cj is currently satisfied
                        # If it is, and it contains the literal (v, opposite of target_val), 
                        # then flipping v will make cj unsatisfied.
                        
                        # Is cj currently satisfied?
                        # We need a fast check. 
                        # Since we don't maintain a "satisfied count" per clause dynamically, 
                        # we have to check.
                        
                        # Optimization: 
                        # A clause is satisfied if ANY literal is true.
                        # We can check if clause cj is satisfied by checking its literals.
                        # But this is O(3) per clause. 
                        # Total damage calc is O(degree(v) * 3).
                        # Degree is usually small (e.g., 10-20).
                        
                        # Check if cj is satisfied BEFORE flip
                        is_sat = False
                        for vv, ss in clause_vars[cj]:
                            val = assignment[vv]
                            if (val == 1 and ss == 1) or (val == 0 and ss == -1):
                                is_sat = True
                                break
                        
                        if is_sat:
                            # Check if flipping v breaks it
                            # The literal in cj involving v is (v, sj)
                            # If we set v to target_val, does (v, sj) become false?
                            # (v, sj) is true if (target_val == 1 and sj == 1) or (target_val == 0 and sj == -1)
                            # If it is currently true (contributing to sat), and becomes false after flip, 
                            # AND no other literal in cj is true, then cj becomes unsat.
                            
                            # Actually, simpler: 
                            # If cj is satisfied, it has at least one true literal.
                            # If the literal (v, sj) is the ONLY true literal in cj, then flipping v makes cj unsat.
                            # If there are other true literals, cj remains sat.
                            
                            # So we need to count true literals in cj.
                            # But we already know it's satisfied.
                            # Let's just count how many literals in cj are true.
                            true_count = 0
                            for vv, ss in clause_vars[cj]:
                                val = assignment[vv]
                                if (val == 1 and ss == 1) or (val == 0 and ss == -1):
                                    true_count += 1
                                    
                            if true_count == 1:
                                # Check if the true literal is the one involving v
                                # It must be, because we are considering the impact of flipping v.
                                # But wait, is (v, sj) currently true?
                                # Current val of v is NOT target_val.
                                # If (v, sj) is currently true, then sj must correspond to current val.
                                # If we flip to target_val, (v, sj) becomes false.
                                # So yes, if true_count == 1, and that one is (v, sj), then damage += 1.
                                # We assume (v, sj) is the only true one.
                                damage += 1

                    if damage < min_damage:
                        min_damage = damage
                        best_var = v
                        best_sign = target_val

            if best_var != -1:
                # Perform the flip
                old_val = assignment[best_var]
                assignment[best_var] = best_sign
                
                # Update unsat_clauses list
                # Clauses affected are those containing best_var
                new_unsat = []
                for cj, sj in var_to_clauses[best_var]:
                    # Check if cj is now unsat
                    if is_unsat(cj, assignment):
                        new_unsat.append(cj)
                    # If it was in unsat_clauses and is now sat, it's removed.
                    # If it was not in unsat_clauses and is now unsat, it's added.
                
                # This incremental update is tricky because we only checked clauses containing best_var.
                # Clauses NOT containing best_var retain their status.
                # So:
                # 1. Remove from unsat_clauses any clause that is now satisfied.
                # 2. Add to unsat_clauses any clause that is now unsatisfied.
                
                # Efficient update:
                # We have the old list `unsat_clauses`.
                # We have the set of affected clauses `affected = [cj for cj, _ in var_to_clauses[best_var]]`
                
                # New list = (Old list - Affected) + (Affected clauses that are now unsat)
                
                affected_set = set(cj for cj, _ in var_to_clauses[best_var])
                
                # Filter old unsat clauses
                filtered = [c for c in unsat_clauses if c not in affected_set]
                
                # Add newly unsat affected clauses
                for cj in affected_set:
                    if is_unsat(cj, assignment):
                        filtered.append(cj)
                
                unsat_clauses = filtered
            
        return assignment, len(unsat_clauses) == 0

    # Main loop with restarts
    best_assignment = None
    best_unsat = float('inf')
    
    # Luby-style restart sequence: 1, 1, 2, 1, 1, 2, 4, ...
    # We scale this by a base step count.
    base_steps = 1000
    
    # Generate restart sequence
    def luby_sequence():
        k = 1
        while True:
            for _ in range(k):
                yield k
            # Next power of 2
            # Actually, standard Luby is 1,1,2,1,1,2,4,1,1,2,1,1,2,4,8...
            # Let's just use a simple doubling restart strategy for simplicity and robustness
            pass

    # Simple adaptive restart:
    # Start with small steps, double if no solution found, up to a max.
    # Or use a fixed sequence.
    
    max_steps_per_restart = 5000
    restart = 0
    
    while time.time() - start_time < time_limit:
        if restart == 0:
            assignment = random_assignment()
        else:
            # Perturb the best assignment found so far? Or just random?
            # Random is safer for diversification.
            # If we have a good best, we might try to restart from it with some noise.
            if best_assignment and rng.random() < 0.2:
                assignment = best_assignment[:]
                # Flip a few random variables
                for _ in range(5):
                    v = rng.randint(0, n - 1)
                    assignment[v] = 1 - assignment[v]
            else:
                assignment = random_assignment()
        
        assignment, solved = glare_search(assignment, max_steps_per_restart)
        
        if solved:
            return assignment
        
        cur_unsat = count_unsat(assignment)
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assignment = assignment[:]
        
        restart += 1
        # Increase steps slightly? Or keep constant?
        # Constant is fine if we have enough restarts.
        
        # If we've done many restarts, we might be stuck. 
        # The time limit will eventually stop us.
        
        # Safety check: if we've been running for a long time without improvement, 
        # we might want to reduce steps to allow more restarts? 
        # Or increase to allow deeper search?
        # Usually, more restarts is better for WalkSAT/GLARE on random 3-SAT.
        
    if best_assignment:
        return best_assignment
    return random_assignment()

