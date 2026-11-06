"""Improved planted 3-SAT solver with WalkSAT + DPLL hybrid and time guard."""
import time
import random

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 280  # safety margin before 300s limit

    # Pre-compute clause structure for speed
    # Store clauses as list of (var_index, sign) where var_index is 0-based
    clause_vars = []
    for c in clauses:
        cv = []
        for l in c:
            var_idx = abs(l) - 1
            sign = 1 if l > 0 else -1
            cv.append((var_idx, sign))
        clause_vars.append(cv)

    def is_satisfied(assignment):
        for cv in clause_vars:
            sat = False
            for var_idx, sign in cv:
                if assignment[var_idx] == (sign == 1):
                    sat = True
                    break
            if not sat:
                return False
        return True

    def count_unsat(assignment):
        count = 0
        for cv in clause_vars:
            sat = False
            for var_idx, sign in cv:
                if assignment[var_idx] == (sign == 1):
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    def walksat(assignment, max_steps):
        """WalkSAT with adaptive noise."""
        rng = random.Random()
        steps = 0
        while steps < max_steps:
            if time.time() > deadline:
                return assignment
            # Find unsatisfied clauses
            unsat_clauses = []
            for i, cv in enumerate(clause_vars):
                sat = False
                for var_idx, sign in cv:
                    if assignment[var_idx] == (sign == 1):
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(i)
            
            if not unsat_clauses:
                return assignment
            
            # Pick a random unsatisfied clause
            ci = rng.choice(unsat_clauses)
            cv = clause_vars[ci]
            
            # Count how many literals in this clause would be satisfied if flipped
            # Prefer flipping a literal that satisfies the clause
            best_var = None
            best_score = float('inf')
            for var_idx, sign in cv:
                # If we flip var_idx, does it satisfy this clause?
                # Currently assignment[var_idx] != (sign == 1) since clause is unsat
                # After flip, assignment[var_idx] becomes (sign == 1), so it satisfies
                # Count how many OTHER clauses this flip would break
                score = 0
                for j, cj in enumerate(clause_vars):
                    if j == ci:
                        continue
                    # Check if cj is currently satisfied only by var_idx
                    # i.e., cj has var_idx as its only satisfying literal
                    has_var = False
                    other_sat = False
                    for vi, si in cj:
                        if vi == var_idx:
                            has_var = True
                            # Currently assignment[vi] == (si == 1) means this literal is true
                            if assignment[vi] == (si == 1):
                                # This literal is currently true; if we flip, it becomes false
                                pass
                        else:
                            if assignment[vi] == (si == 1):
                                other_sat = True
                                break
                    if has_var and not other_sat:
                        # Flipping var_idx would break clause cj
                        score += 1
                
                if score < best_score:
                    best_score = score
                    best_var = var_idx
            
            # With some probability, flip a random variable in the clause instead
            if rng.random() < 0.2:
                var_idx, sign = rng.choice(cv)
            else:
                var_idx, sign = best_var
            
            assignment[var_idx] = not assignment[var_idx]
            steps += 1
        
        return assignment

    def dpll(assignment, time_limit=10.0):
        """DPLL with unit propagation and pure literal elimination."""
        start_time = time.time()
        
        # Work on a copy
        assign = assignment[:]
        
        # Track which variables are assigned
        # Use a dict for partial assignment, or just work with the list and a set of unassigned
        unassigned = set(range(n))
        
        # Clauses that are still active (not satisfied, not empty)
        # Represent clauses as sets of (var_idx, sign)
        active_clauses = [set(cv) for cv in clause_vars]
        
        def propagate():
            """Unit propagation and pure literal elimination."""
            changed = True
            while changed:
                changed = False
                
                # Unit propagation
                for i, clause in enumerate(active_clauses):
                    if not clause:
                        return False  # Contradiction
                    if len(clause) == 1:
                        var_idx, sign = next(iter(clause))
                        if var_idx not in unassigned:
                            # Already assigned; check consistency
                            if assign[var_idx] != (sign == 1):
                                return False
                            continue
                        # Assign this variable
                        assign[var_idx] = (sign == 1)
                        unassigned.discard(var_idx)
                        changed = True
                        # Remove this literal from all clauses, remove clauses that become empty
                        for j, cj in enumerate(active_clauses):
                            if (var_idx, sign) in cj:
                                cj.discard((var_idx, sign))
                            if (var_idx, -sign) in cj:
                                cj.discard((var_idx, -sign))
                        # Remove satisfied clauses
                        active_clauses = [cj for cj in active_clauses if cj]
                        break
                
                if not active_clauses:
                    return True  # All clauses satisfied
                
                # Pure literal elimination
                if unassigned:
                    lit_occurrences = {}
                    for clause in active_clauses:
                        for var_idx, sign in clause:
                            if var_idx in unassigned:
                                key = (var_idx, sign)
                                lit_occurrences[key] = lit_occurrences.get(key, 0) + 1
                    
                    for (var_idx, sign), count in lit_occurrences.items():
                        opposite = (var_idx, -sign)
                        if opposite not in lit_occurrences and var_idx in unassigned:
                            assign[var_idx] = (sign == 1)
                            unassigned.discard(var_idx)
                            changed = True
                            for j, cj in enumerate(active_clauses):
                                if (var_idx, sign) in cj:
                                    cj.discard((var_idx, sign))
                                if (var_idx, -sign) in cj:
                                    cj.discard((var_idx, -sign))
                            active_clauses = [cj for cj in active_clauses if cj]
                            break
                    else:
                        # No pure literals found, continue
                        pass
                    if not active_clauses:
                        return True
            
            return bool(active_clauses) or True  # No contradiction
        
        # Main DPLL loop
        while True:
            if time.time() - start_time > time_limit:
                return None  # Timeout
            
            if not propagate():
                return None  # Contradiction
            
            if not active_clauses:
                return assign  # Satisfied
            
            if not unassigned:
                return None  # All assigned but not satisfied
            
            # Choose a variable to branch on
            # Pick a variable that appears in the most clauses
            var_count = {}
            for clause in active_clauses:
                for var_idx, sign in clause:
                    if var_idx in unassigned:
                        var_count[var_idx] = var_count.get(var_idx, 0) + 1
            
            if not var_count:
                return None
            
            var = max(var_count, key=var_count.get)
            
            # Try both assignments
            for val in [True, False]:
                assign[var] = val
                unassigned.discard(var)
                
                # Copy state for recursion (simplified: just try and backtrack)
                saved_clauses = [set(cj) for cj in active_clauses]
                saved_unassigned = set(unassigned)
                
                # Apply assignment
                for i, cj in enumerate(active_clauses):
                    if (var, 1) in cj:
                        cj.discard((var, 1))
                    if (var, -1) in cj:
                        cj.discard((var, -1))
                active_clauses = [cj for cj in active_clauses if cj]
                
                result = dpll(assign, time_limit)
                if result is not None:
                    return result
                
                # Backtrack
                active_clauses = saved_clauses
                unassigned = saved_unassigned
                assign[var] = not val
        
        return None

    # Strategy: Try WalkSAT first with limited budget, then DPLL if needed
    rng = random.Random(42)
    
    # Initial random assignment
    assignment = [rng.random() < 0.5 for _ in range(n)]
    
    # Phase 1: WalkSAT with moderate budget
    assignment = walksat(assignment, 5000)
    
    if is_satisfied(assignment):
        return assignment
    
    # Phase 2: Try DPLL from current assignment
    result = dpll(assignment, time_limit=20.0)
    if result is not None and is_satisfied(result):
        return result
    
    # Phase 3: Restart with different seeds for WalkSAT
    for seed in range(10):
        if time.time() > deadline:
            break
        rng2 = random.Random(seed * 12345 + 67890)
        assignment2 = [rng2.random() < 0.5 for _ in range(n)]
        assignment2 = walksat(assignment2, 3000)
        if is_satisfied(assignment2):
            return assignment2
        
        # Try DPLL from this assignment
        result2 = dpll(assignment2, time_limit=15.0)
        if result2 is not None and is_satisfied(result2):
            return result2
    
    # Phase 4: Final attempt with longer WalkSAT
    if time.time() < deadline - 10:
        rng3 = random.Random(99999)
        assignment3 = [rng3.random() < 0.5 for _ in range(n)]
        assignment3 = walksat(assignment3, 10000)
        if is_satisfied(assignment3):
            return assignment3
    
    # Return best effort
    return assignment
