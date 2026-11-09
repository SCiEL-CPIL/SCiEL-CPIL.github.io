"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    start_time = time.time()
    deadline = start_time + 280.0  # 280s safety margin before 300s kill

    # Preprocess: convert clauses to 0-based signed literals
    # literal l -> variable index abs(l)-1, positive means True
    # We'll work with a list of lists of signed ints (1-based)
    
    # Check if already satisfied with all-True or all-False
    def is_satisfied(assignment, cl):
        for l in cl:
            v = abs(l) - 1
            if assignment[v] == (l > 0):
                return True
        return False

    def all_satisfied(assignment):
        for cl in clauses:
            if not is_satisfied(assignment, cl):
                return False
        return True

    # Try simple assignments first
    for init_val in [True, False]:
        assignment = [init_val] * n
        if all_satisfied(assignment):
            return assignment

    # DPLL with watched literals, unit propagation, and pure literal elimination
    def dpll_solve():
        # Returns assignment list if solved, None if failed
        # Use recursive DPLL with watched literals
        
        # Build watched literal structure
        # For each clause, watch two literals
        # watches[lit] = list of clause indices that watch lit
        
        # Convert to 0-based variable indices with sign
        # lit = (var_idx, is_positive) where var_idx in [0, n)
        
        def make_lit(l):
            return (abs(l) - 1, l > 0)
        
        # Initialize watches
        watches = {i: [] for i in range(n)}  # watches[var] tracks clauses watching var
        # Actually, let's use a different structure: for each variable, track which clauses watch it
        # and for each clause, track which two literals are watched
        
        # Simpler approach: use a set-based DPLL
        # Given the small size (50 vars), we can be less optimized
        
        # Let's do a simpler DPLL without watched literals but with unit prop
        # This should be fast enough for 50 variables
        
        def dpll_rec(clauses_remaining, assignment_partial):
            """
            clauses_remaining: list of clauses (each a list of signed 1-based lits)
            assignment_partial: dict var_idx -> bool
            Returns: dict if solved, None if unsat
            """
            if time.time() > deadline:
                return None
            
            # Check for empty clause (contradiction)
            for cl in clauses_remaining:
                if len(cl) == 0:
                    return None
            
            # Unit propagation
            changed = True
            while changed:
                changed = False
                new_clauses = []
                for cl in clauses_remaining:
                    # Check if clause is satisfied
                    sat = False
                    new_cl = []
                    for l in cl:
                        v = abs(l) - 1
                        if v in assignment_partial:
                            if assignment_partial[v] == (l > 0):
                                sat = True
                                break
                        else:
                            new_cl.append(l)
                    if sat:
                        continue
                    if len(new_cl) == 0:
                        return None  # contradiction
                    if len(new_cl) == 1:
                        # Unit clause: assign the literal
                        l = new_cl[0]
                        v = abs(l) - 1
                        assignment_partial[v] = (l > 0)
                        changed = True
                    else:
                        new_clauses.append(new_cl)
                clauses_remaining = new_clauses
            
            if not clauses_remaining:
                return dict(assignment_partial)
            
            # Pure literal elimination
            # Find literals that appear only with one sign
            lit_pos = set()
            lit_neg = set()
            for cl in clauses_remaining:
                for l in cl:
                    if l > 0:
                        lit_pos.add(l)
                    else:
                        lit_neg.add(l)
            
            pure = set()
            for l in lit_pos:
                if -l not in lit_neg:
                    pure.add(l)
            for l in lit_neg:
                if -l not in lit_pos:
                    pure.add(l)
            
            if pure:
                # Assign all pure literals
                for l in pure:
                    v = abs(l) - 1
                    if v not in assignment_partial:
                        assignment_partial[v] = (l > 0)
                # Remove satisfied clauses and re-propagate
                new_clauses = []
                for cl in clauses_remaining:
                    sat = False
                    new_cl = []
                    for l in cl:
                        v = abs(l) - 1
                        if v in assignment_partial:
                            if assignment_partial[v] == (l > 0):
                                sat = True
                                break
                        else:
                            new_cl.append(l)
                    if not sat:
                        if len(new_cl) == 0:
                            return None
                        new_clauses.append(new_cl)
                clauses_remaining = new_clauses
                # Re-run unit propagation
                changed = True
                while changed:
                    changed = False
                    new_clauses2 = []
                    for cl in clauses_remaining:
                        sat = False
                        new_cl = []
                        for l in cl:
                            v = abs(l) - 1
                            if v in assignment_partial:
                                if assignment_partial[v] == (l > 0):
                                    sat = True
                                    break
                            else:
                                new_cl.append(l)
                        if sat:
                            continue
                        if len(new_cl) == 0:
                            return None
                        if len(new_cl) == 1:
                            l = new_cl[0]
                            v = abs(l) - 1
                            assignment_partial[v] = (l > 0)
                            changed = True
                        else:
                            new_clauses2.append(new_cl)
                    clauses_remaining = new_clauses2
                if not clauses_remaining:
                    return dict(assignment_partial)
            
            # Choose a variable to branch on
            # Pick variable appearing in most clauses
            var_count = {}
            for cl in clauses_remaining:
                for l in cl:
                    v = abs(l) - 1
                    if v not in assignment_partial:
                        var_count[v] = var_count.get(v, 0) + 1
            
            if not var_count:
                return dict(assignment_partial)
            
            var = max(var_count, key=lambda v: var_count[v])
            
            # Branch: try True first
            for val in [True, False]:
                test_assign = dict(assignment_partial)
                test_assign[var] = val
                test_clauses = []
                for cl in clauses_remaining:
                    sat = False
                    new_cl = []
                    for l in cl:
                        v = abs(l) - 1
                        if v in test_assign:
                            if test_assign[v] == (l > 0):
                                sat = True
                                break
                        else:
                            new_cl.append(l)
                    if not sat:
                        if len(new_cl) == 0:
                            break
                        test_clauses.append(new_cl)
                else:
                    result = dpll_rec(test_clauses, test_assign)
                    if result is not None:
                        # Fill in unassigned variables
                        for i in range(n):
                            if i not in result:
                                result[i] = False
                        return result
            
            return None
        
        result = dpll_rec(list(clauses), {})
        if result is not None:
            return [result[i] for i in range(n)]
        return None

    # Try DPLL first
    dpll_result = dpll_solve()
    if dpll_result is not None:
        return dpll_result

    # Fallback: WalkSAT with adaptive noise
    if time.time() > deadline:
        return [random.Random(0).random() < 0.5 for _ in range(n)]

    def walk_sat():
        rng = random.Random(0)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        def count_unsat(assignment):
            count = 0
            for cl in clauses:
                sat = False
                for l in cl:
                    v = abs(l) - 1
                    if assignment[v] == (l > 0):
                        sat = True
                        break
                if not sat:
                    count += 1
            return count
        
        def get_unsat_clauses(assignment):
            unsat = []
            for cl in clauses:
                sat = False
                for l in cl:
                    v = abs(l) - 1
                    if assignment[v] == (l > 0):
                        sat = True
                        break
                if not sat:
                    unsat.append(cl)
            return unsat
        
        best_assignment = list(assignment)
        best_score = count_unsat(assignment)
        
        max_steps = 50000
        step = 0
        temperature = 0.3  # probability of random flip
        
        while step < max_steps and time.time() < deadline:
            unsat = get_unsat_clauses(assignment)
            if not unsat:
                return assignment
            
            # Update best
            current_score = len(unsat)
            if current_score < best_score:
                best_score = current_score
                best_assignment = list(assignment)
            
            # Pick an unsatisfied clause
            clause = rng.choice(unsat)
            
            # Score each literal in the clause: number of other clauses that would be satisfied
            best_lit = None
            best_gain = -1
            second_best_gain = -1
            second_best_lit = None
            
            for l in clause:
                v = abs(l) - 1
                # What if we flip v to satisfy this literal?
                new_val = (l > 0)
                if assignment[v] == new_val:
                    # Already set correctly, flipping would break this clause
                    gain = -100
                else:
                    # Count how many other unsat clauses would become satisfied
                    gain = 0
                    for cl2 in unsat:
                        if cl2 is clause:
                            continue
                        for l2 in cl2:
                            if abs(l2) - 1 == v:
                                if (l2 > 0) == new_val:
                                    gain += 1
                                break
                if gain > best_gain:
                    second_best_gain = best_gain
                    second_best_lit = best_lit
                    best_gain = gain
                    best_lit = l
                elif gain > second_best_gain:
                    second_best_gain = gain
                    second_best_lit = l
            
            # Adaptive noise: with probability temperature, pick a random literal
            if rng.random() < temperature:
                chosen_lit = rng.choice(clause)
            else:
                # With some probability pick second best for diversity
                if second_best_lit is not None and rng.random() < 0.2:
                    chosen_lit = second_best_lit
                else:
                    chosen_lit = best_lit
            
            v = abs(chosen_lit) - 1
            assignment[v] = (chosen_lit > 0)
            
            step += 1
            
            # Occasionally restart with best assignment
            if step % 5000 == 0 and best_score < count_unsat(assignment):
                assignment = list(best_assignment)
                temperature = max(0.1, temperature * 0.9)
        
        # Return best found
        if best_score == 0:
            return best_assignment
        # Check if current is better
        if count_unsat(assignment) < best_score:
            return assignment
        return best_assignment

    result = walk_sat()
    
    # Final verification
    if all_satisfied(result):
        return result
    
    # Try a few more quick WalkSAT restarts if time permits
    if time.time() < deadline - 5:
        for seed in [1, 2, 3, 4, 5]:
            if time.time() > deadline - 2:
                break
            rng = random.Random(seed)
            assignment = [rng.random() < 0.5 for _ in range(n)]
            for _ in range(10000):
                if time.time() > deadline - 1:
                    break
                unsat = [c for c in clauses if not is_satisfied(assignment, c)]
                if not unsat:
                    return assignment
                clause = rng.choice(unsat)
                literal = rng.choice(clause)
                assignment[abs(literal) - 1] = not assignment[abs(literal) - 1]
            if all_satisfied(assignment):
                return assignment
    
    return result
# EVOLVE-BLOCK-END
