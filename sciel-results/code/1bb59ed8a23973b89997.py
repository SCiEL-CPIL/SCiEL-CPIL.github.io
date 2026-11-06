import random
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Time limit with safety margin
    deadline = time.time() + 290.0
    
    # Pre-process
    # var_to_clauses: list of lists of (clause_idx, sign)
    # sign: 1 if positive, 0 if negative
    var_to_clauses = [[] for _ in range(n)]
    for i, c in enumerate(clauses):
        for l in c:
            v = abs(l) - 1
            sign = 1 if l > 0 else 0
            var_to_clauses[v].append((i, sign))
            
    m = len(clauses)
    
    # Helper to check if a clause is satisfied given an assignment
    def is_satisfied(clause_idx, assignment):
        c = clauses[clause_idx]
        for l in c:
            v = abs(l) - 1
            if assignment[v] == (l > 0):
                return True
        return False

    # Helper to count unsat clauses
    def count_unsat(assignment):
        count = 0
        for i in range(m):
            if not is_satisfied(i, assignment):
                count += 1
        return count

    rng = random.Random(0)
    best_assignment = None
    best_unsat = float('inf')
    
    # Parameters
    max_restarts = 200
    steps_per_restart = 2000
    noise = 0.1 # Probability of random flip
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Random initialization
        assignment = [rng.random() < 0.5 for _ in range(n)]
        current_unsat = count_unsat(assignment)
        
        if current_unsat == 0:
            return assignment
        
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assignment = assignment[:]
            
        local_best_unsat = current_unsat
        no_improve = 0
        
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
                
            # Find an unsatisfied clause
            # To avoid O(M) scan every step, we could maintain a list of unsat clauses.
            # But for N=150, M~500, O(M) is 500 checks. 2000 steps * 500 = 1M checks.
            # This is fine.
            
            unsat_clauses = [i for i in range(m) if not is_satisfied(i, assignment)]
            
            if not unsat_clauses:
                return assignment
            
            c_idx = rng.choice(unsat_clauses)
            c = clauses[c_idx]
            
            # Find best variable to flip in this clause
            best_var = -1
            best_delta = float('inf')
            
            candidates = []
            for l in c:
                v = abs(l) - 1
                # Calculate delta for flipping v
                # Flipping v satisfies c_idx (since c_idx is unsat).
                # So we gain 1 satisfied clause.
                # We might break other clauses containing v.
                
                breaks = 0
                for (ci, sign) in var_to_clauses[v]:
                    if ci == c_idx:
                        continue
                    # Check if clause ci is currently satisfied
                    # If it is satisfied, and we flip v, it might become unsatisfied.
                    # If it is already unsatisfied, flipping v might satisfy it (but we only count breaks).
                    # Actually, delta = (newly unsatisfied) - (newly satisfied).
                    # We know c_idx becomes satisfied.
                    # For other clauses containing v:
                    # If clause ci is currently SAT:
                    #   If v is the ONLY literal in ci that is True, flipping v makes ci UNSAT.
                    #   If v is True and other literals are False -> flipping v makes v False. If no other True, ci becomes UNSAT.
                    #   Wait, simpler:
                    #   Current state of ci: Sat or Unsat.
                    #   If we flip v, does ci change state?
                    #   It changes state if v is the "deciding" literal.
                    
                    # Let's just evaluate the state of ci before and after?
                    # That's expensive.
                    
                    # Efficient check:
                    # A clause is satisfied if ANY literal is true.
                    # If we flip v, the value of literal l (in ci) changes.
                    # If ci was SAT:
                    #   It becomes UNSAT only if v was the ONLY true literal in ci.
                    # If ci was UNSAT:
                    #   It becomes SAT only if v becomes the true literal (and it was false before).
                    
                    # We need to know if v is the only true literal in ci (if ci is sat).
                    # Or if v is the only false literal... no, if ci is unsat, all are false.
                    # If we flip v, v becomes true. So ci becomes SAT.
                    
                    # So:
                    # For each other clause ci containing v:
                    # 1. Is ci currently SAT?
                    #    If yes, check if v is the ONLY true literal in ci.
                    #       If yes, flipping v breaks it (+1 break).
                    #    If no, flipping v satisfies it (-1 break, i.e., +1 sat).
                    
                    # We need to check "is v the only true literal in ci".
                    # This requires scanning ci.
                    # ci has 3 literals.
                    
                    # This is getting complex for a pure Python loop.
                    # Let's simplify: Just count unsat clauses from scratch after each flip?
                    # No, that's O(M) per flip.
                    
                    # Alternative: Use a simpler heuristic.
                    # Pick the variable in the clause that appears in the FEWEST other clauses.
                    # This is a common approximation (Min-Conflict).
                    
                    # Let's stick to the standard WalkSAT:
                    # For each var v in clause c:
                    #   Count how many other clauses containing v are currently UNSAT.
                    #   Let this be `unsat_count_v`.
                    #   If we flip v:
                    #     c becomes SAT.
                    #     Any UNSAT clause containing v becomes SAT (because v flips from False to True? 
                    #     Wait. If clause is UNSAT, all literals are False.
                    #     If v is in the clause, literal l_v is False.
                    #     Flipping v makes l_v True. So the clause becomes SAT.
                    #     So flipping v satisfies ALL unsat clauses containing v.
                    #     It also breaks SAT clauses containing v where v is the only true literal.
                    
                    # So:
                    #   Satisfied = 1 (for c) + (number of other UNSAT clauses containing v)
                    #   Broken = (number of other SAT clauses containing v where v is the only true literal)
                    
                    #   Delta = Broken - Satisfied
                    #   We want to minimize Delta (maximize Satisfied - Broken).
                    
                    # Let's implement this.
                    
                    break_count = 0
                    sat_count = 0
                    
                    for (ci, sign) in var_to_clauses[v]:
                        if ci == c_idx:
                            continue
                        # Check if ci is SAT
                        ci_sat = is_satisfied(ci, assignment)
                        if ci_sat:
                            # Check if v is the only true literal
                            only_true = True
                            for ll in clauses[ci]:
                                vv = abs(ll) - 1
                                if vv == v:
                                    continue
                                if assignment[vv] == (ll > 0):
                                    only_true = False
                                    break
                            if only_true:
                                break_count += 1
                        else:
                            # ci is UNSAT. Flipping v (which is in ci) will make it SAT.
                            sat_count += 1
                    
                    # Total satisfied = 1 (c_idx) + sat_count
                    # Total broken = break_count
                    # Score to maximize = (1 + sat_count) - break_count
                    # Or minimize Delta = break_count - (1 + sat_count)
                    
                    score = (1 + sat_count) - break_count
                    # Add some noise
                    score += rng.random() * 0.5
                    
                    candidates.append((score, v))
            
            # Pick the best candidate
            candidates.sort(reverse=True)
            best_score, best_var = candidates[0]
            
            # Flip
            assignment[best_var] = not assignment[best_var]
            
            # Update current_unsat
            # We could update incrementally, but let's just recount to be safe and simple
            # Given the small size, recounting is okay.
            current_unsat = count_unsat(assignment)
            
            if current_unsat < local_best_unsat:
                local_best_unsat = current_unsat
                no_improve = 0
                if current_unsat < best_unsat:
                    best_unsat = current_unsat
                    best_assignment = assignment[:]
            else:
                no_improve += 1
                
            if no_improve > 100:
                break # Restart early if stuck

    return best_assignment
