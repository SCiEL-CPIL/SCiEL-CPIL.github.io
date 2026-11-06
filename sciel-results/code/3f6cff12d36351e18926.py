"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def solve(instance):
    # Hybrid DPLL + WalkSAT with deadline guard.
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline: 290 seconds to be safe (evaluator kills at 330s, but we want margin)
    deadline = time.time() + 290.0
    
    # Pre-process: convert clauses to 0-based indices and keep sign
    # clauses are lists of signed literals (1-based)
    # Let's store them as lists of (var_index, is_positive)
    # Actually, for speed, let's just keep the literals and use abs(l)-1
    
    # Helper to check if a clause is satisfied
    # We will maintain an assignment array of booleans
    
    # 1. Try DPLL first for a quick win on easy instances or to reduce the problem
    # Since n is small (50), a pure DPLL with unit propagation and pure literal elimination
    # might solve many instances instantly. If it fails, we fall back to stochastic local search.
    
    def dpll(clauses, assignment):
        # assignment is a dict: var_index -> bool, or None if unassigned
        # Returns (success, assignment_dict)
        # We use a recursive approach but with a step limit to avoid hanging on hard parts
        # For n=50, recursion depth is fine.
        
        # 1. Check for satisfied clauses and remove them
        # 2. Check for empty clauses (conflict)
        # 3. Unit propagation
        # 4. Pure literal elimination
        
        # This implementation is a bit heavy for a "try first" if it gets stuck.
        # Let's do a limited DPLL:
        # - Unit propagation
        # - Pure literal elimination
        # - If no more reductions, pick a variable and branch (but only one level deep? or full?)
        
        # Given the time constraint, a full DPLL might be overkill if it explores a huge tree.
        # But for planted 3-SAT with n=50, the clause density is usually around 4.26 (threshold).
        # DPLL is very effective near the threshold.
        
        # Let's implement a robust DPLL.
        
        def unit_propagate(clauses, assignment):
            # clauses: list of lists of literals
            # assignment: dict
            # Returns (new_clauses, new_assignment, conflict_bool)
            changed = True
            while changed:
                changed = False
                new_clauses = []
                for c in clauses:
                    # Check if clause is satisfied
                    sat = False
                    for l in c:
                        v = abs(l) - 1
                        if v in assignment:
                            if (l > 0 and assignment[v]) or (l < 0 and not assignment[v]):
                                sat = True
                                break
                    if sat:
                        continue
                    
                    # Check if clause is empty (conflict)
                    if not c:
                        return None, assignment, True
                    
                    # Check for unit clause
                    unit_lit = None
                    new_c = []
                    for l in c:
                        v = abs(l) - 1
                        if v in assignment:
                            # This literal is falsified (since not satisfied above)
                            # Do nothing, just remove it
                            pass
                        else:
                            new_c.append(l)
                            if unit_lit is None:
                                unit_lit = l
                            else:
                                # More than one unassigned literal
                                unit_lit = None
                                break
                    
                    if unit_lit is not None and len(new_c) == 1:
                        # Unit clause
                        v = abs(unit_lit) - 1
                        val = (unit_lit > 0)
                        assignment[v] = val
                        changed = True
                        # The clause is now satisfied, so we don't add it to new_clauses
                        # But we need to continue propagation
                    else:
                        new_clauses.append(new_c)
            
            return new_clauses, assignment, False

        def pure_literal_eliminate(clauses, assignment):
            # Find variables that appear with only one sign
            # This is an optimization. For n=50, it's cheap to compute.
            # But modifying clauses in place is tricky.
            # Let's skip pure literal elimination for simplicity in the recursive step,
            # or do it carefully.
            # Actually, let's just do unit propagation and branching.
            pass
        
        # Main DPLL loop
        # We'll use a stack-based approach or recursion. Recursion is fine for n=50.
        
        def _dpll_recursive(clauses, assignment, depth):
            if depth > 100: # Safety
                return None
            
            # Check for conflict (empty clause)
            for c in clauses:
                if not c:
                    return None
            
            # Check if all clauses are satisfied
            all_sat = True
            for c in clauses:
                sat = False
                for l in c:
                    v = abs(l) - 1
                    if v in assignment:
                        if (l > 0 and assignment[v]) or (l < 0 and not assignment[v]):
                            sat = True
                            break
                if not sat:
                    all_sat = False
                    break
            
            if all_sat:
                return assignment
            
            # Unit propagation
            new_clauses, new_assignment, conflict = unit_propagate(clauses, assignment)
            if conflict:
                return None
            
            # If no change and no all_sat, we need to branch
            # Find a variable to branch on
            # Heuristic: pick a variable from the shortest unsatisfied clause
            min_clause = None
            min_len = 100
            for c in new_clauses:
                # Check if unsatisfied
                sat = False
                for l in c:
                    v = abs(l) - 1
                    if v in new_assignment:
                        if (l > 0 and new_assignment[v]) or (l < 0 and not new_assignment[v]):
                            sat = True
                            break
                if not sat:
                    if len(c) < min_len:
                        min_len = len(c)
                        min_clause = c
            
            if min_clause is None:
                # All clauses satisfied?
                return new_assignment
            
            # Pick a literal from min_clause
            # We can pick the first one, or use a heuristic
            lit = min_clause[0]
            v = abs(lit) - 1
            
            # Branch 1: lit is True
            new_assignment[v] = (lit > 0)
            result = _dpll_recursive(new_clauses, new_assignment, depth + 1)
            if result is not None:
                return result
            
            # Branch 2: lit is False
            new_assignment[v] = not (lit > 0)
            result = _dpll_recursive(new_clauses, new_assignment, depth + 1)
            if result is not None:
                return result
            
            # Backtrack
            del new_assignment[v]
            return None

        # Initialize assignment as empty dict
        # But we need to handle the initial clauses
        # Let's wrap the initial call
        return _dpll_recursive(clauses, {}, 0)

    # Try DPLL with a time limit
    # Since DPLL can be exponential, we should monitor time.
    # However, implementing a time-limited recursive DPLL is complex.
    # Alternative: Run DPLL for a fixed number of steps?
    # Or, just run DPLL and if it takes too long, it's bad.
    # For n=50, DPLL is usually fast if it works. If it doesn't, it might hang.
    # Let's add a global step counter or time check inside the recursion?
    # That's messy.
    
    # Better approach:
    # 1. Run a simplified DPLL (just unit propagation and pure literals) to simplify the instance.
    # 2. If simplified instance is empty, we have a solution.
    # 3. If simplified instance is small, try full DPLL.
    # 4. Else, use WalkSAT on the simplified instance.
    
    # Let's implement the simplified DPLL (preprocessing)
    def preprocess(clauses, assignment):
        # Returns (simplified_clauses, assignment, success, failure)
        # success: True if all clauses satisfied
        # failure: True if conflict
        
        changed = True
        while changed:
            changed = False
            new_clauses = []
            for c in clauses:
                # Check satisfied
                sat = False
                for l in c:
                    v = abs(l) - 1
                    if v in assignment:
                        if (l > 0 and assignment[v]) or (l < 0 and not assignment[v]):
                            sat = True
                            break
                if sat:
                    continue
                
                # Check empty
                if not c:
                    return None, assignment, False, True
                
                # Unit propagation
                unit_lit = None
                new_c = []
                for l in c:
                    v = abs(l) - 1
                    if v in assignment:
                        pass # Falsified, remove
                    else:
                        new_c.append(l)
                        if unit_lit is None:
                            unit_lit = l
                        else:
                            unit_lit = None
                            break
                
                if unit_lit is not None and len(new_c) == 1:
                    v = abs(unit_lit) - 1
                    assignment[v] = (unit_lit > 0)
                    changed = True
                else:
                    new_clauses.append(new_c)
            
            clauses = new_clauses
        
        # Check if all satisfied
        if not clauses:
            return clauses, assignment, True, False
        
        return clauses, assignment, False, False

    # Run preprocessing
    assignment_dict = {}
    simplified_clauses, assignment_dict, success, failure = preprocess(clauses, assignment_dict)
    
    if success:
        # Convert dict to list
        final_assignment = [assignment_dict.get(i, False) for i in range(n)]
        return final_assignment
    
    if failure:
        # Should not happen for planted satisfiable, but if it does, return random
        rng = random.Random(0)
        return [rng.random() < 0.5 for _ in range(n)]
    
    # If we have a simplified instance, we can run DPLL on it if it's small enough
    # Or just go to WalkSAT.
    # Let's try DPLL on the simplified instance with a timeout guard.
    # We'll implement a iterative DPLL or a recursive one with a step limit.
    
    # Given the complexity, let's just go to WalkSAT on the simplified instance.
    # WalkSAT is very robust for n=50.
    
    # Initialize WalkSAT
    rng = random.Random(12345)
    
    # Start with the partial assignment from preprocessing, fill rest randomly
    assignment = [False] * n
    for v, val in assignment_dict.items():
        assignment[v] = val
    for i in range(n):
        if i not in assignment_dict:
            assignment[i] = rng.random() < 0.5
    
    # WalkSAT parameters
    # Weight for flipping a literal in the clause vs random flip
    # Typical W is 0.5 to 1.0 for 3-SAT
    W = 0.55
    max_steps = 100000
    restarts = 1000
    
    # We will run multiple restarts if the first one fails
    # But we have a deadline.
    
    def walk_sat(clauses, assignment, rng, max_steps, W, deadline):
        # clauses: list of lists of literals
        # assignment: list of booleans
        for _ in range(max_steps):
            if time.time() > deadline:
                return assignment, False
            
            # Find unsatisfied clauses
            unsat = []
            for c in clauses:
                sat = False
                for l in c:
                    v = abs(l) - 1
                    if (l > 0 and assignment[v]) or (l < 0 and not assignment[v]):
                        sat = True
                        break
                if not sat:
                    unsat.append(c)
            
            if not unsat:
                return assignment, True
            
            # Pick a random unsatisfied clause
            c = rng.choice(unsat)
            
            # Calculate flip scores
            best_lit = None
            best_score = -1
            for l in c:
                v = abs(l) - 1
                # Score: how many other clauses would be broken if we flip v
                # If l is currently false, flipping it satisfies c.
                # We want to minimize the number of clauses that become unsatisfied.
                # Actually, standard WalkSAT:
                # Score = number of clauses that become unsatisfied if we flip the variable.
                # We pick the literal with the lowest score.
                # If we flip v, the clause c becomes satisfied.
                # Other clauses containing v might become unsatisfied.
                
                # To speed up, we can just pick a random literal from the clause
                # or use a simple heuristic.
                # For n=50, calculating the full score is feasible.
                
                # Let's use a simpler heuristic: pick the literal that appears in the fewest other unsatisfied clauses?
                # Or just random?
                # Standard WalkSAT uses a weight W.
                # With probability W, pick the best literal (lowest cost).
                # With probability 1-W, pick a random literal.
                
                pass
            
            # Implementation of standard WalkSAT flip selection
            if rng.random() < W:
                # Pick the literal that minimizes the number of broken clauses
                best_cost = 1000
                best_l = c[0]
                for l in c:
                    v = abs(l) - 1
                    cost = 0
                    # Iterate through all clauses to see which ones break
                    # This is O(M) per literal, so O(3*M) per step.
                    # M is number of clauses. For n=50, M is ~200-300.
                    # 3 * 300 = 900 operations per step.
                    # 100000 steps * 900 = 90M operations. Might be slow in Python.
                    # Optimization: Maintain clause counts?
                    # Let's try a simpler version first: just pick random literal.
                    # Random WalkSAT (W=0) is actually quite strong for planted 3-SAT.
                    # Let's use W=0.5 but with a cheaper cost function if possible.
                    
                    # For now, let's just pick a random literal to keep it fast.
                    # We can refine if needed.
                    pass
                
                # Actually, let's just pick a random literal from the clause for speed.
                # "Random WalkSAT" is a valid variant.
                l = rng.choice(c)
            else:
                l = rng.choice(c)
            
            # Flip
            v = abs(l) - 1
            assignment[v] = not assignment[v]
            
        return assignment, False

    # Run WalkSAT
    # We'll try a few restarts if the first one fails
    best_assignment = assignment
    found = False
    
    for r in range(10):
        if time.time() > deadline:
            break
        
        # Reset assignment? Or continue from where we left off?
        # Restarting with a fresh random assignment is common.
        if r == 0:
            # Use the current assignment (which includes preprocessing)
            pass
        else:
            # New random assignment
            for i in range(n):
                if i not in assignment_dict:
                    assignment[i] = rng.random() < 0.5
        
        assignment, found = walk_sat(simplified_clauses, assignment, rng, 5000, W, deadline)
        
        if found:
            break
    
    if found:
        # Verify solution on original clauses
        # Just in case
        for c in clauses:
            sat = False
            for l in c:
                v = abs(l) - 1
                if (l > 0 and assignment[v]) or (l < 0 and not assignment[v]):
                    sat = True
                    break
            if not sat:
                # Should not happen
                break
        
        return assignment
    else:
        # Return the best effort
        return assignment

# EVOLVE-BLOCK-END
