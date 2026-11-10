"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    # Planted 3-SAT solver with a time-bounded local search portfolio.
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Deadline guard: aim to finish well before the 330s wall-clock kill.
    deadline = time.time() + 280.0
    
    # Preprocess: convert clauses to arrays of (var_idx, is_pos) for speed.
    # var_idx is 0-based, is_pos indicates if the literal is positive.
    clause_data = []
    for c in clauses:
        cd = [(abs(l) - 1, l > 0) for l in c]
        clause_data.append(cd)
    
    # Check for trivially satisfied clauses or unit clauses? 
    # For planted SAT, usually random. Let's just run local search.
    
    # Strategy: 
    # 1. Try a few random restarts with WalkSAT variants.
    # 2. Use a better heuristic: pick the unsatisfied clause with the fewest flippable literals (VSIDS-like or just min flippable).
    # 3. Within a clause, pick the literal with the highest weight (number of unsatisfied clauses it would satisfy if flipped) or random.
    
    # We'll implement a robust WalkSAT with:
    # - Random restarts
    # - Clause selection: random among unsatisfied, or weighted by flippable count.
    # - Variable selection: weighted by impact (number of other unsat clauses containing this var).
    
    # To be fast, we maintain counts of unsatisfied clauses per variable.
    # However, for 50 variables and likely < 1000 clauses, a simple approach might suffice if optimized.
    # Let's try to be a bit smarter.
    
    def check_satisfied(assignment):
        for cd in clause_data:
            sat = False
            for var, is_pos in cd:
                if assignment[var] == is_pos:
                    sat = True
                    break
            if not sat:
                return False
        return True
    
    def get_unsat_clauses(assignment):
        unsat = []
        for i, cd in enumerate(clause_data):
            sat = False
            for var, is_pos in cd:
                if assignment[var] == is_pos:
                    sat = True
                    break
            if not sat:
                unsat.append(i)
        return unsat
    
    # Heuristic: For each variable, count how many unsat clauses it appears in.
    # Flipping a variable in an unsat clause that appears in many other unsat clauses is good.
    
    def walk_sat(assignment, rng, max_steps=50000):
        # Precompute clause indices for each variable? 
        # var_to_clauses[var] = list of clause indices containing var
        # This helps to quickly find unsat clauses containing a variable.
        # But we need to know if the clause is unsat.
        
        # Let's just iterate. n=50 is small.
        
        unsat_indices = get_unsat_clauses(assignment)
        if not unsat_indices:
            return assignment
        
        # Build a map: var -> count of unsat clauses it is in
        # And for each unsat clause, the variables in it.
        
        # To speed up, let's precompute the list of variables for each clause.
        # clause_vars[i] = [var1, var2, var3]
        clause_vars = [[var for var, _ in cd] for cd in clause_data]
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            if not unsat_indices:
                return assignment
            
            # Select a clause: 
            # Option A: Random
            # Option B: Weighted by 1/len(flippable) - not easy to get flippable count without checking.
            # Let's stick to random clause selection for simplicity, but choose variable smartly.
            
            clause_idx = rng.choice(unsat_indices)
            cd = clause_data[clause_idx]
            vars_in_clause = clause_vars[clause_idx]
            
            # Calculate impact for each variable in this clause
            # Impact = number of other unsat clauses that contain this variable
            # We need to know which clauses are unsat.
            # unsat_set = set(unsat_indices)
            
            # For each var in vars_in_clause:
            #   count = 0
            #   for other_clause_idx in unsat_indices:
            #       if var in clause_vars[other_clause_idx]:
            #           count += 1
            #   impact[var] = count
            
            # This is O(m * 3) per step, which is fine for m~1000, steps~50000? 
            # 50000 * 3000 = 150M ops. Might be slow in Python.
            
            # Optimization: Maintain unsat_count_per_var dynamically?
            # Too complex for now. Let's try a simpler heuristic first: 
            # Pick the variable that appears in the most unsat clauses.
            
            # To make it faster, let's precompute a list of unsat clause indices for each variable?
            # No, unsat status changes.
            
            # Alternative: Just pick random variable from the clause. 
            # The seed program did that and got 1.0. 
            # Let's try: Pick the variable with the highest frequency in the unsat set.
            
            # To speed up the frequency count:
            # We can maintain a global count of unsat clauses per variable, updating it on each flip.
            
            # Let's restructure to maintain unsat_count_per_var.
            
            # This requires careful bookkeeping.
            # Let's try a different approach: 
            # Since n=50 is very small, we can try all 2^50 assignments? No.
            # But we can try a greedy approach or a better local search.
            
            # Let's implement the dynamic count version.
            
            # Actually, let's just run multiple restarts of the simple WalkSAT with a better variable selection:
            # Pick the variable in the selected clause that is involved in the most *currently unsatisfied* clauses.
            
            # To avoid O(m) per step, we can approximate or use a simpler heuristic.
            # Let's try: Pick the variable that is negated in the most unsat clauses? No.
            
            # Let's try a "MiniSAT" style approach is overkill.
            
            # Let's optimize the simple WalkSAT:
            # 1. Random clause.
            # 2. Random variable in that clause.
            # This is the seed. It got 1.0.
            
            # Why did it get 1.0? Maybe the instances are easy.
            # Let's try to improve robustness:
            # - More restarts if stuck.
            # - Better initial assignment? Random is fine.
            
            # Let's try: Weighted clause selection.
            # Clauses with fewer "free" variables (variables not appearing in other unsat clauses) are easier to satisfy.
            
            # Let's try a different heuristic: 
            # Pick the variable that, if flipped, satisfies the most clauses.
            
            # For the selected clause, for each var v in it:
            #   gain = number of unsat clauses containing v (excluding the selected one)
            #   loss = number of sat clauses containing v (excluding the selected one) - we don't track sat clauses easily.
            
            # Let's just stick to a robust WalkSAT with random restarts and a time limit.
            # The seed used 20000 steps. Let's use more steps per restart, but fewer restarts if time is tight?
            # Or many short restarts?
            
            # Let's try: 
            # While time < deadline:
            #   Start a new WalkSAT run with a random assignment.
            #   Run for max(1000, remaining_time / 10) steps.
            #   If solved, return.
            
            # Let's implement this outer loop.
            
            pass # Placeholder, we will restructure the function.
        
        return assignment # Return current assignment if not solved
    
    # Restructured solve:
    rng = random.Random(42)
    
    # Outer loop: Restart strategy
    # We'll run WalkSAT for a certain number of steps, then restart if not solved.
    # Steps per run: adaptive.
    
    # First, let's try a single long run with a good heuristic.
    # Heuristic: 
    # 1. Select unsat clause randomly.
    # 2. Select variable from that clause with probability proportional to its "badness" (number of unsat clauses it's in).
    
    # To compute badness efficiently, we can maintain an array unsat_count[var].
    # Initially, compute it.
    # On flip, update it.
    
    def compute_unsat_counts(assignment):
        counts = [0] * n
        unsat_indices = []
        for i, cd in enumerate(clause_data):
            sat = False
            for var, is_pos in cd:
                if assignment[var] == is_pos:
                    sat = True
                    break
            if not sat:
                unsat_indices.append(i)
                for var, _ in cd:
                    counts[var] += 1
        return counts, unsat_indices
    
    def flip_var(var, assignment, unsat_counts, unsat_indices):
        # Flip var
        old_val = assignment[var]
        assignment[var] = not old_val
        
        # Update unsat counts and unsat_indices
        # This is tricky to do incrementally without checking all clauses.
        # Given n=50, m is likely small (< 500).
        # Let's just recompute unsat_indices and unsat_counts from scratch if needed?
        # Or update incrementally.
        
        # Incremental update:
        # 1. Find all clauses containing var.
        # 2. For each such clause, check if it was sat/unsat before, and is sat/unsat now.
        # 3. Update counts accordingly.
        
        # This requires a map var -> list of clause indices.
        
        pass # We'll implement this in the main loop for clarity, or use a simpler approach.
    
    # Given the complexity of incremental updates in Python, and n=50 is small,
    # let's try a simpler approach: 
    # Just use the seed's logic but with a time limit and more steps.
    # The seed got 1.0, so it probably solves all instances.
    # The risk is timeout. The seed has a fixed 20000 steps.
    # If an instance is hard, 20000 steps might not be enough.
    # Let's use the time budget to do more steps.
    
    assignment = [rng.random() < 0.5 for _ in range(n)]
    
    # Precompute clause vars for faster access
    clause_vars = [[var for var, _ in cd] for cd in clause_data]
    
    # Main loop
    steps = 0
    while time.time() < deadline:
        unsat = [i for i, cd in enumerate(clause_data) if not any(assignment[var] == is_pos for var, is_pos in cd)]
        if not unsat:
            return assignment
        
        # Pick a random unsat clause
        clause_idx = rng.choice(unsat)
        cd = clause_data[clause_idx]
        
        # Pick a variable from this clause
        # Heuristic: Pick the variable that appears in the most unsat clauses.
        # To avoid O(m) cost, let's just pick randomly for now, but maybe weight by something simple?
        # Let's try: Pick the variable that is negated in the most unsat clauses? No.
        # Let's try: Random selection.
        var, is_pos = rng.choice(cd)
        
        # Flip
        assignment[var] = not assignment[var]
        
        steps += 1
        # Check time every 1000 steps to avoid overhead
        if steps % 1000 == 0 and time.time() > deadline:
            break
    
    return assignment
# EVOLVE-BLOCK-END
