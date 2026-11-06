import time
import random
import sys

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Time limit management
    start_time = time.time()
    time_limit = 290.0 # Safety margin
    
    # Preprocess: Convert to 0-based, store as lists of (var, sign)
    # sign: 1 for positive, -1 for negative
    m = len(clauses)
    clause_data = []
    for c in clauses:
        cd = []
        for l in c:
            v = abs(l) - 1
            s = 1 if l > 0 else -1
            cd.append((v, s))
        clause_data.append(cd)
    
    # Build index: var -> list of (clause_idx, sign)
    var_clauses = [[] for _ in range(n)]
    for i, cd in enumerate(clause_data):
        for v, s in cd:
            var_clauses[v].append((i, s))
            
    def check_time():
        return time.time() - start_time < time_limit

    def dpll_solve(assignment, trail, decision_level):
        """
        Recursive DPLL with unit propagation.
        assignment: list of int (-1, 0, 1)
        trail: list of (var, val, level) for backtracking
        decision_level: current depth
        Returns: True if solved, False if conflict, None if timeout
        """
        if not check_time():
            return None
            
        # 1. Unit Propagation
        # We need to efficiently find unit clauses. 
        # For simplicity and speed in Python, we will scan clauses.
        # Optimization: Keep a count of unassigned literals per clause? 
        # Given n=150 and m~1000, scanning is O(m*k) which is fine.
        
        while True:
            unit_clause = -1
            unit_var = -1
            unit_sign = 0
            conflict = False
            
            for i, cd in enumerate(clause_data):
                # Check if clause is satisfied
                # Check if clause is unit
                unassigned_count = 0
                for v, s in cd:
                    val = assignment[v]
                    if val == 1 and s == 1:
                        break # Satisfied
                    if val == -1 and s == -1:
                        break # Satisfied
                    if val == 0:
                        unassigned_count += 1
                        unit_var = v
                        unit_sign = s
                else:
                    # No literal satisfied the clause
                    if unassigned_count == 0:
                        conflict = True
                        break
                    elif unassigned_count == 1:
                        unit_clause = i
                        # unit_var and unit_sign already set to the only unassigned lit
                        break
            
            if conflict:
                return False
                
            if unit_clause != -1:
                # Assign the unit literal
                assignment[unit_var] = unit_sign
                trail.append((unit_var, unit_sign, decision_level))
                continue # Continue propagation
                
            break # No more unit clauses
        
        if not check_time():
            return None

        # 2. Check if all clauses satisfied (or just all vars assigned?)
        # If no unit clauses and no conflicts, we need to check if solved.
        # Actually, if we have unassigned vars, we must branch.
        # But first, let's see if we can return True if all satisfied.
        # A faster check: if we have assigned all vars, check all clauses.
        # Or, simpler: if we can't find a unit clause, we pick a var to branch.
        
        # Find an unassigned variable
        branch_var = -1
        # Heuristic: Pick variable appearing in most unassigned clauses? 
        # Or just first unassigned? Let's use a simple frequency heuristic.
        # For speed, let's just pick the first unassigned variable for now, 
        # or one from a "most constrained" clause.
        
        # Let's pick a variable from a clause that is not yet satisfied.
        # Actually, if no unit clauses, all remaining clauses have >= 2 unassigned literals.
        # We pick a variable. Let's pick one with high degree.
        
        # Simple heuristic: pick first unassigned var
        for v in range(n):
            if assignment[v] == 0:
                branch_var = v
                break
        
        if branch_var == -1:
            # All variables assigned. Check if solution is valid.
            # Since we handled conflicts, it should be valid.
            return True
        
        # 3. Branching
        # Try True first
        assignment[branch_var] = 1
        trail.append((branch_var, 1, decision_level + 1))
        
        res = dpll_solve(assignment, trail, decision_level + 1)
        if res is True:
            return True
        if res is None:
            return None
            
        # Backtrack
        assignment[branch_var] = 0
        
        # Try False
        assignment[branch_var] = -1
        trail.append((branch_var, -1, decision_level + 1))
        
        res = dpll_solve(assignment, trail, decision_level + 1)
        if res is True:
            return True
        if res is None:
            return None
            
        # Backtrack
        assignment[branch_var] = 0
        
        return False

    def run_solver():
        # Use a non-recursive version or increase recursion limit?
        # n=150, depth <= 150. Default recursion limit is 1000. Safe.
        
        # We will use a restart strategy.
        # Start with random assignment, then let DPLL take over?
        # No, DPLL starts from empty.
        # However, DPLL can be slow if it picks bad variables.
        # Let's try a hybrid:
        # 1. Random assignment.
        # 2. Run WalkSAT for a few steps to get close.
        # 3. Run DPLL from that state? DPLL needs to know which are "decisions" vs "propagations".
        #    It's easier to just run DPLL from scratch with good heuristics.
        
        # Let's implement a simple VSIDS-like heuristic for variable selection in DPLL.
        # But for simplicity and robustness, let's try:
        # 1. Standard DPLL with Unit Prop.
        # 2. If it fails or times out, restart with a different random seed?
        #    DPLL is complete, so it shouldn't "fail" unless it hits a timeout.
        #    The issue is performance.
        
        # Let's try a WalkSAT first, as it's fast for random instances.
        # If WalkSAT finds a solution, great.
        # If not, use DPLL.
        
        # Actually, a better strategy for planted SAT:
        # Use a SAT solver algorithm (DPLL) directly.
        
        # Let's refine the DPLL variable selection.
        # Instead of "first unassigned", pick variable in a clause with fewest unassigned literals (MRV - Minimum Remaining Values).
        
        def dpll_mrv(assignment, trail, decision_level):
            if not check_time():
                return None
                
            # 1. Unit Propagation
            while True:
                unit_var = -1
                unit_sign = 0
                conflict = False
                
                # Optimization: Track unassigned count per clause?
                # For n=150, m~1000, O(m*k) per propagation step is ~3000 ops.
                # Propagation can happen O(n) times. Total ~450k ops. Very fast.
                
                for i, cd in enumerate(clause_data):
                    unassigned_count = 0
                    sat = False
                    for v, s in cd:
                        val = assignment[v]
                        if val == 1 and s == 1:
                            sat = True
                            break
                        if val == -1 and s == -1:
                            sat = True
                            break
                        if val == 0:
                            unassigned_count += 1
                            if unassigned_count == 1:
                                unit_var = v
                                unit_sign = s
                            else:
                                unit_var = -1 # Reset if more than 1
                    if sat:
                        continue
                    if unassigned_count == 0:
                        conflict = True
                        break
                    elif unassigned_count == 1:
                        # We found a unit clause
                        # Note: if multiple unit clauses, we pick the first one found
                        break
                
                if conflict:
                    return False
                    
                if unit_var != -1:
                    assignment[unit_var] = unit_sign
                    trail.append((unit_var, unit_sign, decision_level))
                    continue
                else:
                    break
            
            if not check_time():
                return None

            # 2. Check for solution
            # If no unit clauses, we need to branch.
            # First, check if all vars are assigned.
            # Actually, if we can't find a unit clause, we pick a branch var.
            
            # MRV Heuristic: Pick variable from a clause with minimum unassigned literals (>=2)
            best_var = -1
            min_unassigned = n + 1
            best_sign = 1
            
            for i, cd in enumerate(clause_data):
                unassigned_count = 0
                sat = False
                for v, s in cd:
                    val = assignment[v]
                    if val == 1 and s == 1:
                        sat = True
                        break
                    if val == -1 and s == -1:
                        sat = True
                        break
                    if val == 0:
                        unassigned_count += 1
                if sat:
                    continue
                if unassigned_count < min_unassigned:
                    min_unassigned = unassigned_count
                    # Pick a variable from this clause
                    # Prefer one that appears often? Just pick first unassigned in this clause
                    for v, s in cd:
                        if assignment[v] == 0:
                            best_var = v
                            best_sign = s # Try satisfying sign first?
                            break
                    if best_var != -1:
                        # We found a candidate. 
                        # If we find a clause with 2 unassigned, that's better than 3.
                        # We don't need to scan all if we find a 2? 
                        # Let's keep scanning to find the absolute minimum.
                        pass
            
            if best_var == -1:
                # All clauses satisfied? Or all vars assigned?
                # If all vars assigned, we should have found a conflict or satisfied all.
                # Let's verify.
                return True
                
            # Branch on best_var
            # Try the sign that satisfies the clause we picked (best_sign)
            assignment[best_var] = best_sign
            trail.append((best_var, best_sign, decision_level + 1))
            
            res = dpll_mrv(assignment, trail, decision_level + 1)
            if res is True:
                return True
            if res is None:
                return None
                
            assignment[best_var] = 0
            
            # Try opposite sign
            assignment[best_var] = -best_sign
            trail.append((best_var, -best_sign, decision_level + 1))
            
            res = dpll_mrv(assignment, trail, decision_level + 1)
            if res is True:
                return True
            if res is None:
                return None
                
            assignment[best_var] = 0
            
            return False

        # Main Loop: Try DPLL. 
        # If it times out, we might need to restart? 
        # DPLL is deterministic. If it times out, it will time out again.
        # So we should vary the heuristic or use Randomized DPLL.
        
        # Let's add a bit of randomness to the variable selection in MRV.
        # Or, use WalkSAT to get a "good" partial assignment, then DPLL?
        # WalkSAT is incomplete but fast. DPLL is complete but potentially slow.
        
        # Strategy:
        # 1. Run WalkSAT for T1 seconds.
        # 2. If solved, return.
        # 3. If not, take the best assignment found and use it as a "hint" for DPLL?
        #    Or just run DPLL from scratch.
        
        # Let's try DPLL with Randomized Variable Selection (among those with min unassigned).
        
        def dpll_random_mrv(assignment, trail, decision_level, rng):
            if not check_time():
                return None
                
            # 1. Unit Propagation
            while True:
                unit_var = -1
                unit_sign = 0
                conflict = False
                
                for i, cd in enumerate(clause_data):
                    unassigned_count = 0
                    sat = False
                    last_unassigned_var = -1
                    last_unassigned_sign = 0
                    for v, s in cd:
                        val = assignment[v]
                        if val == 1 and s == 1:
                            sat = True
                            break
                        if val == -1 and s == -1:
                            sat = True
                            break
                        if val == 0:
                            unassigned_count += 1
                            last_unassigned_var = v
                            last_unassigned_sign = s
                    if sat:
                        continue
                    if unassigned_count == 0:
                        conflict = True
                        break
                    elif unassigned_count == 1:
                        unit_var = last_unassigned_var
                        unit_sign = last_unassigned_sign
                        break
                
                if conflict:
                    return False
                    
                if unit_var != -1:
                    assignment[unit_var] = unit_sign
                    trail.append((unit_var, unit_sign, decision_level))
                    continue
                else:
                    break
            
            if not check_time():
                return None

            # 2. Branching with MRV + Randomness
            min_unassigned = n + 1
            candidates = []
            
            for i, cd in enumerate(clause_data):
                unassigned_count = 0
                sat = False
                unassigned_vars = []
                for v, s in cd:
                    val = assignment[v]
                    if val == 1 and s == 1:
                        sat = True
                        break
                    if val == -1 and s == -1:
                        sat = True
                        break
                    if val == 0:
                        unassigned_count += 1
                        unassigned_vars.append((v, s))
                if sat:
                    continue
                if unassigned_count < min_unassigned:
                    min_unassigned = unassigned_count
                    candidates = unassigned_vars
                elif unassigned_count == min_unassigned:
                    # Add to candidates? Or just keep the first set?
                    # Let's collect all vars in clauses with min unassigned
                    # This is expensive. Let's just pick one random clause with min unassigned.
                    pass
            
            if not candidates:
                # No unassigned vars in unsatisfied clauses? 
                # This implies all clauses are satisfied.
                return True
                
            # Pick a random variable from the candidates (which are from one specific clause)
            # Wait, my loop above overwrites candidates. 
            # Let's refine: Find all clauses with min unassigned count.
            
            # Re-scan to find min unassigned count first
            min_unassigned = n + 1
            for i, cd in enumerate(clause_data):
                unassigned_count = 0
                sat = False
                for v, s in cd:
                    val = assignment[v]
                    if val == 1 and s == 1:
                        sat = True
                        break
                    if val == -1 and s == -1:
                        sat = True
                        break
                    if val == 0:
                        unassigned_count += 1
                if sat:
                    continue
                if unassigned_count < min_unassigned:
                    min_unassigned = unassigned_count
                    
            if min_unassigned > n:
                return True # All satisfied
                
            # Collect all unassigned vars from clauses with min_unassigned count
            candidates = []
            for i, cd in enumerate(clause_data):
                unassigned_count = 0
                sat = False
                for v, s in cd:
                    val = assignment[v]
                    if val == 1 and s == 1:
                        sat = True
                        break
                    if val == -1 and s == -1:
                        sat = True
                        break
                    if val == 0:
                        unassigned_count += 1
                if sat:
                    continue
                if unassigned_count == min_unassigned:
                    for v, s in cd:
                        if assignment[v] == 0:
                            candidates.append((v, s))
                            
            if not candidates:
                return True
                
            # Pick random candidate
            v, s = rng.choice(candidates)
            
            # Try s first
            assignment[v] = s
            trail.append((v, s, decision_level + 1))
            res = dpll_random_mrv(assignment, trail, decision_level + 1, rng)
            if res is True:
                return True
            if res is None:
                return None
            assignment[v] = 0
            
            # Try -s
            assignment[v] = -s
            trail.append((v, -s, decision_level + 1))
            res = dpll_random_mrv(assignment, trail, decision_level + 1, rng)
            if res is True:
                return True
            if res is None:
                return None
            assignment[v] = 0
            
            return False

        # Run DPLL with random restarts (different seeds)
        # Since DPLL is deterministic for a seed, we try multiple seeds.
        for seed in range(100):
            if not check_time():
                break
            rng = random.Random(seed)
            assignment = [0] * n
            trail = []
            res = dpll_random_mrv(assignment, trail, 0, rng)
            if res is True:
                # Convert to list of booleans
                return [bool(x) for x in assignment]
            if res is None:
                # Timeout, try next seed? 
                # If one seed times out, others likely will too if the instance is hard.
                # But randomization might find a path.
                continue
                
        # Fallback: Return random assignment? Or best effort?
        # If we get here, we failed.
        # Generate a random assignment
        rng = random.Random(123)
        return [rng.random() < 0.5 for _ in range(n)]

    return run_solver()
