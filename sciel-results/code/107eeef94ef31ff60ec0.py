import time
import numpy as np
from collections import defaultdict

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    if m == 0:
        return [True] * n

    start_time = time.time()
    # The evaluator kills at 330s. We set a strict internal deadline to return safely.
    # We allocate 320s to leave a 10s buffer for Python overhead and return.
    deadline = start_time + 320.0

    # Convert to 0-based indices for internal processing
    # literals are signed, 1-based
    # 0 is unused in SAT, so we just keep them as is but map var index - 1
    # For DPLL, we need efficient access.
    
    # Preprocess: Remove tautologies, duplicate literals in clauses, empty clauses
    processed_clauses = []
    for c in clauses:
        seen = set()
        new_c = []
        has_pos = False
        has_neg = False
        for l in c:
            v = abs(l)
            if v in seen:
                continue
            seen.add(v)
            if l > 0:
                has_pos = True
            else:
                has_neg = True
            new_c.append(l)
        if has_pos and has_neg:
            continue # Tautology
        if len(new_c) == 0:
            return [] # UNSAT (should not happen for planted satisfiable)
        processed_clauses.append(tuple(new_c))
    
    clauses = processed_clauses
    m = len(clauses)
    
    # Build variable index
    # var_to_clauses[v] = list of (clause_idx, sign)
    var_to_clauses = defaultdict(list)
    for i, c in enumerate(clauses):
        for l in c:
            v = abs(l) - 1
            sign = 1 if l > 0 else 0
            var_to_clauses[v].append((i, sign))

    # 1. DPLL with Unit Propagation
    # We use a stack-based or recursive approach. Given n=150, recursion depth is manageable.
    # However, to be safe and allow timeouts, we implement a bounded DPLL.
    
    # Heuristic: Pick variable with highest frequency in unsatisfied clauses? 
    # Or simply first unassigned variable with highest degree.
    
    # Let's implement a simple but effective DPLL.
    # State: assignment (array of -1, 0, 1), watchlists or just iterate clauses.
    # For n=150, iterating all m clauses per step is O(m). m is likely ~1000-2000.
    # 150 * 2000 = 300k ops per full path. Very fast.
    
    assignment = np.full(n, -1, dtype=np.int8)
    
    def unit_propagate(assignment):
        """
        Returns False if conflict (empty clause).
        Updates assignment in place.
        """
        changed = True
        while changed:
            changed = False
            for i, c in enumerate(clauses):
                num_unassigned = 0
                num_false = 0
                last_unassigned = -1
                last_val = 0
                for l in c:
                    v = abs(l) - 1
                    sign = 1 if l > 0 else 0
                    val = assignment[v]
                    if val == -1:
                        num_unassigned += 1
                        last_unassigned = v
                        last_val = sign
                    elif val != sign:
                        num_false += 1
                if num_false == len(c):
                    return False # Conflict
                if num_unassigned == 1:
                    assignment[last_unassigned] = last_val
                    changed = True
        return True

    # Initial Unit Propagation
    if not unit_propagate(assignment):
        # If initial UP leads to conflict, the instance might be hard or my UP is too aggressive?
        # Planted SAT should be satisfiable. If UP finds conflict, it's UNSAT.
        # But task says satisfiable. So this shouldn't happen unless bug.
        # Let's reset and try local search immediately if DPLL fails early.
        pass

    def dpll(assignment, depth):
        """
        Recursive DPLL.
        Returns True if satisfied, False otherwise.
        """
        if time.time() > deadline - 10.0:
            return False # Timeout, abort DPLL, switch to local search
        
        # Check if solved
        for c in clauses:
            sat = False
            for l in c:
                v = abs(l) - 1
                sign = 1 if l > 0 else 0
                if assignment[v] == sign:
                    sat = True
                    break
            if not sat:
                break
        else:
            return True # All clauses satisfied
            
        # Unit Propagation
        if not unit_propagate(assignment):
            return False
            
        # Re-check if solved after UP
        for c in clauses:
            sat = False
            for l in c:
                v = abs(l) - 1
                sign = 1 if l > 0 else 0
                if assignment[v] == sign:
                    sat = True
                    break
            if not sat:
                break
        else:
            return True

        # Variable Selection Heuristic
        # Pick unassigned variable with highest occurrence count in unsatisfied clauses
        # Or simply first unassigned variable.
        # Let's use a simple heuristic: highest degree among unassigned.
        unassigned_vars = [v for v in range(n) if assignment[v] == -1]
        if not unassigned_vars:
            return False # All assigned but not satisfied? Should be caught by UP or check.
            
        # Calculate score for each unassigned variable
        # Score = number of unsatisfied clauses it appears in
        unsat_clauses = []
        for i, c in enumerate(clauses):
            is_sat = False
            for l in c:
                v = abs(l) - 1
                sign = 1 if l > 0 else 0
                if assignment[v] == sign:
                    is_sat = True
                    break
            if not is_sat:
                unsat_clauses.append(c)
                
        if not unsat_clauses:
            return True # All satisfied
            
        var_scores = defaultdict(int)
        for c in unsat_clauses:
            for l in c:
                v = abs(l) - 1
                if assignment[v] == -1:
                    var_scores[v] += 1
                    
        if not var_scores:
            return False # All variables in unsat clauses are assigned but don't satisfy? Contradiction.
            
        # Pick variable with max score
        best_var = max(var_scores, key=var_scores.get)
        
        # Try True (1)
        assignment[best_var] = 1
        if dpll(assignment, depth + 1):
            return True
        assignment[best_var] = 0
        if dpll(assignment, depth + 1):
            return True
        assignment[best_var] = -1
        return False

    # Run DPLL with a time limit
    # We give DPLL a few seconds to try to solve it.
    dpll_deadline = start_time + 5.0 
    old_deadline = deadline
    deadline = dpll_deadline
    
    # Reset assignment to -1 for clean DPLL start
    assignment = np.full(n, -1, dtype=np.int8)
    
    solved = False
    try:
        solved = dpll(assignment, 0)
    except RecursionError:
        pass
    
    deadline = old_deadline # Restore deadline for local search
    
    if solved:
        return assignment.astype(bool).tolist()
        
    # 2. Local Search (WalkSAT / Min-Conflict Hybrid)
    # If DPLL fails, use local search.
    # Initialize randomly or using the partial assignment from DPLL if helpful?
    # Random is usually fine for planted SAT.
    
    rng = np.random.default_rng(42)
    
    # Precompute for fast local search
    # var_idx: [m, 3] 0-based
    # sign: [m, 3] 1 if positive, 0 if negative
    lits = np.empty((m, 3), dtype=np.int32)
    for i, c in enumerate(clauses):
        for j, l in enumerate(c):
            lits[i, j] = l
            
    var_idx = np.abs(lits) - 1
    sign = (lits > 0).astype(np.int32)
    
    # Variable -> (clauses, positions)
    var_clauses = [[] for _ in range(n)]
    var_pos = [[] for _ in range(n)]
    for i in range(m):
        for j in range(3):
            v = int(var_idx[i, j])
            var_clauses[v].append(i)
            var_pos[v].append(j)
            
    var_clauses_np = [np.array(vc, dtype=np.int32) if vc else np.empty(0, dtype=np.int32) for vc in var_clauses]
    var_pos_np = [np.array(vp, dtype=np.int32) if vp else np.empty(0, dtype=np.int32) for vp in var_pos]
    var_signs = [np.array([sign[c, p] for c, p in zip(var_clauses[v], var_pos[v])], dtype=np.int32) if var_clauses[v] else np.empty(0, dtype=np.int32) for v in range(n)]

    def init_assignment():
        return (rng.random(n) < 0.5).astype(np.int8)

    def init_num_true(assignment):
        satisfied = (assignment[var_idx] == sign)
        return satisfied.sum(axis=1).astype(np.int32)

    def update_var(assignment, num_true, v, new_val):
        c_list = var_clauses_np[v]
        p_list = var_pos_np[v]
        if c_list.size == 0:
            return
        s = sign[c_list, p_list]
        now_sat = (new_val == s)
        delta = (2 * now_sat.astype(np.int32)) - 1
        num_true[c_list] += delta

    best_assign = None
    best_unsat = float('inf')
    
    max_restarts = 1000
    base_steps = 2000
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
            
        assignment = init_assignment()
        num_true = init_num_true(assignment)
        
        if num_true.min() > 0:
            return assignment.astype(bool).tolist()
            
        current_unsat = int((num_true == 0).sum())
        if current_unsat < best_unsat:
            best_unsat = current_unsat
            best_assign = assignment.copy()
            
        # WalkSAT Loop
        for step in range(base_steps):
            if time.time() > deadline - 5.0:
                break
                
            unsat_mask = (num_true == 0)
            if not unsat_mask.any():
                return assignment.astype(bool).tolist()
                
            current_unsat = int(unsat_mask.sum())
            if current_unsat < best_unsat:
                best_unsat = current_unsat
                best_assign = assignment.copy()
                
            # Pick an unsatisfied clause
            unsat_idx = np.where(unsat_mask)[0]
            c = int(unsat_idx[rng.integers(unsat_idx.size)])
            
            # Pick a variable from this clause
            # Heuristic: 
            # 1. With probability p (e.g., 0.5), pick variable that minimizes total unsatisfied clauses after flip (Min-Conflict)
            # 2. With probability 1-p, pick random variable (WalkSAT)
            
            cands = var_idx[c]
            
            if rng.random() < 0.6:
                # Min-Conflict style: Calculate delta for each candidate
                best_v = -1
                best_delta = float('inf')
                for v in cands:
                    v = int(v)
                    new_val = 1 - int(assignment[v])
                    # Calculate how many unsat clauses this flip fixes vs breaks
                    c_list = var_clauses_np[v]
                    s = var_signs[v]
                    cur_val = assignment[v]
                    
                    # Gain: unsat clauses containing v that become sat
                    unsat_c = (num_true[c_list] == 0)
                    v_satisfies_new = (new_val == s)
                    gain = (unsat_c & v_satisfies_new).sum()
                    
                    # Loss: sat clauses (num_true==1) containing v that become unsat
                    # If v currently satisfies it, and num_true is 1, flipping breaks it
                    was_v_sat = (cur_val == s)
                    critical = (num_true[c_list] == 1)
                    loss = (was_v_sat & critical).sum()
                    
                    delta = loss - gain
                    if delta < best_delta:
                        best_delta = delta
                        best_v = v
                v = best_v
            else:
                # Random
                v = int(cands[rng.integers(3)])
                
            new_val = 1 - int(assignment[v])
            assignment[v] = new_val
            update_var(assignment, num_true, v, new_val)

    if best_assign is not None:
        return best_assign.astype(bool).tolist()
        
    return init_assignment().astype(bool).tolist()
