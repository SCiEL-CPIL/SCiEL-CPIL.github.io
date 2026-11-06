"""Synthetic planted 3-SAT solver (50 variables) with GF(2) algebraic approach + local search."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np
from collections import deque

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    # Deadline guard: 325s wall-clock, leave 5s margin
    deadline = time.time() + 325.0
    
    # Preprocess: convert clauses to 0-based indices and store signs
    clause_vars = np.zeros((m, 3), dtype=np.int32)
    clause_signs = np.zeros((m, 3), dtype=bool)
    for i, c in enumerate(clauses):
        for j, lit in enumerate(c):
            clause_vars[i, j] = abs(lit) - 1
            clause_signs[i, j] = lit > 0
    
    # Build reverse mapping: variable -> list of (clause_idx, sign)
    var_to_clauses = [[] for _ in range(n)]
    for ci in range(m):
        for j in range(3):
            v = int(clause_vars[ci, j])
            s = bool(clause_signs[ci, j])
            var_to_clauses[v].append((ci, s))
    
    # Build lookup tables for fast unsat counting
    var_clause_indices = [np.array([ci for ci, s in var_to_clauses[v]], dtype=np.int32) for v in range(n)]
    var_clause_signs = [np.array([s for ci, s in var_to_clauses[v]], dtype=bool) for v in range(n)]
    
    def is_satisfied(assignment):
        """Check if assignment satisfies all clauses using vectorized ops."""
        assignment_arr = np.array(assignment, dtype=bool)
        lit_sat = assignment_arr[clause_vars] == clause_signs
        clause_sat = lit_sat.any(axis=1)
        return bool(clause_sat.all())
    
    def count_unsat(assignment):
        """Count number of unsatisfied clauses."""
        assignment_arr = np.array(assignment, dtype=bool)
        lit_sat = assignment_arr[clause_vars] == clause_signs
        clause_sat = lit_sat.any(axis=1)
        return int((~clause_sat).sum())
    
    def fast_count_unsat(assignment):
        """Fast unsat count. Returns (count, unsat_indices)."""
        assignment_arr = np.array(assignment, dtype=bool)
        lit_falsified = assignment_arr[clause_vars] != clause_signs
        clause_unsat = lit_falsified.all(axis=1)
        unsat_idx = np.where(clause_unsat)[0]
        return int(len(unsat_idx)), unsat_idx
    
    def dpll_solve(assignment, time_limit):
        """
        DPLL solver with unit propagation and pure literal elimination.
        Returns a satisfying assignment if found within time_limit, else None.
        """
        dpll_assign = [None] * n
        
        def propagate(assign, active_clauses):
            """
            Perform unit propagation and pure literal elimination.
            Returns (success, new_active_clauses)
            success is False if a contradiction is found.
            """
            changed = True
            while changed:
                changed = False
                
                # Unit propagation
                new_active = []
                for ci in active_clauses:
                    vars_ = clause_vars[ci]
                    signs = clause_signs[ci]
                    
                    unassigned = []
                    satisfied = False
                    for j in range(3):
                        v = int(vars_[j])
                        s = bool(signs[j])
                        if assign[v] is None:
                            unassigned.append((v, s))
                        elif assign[v] == s:
                            satisfied = True
                            break
                    
                    if satisfied:
                        continue
                    elif len(unassigned) == 0:
                        return False, active_clauses
                    elif len(unassigned) == 1:
                        v, s = unassigned[0]
                        if assign[v] is not None:
                            continue
                        assign[v] = s
                        changed = True
                    else:
                        new_active.append(ci)
                
                active_clauses = new_active
                
                # Pure literal elimination
                if active_clauses:
                    pure_vars = {}
                    for ci in active_clauses:
                        vars_ = clause_vars[ci]
                        signs = clause_signs[ci]
                        for j in range(3):
                            v = int(vars_[j])
                            s = bool(signs[j])
                            if assign[v] is None:
                                if v not in pure_vars:
                                    pure_vars[v] = set()
                                pure_vars[v].add(s)
                    
                    for v, signs_set in pure_vars.items():
                        if len(signs_set) == 1:
                            s = signs_set.pop()
                            assign[v] = s
                            changed = True
                    
                    if changed:
                        new_active = []
                        for ci in active_clauses:
                            vars_ = clause_vars[ci]
                            signs = clause_signs[ci]
                            satisfied = False
                            for j in range(3):
                                v = int(vars_[j])
                                s = bool(signs[j])
                                if assign[v] is not None:
                                    if assign[v] == s:
                                        satisfied = True
                                        break
                            if not satisfied:
                                new_active.append(ci)
                        active_clauses = new_active
            
            return True, active_clauses
        
        # Initial propagation
        success, active_clauses = propagate(dpll_assign, list(range(m)))
        
        if not success:
            return None
        
        if not active_clauses:
            for v in range(n):
                if dpll_assign[v] is None:
                    dpll_assign[v] = assignment[v] if assignment[v] is not None else False
            return dpll_assign
        
        def dpll_recurse(current_assign, active_clauses, depth=0):
            if time.time() > time_limit:
                return None
            
            if depth > 60:
                return None
            
            # Propagate
            success, active_clauses = propagate(current_assign, active_clauses)
            if not success:
                return None
            
            if not active_clauses:
                for v in range(n):
                    if current_assign[v] is None:
                        current_assign[v] = assignment[v] if assignment[v] is not None else False
                return current_assign[:]
            
            # Pick branch variable using activity heuristic
            var_count = {}
            for ci in active_clauses:
                vars_ = clause_vars[ci]
                for j in range(3):
                    v = int(vars_[j])
                    if current_assign[v] is None:
                        var_count[v] = var_count.get(v, 0) + 1
            
            if not var_count:
                for v in range(n):
                    if current_assign[v] is None:
                        current_assign[v] = assignment[v] if assignment[v] is not None else False
                return current_assign[:]
            
            branch_var = max(var_count, key=var_count.get)
            
            # Try both values, prefer the one that satisfies more clauses
            pos_sat = 0
            neg_sat = 0
            for ci in active_clauses:
                vars_ = clause_vars[ci]
                signs = clause_signs[ci]
                for j in range(3):
                    if int(vars_[j]) == branch_var:
                        if bool(signs[j]):
                            pos_sat += 1
                        else:
                            neg_sat += 1
            
            if pos_sat >= neg_sat:
                order = [True, False]
            else:
                order = [False, True]
            
            for val in order:
                if time.time() > time_limit:
                    return None
                
                saved_assign = current_assign[:]
                current_assign[branch_var] = val
                
                success, new_active = propagate(current_assign, active_clauses)
                
                if success:
                    if not new_active:
                        for v in range(n):
                            if current_assign[v] is None:
                                current_assign[v] = assignment[v] if assignment[v] is not None else False
                        return current_assign[:]
                    else:
                        result = dpll_recurse(current_assign, new_active, depth + 1)
                        if result is not None:
                            return result
                
                # Restore
                current_assign = saved_assign
            
            return None
        
        return dpll_recurse(dpll_assign, active_clauses)
    
    def gf2_solve():
        """
        Attempt to solve using GF(2) algebraic approach.
        Maps each clause to a polynomial constraint over GF(2).
        For a clause (l1 ∨ l2 ∨ l3), we use the identity:
        l1 ∨ l2 ∨ l3 = 1 - (1-l1)(1-l2)(1-l3)
        Over GF(2), with x_i = 1 if variable i is True, 0 if False:
        A literal x (positive) maps to x_i, ¬x (negative) maps to 1-x_i.
        The clause is satisfied iff the product (1-l1)(1-l2)(1-l3) = 0.
        We can write this as a polynomial equation = 0.
        However, this gives nonlinear equations. For planted instances,
        we can try to find the planted assignment by looking at the structure.
        
        Alternative: Use the fact that for a planted instance, there's a unique
        or few satisfying assignments. We can set up a system where we try to
        recover the planted assignment by solving linear constraints derived
        from the clause structure.
        
        A simpler algebraic approach: For each clause, at least one literal must
        be true. We can write this as a constraint. For planted instances with
        a specific structure, we can often recover the assignment by:
        1. Identifying variables that appear in many clauses
        2. Using the clause structure to determine their values
        
        Let's try a different approach: use the fact that in a planted instance,
        the assignment satisfies all clauses. We can set up a system of equations
        where for each clause, we require that not all literals are false.
        
        Over GF(2), let's define for each variable x_i ∈ {0,1}.
        For a clause (a, b, c) where each is a signed literal:
        The clause is unsatisfied iff all three literals are false.
        Literal l is false iff:
          - If l = x_i (positive): x_i = 0
          - If l = ¬x_i (negative): x_i = 1
        
        We want to find x such that for all clauses, not all literals are false.
        This is a 3-CNF formula, which is NP-hard in general, but for planted
        instances with specific structure, we might be able to solve it.
        
        Let's try a heuristic algebraic approach:
        For each variable, count how many times it appears positively and negatively.
        In a planted instance, the planted assignment should make most clauses satisfied.
        
        Actually, let's try a more direct approach: use Gaussian elimination on a
        linear system derived from the clause structure. For each clause, we can
        write a linear constraint that is satisfied by the planted assignment.
        
        For a clause (l1, l2, l3), the constraint is:
        l1 + l2 + l3 >= 1 (in integer arithmetic)
        
        Over GF(2), this doesn't directly translate. But we can use the following:
        For each clause, at least one literal is true. This means:
        If we define y_i = x_i for positive literals and y_i = 1-x_i for negative,
        then for each clause: y_1 + y_2 + y_3 >= 1.
        
        This is a covering problem. For planted instances, we can try to find
        the assignment that maximizes the number of satisfied clauses, which
        should be all of them.
        
        Let's try a simpler approach: use the structure of the planted instance.
        In many planted 3-SAT instances, the assignment can be recovered by
        looking at the parity of literal occurrences.
        
        For each variable, compute the sum of signs of its occurrences.
        If the planted assignment is x_i, then the expected contribution to
        clause satisfaction is related to x_i.
        
        Let's try: for each variable v, let pos_count[v] be the number of clauses
        where v appears positively, and neg_count[v] be the number where v appears
        negatively. In the planted assignment, if x_v = True, then pos_count[v]
        clauses have this literal satisfied, and if x_v = False, neg_count[v] clauses
        have this literal satisfied.
        
        We want to choose x_v to maximize the total number of satisfied clauses.
        This is a max-cut-like problem, which is NP-hard, but for small n=50,
        we can use a greedy or local search approach.
        
        Let's use a simple greedy approach:
        1. Start with all variables False
        2. For each variable, try setting it to True and see if it increases
           the number of satisfied clauses
        3. Repeat until no improvement
        """
        # Count positive and negative occurrences
        pos_count = np.zeros(n, dtype=np.int32)
        neg_count = np.zeros(n, dtype=np.int32)
        for ci in range(m):
            for j in range(3):
                v = int(clause_vars[ci, j])
                if bool(clause_signs[ci, j]):
                    pos_count[v] += 1
                else:
                    neg_count[v] += 1
        
        # Greedy assignment: start with all False, flip variables that improve
        assignment = [False] * n
        
        def eval_assignment(assignment):
            assignment_arr = np.array(assignment, dtype=bool)
            lit_sat = assignment_arr[clause_vars] == clause_signs
            clause_sat = lit_sat.any(axis=1)
            return int(clause_sat.sum())
        
        best_score = eval_assignment(assignment)
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for v in range(n):
                old_val = assignment[v]
                assignment[v] = not old_val
                new_score = eval_assignment(assignment)
                if new_score > best_score:
                    best_score = new_score
                    improved = True
                else:
                    assignment[v] = old_val
        
        if best_score == m:
            return assignment
        
        # Try starting with all True
        assignment = [True] * n
        best_score = eval_assignment(assignment)
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for v in range(n):
                old_val = assignment[v]
                assignment[v] = not old_val
                new_score = eval_assignment(assignment)
                if new_score > best_score:
                    best_score = new_score
                    improved = True
                else:
                    assignment[v] = old_val
        
        if best_score == m:
            return assignment
        
        # Try a randomized approach: for each variable, set it based on the
        # majority of its literal occurrences
        rng_gf2 = random.Random(42)
        for trial in range(10):
            if time.time() > deadline - 10:
                break
            assignment = []
            for v in range(n):
                # Set variable based on occurrence count with some noise
                total = pos_count[v] + neg_count[v]
                if total == 0:
                    assignment.append(rng_gf2.random() < 0.5)
                else:
                    # Probability of being True proportional to positive occurrences
                    p = pos_count[v] / total
                    # Add some noise
                    p = 0.5 + 0.5 * (p - 0.5) * (1 - 0.1 * trial)
                    assignment.append(rng_gf2.random() < p)
            
            # Local search from this starting point
            best_score = eval_assignment(assignment)
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                for v in range(n):
                    old_val = assignment[v]
                    assignment[v] = not old_val
                    new_score = eval_assignment(assignment)
                    if new_score > best_score:
                        best_score = new_score
                        improved = True
                    else:
                        assignment[v] = old_val
            
            if best_score == m:
                return assignment
        
        return None
    
    best_assignment = None
    best_unsat = float('inf')
    
    rng = random.Random()
    
    # Strategy:
    # 1. Try GF(2) algebraic approach first (fast, might work for planted instances)
    # 2. Try DPLL with a time limit
    # 3. Use WalkSAT with min-unsat heuristic and restarts
    # 4. Use DPLL as a polish step for near-satisfying assignments
    
    # Phase 0: GF(2) algebraic approach
    gf2_result = gf2_solve()
    if gf2_result is not None:
        if is_satisfied(gf2_result):
            return gf2_result
    
    # Phase 1: DPLL with time limit
    dpll_deadline = time.time() + 30.0
    dpll_result = dpll_solve([False] * n, dpll_deadline)
    if dpll_result is not None:
        if is_satisfied(dpll_result):
            return dpll_result
    
    # Phase 2: WalkSAT with min-unsat heuristic and restarts
    def walksat_step(assignment, noise=0.05):
        """
        One WalkSAT step with min-unsat heuristic.
        Returns the new assignment.
        """
        assignment_arr = np.array(assignment, dtype=bool)
        
        # Find unsatisfied clauses
        lit_falsified = assignment_arr[clause_vars] != clause_signs
        clause_unsat = lit_falsified.all(axis=1)
        unsat_idx = np.where(clause_unsat)[0]
        
        if len(unsat_idx) == 0:
            return assignment, 0
        
        # Pick a random unsatisfied clause
        clause_idx = int(rng.choice(unsat_idx))
        vars_ = clause_vars[clause_idx]
        signs = clause_signs[clause_idx]
        
        # Find false literals in this clause
        false_mask = assignment_arr[vars_] != signs
        false_vars = vars_[false_mask]
        
        if len(false_vars) == 0:
            v = int(rng.choice(vars_))
        else:
            best_delta = float('inf')
            best_var = int(false_vars[0])
            candidates = false_vars.tolist()
            
            for v in candidates:
                affected = var_clause_indices[v]
                
                if len(affected) == 0:
                    delta = 0
                else:
                    cur_lit_fals = assignment_arr[clause_vars[affected]] != clause_signs[affected]
                    cur_unsat = cur_lit_fals.all(axis=1)
                    cur_unsat_count = int(cur_unsat.sum())
                    
                    new_assign = assignment_arr.copy()
                    new_assign[v] = not new_assign[v]
                    new_lit_fals = new_assign[clause_vars[affected]] != clause_signs[affected]
                    new_unsat = new_lit_fals.all(axis=1)
                    new_unsat_count = int(new_unsat.sum())
                    
                    delta = new_unsat_count - cur_unsat_count
                
                if delta < best_delta:
                    best_delta = delta
                    best_var = v
                elif delta == best_delta and rng.random() < noise:
                    best_var = v
            
            v = best_var
        
        assignment[v] = not assignment[v]
        return assignment, len(unsat_idx)
    
    # WalkSAT with adaptive noise and restarts
    max_restarts = 200
    base_steps = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline - 2.0:
            break
        
        # Random initial assignment
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        # Adaptive noise: start high, decrease over restarts
        noise = 0.1 * (0.95 ** restart) + 0.01
        
        # Run WalkSAT
        for step in range(base_steps):
            if time.time() > deadline - 2.0:
                break
            
            assignment, unsat_count = walksat_step(assignment, noise)
            
            if unsat_count == 0:
                return assignment
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment[:]
        
        # If we got close, try more steps with lower noise
        if best_unsat <= 3:
            extra_steps = 2000
            for step in range(extra_steps):
                if time.time() > deadline - 2.0:
                    break
                assignment, unsat_count = walksat_step(assignment, 0.01)
                if unsat_count == 0:
                    return assignment
                if unsat_count < best_unsat:
                    best_unsat = unsat_count
                    best_assignment = assignment[:]
    
    # Phase 3: If we have a near-satisfying assignment, try DPLL polish
    if best_assignment is not None and best_unsat <= 5:
        polish_deadline = time.time() + 20.0
        polish_result = dpll_solve(best_assignment, polish_deadline)
        if polish_result is not None:
            if is_satisfied(polish_result):
                return polish_result
    
    # Return best assignment found
    if best_assignment is not None:
        return best_assignment
    else:
        return [rng.random() < 0.5 for _ in range(n)]

# EVOLVE-BLOCK-END
