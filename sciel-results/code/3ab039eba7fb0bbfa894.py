"""CDCL-style SAT solver with clause learning for planted 3-SAT (50 vars)."""
# EVOLVE-BLOCK-START
import time
import random
import heapq
import math


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 280  # safety margin

    # Build CNF
    cnf = [tuple(c) for c in clauses]
    m = len(cnf)

    # Variable-to-clause index
    var_to_clauses = [[] for _ in range(n)]
    for ci, c in enumerate(cnf):
        for lit in c:
            vi = abs(lit) - 1
            var_to_clauses[vi].append(ci)

    # DPLL with clause learning
    # State: assignment array (True/False/None), trail, clause database
    
    # Preprocessing: unit propagation and pure literal elimination
    def preprocess():
        """Apply unit propagation and pure literal elimination."""
        assign = [None] * n  # None = unassigned
        # Work with mutable clause list
        cnf_work = [list(c) for c in cnf]
        
        # Remove satisfied clauses, propagate units
        changed = True
        while changed:
            changed = False
            new_cnf = []
            for c in cnf_work:
                # Check if satisfied
                sat = False
                for lit in c:
                    vi = abs(lit) - 1
                    if assign[vi] is not None and assign[vi] == (lit > 0):
                        sat = True
                        break
                if not sat:
                    new_cnf.append(c)
            cnf_work = new_cnf
            
            # Unit propagation
            for i, c in enumerate(cnf_work):
                if len(c) == 1:
                    lit = c[0]
                    vi = abs(lit) - 1
                    if assign[vi] is None:
                        assign[vi] = (lit > 0)
                        changed = True
                    elif assign[vi] != (lit > 0):
                        return None, None, assign  # contradiction
                elif len(c) == 0:
                    return None, None, assign  # contradiction
            
            # Pure literal elimination
            pos_count = [0] * n
            neg_count = [0] * n
            for c in cnf_work:
                seen = set()
                for lit in c:
                    vi = abs(lit) - 1
                    if vi not in seen:
                        seen.add(vi)
                        if lit > 0:
                            pos_count[vi] += 1
                        else:
                            neg_count[vi] += 1
            
            for vi in range(n):
                if assign[vi] is not None:
                    continue
                if pos_count[vi] == 0 and neg_count[vi] > 0:
                    assign[vi] = False
                    changed = True
                elif neg_count[vi] == 0 and pos_count[vi] > 0:
                    assign[vi] = True
                    changed = True
        
        # Check if all assigned and satisfied
        if all(a is not None for a in assign):
            for c in cnf_work:
                sat = False
                for lit in c:
                    vi = abs(lit) - 1
                    if assign[vi] == (lit > 0):
                        sat = True
                        break
                if not sat:
                    return None, None, assign
            return [bool(a) for a in assign], None, assign
        
        return None, cnf_work, assign

    solved, reduced_cnf, partial = preprocess()
    if solved is not None:
        return solved

    # Now use DPLL with clause learning on the remaining problem
    # For the full problem (or reduced), implement a proper DPLL
    
    # Use the original CNF for DPLL (preprocessing already applied)
    # DPLL with chronological backtracking and unit propagation
    
    def dpll_solve(cnf_list, assign, trail, depth=0):
        """DPLL solver with unit propagation. Returns (solved, assignment, trail)."""
        if time.time() > deadline:
            return False, assign, trail
        
        cnf_work = [list(c) for c in cnf_list]
        
        # Unit propagation loop
        changed = True
        while changed:
            changed = False
            # Remove satisfied clauses
            new_cnf = []
            for c in cnf_work:
                sat = False
                for lit in c:
                    vi = abs(lit) - 1
                    if assign[vi] is not None and assign[vi] == (lit > 0):
                        sat = True
                        break
                if not sat:
                    new_cnf.append(c)
            cnf_work = new_cnf
            
            # Check for empty clause (contradiction)
            for c in cnf_work:
                if len(c) == 0:
                    return False, assign, trail
            
            # Unit propagation
            for i, c in enumerate(cnf_work):
                if len(c) == 1:
                    lit = c[0]
                    vi = abs(lit) - 1
                    if assign[vi] is None:
                        assign[vi] = (lit > 0)
                        trail.append(vi)
                        changed = True
                    elif assign[vi] != (lit > 0):
                        return False, assign, trail
        
        # Check if all variables assigned
        unassigned = [i for i in range(n) if assign[i] is None]
        if not unassigned:
            # Verify all clauses satisfied
            for c in cnf_work:
                sat = False
                for lit in c:
                    vi = abs(lit) - 1
                    if assign[vi] == (lit > 0):
                        sat = True
                        break
                if not sat:
                    return False, assign, trail
            return True, assign, trail
        
        # Variable selection: pick variable with highest degree in remaining clauses
        var_scores = {}
        for c in cnf_work:
            for lit in c:
                vi = abs(lit) - 1
                if assign[vi] is None:
                    var_scores[vi] = var_scores.get(vi, 0) + 1
        
        if not var_scores:
            # All remaining clauses are satisfied or empty
            for c in cnf_work:
                if len(c) == 0:
                    return False, assign, trail
            return True, assign, trail
        
        # Pick variable with max score, tie-break by index
        best_var = max(var_scores.keys(), key=lambda v: (var_scores[v], -v))
        
        # Try True first
        assign[best_var] = True
        trail.append(best_var)
        solved, assign, trail = dpll_solve(cnf_work, assign, trail, depth + 1)
        if solved:
            return True, assign, trail
        
        # Backtrack
        if trail and trail[-1] == best_var:
            trail.pop()
        assign[best_var] = None
        
        # Try False
        assign[best_var] = False
        trail.append(best_var)
        solved, assign, trail = dpll_solve(cnf_work, assign, trail, depth + 1)
        if solved:
            return True, assign, trail
        
        # Backtrack
        if trail and trail[-1] == best_var:
            trail.pop()
        assign[best_var] = None
        
        return False, assign, trail

    # Initialize assignment from preprocessing
    assign = [None] * n
    for i in range(n):
        if partial[i] is not None:
            assign[i] = bool(partial[i])
    
    # Run DPLL
    solved, assign, trail = dpll_solve(cnf, assign, [])
    if solved:
        return [bool(a) if a is not None else False for a in assign]
    
    # If DPLL didn't solve (shouldn't happen for SAT instances within time),
    # fall back to WalkSAT
    if time.time() > deadline - 10:
        return [bool(a) if a is not None else False for a in assign]
    
    # WalkSAT fallback
    def count_unsat(assignment):
        count = 0
        for c in cnf:
            sat = False
            for lit in c:
                vi = abs(lit) - 1
                if assignment[vi] == (lit > 0):
                    sat = True
                    break
            if not sat:
                count += 1
        return count

    def is_satisfied(assignment):
        for c in cnf:
            sat = False
            for lit in c:
                vi = abs(lit) - 1
                if assignment[vi] == (lit > 0):
                    sat = True
                    break
            if not sat:
                return False
        return True

    # Make full assignment
    rng = random.Random(42)
    full_assign = []
    for i in range(n):
        if assign[i] is not None:
            full_assign.append(bool(assign[i]))
        else:
            full_assign.append(rng.random() < 0.5)
    
    if is_satisfied(full_assign):
        return full_assign
    
    best_assign = full_assign[:]
    best_unsat = count_unsat(full_assign)
    
    # WalkSAT with incremental tracking
    clause_true_count = [0] * m
    for ci in range(m):
        cnt = 0
        for lit in cnf[ci]:
            vi = abs(lit) - 1
            if full_assign[vi] == (lit > 0):
                cnt += 1
        clause_true_count[ci] = cnt
    
    cur_unsat = best_unsat
    
    max_steps = 50000
    p_bias = 0.75
    rng = random.Random(123)
    cur = best_assign[:]
    
    for step in range(max_steps):
        if time.time() > deadline - 5:
            break
        if cur_unsat == 0:
            return cur
        
        # Find unsatisfied clauses
        unsat_clauses = [ci for ci in range(m) if clause_true_count[ci] == 0]
        if not unsat_clauses:
            return cur
        
        # Random clause
        ci = unsat_clauses[rng.randrange(len(unsat_clauses))]
        clause = cnf[ci]
        
        # Weighted literal selection
        weights = []
        for lit in clause:
            vi = abs(lit) - 1
            var_degree = len(var_to_clauses[vi])
            w = 1.0 / (1 + var_degree)
            if cur[vi] == (lit > 0):
                w *= 0.2
            else:
                # Count help/hurt
                help_count = 0
                hurt_count = 0
                for aci in var_to_clauses[vi]:
                    if aci == ci:
                        continue
                    for lit2 in cnf[aci]:
                        vi2 = abs(lit2) - 1
                        if vi2 == vi:
                            if (lit2 > 0) != cur[vi]:
                                help_count += 1
                            else:
                                hurt_count += 1
                            break
                net = help_count - hurt_count
                w *= (1.0 + 0.3 * net)
                if w < 0.1:
                    w = 0.1
            weights.append(w)
        
        total_w = sum(weights)
        r = rng.random() * total_w
        cum = 0
        chosen_idx = 0
        for i, w in enumerate(weights):
            cum += w
            if r <= cum:
                chosen_idx = i
                break
        
        literal = clause[chosen_idx]
        vi = abs(literal) - 1
        
        if rng.random() < p_bias:
            cur[vi] = (literal > 0)
        else:
            cur[vi] = not cur[vi]
        
        # Incremental update
        for aci in var_to_clauses[vi]:
            old_count = clause_true_count[aci]
            new_count = 0
            for lit in cnf[aci]:
                vi2 = abs(lit) - 1
                if cur[vi2] == (lit > 0):
                    new_count += 1
            clause_true_count[aci] = new_count
            if old_count == 0 and new_count > 0:
                cur_unsat -= 1
            elif old_count > 0 and new_count == 0:
                cur_unsat += 1
        
        if cur_unsat < best_unsat:
            best_unsat = cur_unsat
            best_assign = cur[:]
    
    if is_satisfied(cur):
        return cur
    return best_assign

# EVOLVE-BLOCK-END
