"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np


def _build_clause_data(n, clauses):
    """Precompute clause structure as numpy arrays for vectorized operations."""
    m = len(clauses)
    clause_vars = np.zeros((m, 3), dtype=np.int32)
    clause_sgn = np.zeros((m, 3), dtype=np.int8)
    for i, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            clause_vars[i, j] = abs(lit) - 1
            clause_sgn[i, j] = 1 if lit > 0 else -1
    # For each variable, precompute which clauses contain it and with what sign
    var_to_clauses = {}
    for v in range(n):
        var_to_clauses[v] = []
    for i in range(m):
        for j in range(3):
            v = clause_vars[i, j]
            var_to_clauses[v].append((i, j))
    return clause_vars, clause_sgn, var_to_clauses


def _compute_sat_state(clause_vars, clause_sgn, assignment):
    """Compute per-clause satisfaction state: sat_count (how many literals satisfied)."""
    m, k = clause_vars.shape
    # lit_satisfied[m, k]: True if literal is satisfied by current assignment
    # assignment is a numpy bool array of length n
    # For literal with sign s and variable v: satisfied if s * (2*assignment[v] - 1) > 0
    # i.e., if s=1, need assignment[v]=True; if s=-1, need assignment[v]=False
    var_idx = clause_vars  # (m, k)
    signs = clause_sgn     # (m, k)
    # Get current truth values for all variables in all clauses
    truth = assignment[var_idx]  # (m, k) bool
    # Literal satisfied if (sign==1 and truth) or (sign==-1 and not truth)
    # = (sign * (truth*2 - 1)) > 0  =>  sign * truth == sign*1 or sign*(-1)==sign*-1
    # Simpler: satisfied = (sign == 1) & truth | (sign == -1) & ~truth
    sat_lit = (signs == 1) & truth | (signs == -1) & (~truth)
    sat_count = sat_lit.sum(axis=1)  # (m,)
    return sat_count


def _phase_seed(n, clauses, rng):
    """Initialize assignment using phase information from clause polarities."""
    pos_count = [0] * n
    neg_count = [0] * n
    for clause in clauses:
        for lit in clause:
            if lit > 0:
                pos_count[lit - 1] += 1
            else:
                neg_count[-lit - 1] += 1
    assignment = []
    for i in range(n):
        if pos_count[i] > neg_count[i]:
            assignment.append(True)
        elif neg_count[i] > pos_count[i]:
            assignment.append(False)
        else:
            assignment.append(rng.random() < 0.5)
    return assignment


def _count_unsat(assignment, clauses):
    """Count number of unsatisfied clauses."""
    count = 0
    for clause in clauses:
        sat = False
        for lit in clause:
            if assignment[abs(lit) - 1] == (lit > 0):
                sat = True
                break
        if not sat:
            count += 1
    return count


def _vectorized_walksat(assignment, clause_vars, clause_sgn, var_to_clauses, n, m, rng, max_steps, deadline):
    """
    Vectorized WalkSAT:
    - Maintains sat_count per clause
    - On flip of variable v, updates sat_count for all clauses containing v
    - Greedy: pick literal in unsat clause that maximizes net gain (satisfies most unsat clauses)
    - Random: with probability 0.2, flip random literal in clause
    """
    assignment = np.array(assignment, dtype=bool)
    sat_count = _compute_sat_state(clause_vars, clause_sgn, assignment)
    
    for step in range(max_steps):
        if time.time() >= deadline:
            break
        
        unsat_mask = sat_count == 0
        n_unsat = int(unsat_mask.sum())
        if n_unsat == 0:
            break
        
        # Pick a random unsatisfied clause
        unsat_indices = np.where(unsat_mask)[0]
        ci = int(rng.choice(unsat_indices.tolist()))
        
        # Get the 3 literals in this clause
        lits_vars = clause_vars[ci]  # (3,)
        lits_sgn = clause_sgn[ci]    # (3,)
        
        # Decide: random flip or greedy
        if rng.random() < 0.2:
            # Random flip of one literal in the clause
            li = rng.randint(0, 2)
            v = int(lits_vars[li])
            old_val = assignment[v]
            assignment[v] = not assignment[v]
            # Update sat_count for all clauses containing v
            for (ci2, li2) in var_to_clauses[v]:
                ci2 = int(ci2)
                li2 = int(li2)
                v2 = int(clause_vars[ci2, li2])
                s2 = int(clause_sgn[ci2, li2])
                # Old literal sat state
                if s2 == 1:
                    old_lit_sat = old_val
                else:
                    old_lit_sat = not old_val
                # New literal sat state
                new_val = not old_val
                if s2 == 1:
                    new_lit_sat = new_val
                else:
                    new_lit_sat = not new_val
                if new_lit_sat and not old_lit_sat:
                    sat_count[ci2] += 1
                elif old_lit_sat and not new_lit_sat:
                    sat_count[ci2] -= 1
        else:
            # Greedy: for each of the 3 literals, compute net gain
            best_gain = -999
            best_v = -1
            
            for li in range(3):
                v = int(lits_vars[li])
                s = int(lits_sgn[li])
                old_val = assignment[v]
                new_val = not old_val
                
                # Compute gain: for each clause containing v, check if flipping v changes its sat status
                gain = 0
                for (ci2, li2) in var_to_clauses[v]:
                    ci2 = int(ci2)
                    li2 = int(li2)
                    v2 = int(clause_vars[ci2, li2])
                    s2 = int(clause_sgn[ci2, li2])
                    
                    # Current sat status of this literal in clause ci2
                    if s2 == 1:
                        cur_lit_sat = old_val
                    else:
                        cur_lit_sat = not old_val
                    
                    # New sat status after flip
                    if s2 == 1:
                        new_lit_sat = new_val
                    else:
                        new_lit_sat = not new_val
                    
                    if ci2 == ci:
                        # This is the target clause (currently unsat)
                        # After flip, check if clause becomes sat
                        new_sc = 0
                        for jj in range(3):
                            vv = int(clause_vars[ci, jj])
                            ss = int(clause_sgn[ci, jj])
                            val = new_val if vv == v else assignment[vv]
                            if ss == 1:
                                if val:
                                    new_sc += 1
                            else:
                                if not val:
                                    new_sc += 1
                        if new_sc > 0:
                            gain += 1
                    else:
                        # Other clause
                        if new_lit_sat and not cur_lit_sat:
                            if sat_count[ci2] == 0:
                                gain += 1
                        elif cur_lit_sat and not new_lit_sat:
                            if sat_count[ci2] == 1:
                                gain -= 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
            
            if best_v >= 0:
                old_val = assignment[best_v]
                assignment[best_v] = not assignment[best_v]
                new_val = not old_val
                # Update sat_count for affected clauses
                for (ci2, li2) in var_to_clauses[best_v]:
                    ci2 = int(ci2)
                    li2 = int(li2)
                    v2 = int(clause_vars[ci2, li2])
                    s2 = int(clause_sgn[ci2, li2])
                    if s2 == 1:
                        old_lit_sat = old_val
                        new_lit_sat = new_val
                    else:
                        old_lit_sat = not old_val
                        new_lit_sat = not new_val
                    if new_lit_sat and not old_lit_sat:
                        sat_count[ci2] += 1
                    elif old_lit_sat and not new_lit_sat:
                        sat_count[ci2] -= 1
    
    return assignment.tolist()


def _minconflicts(assignment, clauses, n, rng, max_steps, deadline):
    """MinConflicts: pick most violated clause, flip var that minimizes violations."""
    for _ in range(max_steps):
        if time.time() >= deadline:
            return assignment
        # Find all unsatisfied clauses
        unsat_clauses = []
        for c in clauses:
            sat = False
            for lit in c:
                if assignment[abs(lit) - 1] == (lit > 0):
                    sat = True
                    break
            if not sat:
                unsat_clauses.append(c)
        if not unsat_clauses:
            return assignment
        # Pick a random unsatisfied clause
        clause = rng.choice(unsat_clauses)
        # For each literal in the clause, compute the cost of flipping that variable
        best_var = -1
        best_cost = float('inf')
        for lit in clause:
            var = abs(lit) - 1
            # Temporarily flip and count violations
            old_val = assignment[var]
            assignment[var] = not assignment[var]
            cost = 0
            for c in clauses:
                sat = False
                for l2 in c:
                    if assignment[abs(l2) - 1] == (l2 > 0):
                        sat = True
                        break
                if not sat:
                    cost += 1
            assignment[var] = old_val
            if cost < best_cost:
                best_cost = cost
                best_var = var
            elif cost == best_cost and rng.random() < 0.5:
                best_var = var
        if best_var >= 0:
            assignment[best_var] = not assignment[best_var]
    return assignment


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 280  # Leave 50s margin before 330s kill
    
    m = len(clauses)
    
    # Precompute clause structure
    clause_vars, clause_sgn, var_to_clauses = _build_clause_data(n, clauses)
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    # Strategy 1: Phase-seeded vectorized WalkSAT with multiple restarts
    for attempt in range(80):
        if time.time() >= deadline:
            break
        seed = attempt * 7919 + 12345
        rng = random.Random(seed)
        assignment = _phase_seed(n, clauses, rng)
        
        # Initial check
        unsat_count = _count_unsat(assignment, clauses)
        if unsat_count == 0:
            return assignment
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
        
        # Vectorized WalkSAT
        steps = 2000 + attempt * 300
        assignment = _vectorized_walksat(assignment, clause_vars, clause_sgn, var_to_clauses, n, m, rng, steps, deadline)
        
        unsat_count = _count_unsat(assignment, clauses)
        if unsat_count == 0:
            return assignment
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
    
    # Strategy 2: MinConflicts from best assignment found so far
    if best_assignment is not None:
        rng = random.Random(99999)
        assignment = best_assignment[:]
        assignment = _minconflicts(assignment, clauses, n, rng, 5000, deadline)
        unsat_count = _count_unsat(assignment, clauses)
        if unsat_count == 0:
            return assignment
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
    
    # Strategy 3: Pure random vectorized WalkSAT from scratch
    for attempt in range(30):
        if time.time() >= deadline:
            break
        seed = 100000 + attempt * 31337
        rng = random.Random(seed)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        
        unsat_count = _count_unsat(assignment, clauses)
        if unsat_count == 0:
            return assignment
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
        
        assignment = _vectorized_walksat(assignment, clause_vars, clause_sgn, var_to_clauses, n, m, rng, 3000, deadline)
        
        unsat_count = _count_unsat(assignment, clauses)
        if unsat_count == 0:
            return assignment
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_assignment = assignment[:]
    
    # Final: return best assignment found
    if best_assignment is not None:
        return best_assignment
    
    # Fallback: all False
    return [False] * n
# EVOLVE-BLOCK-END
