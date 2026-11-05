"""Planted 3-SAT solver (50 vars) using DPLL + unit propagation with local search fallback."""
# EVOLVE-BLOCK-START
import random
import time
import sys

sys.setrecursionlimit(100000)


def solve(instance):
    n = instance["n"]
    clauses_in = instance["clauses"]

    deadline = time.monotonic() + 280.0

    # Convert to 0-based internal representation
    # Each literal: (var_index, sign) where sign is +1 or -1
    clauses = []
    for c in clauses_in:
        clauses.append(tuple((abs(l) - 1, 1 if l > 0 else -1) for l in c))
    m = len(clauses)

    # Build var -> [(clause_idx, pos_in_clause)] map
    var_clause_map = [[] for _ in range(n)]
    for ci, cl in enumerate(clauses):
        for idx, (v, s) in enumerate(cl):
            var_clause_map[v].append((ci, idx))

    def convert_back(assign):
        """Convert internal bool list to the expected output format."""
        return assign

    # DPLL with unit propagation
    def dpll_solve(clauses, var_clause_map, n, deadline, rng):
        """
        DPLL solver with unit propagation.
        Returns (assignment, success) where assignment is a list of (val, decided)
        or None if failed.
        """
        # assignment: dict var -> bool
        # We use a stack-based approach with backtracking
        # For efficiency, we track clauses by their satisfied status
        
        # State: 
        #   assignment: dict var->bool
        #   active_clauses: list of clause indices still active (not satisfied)
        #   For each active clause, track which literals are satisfied/unsatisfied
        
        # Simpler approach: recursive DPLL with unit propagation
        # Since n=50, we can use a cleaner recursive approach
        
        # assignment: list of (var, value) decisions made
        # clause_state: for each clause, list of (lit_satisfied, lit_unsatisfied)
        
        # Let's use a different approach: maintain clause satisfaction incrementally
        # clause_sat[ci] = number of satisfied literals
        # clause_unsat[ci] = number of unsatisfied literals (not yet assigned)
        # A clause is satisfied if clause_sat > 0
        # A clause is falsified if clause_unsat == 0 and clause_sat == 0
        # A clause is unit if clause_sat == 0 and clause_unsat == 1
        
        def try_dpll(clauses, var_clause_map, n, deadline, rng):
            # Initialize
            clause_sat = [0] * len(clauses)
            clause_unsat = [0] * len(clauses)
            for ci, cl in enumerate(clauses):
                clause_unsat[ci] = len(cl)
            
            assignment = {}  # var -> bool
            
            def propagate(assignment, clause_sat, clause_unsat, clauses, var_clause_map, n):
                """
                Unit propagation. Returns (success, assignment, clause_sat, clause_unsat)
                or (False, ...) if contradiction.
                """
                changed = True
                while changed:
                    changed = False
                    # Find unit clauses
                    for ci, cl in enumerate(clauses):
                        if clause_sat[ci] > 0:
                            continue  # already satisfied
                        if clause_unsat[ci] == 0:
                            return False, assignment, clause_sat, clause_unsat  # contradiction
                        if clause_unsat[ci] == 1:
                            # Find the unassigned literal
                            for v, s in cl:
                                if v not in assignment:
                                    # Must assign v to satisfy this literal
                                    val = (s > 0)
                                    assignment[v] = val
                                    changed = True
                                    # Update all clauses containing v
                                    for cj, idx in var_clause_map[v]:
                                        if clause_sat[cj] > 0:
                                            continue  # already satisfied, skip
                                        v, s2 = clauses[cj][idx]
                                        if assignment[v] == (s2 > 0):
                                            clause_sat[cj] += 1
                                            clause_unsat[cj] -= 1
                                        else:
                                            clause_unsat[cj] -= 1
                                    break  # v is now assigned, move on
                return True, assignment, clause_sat, clause_unsat
            
            # Try unit propagation first
            ok, assignment, clause_sat, clause_unsat = propagate(
                assignment, clause_sat, clause_unsat, clauses, var_clause_map, n
            )
            if not ok:
                return None
            
            # Check if all clauses satisfied
            all_sat = all(clause_sat[ci] > 0 for ci in range(len(clauses)))
            if all_sat:
                # Fill in unassigned vars arbitrarily
                result = {}
                for v in range(n):
                    if v in assignment:
                        result[v] = assignment[v]
                    else:
                        result[v] = rng.random() < 0.5
                return result
            
            # Choose a variable to branch on
            # Heuristic: pick variable in most constrained clause
            # Find unsatisfied clauses with most unassigned literals
            best_var = None
            best_score = -1
            for ci, cl in enumerate(clauses):
                if clause_sat[ci] > 0:
                    continue
                # Count unassigned literals in this clause
                unassigned = []
                for v, s in cl:
                    if v not in assignment:
                        unassigned.append((v, s))
                if len(unassigned) > best_score:
                    best_score = len(unassigned)
                    best_var = unassigned[0][0] if unassigned else None
            
            if best_var is None:
                # All vars assigned but not all clauses satisfied? Shouldn't happen
                result = {}
                for v in range(n):
                    result[v] = assignment.get(v, rng.random() < 0.5)
                return result
            
            # Try both values
            for val in [True, False]:
                if time.monotonic() > deadline:
                    return None
                # Save state
                assignment_save = dict(assignment)
                clause_sat_save = clause_sat[:]
                clause_unsat_save = clause_unsat[:]
                
                # Assign and propagate
                assignment[best_var] = val
                # Update clauses
                for cj, idx in var_clause_map[best_var]:
                    if clause_sat[cj] > 0:
                        continue
                    v2, s2 = clauses[cj][idx]
                    if v2 == best_var:
                        if val == (s2 > 0):
                            clause_sat[cj] += 1
                            clause_unsat[cj] -= 1
                        else:
                            clause_unsat[cj] -= 1
                
                ok, assignment, clause_sat, clause_unsat = propagate(
                    assignment, clause_sat, clause_unsat, clauses, var_clause_map, n
                )
                
                if ok:
                    all_sat = all(clause_sat[ci] > 0 for ci in range(len(clauses)))
                    if all_sat:
                        result = {}
                        for v in range(n):
                            result[v] = assignment.get(v, rng.random() < 0.5)
                        return result
                    else:
                        # Recurse
                        result = try_dpll(clauses, var_clause_map, n, deadline, rng)
                        if result is not None:
                            return result
                
                # Restore state
                assignment.clear()
                assignment.update(assignment_save)
                clause_sat[:] = clause_sat_save
                clause_unsat[:] = clause_unsat_save
            
            return None
        
        return try_dpll(clauses, var_clause_map, n, deadline, rng)

    # Local search: improved WalkSAT with better heuristics
    def local_search(clauses, var_clause_map, n, m, deadline, rng, max_restarts=5000, max_steps=20000):
        """
        WalkSAT-style local search with min-conflict heuristic.
        Returns best assignment found.
        """
        best_assign = None
        best_unsat = m + 1
        
        for restart in range(max_restarts):
            if time.monotonic() > deadline:
                break
            
            assign = [rng.random() < 0.5 for _ in range(n)]
            
            # Build state
            clause_sat = [0] * m
            sat_lit = []
            for ci, cl in enumerate(clauses):
                cnt = 0
                sl = []
                for v, s in cl:
                    ok = assign[v] == (s > 0)
                    sl.append(ok)
                    if ok:
                        cnt += 1
                sat_lit.append(sl)
                clause_sat[ci] = cnt
            
            uc = m - sum(1 for x in clause_sat if x > 0)
            
            if uc < best_unsat:
                best_unsat = uc
                best_assign = assign[:]
            if uc == 0:
                return assign
            
            steps = 0
            while steps < max_steps:
                if uc == 0:
                    return assign
                if time.monotonic() > deadline:
                    break
                
                # Pick random unsatisfied clause
                ci = rng.randrange(m)
                if clause_sat[ci] > 0:
                    found = False
                    for k in range(m):
                        j = (ci + k) % m
                        if clause_sat[j] == 0:
                            ci = j
                            found = True
                            break
                    if not found:
                        break
                
                cl = clauses[ci]
                
                # Min-conflict: pick variable whose flip reduces unsat count most
                best_v = None
                best_delta = None
                for idx, (v, s) in enumerate(cl):
                    delta = 0
                    for cj, _ in var_clause_map[v]:
                        if clause_sat[cj] > 0:
                            # Clause is satisfied; check if v is the only satisfying literal
                            vpos = None
                            for p, (vv, ss) in enumerate(clauses[cj]):
                                if vv == v:
                                    vpos = p
                                    break
                            if vpos is not None and sat_lit[cj][vpos] and clause_sat[cj] == 1:
                                delta += 1
                        else:
                            # Clause is unsatisfied; flipping v might satisfy it
                            vpos = None
                            for p, (vv, ss) in enumerate(clauses[cj]):
                                if vv == v:
                                    vpos = p
                                    break
                            if vpos is not None and not sat_lit[cj][vpos]:
                                delta -= 1
                    if best_delta is None or delta < best_delta:
                        best_delta = delta
                        best_v = v
                
                # Random escape
                if rng.random() < 0.15:
                    best_v = cl[rng.randrange(len(cl))][0]
                
                # Flip
                for cj, idx in var_clause_map[best_v]:
                    was = sat_lit[cj][idx]
                    if was:
                        clause_sat[cj] -= 1
                    else:
                        clause_sat[cj] += 1
                    sat_lit[cj][idx] = not was
                assign[best_v] = not assign[best_v]
                
                uc = m - sum(1 for x in clause_sat if x > 0)
                if uc < best_unsat:
                    best_unsat = uc
                    best_assign = assign[:]
                
                steps += 1
        
        return best_assign

    rng = random.Random(0)

    # First try DPLL
    result = dpll_solve(clauses, var_clause_map, n, deadline, rng)
    if result is not None:
        # Convert dict to list
        assign = [result.get(v, False) for v in range(n)]
        return assign

    # Fall back to local search
    result = local_search(clauses, var_clause_map, n, m, deadline, rng)
    if result is not None:
        return result

    # Last resort: random assignment
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
