import random
import time
import sys

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 285.0  # Safety margin
    
    # Convert clauses to (var, pos) tuples for DPLL
    # var: 0..n-1, pos: True/False
    def to_internal(clause):
        return [(abs(l) - 1, l > 0) for l in clause]
    
    internal_clauses = [to_internal(c) for c in clauses]
    
    # DPLL solver with unit propagation and pure literal elimination
    def dpll(clauses, assign):
        """
        clauses: list of lists of (var, pos)
        assign: list of -1 (unassigned), 0 (False), 1 (True)
        Returns: (solved, final_assign) where solved is True/False/None (timeout)
        """
        
        def propagate(clauses, assign):
            """Apply unit propagation. Returns (ok, new_clauses, new_assign, conflict)"""
            changed = True
            while changed:
                changed = False
                new_clauses = []
                for clause in clauses:
                    sat = False
                    unassigned = []
                    for var, pos in clause:
                        if assign[var] == 1:
                            if pos:
                                sat = True
                                break
                        elif assign[var] == 0:
                            if not pos:
                                sat = True
                                break
                        else:
                            unassigned.append((var, pos))
                    if sat:
                        continue
                    if not unassigned:
                        return False, None, None, True
                    if len(unassigned) == 1:
                        var, pos = unassigned[0]
                        assign[var] = 1 if pos else 0
                        changed = True
                    else:
                        new_clauses.append(unassigned)
            return True, new_clauses, assign, False

        def pure_literal_elim(clauses, assign):
            """Apply pure literal elimination. Returns (new_clauses, new_assign) or (None, None) on conflict"""
            if not clauses:
                return clauses, assign
            
            pos_count = [0] * n
            neg_count = [0] * n
            for clause in clauses:
                for var, pos in clause:
                    if assign[var] != -1:
                        continue
                    if pos:
                        pos_count[var] += 1
                    else:
                        neg_count[var] += 1

            # Find pure literals
            pure_assignments = []
            for var in range(n):
                if assign[var] != -1:
                    continue
                if pos_count[var] > 0 and neg_count[var] == 0:
                    pure_assignments.append((var, 1))
                elif neg_count[var] > 0 and pos_count[var] == 0:
                    pure_assignments.append((var, 0))

            if not pure_assignments:
                return clauses, assign

            # Apply pure literal assignments
            for var, val in pure_assignments:
                assign[var] = val

            # Simplify clauses
            new_clauses = []
            for clause in clauses:
                sat = False
                remaining = []
                for var, pos in clause:
                    if assign[var] == 1:
                        if pos:
                            sat = True
                            break
                    elif assign[var] == 0:
                        if not pos:
                            sat = True
                            break
                    else:
                        remaining.append((var, pos))
                if sat:
                    continue
                if not remaining:
                    return None, None
                new_clauses.append(remaining)
            return new_clauses, assign

        # Main DPLL loop
        while True:
            if time.time() > deadline:
                return None, None

            # Unit propagation
            ok, clauses, assign, conflict = propagate(clauses, assign)
            if conflict:
                return False, None
            if not clauses:
                return True, assign

            # Pure literal elimination
            clauses, assign = pure_literal_elim(clauses, assign)
            if clauses is None:
                return False, None
            if not clauses:
                return True, assign

            # If all variables assigned, check satisfaction
            unassigned_vars = [v for v in range(n) if assign[v] == -1]
            if not unassigned_vars:
                # Verify all clauses satisfied
                for clause in clauses:
                    sat = False
                    for var, pos in clause:
                        if assign[var] == (1 if pos else 0):
                            sat = True
                            break
                    if not sat:
                        return False, None
                return True, assign

            # Select variable: most frequent in remaining clauses
            var_freq = {}
            for clause in clauses:
                for var, pos in clause:
                    if assign[var] == -1:
                        var_freq[var] = var_freq.get(var, 0) + 1

            if not var_freq:
                return True, assign

            # Pick variable with highest frequency
            var = max(var_freq, key=lambda v: var_freq[v])

            # Try True first, then False
            for val in [1, 0]:
                if time.time() > deadline:
                    return None, None

                new_assign = assign[:]
                new_assign[var] = val
                
                # Simplify clauses with this assignment
                new_clauses = []
                conflict_found = False
                for clause in clauses:
                    sat = False
                    remaining = []
                    for v, pos in clause:
                        if v == var:
                            if (val == 1 and pos) or (val == 0 and not pos):
                                sat = True
                                break
                        elif new_assign[v] == -1:
                            remaining.append((v, pos))
                        else:
                            if (new_assign[v] == 1 and pos) or (new_assign[v] == 0 and not pos):
                                sat = True
                                break
                    if sat:
                        continue
                    if not remaining:
                        conflict_found = True
                        break
                    new_clauses.append(remaining)

                if conflict_found:
                    continue

                result, new_assign2 = dpll(new_clauses, new_assign)
                if result is True:
                    return True, new_assign2
                elif result is None:
                    return None, None
                # result is False, try other value

            return False, None

    # Run DPLL
    initial_assign = [-1] * n
    success, dpll_assign = dpll(internal_clauses, initial_assign)
    
    if success and dpll_assign is not None:
        result = []
        for v in range(n):
            if dpll_assign[v] == -1:
                result.append(random.random() < 0.5)
            else:
                result.append(dpll_assign[v] == 1)
        return result

    # Fallback: Optimized WalkSAT with incremental tracking
    rng = random.Random(42)
    
    # Build index for fast lookup
    var_to_clauses = [[] for _ in range(n)]
    for i, clause in enumerate(clauses):
        for l in clause:
            v = abs(l) - 1
            var_to_clauses[v].append((i, l > 0))
    
    m = len(clauses)
    
    # Initial random assignment
    assignment = [rng.random() < 0.5 for _ in range(n)]
    
    # Track satisfied status of each clause
    clause_sat = [False] * m
    clause_unsat_count = [0] * m  # Number of unsatisfied literals in clause
    
    def update_clause(ci):
        """Update satisfaction status of clause ci"""
        count = 0
        for l in clauses[ci]:
            v = abs(l) - 1
            if assignment[v] != (l > 0):
                count += 1
        clause_unsat_count[ci] = count
        clause_sat[ci] = (count == 0)
    
    # Initialize clause states
    for ci in range(m):
        update_clause(ci)
    
    unsat_count = sum(1 for s in clause_sat if not s)
    
    max_restarts = 100
    flips_per_restart = 80000
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break

        if restart > 0:
            assignment = [rng.random() < 0.5 for _ in range(n)]
            for ci in range(m):
                update_clause(ci)
            unsat_count = sum(1 for s in clause_sat if not s)
        
        for step in range(flips_per_restart):
            if time.time() > deadline - 5:
                return assignment

            if unsat_count == 0:
                return assignment

            # Pick a random unsatisfied clause
            unsat_clauses = [ci for ci in range(m) if not clause_sat[ci]]
            if not unsat_clauses:
                return assignment
            best_clause = unsat_clauses[rng.randrange(len(unsat_clauses))]

            # Pick literal to flip: minimize net increase in unsatisfied clauses
            best_var = -1
            best_net = float('inf')
            
            for l in clauses[best_clause]:
                var = abs(l) - 1
                new_val = not assignment[var]
                
                # Count change in unsatisfied clauses
                would_break = 0  # Satisfied clauses that become unsatisfied
                would_satisfy = 0  # Unsatisfied clauses that become satisfied
                
                for ci, pos in var_to_clauses[var]:
                    if clause_sat[ci]:
                        # Check if this clause would become unsatisfied
                        # It's currently satisfied. After flip, check if still satisfied
                        still_sat = False
                        for ll in clauses[ci]:
                            vv = abs(ll) - 1
                            val_after = new_val if vv == var else assignment[vv]
                            if val_after == (ll > 0):
                                still_sat = True
                                break
                        if not still_sat:
                            would_break += 1
                    else:
                        # Clause is currently unsatisfied
                        # Check if it becomes satisfied after flip
                        # It becomes satisfied if the literal for var becomes true
                        # AND it was the only unsatisfied literal (or at least one becomes true)
                        # Actually, a clause becomes satisfied if ANY literal is true
                        # After flip, check if any literal is true
                        now_sat = False
                        for ll in clauses[ci]:
                            vv = abs(ll) - 1
                            val_after = new_val if vv == var else assignment[vv]
                            if val_after == (ll > 0):
                                now_sat = True
                                break
                        if now_sat:
                            would_satisfy += 1
                
                net_change = would_break - would_satisfy
                if net_change < best_net:
                    best_net = net_change
                    best_var = var

            if best_var != -1:
                old_val = assignment[best_var]
                assignment[best_var] = not old_val
                
                # Update affected clauses
                for ci, pos in var_to_clauses[best_var]:
                    old_sat = clause_sat[ci]
                    old_unsat = clause_unsat_count[ci]
                    update_clause(ci)
                    new_sat = clause_sat[ci]
                    new_unsat = clause_unsat_count[ci]
                    
                    if old_sat and not new_sat:
                        unsat_count += 1
                    elif not old_sat and new_sat:
                        unsat_count -= 1

    return assignment
