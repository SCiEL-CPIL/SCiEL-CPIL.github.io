"""Improved planted 3-SAT solver: DPLL + WalkSAT with clause tracking."""
# EVOLVE-BLOCK-START
import random
import time


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 290  # safety margin before 330s kill

    # Precompute clause data
    clause_vars = [[abs(l) - 1 for l in c] for c in clauses]
    clause_signs = [[(l > 0) for l in c] for c in clauses]
    m = len(clauses)

    # Build variable-to-clauses index
    var_to_clauses = [[] for _ in range(n)]
    for ci in range(m):
        for vi in clause_vars[ci]:
            var_to_clauses[vi].append(ci)

    def is_clause_sat(ci, assignment):
        for vi, s in zip(clause_vars[ci], clause_signs[ci]):
            if assignment[vi] == s:
                return True
        return False

    def dpll_with_up(assignment, deadline):
        """DPLL with unit propagation. Returns True if solved, False if failed."""
        # Deep copy assignment
        asg = list(assignment)
        unassigned = set(range(n))
        
        # Track which clauses are satisfied
        sat_clauses = set()
        for ci in range(m):
            if is_clause_sat(ci, asg):
                sat_clauses.add(ci)
        
        # Active clauses (not yet satisfied)
        active = [ci for ci in range(m) if ci not in sat_clauses]
        
        # Unit propagation loop
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_active = []
            for ci in active:
                # Check if clause is satisfied
                sat = False
                unassigned_lits = []
                for vi, s in zip(clause_vars[ci], clause_signs[ci]):
                    if vi not in unassigned:
                        if asg[vi] == s:
                            sat = True
                            break
                    else:
                        unassigned_lits.append((vi, s))
                
                if sat:
                    continue
                
                if len(unassigned_lits) == 0:
                    return False  # Empty clause, unsatisfiable
                
                if len(unassigned_lits) == 1:
                    # Unit clause: force assignment
                    vi, s = unassigned_lits[0]
                    if vi in unassigned:
                        asg[vi] = s
                        unassigned.discard(vi)
                        changed = True
                        # Don't add to new_active, it's now satisfied
                        continue
                
                new_active.append(ci)
            
            active = new_active
        
        if time.time() > deadline:
            return False
        
        if not active:
            return True
        
        # If all variables assigned and not all satisfied, fail
        if not unassigned:
            return False
        
        # Pick a variable to branch on
        var = next(iter(unassigned))
        
        # Try True
        asg[var] = True
        unassigned.discard(var)
        
        # Check if this satisfies any active clauses
        new_sat = set()
        for ci in active:
            if is_clause_sat(ci, asg):
                new_sat.add(ci)
        
        remaining = [ci for ci in active if ci not in new_sat]
        
        # Quick check for empty clauses
        for ci in remaining:
            empty = True
            for vi, s in zip(clause_vars[ci], clause_signs[ci]):
                if vi in unassigned:
                    empty = False
                    break
            if empty:
                # Backtrack: try False
                asg[var] = False
                unassigned.add(var)
                # Try False branch
                asg[var] = False
                unassigned.discard(var)
                new_sat2 = set()
                for ci in active:
                    if is_clause_sat(ci, asg):
                        new_sat2.add(ci)
                remaining2 = [ci for ci in active if ci not in new_sat2]
                for ci in remaining2:
                    empty2 = True
                    for vi, s in zip(clause_vars[ci], clause_signs[ci]):
                        if vi in unassigned:
                            empty2 = False
                            break
                    if empty2:
                        return False
                # Recurse
                if dpll_with_up(asg, deadline):
                    return True
                return False
            return False
        
        # Recurse with True assignment
        if dpll_with_up(asg, deadline):
            return True
        
        # Backtrack and try False
        asg[var] = False
        unassigned.add(var)
        asg[var] = False
        unassigned.discard(var)
        
        new_sat = set()
        for ci in active:
            if is_clause_sat(ci, asg):
                new_sat.add(ci)
        remaining = [ci for ci in active if ci not in new_sat]
        
        for ci in remaining:
            empty = True
            for vi, s in zip(clause_vars[ci], clause_signs[ci]):
                if vi in unassigned:
                    empty = False
                    break
            if empty:
                return False
        
        if dpll_with_up(asg, deadline):
            return True
        
        return False

    def walksat(assignment, max_steps, p_noise, deadline):
        """Efficient WalkSAT with clause tracking. Returns True if solved."""
        if time.time() > deadline:
            return False
        
        # Initialize unsat set
        unsat = set()
        for ci in range(m):
            if not is_clause_sat(ci, assignment):
                unsat.add(ci)
        
        if not unsat:
            return True
        
        for step in range(max_steps):
            if time.time() > deadline:
                return False
            
            if not unsat:
                return True
            
            # Pick random unsatisfied clause
            ci = random.choice(list(unsat))
            vars_in_clause = clause_vars[ci]
            signs_in_clause = clause_signs[ci]
            
            # Score each variable in the clause
            best_var = -1
            best_score = -1
            
            for vi, s in zip(vars_in_clause, signs_in_clause):
                # If we flip vi, this clause becomes satisfied
                # Count: how many other unsat clauses contain vi and would be satisfied?
                # And how many would be broken?
                score = 0
                for ci2 in unsat:
                    if ci2 == ci:
                        continue
                    # Does ci2 contain vi?
                    for vi2, s2 in zip(clause_vars[ci2], clause_signs[ci2]):
                        if vi2 == vi:
                            # If we flip vi, does it satisfy ci2?
                            if (not assignment[vi]) == s2:
                                score += 1
                            break
                
                if score > best_score:
                    best_score = score
                    best_var = vi
            
            if best_var == -1:
                vi = random.choice(vars_in_clause)
            else:
                vi = best_var
            
            # With probability p_noise, pick random variable from clause
            if random.random() < p_noise:
                vi = random.choice(vars_in_clause)
            
            # Flip vi
            old_val = assignment[vi]
            assignment[vi] = not old_val
            
            # Update unsat set
            # Clauses that might become satisfied: those containing vi
            for ci2 in var_to_clauses[vi]:
                if ci2 in unsat:
                    if is_clause_sat(ci2, assignment):
                        unsat.discard(ci2)
            
            # Clauses that might become unsatisfied: those containing vi
            for ci2 in var_to_clauses[vi]:
                if ci2 not in unsat:
                    if not is_clause_sat(ci2, assignment):
                        unsat.add(ci2)
        
        return len(unsat) == 0

    # Phase 1: Try DPLL with unit propagation on random assignments
    dpll_attempts = 5
    for attempt in range(dpll_attempts):
        if time.time() > deadline:
            break
        asg = [random.random() < 0.5 for _ in range(n)]
        if dpll_with_up(asg, deadline):
            return asg

    # Phase 2: WalkSAT with restarts and adaptive noise
    base_steps = 3000
    max_restarts = 100
    restarts = 0
    
    best_asg = None
    best_unsat_count = m + 1
    
    while time.time() < deadline and restarts < max_restarts:
        asg = [random.random() < 0.5 for _ in range(n)]
        
        # Adaptive noise: start high, decrease
        p = max(0.1, 0.5 * (0.9 ** restarts))
        steps = base_steps + restarts * 500
        
        if walksat(asg, steps, p, deadline):
            return asg
        
        # Track best
        unsat_count = sum(1 for ci in range(m) if not is_clause_sat(ci, asg))
        if unsat_count < best_unsat_count:
            best_unsat_count = unsat_count
            best_asg = list(asg)
        
        restarts += 1

    # Phase 3: Polish with focused WalkSAT from best assignment
    if best_asg is not None and time.time() < deadline:
        polish_deadline = min(deadline, time.time() + 30)
        asg = list(best_asg)
        # More steps, lower noise
        if walksat(asg, 20000, 0.05, polish_deadline):
            return asg
    
    if best_asg is not None:
        return best_asg
    return [True] * n
# EVOLVE-BLOCK-END
