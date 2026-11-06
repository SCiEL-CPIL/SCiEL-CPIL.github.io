"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import sys

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)
    
    if m == 0:
        return [False] * n

    # Data structures for efficient access
    # var_clauses[v] = list of (clause_idx, sign) where sign is 1 for positive, -1 for negative
    var_clauses = [[] for _ in range(n)]
    # clause_vars[ci] = list of (var_idx, sign)
    clause_vars = [[] for _ in range(m)]
    
    for ci, clause in enumerate(clauses):
        for lit in clause:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            var_clauses[v].append((ci, s))
            clause_vars[ci].append((v, s))

    # Helper to check if a specific literal is satisfied given assignment
    # assignment is a list of 0s and 1s
    def lit_sat(assignment, v, s):
        return (assignment[v] == 1) if s == 1 else (assignment[v] == 0)

    # Phase 1: Fast DPLL
    # We implement a simple DPLL with unit propagation and pure literal elimination
    # but with a strict time limit to avoid getting stuck on hard instances.
    
    dpll_deadline = time.time() + 2.0 # Give DPLL 2 seconds
    
    def dpll(assignment, remaining_clauses):
        """
        assignment: list of None, 0, 1
        remaining_clauses: list of clause indices
        Returns True if satisfied, updates assignment
        """
        if time.time() > dpll_deadline:
            return False
        
        if not remaining_clauses:
            return True
            
        # Unit Propagation
        changed = True
        while changed:
            changed = False
            new_remaining = []
            for ci in remaining_clauses:
                # Check status of clause ci
                sat = False
                undecided = []
                for v, s in clause_vars[ci]:
                    val = assignment[v]
                    if val is not None:
                        if lit_sat(assignment, v, s):
                            sat = True
                            break
                    else:
                        undecided.append((v, s))
                
                if sat:
                    continue
                if not undecided:
                    # Falsified
                    return False
                if len(undecided) == 1:
                    # Unit clause
                    v, s = undecided[0]
                    assignment[v] = 1 if s == 1 else 0
                    changed = True
                else:
                    new_remaining.append(ci)
            remaining_clauses = new_remaining
            
        if not remaining_clauses:
            return True
            
        if time.time() > dpll_deadline:
            return False

        # Pure Literal Elimination
        # Check if any variable appears only with one sign in remaining clauses
        var_signs = {}
        for ci in remaining_clauses:
            for v, s in clause_vars[ci]:
                if assignment[v] is None:
                    if v not in var_signs:
                        var_signs[v] = set()
                    var_signs[v].add(s)
                    
        pure_found = False
        for v, signs in var_signs.items():
            if len(signs) == 1:
                assignment[v] = 1 if 1 in signs else 0
                pure_found = True
                
        if pure_found:
            # Re-filter clauses
            new_remaining = []
            for ci in remaining_clauses:
                sat = False
                for v, s in clause_vars[ci]:
                    if assignment[v] is not None and lit_sat(assignment, v, s):
                        sat = True
                        break
                if not sat:
                    new_remaining.append(ci)
            return dpll(assignment, new_remaining)

        if time.time() > dpll_deadline:
            return False

        # Branching
        # Pick variable with highest degree in remaining clauses
        var_count = {}
        for ci in remaining_clauses:
            for v, s in clause_vars[ci]:
                if assignment[v] is None:
                    var_count[v] = var_count.get(v, 0) + 1
        
        if not var_count:
            return True

        best_var = max(var_count, key=lambda v: var_count[v])
        
        # Try True
        assignment[best_var] = 1
        new_remaining = []
        for ci in remaining_clauses:
            sat = False
            for v, s in clause_vars[ci]:
                if assignment[v] is not None and lit_sat(assignment, v, s):
                    sat = True
                    break
            if not sat:
                new_remaining.append(ci)
                
        if dpll(assignment, new_remaining):
            return True
            
        # Try False
        assignment[best_var] = 0
        new_remaining = []
        for ci in remaining_clauses:
            sat = False
            for v, s in clause_vars[ci]:
                if assignment[v] is not None and lit_sat(assignment, v, s):
                    sat = True
                    break
            if not sat:
                new_remaining.append(ci)
                
        if dpll(assignment, new_remaining):
            return True
            
        # Backtrack
        assignment[best_var] = None
        return False

    dpll_assign = [None] * n
    all_clauses = list(range(m))
    if dpll(dpll_assign, all_clauses):
        # Convert to boolean list
        result = [bool(v) for v in dpll_assign if v is not None]
        # If some vars are None (shouldn't happen if all clauses sat, but just in case)
        for i in range(n):
            if dpll_assign[i] is None:
                result.append(False) # Arbitrary
            else:
                pass # Already handled
        # Actually, let's just construct it properly
        result = [False] * n
        for i in range(n):
            if dpll_assign[i] is not None:
                result[i] = bool(dpll_assign[i])
        # Verify
        is_sat = True
        for ci, clause in enumerate(clauses):
            sat = False
            for lit in clause:
                v = abs(lit) - 1
                if result[v] == (lit > 0):
                    sat = True
                    break
            if not sat:
                is_sat = False
                break
        if is_sat:
            return result

    # Phase 2: Optimized WalkSAT
    # State:
    # assignment: list of 0/1
    # clause_sat_count: list of int, number of satisfied literals in each clause
    # unsat_count: total number of unsatisfied clauses
    
    deadline = time.time() + 285.0
    best_assignment = None
    best_unsat = float('inf')
    
    def init_state(assignment):
        clause_sat_count = [0] * m
        for ci in range(m):
            count = 0
            for v, s in clause_vars[ci]:
                if lit_sat(assignment, v, s):
                    count += 1
            clause_sat_count[ci] = count
        unsat = sum(1 for c in clause_sat_count if c == 0)
        return clause_sat_count, unsat

    def flip_var(v, assignment, clause_sat_count, unsat_count):
        """
        Flip variable v, update clause_sat_count and unsat_count.
        Returns new unsat_count.
        """
        old_val = assignment[v]
        new_val = 1 - old_val
        assignment[v] = new_val
        
        delta_unsat = 0
        for ci, s in var_clauses[v]:
            old_sat = lit_sat(assignment, v, s) # This is now the NEW value's satisfaction status
            # Wait, lit_sat uses current assignment.
            # Before flip: val was old_val.
            # After flip: val is new_val.
            
            # Let's compute old_sat_lit and new_sat_lit explicitly to avoid confusion
            # old_val is the value BEFORE flip.
            # new_val is the value AFTER flip.
            
            old_lit_sat = (old_val == 1) if s == 1 else (old_val == 0)
            new_lit_sat = (new_val == 1) if s == 1 else (new_val == 0)
            
            old_count = clause_sat_count[ci]
            
            if old_lit_sat and not new_lit_sat:
                # Literal was satisfied, now unsatisfied
                clause_sat_count[ci] = old_count - 1
                if old_count - 1 == 0:
                    delta_unsat += 1
            elif not old_lit_sat and new_lit_sat:
                # Literal was unsatisfied, now satisfied
                clause_sat_count[ci] = old_count + 1
                if old_count == 0:
                    delta_unsat -= 1
            # else: no change in satisfaction status of this literal
            
        return unsat_count + delta_unsat

    max_restarts = 5000
    base_steps = 500
    
    for restart in range(max_restarts):
        if time.time() >= deadline:
            break
            
        rng = random.Random(restart * 12345 + 6789)
        assignment = [rng.random() < 0.5 for _ in range(n)]
        # Convert to 0/1 for easier logic
        assignment = [1 if x else 0 for x in assignment]
        
        clause_sat_count, unsat_count = init_state(assignment)
        
        if unsat_count == 0:
            return [bool(x) for x in assignment]
            
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment[:]
            
        # Adaptive steps: more steps if close to solution
        steps = base_steps + int(200 * (1.0 - min(unsat_count / max(m, 1), 1.0)))
        
        for step in range(steps):
            if time.time() >= deadline:
                break
                
            if unsat_count == 0:
                return [bool(x) for x in assignment]
                
            # Pick a random unsatisfied clause
            # To pick efficiently, we can just iterate or maintain a list of unsat clauses.
            # For n=50, m~200, iterating is fast enough.
            unsat_clause_idx = -1
            for ci in range(m):
                if clause_sat_count[ci] == 0:
                    unsat_clause_idx = ci
                    break
                    
            if unsat_clause_idx == -1:
                # Should not happen if unsat_count > 0, but safety
                break
                
            # Evaluate gain for each variable in this clause
            best_var = -1
            best_gain = float('inf')
            
            for v, s in clause_vars[unsat_clause_idx]:
                # Calculate gain for flipping v
                # Gain = change in unsat_count
                # We simulate the flip without actually flipping to save time?
                # Or just calculate delta directly.
                
                old_val = assignment[v]
                new_val = 1 - old_val
                
                old_lit_sat = (old_val == 1) if s == 1 else (old_val == 0)
                new_lit_sat = (new_val == 1) if s == 1 else (new_val == 0)
                
                delta = 0
                for ci, s_c in var_clauses[v]:
                    old_count = clause_sat_count[ci]
                    
                    # Determine if this literal's status changes
                    # Note: s_c is the sign in clause ci.
                    # We need to check if the literal in clause ci is satisfied by old_val and new_val.
                    
                    old_sat_c = (old_val == 1) if s_c == 1 else (old_val == 0)
                    new_sat_c = (new_val == 1) if s_c == 1 else (new_val == 0)
                    
                    if old_sat_c and not new_sat_c:
                        # Becomes unsatisfied
                        if old_count == 1:
                            # Clause becomes unsatisfied
                            delta += 1
                    elif not old_sat_c and new_sat_c:
                        # Becomes satisfied
                        if old_count == 0:
                            # Clause becomes satisfied
                            delta -= 1
                            
                # Add small noise to break ties and allow exploration
                noise = rng.random() * 0.1
                score = delta + noise
                
                if score < best_gain:
                    best_gain = score
                    best_var = v
                    
            if best_var != -1:
                unsat_count = flip_var(best_var, assignment, clause_sat_count, unsat_count)
                
                if unsat_count < best_unsat:
                    best_unsat = unsat_count
                    best_assignment = assignment[:]
                    
        if time.time() >= deadline:
            break

    if best_assignment is not None:
        return [bool(x) for x in best_assignment]
    
    # Fallback random
    rng = random.Random(0)
    return [rng.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
