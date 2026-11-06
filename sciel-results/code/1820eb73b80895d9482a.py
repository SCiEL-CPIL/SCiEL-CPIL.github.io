"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random

def solve(instance):
    """
    Solve a 3-CNF SAT instance using DPLL with unit propagation and pure literal elimination.
    If DPLL fails or times out, fallback to WalkSAT.
    Returns a list of booleans representing the satisfying assignment.
    """
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline guard: 330s limit, use 300s for safety
    start_time = time.time()
    deadline = start_time + 300.0
    
    def check_time():
        return time.time() > deadline

    def get_var(lit):
        return abs(lit) - 1

    def is_satisfied_clause(clause, assign):
        for lit in clause:
            v = get_var(lit)
            if (lit > 0 and assign[v]) or (lit < 0 and not assign[v]):
                return True
        return False

    # --- DPLL Implementation ---
    
    def simplify_clauses(clauses, assign):
        new_clauses = []
        for clause in clauses:
            new_clause = []
            satisfied = False
            for lit in clause:
                v = get_var(lit)
                if v in assign:
                    val = assign[v]
                    if (lit > 0 and val) or (lit < 0 and not val):
                        satisfied = True
                        break
                else:
                    new_clause.append(lit)
            
            if satisfied:
                continue
            
            if len(new_clause) == 0:
                return None # Contradiction
            
            new_clauses.append(new_clause)
        
        return new_clauses

    def unit_propagate(clauses, assign):
        changed = True
        while changed:
            changed = False
            unit_clause = None
            for c in clauses:
                if len(c) == 1:
                    unit_clause = c
                    break
            
            if not unit_clause:
                break
            
            lit = unit_clause[0]
            v = get_var(lit)
            
            if v in assign:
                val = assign[v]
                if (lit > 0 and not val) or (lit < 0 and val):
                    return False, None, None
                # If satisfied, it should be removed.
                # If we are here, it's not removed.
                # Remove it.
                clauses = [c for c in clauses if c != unit_clause]
                continue

            assign[v] = (lit > 0)
            changed = True
            clauses = simplify_clauses(clauses, assign)
            if clauses is None:
                return False, None, None
            
            if check_time():
                return True, clauses, assign
        
        return True, clauses, assign

    def find_pure_literal(clauses):
        if not clauses:
            return None, None
        
        pos_vars = set()
        neg_vars = set()
        
        for clause in clauses:
            for lit in clause:
                v = get_var(lit)
                if lit > 0:
                    pos_vars.add(v)
                else:
                    neg_vars.add(v)
        
        for v in range(n):
            in_pos = v in pos_vars
            in_neg = v in neg_vars
            
            if in_pos and not in_neg:
                return v + 1, v
            if in_neg and not in_pos:
                return -(v + 1), v
                
        return None, None

    def select_variable(clauses, assign):
        counts = {}
        for c in clauses:
            for lit in c:
                v = get_var(lit)
                if v not in assign:
                    counts[v] = counts.get(v, 0) + 1
        
        if not counts:
            return None, None
            
        max_count = max(counts.values())
        candidates = [v for v, c in counts.items() if c == max_count]
        v = random.choice(candidates)
        
        for c in clauses:
            for lit in c:
                if get_var(lit) == v:
                    return v, lit
        return None, None

    def dpll(clauses, assign):
        if check_time():
            return assign
        
        if not clauses:
            return assign
        
        success, clauses, assign = unit_propagate(clauses, assign)
        if not success:
            return None
        if check_time():
            return assign
            
        if not clauses:
            return assign
        
        lit, v = find_pure_literal(clauses)
        if lit is not None:
            assign[v] = (lit > 0)
            clauses = simplify_clauses(clauses, assign)
            if clauses is None:
                return None
            if check_time():
                return assign
            return dpll(clauses, assign)
        
        v, lit = select_variable(clauses, assign)
        if v is None:
            return None

        assign_copy = assign.copy()
        assign_copy[v] = (lit > 0)
        clauses_1 = simplify_clauses(clauses, assign_copy)
        if clauses_1 is not None:
            result = dpll(clauses_1, assign_copy)
            if result is not None:
                return result
        
        assign_copy = assign.copy()
        assign_copy[v] = not (lit > 0)
        clauses_2 = simplify_clauses(clauses, assign_copy)
        if clauses_2 is not None:
            result = dpll(clauses_2, assign_copy)
            if result is not None:
                return result
        
        return None

    # Try DPLL first
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(10000)
    
    result_assign = None
    try:
        current_clauses = simplify_clauses(clauses, {})
        if current_clauses is not None:
            result_assign = dpll(current_clauses, {})
    finally:
        sys.setrecursionlimit(old_limit)

    # Prepare assignment for WalkSAT
    if result_assign is not None and len(result_assign) == n:
        final_assign = [False] * n
        for v, val in result_assign.items():
            if 0 <= v < n:
                final_assign[v] = val
        # Verify
        if all(is_satisfied_clause(c, final_assign) for c in clauses):
            return final_assign

    # Fallback: WalkSAT
    if result_assign is not None:
        assignment = [False] * n
        for v, val in result_assign.items():
            if 0 <= v < n:
                assignment[v] = val
    else:
        assignment = [random.random() < 0.5 for _ in range(n)]
    
    rng = random.Random(42)
    
    # Build var_to_clauses
    var_to_clauses = {}
    for i, c in enumerate(clauses):
        for lit in c:
            v = get_var(lit)
            if v not in var_to_clauses:
                var_to_clauses[v] = []
            var_to_clauses[v].append(i)
    
    while time.time() < deadline - 1.0:
        unsat_indices = [i for i, c in enumerate(clauses) if not is_satisfied_clause(c, assignment)]
        
        if not unsat_indices:
            return assignment
        
        idx = rng.choice(unsat_indices)
        clause = clauses[idx]
        
        best_var = None
        best_delta = float('inf')
        
        for lit in clause:
            v = get_var(lit)
            
            delta = 0
            for ci in var_to_clauses.get(v, []):
                c = clauses[ci]
                
                cur_sat = is_satisfied_clause(c, assignment)
                
                old_val = assignment[v]
                assignment[v] = not old_val
                new_sat = is_satisfied_clause(c, assignment)
                assignment[v] = old_val
                
                if cur_sat and not new_sat:
                    delta += 1
                elif not cur_sat and new_sat:
                    delta -= 1
            
            noisy_delta = delta + rng.random() * 0.5
            if noisy_delta < best_delta:
                best_delta = noisy_delta
                best_var = v
        
        if best_var is not None:
            assignment[best_var] = not assignment[best_var]
            
    return assignment

# EVOLVE-BLOCK-END
