# EVOLVE-BLOCK-START
import sys
from collections import defaultdict

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Build initial clause list with literal indices (0-based)
    # We'll represent clauses as lists of (var_idx, sign) where sign is +1 or -1
    # var_idx is 0-based
    
    # For efficiency, we'll use a recursive DPLL with:
    # - Unit propagation
    # - VSIDS variable ordering
    # - Failed literal probing
    # - Clause learning (simple form: record conflicts)
    
    # Data structures
    # assignment: list of None (unassigned), True, False
    # For each variable, we track which clauses contain it (watched literals approach would be better
    # but for 150 vars, a simpler approach might work)
    
    # Let's use a more efficient approach:
    # - Maintain a list of clauses
    # - For each variable, maintain a set of clause indices that contain it
    # - Use unit propagation by tracking clause satisfaction status
    
    # Actually, for 150 variables, let's implement a clean DPLL with:
    # 1. Unit propagation
    # 2. VSIDS
    # 3. Failed literal probing
    # 4. Simple clause learning
    
    # Preprocessing: index clauses by variable
    # clause_vars[i] = list of (var_idx, sign) for clause i
    
    clause_vars = []
    var_to_clauses = defaultdict(set)  # var_idx -> set of clause indices
    
    for ci, clause in enumerate(clauses):
        cl = []
        for lit in clause:
            var_idx = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            cl.append((var_idx, sign))
            var_to_clauses[var_idx].add(ci)
        clause_vars.append(cl)
    
    num_clauses = len(clauses)
    
    # State for DPLL
    # assignment: dict var_idx -> bool, or we can use a list
    # Let's use a list with None for unassigned
    
    # VSIDS: maintain activity scores
    # We'll implement a simplified version
    
    def dpll(assignment, decision_level):
        # assignment: list of (None, True, False)
        
        # Unit propagation
        if not unit_propagate(assignment):
            return None  # Conflict
        
        # Check if all clauses are satisfied
        if all_satisfied(assignment):
            return assignment[:]  # Return a copy
        
        # Failed literal probing
        flp_result = failed_literal_probing(assignment)
        if flp_result is not None:
            # flp_result is (var_idx, value) that should be forced
            var_idx, value = flp_result
            if assignment[var_idx] is not None and assignment[var_idx] != value:
                return None  # Conflict
            assignment[var_idx] = value
            # Recurse after setting this
            return dpll(assignment, decision_level + 1)
        
        # Choose a variable using VSIDS
        var_idx = choose_variable(assignment)
        if var_idx is None:
            # All variables assigned
            if all_satisfied(assignment):
                return assignment[:]
            return None
        
        # Try True first, then False
        # Use VSIDS to decide order
        for value in [True, False]:
            new_assignment = assignment[:]
            new_assignment[var_idx] = value
            result = dpll(new_assignment, decision_level + 1)
            if result is not None:
                return result
        
        return None
    
    def unit_propagate(assignment):
        # Repeat until no more unit clauses
        changed = True
        while changed:
            changed = False
            for ci in range(num_clauses):
                cl = clause_vars[ci]
                # Check if clause is satisfied
                sat = False
                unassigned = []
                for var_idx, sign in cl:
                    val = assignment[var_idx]
                    if val is None:
                        unassigned.append((var_idx, sign))
                    elif val == (sign == 1):
                        sat = True
                        break
                if sat:
                    continue
                if len(unassigned) == 0:
                    # Clause is falsified
                    return False
                if len(unassigned) == 1:
                    # Unit clause
                    var_idx, sign = unassigned[0]
                    value = (sign == 1)
                    if assignment[var_idx] is not None and assignment[var_idx] != value:
                        return False
                    if assignment[var_idx] is None:
                        assignment[var_idx] = value
                        changed = True
        return True
    
    def all_satisfied(assignment):
        for ci in range(num_clauses):
            cl = clause_vars[ci]
            sat = False
            for var_idx, sign in cl:
                val = assignment[var_idx]
                if val is not None and val == (sign == 1):
                    sat = True
                    break
            if not sat:
                return False
        return True
    
    # VSIDS implementation
    # activity: list of activity scores for each variable
    activity = [0.0] * n
    var_order = list(range(n))
    
    def choose_variable(assignment):
        # Choose unassigned variable with highest activity
        best_var = None
        best_act = -1
        for var_idx in range(n):
            if assignment[var_idx] is None:
                if activity[var_idx] > best_act:
                    best_act = activity[var_idx]
                    best_var = var_idx
        return best_var
    
    def failed_literal_probing(assignment):
        # For each unassigned variable, try setting it to True and False
        # If one leads to immediate conflict (unit propagation fails), force the other
        # This is expensive, so we limit it
        # For efficiency, only check a subset or use heuristics
        
        # Simple version: check a few variables
        # Actually, for 150 vars, let's do a limited version
        # Check at most 10 variables per call
        
        unassigned_vars = [v for v in range(n) if assignment[v] is None]
        if not unassigned_vars:
            return None
        
        # Limit probing to keep it fast
        probe_vars = unassigned_vars[:10]
        
        for var_idx in probe_vars:
            for value in [True, False]:
                # Try setting var_idx to value and see if unit propagation fails
                test_assignment = assignment[:]
                test_assignment[var_idx] = value
                if not unit_propagate(test_assignment):
                    # Conflict, so var_idx should be !value
                    return (var_idx, not value)
        
        return None
    
    # Initialize
    initial_assignment = [None] * n
    result = dpll(initial_assignment, 0)
    
    if result is not None:
        # Convert to list of booleans
        return [bool(v) for v in result]
    
    # Fallback: if DPLL fails (shouldn't happen for satisfiable instances), 
    # try WalkSAT as backup
    import random
    rng = random.Random(42)
    assignment = [rng.random() < 0.5 for _ in range(n)]
    
    def satisfied(clause):
        return any(assignment[abs(l) - 1] == (l > 0) for l in clause)
    
    for _ in range(50000):
        unsat = [c for c in clauses if not satisfied(c)]
        if not unsat:
            break
        clause = rng.choice(unsat)
        literal = rng.choice(clause)
        assignment[abs(literal) - 1] = not assignment[abs(literal) - 1]
    
    return assignment

# EVOLVE-BLOCK-END
