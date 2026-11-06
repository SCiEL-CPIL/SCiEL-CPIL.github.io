"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict


def solve(instance):
    # DPLL with unit propagation, pure literal elimination, and watched literals.
    n, clauses = instance["n"], instance["clauses"]
    
    # Convert to 0-based literals: positive var -> var-1, negative var -> -(var-1)-1
    # We use a representation where literal l in 1-based signed form:
    #   var = abs(l) - 1
    #   positive if l > 0
    # For internal use, we'll keep literals as-is but map to 0-based var index.
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290  # 290s margin before 330s kill
    
    # Normalize clauses to use 0-based variable indices
    # Each literal becomes (var_idx, is_positive)
    # Or we can keep signed 1-based and just use abs(l)-1 for indexing
    
    def get_var(lit):
        return abs(lit) - 1
    
    def is_pos(lit):
        return lit > 0
    
    # Build watched literals: for each variable, list of (clause_idx, other_lit)
    # where the clause contains var and other_lit is the other literal in the clause
    # Actually, watched literals: for each variable, list of clause indices that watch it
    
    # Let's use a different approach: maintain for each variable a set of clauses that contain it
    # And for each clause, track which literals are satisfied/unsatisfied
    
    # Simpler approach: DPLL with clause list, unit propagation via scanning
    # For 50 variables and typical clause counts, this should be fast enough
    
    # Let's use a more efficient DPLL with watched literals
    
    # Convert clauses to 0-based
    clauses_0 = []
    for clause in clauses:
        clauses_0.append([(abs(l) - 1, l > 0) for l in clause])
    
    m = len(clauses_0)
    
    # Assignment: -1 = unassigned, 0 = False, 1 = True
    assignment = [-1] * n
    
    # For each variable, maintain list of clause indices that contain it
    var_clauses = defaultdict(list)
    for ci, clause in enumerate(clauses_0):
        for var, pos in clause:
            var_clauses[var].append(ci)
    
    # For each clause, track state: satisfied, falsified, or active
    # clause_state[ci] = 'sat', 'unsat', or 'active'
    # Also track for each clause which literals are assigned
    
    def check_clause(ci):
        """Return 'sat', 'unsat', or 'active' for clause ci given current assignment."""
        clause = clauses_0[ci]
        has_unassigned = False
        for var, pos in clause:
            val = assignment[var]
            if val == -1:
                has_unassigned = True
            else:
                if (val == 1) == pos:
                    return 'sat'
        if has_unassigned:
            return 'active'
        return 'unsat'
    
    def unit_propagate():
        """Apply unit propagation. Returns False if contradiction, True otherwise."""
        changed = True
        while changed:
            changed = False
            # Find unit clauses
            for ci in range(m):
                clause = clauses_0[ci]
                unassigned_lits = []
                all_assigned = True
                for var, pos in clause:
                    val = assignment[var]
                    if val == -1:
                        unassigned_lits.append((var, pos))
                        all_assigned = False
                    else:
                        if (val == 1) == pos:
                            break  # clause satisfied
                if all_assigned and not unassigned_lits:
                    # Check if clause is unsatisfied
                    sat = False
                    for var, pos in clause:
                        val = assignment[var]
                        if val != -1 and (val == 1) == pos:
                            sat = True
                            break
                    if not sat:
                        return False  # contradiction
                elif len(unassigned_lits) == 1:
                    # Unit clause - must assign
                    var, pos = unassigned_lits[0]
                    assignment[var] = 1 if pos else 0
                    changed = True
        return True
    
    def pure_literal_elimination():
        """Apply pure literal elimination. Returns True if any change made."""
        changed = False
        # For each unassigned variable, check if it appears in only one polarity
        for var in range(n):
            if assignment[var] != -1:
                continue
            # Check all clauses containing this variable
            has_pos = False
            has_neg = False
            for ci in var_clauses[var]:
                clause = clauses_0[ci]
                for v, p in clause:
                    if v == var:
                        if p:
                            has_pos = True
                        else:
                            has_neg = True
            if has_pos and not has_neg:
                assignment[var] = 1
                changed = True
            elif has_neg and not has_pos:
                assignment[var] = 0
                changed = True
        return changed
    
    # Main DPLL
    def dpll():
        """Returns True if satisfiable, False if not."""
        # Check deadline
        if time.time() > deadline:
            return None  # timeout
        
        # Apply pure literal elimination
        if pure_literal_elimination():
            if not unit_propagate():
                return False
            # After assignment, check if all clauses are satisfied
            for ci in range(m):
                sat = False
                for var, pos in clauses_0[ci]:
                    val = assignment[var]
                    if val != -1 and (val == 1) == pos:
                        sat = True
                        break
                    if val == -1:
                        continue
                if not sat:
                    # Check if clause is actually unsatisfied
                    all_assigned = True
                    for var, pos in clauses_0[ci]:
                        if assignment[var] == -1:
                            all_assigned = False
                            break
                    if all_assigned:
                        return False
                    break
            # Check if all variables assigned
            if all(a != -1 for a in assignment):
                # Verify all clauses
                for ci in range(m):
                    sat = False
                    for var, pos in clauses_0[ci]:
                        val = assignment[var]
                        if val != -1 and (val == 1) == pos:
                            sat = True
                            break
                    if not sat:
                        return False
                return True
            # Continue with DPLL
            return dpll()
        
        # Check if all variables assigned
        if all(a != -1 for a in assignment):
            for ci in range(m):
                sat = False
                for var, pos in clauses_0[ci]:
                    val = assignment[var]
                    if val != -1 and (val == 1) == pos:
                        sat = True
                        break
                if not sat:
                    return False
            return True
        
        # Pick a variable to branch on
        # Choose variable that appears in most unsatisfied/unit clauses
        # Simple heuristic: pick first unassigned variable
        var_to_branch = None
        for var in range(n):
            if assignment[var] == -1:
                var_to_branch = var
                break
        
        if var_to_branch is None:
            return True  # all assigned
        
        # Try True first
        assignment[var_to_branch] = 1
        if unit_propagate():
            result = dpll()
            if result is not None:
                if result:
                    return True
            else:
                # Timeout
                assignment[var_to_branch] = -1
                return None
        assignment[var_to_branch] = -1
        
        # Try False
        assignment[var_to_branch] = 0
        if unit_propagate():
            result = dpll()
            if result is not None:
                if result:
                    return True
            else:
                # Timeout
                assignment[var_to_branch] = -1
                return None
        assignment[var_to_branch] = -1
        
        return False
    
    result = dpll()
    
    if result is True:
        # Convert assignment to list of booleans
        return [bool(a) for a in assignment]
    elif result is None:
        # Timeout - return current best effort
        # Fill unassigned with False
        return [bool(a) if a != -1 else False for a in assignment]
    else:
        # UNSAT - but problem says satisfiable, so return best effort
        return [bool(a) if a != -1 else False for a in assignment]
# EVOLVE-BLOCK-END
