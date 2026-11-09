import random
import time
from collections import deque

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    rng = random.Random(0)
    
    # Preprocessing: Map variable to list of (clause_index, is_positive)
    var_to_clauses = [[] for _ in range(n)]
    for ci, clause in enumerate(clauses):
        for l in clause:
            v = abs(l) - 1
            is_pos = l > 0
            var_to_clauses[v].append((ci, is_pos))
    
    m = len(clauses)
    
    def dpll_with_unit_propagation(clauses, n, deadline):
        """
        Try DPLL with unit propagation. Returns (success, assignment) 
        where assignment is a list of n booleans (None if failed).
        """
        # assignment: -1 = unassigned, 0 = False, 1 = True
        assignment = [-1] * n
        
        # Worklist of unit clauses (variables that must be set)
        unit_queue = deque()
        
        # Track clause satisfaction state
        # clause_status: 0 = unsatisfied, 1 = satisfied, 2 = false
        clause_status = [0] * len(clauses)
        clause_true_count = [0] * len(clauses)
        clause_false_count = [0] * len(clauses)
        
        # Initialize clause counts
        for ci, clause in enumerate(clauses):
            for l in clause:
                v = abs(l) - 1
                # Initially all unassigned, so count is 0
                pass
            clause_true_count[ci] = 0
            clause_false_count[ci] = 0
        
        # Set a variable and propagate
        def set_variable(v, val):
            """Set variable v to val (0 or 1). Return False if conflict."""
            if assignment[v] != -1:
                return assignment[v] == val
            
            assignment[v] = val
            
            # Update all clauses containing v
            for (ci, is_pos) in var_to_clauses[v]:
                if clause_status[ci] == 2:  # Already false
                    continue
                    
                if is_pos == (val == 1):
                    # This literal is satisfied
                    clause_true_count[ci] += 1
                    if clause_true_count[ci] >= 1:
                        clause_status[ci] = 1
                else:
                    # This literal is false
                    clause_false_count[ci] += 1
                    # Check if clause becomes false (all literals false)
                    if clause_false_count[ci] == len(clauses[ci]):
                        clause_status[ci] = 2
                        return False  # Conflict!
            
            return True
        
        # Initialize: find any unit clauses (none initially since all unassigned)
        # Actually, we need to check for clauses that become unit after assignments
        
        # For now, let's do a simple approach: try to find a satisfying assignment
        # using DPLL with unit propagation and pure literal elimination
        
        # Check for pure literals
        def find_pure_literal():
            """Find a variable that only appears in one polarity."""
            for v in range(n):
                if assignment[v] != -1:
                    continue
                has_pos = False
                has_neg = False
                for (ci, is_pos) in var_to_clauses[v]:
                    if clause_status[ci] == 2:
                        continue
                    if is_pos:
                        has_pos = True
                    else:
                        has_neg = True
                    if has_pos and has_neg:
                        break
                if has_pos and not has_neg:
                    return v, True
                elif has_neg and not has_pos:
                    return v, False
            return None, None
        
        # Check for unit clauses
        def find_unit_clause():
            """Find a clause with exactly one unassigned literal."""
            for ci, clause in enumerate(clauses):
                if clause_status[ci] == 1:  # Already satisfied
                    continue
                if clause_status[ci] == 2:  # False
                    return None, None, True  # Conflict
                
                unassigned = []
                for l in clause:
                    v = abs(l) - 1
                    if assignment[v] == -1:
                        unassigned.append(l)
                
                if len(unassigned) == 1:
                    return ci, unassigned[0], False
                elif len(unassigned) == 0:
                    return None, None, True  # Conflict: empty clause
                
            return None, None, False
        
        # Main DPLL loop
        while time.time() < deadline:
            # Check if all clauses satisfied
            all_satisfied = True
            for ci in range(len(clauses)):
                if clause_status[ci] == 0:
                    all_satisfied = False
                    break
                if clause_status[ci] == 2:
                    return False, None
            
            if all_satisfied:
                # Convert to boolean list
                result = [False] * n
                for v in range(n):
                    if assignment[v] == 1:
                        result[v] = True
                    elif assignment[v] == -1:
                        result[v] = rng.random() < 0.5
                    # else False
                return True, result
            
            # Try unit propagation
            ci, unit_lit, conflict = find_unit_clause()
            if conflict:
                return False, None
            if unit_lit is not None:
                v = abs(unit_lit) - 1
                val = 1 if unit_lit > 0 else 0
                if not set_variable(v, val):
                    return False, None
                continue
            
            # Try pure literal elimination
            v, val = find_pure_literal()
            if v is not None:
                if not set_variable(v, 1 if val else 0):
                    return False, None
                continue
            
            # No unit clauses or pure literals, pick a variable to branch on
            # Pick variable that appears in most unsatisfied clauses
            best_v = -1
            best_count = -1
            for v in range(n):
                if assignment[v] != -1:
                    continue
                count = 0
                for (ci, _) in var_to_clauses[v]:
                    if clause_status[ci] == 0:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_v = v
            
            if best_v == -1:
                # All variables assigned but not all satisfied? Shouldn't happen
                return False, None
            
            # Branch: try True first
            if set_variable(best_v, 1):
                # Check if this leads to solution
                # For simplicity, we'll just continue the loop
                pass
            else:
                # Conflict, try False
                if not set_variable(best_v, 0):
                    return False, None
        
        return False, None
    
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s
    
    # First, try DPLL with unit propagation for a short time
    dpll_deadline = start_time + 5  # Give DPLL 5 seconds
    success, dpll_assignment = dpll_with_unit_propagation(clauses, n, dpll_deadline)
    if success:
        return dpll_assignment
    
    # Fall back to WalkSAT
    def initialize():
        """Initialize assignment and clause counts."""
        assignment = [rng.random() < 0.5 for _ in range(n)]
        clause_true_count = [0] * m
        for ci, clause in enumerate(clauses):
            count = 0
            for l in clause:
                v = abs(l) - 1
                if assignment[v] == (l > 0):
                    count += 1
            clause_true_count[ci] = count
        return assignment, clause_true_count
    
    def update_clause(ci, assignment):
        """Recalculate true count for a specific clause."""
        count = 0
        for l in clauses[ci]:
            v = abs(l) - 1
            if assignment[v] == (l > 0):
                count += 1
        return count
    
    def is_solved(clause_true_count):
        return all(c >= 1 for c in clause_true_count)
    
    assignment, clause_true_count = initialize()
    
    restart_threshold = 2000
    steps = 0
    p_walk = 0.2  # Probability to pick a random literal instead of best
    
    while time.time() < deadline:
        if is_solved(clause_true_count):
            return assignment
        
        # Check for restart
        if steps >= restart_threshold:
            assignment, clause_true_count = initialize()
            steps = 0
            continue
        
        # Find unsatisfied clauses
        unsat_indices = [i for i, c in enumerate(clause_true_count) if c == 0]
        if not unsat_indices:
            return assignment
        
        # Select clause: Pick one with minimum true count (all are 0 here, so random)
        ci = rng.choice(unsat_indices)
        
        # Select variable to flip
        best_var = -1
        best_gain = -1000
        gains = []
        
        for l in clauses[ci]:
            v = abs(l) - 1
            # Calculate gain of flipping v
            gain = 0
            for (other_ci, is_pos) in var_to_clauses[v]:
                # Check current state of this variable in other_ci
                if assignment[v] == is_pos:
                    gain -= 1
                else:
                    gain += 1
            gains.append((gain, v))
            
            if gain > best_gain:
                best_gain = gain
                best_var = v
        
        # Apply WalkSAT heuristic: with prob p_walk, pick random, else pick best
        if rng.random() < p_walk:
            # Pick random variable from the clause
            v = rng.choice([abs(l) - 1 for l in clauses[ci]])
        else:
            v = best_var
        
        # Flip variable v
        assignment[v] = not assignment[v]
        
        # Update clause_true_count for all clauses containing v
        for (affected_ci, _) in var_to_clauses[v]:
            clause_true_count[affected_ci] = update_clause(affected_ci, assignment)
            
        steps += 1
    
    # If deadline reached, return best effort
    return assignment
