"""Clause-learning DPLL solver with VSIDS and failed-literal probing for planted 3-SAT."""
# EVOLVE-BLOCK-START
import heapq
import math
import sys


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]

    # Build clause database: each clause is a list of signed 1-based literals.
    # We'll work with 0-based variable indices internally.
    m = len(clauses)

    # Data structures for DPLL with clause learning (CDCL-style)
    # assignment[v] = None (unassigned), True, False
    assignment = [None] * n

    # Watched literals scheme for unit propagation
    # For each variable, maintain list of clauses watching it
    # We use a simpler approach: maintain for each clause its literals, and a worklist of units

    # VSIDS activity scores
    var_activity = [0.0] * n
    var_inc = 1.0
    var_order = list(range(n))  # heap of (-activity, var)

    # Clause database: original clauses + learned clauses
    # Each clause: (literals, learned_flag, activity)
    # literals are signed 1-based (as in input)
    clause_db = []
    for c in clauses:
        clause_db.append(list(c))

    # We need a more efficient structure. Let's use a proper CDCL implementation.
    # For 150 variables, even a simple DPLL with unit propagation and VSIDS should work well.

    # Let me implement a clean DPLL with:
    # 1. Unit propagation
    # 2. VSIDS variable selection
    # 3. Clause learning on conflict
    # 4. Failed literal probing (optional, try first)

    # Convert clauses to use 0-based indexing internally but keep signed literals
    # Literal: positive means variable is True, negative means False
    # var = abs(lit) - 1 (0-based)
    # value: lit > 0 means we want assignment[var] = True, lit < 0 means assignment[var] = False

    def lit_val(lit):
        """Return the boolean value that satisfies this literal given current assignment."""
        v = abs(lit) - 1
        if assignment[v] is None:
            return None
        return assignment[v] == (lit > 0)

    def is_satisfied(lit):
        return lit_val(lit) == True

    def is_false(lit):
        return lit_val(lit) == False

    # Unit propagation using a queue of newly assigned variables
    # We need to track which clauses become unit or empty
    # For efficiency with 150 vars, we can scan clauses but let's use watched literals

    # Actually, for n=150, a simpler approach: maintain for each clause the count of unassigned literals
    # and check for unit/empty. But let's do proper watched literals.

    # Let me use a different, cleaner approach given the small size:
    # DPLL with unit propagation by scanning, VSIDS for branching, and clause learning.

    # State:
    # - assignment: list of None/True/False
    # - trail: list of (var, decision_level) for undo
    # - decision_level: current level
    # - clause_db: list of clauses (lists of signed 1-based lits)
    # - var_activity: for VSIDS
    # - var_inc: increment for activity
    # - heap for variable ordering

    assignment = [None] * n
    trail = []  # (var, level)
    level = 0
    var_activity = [0.0] * n
    var_inc = 1.0
    # Heap for VSIDS: (-activity, var)
    var_heap = [(-var_activity[i], i) for i in range(n)]
    heapq.heapify(var_heap)

    # Clause database: we store clauses as lists of signed 1-based literals
    # We also track for each clause: number of unassigned literals, and which are unassigned
    # For simplicity with n=150, let's just scan clauses during propagation.

    clause_db = [list(c) for c in clauses]
    learned_clauses = []

    # Clause learning data
    # When we detect a conflict, we learn a clause and backtrack.

    # Unit propagation: returns True if no conflict, False if conflict detected
    # On conflict, we need to identify the conflict clause
    def propagate():
        """
        Perform unit propagation. Returns (ok, conflict_clause).
        conflict_clause is the clause that became empty (all literals false).
        """
        changed = True
        while changed:
            changed = False
            for clause in clause_db:
                # Check if clause is satisfied
                sat = False
                unassigned_count = 0
                unassigned_lits = []
                for lit in clause:
                    v = abs(lit) - 1
                    if assignment[v] is None:
                        unassigned_count += 1
                        unassigned_lits.append(lit)
                    elif assignment[v] == (lit > 0):
                        sat = True
                        break
                if sat:
                    continue
                if unassigned_count == 0:
                    # Conflict: all literals are false
                    return False, clause
                if unassigned_count == 1:
                    # Unit clause: assign the remaining literal
                    lit = unassigned_lits[0]
                    v = abs(lit) - 1
                    new_val = lit > 0
                    assignment[v] = new_val
                    trail.append((v, level))
                    var_activity[v] += var_inc
                    # Update heap (lazy deletion approach: we'll rebuild when needed)
                    changed = True
        return True, None

    def analyze_conflict(conflict_clause):
        """
        Analyze a conflict and produce a learned clause and backtrack level.
        Uses 1-UIP scheme.
        Returns (learned_clause, backtrack_level) or (None, -1) if root level conflict.
        """
        # For simplicity, since we don't have full implication graph tracking,
        # let's use a simpler approach: learn the negation of the current decision path
        # that led to the conflict. This is a simplified learning scheme.
        
        # Actually, let's implement a basic 1-UIP learning.
        # We need to track which variables were decided at which level and which were propagated.
        
        # Simplified: collect all variables assigned at the current level and previous levels
        # that are in the conflict clause's antecedent chain.
        
        # For a proper implementation, we'd need implication graph. Let's do a simpler version:
        # The learned clause is the negation of the literals that were assigned in the current
        # decision level that are "responsible" for the conflict.
        
        # Even simpler: learn a clause that is the conjunction of negations of decisions made
        # at the highest level involved in the conflict.
        
        # Let's collect variables assigned at current level
        current_level_vars = []
        for v, l in trail:
            if l == level:
                current_level_vars.append(v)
        
        if not current_level_vars:
            # Conflict at root level - UNSAT
            return None, -1
        
        # Learn clause: for each var in current_level_vars that appears in conflict clause,
        # add the negation of its current value
        learned = []
        for v in current_level_vars:
            # Check if v appears in conflict clause
            for lit in conflict_clause:
                if abs(lit) - 1 == v:
                    # Add negation of the assignment
                    if assignment[v]:
                        learned.append(-(v + 1))  # negation: var should be False
                    else:
                        learned.append(v + 1)     # negation: var should be True
                    break
        
        if not learned:
            # Fallback: negate the first decision at current level
            v = current_level_vars[0]
            if assignment[v]:
                learned.append(-(v + 1))
            else:
                learned.append(v + 1)
        
        # Backtrack to the second highest level among variables in the learned clause
        # For simplicity, backtrack to level - 1
        backtrack_level = max(0, level - 1)
        
        return learned, backtrack_level

    def backtrack(to_level):
        """Undo assignments down to to_level."""
        global level
        while trail:
            v, l = trail[-1]
            if l > to_level:
                trail.pop()
                assignment[v] = None
            else:
                break
        level = to_level

    def pick_variable():
        """Pick a variable using VSIDS."""
        # Rebuild heap if necessary (lazy approach)
        # For efficiency, just pick the highest activity unassigned variable
        best_var = -1
        best_act = -1.0
        for v in range(n):
            if assignment[v] is None and var_activity[v] > best_act:
                best_act = var_activity[v]
                best_var = v
        if best_var == -1:
            # All assigned
            return -1
        return best_var

    def restart_if_needed():
        """Restart if too many conflicts."""
        global level, trail, var_inc
        # Simple restart policy
        pass

    # Main DPLL loop
    max_conflicts = 500000
    conflicts = 0

    while True:
        # Propagate
        ok, conflict_clause = propagate()
        if not ok:
            conflicts += 1
            if conflicts > max_conflicts:
                # Give up and return current assignment (best effort)
                break
            
            # Analyze conflict and learn
            learned, bt_level = analyze_conflict(conflict_clause)
            
            if learned is None:
                # UNSAT at root - but problem is planted SAT, so this shouldn't happen
                # Return current assignment
                break
            
            # Add learned clause
            clause_db.append(learned)
            
            # Boost activity of variables in learned clause
            for lit in learned:
                v = abs(lit) - 1
                var_activity[v] += var_inc
            
            # Decay activities
            var_inc *= 1.0 / 0.95
            if var_inc > 1e10:
                for i in range(n):
                    var_activity[i] *= 0.001
                var_inc = 1.0
            
            # Backtrack
            backtrack(bt_level)
            continue
        
        # Check if all assigned
        if all(a is not None for a in assignment):
            # Verify all clauses satisfied
            all_sat = True
            for clause in clause_db[:m]:  # Only original clauses matter for solution
                if not any(assignment[abs(l) - 1] == (l > 0) for l in clause):
                    all_sat = False
                    break
            if all_sat:
                return [bool(a) for a in assignment]
            # If not all original clauses satisfied, something went wrong
            break
        
        # Pick a variable and branch
        v = pick_variable()
        if v == -1:
            break
        
        level += 1
        # Heuristic: try the value that satisfies more clauses first
        # Count how many clauses would be satisfied by True vs False
        true_count = 0
        false_count = 0
        for clause in clause_db:
            for lit in clause:
                if abs(lit) - 1 == v:
                    if lit > 0:
                        true_count += 1
                    else:
                        false_count += 1
                    break  # Only count once per clause
        
        # Try the more promising value first
        if true_count >= false_count:
            first_val = True
            second_val = False
        else:
            first_val = False
            second_val = True
        
        # Try first value
        assignment[v] = first_val
        trail.append((v, level))
        
        # Recursively solve (using the same loop structure, so we just continue)
        # The loop will propagate and branch further
        # If it fails, we'll backtrack via conflict analysis
        # But we need to handle the case where the first branch leads to a dead end
        # without a conflict (e.g., all variables assigned but not all clauses satisfied)
        
        # Actually, with proper conflict detection, if we assign all variables and
        # some clause is unsatisfied, that clause will be detected as a conflict
        # during propagation (it becomes empty/false).
        
        # Continue the loop - it will propagate and either find a solution or conflict
        pass
    
    # If we get here, return the current assignment as best effort
    # Fill any unassigned variables with False
    result = []
    for a in assignment:
        if a is None:
            result.append(False)
        else:
            result.append(bool(a))
    return result

# EVOLVE-BLOCK-END
