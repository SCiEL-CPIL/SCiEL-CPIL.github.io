"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(10000)

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # CDCL Solver Implementation
    # Literals are 1-based signed integers. Var i is in [1, n].
    # value[lit] is not used directly, we use value[var] which is 0, 1, or None.
    
    # Data structures
    m = len(clauses)
    # Convert clauses to list of (var, sign) for faster processing? 
    # Or keep signed literals and use helper functions.
    # For speed in Python, pre-processing clauses into (var, sign) tuples might help, 
    # but standard signed literal handling is also fine if optimized.
    
    # Let's stick to signed literals for clause storage to match input.
    # var = abs(lit), sign = 1 if lit>0 else -1.
    
    # Watchlists: dict from literal to list of clause indices
    # We watch clauses on their non-unit literals.
    # Initially, all clauses are watched on their first two literals.
    watched = {lit: [] for lit in range(-n, n+1) if lit != 0}
    
    # Initialize watchlists
    for i, clause in enumerate(clauses):
        # Assume clauses are length 3
        # Watch on first two literals
        l1, l2 = clause[0], clause[1]
        watched[l1].append(i)
        watched[l2].append(i)
        
    # Value assignment: 0 for False, 1 for True, None for Unassigned
    # We index by variable number (1..n)
    value = [None] * (n + 1)
    level = [0] * (n + 1)
    reason = [-1] * (n + 1)
    
    # Trail of assigned variables
    trail = []
    trail_lim = []
    
    # Activity for variable selection (VSIDS-like)
    activity = [0.0] * (n + 1)
    var_inc = 1.0
    var_decay = 0.95
    
    # Clause learning
    learnt_clauses = []
    
    def is_true(lit):
        v = abs(lit)
        s = 1 if lit > 0 else -1
        val = value[v]
        if val is None:
            return False
        return (val == 1 and s == 1) or (val == 0 and s == -1)
        
    def is_false(lit):
        v = abs(lit)
        s = 1 if lit > 0 else -1
        val = value[v]
        if val is None:
            return False
        return (val == 1 and s == -1) or (val == 0 and s == 1)
        
    def enqueue(var, lev, rs):
        value[var] = 1 if rs else 0 # Wait, rs is reason clause, we need to know the value.
        # Actually, enqueue is called with the value determined by propagation.
        # Let's refactor: enqueue(var, val, lev, rs)
        pass # Placeholder, will define properly below
        
    # Proper enqueue
    def _enqueue(var, val, lev, rs):
        value[var] = val
        level[var] = lev
        reason[var] = rs
        trail.append(var)
        
    def propagate():
        while len(trail) > len(trail_lim):
            p = trail[-1]
            trail.pop()
            # p was just assigned
            # Check watched clauses for literal p (True) and -p (False)
            # If p is True, check clauses watching -p (which is false)
            # If p is False, check clauses watching p (which is false)
            
            # Determine which literal is now false
            if value[p] == 1:
                false_lit = -p
            else:
                false_lit = p
                
            # Iterate over clauses watching this false literal
            # We must be careful with modifying the list while iterating
            i = 0
            watchers = watched[false_lit]
            while i < len(watchers):
                ci = watchers[i]
                clause = clauses[ci] if ci < m else learnt_clauses[ci - m]
                
                # Check if clause is satisfied
                sat = False
                # Check all literals in clause
                # Optimization: check if any literal is true
                for lit in clause:
                    if is_true(lit):
                        sat = True
                        break
                if sat:
                    i += 1
                    continue
                    
                # Check if clause is unit (only one unassigned literal)
                # If it is unit, enqueue that literal
                # If it is empty (0 unassigned), conflict
                
                unassigned_lit = None
                unassigned_count = 0
                # We know at least one literal is false (false_lit)
                # Check others
                for lit in clause:
                    if lit == false_lit:
                        continue
                    if value[abs(lit)] is None:
                        unassigned_lit = lit
                        unassigned_count += 1
                        if unassigned_count > 1:
                            break # More than one unassigned, not unit
                    
                if unassigned_count == 0:
                    # Conflict
                    return ci # Return conflicting clause index
                elif unassigned_count == 1:
                    # Unit clause
                    # Enqueue unassigned_lit
                    v = abs(unassigned_lit)
                    val = 1 if unassigned_lit > 0 else 0
                    lev = level[false_lit] # Inherited level
                    rs = ci
                    _enqueue(v, val, lev, rs)
                    
                    # Update watchlists: remove this clause from watching false_lit
                    # and add it to watching unassigned_lit
                    # Actually, we swap the watched literal.
                    # We want to watch unassigned_lit instead of false_lit.
                    # Remove ci from watched[false_lit]
                    # Add ci to watched[unassigned_lit]
                    
                    # Efficient removal: swap with last element
                    last = watchers[-1]
                    watchers[i] = last
                    watchers.pop()
                    watched[unassigned_lit].append(ci)
                    # Do not increment i, because we swapped a new clause in
                else:
                    i += 1
                    
        return -1 # No conflict
        
    def analyze(conflict_clause_idx):
        # Compute learnt clause and backjump level
        clause = clauses[conflict_clause_idx] if conflict_clause_idx < m else learnt_clauses[conflict_clause_idx - m]
        
        seen = set()
        learnt = []
        cnt = 0
        max_level = 0
        
        p = -1 # Variable to resolve
        # Initialize with the conflicting clause
        # We resolve until we have one literal at the current decision level (or none, if root)
        
        # Standard analysis:
        # 1. Add all literals in the conflict clause that are at the current level to 'learnt'
        # 2. Resolve with reasons for those literals
        
        # Let's use a simpler, robust analysis for small instances
        # This is a simplified version of MiniSat's analysis
        
        # Get the literals in the conflict clause
        # We need to find the "asserting" literal
        
        # Variables involved in the conflict clause at the current level
        current_level = len(trail_lim) - 1
        
        # Collect variables to resolve
        to_resolve = []
        for lit in clause:
            v = abs(lit)
            if level[v] == current_level:
                to_resolve.append(v)
                
        # If no variables at current level, it's a root conflict
        if not to_resolve:
            # Learnt clause is the negation of the conflict clause
            # Backjump to 0
            learnt_clause = [-lit for lit in clause]
            # Remove duplicates? Usually not needed if clauses are simplified
            return learnt_clause, 0

        # Resolve
        # We maintain a set of seen variables to avoid cycles
        # We build the learnt clause incrementally
        
        # This simplified analysis might be tricky to get right quickly.
        # Alternative: Use a simpler "first UIP" approximation or just backjump to the max level of the conflict clause.
        # For n=50, a simpler backjump strategy might suffice.
        # Let's try a simple backjump: backjump to the max level among variables in the conflict clause.
        
        max_lev = 0
        for lit in clause:
            v = abs(lit)
            if level[v] > max_lev:
                max_lev = level[v]
                
        # Create learnt clause: negate all literals in the conflict clause
        # This is a valid learnt clause (it must be satisfied).
        # It might not be the "best" learnt clause, but it is correct.
        learnt_clause = [-lit for lit in clause]
        
        # Simplify learnt clause: remove literals that are false at the current state?
        # No, learnt clause must be satisfied in the current state (after backjump).
        # The negation of the conflict clause is satisfied if the conflict clause is falsified.
        
        return learnt_clause, max_lev

    def pick_var():
        # Pick an unassigned variable with highest activity
        best_var = -1
        best_act = -1.0
        for v in range(1, n + 1):
            if value[v] is None:
                if activity[v] > best_act:
                    best_act = activity[v]
                    best_var = v
        if best_var == -1:
            return -1
        return best_var

    def decay_activity():
        nonlocal var_inc
        var_inc /= var_decay
        for v in range(1, n + 1):
            activity[v] /= var_decay

    def bump_activity(var):
        nonlocal var_inc
        activity[var] += var_inc
        if activity[var] > 1e100:
            # Rescale
            for v in range(1, n + 1):
                activity[v] /= 1e100
            var_inc /= 1e100

    def search():
        num_conflicts = 0
        while True:
            conflict = propagate()
            
            if conflict != -1:
                # Conflict analysis
                learnt_clause, backjump_level = analyze(conflict)
                
                # Backjump
                # Remove trail up to backjump_level
                # trail_lim[backjump_level] is the index where level backjump_level starts
                # We need to undo assignments with level > backjump_level
                
                # Find the index in trail where we stop
                # trail_lim[i] is the index of the first variable assigned at level i
                # So we want to keep trail[0 : trail_lim[backjump_level + 1]] ?
                # Actually, trail_lim[k] is the index of the start of level k.
                # If we backjump to level L, we keep assignments up to the end of level L-1?
                # No, we keep assignments up to level L. The variable at level L is the one we might re-assign or keep?
                # Standard CDCL: backjump to level L. We undo all variables with level > L.
                # The variable that caused the conflict at level L+1 is undone.
                
                # Let's implement undo carefully
                # We need to find the last variable assigned at level <= backjump_level
                # Actually, we just pop from trail until level[trail[-1]] <= backjump_level
                # But we must be careful with trail_lim.
                
                # Simpler: 
                # 1. Pop trail until we reach a variable with level <= backjump_level
                # 2. Adjust trail_lim to match the new trail length
                
                while trail and level[trail[-1]] > backjump_level:
                    v = trail.pop()
                    value[v] = None
                    level[v] = 0
                    reason[v] = -1
                    # Note: We don't un-watch clauses here, they are re-watched if needed?
                    # Actually, watchlists are updated during propagation.
                    # When we undo, the literal becomes unassigned again.
                    # The watchlist entries are still there, but the clause might be watched on a literal that is now unassigned.
                    # This is fine, propagation will handle it.
                    
                # Update trail_lim
                # trail_lim should have an entry for each level.
                # If we popped back to level L, we need to remove trail_lim entries for levels > L
                # and ensure trail_lim[L] is correct?
                # Actually, trail_lim is a list where trail_lim[i] is the index in trail where level i starts.
                # If we pop, the indices change.
                # It is easier to just rebuild trail_lim or maintain it carefully.
                # For simplicity and correctness in this small scope, let's rebuild trail_lim?
                # Or better: maintain trail_lim such that trail_lim[k] is the index of the first var at level k.
                # When we pop, we just remove the corresponding trail_lim entries.
                
                # Let's check the current structure.
                # trail_lim is a list. len(trail_lim) == current_level + 1.
                # If we backjump to L, we remove trail_lim[L+1:]
                
                # But wait, if we popped variables, the indices in trail_lim for lower levels might be invalid if they were > current len(trail).
                # However, we only popped vars with level > L.
                # So the indices for levels <= L are still valid relative to the new trail?
                # No, trail_lim[k] is an index into the OLD trail.
                # If we pop, the indices shift.
                
                # Let's use a different structure: 
                # Instead of trail_lim being indices, let's just store the levels in a separate stack or recompute.
                # Given the complexity, let's simplify the solver to a DPLL with clause learning but simpler backjumping?
                # Or, let's just use a standard WalkSAT which is much easier to implement correctly in Python for n=50.
                
                # RE-EVALUATION:
                # Writing a bug-free CDCL solver in Python under time pressure is risky.
                # The seed is WalkSAT.
                # A better "new approach" that is still local search but different from basic WalkSAT is 
                # **Gallois** or **WalkSAT with Clause Weighting**.
                # Or **Minisat** is too complex to write from scratch safely.
                
                # Let's switch to **WalkSAT with Clause Weighting**.
                # This is a distinct improvement over the seed (which has fixed weights/no weights).
                # It is robust and fast.
                
                pass # Abort CDCL attempt, switch to Weighted WalkSAT

# Since the CDCL attempt is getting complex and risky, I will provide a **Weighted WalkSAT** implementation.
# This is a different "approach family" in the sense of heuristic tuning and data structure usage (clause weights) compared to the seed's static WalkSAT.
# Actually, the directive says "from scratch... do not reuse current designs".
# Weighted WalkSAT is still WalkSAT.
# Is there another family? **Simulated Annealing**? **Genetic Algorithm**?
# For SAT, **WalkSAT** and **Chaff/CDCL** are the main ones.
# **Gallois** is a local search.
# Let's try a **Simple DPLL** with **Unit Propagation** and **Pure Literal Elimination**.
# For n=50, a well-optimized DPLL can solve planted 3-SAT instances very quickly.
# It is a complete search, unlike local search. This is a distinct approach.

# Let's implement a basic DPLL.

# RESTARTING THE CODE GENERATION WITH DPLL

import sys
sys.setrecursionlimit(10000)

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # DPLL Solver
    # Clauses are lists of signed literals.
    # We will modify a copy of the clauses to simplify them.
    
    def simplify(clauses, assignment):
        # assignment is a dict var -> bool
        # Returns (new_clauses, conflict)
        # conflict is True if empty clause found
        # new_clauses is the list of simplified clauses
        new_clauses = []
        for clause in clauses:
            new_clause = []
            clause_satisfied = False
            for lit in clause:
                var = abs(lit)
                sign = lit > 0
                if var in assignment:
                    val = assignment[var]
                    if val == sign:
                        clause_satisfied = True
                        break
                    else:
                        # Literal is false, skip it
                        continue
                else:
                    new_clause.append(lit)
            if clause_satisfied:
                continue
            if not new_clause:
                return None, True # Conflict
            new_clauses.append(new_clause)
        return new_clauses, False

    def pure_literal_elimination(clauses, assignment):
        # Find variables that appear only with one sign
        # If found, assign it and simplify
        # This is a loop because assignments change the clause set
        changed = True
        while changed:
            changed = False
            # Count occurrences
            pos_count = [0] * (n + 1)
            neg_count = [0] * (n + 1)
            for clause in clauses:
                for lit in clause:
                    var = abs(lit)
                    if lit > 0:
                        pos_count[var] += 1
                    else:
                        neg_count[var] += 1
                        
            # Find pure literals
            for var in range(1, n + 1):
                if var in assignment:
                    continue
                if pos_count[var] == 0:
                    # Assign var = False
                    assignment[var] = False
                    changed = True
                elif neg_count[var] == 0:
                    # Assign var = True
                    assignment[var] = True
                    changed = True
                    
            if changed:
                clauses, conflict = simplify(clauses, assignment)
                if conflict:
                    return False
        return True

    def unit_propagation(clauses, assignment):
        # Find unit clauses and assign
        changed = True
        while changed:
            changed = False
            for i, clause in enumerate(clauses):
                if len(clause) == 1:
                    lit = clause[0]
                    var = abs(lit)
                    sign = lit > 0
                    if var not in assignment:
                        assignment[var] = sign
                        changed = True
                        # Simplify and restart loop
                        clauses, conflict = simplify(clauses, assignment)
                        if conflict:
                            return False
                        break
        return True

    def dpll(clauses, assignment, depth=0):
        # Base case: all clauses satisfied (empty list)
        if not clauses:
            return assignment
            
        # Check for conflict (should be handled by simplify, but just in case)
        # Simplify again to ensure consistency
        clauses, conflict = simplify(clauses, assignment)
        if conflict:
            return None
            
        if not clauses:
            return assignment

        # Pure Literal Elimination
        if not pure_literal_elimination(clauses, assignment):
            return None
            
        # Unit Propagation
        if not unit_propagation(clauses, assignment):
            return None
            
        # Simplify again after UP
        clauses, conflict = simplify(clauses, assignment)
        if conflict:
            return None
        if not clauses:
            return assignment

        # Choose a variable
        # Heuristic: pick variable from the shortest clause
        min_len = min(len(c) for c in clauses)
        if min_len == 0:
            return None
            
        # Pick a clause with min length
        clause = None
        for c in clauses:
            if len(c) == min_len:
                clause = c
                break
                
        var = abs(clause[0])
        
        # Branch 1: var = True
        new_assignment = assignment.copy()
        new_assignment[var] = True
        result = dpll(clauses, new_assignment, depth + 1)
        if result is not None:
            return result
            
        # Branch 2: var = False
        new_assignment = assignment.copy()
        new_assignment[var] = False
        result = dpll(clauses, new_assignment, depth + 1)
        if result is not None:
            return result
            
        return None

    # Initial assignment is empty
    assignment = {}
    
    # Run DPLL
    result = dpll(clauses, assignment)
    
    if result is None:
        # Instance is unsatisfiable (should not happen for planted SAT)
        # Return a random assignment or all False
        return [False] * n
        
    # Convert dict to list of booleans
    # result is a dict var -> bool
    final_assignment = [False] * n
    for var in range(1, n + 1):
        if var in result:
            final_assignment[var - 1] = result[var]
        else:
            # Unassigned variables can be set arbitrarily
            final_assignment[var - 1] = False
            
    return final_assignment

# EVOLVE-BLOCK-END
