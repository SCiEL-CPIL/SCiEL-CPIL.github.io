"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    deadline = time.time() + 290.0  # safety margin before 330s kill

    # Preprocess: build watch lists and detect trivial unit clauses
    # Represent each literal as signed 1-based integer.
    # assignment: list of booleans, index 0..n-1 corresponds to var 1..n
    # lit_value(lit, assignment): True if lit is satisfied
    #   lit > 0: assignment[lit-1]
    #   lit < 0: not assignment[-lit-1]

    # Build watch lists: for each literal, list of (clause_index, other_literal)
    # where the clause contains that literal and the other literal is its watch partner.
    # Standard 2-watching: each clause watches two distinct literals.
    # For 3-SAT, each clause has 3 literals; we watch 2 of them.

    num_clauses = len(clauses)
    # watch[lit] = list of clause indices that watch lit
    # For each clause, we store which two literals it watches
    # watch_lits[c] = [lit_a, lit_b] the two watched literals in clause c
    # other_lit[c] for each watched position: the third literal (unwatched)

    # Initialize: for each clause, watch the first two literals
    watch = [[] for _ in range(2 * n + 1)]  # index by abs(lit), but we need sign
    # Actually, let's use a dict or array indexed by literal value (1..n, -1..-n)
    # Simpler: use a dictionary mapping literal -> list of (clause_idx, position)
    # But for speed, use two arrays: watch_pos[2][n+1] where pos 0,1 are the two watch slots
    # Let's use: watch[2][n+1] where watch[pos][var] = list of (clause_idx, other_var, other_sign)
    # This is getting complex. Let me use a simpler but effective approach.

    # Alternative: Use a well-tuned WalkSAT with restarts and clause learning lite.
    # But the directive says to try DPLL with VSIDS. Let me implement a clean DPLL.

    # DPLL with VSIDS and unit propagation
    # For planted 3-SAT with 150 vars, DPLL should be very fast.

    # Let's implement a proper DPLL solver.

    # Data structures:
    # - assignment: array of -1, 0, 1 (false, unassigned, true)
    # - trail: list of (var, value) for undo
    # - watch lists for unit propagation
    # - clause list: list of lists of literals
    # - activity scores for VSIDS

    # Build initial watch lists
    # For each clause, pick two literals to watch.
    # watch[lit] = list of clause indices
    # For each clause, store the two watched literals and the unwatched one.

    # Let me use a different, cleaner implementation:

    # Convert clauses to 0-based variables for internal use, but keep signed literals.
    # Actually, let's keep 1-based as given.

    # assignment[var] where var is 1..n, value is -1, 0, 1
    assignment = [0] * (n + 1)  # index 0 unused
    trail = []  # list of vars in order of assignment

    # Watch lists: for each literal (signed), list of clause indices
    # We'll use a dict: watch[lit] = set of clause indices
    # But for efficiency, use list.
    
    # For each clause, we track which literals are watched.
    # Let's use: for clause c, watched_lits[c] = [lit1, lit2] (the two watched)
    # and we can find the third by checking which of the 3 is not in watched.
    
    # Actually, let's just implement a straightforward DPLL without watching,
    # since n=150 is small. Unit propagation by scanning all clauses each time
    # might be slow, but with good variable ordering it should be fine.
    
    # Better: use watching. Let me implement it properly.
    
    # watch: dict from literal to list of clause indices
    watch = {}
    # For each clause, the two watched literals
    clause_watched = []  # clause_watched[c] = (lit_a, lit_b)
    
    for c, clause in enumerate(clauses):
        # Pick two literals to watch (first two)
        lit_a, lit_b = clause[0], clause[1]
        clause_watched.append((lit_a, lit_b))
        if lit_a not in watch:
            watch[lit_a] = []
        watch[lit_a].append(c)
        if lit_b not in watch:
            watch[lit_b] = []
        watch[lit_b].append(c)

    # Activities for VSIDS
    activity = [0.0] * (n + 1)
    # Variable selection: pick unassigned var with highest activity
    # We'll use a simple approach: maintain a heap or just scan.
    # For n=150, scanning is fine.

    import heapq

    # Let me implement a cleaner DPLL:

    def is_satisfied(lit, assign):
        v = abs(lit)
        val = assign[v]
        if val == 0:
            return None
        return (val == 1) if lit > 0 else (val == -1)

    def is_false(lit, assign):
        v = abs(lit)
        val = assign[v]
        if val == 0:
            return False
        return (val == -1) if lit > 0 else (val == 1)

    def propagate():
        """Unit propagation using watch lists. Returns False on conflict."""
        # We need a queue of unit clauses (clauses with exactly one unassigned literal)
        # When we assign a variable, some clauses become unit.
        
        # Let's use a worklist of clauses to check.
        # Actually, with watching, when we assign var v to value val:
        #   For each clause watching a literal that becomes false:
        #     Check if the other watched literal is satisfied -> if so, done
        #     Else if there's an unassigned literal in the clause, rewatch it
        #     Else if all literals are false -> conflict
        #     Else the clause is unit -> add to unit queue
        
        # Let me implement step-by-step.
        
        # We'll process assignments one at a time and update watches.
        # This is getting complex. Let me use a simpler approach for n=150:
        # Just scan all clauses for unit propagation. It's O(num_clauses) per assignment,
        # and with ~150 assignments max, that's 150 * num_clauses which is fine.
        
        pass  # We'll do simple propagation

    # Actually, let me just implement a simple but effective DPLL:

    def unit_propagate():
        """Scan all clauses, find unit clauses, propagate. Returns (conflict, unit_var, unit_val) or (False, None, None)."""
        changed = True
        while changed:
            changed = False
            for clause in clauses:
                # Count unassigned and check for satisfaction/conflict
                unassigned_lits = []
                satisfied = False
                for lit in clause:
                    v = abs(lit)
                    val = assignment[v]
                    if val == 0:
                        unassigned_lits.append(lit)
                    else:
                        if (val == 1) == (lit > 0):
                            satisfied = True
                            break
                if satisfied:
                    continue
                if not unassigned_lits:
                    return (True, None, None)  # conflict
                if len(unassigned_lits) == 1:
                    # Unit clause
                    lit = unassigned_lits[0]
                    v = abs(lit)
                    val = 1 if lit > 0 else -1
                    if assignment[v] == 0:
                        assignment[v] = val
                        trail.append(v)
                        changed = True
        return (False, None, None)

    def pick_variable():
        """Pick an unassigned variable using VSIDS-like heuristic."""
        best_var = -1
        best_score = -1
        for v in range(1, n + 1):
            if assignment[v] == 0:
                # Score based on activity and occurrence in unsatisfied clauses
                score = activity[v]
                # Boost by number of unassigned occurrences in currently unsatisfied clauses
                # For simplicity, just use activity
                if score > best_score:
                    best_score = score
                    best_var = v
        return best_var

    def bump_activity():
        """Bump activity of unassigned variables in unsatisfied clauses."""
        for clause in clauses:
            # Check if clause is currently unsatisfied (no assigned true literal)
            has_assigned_true = False
            for lit in clause:
                v = abs(lit)
                val = assignment[v]
                if val != 0 and (val == 1) == (lit > 0):
                    has_assigned_true = True
                    break
            if not has_assigned_true:
                for lit in clause:
                    v = abs(lit)
                    if assignment[v] == 0:
                        activity[v] += 1.0

    def backtrack(level):
        """Unassign variables down to the given trail level."""
        while len(trail) > level:
            v = trail.pop()
            assignment[v] = 0

    def dpll():
        """Recursive DPLL with restarts handled externally."""
        # Check for conflict
        conflict, _, _ = unit_propagate()
        if conflict:
            return False
        
        # Check if all assigned
        for v in range(1, n + 1):
            if assignment[v] == 0:
                break
        else:
            return True  # All assigned and no conflict
        
        # Pick variable
        var = pick_variable()
        if var == -1:
            return True
        
        # Try true
        trail_level = len(trail)
        assignment[var] = 1
        trail.append(var)
        if dpll():
            return True
        backtrack(trail_level)
        
        # Try false
        assignment[var] = -1
        trail.append(var)
        if dpll():
            return True
        backtrack(trail_level)
        
        return False

    # Simple DPLL might be slow without good heuristics. Let me add restarts and
    # a more careful activity scheme. But for planted 3-SAT with 150 vars,
    # even a simple DPLL should work. Let's try it.

    # Actually, the recursive approach with full unit propagation scanning all clauses
    # each time is O(num_clauses * depth) which could be slow. Let me limit the search.

    # Let me try a hybrid: use DPLL for a limited number of decisions, then fall back to WalkSAT.

    # First, try DPLL with a decision limit.
    decision_limit = 1000
    decisions_made = [0]

    def dpll_limited():
        if time.time() > deadline:
            return False
        if decisions_made[0] >= decision_limit:
            return None  # timeout/limit

        conflict, _, _ = unit_propagate()
        if conflict:
            return False

        # Check if all assigned
        all_assigned = True
        for v in range(1, n + 1):
            if assignment[v] == 0:
                all_assigned = False
                break
        if all_assigned:
            return True

        var = pick_variable()
        if var == -1:
            return True

        decisions_made[0] += 1
        bump_activity()

        # Try true
        trail_level = len(trail)
        assignment[var] = 1
        trail.append(var)
        result = dpll_limited()
        if result is True:
            return True
        if result is None:
            backtrack(trail_level)
            return None
        backtrack(trail_level)

        # Try false
        assignment[var] = -1
        trail.append(var)
        result = dpll_limited()
        if result is True:
            return True
        if result is None:
            backtrack(trail_level)
            return None
        backtrack(trail_level)

        return False

    result = dpll_limited()
    if result is True:
        # Extract assignment
        return [bool(assignment[v]) for v in range(1, n + 1)]

    # If DPLL didn't solve it, fall back to WalkSAT with the current partial assignment
    # or a fresh random assignment.
    
    # Let's do a fresh WalkSAT with restarts.
    rng = random.Random(0)
    
    best_assignment = None
    best_unsat_count = float('inf')
    
    max_restarts = 50
    max_flips_per_restart = 5000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Initialize assignment
        if restart == 0 and len(trail) > 0:
            # Use current partial assignment, fill rest randomly
            assignment = [0] * (n + 1)
            for v in trail:
                assignment[v] = 1 if rng.random() < 0.5 else -1
            # Wait, we need to be careful. Let's just start fresh.
            pass
        
        assignment = [0] * (n + 1)
        for v in range(1, n + 1):
            assignment[v] = 1 if rng.random() < 0.5 else -1
        
        def count_unsat():
            count = 0
            for clause in clauses:
                sat = False
                for lit in clause:
                    v = abs(lit)
                    val = assignment[v]
                    if (val == 1) == (lit > 0):
                        sat = True
                        break
                if not sat:
                    count += 1
            return count
        
        current_unsat = count_unsat()
        if current_unsat == 0:
            return [bool(assignment[v]) for v in range(1, n + 1)]
        
        if current_unsat < best_unsat_count:
            best_unsat_count = current_unsat
            best_assignment = assignment[:]
        
        for flip in range(max_flips_per_restart):
            if time.time() > deadline:
                break
            
            # Find unsatisfied clauses
            unsat_clauses = []
            for clause in clauses:
                sat = False
                for lit in clause:
                    v = abs(lit)
                    val = assignment[v]
                    if (val == 1) == (lit > 0):
                        sat = True
                        break
                if not sat:
                    unsat_clauses.append(clause)
            
            if not unsat_clauses:
                return [bool(assignment[v]) for v in range(1, n + 1)]
            
            # Pick a random unsatisfied clause
            clause = rng.choice(unsat_clauses)
            
            # Pick a literal to flip with probability based on how many literals in clause are false
            # Standard WalkSAT: flip with prob p if it satisfies the clause, else 1-p
            p = 0.5
            num_false_in_clause = 0
            for lit in clause:
                v = abs(lit)
                val = assignment[v]
                if (val == -1) == (lit > 0):
                    num_false_in_clause += 1
            
            # Pick a literal
            if rng.random() < p:
                # Pick a false literal from the clause
                false_lits = [lit for lit in clause if (assignment[abs(lit)] == -1) == (lit > 0)]
                if false_lits:
                    lit = rng.choice(false_lits)
                else:
                    lit = rng.choice(clause)
            else:
                lit = rng.choice(clause)
            
            v = abs(lit)
            assignment[v] = -assignment[v]
            
            current_unsat = count_unsat()
            if current_unsat == 0:
                return [bool(assignment[v]) for v in range(1, n + 1)]
            
            if current_unsat < best_unsat_count:
                best_unsat_count = current_unsat
                best_assignment = assignment[:]

    # Return best assignment found
    if best_assignment is not None:
        return [bool(best_assignment[v]) for v in range(1, n + 1)]
    
    # Fallback: all false
    return [False] * n
# EVOLVE-BLOCK-END
