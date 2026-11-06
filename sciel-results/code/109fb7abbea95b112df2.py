"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

# Increase recursion limit just in case, though CDCL is iterative
sys.setrecursionlimit(10000)

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Deadline guard
    deadline = time.time() + 290
    
    # Internal representation:
    # Variables are 0..n-1
    # Literals are integers: var (positive) or ~var (negative)
    # We use 0-based indexing internally.
    # Clause is a list of literals.
    
    # Convert input clauses to internal format
    # Input: list of lists of signed 1-based ints
    # Output: list of lists of signed 0-based ints
    internal_clauses = []
    for clause in clauses:
        int_clause = []
        for lit in clause:
            var = abs(lit) - 1
            sign = 1 if lit > 0 else -1
            int_clause.append(var * sign)
        internal_clauses.append(int_clause)
    
    # CDCL Solver Implementation
    
    # Data structures
    # assignment: list of 0, 1, or -1 (unassigned, true, false)
    # trail: list of literals assigned in order
    # trail_lim: list of trail indices for decision levels
    # levels: list of decision levels for each variable
    # reason: list of clause indices that caused the assignment, or None for decisions
    # learnts: list of learnt clauses
    # watched: dict mapping literal to list of clause indices watching it
    
    assignment = [0] * n
    trail = []
    trail_lim = []
    levels = [0] * n
    reasons = [None] * n
    learnts = []
    watched = {}
    
    # Initialize watched literals
    # For each clause, pick two literals to watch
    # If a clause has < 2 literals, handle separately (unit or empty)
    
    def add_clause(clause):
        """Add a clause to the solver, returning False if it's empty or unit (handled by propagation)"""
        # Remove duplicates and tautologies
        if len(clause) == 0:
            return False
        if len(clause) == 1:
            # Unit clause, handled by propagation loop
            return True 
        
        # Simple deduplication
        seen = set()
        new_clause = []
        for lit in clause:
            if lit not in seen:
                new_clause.append(lit)
                seen.add(lit)
            else:
                # Tautology check: if -lit is also in clause
                # Actually, if we see a duplicate, it's not a tautology yet.
                # Tautology is if lit and -lit are both present.
                pass
        
        # Check for tautology
        lit_set = set(new_clause)
        for lit in new_clause:
            if -lit in lit_set:
                return True # Tautology, always satisfied
        
        new_clause = list(lit_set)
        
        if len(new_clause) == 0:
            return False
        if len(new_clause) == 1:
            return True
            
        # Pick two literals to watch
        # Prefer unassigned literals for watching
        unassigned = [lit for lit in new_clause if assignment[abs(lit)] == 0]
        if len(unassigned) >= 2:
            w1, w2 = unassigned[0], unassigned[1]
        elif len(unassigned) == 1:
            w1 = unassigned[0]
            w2 = new_clause[0]
        else:
            w1, w2 = new_clause[0], new_clause[1]
        
        clause_idx = len(internal_clauses) + len(learnts)
        # We need a unique ID for the clause.
        # Let's use a separate list for all clauses to manage indices easily.
        # For simplicity in this compact implementation, we'll append to a master list if we want to reference them by index.
        # However, standard CDCL often just keeps the clause in the learnts list and references it.
        # Let's use a global clause list for all clauses (original + learnt).
        
        # Actually, to keep it simple and fast, let's just store the clause in a list `all_clauses` 
        # and `watched` maps literal -> list of clause indices in `all_clauses`.
        
        # Refactoring: Use a single list for all clauses.
        # `internal_clauses` are the original ones. `learnts` will be appended to a shared list?
        # No, let's just use `all_clauses` and `watched`.
        
        return new_clause

    # Let's restart the data structure setup to be cleaner.
    # We will maintain `all_clauses` which contains original and learnt clauses.
    # `watched` maps literal -> list of indices into `all_clauses`.
    
    all_clauses = []
    watched = {}
    
    def add_clause(clause):
        """
        Adds a clause. Returns False if the clause is empty (conflict).
        Returns True if the clause is satisfied or unit (and enqueues unit).
        """
        # Check for tautology and duplicates
        if len(clause) == 0:
            return False
        
        # Remove tautologies
        lit_set = set(clause)
        for lit in clause:
            if -lit in lit_set:
                return True # Satisfied by tautology
        
        # Remove duplicates
        new_clause = list(lit_set)
        
        if len(new_clause) == 0:
            return False
            
        if len(new_clause) == 1:
            # Unit clause
            lit = new_clause[0]
            var = abs(lit)
            if assignment[var] == 0:
                enqueue(lit)
            elif assignment[var] == 1 and lit < 0:
                return False # Conflict
            elif assignment[var] == -1 and lit > 0:
                return False # Conflict
            return True
        
        # Add to all_clauses
        clause_idx = len(all_clauses)
        all_clauses.append(new_clause)
        
        # Pick two literals to watch
        # Prefer unassigned
        unassigned = [lit for lit in new_clause if assignment[abs(lit)] == 0]
        if len(unassigned) >= 2:
            w1, w2 = unassigned[0], unassigned[1]
        elif len(unassigned) == 1:
            w1 = unassigned[0]
            w2 = new_clause[0]
        else:
            w1, w2 = new_clause[0], new_clause[1]
            
        if w1 not in watched:
            watched[w1] = []
        watched[w1].append(clause_idx)
        
        if w2 not in watched:
            watched[w2] = []
        watched[w2].append(clause_idx)
        
        return True

    def enqueue(lit):
        var = abs(lit)
        val = 1 if lit > 0 else -1
        assignment[var] = val
        trail.append(lit)
        levels[var] = len(trail_lim)
        # reasons[var] = None for now, set during propagation

    def propagate():
        """
        Perform unit propagation.
        Returns a conflict clause (list of literals) if a conflict is found, else None.
        """
        while len(trail) > (len(trail_lim) if trail_lim else 0):
            p = trail[-1]
            var = abs(p)
            
            # Watched literals for p
            # We need to find clauses where p is a watched literal and p is now False (i.e., we assigned -p)
            # Wait, if p is on the trail, it means p is True.
            # So we look for clauses watching -p (the negation of the assigned literal).
            # If a clause watches -p, and -p becomes False (because p is True), we need to update the watch.
            
            neg_p = -p
            if neg_p not in watched:
                continue
                
            i = 0
            to_remove = []
            
            for c_idx in watched[neg_p]:
                clause = all_clauses[c_idx]
                
                # Check if clause is satisfied
                # If any literal in clause is True, it's satisfied.
                # Since we are propagating, we assume other literals might be false or unassigned.
                # We specifically check if the clause is satisfied by any literal other than the watched ones?
                # Standard algorithm:
                # 1. If clause is satisfied, remove watch? No, keep it but don't worry.
                # Actually, if satisfied, we can just skip.
                
                # Check satisfaction
                satisfied = False
                for lit in clause:
                    v = assignment[abs(lit)]
                    if v != 0:
                        if (lit > 0 and v == 1) or (lit < 0 and v == -1):
                            satisfied = True
                            break
                
                if satisfied:
                    continue
                    
                # Not satisfied.
                # One of the watched literals is neg_p, which is now False (since p is True).
                # We need to find a new literal to watch.
                
                # Find a new literal in the clause that is not neg_p and not the other watched literal?
                # Actually, we just need any unassigned literal to watch.
                
                new_watch = None
                unit_lit = None
                conflict = False
                
                for lit in clause:
                    if lit == neg_p:
                        continue
                    v = assignment[abs(lit)]
                    if v == 0:
                        new_watch = lit
                        break
                    else:
                        # Check if this literal makes the clause satisfied?
                        # We already checked satisfaction above, so if v != 0, it must be False (unsatisfied).
                        # Wait, if v != 0 and not satisfied, then lit is False.
                        # If we find a lit that is False, it's not a new watch.
                        # If we find a lit that is True, it would have been satisfied.
                        # So all non-neg_p literals are either Unassigned or False.
                        pass
                
                if new_watch is None:
                    # All other literals are False.
                    # This means the clause is a unit clause (only neg_p was unassigned, now False)
                    # Or if neg_p was the only unassigned, and it's now False, the clause is False.
                    # Wait, if neg_p is False, and all others are False, the clause is UNSAT.
                    # This is a conflict.
                    conflict = True
                    break
                else:
                    # We found a new watch.
                    # Remove neg_p from watched list?
                    # We will replace neg_p with new_watch in the watched list for this clause.
                    # Actually, we just need to add new_watch to watched[new_watch].
                    # And we will skip neg_p in the next iteration of the outer loop?
                    # No, we are iterating watched[neg_p].
                    # We need to update the clause's watched literals.
                    # The clause currently watches neg_p and some other lit (w2).
                    # We change the watch from neg_p to new_watch.
                    
                    if new_watch not in watched:
                        watched[new_watch] = []
                    watched[new_watch].append(c_idx)
                    
                    # Mark to remove neg_p from this clause's watch?
                    # We can just leave it in watched[neg_p] but it will be processed again?
                    # Better to remove it from the current list to avoid re-processing.
                    # Since we are iterating a list, we can't easily remove while iterating.
                    # We can just add to a "to_remove" list.
                    to_remove.append(c_idx)
                    
                    # If the other watched literal was also False?
                    # We checked satisfaction. If not satisfied, the other watch must be Unassigned or False.
                    # If the other watch was False, we would have found a conflict earlier?
                    # No, we only checked neg_p.
                    # Let's verify:
                    # Clause: {neg_p, w2, ...}
                    # neg_p is False.
                    # If w2 is False, and all others are False, then conflict.
                    # My loop `for lit in clause` checks ALL literals.
                    # If it finds `new_watch` (Unassigned), it stops.
                    # If it doesn't find any Unassigned, `new_watch` remains None -> Conflict.
                    # So if `new_watch` is found, it is Unassigned.
                    # The clause is now watched by `new_watch` and `w2` (or whatever the other watch was).
                    # We don't need to do anything else.
            
            # Remove clauses that had their watch changed
            if to_remove:
                # This is O(N) removal. For n=50, this is fine.
                new_watched_list = []
                for c_idx in watched[neg_p]:
                    if c_idx not in to_remove:
                        new_watched_list.append(c_idx)
                watched[neg_p] = new_watched_list
            
            if conflict:
                return all_clauses[c_idx]
                
        return None

    # Initialize with original clauses
    for clause in internal_clauses:
        if not add_clause(clause):
            # Unsat
            return [False] * n
            
    # Propagate initial units
    conflict_clause = propagate()
    if conflict_clause:
        return [False] * n
        
    # Main Search Loop
    # VSIDS Heuristic
    activity = [1.0] * n
    # We need a heap for variable selection, but for n=50, sorting is fine.
    
    def pick_var():
        # Pick unassigned variable with highest activity
        best_var = -1
        best_act = -1
        for v in range(n):
            if assignment[v] == 0:
                if activity[v] > best_act:
                    best_act = activity[v]
                    best_var = v
        return best_var

    def analyze(conflict_clause):
        """
        First UIP clause learning.
        Returns (learnt_clause, backtrack_level)
        """
        learnt = []
        seen = set()
        p = None
        index = len(trail) - 1
        cnt = 0
        
        # conflict_clause is the clause that became False
        # It contains at least one literal that was just assigned (the one that triggered propagation)
        # Actually, the conflict clause is the one that is now False.
        # All literals in it are False.
        
        # We need to find the First UIP.
        
        # Initialize seen with the literal that caused the conflict?
        # The conflict clause is all False.
        # We trace back.
        
        # Standard Algorithm:
        # 1. Start with the conflict clause.
        # 2. While True:
        #    - Pick a literal from the current clause that is not in seen.
        #    - If it's a decision literal, break (this is the UIP).
        #    - Add it to seen.
        #    - Get the reason clause for this literal.
        #    - Add all literals from the reason clause to the current clause (if not in seen).
        #    - ...
        
        # Let's implement carefully.
        
        # The conflict clause is `conflict_clause`.
        # All literals in it are currently False.
        
        # We will build the learnt clause.
        
        # Step 1: Identify the literal in conflict_clause that was assigned last?
        # Actually, we just iterate through the conflict clause.
        
        # Let's use a queue or stack of literals to process.
        
        # The "current" clause is the conflict clause.
        # We want to find a literal in it that is a decision.
        
        # Algorithm:
        # learnt = []
        # seen = set()
        # 
        # # Add all literals from conflict clause to a processing list?
        # # No, we pick one literal to "explain".
        
        # Let's follow the standard First-UIP algorithm:
        
        # 1. Let C be the conflict clause.
        # 2. Let S be an empty set.
        # 3. While True:
        #    a. Let l be a literal in C \ S.
        #    b. If l is a decision literal, add l to learnt, break.
        #    c. Add l to S.
        #    d. Let R be the reason clause for l.
        #    e. For each literal m in R:
        #       - If m is not in S and m is not l:
        #         - Add m to C.
        #    f. C = C \ {l} (remove l from C, as it's been explained)
        
        # Wait, the learnt clause is the set of literals that led to the conflict.
        
        # Let's trace:
        # conflict_clause = [l1, l2, ...] all False.
        
        # We need to find the First UIP.
        
        # Implementation:
        
        # Get the decision level of the current conflict.
        # All literals in conflict_clause are False.
        # The highest level among them is the current level (since we just propagated).
        
        # We need to backtrack to the level before the current one.
        
        # Let's pick the literal in conflict_clause with the highest level.
        # If there are multiple, pick any.
        
        # Actually, the standard algorithm works on the "reason" graph.
        
        # Let's simplify:
        
        # 1. `learnt` will contain the literals that are "remote" (assigned at lower levels) and the UIP.
        
        # Let's use a list `to_analyze` initialized with literals from `conflict_clause`.
        
        # But we need to be careful which one to pick.
        
        # Let's pick the literal in `conflict_clause` that was assigned most recently?
        # No, just pick any.
        
        # Let's iterate.
        
        # `seen` keeps track of literals we have "used" to explain.
        
        # `current_clause` starts as `conflict_clause`.
        
        # We will maintain a list of literals in `current_clause`.
        
        # 1. Pick a literal `l` from `current_clause` that is not in `seen`.
        #    Prefer literals with highest level?
        
        # 2. If `l` is a decision:
        #    - Add `l` to `learnt`.
        #    - Break.
        
        # 3. Else:
        #    - Add `l` to `seen`.
        #    - Get `reason` clause for `l`.
        #    - Add all literals from `reason` to `current_clause` (if not in `seen` and not `l`).
        #    - Remove `l` from `current_clause`.
        
        # After loop:
        # `learnt` contains the UIP and all "remote" literals?
        # No, the learnt clause is the set of literals in `current_clause` that are NOT in `seen`?
        # Actually, the learnt clause is the set of literals that were added to `learnt` (the UIP) 
        # PLUS the literals that were in `current_clause` but not in `seen`?
        
        # Correct First-UIP:
        # The learnt clause consists of:
        # 1. The UIP literal.
        # 2. All literals from the reason clauses that were NOT in `seen` and NOT the UIP.
        
        # Let's implement this.
        
        seen = set()
        learnt = []
        
        # Start with the conflict clause
        # We need to pick a literal to start explaining.
        # Let's pick the one with the highest level in the conflict clause.
        
        # Get levels of literals in conflict_clause
        max_level = -1
        start_lit = None
        for lit in conflict_clause:
            lvl = levels[abs(lit)]
            if lvl > max_level:
                max_level = lvl
                start_lit = lit
                
        if start_lit is None:
            return [], 0
            
        # We will use a stack of literals to process?
        # No, the algorithm is iterative.
        
        # Let's maintain `current_clause` as a set of literals.
        current_clause = set(conflict_clause)
        
        # We need to pick a literal from `current_clause` to explain.
        # We should pick the one with the highest level to ensure we move towards the decision.
        
        while True:
            # Pick literal with highest level in current_clause
            best_lit = None
            best_lvl = -1
            for lit in current_clause:
                if lit in seen:
                    continue
                lvl = levels[abs(lit)]
                if lvl > best_lvl:
                    best_lvl = lvl
                    best_lit = lit
                    
            if best_lit is None:
                # Should not happen
                break
                
            # Check if decision
            if reasons[abs(best_lit)] is None:
                # It's a decision
                learnt.append(best_lit)
                break
            else:
                # It's a propagated literal
                seen.add(best_lit)
                reason_clause = all_clauses[reasons[abs(best_lit)]]
                
                # Add literals from reason clause to current_clause
                for lit in reason_clause:
                    if lit == best_lit:
                        continue
                    if lit in seen:
                        continue
                    current_clause.add(lit)
                    
                # Remove best_lit from current_clause
                current_clause.remove(best_lit)
                
        # The learnt clause is `learnt` (UIP) + all literals in `current_clause` that are not in `seen`?
        # Wait, the literals in `current_clause` that are NOT in `seen` are the ones we haven't explained yet.
        # But we only break when we find a decision.
        # The other literals in `current_clause` are the ones that were added from reasons but not yet "processed" (i.e., not in `seen`).
        # These are the "remote" literals.
        
        for lit in current_clause:
            if lit not in seen:
                learnt.append(lit)
                
        # Simplify learnt clause:
        # Remove literals that are already False? No, they are all False by definition of conflict.
        # Remove duplicates.
        
        # Determine backtrack level
        # Find the second highest level in `learnt`.
        
        levels_in_learnt = [levels[abs(lit)] for lit in learnt]
        
        if len(levels_in_learnt) == 0:
            bt_level = 0
        elif len(levels_in_learnt) == 1:
            bt_level = 0
        else:
            # Sort levels
            sorted_levels = sorted(levels_in_learnt, reverse=True)
            # The highest is the current level.
            # The second highest is the backtrack level.
            bt_level = sorted_levels[1]
            
        return learnt, bt_level

    # Main loop
    num_decisions = 0
    conflict_count = 0
    restart_interval = 100
    restarts = 0
    activity_decay = 0.95
    
    while True:
        if time.time() > deadline:
            # Return current assignment (might be partial)
            # Fill unassigned with random
            for i in range(n):
                if assignment[i] == 0:
                    assignment[i] = 1 if random.random() < 0.5 else -1
            return [a == 1 for a in assignment]
            
        # Check for conflict
        conflict_clause = propagate()
        
        if conflict_clause:
            # Analyze
            learnt, bt_level = analyze(conflict_clause)
            
            # Add learnt clause
            # If learnt is empty, Unsat
            if not learnt:
                return [False] * n
                
            # Simplify learnt clause
            # Remove literals that are already False?
            # All literals in learnt are currently False.
            # But some might be at lower levels.
            
            # Add to solver
            if not add_clause(learnt):
                return [False] * n
                
            # Backtrack to bt_level
            # Remove trail entries above bt_level
            
            while len(trail_lim) > bt_level:
                trail_lim.pop()
                
            # Unassign variables above bt_level
            while len(trail) > (trail_lim[-1] if trail_lim else 0):
                lit = trail.pop()
                var = abs(lit)
                assignment[var] = 0
                levels[var] = 0
                reasons[var] = None
                
            # Update activity
            for lit in learnt:
                var = abs(lit)
                activity[var] += 1.0
                
            # Decay activity
            for i in range(n):
                activity[i] *= activity_decay
                
            conflict_count += 1
            if conflict_count % restart_interval == 0:
                # Restart
                # Unassign all non-decisions?
                # Simple restart: unassign everything and re-propagate
                # This is expensive. 
                # Better: just reset the decision stack?
                # For n=50, full restart is fine.
                for i in range(n):
                    if assignment[i] != 0:
                        # If it's a decision, keep it? No, restart usually clears all.
                        # Let's clear all and re-propagate.
                        pass
                        
                # Actually, let's just clear the trail and re-propagate.
                # Keep the learnt clauses.
                
                # Unassign all
                for i in range(n):
                    assignment[i] = 0
                    levels[i] = 0
                    reasons[i] = None
                trail = []
                trail_lim = []
                
                # Re-propagate
                conflict_clause = propagate()
                if conflict_clause:
                    # Conflict at root? Unsat
                    return [False] * n
                    
                continue
        
        # No conflict
        # Pick a variable
        var = pick_var()
        
        if var == -1:
            # All assigned, Satisfiable
            break
            
        # Make decision
        # Try positive first (arbitrary)
        enqueue(var) # Positive literal
        trail_lim.append(len(trail))
        reasons[var] = None # Mark as decision
        
        # Update activity?
        activity[var] += 1.0
        
        num_decisions += 1
        
    # Satisfiable
    return [a == 1 for a in assignment]

# EVOLVE-BLOCK-END
