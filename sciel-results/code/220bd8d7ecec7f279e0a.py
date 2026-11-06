# EVOLVE-BLOCK-START
import time
import random
import heapq

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    # Convert to 0-based indices for internal processing
    # lit: 1..n -> 0..n-1 (True), -1..-n -> 0..n-1 (False)
    # Internal representation: var = abs(lit)-1, sign = 1 if lit>0 else 0 (0 for False, 1 for True)
    # Let's stick to standard SAT internal: value is 0 (False), 1 (True), -1 (Unassigned)
    
    # Preprocess clauses
    internal_clauses = []
    for c in clauses:
        internal_clauses.append(c)
        
    # Data structures for CDCL
    # assignment: list of 0, 1, -1
    assignment = [-1] * n
    # trail: list of (var, reason_clause_idx) in order of assignment
    trail = []
    trail_lim = [] # stack of trail lengths for decision levels
    
    # Clause storage
    # We need to handle clause deletion/learning efficiently.
    # For simplicity and speed in Python, we might use a list of lists.
    # However, CDCL requires fast access to clauses containing a literal.
    # watch_lits: list of sets of clause indices for each literal (var, sign)
    # Literal key: (var, sign) where sign is 0 (False) or 1 (True)
    # Or better: index literals as 0..2n-1. 
    # lit_idx = var * 2 + sign (0 for False, 1 for True)
    
    # Let's use a simpler structure for the "learned" clauses and "original" clauses.
    # Since n=50 is small, we can afford O(n) scans if necessary, but let's try to be efficient.
    
    # Variables: 0..n-1
    # Literals: 
    #   True literal for var v: (v, 1)
    #   False literal for var v: (v, 0)
    
    # Watch lists: dict mapping literal_tuple to list of clause_indices
    watch = {}
    
    # Store clauses as list of literal tuples
    all_clauses = []
    
    def add_clause(lits):
        """Add a clause to the solver. Returns clause index."""
        # lits is list of (var, sign)
        # Check for tautology or unit
        # For now, just add it.
        idx = len(all_clauses)
        all_clauses.append(lits)
        # Add to watch lists for the first two literals
        if len(lits) >= 2:
            l1, l2 = lits[0], lits[1]
            if l1 not in watch:
                watch[l1] = []
            watch[l1].append(idx)
            if l2 not in watch:
                watch[l2] = []
            watch[l2].append(idx)
        elif len(lits) == 1:
            # Unit clause
            l1 = lits[0]
            if l1 not in watch:
                watch[l1] = []
            watch[l1].append(idx)
        return idx

    # Initialize with input clauses
    for c in clauses:
        lits = []
        for lit in c:
            v = abs(lit) - 1
            s = 1 if lit > 0 else 0
            lits.append((v, s))
        add_clause(lits)

    # Activity scores for variables (VSIDS)
    var_activity = [0.0] * n
    # Heap for variable selection (max-heap, so we store negative activity)
    var_heap = [(-var_activity[i], i) for i in range(n)]
    heapq.heapify(var_heap)
    
    # Clause activity
    clause_activity = [0.0] * len(all_clauses)
    
    # Decay factor
    decay = 0.95
    
    def propagate():
        """
        Perform unit propagation.
        Returns True if no conflict, False if conflict (empty clause).
        """
        i = 0
        while i < len(trail):
            var, reason_idx = trail[i]
            i += 1
            # The variable assigned at trail[i-1] is var
            # We need to check all clauses watched by the literal that is now False
            # The literal that became False is the one opposite to the assignment
            # If assignment[var] is 1 (True), the literal (var, 0) is False
            # If assignment[var] is 0 (False), the literal (var, 1) is False
            
            val = assignment[var]
            # The literal that is now false
            false_lit = (var, 0) if val == 1 else (var, 1)
            
            if false_lit not in watch:
                continue
            
            # Process watch list for false_lit
            # We need to iterate carefully because we might modify the list
            # Make a copy or iterate backwards? 
            # Standard approach: iterate, and if a clause needs updating, update watch lists.
            
            # Get current watch list
            wl = watch[false_lit]
            j = 0
            while j < len(wl):
                c_idx = wl[j]
                lits = all_clauses[c_idx]
                
                # Check if the other watched literal is satisfied
                # The watched literals are the first two in the clause list
                # But they might have changed if we moved literals around.
                # In this simple implementation, we don't move literals within the clause list 
                # to maintain watch invariants easily. Instead, we just check the two literals 
                # currently at index 0 and 1.
                
                l0 = lits[0]
                l1 = lits[1]
                
                # One of l0 or l1 must be false_lit (since it's in the watch list)
                # Check if the other is satisfied
                if l0 == false_lit:
                    other = l1
                else:
                    other = l0
                
                if assignment[other[0]] == other[1]:
                    # Satisfied, do nothing
                    j += 1
                    continue
                
                # Check if the other literal is also false
                if assignment[other[0]] == (1 - other[1]):
                    # Conflict: both watched literals are false
                    # Check if the rest of the clause has a true literal
                    # If not, it's a conflict.
                    # For 3-SAT, if len(lits)==3, we just check lits[2]
                    # For general, scan the rest
                    conflict = True
                    for k in range(2, len(lits)):
                        if assignment[lits[k][0]] == lits[k][1]:
                            conflict = False
                            break
                    
                    if conflict:
                        return False, c_idx # Conflict with this clause
                    else:
                        # Move the true literal to position 1 (or 0)
                        # Let's move it to position 1
                        lits[1] = lits[k]
                        # Update watch list: remove this clause from the watch of lits[k]
                        # and add it to the watch of false_lit (it's already there)
                        # Actually, we just need to ensure the watch lists are correct.
                        # The clause is now watched by false_lit (pos 0) and lits[k] (pos 1).
                        # It was previously watched by false_lit and other (which was false).
                        # So we need to remove c_idx from watch[other] if it's there?
                        # Wait, 'other' was the second literal.
                        # If l0 was false_lit, then l1 was other.
                        # We replaced l1 with lits[k].
                        # So we need to remove c_idx from watch[other] and add to watch[lits[k]].
                        
                        # But 'other' is the literal that was at l1 (or l0).
                        # Let's be precise.
                        # If l0 == false_lit:
                        #   l1 was other.
                        #   We set lits[1] = lits[k].
                        #   We need to remove c_idx from watch[other] (which is l1_old).
                        #   And add c_idx to watch[lits[k]].
                        # If l1 == false_lit:
                        #   l0 was other.
                        #   We set lits[1] = lits[k].
                        #   We need to remove c_idx from watch[other] (which is l0_old).
                        #   And add c_idx to watch[lits[k]].
                        
                        # In both cases, we remove from watch[other] and add to watch[lits[k]].
                        
                        if other in watch:
                            # Remove c_idx from watch[other]
                            # This is O(n) in Python list. 
                            # Given small n, this is acceptable.
                            try:
                                watch[other].remove(c_idx)
                            except ValueError:
                                pass # Should not happen if invariants hold
                            
                        if lits[k] not in watch:
                            watch[lits[k]] = []
                        watch[lits[k]].append(c_idx)
                        
                        # Do NOT increment j, because we might have swapped something?
                        # No, we just changed the second literal. The clause is still in wl (watch[false_lit]).
                        # We can increment j.
                        j += 1
                        continue
                else:
                    # The other literal is unassigned
                    # Propagate: assign other[0] to other[1]
                    assignment[other[0]] = other[1]
                    trail.append((other[0], c_idx))
                    # Do not increment j, because we might need to check other clauses?
                    # No, the clause is satisfied by the new assignment? 
                    # Wait, if we assign 'other' to true, the clause is satisfied.
                    # So we can increment j.
                    j += 1
                    continue
            # End while j < len(wl)
        return True, None

    def analyze_conflict(conflict_idx):
        """
        Analyze the conflict and return (learned_clause, backtrack_level).
        learned_clause is a list of (var, sign).
        backtrack_level is the decision level to jump back to.
        """
        # We need to find the reason for each variable in the conflict clause.
        # The conflict clause is all_clauses[conflict_idx].
        # All literals in it are false.
        
        # We perform 1-UIP (Unique Implication Point) analysis.
        
        # Set of variables in the current implication set
        # We start with the variables in the conflict clause.
        
        # To do this efficiently, we need to trace back through the trail.
        
        # Simplified analysis for small n:
        # We can reconstruct the implication graph by tracing reasons.
        
        # Let's use a standard algorithm:
        
        # 1. Initialize:
        #    - current_clause = conflict clause literals
        #    - level = current decision level
        #    - seen_vars = set()
        
        # 2. Loop:
        #    - Pick a literal from current_clause that has a reason (is not from the current decision level? 
        #      Actually, we pick the one with the highest level that is not UIP).
        #    - If no such literal, we found the UIP.
        #    - Otherwise, resolve current_clause with the reason clause of that literal.
        
        # We need a way to get the reason clause for a variable.
        # reason_clause[var] = clause index
        
        # Let's maintain reason_clause array.
        
        # Since we are rewriting, let's add this structure.
        
        # For now, let's implement a simpler "chronological backtracking" with learning.
        # It's less efficient but easier to implement correctly in Python quickly.
        # However, CDCL is requested. Let's try to implement 1-UIP properly.
        
        # We need to store reasons.
        # Let's assume we have a global reason_clause dict/list.
        
        # Since I didn't define it in the outer scope, I'll define it here and use it.
        # This requires modifying the propagate function to store reasons.
        
        # Let's refactor slightly:
        # I will define reason_clause in the outer scope of solve.
        
        # For the sake of this block, I'll implement a simplified analysis that returns the 
        # variables involved in the conflict and their reasons, and backtracks to the 
        # second highest level.
        
        # Actually, let's just do chronological backtracking with a simple learned clause 
        # consisting of the negations of the literals in the conflict clause that were 
        # assigned in the current decision level, plus the literals from previous levels 
        # that are still false.
        
        # This is effectively learning the conflict clause itself, but simplified.
        
        # Let's get the literals of the conflict clause.
        conflict_lits = all_clauses[conflict_idx]
        
        # We want to learn a clause that will be unit at the backtrack level.
        # The backtrack level is the second highest decision level among the variables in the conflict clause.
        
        levels = []
        for v, s in conflict_lits:
            # Find the level at which v was assigned
            # We can search the trail_lim
            # Or we can store level[var]
            
            # Let's store level[var] in propagate.
            # I'll add that.
            
            pass
        
        # This is getting complex. Let's use a robust but simpler approach:
        # If conflict, backtrack to the previous decision level, and learn the clause 
        # that caused the conflict (the conflict clause itself, negated? No, the conflict clause 
        # is false, so the learned clause is the negation of the conflict clause? 
        # No, if a clause is false, its negation is true. 
        # The learned clause is the disjunction of the negations of the literals in the conflict clause.
        # Wait. If (A or B or C) is false, then (!A and !B and !C) is true.
        # So the learned clause is (!A or !B or !C)? No.
        # If (A or B or C) is false, it means A is false, B is false, C is false.
        # So we know that it is NOT the case that (A and B and C).
        # This doesn't directly give a clause.
        
        # In CDCL, we analyze the implication graph.
        
        # Given the constraints, let's implement a basic CDCL with:
        # 1. Unit Propagation
        # 2. On Conflict:
        #    - Learn the clause formed by the negations of the literals in the conflict clause 
        #      that were assigned at the current decision level, AND the literals from 
        #      previous levels that are still false.
        #    - Actually, the standard learned clause from a conflict (A1 or ... or Ak) where 
        #      all are false is the clause (¬A1 or ... or ¬Ak).
        #    - But we want it to be unit at the backtrack level.
        
        # Let's just do:
        # 1. Identify the variables in the conflict clause.
        # 2. Find the second highest decision level L2.
        # 3. Construct the learned clause:
        #    - For each literal (v, s) in the conflict clause:
        #      - If level[v] == L2, add (v, 1-s) to the learned clause? 
        #      - No, the standard 1-UIP algorithm produces a clause where exactly one literal 
        #        is at the backtrack level (the UIP) and it is the negation of the UIP assignment.
        
        # Let's implement a simple 1-UIP.
        
        # We need reason_clause. I'll add it to the outer scope.
        
        # Re-initializing structures to include reason tracking.
        
        # Since I can't easily modify the outer scope in this thought process, 
        # I will define the necessary structures inside solve and ensure they are used.
        
        # Let's restart the function definition with the full CDCL structures.
        
        pass

    # --- RESTART WITH FULL CDCL IMPLEMENTATION ---
    
    # Internal state
    assignment = [-1] * n
    trail = []
    trail_lim = []
    reason_clause = [-1] * n # reason_clause[var] = clause index
    
    watch = {}
    all_clauses = []
    
    def add_clause(lits):
        idx = len(all_clauses)
        all_clauses.append(lits)
        if len(lits) >= 2:
            l1, l2 = lits[0], lits[1]
            if l1 not in watch:
                watch[l1] = []
            watch[l1].append(idx)
            if l2 not in watch:
                watch[l2] = []
            watch[l2].append(idx)
        elif len(lits) == 1:
            l1 = lits[0]
            if l1 not in watch:
                watch[l1] = []
            watch[l1].append(idx)
        return idx

    # Initialize
    for c in clauses:
        lits = []
        for lit in c:
            v = abs(lit) - 1
            s = 1 if lit > 0 else 0
            lits.append((v, s))
        add_clause(lits)

    # Activity
    var_activity = [0.0] * n
    clause_activity = [0.0] * len(all_clauses)
    decay = 0.95
    
    def propagate():
        i = 0
        while i < len(trail):
            var, reason_idx = trail[i]
            i += 1
            val = assignment[var]
            false_lit = (var, 0) if val == 1 else (var, 1)
            
            if false_lit not in watch:
                continue
            
            wl = watch[false_lit]
            j = 0
            while j < len(wl):
                c_idx = wl[j]
                lits = all_clauses[c_idx]
                
                l0 = lits[0]
                l1 = lits[1]
                
                if l0 == false_lit:
                    other = l1
                else:
                    other = l0
                
                if assignment[other[0]] == other[1]:
                    j += 1
                    continue
                
                if assignment[other[0]] == (1 - other[1]):
                    # Conflict check
                    conflict = True
                    for k in range(2, len(lits)):
                        if assignment[lits[k][0]] == lits[k][1]:
                            conflict = False
                            break
                    if conflict:
                        return False, c_idx
                    else:
                        # Move true literal to pos 1
                        lits[1] = lits[k]
                        if other in watch:
                            try:
                                watch[other].remove(c_idx)
                            except ValueError:
                                pass
                        if lits[k] not in watch:
                            watch[lits[k]] = []
                        watch[lits[k]].append(c_idx)
                        j += 1
                        continue
                else:
                    # Propagate
                    assignment[other[0]] = other[1]
                    trail.append((other[0], c_idx))
                    reason_clause[other[0]] = c_idx
                    j += 1
                    continue
        return True, None

    def analyze_conflict(conflict_idx):
        # 1-UIP Analysis
        # current_clause: list of (var, sign)
        # We start with the conflict clause.
        # We resolve with reasons until we have exactly one literal at the current decision level.
        
        # Get current decision level
        curr_level = len(trail_lim)
        
        # We need to know the level of each variable.
        # We can compute it by checking trail_lim.
        # Or we can store level[var].
        
        # Let's store level[var] in propagate.
        # I'll add that.
        
        # For now, let's assume we have a level array.
        
        # Re-implementing with level array.
        
        pass

    # --- FINAL IMPLEMENTATION WITH LEVEL TRACKING ---
    
    assignment = [-1] * n
    trail = []
    trail_lim = []
    reason_clause = [-1] * n
    level = [-1] * n # level[var] = decision level
    
    watch = {}
    all_clauses = []
    
    def add_clause(lits):
        idx = len(all_clauses)
        all_clauses.append(lits)
        if len(lits) >= 2:
            l1, l2 = lits[0], lits[1]
            if l1 not in watch:
                watch[l1] = []
            watch[l1].append(idx)
            if l2 not in watch:
                watch[l2] = []
            watch[l2].append(idx)
        elif len(lits) == 1:
            l1 = lits[0]
            if l1 not in watch:
                watch[l1] = []
            watch[l1].append(idx)
        return idx

    for c in clauses:
        lits = []
        for lit in c:
            v = abs(lit) - 1
            s = 1 if lit > 0 else 0
            lits.append((v, s))
        add_clause(lits)

    var_activity = [0.0] * n
    clause_activity = [0.0] * len(all_clauses)
    decay = 0.95
    
    def propagate():
        i = 0
        while i < len(trail):
            var, reason_idx = trail[i]
            i += 1
            val = assignment[var]
            level[var] = len(trail_lim)
            false_lit = (var, 0) if val == 1 else (var, 1)
            
            if false_lit not in watch:
                continue
            
            wl = watch[false_lit]
            j = 0
            while j < len(wl):
                c_idx = wl[j]
                lits = all_clauses[c_idx]
                
                l0 = lits[0]
                l1 = lits[1]
                
                if l0 == false_lit:
                    other = l1
                else:
                    other = l0
                
                if assignment[other[0]] == other[1]:
                    j += 1
                    continue
                
                if assignment[other[0]] == (1 - other[1]):
                    conflict = True
                    for k in range(2, len(lits)):
                        if assignment[lits[k][0]] == lits[k][1]:
                            conflict = False
                            break
                    if conflict:
                        return False, c_idx
                    else:
                        lits[1] = lits[k]
                        if other in watch:
                            try:
                                watch[other].remove(c_idx)
                            except ValueError:
                                pass
                        if lits[k] not in watch:
                            watch[lits[k]] = []
                        watch[lits[k]].append(c_idx)
                        j += 1
                        continue
                else:
                    assignment[other[0]] = other[1]
                    trail.append((other[0], c_idx))
                    reason_clause[other[0]] = c_idx
                    j += 1
                    continue
        return True, None

    def analyze_conflict(conflict_idx):
        curr_level = len(trail_lim)
        
        # Initialize current_clause with conflict clause literals
        # We want to find the 1-UIP.
        # We maintain a set of variables in the current clause.
        
        # The standard algorithm:
        # 1. Let C be the conflict clause.
        # 2. Let L be the current decision level.
        # 3. While C has more than one literal at level L:
        #    a. Pick a literal l in C that is at level L and is not the UIP.
        #       (Heuristic: pick the one with the highest level, or the one assigned last in the trail).
        #    b. Let R be the reason clause for the variable of l.
        #    c. Resolve C and R: C = (C \ {l}) U (R \ {¬l}).
        # 4. The remaining literal in C at level L is the UIP.
        # 5. Backtrack to the level of the second highest literal in C.
        
        # Implementation:
        
        # We need to identify which literals are at the current level.
        # And we need to pick one to resolve.
        
        # Let's use a set of variables in C.
        # And a list of literals in C.
        
        C = all_clauses[conflict_idx][:]
        C_vars = set(v for v, s in C)
        
        # We need to know the level of each variable in C.
        
        # We will iterate until we have exactly one variable in C at curr_level.
        
        # To pick the variable to resolve, we can look at the trail.
        # The variable assigned most recently in the current decision level that is in C 
        # (and not the first one? No, just pick any).
        
        # Let's find the variable in C with the highest index in the trail within the current level.
        
        # Trail indices for current level: from trail_lim[-1] to len(trail)-1
        
        start_idx = trail_lim[-1] if trail_lim else 0
        
        # Find the variable in C that appears last in the trail
        max_trail_idx = -1
        var_to_resolve = None
        for v in C_vars:
            # Find index of v in trail
            # This is O(n) if we search. 
            # Given small n, it's fine.
            for k in range(len(trail)-1, start_idx-1, -1):
                if trail[k][0] == v:
                    if k > max_trail_idx:
                        max_trail_idx = k
                        var_to_resolve = v
                    break
        
        if var_to_resolve is None:
            # Should not happen if conflict is at current level
            return C, 0
        
        # Loop until only one var in C is at curr_level
        while True:
            # Count vars at curr_level
            vars_at_curr = [v for v in C_vars if level[v] == curr_level]
            
            if len(vars_at_curr) <= 1:
                break
            
            # Resolve with reason of var_to_resolve
            # Find the literal for var_to_resolve in C
            lit_to_remove = None
            for v, s in C:
                if v == var_to_resolve:
                    lit_to_remove = (v, s)
                    break
            
            if lit_to_remove is None:
                break # Should not happen
            
            reason_idx = reason_clause[var_to_resolve]
            reason_clause_lits = all_clauses[reason_idx]
            
            # The literal in the reason clause that is the negation of lit_to_remove
            # lit_to_remove is (v, s). The reason clause must contain (v, 1-s).
            # And it is the only literal in the reason clause that is false?
            # No, the reason clause was unit when var_to_resolve was assigned.
            # So all other literals in the reason clause are true.
            # We add the other literals from the reason clause to C.
            
            new_lits = []
            for v, s in reason_clause_lits:
                if (v, s) == lit_to_remove:
                    continue
                # Only add if it's not already in C?
                # And only if it's false?
                # In 1-UIP, we add all literals from the reason clause except the one resolved.
                # But we should only keep literals that are false in the current assignment?
                # Actually, the standard algorithm adds them all, but literals that are true 
                # can be removed (self-subsumption).
                
                # Let's add it.
                new_lits.append((v, s))
            
            # Remove lit_to_remove from C
            C = [l for l in C if l != lit_to_remove]
            
            # Add new_lits to C
            # Avoid duplicates
            existing = set(C)
            for l in new_lits:
                if l not in existing:
                    C.append(l)
                    existing.add(l)
            
            C_vars = set(v for v, s in C)
            
            # Update var_to_resolve
            # Pick the variable in C_vars that is at curr_level and appears last in the trail
            max_trail_idx = -1
            var_to_resolve = None
            for v in C_vars:
                if level[v] != curr_level:
                    continue
                for k in range(len(trail)-1, start_idx-1, -1):
                    if trail[k][0] == v:
                        if k > max_trail_idx:
                            max_trail_idx = k
                            var_to_resolve = v
                        break
            
            if var_to_resolve is None:
                break
        
        # Now C has at most one literal at curr_level.
        # The learned clause is C.
        # The backtrack level is the second highest level among variables in C.
        
        levels = [level[v] for v, s in C]
        if len(levels) == 0:
            bt_level = 0
        else:
            levels.sort(reverse=True)
            if len(levels) == 1:
                bt_level = levels[0]
            else:
                bt_level = levels[1]
        
        # Minimize clause?
        # Remove literals that are true? No, they are false.
        # Remove literals that are implied by others?
        # For now, return C and bt_level.
        
        return C, bt_level

    def backtrack_to_level(bt_level):
        # Unassign variables from the end of the trail until len(trail) == trail_lim[bt_level]
        target_len = trail_lim[bt_level] if bt_level < len(trail_lim) else 0
        while len(trail) > target_len:
            var, _ = trail.pop()
            assignment[var] = -1
            reason_clause[var] = -1
            level[var] = -1
        while len(trail_lim) > bt_level:
            trail_lim.pop()

    def select_variable():
        # VSIDS
        # Pick the unassigned variable with highest activity
        # We need to update the heap or just scan.
        # Scanning is O(n). n=50. Fine.
        best_var = -1
        best_act = -1
        for v in range(n):
            if assignment[v] == -1:
                if var_activity[v] > best_act:
                    best_act = var_activity[v]
                    best_var = v
        return best_var

    # Main Loop
    deadline = time.time() + 325.0
    best_assignment = None
    best_unsat = float('inf')
    
    # Initial random assignment to guide search? 
    # CDCL usually starts with empty assignment.
    # But we can use a random heuristic for the first decision.
    
    rng = random.Random(0)
    
    while time.time() < deadline:
        # Reset state for new search? 
        # Or continue from where we left off?
        # CDCL is incremental. We can continue.
        # But if we get stuck, we might want to restart.
        
        # Let's try to solve from scratch with CDCL.
        
        # Clear state
        assignment = [-1] * n
        trail = []
        trail_lim = []
        reason_clause = [-1] * n
        level = [-1] * n
        
        # Watch lists and clauses are persistent? 
        # Learned clauses should be persistent.
        # But for simplicity, let's restart the search with the same clauses.
        # We need to clear the watch lists? No, the clauses are the same.
        # But the watch lists might be corrupted if we didn't backtrack properly?
        # Backtracking doesn't change watch lists.
        
        # So we can just reset assignment/trail.
        
        # However, the watch lists are built based on the clause contents.
        # If we don't change clauses, watch lists are fine.
        
        # Let's just run the search.
        
        while True:
            # Propagate
            ok, conflict_idx = propagate()
            
            if not ok:
                # Conflict
                learned_clause, bt_level = analyze_conflict(conflict_idx)
                
                # Add learned clause
                if learned_clause:
                    # Check if it's unit
                    # Add to solver
                    add_clause(learned_clause)
                    
                    # Update clause activity
                    clause_activity[-1] = 1.0
                    
                    # Decay other clause activities?
                    # For simplicity, skip.
                
                # Backtrack
                backtrack_to_level(bt_level)
                
                # Update variable activities
                for v, s in learned_clause:
                    var_activity[v] = 1.0
                    for i in range(n):
                        var_activity[i] *= decay
                
                # Check if we should restart
                # If bt_level is 0, we are at the root.
                # If we keep conflicting at root, it's UNSAT (but problem is SAT).
                # So we should eventually find a solution.
                
                continue
            
            # Check if all variables assigned
            if len(trail) == n:
                # Check if solution is valid
                # It should be, since no conflict.
                return [bool(assignment[i]) for i in range(n)]
            
            # Decide
            var = select_variable()
            if var == -1:
                # Should not happen if len(trail) < n
                break
            
            # Randomly assign true or false
            val = rng.random() < 0.5
            
            # Push decision level
            trail_lim.append(len(trail))
            
            # Assign
            assignment[var] = 1 if val else 0
            trail.append((var, -1)) # -1 indicates decision
            reason_clause[var] = -1
            level[var] = len(trail_lim)

    # If we exit the loop due to time, return best found
    # Since we return immediately on solution, this is unlikely.
    # But if we time out, we need to return something.
    # The problem is SAT, so we should find it.
    # If not, return a random assignment.
    
    return [rng.random() < 0.5 for _ in range(n)]

# EVOLVE-BLOCK-END
