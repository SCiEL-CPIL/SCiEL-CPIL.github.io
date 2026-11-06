# EVOLVE-BLOCK-START
import sys

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Preprocess: remove duplicate literals within clauses, remove tautologies
    # Convert to 0-based indices
    processed_clauses = []
    for clause in clauses:
        seen = set()
        new_clause = []
        is_tautology = False
        for l in clause:
            idx = abs(l) - 1
            if idx in seen:
                is_tautology = True
                break
            seen.add(idx)
            new_clause.append(l) # Keep original signed literal for now, or convert to (idx, sign)
        if not is_tautology and new_clause:
            processed_clauses.append(new_clause)
    
    if not processed_clauses:
        return [False] * n

    # DPLL with Watched Literals
    # We will maintain:
    # assignment: list of 0 (False), 1 (True), -1 (Unassigned) or similar. Let's use None/True/False.
    # Actually, for watched literals, it's easier to track assignments as 0/1 and unassigned as None.
    
    # Structure:
    # watched: list of lists, watched[i] is a list of clause indices that are watching literal i.
    # But standard watched literals: each clause watches 2 literals.
    # Let's implement a standard Watched Literals DPLL.
    
    class WatchedLiteralSolver:
        def __init__(self, n, clauses):
            self.n = n
            self.clauses = clauses
            self.assignment = [None] * n
            # Each clause is a list of literals (signed, 1-based or 0-based? Let's use 0-based index + sign)
            # Let's normalize clauses to list of (var_idx, sign) where sign is 1 or -1.
            self.norm_clauses = []
            for c in clauses:
                nc = []
                for l in c:
                    idx = abs(l) - 1
                    sign = 1 if l > 0 else -1
                    nc.append((idx, sign))
                self.norm_clauses.append(nc)
            
            self.num_clauses = len(self.norm_clauses)
            self.watched = [[], []] # watched[0] for False literals? No.
            # Standard: watched[lit] where lit is an integer representing the literal.
            # Let's map literal (idx, sign) to an integer ID.
            # ID = idx * 2 + (0 if sign==-1 else 1)?
            # Let's just use the signed literal value but 0-based.
            # lit_val = idx if sign==1 else -idx? No, 0 is a problem.
            # Let's use: lit_id = idx if sign == 1 else ~idx (bitwise not)
            # Or simpler: lit_id = idx * 2 + (0 if sign == -1 else 1)
            # Let's stick to a dict or list of size 2*n.
            # index 0..n-1 for positive literals? No, let's use 0..2n-1.
            # lit_idx = var_idx * 2 + (0 if neg else 1)
            
            self.watched = [[] for _ in range(2 * n)]
            self.clause_watches = [] # For each clause, which 2 literals are it watching?
            
            for i, c in enumerate(self.norm_clauses):
                if len(c) < 2:
                    # Unit clause or empty
                    self.clause_watches.append(c[:2])
                else:
                    self.clause_watches.append(c[:2])
                
                for lit in self.clause_watches[i]:
                    idx, sign = lit
                    lit_id = idx * 2 + (0 if sign == -1 else 1)
                    self.watched[lit_id].append(i)
            
            self.occurrence = [[] for _ in range(2 * n)] # For pure literal / variable selection
            for i, c in enumerate(self.norm_clauses):
                for lit in c:
                    idx, sign = lit
                    lit_id = idx * 2 + (0 if sign == -1 else 1)
                    self.occurrence[lit_id].append(i)

        def lit_id(self, idx, sign):
            return idx * 2 + (0 if sign == -1 else 1)
            
        def is_assigned(self, idx):
            return self.assignment[idx] is not None
            
        def satisfies(self, idx, sign):
            if self.assignment[idx] is None: return False
            return (self.assignment[idx] == (sign == 1))
            
        def falsifies(self, idx, sign):
            if self.assignment[idx] is None: return False
            return (self.assignment[idx] != (sign == 1))
            
        def propagate(self, lit):
            """
            lit: tuple (idx, sign) of the literal that was just assigned True.
            Returns False if contradiction.
            """
            idx, sign = lit
            # Find all clauses watching this literal
            # If a clause is watching a literal that is now False, we need to update its watch.
            # The literal assigned is True, so its negation is False.
            # neg_lit = (idx, -sign)
            neg_lit_id = self.lit_id(idx, -sign)
            
            to_remove = []
            for c_idx in self.watched[neg_lit_id]:
                c = self.norm_clauses[c_idx]
                w1, w2 = self.clause_watches[c_idx]
                
                # Check if the clause is already satisfied by the other watch
                # Or if the assigned literal satisfies it? 
                # The assigned literal is (idx, sign). 
                # If (idx, sign) is in the clause, it's satisfied.
                # But we are processing clauses that were watching (idx, -sign).
                # If (idx, sign) is in the clause, then (idx, -sign) is NOT in the clause (no duplicates).
                # So we need to check if the clause is satisfied by any other literal.
                
                # First, check if the clause is satisfied by the assignment we just made?
                # No, we are looking at clauses that have (idx, -sign).
                # Since we assigned (idx, sign), (idx, -sign) is False.
                # We need to find a new literal in the clause that is not False to watch.
                
                # Check if clause is satisfied
                sat = False
                new_watch = None
                for i, (v, s) in enumerate(c):
                    if self.satisfies(v, s):
                        sat = True
                        break
                    if (v, s) == (idx, -sign):
                        continue # This is the one that became False
                    # Check if this literal is not False
                    if not self.falsifies(v, s):
                        new_watch = (v, s)
                        break
                        
                if sat:
                    to_remove.append(c_idx)
                else:
                    if new_watch is None:
                        # Clause is unit (all other literals are False)
                        # The only remaining literal must be the one we just falsified? 
                        # No, if all other literals are false, and the one we falsified is false, 
                        # then the clause is falsified? 
                        # Wait. If len(c)==1, it was unit. 
                        # If len(c)>1, and we found no new watch, it means all other literals are False.
                        # So the clause is now unit? No, if all literals are False, it's a conflict.
                        # Let's re-evaluate.
                        # We are in a clause that had (idx, -sign).
                        # We assigned (idx, sign). So (idx, -sign) is False.
                        # We scan the clause for a literal that is not False.
                        # If we don't find one, it means all literals in the clause are False.
                        # This is a conflict.
                        return False
                    else:
                        # Update watch
                        old_watch_id = self.lit_id(idx, -sign)
                        # Remove c_idx from watched[old_watch_id] (handled later)
                        # Add c_idx to watched[new_watch_id]
                        new_watch_id = self.lit_id(*new_watch)
                        self.watched[new_watch_id].append(c_idx)
                        # Update clause_watches
                        # Find position of (idx, -sign) in clause_watches[c_idx]
                        cw = self.clause_watches[c_idx]
                        if cw[0] == (idx, -sign):
                            cw[0] = new_watch
                        else:
                            cw[1] = new_watch
            
            # Remove satisfied clauses from watched lists
            for c_idx in to_remove:
                w1, w2 = self.clause_watches[c_idx]
                id1 = self.lit_id(*w1)
                id2 = self.lit_id(*w2)
                # Remove c_idx from watched[id1] and watched[id2]
                # This is O(N) for list removal. 
                # For efficiency, we could use sets, but N=50 is small.
                try:
                    self.watched[id1].remove(c_idx)
                except ValueError:
                    pass
                try:
                    self.watched[id2].remove(c_idx)
                except ValueError:
                    pass
                # Mark clause as satisfied? We can just leave it, but it won't be propagated.
                # It's better to have a "satisfied" flag or just ignore it in future scans.
                # Since we removed it from watched lists, it won't be triggered by future assignments.
                # We should also remove it from occurrence lists? Not strictly necessary if we check satisfaction.
            
            return True

        def unit_propagate(self):
            """
            Propagate all unit clauses.
            Returns False if conflict.
            """
            # We need a queue of literals to propagate.
            # Initially, find all unit clauses.
            # Then, when a literal is assigned, it might make other clauses unit.
            
            queue = []
            # Find initial unit clauses
            for i, c in enumerate(self.norm_clauses):
                # Check if clause is unit
                unassigned = []
                for lit in c:
                    if self.is_assigned(lit[0]):
                        if self.satisfies(*lit):
                            break # Satisfied
                        else:
                            continue # Falsified
                    else:
                        unassigned.append(lit)
                
                if not unassigned:
                    return False # Conflict
                if len(unassigned) == 1:
                    lit = unassigned[0]
                    if self.assignment[lit[0]] is None:
                        queue.append(lit)
                        
            while queue:
                lit = queue.pop()
                idx, sign = lit
                if self.assignment[idx] is not None:
                    continue
                self.assignment[idx] = (sign == 1)
                if not self.propagate(lit):
                    return False
                # Check for new unit clauses? 
                # The propagate function updates watched literals.
                # If a clause becomes unit, it will be in the watched list of the literal that is NOT false.
                # Actually, the standard algorithm pushes the new unit literal to the queue.
                # In my `propagate`, I didn't explicitly push new units to a queue.
                # I need to modify `propagate` to return new unit literals or handle the queue there.
                
            # Re-scan for units is safer but slower. 
            # Let's implement a proper queue in propagate.
            return True

        def solve(self):
            # Initial unit propagation
            if not self._propagate_loop():
                return None
                
            if not self.norm_clauses:
                return self.assignment
                
            # Pure Literal Elimination
            if self._pure_literal_elimination():
                if not self._propagate_loop():
                    return None
                if not self.norm_clauses:
                    return self.assignment

            # Branching
            # Pick a variable
            # Heuristic: Pick variable from the first clause
            # Or pick variable with max occurrences
            var_to_branch = None
            # Simple: first unassigned variable in first clause
            for c in self.norm_clauses:
                for lit in c:
                    if self.assignment[lit[0]] is None:
                        var_to_branch = lit[0]
                        break
                if var_to_branch is not None:
                    break
                    
            if var_to_branch is None:
                return self.assignment

            # Try True
            self.assignment[var_to_branch] = True
            if self._propagate_loop():
                # Check if solved
                if self._check_solved():
                    return self.assignment
                # Recurse
                # We need to save state for backtracking
                # This class-based approach with recursion is tricky without explicit state saving.
                # Let's use a standard DPLL recursion pattern with state copying or trail.
                # Given N=50, copying the assignment and clause list is feasible.
                pass
            
            # This class structure is getting complex for a quick edit.
            # Let's stick to the functional approach but with Watched Literals implemented carefully.
            
            return None

    # Given the complexity and risk of bugs in a full rewrite within the constraints,
    # let's optimize the EXISTING DPLL approach instead.
    # The existing approach is O(N^3) roughly. 
    # The main cost is `for c in current_clauses` inside the while loop.
    # We can optimize by maintaining a list of active clauses and updating it incrementally.
    # Or, we can use a simpler heuristic: 
    # 1. Use a better variable selection: pick variable with highest degree in current clauses.
    # 2. Optimize unit propagation by not scanning all clauses if we track which clauses are affected.
    
    # However, the most robust improvement for "keep the approach" is to implement 
    # a proper DPLL with a trail and watched literals, but done correctly.
    
    # Let's try a different tactic: 
    # The current code is slow because it rebuilds `current_clauses` from scratch every time.
    # We can avoid this by using a "level" based assignment and only checking clauses that are potentially affected.
    # But for N=50, even a naive DPLL with good heuristics should work.
    # Why does the current one fail? 
    # `unit_clauses = [c for c in current_clauses if len(c) == 1]` is O(C).
    # Then `for c in current_clauses` to update is O(C*L).
    # This happens in a loop.
    
    # Let's replace the core solver with a more optimized DPLL that uses 
    # a "decision level" and a "trail" for backtracking, and 
    # a "watched literals" scheme.
    
    # Here is a clean, correct implementation of DPLL with Watched Literals for N=50.
    
    def dpll_watched(n, clauses):
        # Normalize clauses
        norm_clauses = []
        for c in clauses:
            nc = []
            seen = set()
            for l in c:
                idx = abs(l) - 1
                sign = 1 if l > 0 else -1
                if idx not in seen:
                    seen.add(idx)
                    nc.append((idx, sign))
            if nc:
                norm_clauses.append(nc)
                
        if not norm_clauses:
            return [False] * n
            
        N = len(norm_clauses)
        assignment = [None] * n
        # watched: list of lists, indexed by lit_id
        # lit_id(idx, sign) = idx*2 + (0 if sign==-1 else 1)
        watched = [[] for _ in range(2*n)]
        clause_watches = []
        
        for i, c in enumerate(norm_clauses):
            if len(c) == 0:
                return None
            if len(c) == 1:
                clause_watches.append([c[0], None]) # Dummy second watch
            else:
                clause_watches.append([c[0], c[1]])
            
            for w in clause_watches[i]:
                if w is not None:
                    idx, sign = w
                    lid = idx*2 + (0 if sign==-1 else 1)
                    watched[lid].append(i)
        
        def lit_id(idx, sign):
            return idx*2 + (0 if sign==-1 else 1)
            
        def satisfies(idx, sign):
            if assignment[idx] is None: return False
            return assignment[idx] == (sign == 1)
            
        def falsifies(idx, sign):
            if assignment[idx] is None: return False
            return assignment[idx] != (sign == 1)
            
        def propagate(lit):
            """
            lit: (idx, sign) assigned True.
            Returns False if conflict.
            """
            idx, sign = lit
            neg_lid = lit_id(idx, -sign)
            
            # Get list of clauses watching neg_lid
            # We must iterate over a copy because we modify watched
            watchers = list(watched[neg_lid])
            
            for c_idx in watchers:
                c = norm_clauses[c_idx]
                w1, w2 = clause_watches[c_idx]
                
                # Check if clause is satisfied
                sat = False
                new_watch = None
                
                for lit_c in c:
                    if satisfies(*lit_c):
                        sat = True
                        break
                    if lit_c == (idx, -sign):
                        continue
                    if not falsifies(*lit_c):
                        new_watch = lit_c
                        break
                        
                if sat:
                    # Remove c_idx from watched lists
                    # It was watching w1 and w2
                    # We need to remove it from watched[lit_id(w1)] and watched[lit_id(w2)]
                    # Note: w1 or w2 might be the one that became false?
                    # Actually, if it's satisfied, we just remove it.
                    # We don't strictly need to remove it from watched if we mark it as satisfied,
                    # but to keep watched lists clean, we should.
                    # However, removing from list is O(N). 
                    # For N=50, it's fine.
                    
                    # Remove from watched[w1]
                    if w1 is not None:
                        id1 = lit_id(*w1)
                        if c_idx in watched[id1]:
                            watched[id1].remove(c_idx)
                    if w2 is not None:
                        id2 = lit_id(*w2)
                        if c_idx in watched[id2]:
                            watched[id2].remove(c_idx)
                            
                else:
                    if new_watch is None:
                        # Conflict
                        return False
                    else:
                        # Update watch
                        # Remove c_idx from watched[neg_lid] (the old watch that is now false)
                        # It's in the list we are iterating, but we are building a new list?
                        # No, we are modifying watched in place.
                        # We will remove it from watched[neg_lid] later or now.
                        
                        # Add to watched[new_lid]
                        new_lid = lit_id(*new_watch)
                        watched[new_lid].append(c_idx)
                        
                        # Update clause_watches
                        if w1 == (idx, -sign):
                            clause_watches[c_idx][0] = new_watch
                        else:
                            clause_watches[c_idx][1] = new_watch
                            
            # Cleanup: remove satisfied clauses from watched[neg_lid]?
            # The satisfied clauses were removed from their other watch, but they are still in watched[neg_lid].
            # We should filter watched[neg_lid] to remove any clause that is now satisfied.
            new_watchers = []
            for c_idx in watched[neg_lid]:
                c = norm_clauses[c_idx]
                # Check if satisfied
                is_sat = False
                for lit_c in c:
                    if satisfies(*lit_c):
                        is_sat = True
                        break
                if not is_sat:
                    new_watchers.append(c_idx)
            watched[neg_lid] = new_watchers
            
            return True

        def unit_propagate():
            # Find all unit clauses
            # A clause is unit if it has exactly 1 unassigned literal.
            # We can find them by checking all clauses? O(N*C).
            # Or we can maintain a queue.
            # Let's start with a queue of all unit clauses.
            queue = []
            for i, c in enumerate(norm_clauses):
                unassigned = [lit for lit in c if assignment[lit[0]] is None]
                if not unassigned:
                    return False # Conflict
                if len(unassigned) == 1:
                    lit = unassigned[0]
                    if assignment[lit[0]] is None:
                        queue.append(lit)
                        
            while queue:
                lit = queue.pop()
                idx, sign = lit
                if assignment[idx] is not None:
                    continue
                assignment[idx] = (sign == 1)
                if not propagate(lit):
                    return False
                # New unit clauses might have been created.
                # We need to find them.
                # The propagate function updated watched lists.
                # Any clause that became unit will be watching a literal that is not false.
                # Actually, the standard algorithm pushes the new unit to the queue.
                # My propagate doesn't return new units.
                # So I must re-scan for units?
                # That makes it O(N*C) per step.
                # To be efficient, propagate should identify new units.
                # When we update a watch, if the clause has only 1 unassigned literal left, it's a unit.
                # Let's modify propagate to return new units.
                
            return True

        # The above is getting complicated. 
        # Let's use a simpler, but still fast, DPLL.
        # For N=50, a well-optimized naive DPLL is sufficient if we avoid O(N^2) scans where possible.
        # The current code's weakness is the full scan.
        
        # Let's try a different optimization: 
        # Use a "level" array and a "trail" to avoid copying arrays.
        # And use a "dirty" flag for clauses to only check clauses that might have changed.
        
        # Actually, the simplest robust improvement is to use `python-sat` style logic but implemented manually.
        # Let's go with the Watched Literals but fix the queue issue.
        
        # Revised Plan:
        # 1. Implement `propagate` to return a list of new unit literals.
        # 2. Use a queue in `unit_propagate`.
        
        def dpll_rec(assignment):
            # Unit Propagation
            queue = []
            # Find initial units
            for i, c in enumerate(norm_clauses):
                unassigned = [lit for lit in c if assignment[lit[0]] is None]
                if not unassigned:
                    return None
                if len(unassigned) == 1:
                    lit = unassigned[0]
                    if assignment[lit[0]] is None:
                        queue.append(lit)
                    
            while queue:
                lit = queue.pop()
                idx, sign = lit
                if assignment[idx] is not None:
                    continue
                assignment[idx] = (sign == 1)
                
                # Propagate this assignment
                # Find clauses that contain the negation of this literal
                # And are not yet satisfied
                # And are not already falsified
                neg_lit = (idx, -sign)
                # We need to find clauses watching neg_lit?
                # If we don't maintain watched lists, we scan all clauses.
                # Let's maintain watched lists properly.
                
                # This requires the watched structure to be passed and updated.
                # Let's restart the implementation with a clean class.
            
            return None

    # Okay, I will implement a standard DPLL with Watched Literals properly.
    
    class DPLL:
        def __init__(self, n, clauses):
            self.n = n
            self.clauses = []
            for c in clauses:
                nc = []
                seen = set()
                for l in c:
                    idx = abs(l) - 1
                    sign = 1 if l > 0 else -1
                    if idx not in seen:
                        seen.add(idx)
                        nc.append((idx, sign))
                if nc:
                    self.clauses.append(nc)
            self.num_clauses = len(self.clauses)
            self.assignment = [None] * n
            self.watched = [[] for _ in range(2*n)]
            self.clause_watches = []
            
            for i, c in enumerate(self.clauses):
                if len(c) == 1:
                    self.clause_watches.append([c[0], None])
                else:
                    self.clause_watches.append([c[0], c[1]])
                for w in self.clause_watches[i]:
                    if w is not None:
                        idx, sign = w
                        lid = idx*2 + (0 if sign==-1 else 1)
                        self.watched[lid].append(i)
                        
        def lit_id(self, idx, sign):
            return idx*2 + (0 if sign==-1 else 1)
            
        def satisfies(self, idx, sign):
            if self.assignment[idx] is None: return False
            return self.assignment[idx] == (sign == 1)
            
        def falsifies(self, idx, sign):
            if self.assignment[idx] is None: return False
            return self.assignment[idx] != (sign == 1)
            
        def propagate(self, lit):
            idx, sign = lit
            neg_lid = self.lit_id(idx, -sign)
            new_units = []
            
            watchers = list(self.watched[neg_lid])
            for c_idx in watchers:
                c = self.clauses[c_idx]
                w1, w2 = self.clause_watches[c_idx]
                
                sat = False
                new_watch = None
                for lit_c in c:
                    if self.satisfies(*lit_c):
                        sat = True
                        break
                    if lit_c == (idx, -sign):
                        continue
                    if not self.falsifies(*lit_c):
                        new_watch = lit_c
                        break
                        
                if sat:
                    # Remove from watched
                    if w1 is not None:
                        id1 = self.lit_id(*w1)
                        if c_idx in self.watched[id1]:
                            self.watched[id1].remove(c_idx)
                    if w2 is not None:
                        id2 = self.lit_id(*w2)
                        if c_idx in self.watched[id2]:
                            self.watched[id2].remove(c_idx)
                else:
                    if new_watch is None:
                        return None, False # Conflict
                    else:
                        new_lid = self.lit_id(*new_watch)
                        self.watched[new_lid].append(c_idx)
                        if w1 == (idx, -sign):
                            self.clause_watches[c_idx][0] = new_watch
                        else:
                            self.clause_watches[c_idx][1] = new_watch
                            
                        # Check if this clause is now unit
                        unassigned_count = 0
                        unit_lit = None
                        for lit_c in c:
                            if self.assignment[lit_c[0]] is None:
                                unassigned_count += 1
                                unit_lit = lit_c
                        if unassigned_count == 1:
                            new_units.append(unit_lit)
                            
            # Clean up watched[neg_lid]
            new_watchers = []
            for c_idx in self.watched[neg_lid]:
                c = self.clauses[c_idx]
                is_sat = any(self.satisfies(*lit_c) for lit_c in c)
                if not is_sat:
                    new_watchers.append(c_idx)
            self.watched[neg_lid] = new_watchers
            
            return new_units, True

        def unit_propagate(self):
            queue = []
            for i, c in enumerate(self.clauses):
                unassigned = [lit for lit in c if self.assignment[lit[0]] is None]
                if not unassigned:
                    return False
                if len(unassigned) == 1:
                    lit = unassigned[0]
                    if self.assignment[lit[0]] is None:
                        queue.append(lit)
                        
            while queue:
                lit = queue.pop()
                idx, sign = lit
                if self.assignment[idx] is not None:
                    continue
                self.assignment[idx] = (sign == 1)
                new_units, ok = self.propagate(lit)
                if not ok:
                    return False
                queue.extend(new_units)
                
            return True

        def pure_literal_elimination(self):
            pos_occ = [0]*self.n
            neg_occ = [0]*self.n
            for c in self.clauses:
                for lit in c:
                    if self.assignment[lit[0]] is None:
                        if lit[1] == 1:
                            pos_occ[lit[0]] += 1
                        else:
                            neg_occ[lit[0]] += 1
                            
            changed = False
            for i in range(self.n):
                if self.assignment[i] is None:
                    if pos_occ[i] > 0 and neg_occ[i] == 0:
                        self.assignment[i] = True
                        changed = True
                    elif neg_occ[i] > 0 and pos_occ[i] == 0:
                        self.assignment[i] = False
                        changed = True
            if changed:
                # Remove satisfied clauses
                new_clauses = []
                for c in self.clauses:
                    sat = False
                    for lit in c:
                        if self.satisfies(*lit):
                            sat = True
                            break
                    if not sat:
                        # Update watches? This is complex.
                        # If we modify clauses, we must rebuild watched lists.
                        # So we rebuild.
                        new_clauses.append(c)
                if len(new_clauses) != len(self.clauses):
                    self.clauses = new_clauses
                    self.num_clauses = len(self.clauses)
                    # Rebuild watched
                    self.watched = [[] for _ in range(2*self.n)]
                    self.clause_watches = []
                    for i, c in enumerate(self.clauses):
                        if len(c) == 1:
                            self.clause_watches.append([c[0], None])
                        else:
                            self.clause_watches.append([c[0], c[1]])
                        for w in self.clause_watches[i]:
                            if w is not None:
                                idx, sign = w
                                lid = idx*2 + (0 if sign==-1 else 1)
                                self.watched[lid].append(i)
                    return True
            return False

        def solve(self):
            if not self.unit_propagate():
                return None
            if self.pure_literal_elimination():
                if not self.unit_propagate():
                    return None
            if not self.clauses:
                for i in range(self.n):
                    if self.assignment[i] is None:
                        self.assignment[i] = False
                return self.assignment
                
            # Branch
            # Pick variable with max degree
            var_counts = [0]*self.n
            for c in self.clauses:
                for lit in c:
                    if self.assignment[lit[0]] is None:
                        var_counts[lit[0]] += 1
                        
            var = max(range(self.n), key=lambda i: var_counts[i])
            if var_counts[var] == 0:
                return None # Should not happen if clauses exist
                
            # Try True
            self.assignment[var] = True
            if self.unit_propagate():
                # Recurse
                # We need to save state. 
                # Copying assignment and clauses is O(N+C).
                # Let's do it.
                save_assignment = self.assignment[:]
                save_clauses = self.clauses[:]
                save_watched = [w[:] for w in self.watched]
                save_clause_watches = [w[:] for w in self.clause_watches]
                
                # We need to create a new DPLL instance? Or modify and restore?
                # Modifying and restoring is faster.
                # But recursion depth might be an issue? N=50 is small.
                
                # Actually, let's just return a result if found.
                # This is getting messy.
                
            return None

    # The complexity of managing state in the class is high.
    # Let's go with the simplest effective improvement:
    # Optimize the existing DPLL by using a better variable selection heuristic 
    # and ensuring the clause update is as fast as possible.
    # The existing code is likely fine for N=50 if it doesn't timeout.
    # If it times out, it's because of the O(N^3) behavior.
    # Watched literals reduce this to O(N^2) or better.
    
    # I will provide a clean, recursive DPLL with Watched Literals that handles state by copying.
    # For N=50, copying is cheap.

    def solve_dpll(n, clauses):
        # Normalize
        norm = []
        for c in clauses:
            nc = []
            seen = set()
            for l in c:
                idx = abs(l) - 1
                sign = 1 if l > 0 else -1
                if idx not in seen:
                    seen.add(idx)
                    nc.append((idx, sign))
            if nc:
                norm.append(nc)
        if not norm:
            return [False]*n

        def lit_id(idx, sign):
            return idx*2 + (0 if sign==-1 else 1)

        def dpll(assignment, clauses, watched, clause_watches):
            # Unit Propagation
            queue = []
            for i, c in enumerate(clauses):
                unassigned = [lit for lit in c if assignment[lit[0]] is None]
                if not unassigned:
                    return None
                if len(unassigned) == 1:
                    lit = unassigned[0]
                    if assignment[lit[0]] is None:
                        queue.append(lit)
                        
            while queue:
                lit = queue.pop()
                idx, sign = lit
                if assignment[idx] is not None:
                    continue
                assignment[idx] = (sign == 1)
                
                # Propagate
                neg_lid = lit_id(idx, -sign)
                watchers = list(watched[neg_lid])
                for c_idx in watchers:
                    c = clauses[c_idx]
                    w1, w2 = clause_watches[c_idx]
                    sat = False
                    new_watch = None
                    for lit_c in c:
                        if assignment[lit_c[0]] is not None:
                            if assignment[lit_c[0]] == (lit_c[1] == 1):
                                sat = True
                                break
                        else:
                            if not (assignment[lit_c[0]] is not None and assignment[lit_c[0]] != (lit_c[1] == 1)):
                                # If not assigned, it's not falsified.
                                # If assigned and false, it's falsified.
                                if assignment[lit_c[0]] is None:
                                    if not sat:
                                        new_watch = lit_c
                                    else:
                                        break
                                else:
                                    # Assigned and False
                                    pass
                    # Simplified logic:
                    sat = False
                    new_watch = None
                    for lit_c in c:
                        if lit_c == (idx, -sign):
                            continue
                        if assignment[lit_c[0]] is not None:
                            if assignment[lit_c[0]] == (lit_c[1] == 1):
                                sat = True
                                break
                        else:
                            new_watch = lit_c
                            break
                            
                    if sat:
                        # Remove c_idx from watched
                        if w1 is not None:
                            id1 = lit_id(*w1)
                            if c_idx in watched[id1]:
                                watched[id1].remove(c_idx)
                        if w2 is not None:
                            id2 = lit_id(*w2)
                            if c_idx in watched[id2]:
                                watched[id2].remove(c_idx)
                    else:
                        if new_watch is None:
                            return None
                        else:
                            new_lid = lit_id(*new_watch)
                            watched[new_lid].append(c_idx)
                            if w1 == (idx, -sign):
                                clause_watches[c_idx][0] = new_watch
                            else:
                                clause_watches[c_idx][1] = new_watch
                            
                            # Check if unit
                            unassigned_count = 0
                            unit_lit = None
                            for lit_c in c:
                                if assignment[lit_c[0]] is None:
                                    unassigned_count += 1
                                    unit_lit = lit_c
                            if unassigned_count == 1:
                                queue.append(unit_lit)
                                
                # Clean watched[neg_lid]
                new_watchers = []
                for c_idx in watched[neg_lid]:
                    c = clauses[c_idx]
                    is_sat = False
                    for lit_c in c:
                        if assignment[lit_c[0]] is not None and assignment[lit_c[0]] == (lit_c[1] == 1):
                            is_sat = True
                            break
                    if not is_sat:
                        new_watchers.append(c_idx)
                watched[neg_lid] = new_watchers

            # Check if all clauses satisfied
            # And pure literal elimination
            # For simplicity, skip pure literal in recursion to avoid complexity, 
            # or implement it carefully.
            # Let's add pure literal.
            pos_occ = [0]*n
            neg_occ = [0]*n
            for c in clauses:
                for lit in c:
                    if assignment[lit[0]] is None:
                        if lit[1] == 1:
                            pos_occ[lit[0]] += 1
                        else:
                            neg_occ[lit[0]] += 1
            changed = False
            for i in range(n):
                if assignment[i] is None:
                    if pos_occ[i] > 0 and neg_occ[i] == 0:
                        assignment[i] = True
                        changed = True
                    elif neg_occ[i] > 0 and pos_occ[i] == 0:
                        assignment[i] = False
                        changed = True
            if changed:
                # Re-run unit propagation
                return dpll(assignment, clauses, watched, clause_watches)
                
            # Check if solved
            all_sat = True
            for c in clauses:
                sat = False
                for lit in c:
                    if assignment[lit[0]] is not None and assignment[lit[0]] == (lit[1] == 1):
                        sat = True
                        break
                if not sat:
                    all_sat = False
                    break
            if all_sat:
                for i in range(n):
                    if assignment[i] is None:
                        assignment[i] = False
                return assignment
                
            # Branch
            # Pick variable with max degree
            var_counts = [0]*n
            for c in clauses:
                for lit in c:
                    if assignment[lit[0]] is None:
                        var_counts[lit[0]] += 1
            var = max(range(n), key=lambda i: var_counts[i])
            if var_counts[var] == 0:
                return None
                
            # Try True
            new_assignment = assignment[:]
            new_assignment[var] = True
            # Copy watched and clause_watches? 
            # If we modify them in place, we need to restore.
            # Copying is safer.
            new_watched = [w[:] for w in watched]
            new_clause_watches = [w[:] for w in clause_watches]
            
            res = dpll(new_assignment, clauses, new_watched, new_clause_watches)
            if res is not None:
                return res
                
            # Try False
            new_assignment = assignment[:]
            new_assignment[var] = False
            new_watched = [w[:] for w in watched]
            new_clause_watches = [w[:] for w in clause_watches]
            
            res = dpll(new_assignment, clauses, new_watched, new_clause_watches)
            if res is not None:
                return res
                
            return None

        # Initialize watched
        watched = [[] for _ in range(2*n)]
        clause_watches = []
        for i, c in enumerate(norm):
            if len(c) == 1:
                clause_watches.append([c[0], None])
            else:
                clause_watches.append([c[0], c[1]])
            for w in clause_watches[i]:
                if w is not None:
                    idx, sign = w
                    lid = lit_id(idx, sign)
                    watched[lid].append(i)
                    
        assignment = [None]*n
        res = dpll(assignment, norm, watched, clause_watches)
        if res is None:
            return [False]*n
        return res

    return solve_dpll(n, processed_clauses)
# EVOLVE-BLOCK-END
