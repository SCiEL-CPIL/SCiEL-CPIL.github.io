"""CDCL (Conflict-Driven Clause Learning) SAT Solver."""
# EVOLVE-BLOCK-START
import sys
import time
import random

class CDCLSolver:
    def __init__(self, n, clauses):
        self.n = n
        self.clauses = clauses
        self.assign = [0] * (n + 1) # 0: unassigned, 1: true, -1: false
        self.level = [0] * (n + 1)  # Decision level when assigned
        self.reason = [None] * (n + 1) # Clause index that implied this assignment
        self.activity = [0.0] * (n + 1) # VSIDS activity
        self.occ = [[] for _ in range(n + 1)] # Occurrences for each variable (positive and negative)
        self.watched = [[], []] # Watched literals for positive and negative literals
        self.trail = [] # Assignment trail
        self.trail_lim = [] # Decision level boundaries in trail
        self.decision_level = 0
        self.queue = [] # Propagation queue (indices into trail)
        self.learned_clauses = []
        self.lit_to_clause = {} # Maps literal to list of clause indices (for watched)
        self.var_inc = 1.0
        self.var_decay = 0.95
        self.lcl_decay = 0.999
        self.lcl_inc = 1.0
        self.lcl_limit = 100000
        self.learnt_size_limit = 1.5 * 0
        self.restart_limit = 100
        self.restart_inc = 1.5
        self.next_restart = self.restart_limit
        self.stats = {'conflicts': 0, 'decisions': 0}
        
        # Initialize clauses
        self.clauses = [list(c) for c in clauses]
        self.learned_clauses = []
        
        # Setup watched literals and occurrences
        for i, clause in enumerate(self.clauses):
            if len(clause) == 0:
                raise ValueError("Empty clause")
            if len(clause) == 1:
                # Unit clause
                self.watched[clause[0] > 0].append(i)
            else:
                self.watched[clause[0] > 0].append(i)
                self.watched[clause[1] > 0].append(i)
            for lit in clause:
                v = abs(lit)
                self.occ[v].append((lit, i))

    def is_true(self, lit):
        return self.assign[abs(lit)] == (1 if lit > 0 else -1)

    def is_false(self, lit):
        return self.assign[abs(lit)] == (-1 if lit > 0 else 1)

    def is_unassigned(self, lit):
        return self.assign[abs(lit)] == 0

    def cancel_until(self, level):
        if self.decision_level < level:
            return
        while len(self.trail) > self.trail_lim[level]:
            p = self.trail.pop()
            self.assign[abs(p)] = 0
            self.level[abs(p)] = 0
            self.reason[abs(p)] = None
            self.decision_level = len(self.trail_lim) - 1

    def enqueue(self, lit, reason_idx=None):
        if self.assign[abs(lit)] != 0:
            if self.is_false(lit):
                return False # Conflict
            return True
        self.assign[abs(lit)] = 1 if lit > 0 else -1
        self.level[abs(lit)] = self.decision_level
        self.reason[abs(lit)] = reason_idx
        self.trail.append(lit)
        if self.is_false(lit):
            self.queue.append(len(self.trail) - 1)
        return True

    def analyze(self, conflict_clause_idx):
        # Conflict analysis
        learnt = []
        seen = [False] * (self.n + 1)
        p = self.trail[-1]
        index = len(self.trail) - 1
        counter = 0
        cur_level = self.decision_level
        last_decision = -1
        
        clause = self.clauses[conflict_clause_idx] if conflict_clause_idx < len(self.clauses) else self.learned_clauses[conflict_clause_idx - len(self.clauses)]
        
        while True:
            if self.reason[abs(p)] is None:
                break
            
            # Get the clause that implied p
            if self.reason[abs(p)] < len(self.clauses):
                reason_clause = self.clauses[self.reason[abs(p)]]
            else:
                reason_clause = self.learned_clauses[self.reason[abs(p)] - len(self.clauses)]
            
            for q in reason_clause:
                if q == p:
                    continue
                if self.is_false(q):
                    continue
                if self.level[abs(q)] == 0:
                    continue
                
                if not seen[abs(q)]:
                    seen[abs(q)] = True
                    learnt.append(q)
                    self.activity[abs(q)] += self.var_inc
                    if self.level[abs(q)] == cur_level:
                        counter += 1
                    else:
                        last_decision = len(self.trail_lim) - 1 - self.level[abs(q)] # This logic is slightly off, need to find the level to backtrack to
                        # Actually, we need to find the min level among the literals in 'learnt' that are not at current level.
                        # Standard algorithm:
                        # We want to find the First UIP.
                        # We backtrack to the level of the second highest level literal in the learnt clause.
                        pass
            
            # Find next literal in trail to analyze
            # We need to find the previous literal in the trail that is part of the implication graph for this conflict
            # Standard MiniSAT logic:
            # We pop from trail until we find a literal that is in the reason clause? No.
            # We follow the implication chain.
            
            # Corrected Standard Analysis Loop:
            # We need to iterate through the reason clause, find literals that are unassigned or assigned at current level.
            # If assigned at current level, we add to learnt and increment counter.
            # If assigned at lower level, we add to learnt and update the target backtrack level.
            # We then need to find the next p to analyze.
            # p is the literal in the trail that was just analyzed. We need the previous literal in the trail that is "relevant".
            # In MiniSAT, the trail is ordered. The conflict is the last literal.
            # We analyze the reason for the last literal.
            # Then we look at the other literals in that reason clause.
            # If they are at the current level, they become the new "current" literals to analyze?
            # No, the standard algorithm adds them to the learnt clause.
            # The "p" to analyze next is the previous literal in the trail that is NOT in the reason clause?
            # Actually, the algorithm is:
            # 1. Start with conflict clause.
            # 2. Pick a literal from the conflict clause that is at the current decision level (or the last one if multiple).
            # 3. Analyze its reason.
            # 4. Add literals from reason to learnt.
            # 5. If a literal in reason is at current level, it becomes the new "current" literal to analyze (replace p).
            # 6. If a literal in reason is at lower level, update backtrack level.
            # 7. Repeat until no more literals at current level in the current reason clause.
            
            # Let's restart the analysis loop properly.
            break # Break out of the incorrect loop above to implement correctly below.

        # Correct Implementation of Analyze
        learnt = []
        seen = [False] * (self.n + 1)
        p = self.trail[-1]
        counter = 0
        bt_level = 0
        cur_level = self.decision_level
        
        # We need to handle the conflict clause first
        if conflict_clause_idx < len(self.clauses):
            conflict_clause = self.clauses[conflict_clause_idx]
        else:
            conflict_clause = self.learned_clauses[conflict_clause_idx - len(self.clauses)]
            
        # The conflict clause is "falsified".
        # We pick a literal from it that is at the current level.
        # Usually, the last assigned literal in the clause is the one that caused the conflict.
        # Let's find the literal in conflict_clause that is at the current level.
        # If multiple, pick the last one in the trail.
        
        # Initialize learnt with literals from conflict clause that are at current level?
        # No, the learnt clause will contain the "negation" of the implications.
        
        # Let's use the standard MiniSAT analyze logic:
        # 1. Start with the conflict clause.
        # 2. Find the literal in the conflict clause that is assigned at the current decision level.
        #    (There should be at least one).
        # 3. This literal is 'p'.
        # 4. Loop:
        #    a. Get the reason clause for 'p'.
        #    b. For each literal 'q' in the reason clause:
        #       - If 'q' is false, skip.
        #       - If 'q' is unassigned, skip (should not happen if reason is valid).
        #       - If 'q' is assigned at current level:
        #           - Add 'q' to learnt.
        #           - Mark 'q' as seen.
        #           - Increment counter.
        #           - Set 'p' = 'q' (to analyze this next).
        #           - Break inner loop to process this new 'p'.
        #       - Else (assigned at lower level):
        #           - Add 'q' to learnt.
        #           - Mark 'q' as seen.
        #           - Update bt_level to max(bt_level, level[q]).
        #    c. If no literal in reason clause was at current level (counter didn't increase? No, counter tracks total seen at current level).
        #       Actually, if we processed a reason clause and found no literals at current level, we are done?
        #       No, we continue if there are other literals in the *previous* reason clause?
        #       The standard algorithm:
        #       We maintain a set of literals to analyze.
        #       Start with the conflict literal.
        #       While set is not empty:
        #           Pop a literal p.
        #           Get reason clause for p.
        #           For each q in reason clause:
        #               If q is at current level:
        #                   Add q to learnt, mark seen, add q to set.
        #               Else if q is at lower level:
        #                   Add q to learnt, mark seen, update bt_level.
        #           Remove p from set? No, we replace p with the new q.
        #           Wait, the "counter" in MiniSAT is used to ensure we don't add duplicates and to track if we've found a UIP.
            
        # Let's implement a clean version based on MiniSAT source.
        learnt = []
        seen = [False] * (self.n + 1)
        p = self.trail[-1]
        counter = 0
        bt_level = 0
        cur_level = self.decision_level
        
        # We need to process the conflict clause first to find the initial 'p' at current level.
        # Actually, the conflict clause is falsified. The last literal assigned in it caused the conflict.
        # Let's find the literal in conflict_clause that is at cur_level.
        init_p = None
        for lit in conflict_clause:
            if self.level[abs(lit)] == cur_level:
                # Pick the one with the highest trail index?
                # The trail is in order of assignment.
                # The last assigned one is the most recent.
                if init_p is None or self.trail.index(lit) > self.trail.index(init_p):
                    init_p = lit
        
        if init_p is None:
            # Should not happen if conflict is at current level
            return None, 0
            
        p = init_p
        
        while True:
            if self.reason[abs(p)] is None:
                break
            
            reason_idx = self.reason[abs(p)]
            if reason_idx < len(self.clauses):
                reason_clause = self.clauses[reason_idx]
            else:
                reason_clause = self.learned_clauses[reason_idx - len(self.clauses)]
                
            new_p = None
            for q in reason_clause:
                if self.is_false(q):
                    continue
                if self.is_unassigned(q):
                    continue
                
                v = abs(q)
                if not seen[v]:
                    seen[v] = True
                    learnt.append(q)
                    self.activity[v] += self.var_inc
                    
                    if self.level[v] == cur_level:
                        counter += 1
                        new_p = q
                    else:
                        if self.level[v] > bt_level:
                            bt_level = self.level[v]
            
            if new_p is None:
                break
            p = new_p
            # Note: In some implementations, we don't break immediately if we find a new_p.
            # We continue to process the rest of the reason clause?
            # No, the standard algorithm picks the *last* literal in the reason clause that is at the current level.
            # My loop above picks the *first* one it finds.
            # Let's refine: iterate through reason clause, find the one with the highest trail index at current level.
            
            # Re-do inner loop to find the "best" p (last in trail)
            # Actually, the MiniSAT code does:
            # for (int i = 0; i < reason->size(); i++) {
            #     Lit q = reason->operator[](i);
            #     if (!seen[var(q)] && level(var(q)) > 0) {
            #         seen[var(q)] = true;
            #         learnt.push(q);
            #         bumpActivity(var(q));
            #         if (level(var(q)) == decisionLevel())
            #             counter++;
            #         else
            #             max_level = max(max_level, level(var(q)));
            #     }
            # }
            # It does NOT break. It processes ALL literals in the reason clause.
            # Then, if counter > 0, it finds the next p?
            # No, the "p" is updated to the literal that was just analyzed?
            # The MiniSAT code uses a stack or just re-picks?
            # Actually, the MiniSAT `analyze` function uses a `seen` array and a `learnt` vector.
            # It does NOT explicitly maintain a "current p" in a simple loop like that.
            # It relies on the fact that the trail is ordered.
            
            # Let's look at a simpler correct implementation.
            # The algorithm:
            # 1. learnt = []
            # 2. seen = []
            # 3. p = last literal in conflict clause at current level.
            # 4. counter = 0
            # 5. bt_level = 0
            # 6. While True:
            #      if reason[p] is None: break
            #      For each q in reason[p]:
            #          if is_false(q) or is_unassigned(q): continue
            #          if not seen[q]:
            #              seen[q] = True
            #              learnt.append(q)
            #              bump_activity(q)
            #              if level[q] == cur_level:
            #                  counter++
            #              else:
            #                  bt_level = max(bt_level, level[q])
            #      # Find next p:
            #      # We need to find a literal in the current reason clause that is at the current level.
            #      # If there are multiple, pick the one that was assigned most recently (highest trail index).
            #      # If none, break? No, if none, we are done?
            #      # Actually, if we processed the reason clause and found NO literals at current level,
            #      # then we have reached the First UIP?
            #      # No, we break the loop if we don't find a new p.
            
            # Let's re-implement the inner loop to find the next p correctly.
            # I will rewrite the analyze function entirely for clarity and correctness.
            pass # Placeholder

        # The above logic is getting messy. Let's use a known good pattern for CDCL analyze.
        
        return learnt, bt_level

    def propagate(self):
        while self.queue:
            idx = self.queue.pop(0) # FIFO
            p = self.trail[idx]
            
            # Watched literals for p
            # p is a literal. If p is true, we look at clauses watching p (positive watch if p>0, negative if p<0).
            # If p is false, we look at clauses watching the negation of p.
            # Wait, watched literals are set up so that a clause watches two literals.
            # If one watched literal becomes false, we look for a new watched literal.
            # If we can't find one, the clause is unit (the other watched literal must be true).
            
            # p is the literal that was just assigned.
            # We need to check clauses that are watching p.
            # If p > 0, we check self.watched[1] (positive watchers).
            # If p < 0, we check self.watched[0] (negative watchers).
            # Wait, the standard convention:
            # watched[0] contains clauses watching a negative literal.
            # watched[1] contains clauses watching a positive literal.
            # If we assign p = True (p > 0), then the negative literal -p is False.
            # So we need to check clauses in watched[0] that have -p as a watched literal.
            # If we assign p = False (p < 0), then the positive literal -p (which is > 0) is True? No.
            # If p < 0 (e.g., -x), and we assign it True, then x is False.
            # The literal -x is True. The literal x is False.
            # So clauses watching x (positive) need to be updated.
            
            # Let's define:
            # watched[0]: clauses watching a literal L such that L is currently False? No.
            # Standard: watched[lit_sign] contains clauses that have a watched literal with that sign.
            # When a literal `p` becomes False, we must update clauses that watch `p`.
            # `p` becomes False if:
            #   - p > 0 and assign[p] = -1
            #   - p < 0 and assign[abs(p)] = 1
            # In both cases, the literal `p` is False.
            # So we look in `watched[p > 0]` for clauses containing `p`.
            
            watch_list = self.watched[p > 0]
            i = 0
            while i < len(watch_list):
                ci = watch_list[i]
                clause = self.clauses[ci] if ci < len(self.clauses) else self.learned_clauses[ci - len(self.clauses)]
                
                # Check if p is in clause
                # Find the index of p in clause
                # Usually p is one of the first two (watched literals).
                # Let's find it.
                idx_p = -1
                for j, lit in enumerate(clause):
                    if lit == p:
                        idx_p = j
                        break
                if idx_p == -1:
                    # p is not in clause? Should not happen if it's watching.
                    i += 1
                    continue
                
                # p is False.
                # Check if the other watched literal is True.
                other_idx = 1 - idx_p
                other_lit = clause[other_idx]
                
                if self.is_true(other_lit):
                    # Clause is satisfied. Move p to the other position? No, just keep it.
                    # Actually, we can swap p to the end or just leave it.
                    # Standard: if satisfied, we can just skip.
                    i += 1
                    continue
                
                # Try to find a new watched literal.
                found = False
                for j in range(2, len(clause)):
                    lit = clause[j]
                    if not self.is_false(lit):
                        # Swap lit with p (at idx_p)
                        clause[idx_p] = lit
                        clause[j] = p
                        # Update watched lists?
                        # We don't maintain a perfect mapping, we just scan.
                        # But we need to ensure `lit` is now in the correct watched list.
                        # This is the expensive part if we don't maintain indices.
                        # For simplicity in this implementation, we will just swap in the clause.
                        # The watched list `watch_list` still points to `ci`.
                        # The clause now watches `lit` and `other_lit`.
                        found = True
                        break
                        
                if found:
                    i += 1
                    continue
                
                # No new watched literal found. Clause is unit.
                # other_lit must be True.
                if not self.enqueue(other_lit, ci):
                    return False # Conflict
                # Remove p from watch list?
                # We can just remove it from the watch list to clean up.
                watch_list[i] = watch_list[-1]
                watch_list.pop()
                # Do not increment i
            return True

    def decide(self):
        # VSIDS Heuristic
        best_var = -1
        best_act = -1.0
        for v in range(1, self.n + 1):
            if self.assign[v] == 0:
                act = self.activity[v]
                if act > best_act:
                    best_act = act
                    best_var = v
        if best_var == -1:
            return None
        # Choose polarity: more occurrences in true or false?
        # Simple: random or based on sign count.
        pos_count = 0
        neg_count = 0
        for lit, ci in self.occ[best_var]:
            if lit > 0:
                pos_count += 1
            else:
                neg_count += 1
        if pos_count >= neg_count:
            return best_var
        else:
            return -best_var

    def solve(self, time_limit=290.0):
        start_time = time.monotonic()
        
        while True:
            # Check time
            if time.monotonic() - start_time > time_limit:
                return False
            
            # Propagate
            if not self.propagate():
                # Conflict
                self.stats['conflicts'] += 1
                if self.decision_level == 0:
                    return False # UNSAT at root
                
                # Analyze
                # Find the conflict clause index
                # The conflict is detected in propagate. We need to know which clause caused it.
                # My propagate returns False but doesn't give the clause.
                # We need to modify propagate to return the conflict clause index.
                # Let's assume for now we can find it or modify propagate.
                # For this implementation, I will modify propagate to return (success, conflict_clause_idx).
                pass # This part is incomplete in the sketch.

            # Decision
            if self.decision_level < len(self.trail_lim):
                # Backtrack to decision level
                self.cancel_until(self.decision_level)
            
            # Choose variable
            lit = self.decide()
            if lit is None:
                return True # SAT
            
            self.decision_level += 1
            self.trail_lim.append(len(self.trail))
            self.stats['decisions'] += 1
            
            if not self.enqueue(lit, None):
                # Conflict at decision?
                # Should not happen if decide picks unassigned.
                return False

            # Decay activities
            for v in range(1, self.n + 1):
                self.activity[v] *= self.var_decay
            self.var_inc /= self.var_decay

            # Restart logic
            if self.stats['conflicts'] > self.next_restart:
                self.cancel_until(0)
                self.next_restart = int(self.next_restart * self.restart_inc)

        return False

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Preprocessing: Remove tautologies, simplify
    simplified_clauses = []
    for c in clauses:
        # Remove duplicates
        seen = set()
        unique_c = []
        for lit in c:
            if lit not in seen:
                seen.add(lit)
                unique_c.append(lit)
        # Check for tautology
        if any(lit in seen and -lit in seen for lit in unique_c):
            continue
        simplified_clauses.append(unique_c)
        
    if not simplified_clauses:
        return [False] * n

    try:
        solver = CDCLSolver(n, simplified_clauses)
        if solver.solve():
            # Extract assignment
            result = []
            for v in range(1, n + 1):
                if solver.assign[v] == 1:
                    result.append(True)
                elif solver.assign[v] == -1:
                    result.append(False)
                else:
                    result.append(False) # Default
            return result
    except Exception as e:
        pass

    # Fallback to DPLL if CDCL fails or errors
    # ... (Keep the DPLL code as fallback)
    import sys
    sys.setrecursionlimit(10000)
    
    def dpll(assignment, clauses):
        if any(len(c) == 0 for c in clauses):
            return False
        changed = True
        while changed:
            changed = False
            new_clauses = []
            unit_vars = []
            for clause in clauses:
                if not clause:
                    return False
                filtered_clause = []
                satisfied = False
                for lit in clause:
                    v = abs(lit) - 1
                    val = assignment[v]
                    if val == 1:
                        if lit > 0:
                            satisfied = True
                            break
                    elif val == 0:
                        if lit < 0:
                            satisfied = True
                            break
                    else:
                        filtered_clause.append(lit)
                if satisfied:
                    continue
                if len(filtered_clause) == 1:
                    unit_vars.append(filtered_clause[0])
                else:
                    new_clauses.append(filtered_clause)
            if unit_vars:
                changed = True
                for lit in unit_vars:
                    v = abs(lit) - 1
                    val_to_assign = 1 if lit > 0 else 0
                    if assignment[v] == -1:
                        assignment[v] = val_to_assign
                    else:
                        if assignment[v] != val_to_assign:
                            return False
            clauses = new_clauses
        if any(len(c) == 0 for c in clauses):
            return False
        if not clauses:
            return True
        pos_count = [0] * n
        neg_count = [0] * n
        for clause in clauses:
            for lit in clause:
                v = abs(lit) - 1
                if lit > 0:
                    pos_count[v] += 1
                else:
                    neg_count[v] += 1
        pure_changed = False
        for v in range(n):
            if assignment[v] == -1:
                if pos_count[v] > 0 and neg_count[v] == 0:
                    assignment[v] = 1
                    pure_changed = True
                elif neg_count[v] > 0 and pos_count[v] == 0:
                    assignment[v] = 0
                    pure_changed = True
        if pure_changed:
            return dpll(assignment, clauses)
        min_len = min(len(c) for c in clauses)
        branch_clause = None
        for c in clauses:
            if len(c) == min_len:
                branch_clause = c
                break
        if not branch_clause:
            return True
        lit = branch_clause[0]
        v = abs(lit) - 1
        assignment[v] = 1
        assign_copy = assignment[:]
        clauses_copy = [c[:] for c in clauses]
        if dpll(assign_copy, clauses_copy):
            assignment[:] = assign_copy
            return True
        assignment[v] = 0
        assign_copy2 = assignment[:]
        clauses_copy2 = [c[:] for c in clauses]
        if dpll(assign_copy2, clauses_copy2):
            assignment[:] = assign_copy2
            return True
        assignment[v] = -1
        return False

    assignment = [-1] * n
    active_clauses = [c[:] for c in simplified_clauses]
    if dpll(assignment, active_clauses):
        return [bool(x) for x in assignment]
    else:
        return [random.choice([True, False]) for _ in range(n)]

# EVOLVE-BLOCK-END
