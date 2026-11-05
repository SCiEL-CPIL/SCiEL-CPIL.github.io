"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from collections import defaultdict, deque

sys.setrecursionlimit(100000)


def solve(instance):
    """
    Solve a planted 3-SAT instance using an efficient DPLL solver with
    watched literals, unit clause propagation, and a frequency-based heuristic.
    
    Returns a list of booleans (length n) representing the satisfying assignment,
    or None if UNSAT (though the task guarantees satisfiable instances).
    """
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Normalize clauses to lists of ints
    clauses = [list(c) for c in clauses]
    
    # Remove tautologies and duplicate literals within clauses
    cleaned_clauses = []
    for c in clauses:
        # Remove duplicates
        seen = set()
        new_c = []
        for lit in c:
            if lit not in seen:
                seen.add(lit)
                new_c.append(lit)
        # Check for tautology (x and -x)
        is_taut = False
        for lit in new_c:
            if -lit in seen:
                is_taut = True
                break
        if not is_taut and new_c:
            cleaned_clauses.append(new_c)
        
    clauses = cleaned_clauses
    m = len(clauses)
    
    # If no clauses, any assignment works
    if not clauses:
        return [False] * n

    # Watched Literals Setup
    # watches[lit] = list of clause indices that watch lit
    # lit is 1-based signed. We map to 0-based index for array access?
    # Let's use a dictionary or a large list. 
    # Range of lit: -n to n. 
    # We can map lit to index: lit + n. Size 2n+1.
    max_lit = n
    watch_size = 2 * max_lit + 1
    watches = [set() for _ in range(watch_size)]
    
    def lit_to_idx(lit):
        return lit + max_lit

    # Initialize watches: for each clause, pick first two literals
    for i, c in enumerate(clauses):
        if len(c) >= 2:
            w1, w2 = c[0], c[1]
            watches[lit_to_idx(w1)].add(i)
            watches[lit_to_idx(w2)].add(i)
        elif len(c) == 1:
            w1 = c[0]
            watches[lit_to_idx(w1)].add(i)
            # A unit clause needs immediate propagation or special handling.
            # We'll handle unit clauses in the initial propagation.

    # Assignment state
    # -1: unassigned, 0: False, 1: True
    assignment = [-1] * n
    
    # Trail for backtracking
    trail = []
    
    # Queue for unit propagation
    prop_queue = deque()
    
    # Heuristic: Count variable frequency in unsatisfied clauses
    # We'll compute this on the fly or maintain it.
    # For n=50, computing frequency each time is cheap.
    
    def is_satisfied(clause_idx):
        c = clauses[clause_idx]
        for lit in c:
            var = abs(lit) - 1
            val = assignment[var]
            if val == 1 and lit > 0:
                return True
            if val == 0 and lit < 0:
                return True
        return False

    def is_false(clause_idx):
        c = clauses[clause_idx]
        for lit in c:
            var = abs(lit) - 1
            val = assignment[var]
            if val == -1:
                return False
            if val == 1 and lit > 0:
                return False
            if val == 0 and lit < 0:
                return False
        return True

    def propagate():
        """
        Process the propagation queue.
        Returns False if UNSAT detected, True otherwise.
        """
        while prop_queue:
            lit = prop_queue.popleft()
            var = abs(lit) - 1
            # The literal is assigned to True. So its negation is False.
            # We need to process clauses watching the *negation* of the assigned literal?
            # No, standard watched literal logic:
            # When we assign `lit = True`, the literal `-lit` becomes False.
            # Clauses watching `-lit` might be affected.
            
            neg_lit = -lit
            neg_idx = lit_to_idx(neg_lit)
            
            # Iterate over a copy of the set because we modify it
            for c_idx in list(watches[neg_idx]):
                c = clauses[c_idx]
                
                # Check if the clause is satisfied
                sat = False
                for l in c:
                    v = abs(l) - 1
                    a = assignment[v]
                    if a == 1 and l > 0:
                        sat = True
                        break
                    if a == 0 and l < 0:
                        sat = True
                        break
                if sat:
                    continue
                
                # Check if the clause is false (all literals assigned False)
                # If all literals are assigned False, it's UNSAT.
                # In watched literal scheme, if the watched literal becomes False,
                # we look for a new watch. If no new watch found, the clause is unit or false.
                
                # Find the other watched literal in this clause
                # We don't store which literals are watched explicitly per clause in this simple version,
                # we just know c_idx is in watches[neg_idx].
                # We need to find the other literal in `c` that is not `neg_lit`.
                other_lit = None
                for l in c:
                    if l != neg_lit:
                        other_lit = l
                        break
                if other_lit is None:
                    # This shouldn't happen if we initialized correctly, 
                    # unless the clause had duplicate literals which we removed.
                    # If clause length 1, other_lit is None? No, if len 1, it was neg_lit.
                    # If len 1 and we assigned it False, it's UNSAT.
                    if len(c) == 1:
                        return False
                    # If len > 1, this case implies error in logic or initialization.
                    
                # Check status of other_lit
                o_var = abs(other_lit) - 1
                o_val = assignment[o_var]
                
                if o_val == 1 and other_lit > 0:
                    # Other watch is True -> Clause satisfied
                    continue
                if o_val == 0 and other_lit < 0:
                    # Other watch is False
                    # Find a new watch
                    new_watch = None
                    for l in c:
                        if l == neg_lit or l == other_lit:
                            continue
                        v = abs(l) - 1
                        if assignment[v] == -1:
                            new_watch = l
                            break
                    
                    if new_watch is None:
                        # No unassigned literal left in clause.
                        # Since neg_lit is False and other_lit is False, and no other unassigned,
                        # the clause is UNSAT.
                        return False
                    else:
                        # Replace neg_lit watch with new_watch
                        watches[neg_idx].discard(c_idx)
                        watches[lit_to_idx(new_watch)].add(c_idx)
                        continue
                else:
                    # other_lit is unassigned
                    # This is a unit clause!
                    # Assign other_lit to True.
                    assignment[o_var] = 1
                    trail.append((o_var, 1))
                    prop_queue.append(other_lit)
                    # The clause is now satisfied by other_lit, so we don't need to update watches
                    # for this specific implication step, but the watch on neg_lit is still there?
                    # Actually, since other_lit is now True, the clause is satisfied.
                    # We can leave the watches as is, or clean up.
                    # Leaving them is fine, is_satisfied check will pass.
                    # But we should remove c_idx from watches[neg_idx] to avoid processing it again?
                    # No, if we remove it, and later backtrack, we lose the watch.
                    # Standard practice: if clause becomes satisfied, we can leave it in the watch list.
                    # It will be ignored in future propagations because is_satisfied check passes.
                    # However, to keep watch lists small, we might want to move it.
                    # For now, leave it.
                    pass
        
        return True

    def pick_variable():
        """
        Pick a variable to branch on.
        Heuristic: Most frequent variable in unsatisfied, non-unit clauses.
        """
        freq = [0] * n
        for c_idx in range(len(clauses)):
            # Only consider unsatisfied clauses
            if is_satisfied(c_idx):
                continue
            # Check if unit
            unassigned = []
            for lit in clauses[c_idx]:
                v = abs(lit) - 1
                if assignment[v] == -1:
                    unassigned.append(v)
            if len(unassigned) == 0:
                # Should have been caught by propagation
                continue
            if len(unassigned) == 1:
                # Unit clause, should have been propagated
                continue
            # Count frequency of variables in this clause
            seen_vars = set()
            for lit in clauses[c_idx]:
                v = abs(lit) - 1
                if assignment[v] == -1:
                    if v not in seen_vars:
                        freq[v] += 1
                        seen_vars.add(v)
        
        max_freq = -1
        best_var = -1
        for v in range(n):
            if assignment[v] == -1 and freq[v] > max_freq:
                max_freq = freq[v]
                best_var = v
        
        if best_var == -1:
            # Fallback: first unassigned
            for v in range(n):
                if assignment[v] == -1:
                    return v
            return None
        return best_var

    def dpll():
        # 1. Propagate
        if not propagate():
            return None
        
        # 2. Check if all clauses satisfied
        all_sat = True
        for c_idx in range(len(clauses)):
            if not is_satisfied(c_idx):
                # Check if it's a unit clause that wasn't propagated?
                # propagate() should have handled units.
                # If it's not satisfied and not unit, we need to branch.
                # If it's unit, propagate should have queued it.
                # If it's empty, propagate should have returned False.
                all_sat = False
                break
        
        if all_sat:
            # Check if any clause is false? 
            # If all_sat is True, no clause is false.
            return assignment[:]

        # 3. Pick variable
        var = pick_variable()
        if var is None:
            # No unassigned variables, but not all satisfied?
            # This implies UNSAT.
            return None

        # 4. Branch on True
        assignment[var] = 1
        trail.append((var, 1))
        result = dpll()
        if result is not None:
            return result
        
        # 5. Backtrack
        while trail and trail[-1][0] != var:
            v, val = trail.pop()
            assignment[v] = -1
        if trail:
            v, val = trail.pop()
            assignment[v] = -1

        # 6. Branch on False
        assignment[var] = 0
        trail.append((var, 0))
        result = dpll()
        if result is not None:
            return result
        
        # 7. Backtrack
        while trail and trail[-1][0] != var:
            v, val = trail.pop()
            assignment[v] = -1
        if trail:
            v, val = trail.pop()
            assignment[v] = -1
            
        return None

    # Initialize with any unit clauses in the original set
    for c in clauses:
        if len(c) == 1:
            lit = c[0]
            var = abs(lit) - 1
            # Assign it
            if assignment[var] == -1:
                assignment[var] = 1
                trail.append((var, 1))
                prop_queue.append(lit)
            elif assignment[var] == 0 and lit > 0:
                return [False] * n # Conflict
            elif assignment[var] == 1 and lit < 0:
                return [False] * n # Conflict

    result = dpll()
    
    if result is None:
        return [False] * n
        
    return [bool(x) for x in result]

# EVOLVE-BLOCK-END
