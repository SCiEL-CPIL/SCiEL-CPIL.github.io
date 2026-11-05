# EVOLVE-BLOCK-START
import sys

sys.setrecursionlimit(100000)


def solve(instance):
    """
    Solve a planted 3-SAT instance using a DPLL-based solver with
    watched literals and unit clause propagation.
    
    Returns a list of booleans (length n) representing the satisfying assignment,
    or None if UNSAT (though the task guarantees satisfiable instances).
    """
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Pre-process clauses
    clauses = [list(c) for c in clauses]
    
    # Internal representation:
    # assignment: list of -1 (unassigned), 0 (False), 1 (True)
    # trail: list of (var, value)
    
    # Watched literals:
    # watch[lit + n] = list of clause indices
    # w1[i], w2[i] = watched literals for clause i
    
    num_lits = 2 * n
    
    def init_state(clauses):
        """Initialize solver state for a set of clauses."""
        assignment = [-1] * n
        trail = []
        queue = []
        
        watch = [[] for _ in range(num_lits)]
        w1 = [0] * len(clauses)
        w2 = [0] * len(clauses)
        
        # We need to store the current literals of each clause.
        # As we propagate, we remove assigned literals from clause_lits?
        # Or we just check assignment?
        # Standard watched literals: we keep the list of literals, but we only watch two.
        # When we need a new watch, we scan the clause for unassigned literals.
        # So we need `clause_lits` to remain the original set? 
        # Yes, because we might backtrack.
        # So `clause_lits` is constant.
        
        for i, c in enumerate(clauses):
            if len(c) == 0:
                return None, None, None, None, None, None # UNSAT marker
            if len(c) == 1:
                w1[i] = c[0]
                w2[i] = 0
                watch[c[0] + n].append(i)
            else:
                w1[i] = c[0]
                w2[i] = c[1]
                watch[c[0] + n].append(i)
                watch[c[1] + n].append(i)
                
        return assignment, trail, queue, watch, w1, w2

    # Initial state
    assignment, trail, queue, watch, w1, w2 = init_state(clauses)
    
    if assignment is None:
        return None
        
    def is_sat(lit, assignment):
        var = abs(lit) - 1
        if lit > 0:
            return assignment[var] == 1
        else:
            return assignment[var] == 0
            
    def is_false(lit, assignment):
        var = abs(lit) - 1
        if lit > 0:
            return assignment[var] == 0
        else:
            return assignment[var] == 1

    def propagate(queue, assignment, watch, w1, w2, clauses):
        """
        Process the queue of literals that have been assigned False.
        Modifies assignment, watch, w1, w2 in place.
        Returns True if OK, False if UNSAT.
        """
        while queue:
            lit = queue.pop()
            watchers = watch[lit + n][:]
            for i in watchers:
                if w1[i] == lit:
                    other = w2[i]
                else:
                    other = w1[i]
                
                if other == 0:
                    return False
                
                if is_sat(other, assignment):
                    try:
                        watch[lit + n].remove(i)
                    except ValueError:
                        pass
                    continue
                
                if assignment[abs(other)-1] == -1:
                    found = False
                    for l in clauses[i]:
                        if l == lit or l == other:
                            continue
                        if assignment[abs(l)-1] == -1:
                            try:
                                watch[lit + n].remove(i)
                            except ValueError:
                                pass
                            watch[l + n].append(i)
                            if w1[i] == lit:
                                w1[i] = l
                            else:
                                w2[i] = l
                            found = True
                            break
                    if not found:
                        # Unit clause on 'other'
                        var_other = abs(other) - 1
                        assignment[var_other] = 1
                        trail.append((var_other, 1))
                        # 'other' is now True. 
                        # The clause is satisfied.
                        try:
                            watch[lit + n].remove(i)
                        except ValueError:
                            pass
                        try:
                            watch[other + n].remove(i)
                        except ValueError:
                            pass
                        # Mark as satisfied? 
                        # We can set w1[i] = 0 to indicate satisfied, but we removed from watches.
                        # Let's just continue.
                        continue
                else:
                    # other is False
                    found = False
                    for l in clauses[i]:
                        if l == lit or l == other:
                            continue
                        if assignment[abs(l)-1] == -1:
                            try:
                                watch[lit + n].remove(i)
                            except ValueError:
                                pass
                            watch[l + n].append(i)
                            if w1[i] == lit:
                                w1[i] = l
                            else:
                                w2[i] = l
                            found = True
                            break
                    if not found:
                        return False
        return True

    def assign(var, val, assignment, trail, queue, watch, w1, w2, clauses):
        """
        Assign var to val.
        Returns True if OK, False if UNSAT.
        """
        if assignment[var] != -1:
            if assignment[var] != val:
                return False
            return True
        
        assignment[var] = val
        trail.append((var, val))
        
        if val == 1:
            false_lit = -(var + 1)
        else:
            false_lit = var + 1
            
        queue.append(false_lit)
        
        return propagate(queue, assignment, watch, w1, w2, clauses)

    # Initial Propagation
    for i, c in enumerate(clauses):
        unassigned = [l for l in c if assignment[abs(l)-1] == -1]
        if len(unassigned) == 0:
            return None
        elif len(unassigned) == 1:
            lit = unassigned[0]
            var = abs(lit) - 1
            val = 1 if lit > 0 else 0
            if not assign(var, val, assignment, trail, queue, watch, w1, w2, clauses):
                return None

    def dpll(assignment, trail, queue, watch, w1, w2, clauses):
        # Check if all variables assigned
        var = -1
        for i in range(n):
            if assignment[i] == -1:
                var = i
                break
        
        if var == -1:
            return assignment # SAT
        
        # Copy state for branching
        # We need to deep copy watch, w1, w2, clause_lits? 
        # clause_lits is constant.
        # watch is a list of lists.
        # w1, w2 are lists.
        # assignment is a list.
        # trail is a list.
        # queue is a list.
        
        # Since we modify watch/w1/w2 in place during propagation, we must copy them.
        
        # Try True
        assignment_t = assignment[:]
        trail_t = trail[:]
        queue_t = queue[:]
        watch_t = [lst[:] for lst in watch]
        w1_t = w1[:]
        w2_t = w2[:]
        
        if assign(var, 1, assignment_t, trail_t, queue_t, watch_t, w1_t, w2_t, clauses):
            result = dpll(assignment_t, trail_t, queue_t, watch_t, w1_t, w2_t, clauses)
            if result is not None:
                return result
        
        # Try False
        assignment_f = assignment[:]
        trail_f = trail[:]
        queue_f = queue[:]
        watch_f = [lst[:] for lst in watch]
        w1_f = w1[:]
        w2_f = w2[:]
        
        if assign(var, 0, assignment_f, trail_f, queue_f, watch_f, w1_f, w2_f, clauses):
            result = dpll(assignment_f, trail_f, queue_f, watch_f, w1_f, w2_f, clauses)
            if result is not None:
                return result
        
        return None

    result = dpll(assignment, trail, queue, watch, w1, w2, clauses)
    
    if result is None:
        return [False] * n
        
    return [result[i] == 1 for i in range(n)]

# EVOLVE-BLOCK-END
