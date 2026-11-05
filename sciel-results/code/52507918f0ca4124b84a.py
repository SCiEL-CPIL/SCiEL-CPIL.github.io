"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys

sys.setrecursionlimit(100000)


def solve(instance):
    """
    Solve a planted 3-SAT instance using a DPLL-based solver with
    watched literals, unit clause propagation, and pure literal elimination.
    
    Returns a list of booleans (length n) representing the satisfying assignment,
    or None if UNSAT (though the task guarantees satisfiable instances).
    """
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Normalize: ensure clauses are tuples of ints (signed literals, 1-based)
    clauses = [tuple(c) for c in clauses]
    
    # Data structures for DPLL
    # assignment: array of 0 (False), 1 (True), -1 (Unassigned)
    # We use 1-based indexing for variables in literals, so var_idx = abs(lit)
    # assignment array size n+1, index 0 unused.
    
    def get_lit_idx(lit):
        return lit + n

    num_clauses = len(clauses)
    assignment = [-1] * (n + 1)
    trail = []
    trail_lim = []
    watches = [[] for _ in range(2 * n + 1)]
    watched_lits = []
    to_propagate = []
    
    # Initialize watches
    for i, c in enumerate(clauses):
        if len(c) >= 2:
            w1, w2 = c[0], c[1]
            watched_lits.append([w1, w2])
            watches[get_lit_idx(w1)].append(i)
            watches[get_lit_idx(w2)].append(i)
        elif len(c) == 1:
            # Unit clause
            # Watch the same literal twice to simulate unit behavior in the watcher logic
            # If it becomes false, conflict. If true, satisfied.
            watched_lits.append([c[0], c[0]])
            watches[get_lit_idx(c[0])].append(i)
            watches[get_lit_idx(c[0])].append(i)
        else:
            # Empty clause -> UNSAT
            return None

    def is_false(lit):
        v = abs(lit)
        a = assignment[v]
        if a == -1:
            return False
        if lit > 0:
            return a == 0
        else:
            return a == 1

    def is_true(lit):
        v = abs(lit)
        a = assignment[v]
        if a == -1:
            return False
        if lit > 0:
            return a == 1
        else:
            return a == 0

    def propagate():
        """
        Process the to_propagate queue.
        Returns False if conflict, True if success.
        """
        while to_propagate:
            lit = to_propagate.pop()
            # If lit is already assigned, skip
            if assignment[abs(lit)] != -1:
                continue
                
            # Assign this literal
            var = abs(lit)
            val = 1 if lit > 0 else 0
            assignment[var] = val
            trail.append(var)
            
            # The opposite literal becomes false
            false_lit = -lit
            f_idx = get_lit_idx(false_lit)
            
            # Process clauses watching false_lit
            while watches[f_idx]:
                c_idx = watches[f_idx].pop()
                
                w0, w1 = watched_lits[c_idx]
                
                # Check if clause is satisfied
                if is_true(w0) or is_true(w1):
                    continue
                    
                # Check if both are false (Conflict)
                if is_false(w0) and is_false(w1):
                    return False
                    
                # One is false, one is unassigned
                if is_false(w0):
                    # w0 is false, w1 is unassigned
                    # Move watcher to w1
                    watched_lits[c_idx] = [w1, w0]
                    watches[get_lit_idx(w1)].append(c_idx)
                    
                    # Check if clause is now a unit clause
                    clause = clauses[c_idx]
                    unassigned_count = 0
                    for l in clause:
                        if assignment[abs(l)] == -1:
                            unassigned_count += 1
                            
                    if unassigned_count == 1:
                        to_propagate.append(w1)
                        
                else:
                    # w1 is false, w0 is unassigned
                    watched_lits[c_idx] = [w0, w1]
                    watches[get_lit_idx(w0)].append(c_idx)
                    
                    # Check if clause is now a unit clause
                    clause = clauses[c_idx]
                    unassigned_count = 0
                    for l in clause:
                        if assignment[abs(l)] == -1:
                            unassigned_count += 1
                            
                    if unassigned_count == 1:
                        to_propagate.append(w0)
                        
        return True

    def pick_variable():
        # Degree-based heuristic
        counts = [0] * (n + 1)
        for c in clauses:
            for l in c:
                v = abs(l)
                if assignment[v] == -1:
                    counts[v] += 1
                    
        best_var = -1
        max_count = 0
        for v in range(1, n + 1):
            if assignment[v] == -1 and counts[v] > max_count:
                max_count = counts[v]
                best_var = v
                
        return best_var

    def dpll(level):
        # Propagate
        if not propagate():
            return None
            
        # Pick variable
        var = pick_variable()
        
        if var == -1:
            # All assigned
            # Verify SAT
            for c in clauses:
                sat = False
                for l in c:
                    if is_true(l):
                        sat = True
                        break
                if not sat:
                    return None
            return assignment
            
        # Branch
        trail_lim.append(len(trail))
        
        # Try True
        to_propagate.append(var)
        if dpll(level + 1) is not None:
            return assignment
            
        # Backtrack
        while len(trail) > trail_lim[-1]:
            v = trail.pop()
            assignment[v] = -1
        trail_lim.pop()
        
        # Try False
        to_propagate.append(-var)
        if dpll(level + 1) is not None:
            return assignment
            
        # Backtrack
        while len(trail) > trail_lim[-1]:
            v = trail.pop()
            assignment[v] = -1
        trail_lim.pop()
        
        return None

    # Start solving
    result = dpll(0)
    
    if result is None:
        return [False] * n
        
    return [result[i] == 1 for i in range(1, n + 1)]

# EVOLVE-BLOCK-END
