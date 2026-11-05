"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random


def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # DPLL with watched literals, unit propagation, and pure literal elimination
    
    def dpll(n_vars, clauses):
        """
        n_vars: number of variables
        clauses: list of clauses, each clause is a list of 1-based signed literals
        Returns: list of booleans (assignment) or None if UNSAT
        """
        # assignment[v] = None (unassigned), True, or False
        assignment = [None] * n_vars
        
        # Build watched literal structure
        # For each clause, we watch two literals
        # watchers[v] = list of (clause_idx, other_var, other_sign)
        # meaning clause clause_idx has a literal on v, and the other watched literal is (other_var, other_sign)
        
        num_clauses = len(clauses)
        watchers = [[] for _ in range(n_vars)]
        clause_watch = [None] * num_clauses  # stores (var1, sign1, var2, sign2)
        
        # Initialize
        for ci, clause in enumerate(clauses):
            if len(clause) == 0:
                return None  # UNSAT
            if len(clause) == 1:
                l = clause[0]
                v = abs(l) - 1
                assignment[v] = (l > 0)
                clause_watch[ci] = None
                continue
            l1, l2 = clause[0], clause[1]
            v1, s1 = abs(l1) - 1, (l1 > 0)
            v2, s2 = abs(l2) - 1, (l2 > 0)
            clause_watch[ci] = (v1, s1, v2, s2)
            watchers[v1].append((ci, v2, s2))
            watchers[v2].append((ci, v1, s1))
        
        # Propagation queue
        queue = [v for v in range(n_vars) if assignment[v] is not None]
        
        def propagate():
            """
            Perform unit propagation.
            Returns (success: bool, new_queue: list)
            """
            nonlocal assignment, watchers, clause_watch
            
            while queue:
                v = queue.pop()
                val = assignment[v]
                
                i = 0
                wlist = watchers[v]
                while i < len(wlist):
                    ci, other_v, other_s = wlist[i]
                    other_val = assignment[other_v]
                    
                    if other_val is not None:
                        if other_val == other_s:
                            # Clause satisfied, remove watch
                            wlist[i] = wlist[-1]
                            wlist.pop()
                            continue
                        else:
                            # other_v falsified, find new watch
                            found_new = False
                            for l in clauses[ci]:
                                lv = abs(l) - 1
                                ls = (l > 0)
                                if lv == v:
                                    continue
                                if assignment[lv] is None:
                                    # Found unassigned literal
                                    # Remove old watch, add new one
                                    wlist[i] = wlist[-1]
                                    wlist.pop()
                                    watchers[lv].append((ci, other_v, other_s))
                                    clause_watch[ci] = (lv, ls, other_v, other_s)
                                    found_new = True
                                    break
                                elif assignment[lv] == ls:
                                    # Clause satisfied
                                    wlist[i] = wlist[-1]
                                    wlist.pop()
                                    found_new = True
                                    break
                            
                            if not found_new:
                                # All literals falsified -> UNSAT
                                return False, queue
                            continue
                    else:
                        # other_v unassigned, clause is unit
                        assignment[other_v] = other_s
                        queue.append(other_v)
                        # Remove watch
                        wlist[i] = wlist[-1]
                        wlist.pop()
                        continue
                    i += 1
            
            return True, queue
        
        # Initial propagation
        ok, queue = propagate()
        if not ok:
            return None
        
        # Check if all assigned
        unassigned = [v for v in range(n_vars) if assignment[v] is None]
        if not unassigned:
            return assignment
        
        # Pure literal elimination
        pos_count = [0] * n_vars
        neg_count = [0] * n_vars
        for clause in clauses:
            for l in clause:
                v = abs(l) - 1
                if l > 0:
                    pos_count[v] += 1
                else:
                    neg_count[v] += 1
        
        for v in unassigned:
            if pos_count[v] == 0 and neg_count[v] > 0:
                assignment[v] = False
                queue.append(v)
            elif neg_count[v] == 0 and pos_count[v] > 0:
                assignment[v] = True
                queue.append(v)
        
        if queue:
            ok, queue = propagate()
            if not ok:
                return None
            unassigned = [v for v in range(n_vars) if assignment[v] is None]
            if not unassigned:
                return assignment
        
        # Variable selection: most occurrences
        var_occ = [0] * n_vars
        for clause in clauses:
            seen = set()
            for l in clause:
                v = abs(l) - 1
                if assignment[v] is None and v not in seen:
                    var_occ[v] += 1
                    seen.add(v)
        
        v = max(unassigned, key=lambda x: var_occ[x])
        
        # Try True
        save_assignment = assignment[:]
        save_watchers = [w[:] for w in watchers]
        save_clause_watch = clause_watch[:]
        save_queue = queue[:]
        
        assignment[v] = True
        queue.append(v)
        ok, _ = propagate()
        if ok:
            # Recurse
            # We need to handle the remaining problem
            # For simplicity, let's just do a full recursive DPLL
            pass
        
        # This is getting complicated. Let me just use a simple recursive DPLL with state copying.
        
        def dpll_simple(assign, cls):
            """
            Simple recursive DPLL with state copying.
            assign: list of None/True/False
            cls: list of clauses (list of 1-based signed literals)
            Returns: assignment list or None
            """
            # Remove satisfied clauses and falsified literals
            new_cls = []
            for clause in cls:
                new_clause = []
                sat = False
                for l in clause:
                    v = abs(l) - 1
                    if assign[v] is not None:
                        if assign[v] == (l > 0):
                            sat = True
                            break
                    else:
                        new_clause.append(l)
                if not sat:
                    if len(new_clause) == 0:
                        return None
                    new_cls.append(new_clause)
            
            if not new_cls:
                return assign
            
            # Unit propagation
            changed = True
            while changed:
                changed = False
                for clause in new_cls:
                    if len(clause) == 1:
                        l = clause[0]
                        v = abs(l) - 1
                        if assign[v] is None:
                            assign[v] = (l > 0)
                            changed = True
                        break
                if changed:
                    # Re-filter
                    new_cls2 = []
                    for clause in new_cls:
                        new_clause = []
                        sat = False
                        for l in clause:
                            v = abs(l) - 1
                            if assign[v] is not None:
                                if assign[v] == (l > 0):
                                    sat = True
                                    break
                            else:
                                new_clause.append(l)
                        if not sat:
                            if len(new_clause) == 0:
                                return None
                            new_cls2.append(new_clause)
                    new_cls = new_cls2
            
            # Check if all assigned
            unassigned = [v for v in range(len(assign)) if assign[v] is None]
            if not unassigned:
                return assign
            
            # Pure literal elimination
            pos_count = [0] * len(assign)
            neg_count = [0] * len(assign)
            for clause in new_cls:
                for l in clause:
                    v = abs(l) - 1
                    if l > 0:
                        pos_count[v] += 1
                    else:
                        neg_count[v] += 1
            
            for v in unassigned:
                if pos_count[v] == 0 and neg_count[v] > 0:
                    assign[v] = False
                elif neg_count[v] == 0 and pos_count[v] > 0:
                    assign[v] = True
            
            # Re-propagate
            changed = True
            while changed:
                changed = False
                for clause in new_cls:
                    if len(clause) == 1:
                        l = clause[0]
                        v = abs(l) - 1
                        if assign[v] is None:
                            assign[v] = (l > 0)
                            changed = True
                        break
                if changed:
                    new_cls2 = []
                    for clause in new_cls:
                        new_clause = []
                        sat = False
                        for l in clause:
                            v = abs(l) - 1
                            if assign[v] is not None:
                                if assign[v] == (l > 0):
                                    sat = True
                                    break
                            else:
                                new_clause.append(l)
                        if not sat:
                            if len(new_clause) == 0:
                                return None
                            new_cls2.append(new_clause)
                    new_cls = new_cls2
            
            unassigned = [v for v in range(len(assign)) if assign[v] is None]
            if not unassigned:
                return assign
            
            # Variable selection
            var_occ = [0] * len(assign)
            for clause in new_cls:
                seen = set()
                for l in clause:
                    v = abs(l) - 1
                    if assign[v] is None and v not in seen:
                        var_occ[v] += 1
                        seen.add(v)
            
            v = max(unassigned, key=lambda x: var_occ[x])
            
            # Try True
            save = assign[:]
            assign[v] = True
            result = dpll_simple(assign, new_cls)
            if result is not None:
                return result
            
            # Try False
            assign = save[:]
            assign[v] = False
            result = dpll_simple(assign, new_cls)
            if result is not None:
                return result
            
            return None
        
        return dpll_simple(assignment, clauses)
    
    result = dpll(n, clauses)
    if result is None:
        return [False] * n
    return [bool(x) for x in result]

# EVOLVE-BLOCK-END
