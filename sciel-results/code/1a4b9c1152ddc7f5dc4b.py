# EVOLVE-BLOCK-START
import time
import sys

sys.setrecursionlimit(10000)

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 290.0
    
    def simplify(clauses, assignment):
        """
        Simplifies clauses given assignment.
        Returns (new_clauses, unit_lits) or (None, None) on conflict.
        """
        new_clauses = []
        unit_lits = []
        for clause in clauses:
            sat = False
            simplified = []
            for lit in clause:
                var = abs(lit)
                if var in assignment:
                    val = assignment[var]
                    if (lit > 0) == val:
                        sat = True
                        break
                    # else false, drop
                else:
                    simplified.append(lit)
            
            if sat:
                continue
            if len(simplified) == 0:
                return None, None
            if len(simplified) == 1:
                unit_lits.append(simplified[0])
            else:
                new_clauses.append(simplified)
        return new_clauses, unit_lits

    def dpll(clauses, assignment):
        if time.time() > deadline:
            return None

        # 1. Simplify and Unit Propagate
        new_clauses, units = simplify(clauses, assignment)
        if new_clauses is None:
            return None

        while units:
            lit = units.pop()
            var = abs(lit)
            val = (lit > 0)
            
            if var in assignment:
                if assignment[var] != val:
                    return None
                continue
            
            assignment[var] = val
            new_clauses, units = simplify(new_clauses, assignment)
            if new_clauses is None:
                return None

        if not new_clauses:
            for v in range(1, n + 1):
                if v not in assignment:
                    assignment[v] = True
            return assignment

        # 2. Pure Literal Elimination
        while True:
            if time.time() > deadline:
                return None
                
            pos_count = [0] * (n + 1)
            neg_count = [0] * (n + 1)
            for clause in new_clauses:
                for lit in clause:
                    var = abs(lit)
                    if lit > 0:
                        pos_count[var] += 1
                    else:
                        neg_count[var] += 1
            
            pure_var = -1
            pure_val = None
            for var in range(1, n + 1):
                if var in assignment:
                    continue
                if pos_count[var] == 0 and neg_count[var] > 0:
                    pure_var = var
                    pure_val = False
                    break
                elif neg_count[var] == 0 and pos_count[var] > 0:
                    pure_var = var
                    pure_val = True
                    break
            
            if pure_var == -1:
                break
            
            assignment[pure_var] = pure_val
            new_clauses, units = simplify(new_clauses, assignment)
            if new_clauses is None:
                return None
            
            # Unit propagation after pure literal
            while units:
                lit = units.pop()
                var = abs(lit)
                val = (lit > 0)
                if var in assignment:
                    if assignment[var] != val:
                        return None
                    continue
                assignment[var] = val
                new_clauses, units = simplify(new_clauses, assignment)
                if new_clauses is None:
                    return None
            
            if not new_clauses:
                for v in range(1, n + 1):
                    if v not in assignment:
                        assignment[v] = True
                return assignment

        # 3. Branching
        # Pick variable from shortest clause
        min_len = len(new_clauses[0])
        best_clause = new_clauses[0]
        for c in new_clauses[1:]:
            if len(c) < min_len:
                min_len = len(c)
                best_clause = c
                if min_len == 2:
                    break

        lit = best_clause[0]
        var = abs(lit)
        
        # Try True
        assignment[var] = True
        res = dpll(new_clauses, assignment)
        if res is not None:
            return res
        del assignment[var]
        
        # Try False
        assignment[var] = False
        res = dpll(new_clauses, assignment)
        if res is not None:
            return res
        del assignment[var]
        
        return None

    assignment = {}
    result = dpll(list(clauses), assignment)
    
    if result is not None:
        return [result.get(v, False) for v in range(1, n + 1)]
    
    # Fallback
    import random
    rng = random.Random(42)
    return [rng.random() < 0.5 for _ in range(n)]

# EVOLVE-BLOCK-END
