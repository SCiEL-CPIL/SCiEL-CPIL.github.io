import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin for 120s limit

    def make_paley(q):
        if q % 4 != 1:
            return None
        squares = {x * x % q for x in range(1, q)}
        edges = []
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in squares:
                    edges.append([i, j])
        return {'n': q, 'edges': edges}

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def find_triangles(n, adj):
        triangles = []
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    common = adj[i] & adj[j]
                    for k in common:
                        if k > j:
                            triangles.append((i, j, k))
        return triangles

    def find_indep_sets_5(n, adj):
        indep_sets = []
        
        def backtrack(start, current):
            if len(current) == 5:
                indep_sets.append(tuple(sorted(current)))
                return
            # Pruning: if not enough vertices left
            if n - start < 5 - len(current):
                return
            for v in range(start, n):
                if all(v not in adj[u] for u in current):
                    current.append(v)
                    backtrack(v + 1, current)
                    current.pop()
                    # Limit to avoid explosion, though for n<30 it's usually fine
                    if len(indep_sets) > 5000:
                        return

        backtrack(0, [])
        return indep_sets

    def is_valid(g):
        n = g['n']
        edges = g['edges']
        adj = build_adj(n, edges)
        
        # Efficient K4 check
        for i in range(n):
            for j in range(i+1, n):
                if j in adj[i]:
                    for k in range(j+1, n):
                        if k in adj[i] and k in adj[j]:
                            common = adj[i] & adj[j] & adj[k]
                            if common:
                                return False
                                
        # Check if there's an independent set of size 6
        def has_indep_6(start, current):
            if len(current) == 6:
                return True
            if start + (6 - len(current)) > n:
                return False
            
            for v in range(start, n):
                if all(v not in adj[u] for u in current):
                    if has_indep_6(v + 1, current + [v]):
                        return True
            return False
        
        if has_indep_6(0, []):
            return False
            
        return True

    def dpll(clauses, num_vars):
        """
        Simple DPLL SAT solver.
        clauses: list of lists of (var, sign). sign=1 for positive, -1 for negative.
        Returns assignment list or None.
        """
        # Convert to internal representation: list of sets of integers (var*2 + (1 if pos else 0))
        # Actually, let's stick to (var, sign) tuples for clarity in this context, 
        # but optimize the checking.
        
        def propagate(assignment, clauses):
            """
            Unit propagation and pure literal elimination.
            Returns (assignment, clauses, contradiction)
            """
            # Make a copy of clauses
            remaining_clauses = [list(c) for c in clauses]
            
            changed = True
            while changed:
                changed = False
                # Unit Propagation
                i = 0
                while i < len(remaining_clauses):
                    clause = remaining_clauses[i]
                    # Remove literals that are satisfied
                    new_clause = []
                    satisfied = False
                    for var, sign in clause:
                        if var in assignment:
                            val = assignment[var]
                            if (sign == 1 and val) or (sign == -1 and not val):
                                satisfied = True
                                break
                        else:
                            new_clause.append((var, sign))
                    
                    if satisfied:
                        remaining_clauses.pop(i)
                        changed = True
                        continue
                    
                    if len(new_clause) == 0:
                        return assignment, remaining_clauses, True # Contradiction
                    elif len(new_clause) == 1:
                        var, sign = new_clause[0]
                        assignment[var] = (sign == 1)
                        remaining_clauses[i] = new_clause
                        changed = True
                    else:
                        remaining_clauses[i] = new_clause
                        i += 1
                
                # Pure Literal Elimination (Optional but helps)
                # Find literals that appear only in one polarity
                pos_vars = set()
                neg_vars = set()
                for clause in remaining_clauses:
                    for var, sign in clause:
                        if var in assignment:
                            continue
                        if sign == 1:
                            pos_vars.add(var)
                        else:
                            neg_vars.add(var)
                
                pure_pos = pos_vars - neg_vars
                pure_neg = neg_vars - pos_vars
                
                if pure_pos or pure_neg:
                    for var in pure_pos:
                        assignment[var] = True
                    for var in pure_neg:
                        assignment[var] = False
                    changed = True

            return assignment, remaining_clauses, False

        def solve(assignment, clauses):
            assignment, remaining_clauses, contradiction = propagate(assignment, clauses)
            if contradiction:
                return None
            if not remaining_clauses:
                return assignment
            
            # Choose a variable (heuristic: pick from first clause)
            var = remaining_clauses[0][0][0]
            
            # Try True
            new_assignment = assignment.copy()
            new_assignment[var] = True
            res = solve(new_assignment, remaining_clauses)
            if res is not None:
                return res
            
            # Try False
            new_assignment = assignment.copy()
            new_assignment[var] = False
            res = solve(new_assignment, remaining_clauses)
            if res is not None:
                return res
            
            return None

        return solve({}, clauses)

    def try_extend(n, adj, edges, max_time=5):
        """Try to add one vertex. Returns new (n, adj, edges) or None."""
        triangles = find_triangles(n, adj)
        indep_sets_5 = find_indep_sets_5(n, adj)
        
        clauses = []
        for tri in triangles:
            u, v, w = tri
            clauses.append([(u, -1), (v, -1), (w, -1)])
        for ind in indep_sets_5:
            u, v, w, x, y = ind
            clauses.append([(u, 1), (v, 1), (w, 1), (x, 1), (y, 1)])
        
        if not clauses:
            assignment = [False] * n
        else:
            # Use DPLL
            assignment = dpll(clauses, n)
            
            if assignment is None:
                return None
        
        # Verify
        valid = True
        for c in clauses:
            sat = False
            for var, sign in c:
                val = assignment[var]
                if sign == 1:
                    if val: sat = True; break
                else:
                    if not val: sat = True; break
            if not sat:
                valid = False
                break
        
        if not valid:
            return None
        
        new_n = n + 1
        new_edges = edges[:]
        for i in range(n):
            if assignment[i]:
                new_edges.append([i, new_n - 1])
        
        new_adj = build_adj(new_n, new_edges)
        return (new_n, new_adj, new_edges)

    # Try multiple starting points
    best_g = None
    best_n = 0
    
    for q in [29, 17, 13, 5]:
        if time.time() > deadline - 20:
            break
        
        G = make_paley(q)
        if G is None:
            continue
        
        if not is_valid(G):
            continue
        
        n = G['n']
        edges = G['edges']
        adj = build_adj(n, edges)
        
        if n > best_n:
            best_g = {'n': n, 'edges': edges}
            best_n = n
        
        # Extension loop
        extended = True
        while extended and time.time() < deadline - 5:
            result = try_extend(n, adj, edges)
            if result is None:
                extended = False
            else:
                n, adj, edges = result
                if n > best_n:
                    best_g = {'n': n, 'edges': edges}
                    best_n = n
        
        if best_n >= 35:
            break  # Can't do much better than 35
    
    if best_g is None:
        return {'n': 1, 'edges': []}
    
    # Final verification
    if not is_valid(best_g):
        # Fallback to a known valid small graph
        return {'n': 1, 'edges': []}
    
    return best_g

# EVOLVE-BLOCK-END
