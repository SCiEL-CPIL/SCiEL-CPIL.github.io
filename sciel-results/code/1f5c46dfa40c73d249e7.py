import random
import time
import sys
import itertools
from collections import defaultdict

def construct():
    """
    Constructs a K4-free graph with no independent set of size 6.
    Uses a custom DPLL SAT solver to incrementally add vertices.
    """
    start_time = time.time()
    timeout = 110  # Leave 10s buffer
    
    # Fallback known solution (Paley graph of order 17)
    def get_fallback():
        n = 17
        squares = {x * x % 17 for x in range(1, 17)}
        edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
        return {'n': n, 'edges': edges}

    best_result = get_fallback()
    
    # We will try to build graphs of size n starting from 18 up to 35
    # For each n, we run the SAT solver.
    
    def solve_sat(n, time_limit=5.0):
        """
        Solves for a graph on n vertices.
        Variables: x[i][j] for 0 <= i < j < n. True if edge exists.
        Constraints:
        1. No K4: For any 4 vertices {a,b,c,d}, not (all 6 edges exist).
           Clause: ~x_ab OR ~x_ac OR ~x_ad OR ~x_bc OR ~x_bd OR ~x_cd
        2. No Independent Set of size 6: For any 6 vertices {a,...,f}, not (no edges exist).
           Clause: x_ab OR x_ac OR ... OR x_ef (at least one edge exists)
        """
        # Map edge (i,j) to variable index
        # Total vars = n*(n-1)//2
        num_vars = n * (n - 1) // 2
        
        var_map = {}
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                var_map[(i, j)] = idx
                idx += 1
        
        # Clauses are lists of literals. Positive = True, Negative = False.
        # We use integers 1..num_vars. Literal is var if positive, -var if negative.
        clauses = []
        
        # Constraint 1: No K4
        # Iterate over all 4-sets
        for quad in itertools.combinations(range(n), 4):
            a, b, c, d = quad
            # Edges: ab, ac, ad, bc, bd, cd
            # Clause: ~ab OR ~ac OR ~ad OR ~bc OR ~bd OR ~cd
            clause = []
            for u, v in [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]:
                clause.append(-var_map[(u, v)])
            clauses.append(clause)
            
        # Constraint 2: No Independent Set of size 6
        # Iterate over all 6-sets
        for sext in itertools.combinations(range(n), 6):
            # Edges among these 6 vertices: C(6,2)=15 edges
            # Clause: x_e1 OR x_e2 OR ... OR x_e15
            clause = []
            for u, v in itertools.combinations(sext, 2):
                clause.append(var_map[(u, v)])
            clauses.append(clause)
                
        # DPLL Solver
        # State: assignment dict {var: bool}
        # Clauses: list of lists of literals
        
        assignment = {}
        num_clauses = len(clauses)
        
        # Preprocess: remove duplicate literals in clauses? Not strictly necessary for DPLL if we check carefully.
        # But for efficiency, we can sort and unique.
        
        # Unit propagation and DPLL
        # To keep it simple and fast enough for small n, we will implement a basic DPLL with unit propagation.
        
        def unit_propagate(clauses, assignment):
            """
            Returns (new_clauses, new_assignment, satisfied, contradiction)
            """
            changed = True
            while changed:
                changed = False
                new_clauses = []
                for clause in clauses:
                    # Check if clause is satisfied
                    sat = False
                    # Check if clause is falsified (all literals are false in assignment)
                    # Actually, we remove literals that are satisfied.
                    # If a clause becomes empty, contradiction.
                    new_clause = []
                    for lit in clause:
                        var = abs(lit)
                        if var in assignment:
                            val = assignment[var]
                            if (lit > 0 and val) or (lit < 0 and not val):
                                sat = True
                                break
                            else:
                                # Literal is false, discard
                                continue
                        else:
                            new_clause.append(lit)
                    
                    if sat:
                        continue
                    if not new_clause:
                        return None, assignment, False, True # Contradiction
                    if len(new_clause) == 1:
                        # Unit clause
                        lit = new_clause[0]
                        var = abs(lit)
                        val = lit > 0
                        if var in assignment:
                            if assignment[var] != val:
                                return None, assignment, False, True
                        else:
                            assignment[var] = val
                            changed = True
                            # We don't add to new_clauses here, loop again to propagate
                            continue
                    new_clauses.append(new_clause)
            
            return new_clauses, assignment, True, False

        def dpll(clauses, assignment):
            # Check for contradiction first via unit prop
            clauses, assignment, ok, contra = unit_propagate(clauses, assignment)
            if contra:
                return None
            if not ok:
                return None
            
            if not clauses:
                return assignment
                
            # Choose a variable
            # Heuristic: Pick variable appearing in most clauses? Or just first unassigned.
            # For simplicity and speed in Python, let's pick a variable from the first clause.
            # Better: Pick variable with highest frequency.
            # Given time constraints, let's try picking a variable from a short clause.
            
            # Find a clause with minimal length
            min_clause = min(clauses, key=len)
            
            # Pick a variable from this clause that is not assigned
            var = None
            for lit in min_clause:
                v = abs(lit)
                if v not in assignment:
                    var = v
                    break
            
            if var is None:
                # All variables in min_clause are assigned?
                # This should have been handled by unit propagation if it became unit.
                # If it's still there, it means it has unassigned vars? 
                # Wait, if all vars in min_clause are assigned, unit_prop would have resolved it.
                # So var must be found.
                return None

            # Try True
            assignment[var] = True
            sol = dpll(clauses, assignment)
            if sol is not None:
                return sol
            
            # Try False
            assignment[var] = False
            sol = dpll(clauses, assignment)
            if sol is not None:
                return sol
            
            # Backtrack
            del assignment[var]
            return None

        # Optimization: Limit recursion depth or time?
        # For n=18-25, DPLL might be slow in pure Python.
        # We will add a node limit.
        
        node_count = [0]
        max_nodes = 50000 # Arbitrary limit per call
        
        def dpll_limited(clauses, assignment):
            node_count[0] += 1
            if node_count[0] > max_nodes:
                return "TIMEOUT"
            
            clauses, assignment, ok, contra = unit_propagate(clauses, assignment)
            if contra:
                return None
            if not ok:
                return None
            
            if not clauses:
                return assignment
                
            # Choose variable
            # Heuristic: Min length clause
            min_clause = min(clauses, key=len)
            
            var = None
            for lit in min_clause:
                v = abs(lit)
                if v not in assignment:
                    var = v
                    break
            
            if var is None:
                return None

            # Order: Try True first? Or based on something?
            # Let's try True.
            assignment[var] = True
            sol = dpll_limited(clauses, assignment)
            if sol == "TIMEOUT":
                del assignment[var]
                return "TIMEOUT"
            if sol is not None:
                return sol
            
            assignment[var] = False
            sol = dpll_limited(clauses, assignment)
            if sol == "TIMEOUT":
                del assignment[var]
                return "TIMEOUT"
            if sol is not None:
                return sol
            
            del assignment[var]
            return None

        # Run DPLL
        # To avoid deep recursion limits, we might need to increase recursion limit or use iterative.
        # sys.setrecursionlimit(10000)
        
        import sys
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(10000)
        
        try:
            sol = dpll_limited(clauses, {})
        finally:
            sys.setrecursionlimit(old_limit)
            
        if sol == "TIMEOUT":
            return None
            
        if sol is None:
            return None
            
        # Extract edges
        edges = []
        for (i, j), v in var_map.items():
            if sol.get(v, False):
                edges.append([i, j])
                
        return edges

    # Try to find a solution for n = 18, 19, ...
    # We start from 18 because 17 is the seed.
    # We go up to 35.
    
    for n in range(18, 36):
        if time.time() - start_time > timeout:
            break
            
        # Try a few restarts with different random seeds? 
        # DPLL is deterministic given the variable order.
        # We can randomize the variable order or initial assignment?
        # For now, just run DPLL.
        # If it fails (unsat or timeout), try next n? 
        # Wait, if it's unsat, it's unsat. If timeout, maybe it's solvable but slow.
        
        # To improve success rate, we can try a few different "heuristics" for variable selection.
        # But for now, let's just try once.
        
        edges = solve_sat(n, time_limit=timeout - (time.time() - start_time))
        
        if edges is not None:
            # Verify solution
            # Build adj matrix
            adj = [[False]*n for _ in range(n)]
            for u, v in edges:
                adj[u][v] = True
                adj[v][u] = True
            
            # Check K4
            has_k4 = False
            for i in range(n):
                for j in range(i+1, n):
                    if not adj[i][j]: continue
                    for k in range(j+1, n):
                        if not adj[i][k] or not adj[j][k]: continue
                        for l in range(k+1, n):
                            if adj[i][l] and adj[j][l] and adj[k][l]:
                                has_k4 = True
                                break
                        if has_k4: break
                    if has_k4: break
                if has_k4: break
            
            # Check Alpha 6
            has_alpha6 = False
            # Complement adjacency
            comp_adj = [[not adj[i][j] for j in range(n)] for i in range(n)]
            for i in range(n):
                comp_adj[i][i] = False # No self-loops in independent set check logic usually, but adj[i][i] is False so comp is True. 
                # Wait, independent set is set of vertices with NO edges between them.
                # So we look for clique of size 6 in complement.
                # In complement, edge exists if no edge in G.
                # So comp_adj[i][j] = not adj[i][j] for i!=j.
                # We need to check if there is a K6 in comp_adj.
            
            if not has_k4:
                # Check K6 in comp_adj
                for sext in itertools.combinations(range(n), 6):
                    is_clique = True
                    for u, v in itertools.combinations(sext, 2):
                        if not comp_adj[u][v]:
                            is_clique = False
                            break
                    if is_clique:
                        has_alpha6 = True
                        break
                        
            if not has_k4 and not has_alpha6:
                # Valid!
                return {'n': n, 'edges': edges}
            else:
                # Invalid? Should not happen if DPLL is correct.
                # But maybe DPLL returned a partial assignment or something?
                # Or maybe my verification is wrong?
                # Let's print debug info if we have time? No, just skip.
                pass
                
    # If we reach here, we didn't find anything better than 17
    return best_result

# EVOLVE-BLOCK-END
