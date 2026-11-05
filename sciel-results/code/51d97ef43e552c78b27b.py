"""Few edges in 5-uniform hypergraphs without Property B (m(5))."""
import time
import random
from itertools import combinations

def _is_2_colorable(n, edges, deadline=None):
    """Check if hypergraph is 2-colorable via SAT backtracking with unit propagation.
    Returns True if 2-colorable, False if not."""
    m = len(edges)
    if m == 0:
        return True
    
    # SAT formulation: each vertex v has a boolean x_v (True = color 1, False = color 0)
    # For each edge e = {v1,...,v5}:
    #   Clause 1: (x_v1 ∨ x_v2 ∨ x_v3 ∨ x_v4 ∨ x_v5)  - not all color 0
    #   Clause 2: (¬x_v1 ∨ ¬x_v2 ∨ ¬x_v3 ∨ ¬x_v4 ∨ ¬x_v5)  - not all color 1
    
    # Precompute: for each vertex, which clauses contain it (and with what sign)
    # Clauses are indexed 0..2m-1
    # Clause 2*i: positive clause for edge i
    # Clause 2*i+1: negative clause for edge i
    
    vert_clauses = {v: [] for v in range(n)}
    for i, e in enumerate(edges):
        for v in e:
            vert_clauses[v].append((2 * i, 1))       # positive
            vert_clauses[v].append((2 * i + 1, -1))  # negative
    
    # State: assignment[v] = None (unassigned), True, False
    # clause_satisfied[c] = True if clause c is already satisfied
    # clause_literal_counts: for each clause, count of unassigned literals and their signs
    
    # Use iterative approach with unit propagation
    
    def solve(assignment, deadline):
        """DPLL with unit propagation. Returns True if satisfiable."""
        if deadline is not None and time.time() > deadline:
            return True  # assume satisfiable on timeout (safe for our purpose)
        
        m_clause = 2 * m
        
        # Quick check: if all clauses satisfied, done
        # We'll track which clauses are satisfied and which are unit
        
        satisfied = [False] * m_clause
        # For each clause, track: number of unassigned literals, and if exactly 1 unassigned, what it is
        
        # Build clause status
        clause_unassigned = [0] * m_clause
        clause_unassigned_lit = [None] * m_clause  # (vertex, sign) if exactly 1 unassigned
        
        for ci in range(m_clause):
            edge_idx = ci // 2
            is_positive = (ci % 2 == 0)
            e = edges[edge_idx]
            unassigned_count = 0
            unassigned_v = None
            for v in e:
                if assignment[v] is None:
                    unassigned_count += 1
                    unassigned_v = v
                else:
                    val = assignment[v]
                    # Literal is satisfied if:
                    # positive clause: val is True
                    # negative clause: val is False
                    if is_positive:
                        if val:
                            satisfied[ci] = True
                            break
                    else:
                        if not val:
                            satisfied[ci] = True
                            break
            else:
                if unassigned_count == 1:
                    clause_unassigned_lit[ci] = (unassigned_v, 1 if is_positive else -1)
                clause_unassigned[ci] = unassigned_count
        
        # Unit propagation
        changed = True
        while changed:
            changed = False
            for ci in range(m_clause):
                if satisfied[ci]:
                    continue
                if clause_unassigned[ci] == 0:
                    # Clause is falsified -> unsatisfiable
                    return False
                if clause_unassigned[ci] == 1:
                    v, sign = clause_unassigned_lit[ci]
                    if assignment[v] is not None:
                        continue
                    # Force assignment
                    new_val = (sign == 1)
                    assignment[v] = new_val
                    changed = True
                    # Update all clauses containing v
                    for cj, _ in vert_clauses[v]:
                        if satisfied[cj]:
                            continue
                        edge_idx = cj // 2
                        is_positive = (cj % 2 == 0)
                        e = edges[edge_idx]
                        unassigned_count = 0
                        unassigned_v = None
                        for vv in e:
                            if assignment[vv] is None:
                                unassigned_count += 1
                                unassigned_v = vv
                            else:
                                val = assignment[vv]
                                if is_positive:
                                    if val:
                                        satisfied[cj] = True
                                        break
                                else:
                                    if not val:
                                        satisfied[cj] = True
                                        break
                        else:
                            if unassigned_count == 1:
                                clause_unassigned_lit[cj] = (unassigned_v, 1 if is_positive else -1)
                            clause_unassigned[cj] = unassigned_count
                            if unassigned_count == 0:
                                # Check if falsified
                                # All literals are assigned and none satisfied the clause
                                # This means the clause is falsified
                                return False
        
        # Check if all clauses satisfied
        for ci in range(m_clause):
            if not satisfied[ci]:
                if clause_unassigned[ci] == 0:
                    return False
                # Need to branch
        
        # Find an unassigned variable to branch on
        # Heuristic: pick variable in most constrained clause
        best_v = None
        best_score = -1
        for v in range(n):
            if assignment[v] is not None:
                continue
            score = 0
            for ci, sign in vert_clauses[v]:
                if satisfied[ci]:
                    continue
                if clause_unassigned[ci] == 1:
                    score += 10
                elif clause_unassigned[ci] == 2:
                    score += 5
                elif clause_unassigned[ci] >= 3:
                    score += 1
            if score > best_score:
                best_score = score
                best_v = v
        
        if best_v is None:
            # All assigned, check all clauses satisfied
            for ci in range(m_clause):
                if not satisfied[ci]:
                    if clause_unassigned[ci] == 0:
                        return False
            return True
        
        # Branch on best_v
        for val in [True, False]:
            new_assignment = assignment[:]
            new_assignment[best_v] = val
            if solve(new_assignment, deadline):
                return True
        return False
    
    assignment = [None] * n
    return solve(assignment, deadline)


def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Strategy: search for small non-2-colorable 5-uniform hypergraphs
    # Known: Fano plane (3-uniform, 7 vertices, 7 edges) is non-2-colorable
    # For 5-uniform, we can try various constructions
    
    best_result = None
    best_edge_count = float('inf')
    
    rng = random.Random(12345)
    
    # Try different vertex counts and search for minimal non-2-colorable hypergraphs
    for n in [9, 10, 11, 12, 13, 14]:
        if time.time() > deadline:
            break
        
        all_possible = list(combinations(range(n), 5))
        num_possible = len(all_possible)
        
        # Try multiple random orderings
        for trial in range(30):
            if time.time() > deadline:
                break
            
            # Shuffle edges
            shuffled = all_possible[:]
            rng.shuffle(shuffled)
            
            edges = []
            # Add edges one by one, check when it becomes non-2-colorable
            for i in range(min(num_possible, 80)):
                if time.time() > deadline:
                    break
                
                edges.append(list(shuffled[i]))
                # Check if non-2-colorable (only check periodically to save time)
                if (i + 1) % 3 == 0 or i == num_possible - 1:
                    if not _is_2_colorable(n, edges, deadline):
                        # Found non-2-colorable
                        if len(edges) < best_edge_count:
                            best_edge_count = len(edges)
                            best_result = {'n': n, 'edges': edges[:]}
                        break
    
    if best_result is not None:
        return best_result
    
    # Fallback: complete 5-uniform on 9 vertices
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
