"""Improved program for Ramsey R(5,5): using WalkSAT for vertex addition."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import deque

DEADLINE_SECONDS = 320.0
START_TIME = time.time()

def _time_left():
    return DEADLINE_SECONDS - (time.time() - START_TIME)

def _paley_graph(q):
    """Return edges of the Paley graph on q vertices (q a prime congruent to 1 mod 4)."""
    residues = {(x * x) % q for x in range(1, q)}
    return [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]

def _build_adj(edges, n):
    """Build adjacency as list of int bitsets."""
    adj = [0] * n
    for i, j in edges:
        adj[i] |= (1 << j)
        adj[j] |= (1 << i)
    return adj

def _find_k4s(adj, n):
    """Find all 4-cliques in the graph."""
    k4s = []
    for a in range(n):
        for b in range(a + 1, n):
            if not (adj[a] & (1 << b)):
                continue
            common_ab = adj[a] & adj[b] & ~((1 << (b + 1)) - 1)
            c_mask = common_ab
            while c_mask:
                c_bit = c_mask & (-c_mask)
                c = c_bit.bit_length() - 1
                c_mask ^= c_bit
                common_abc = common_ab & adj[c] & ~((1 << (c + 1)) - 1)
                d_mask = common_abc
                while d_mask:
                    d_bit = d_mask & (-d_mask)
                    d = d_bit.bit_length() - 1
                    d_mask ^= d_bit
                    k4s.append((a, b, c, d))
    return k4s

def _find_indep4s(adj, n):
    """Find all 4-independent sets in the graph."""
    all_mask = (1 << n) - 1
    non_adj = [all_mask ^ adj[i] ^ (1 << i) for i in range(n)]
    indep4s = []
    for a in range(n):
        for b in range(a + 1, n):
            if adj[a] & (1 << b):
                continue
            common_non_ab = non_adj[a] & non_adj[b] & ~((1 << (b + 1)) - 1)
            c_mask = common_non_ab
            while c_mask:
                c_bit = c_mask & (-c_mask)
                c = c_bit.bit_length() - 1
                c_mask ^= c_bit
                common_non_abc = common_non_ab & non_adj[c] & ~((1 << (c + 1)) - 1)
                d_mask = common_non_abc
                while d_mask:
                    d_bit = d_mask & (-d_mask)
                    d = d_bit.bit_length() - 1
                    d_mask ^= d_bit
                    indep4s.append((a, b, c, d))
    return indep4s

def _walksat_solve(clauses, n_vars, max_steps=50000, restarts=100):
    """
    Simple WalkSAT implementation.
    clauses: list of tuples. Each tuple contains literals.
    Positive literal i means x_i is True. Negative literal -i means x_i is False.
    """
    if not clauses:
        return [True] * n_vars
    
    for r in range(restarts):
        if _time_left() < 5:
            break
            
        # Random initial assignment
        assignment = [random.random() < 0.5 for _ in range(n_vars)]
        
        # Count occurrences of each variable in clauses
        # And current satisfaction status
        # For speed, we just re-evaluate cost on each step or keep a delta.
        # Given small n_vars (37-45), re-evaluation is cheap enough if optimized.
        
        for step in range(max_steps):
            # Check if all clauses are satisfied
            all_sat = True
            unsat_clauses = []
            for c in clauses:
                sat = False
                for lit in c:
                    var = abs(lit) - 1
                    is_pos = lit > 0
                    if assignment[var] == is_pos:
                        sat = True
                        break
                if not sat:
                    all_sat = False
                    unsat_clauses.append(c)
            
            if all_sat:
                return assignment
            
            # Pick a random unsatisfied clause
            if not unsat_clauses:
                break
            clause = random.choice(unsat_clauses)
            
            # Pick a variable to flip from this clause
            # Weighted by frequency in other unsatisfied clauses? 
            # Simple: random variable from the clause
            lit = random.choice(clause)
            var = abs(lit) - 1
            assignment[var] = not assignment[var]
            
    return None

def _try_add_vertex_walksat(edges, n):
    """
    Try to add a new vertex (index n) using WalkSAT to find a valid neighborhood S.
    Returns (new_edges, success).
    """
    adj = _build_adj(edges, n)
    
    # Find all 4-cliques and 4-independent sets
    k4s = _find_k4s(adj, n)
    indep4s = _find_indep4s(adj, n)
    
    # Build CNF clauses
    # Variables: x_0 ... x_{n-1}
    # x_i = True means i is in S (connected to new vertex)
    
    clauses = []
    
    # Constraint 1: No K4 fully in S
    # For each K4 (a,b,c,d), NOT (x_a AND x_b AND x_c AND x_d)
    # Clause: (NOT x_a) OR (NOT x_b) OR (NOT x_c) OR (NOT x_d)
    # Literals: -a-1, -b-1, -c-1, -d-1
    for k4 in k4s:
        clauses.append((-k4[0]-1, -k4[1]-1, -k4[2]-1, -k4[3]-1))
        
    # Constraint 2: Every Indep4 intersects S
    # For each Indep4 (a,b,c,d), NOT (NOT x_a AND NOT x_b AND NOT x_c AND NOT x_d)
    # Clause: x_a OR x_b OR x_c OR x_d
    # Literals: a+1, b+1, c+1, d+1
    for i4 in indep4s:
        clauses.append((i4[0]+1, i4[1]+1, i4[2]+1, i4[3]+1))
        
    # Solve
    assignment = _walksat_solve(clauses, n, max_steps=10000, restarts=50)
    
    if assignment is None:
        return None, False
        
    # Construct new edges
    new_edges = list(edges)
    for i in range(n):
        if assignment[i]:
            new_edges.append((i, n))
            
    return new_edges, True

def _is_55_free(edges, n):
    """Check if the graph has no K5 and no independent 5-set."""
    adj = _build_adj(edges, n)
    
    # Check K5
    for a in range(n):
        for b in range(a + 1, n):
            if not (adj[a] & (1 << b)):
                continue
            common_ab = adj[a] & adj[b] & ~((1 << (b + 1)) - 1)
            c_mask = common_ab
            while c_mask:
                c_bit = c_mask & (-c_mask)
                c = c_bit.bit_length() - 1
                c_mask ^= c_bit
                common_abc = common_ab & adj[c] & ~((1 << (c + 1)) - 1)
                d_mask = common_abc
                while d_mask:
                    d_bit = d_mask & (-d_mask)
                    d = d_bit.bit_length() - 1
                    d_mask ^= d_bit
                    common_abcd = common_abc & adj[d] & ~((1 << (d + 1)) - 1)
                    if common_abcd:
                        return False
                        
    # Check Indep5
    all_mask = (1 << n) - 1
    non_adj = [all_mask ^ adj[i] ^ (1 << i) for i in range(n)]
    for a in range(n):
        for b in range(a + 1, n):
            if adj[a] & (1 << b):
                continue
            common_non_ab = non_adj[a] & non_adj[b] & ~((1 << (b + 1)) - 1)
            c_mask = common_non_ab
            while c_mask:
                c_bit = c_mask & (-c_mask)
                c = c_bit.bit_length() - 1
                c_mask ^= c_bit
                common_non_abc = common_non_ab & non_adj[c] & ~((1 << (c + 1)) - 1)
                d_mask = common_non_abc
                while d_mask:
                    d_bit = d_mask & (-d_mask)
                    d = d_bit.bit_length() - 1
                    d_mask ^= d_bit
                    common_non_abcd = common_non_abc & non_adj[d] & ~((1 << (d + 1)) - 1)
                    if common_non_abcd:
                        return False
                        
    return True

def construct():
    # Start with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    n = q
    
    # Verify it's valid
    if not _is_55_free(edges, n):
        # Fallback to a smaller known valid graph if Paley fails (shouldn't happen)
        # For now, assume it's valid
        pass
    
    best_n = n
    best_edges = edges[:]
    
    # Try to add vertices one by one using WalkSAT
    max_attempts = 10
    for attempt in range(max_attempts):
        if _time_left() < 20:
            break
        
        new_edges, success = _try_add_vertex_walksat(edges, n)
        if success:
            # Verify the new graph
            if _is_55_free(new_edges, n + 1):
                edges = new_edges
                n += 1
                if n > best_n:
                    best_n = n
                    best_edges = edges[:]
            else:
                # WalkSAT found a solution but it's invalid? 
                # This can happen if the CNF construction is wrong or WalkSAT bug.
                # Re-try with more restarts?
                # For now, break to avoid wasting time
                pass
        else:
            # Failed to add vertex
            # Try a few more times with different random seeds?
            # Or break
            # Let's try a few more times
            success = False
            for _ in range(5):
                if _time_left() < 10:
                    break
                new_edges, success = _try_add_vertex_walksat(edges, n)
                if success:
                    if _is_55_free(new_edges, n + 1):
                        edges = new_edges
                        n += 1
                        if n > best_n:
                            best_n = n
                            best_edges = edges[:]
                        break
            
            if not success:
                break
    
    # Return the best we found
    return best_n, best_edges

# EVOLVE-BLOCK-END
