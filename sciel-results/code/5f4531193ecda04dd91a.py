# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from itertools import combinations
from scipy.sparse import lil_matrix

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Fallback: known n=9 solution (cyclic coloring)
    n0 = 9
    best_n = n0
    best_colors = [min(j - i, n0 - j + i) - 1 for i in range(n0) for j in range(i + 1, n0)]
    
    # Try to find solutions for n from 17 down to 10
    for target_n in range(17, 9, -1):
        if time.time() > deadline - 15:
            break
        
        solution = _solve_ilp(target_n, deadline)
        if solution is not None:
            best_n = target_n
            best_colors = solution
            break
    
    return {'n': best_n, 'colors': best_colors}


def _solve_ilp(n, deadline):
    """
    Solve 4-coloring of K_n with no monochromatic C6 using ILP.
    Uses a two-phase approach:
    Phase 1: Use a reduced constraint set (only "canonical" C6s) for larger n
    Phase 2: Verify and refine with full C6 check
    """
    # Build edge list
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    edge_to_idx = {(i, j): idx for idx, (i, j) in enumerate(edges)}
    
    # Variables: x[idx][c] for idx in range(num_edges), c in range(4)
    num_vars = num_edges * 4
    
    def var_idx(e_idx, c):
        return e_idx * 4 + c
    
    # Objective: minimize 0 (feasibility)
    c_obj = np.zeros(num_vars)
    
    # Enumerate C6s efficiently
    c6_list = _enumerate_c6s(n)
    
    # For very large n, limit constraints to keep ILP tractable
    max_constraints = 5000
    if len(c6_list) * 4 + num_edges > max_constraints:
        # Use a subset of C6s - prioritize those involving vertex 0 for symmetry breaking
        c6_list = c6_list[:max_constraints // 4]
    
    # Number of constraints: num_edges + len(c6_list) * 4
    num_constraints = num_edges + len(c6_list) * 4
    
    # Build sparse A matrix
    A = lil_matrix((num_constraints, num_vars), dtype=np.float64)
    lb = np.zeros(num_constraints)
    ub = np.zeros(num_constraints)
    
    # Constraint 1: edge sum = 1
    for e_idx in range(num_edges):
        row = e_idx
        for c in range(4):
            A[row, var_idx(e_idx, c)] = 1
        lb[row] = 1
        ub[row] = 1
    
    # Constraint 2: C6 constraints - sum of colors on cycle edges <= 5 for each color
    constraint_row = num_edges
    for cycle in c6_list:
        cycle_edges = []
        for i in range(6):
            u = cycle[i]
            v = cycle[(i + 1) % 6]
            if u > v:
                u, v = v, u
            e_idx = edge_to_idx[(u, v)]
            cycle_edges.append(e_idx)
        
        for c in range(4):
            row = constraint_row
            constraint_row += 1
            for e_idx in cycle_edges:
                A[row, var_idx(e_idx, c)] = 1
            lb[row] = 0
            ub[row] = 5
    
    # Variable bounds with symmetry breaking
    integrality = np.ones(num_vars, dtype=np.int8)
    var_lb = np.zeros(num_vars)
    var_ub = np.ones(num_vars)
    
    # Fix x[edge(0,1)][0] = 1
    if num_edges > 0:
        e01_idx = edge_to_idx[(0, 1)]
        v01_c0 = var_idx(e01_idx, 0)
        var_lb[v01_c0] = 1
        var_ub[v01_c0] = 1
    
    # Fix x[edge(0,2)][1] = 1
    if num_edges > 1:
        e02_idx = edge_to_idx[(0, 2)]
        v02_c1 = var_idx(e02_idx, 1)
        var_lb[v02_c1] = 1
        var_ub[v02_c1] = 1
    
    bounds = Bounds(var_lb, var_ub)
    
    # Solve with time check
    try:
        result = milp(
            c=c_obj,
            constraints=LinearConstraint(A.tocsr(), lb, ub),
            integrality=integrality,
            bounds=bounds
        )
    except Exception:
        return None
    
    if result.status != 0:
        return None
    
    x = result.x
    if x is None:
        return None
    
    # Extract colors
    colors = []
    for e_idx in range(num_edges):
        for c in range(4):
            if x[var_idx(e_idx, c)] > 0.5:
                colors.append(c)
                break
        else:
            return None
    
    # Verify the solution has no C6
    if _has_monochromatic_c6(n, colors):
        return None
    
    return colors


def _enumerate_c6s(n):
    """Enumerate all distinct 6-cycles in K_n efficiently."""
    c6_list = []
    for subset in combinations(range(n), 6):
        verts = list(subset)
        seen_cycles = set()
        # Generate all 60 distinct cycles on these 6 vertices
        # Fix vertex 0 as start, enumerate permutations of remaining 5
        # Divide by 2 for direction
        for perm in _permutations(verts[1:]):
            cycle = [verts[0]] + list(perm)
            # Normalize for uniqueness
            # Two cycles are equivalent if one is a rotation or reversal of the other
            # Since we fixed vertex 0 as start, we just need to check reversal
            rev = [verts[0]] + list(reversed(perm))
            norm = tuple(cycle) if tuple(cycle) < tuple(rev) else tuple(rev)
            if norm not in seen_cycles:
                seen_cycles.add(norm)
                c6_list.append(list(norm))
    
    return c6_list


def _has_monochromatic_c6(n, colors):
    """Check if a 4-coloring of K_n has any monochromatic C6."""
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_to_idx = {(i, j): idx for idx, (i, j) in enumerate(edges)}
    
    # Build adjacency per color
    adj = [[0] * n for _ in range(4)]
    for idx, (i, j) in enumerate(edges):
        c = colors[idx]
        adj[c][i] |= (1 << j)
        adj[c][j] |= (1 << i)
    
    for c in range(4):
        adj_c = adj[c]
        for u in range(n):
            # Try to find a C6 starting from u
            N_u = adj_c[u] & ~((1 << (u + 1)) - 1)
            if N_u == 0:
                continue
            
            def dfs(current, depth, visited, path):
                if depth == 5:
                    if (adj_c[current] >> u) & 1:
                        return True
                    return False
                candidates = adj_c[current] & ~((1 << (u + 1)) - 1) & ~visited
                c_bits = candidates
                while c_bits:
                    c_bit = c_bits & -c_bits
                    cv = c_bit.bit_length() - 1
                    c_bits ^= c_bit
                    if dfs(cv, depth + 1, visited | c_bit, path + [cv]):
                        return True
                return False
            
            s_bits = N_u
            while s_bits:
                s_bit = s_bits & -s_bits
                s = s_bit.bit_length() - 1
                s_bits ^= s_bit
                if dfs(s, 1, s_bit, [s]):
                    return True
    
    return False


def _permutations(iterable):
    """Generate all permutations of an iterable."""
    if len(iterable) <= 1:
        yield tuple(iterable)
        return
    for i, item in enumerate(iterable):
        rest = list(iterable[:i]) + list(iterable[i+1:])
        for perm in _permutations(rest):
            yield (item,) + perm


def _solve_local_search(n, deadline):
    """Improved local search with better C6 detection and restart."""
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    edge_to_idx = {(i, j): idx for idx, (i, j) in enumerate(edges)}
    
    def find_c6_with_edge(colors):
        """Find a monochromatic C6, return (color, edge_idx) or (None, None)."""
        adj = [[0] * n for _ in range(4)]
        for idx, (i, j) in enumerate(edges):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
        
        for c in range(4):
            adj_c = adj[c]
            for u in range(n):
                N_u = adj_c[u] & ~((1 << (u + 1)) - 1)
                if N_u == 0:
                    continue
                
                def dfs(current, depth, visited):
                    if depth == 5:
                        if (adj_c[current] >> u) & 1:
                            return [current]
                        return None
                    candidates = adj_c[current] & ~((1 << (u + 1)) - 1) & ~visited
                    c_bits = candidates
                    while c_bits:
                        c_bit = c_bits & -c_bits
                        cv = c_bit.bit_length() - 1
                        c_bits ^= c_bit
                        result = dfs(cv, depth + 1, visited | c_bit)
                        if result is not None:
                            return [cv] + result
                    return None
                
                s_bits = N_u
                while s_bits:
                    s_bit = s_bits & -s_bits
                    s = s_bit.bit_length() - 1
                    s_bits ^= s_bit
                    result = dfs(s, 1, s_bit)
                    if result is not None:
                        cycle = [u, s] + result
                        k = random.randint(0, 5)
                        i, j = cycle[k], cycle[(k + 1) % 6]
                        if i > j:
                            i, j = j, i
                        return c, edge_to_idx[(i, j)]
        return None, None
    
    # Try multiple random restarts
    max_restarts = 5
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        max_steps = 5000
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            c, idx = find_c6_with_edge(colors)
            if c is None:
                return colors
            
            old_c = colors[idx]
            new_c = random.choice([cc for cc in range(4) if cc != old_c])
            colors[idx] = new_c
        
    return None
# EVOLVE-BLOCK-END
