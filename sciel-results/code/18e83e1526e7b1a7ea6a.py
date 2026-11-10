"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = start_time + 80.0  # Leave 10s margin for safety and fallback
    
    N = 256
    max_sum = 2 * N
    
    # Precompute perfect squares up to 512
    perfect_squares = set()
    for k in range(1, int(math.isqrt(max_sum)) + 2):
        sq = k * k
        if sq <= max_sum:
            perfect_squares.add(sq)
    
    # Identify valid vertices: a such that 2a is NOT a perfect square
    # If 2a is a perfect square, a cannot be in the set (self-loop constraint)
    valid_vertices = []
    for a in range(1, N + 1):
        if (2 * a) not in perfect_squares:
            valid_vertices.append(a)
    
    n = len(valid_vertices)
    
    # If no valid vertices (should not happen for N=256), return [1]
    if n == 0:
        return [1]
    
    # Build adjacency list for the conflict graph
    # adj[i] is a set of indices j such that valid_vertices[i] + valid_vertices[j] is a perfect square
    adj = [set() for _ in range(n)]
    for i in range(n):
        vi = valid_vertices[i]
        for j in range(i + 1, n):
            vj = valid_vertices[j]
            if (vi + vj) in perfect_squares:
                adj[i].add(j)
                adj[j].add(i)
    
    # Strategy 1: Exact ILP using SciPy
    # Variables: x_i for i in 0..n-1
    # Maximize sum(x_i)
    # Constraints: x_i + x_j <= 1 for all edges (i, j)
    
    c = -np.ones(n) # Minimize -sum(x) => Maximize sum(x)
    
    # Build constraint matrix A_ub such that A_ub @ x <= b_ub
    # For each edge (i, j), we need x_i + x_j <= 1
    # We can construct this matrix sparsely or densely.
    # Given n ~ 245 (256 - 11 self-loops), a dense matrix is 245x245, which is small.
    # However, the number of constraints is the number of edges.
    
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j: # Avoid duplicates
                edges.append((i, j))
    
    m_constraints = len(edges)
    
    if m_constraints == 0:
        # No conflicts, take all
        return valid_vertices
    
    # Construct A_ub
    # Using sparse matrix for efficiency if many edges, but dense is fine for n<300
    # Let's use a dense array for simplicity and speed in construction for small N
    # Actually, scipy.milp accepts sparse matrices. Let's build a sparse one to be safe.
    import scipy.sparse as sp
    
    rows = []
    cols = []
    data = []
    
    for (i, j) in edges:
        rows.append(len(rows))
        cols.append(i)
        data.append(1)
        
        rows.append(len(rows))
        cols.append(j)
        data.append(1)
    
    A_ub = sp.csr_matrix((data, (rows, cols)), shape=(m_constraints, n))
    b_ub = np.ones(m_constraints)
    
    # Bounds: 0 <= x_i <= 1
    lb = np.zeros(n)
    ub = np.ones(n)
    bounds = Bounds(lb, ub)
    
    # Integrality: x_i must be integer
    integrality = np.ones(n)
    
    try:
        # Time limit for solver
        solver_time_limit = DEADLINE - time.time() - 2.0 # Leave 2s for post-processing
        if solver_time_limit < 1.0:
            raise TimeoutError("Not enough time for ILP")
            
        res = milp(
            c=c,
            constraints=[LinearConstraint(A_ub, ub=b_ub)],
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": solver_time_limit, "presolve": "default"}
        )
        
        if res.status == 0: # Optimal
            # Extract solution
            x = res.x
            # Due to floating point, x might be 0.9999 or 1.0001
            # Threshold at 0.5
            selected_indices = [i for i in range(n) if x[i] > 0.5]
            
            if len(selected_indices) > 0:
                result = [valid_vertices[i] for i in selected_indices]
                # Validate result
                result_set = set(result)
                valid = True
                for a in result:
                    if (2 * a) in perfect_squares:
                        valid = False
                        break
                    for b in result:
                        if a <= b and (a + b) in perfect_squares:
                            # Only check pairs once? The constraint x_i+x_j<=1 ensures no edge.
                            # But let's double check.
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    return sorted(result)
                else:
                    # ILP gave an invalid solution? Unlikely, but fallback
                    pass
            else:
                # Empty set? ILP maximizes, so it should find at least one if possible.
                # But if all vertices are isolated, it should return all.
                # If it returns empty, something is wrong.
                pass
                
    except Exception as e:
        # Fallback to heuristic if ILP fails or times out
        pass
    
    # Strategy 2: Heuristic Fallback (Greedy + Local Search)
    # This ensures we always return a valid non-empty set.
    
    def greedy_independent_set(vertices, adj, rs):
        remaining_verts = set(range(len(vertices)))
        result = set()
        
        while remaining_verts:
            min_deg = float('inf')
            candidates = []
            for v in remaining_verts:
                deg = len(adj[v] & remaining_verts)
                if deg < min_deg:
                    min_deg = deg
                    candidates = [v]
                elif deg == min_deg:
                    candidates.append(v)
            
            v = rs.choice(candidates)
            result.add(v)
            remaining_verts.remove(v)
            for nb in adj[v]:
                if nb in remaining_verts:
                    remaining_verts.remove(nb)
        
        return result
    
    def local_improvement(ind_set, adj, vertices, rs, max_iterations=500):
        ind_set = set(ind_set)
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            candidates = [v for v in range(len(vertices)) if v not in ind_set]
            rs.shuffle(candidates)
            
            for v in candidates:
                conflicts = adj[v] & ind_set
                if len(conflicts) == 0:
                    ind_set.add(v)
                    improved = True
                else:
                    # Try swap if net gain > 0 (impossible for single add)
                    # Or if we can remove 1 and add 1? No, that's 0 gain.
                    # Only add if 0 conflicts.
                    pass
        
        return ind_set
    
    best_set = set()
    best_size = 0
    
    # Run a few heuristics if time permits
    time_left = DEADLINE - time.time()
    
    if time_left > 5.0:
        for seed in range(10):
            if time.time() > DEADLINE - 2.0:
                break
            rs = random.Random(seed)
            s = greedy_independent_set(valid_vertices, adj, rs)
            s = local_improvement(s, adj, valid_vertices, rs, max_iterations=200)
            if len(s) > best_size:
                best_size = len(s)
                best_set = set(s)
        
        if best_size > 0:
            result = [valid_vertices[i] for i in best_set]
            # Final validation
            result_set = set(result)
            valid = True
            for a in result:
                if (2 * a) in perfect_squares:
                    valid = False
                    break
                for b in result:
                    if a <= b and (a + b) in perfect_squares:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return sorted(result)
    
    # Last resort: single vertex
    # Find a vertex with no self-loop (all valid_vertices have no self-loop by definition)
    # So any valid_vertex is a valid set of size 1.
    return [valid_vertices[0]]

# EVOLVE-BLOCK-END
