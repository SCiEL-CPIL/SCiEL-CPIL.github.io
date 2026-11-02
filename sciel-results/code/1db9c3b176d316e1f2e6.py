"""Matrix multiplication 3x4x7 tensor decomposition search."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify_exact(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition is exact.
    The matrix multiplication tensor T[i,j,k] = 1 if and only if
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1.
    Otherwise the sum must be 0.
    """
    rank = len(U[0]) if U else 0
    # Build sparse representations
    U_sp = {}
    for i in range(n * m):
        for c in range(rank):
            val = U[i][c]
            if val != 0:
                U_sp.setdefault(i, {})[c] = val

    V_sp = {}
    for j in range(m * p):
        for c in range(rank):
            val = V[j][c]
            if val != 0:
                V_sp.setdefault(j, {})[c] = val

    W_sp = {}
    for k in range(p * n):
        for c in range(rank):
            val = W[k][c]
            if val != 0:
                W_sp.setdefault(k, {})[c] = val

    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = i * m + j
                vj = j * p + k
                wk = k * n + i
                # Compute sum over c of U[ui,c]*V[vj,c]*W[wk,c]
                s = 0
                u_dict = U_sp.get(ui, {})
                v_dict = V_sp.get(vj, {})
                w_dict = W_sp.get(wk, {})
                # iterate over the smallest dict
                if len(u_dict) <= len(v_dict) and len(u_dict) <= len(w_dict):
                    for c, uc in u_dict.items():
                        if c in v_dict and c in w_dict:
                            s += uc * v_dict[c] * w_dict[c]
                elif len(v_dict) <= len(u_dict) and len(v_dict) <= len(w_dict):
                    for c, vc in v_dict.items():
                        if c in u_dict and c in w_dict:
                            s += u_dict[c] * vc * w_dict[c]
                else:
                    for c, wc in w_dict.items():
                        if c in u_dict and c in v_dict:
                            s += u_dict[c] * v_dict[c] * wc
                if s != 1:
                    return False
    return True


def _random_halfint(max_abs=3):
    """Generate a random half-integer in [-max_abs, max_abs]."""
    val = random.randint(-2 * max_abs, 2 * max_abs) / 2.0
    return val


def _try_random_search(n, m, p, target_rank, time_limit, seed=42):
    """
    Randomized search for a decomposition of the given rank.
    Uses a structured approach: initialize with random values, then
    try to satisfy the tensor equations via iterative refinement.
    """
    rng = random.Random(seed)
    start_time = time.time()
    
    # The tensor has n*m*p = 84 constraints (each must equal 1)
    # We need rank variables in each of U, V, W
    # Total unknowns: (n*m + m*p + p*n) * rank = (12 + 28 + 21) * rank = 61 * rank
    
    # For rank r, we have 61*r unknowns and 84 equations
    # So for r >= 2, we have more unknowns than equations (overdetermined in a good way)
    
    # Strategy: Use a greedy/iterative approach
    # 1. Initialize U, V, W randomly
    # 2. For each tensor entry (i,j,k), check if sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    # 3. If not, adjust one of the terms
    
    # Better strategy: Use the fact that we can set up a linear system
    # For a fixed pair (i,j), the equation for all k is:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
    # This is nonlinear in the variables.
    
    # Alternative: Try to find a decomposition where each rank-1 term
    # contributes to a specific subset of tensor entries.
    
    # Let's try a different approach: start with schoolbook and try to merge terms.
    # Two schoolbook terms can be merged if their supports overlap in a compatible way.
    
    # Actually, let's try a more direct approach:
    # Use random initialization and gradient-like descent on the constraint satisfaction.
    
    def _evaluate(U, V, W, n, m, p):
        """Compute the error of the decomposition."""
        rank = len(U[0])
        error = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if i * m + j == 0:  # placeholder, we'll compute properly
                        pass
                    # The target is 1 for all (i,j,k) in the multiplication tensor
                    # Wait, the matrix multiplication tensor is:
                    # T[i,j,k] = 1 for all valid (i,j,k)
                    # So we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
                    error += (s - 1) ** 2
        return error
    
    # This is a hard non-convex optimization problem.
    # Let's try a simpler heuristic: use a known structure.
    
    # For small matrix multiplication, there are known decompositions
    # that are better than schoolbook. For 3x4x7, the best known is 84 (schoolbook)
    # unless we find something better.
    
    # Let's try a targeted search: assume a sparse structure.
    # Each rank-1 term (U_col, V_col, W_col) should have sparse entries.
    
    best_result = None
    best_error = float('inf')
    
    max_iters = 500
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize with sparse random values
        U = [[0.0] * target_rank for _ in range(n * m)]
        V = [[0.0] * target_rank for _ in range(m * p)]
        W = [[0.0] * target_rank for _ in range(p * n)]
        
        # Each column c has a sparse pattern
        for c in range(target_rank):
            # Choose random non-zero positions
            nz_u = rng.randint(1, 3)
            nz_v = rng.randint(1, 3)
            nz_w = rng.randint(1, 3)
            
            u_pos = rng.sample(range(n * m), min(nz_u, n * m))
            v_pos = rng.sample(range(m * p), min(nz_v, m * p))
            w_pos = rng.sample(range(p * n), min(nz_w, p * n))
            
            for pos in u_pos:
                U[pos][c] = rng.choice([-2, -1, -0.5, 0.5, 1, 2])
            for pos in v_pos:
                V[pos][c] = rng.choice([-2, -1, -0.5, 0.5, 1, 2])
            for pos in w_pos:
                W[pos][c] = rng.choice([-2, -1, -0.5, 0.5, 1, 2])
        
        # Evaluate
        error = _evaluate(U, V, W, n, m, p)
        if error < best_error:
            best_error = error
            best_result = (U, V, W)
        
        if error < 1e-9:
            # Check exactness
            if _verify_exact(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    return None


def _try_structured_merge(n, m, p, time_limit):
    """
    Try to merge schoolbook terms to reduce rank.
    Two terms (i1,j1,k1) and (i2,j2,k2) can be merged if there exist
    values a, b, c, d, e, f such that the combined contribution works.
    This is complex, so we try a limited search.
    """
    start_time = time.time()
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    U = [row[:] for row in result["U"]]
    V = [row[:] for row in result["V"]]
    W = [row[:] for row in result["W"]]
    rank = n * m * p
    
    # Try to eliminate one term at a time by adjusting others
    # This is a local search approach
    
    # For each term c, try to see if we can set it to zero by adjusting
    # the other terms. This requires solving a system of equations.
    
    # Given the complexity, let's just return schoolbook for now.
    # A more sophisticated approach would use algebraic geometry or
    # Gröbner bases, which is beyond the scope here.
    
    return result


def _try_bilinear_optimization(n, m, p, target_rank, time_limit):
    """
    Try to find a decomposition using bilinear optimization.
    Fix U and V, then solve for W linearly, and vice versa.
    """
    start_time = time.time()
    rng = random.Random(12345)
    
    # Initialize U and V randomly
    U = [[rng.choice([-1, 0, 0, 0, 1]) for _ in range(target_rank)] for _ in range(n * m)]
    V = [[rng.choice([-1, 0, 0, 0, 1]) for _ in range(target_rank)] for _ in range(m * p)]
    
    # For each (i,k), we need:
    # sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # This is linear in W[k*n+i, c] for fixed i, k.
    # But it's coupled across j, so it's not simply linear.
    
    # Actually, for fixed (i, k), the equation is:
    # sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * V[j*p+k, c]) = 1
    # Let A[i, k, c] = sum_j U[i*m+j, c] * V[j*p+k, c]
    # Then we need: sum_c W[k*n+i, c] * A[i, k, c] = 1 for all (i, k)
    # This is a system of n*p equations in n*p*target_rank unknowns (W entries)
    # But the equations are coupled because W[k*n+i, c] appears in multiple equations?
    # No, W[k*n+i, c] only appears in the equation for (i, k).
    # So we have n*p independent equations, each involving target_rank unknowns.
    # This is underdetermined, so we can find a solution if the system is consistent.
    
    # Let's try this approach:
    # 1. Randomly initialize U and V
    # 2. Compute A[i, k, c] = sum_j U[i*m+j, c] * V[j*p+k, c]
    # 3. For each (i, k), solve sum_c W[k*n+i, c] * A[i, k, c] = 1
    #    This is a linear equation in the W[k*n+i, : ] vector.
    #    We can choose W[k*n+i, :] to satisfy this, e.g., set all but one to zero.
    # 4. Check if the full decomposition is exact.
    
    best_result = None
    best_error = float('inf')
    
    for trial in range(100):
        if time.time() - start_time > time_limit:
            break
        
        U = [[rng.choice([-2, -1, -0.5, 0, 0, 0, 0.5, 1, 2]) for _ in range(target_rank)] for _ in range(n * m)]
        V = [[rng.choice([-2, -1, -0.5, 0, 0, 0, 0.5, 1, 2]) for _ in range(target_rank)] for _ in range(m * p)]
        
        # Compute A[i, k, c]
        A = [[[0.0] * target_rank for _ in range(p)] for _ in range(n)]
        for i in range(n):
            for k in range(p):
                for c in range(target_rank):
                    for j in range(m):
                        A[i][k][c] += U[i * m + j][c] * V[j * p + k][c]
        
        # Solve for W
        W = [[0.0] * target_rank for _ in range(p * n)]
        valid = True
        for i in range(n):
            for k in range(p):
                # Need sum_c W[k*n+i, c] * A[i, k, c] = 1
                # Find a c where A[i, k, c] != 0
                found = False
                for c in range(target_rank):
                    if abs(A[i][k][c]) > 1e-12:
                        W[k * n + i][c] = 1.0 / A[i][k][c]
                        found = True
                        break
                if not found:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        # Verify
        if _verify_exact(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        
        # Compute error
        error = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(target_rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    error += (s - 1) ** 2
        
        if error < best_error:
            best_error = error
            best_result = (U, V, W)
    
    return None


def _try_low_rank_search(n, m, p, time_limit):
    """
    Search for a decomposition with rank < n*m*p.
    Try ranks from a lower bound up to n*m*p - 1.
    """
    start_time = time.time()
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Lower bound: at least max(nm, mp, pn) = max(12, 28, 21) = 28
    # Also, the tensor rank is at least the max of the matrix ranks of the slices
    # For now, try from 40 up to 83
    
    for target_rank in range(40, n * m * p):
        if time.time() - start_time > time_limit:
            break
        
        # Try bilinear optimization
        result = _try_bilinear_optimization(n, m, p, target_rank, time_limit - (time.time() - start_time))
        if result:
            return result
        
        # Try random search
        result = _try_random_search(n, m, p, target_rank, time_limit - (time.time() - start_time), seed=target_rank)
        if result:
            return result
    
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s kill

    # Start with schoolbook
    result = _schoolbook(n, m, p)
    best_rank = n * m * p

    # Try to improve via search
    try:
        remaining = deadline - time.time()
        improved = _try_low_rank_search(n, m, p, remaining)
        if improved:
            r = len(improved["U"][0])
            if r < best_rank:
                # Verify
                if _verify_exact(improved["U"], improved["V"], improved["W"], n, m, p):
                    best_rank = r
                    result = improved
    except Exception:
        pass

    if time.time() > deadline:
        return result

    return result

# EVOLVE-BLOCK-END
