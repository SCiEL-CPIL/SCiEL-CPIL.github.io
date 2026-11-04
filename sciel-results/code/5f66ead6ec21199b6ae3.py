import time
import itertools
import random
import numpy as np

DIMENSIONS = [3, 4, 5]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the tensor decomposition exactly reproduces matrix multiplication."""
    n, m, p = n, m, p
    rank = len(U[0]) if len(U) > 0 else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the multiplication tensor T[i,j,k] = (A*B)[i,k] coefficient for A[i,j]*B[j,k]
    # The decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k} (for all i,j,k)
    # Actually T_{i,j,k} should be 1 if the (i,k) output equals sum over j of A[i,j]*B[j,k]
    # So for fixed i,k: sum_j sum_r U[i*m+j,r]*V[j*p+k,r] should = 1 (the coefficient of output[i,k])
    # And for i != i': the contribution to wrong outputs must be 0.
    
    # The condition: for all i, j, k:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  (this is the standard tensor equation)
    # Wait, let me reconsider. The standard formulation:
    #   (A*B)_{i,k} = sum_j A_{i,j} * B_{j,k}
    #   = sum_r U_{i,j,r} * V_{j,k,r} * W_{i,k,r}
    # where W index is [k*n + i] per the problem statement "W[k*n+i, r]"
    
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for ALL i, j, k
    # Wait no - that's not right either. Let me re-read.
    # 
    # The tensor M_{i,j,k} = 1 for all i,j,k (since multiplication tensor has 1s everywhere)
    # The decomposition: M_{i,j,k} = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # 
    # So the check is: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1.0) > tol:
                    return False, (i, j, k, val)
    return True, None


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _try_integer_search(n, m, p, target_rank, deadline, seed_offset=0):
    """Try to find an integer-entry decomposition of given rank via iterative refinement."""
    rng = random.Random(42 + seed_offset)
    n, m, p = n, m, p
    nm, mp, pn = n*m, m*p, p*n
    
    # Start with random integer entries
    U = [[rng.randint(-2, 2) for _ in range(target_rank)] for _ in range(nm)]
    V = [[rng.randint(-2, 2) for _ in range(target_rank)] for _ in range(mp)]
    W = [[rng.randint(-2, 2) for _ in range(target_rank)] for _ in range(pn)]
    
    # We want sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 for all i,j,k
    # This is a system of n*m*p = 60 equations in n*m*rank + m*p*rank + p*n*rank = (nm+mp+pn)*rank variables
    # For rank r: (nm+mp+pn)*r variables, 60 equations. We need a solution.
    # 
    # Strategy: use Newton-like method or try structured decompositions.
    
    # Try a structured approach: use the known Strassen-like patterns
    # For 2x2x2, rank 7 works. Let's try to extend that pattern.
    
    # Alternative: solve as a least-squares problem iteratively
    # Fix two of the three matrices, solve for the third.
    
    def solve_for_W(U, V):
        """Given U and V, solve for W: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1"""
        # For each pair (i,k), we have: sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # Let A[i,k,j] = sum_r U[i*m+j,r]*V[j*p+k,r]
        # Then: sum_j A[i,k,j]*W[k*n+i,r]... 
        # Actually W[k*n+i, r] is indexed by r for fixed (k,i)
        # So: for fixed (i,k): sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # This is: sum_r (sum_j U[i*m+j,r]*V[j*p+k,r]) * W[k*n+i,r] = 1
        # Let B[i,k,r] = sum_j U[i*m+j,r]*V[j*p+k,r]
        # Then: sum_r B[i,k,r]*W[k*n+i,r] = 1
        # This gives n*p = 15 equations, each involving p*n*rank = 15*rank unknowns (but W[k*n+i,r] for fixed k,i gives rank unknowns)
        # Actually W[k*n+i, r] for r=0..rank-1 are rank unknowns per (k,i) pair
        # And we have 15 (k,i) pairs, each with rank unknowns -> 15*rank unknowns
        # 15 equations, 15*rank unknowns -> underdetermined for rank >= 1
        # We can set W[k*n+i, r] = B[i,k,r] / sum_s B[i,k,s]^2 (normalized) or use least squares
        
        # Better: set W[k*n+i, r] such that sum_r B[i,k,r]*W[k*n+i,r] = 1
        # Simplest: pick W[k*n+i, 0] = 1 / B[i,k,0] if B[i,k,0] != 0, else adjust
        # This is fragile. Let's use a different approach.
        
        W = [[0.0]*rank for _ in range(pn)]
        for i in range(n):
            for k in range(p):
                # B[i,k,r] = sum_j U[i*m+j,r] * V[j*p+k,r]
                B = [0.0]*rank
                for j in range(m):
                    for r in range(rank):
                        B[r] += U[i*m+j][r] * V[j*p+k][r]
                # Want sum_r B[r] * W[k*n+i, r] = 1
                s = sum(b*b for b in B)
                if s < 1e-15:
                    # Degenerate; set a default
                    W[k*n+i][0] = 1.0 if B[0] != 0 else 0
                else:
                    for r in range(rank):
                        W[k*n+i][r] = B[r] / s
                # Verify
                dot = sum(B[r]*W[k*n+i][r] for r in range(rank))
                if abs(dot - 1.0) > 1e-6:
                    # Adjust
                    err = 1.0 - dot
                    # Add to first component
                    W[k*n+i][0] += err / B[0] if B[0] != 0 else 0
        return W
    
    def solve_for_U(V, W):
        """Given V and W, solve for U."""
        # sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # For fixed (i,j): sum_k sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # Let C[i,j,r] = sum_k V[j*p+k,r]*W[k*n+i,r]
        # Then: sum_r U[i*m+j,r]*C[i,j,r] = 1
        U = [[0.0]*rank for _ in range(nm)]
        for i in range(n):
            for j in range(m):
                C = [0.0]*rank
                for k in range(p):
                    for r in range(rank):
                        C[r] += V[j*p+k][r] * W[k*n+i][r]
                s = sum(c*c for c in C)
                if s < 1e-15:
                    U[i*m+j][0] = 1.0
                else:
                    for r in range(rank):
                        U[i*m+j][r] = C[r] / s
        return U
    
    def solve_for_V(U, W):
        """Given U and W, solve for V."""
        # For fixed (j,k): sum_i sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # Let D[j,k,r] = sum_i U[i*m+j,r]*W[k*n+i,r]
        # Then: sum_r V[j*p+k,r]*D[j,k,r] = 1
        V = [[0.0]*rank for _ in range(mp)]
        for j in range(m):
            for k in range(p):
                D = [0.0]*rank
                for i in range(n):
                    for r in range(rank):
                        D[r] += U[i*m+j][r] * W[k*n+i][r]
                s = sum(d*d for d in D)
                if s < 1e-15:
                    V[j*p+k][0] = 1.0
                else:
                    for r in range(rank):
                        V[j*p+k][r] = D[r] / s
        return V
    
    rank = target_rank
    # Initialize
    U = [[rng.randint(-1, 1) for _ in range(rank)] for _ in range(nm)]
    V = [[rng.randint(-1, 1) for _ in range(rank)] for _ in range(mp)]
    W = [[rng.randint(-1, 1) for _ in range(rank)] for _ in range(pn)]
    
    best_err = float('inf')
    best_solution = None
    
    for iteration in range(500):
        if time.time() > deadline - 15:
            break
        # Cyclic coordinate descent
        U_new = solve_for_U(V, W)
        V_new = solve_for_V(U_new, W)
        W_new = solve_for_W(U_new, V_new)
        
        U, V, W = U_new, V_new, W_new
        
        # Check error
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r] for r in range(rank))
                    max_err = max(max_err, abs(val - 1.0))
        
        if max_err < best_err:
            best_err = max_err
            best_solution = (
                [[x for x in row] for row in U],
                [[x for x in row] for row in V],
                [[x for x in row] for row in W]
            )
        
        if max_err < 1e-9:
            # Found! Round to rationals
            U_final = [[int(round(x)) if abs(x - round(x)) < 1e-6 else x for x in row] for row in U]
            V_final = [[int(round(x)) if abs(x - round(x)) < 1e-6 else x for x in row] for row in V]
            W_final = [[int(round(x)) if abs(x - round(x)) < 1e-6 else x for x in row] for row in W]
            
            # Verify with integer entries
            ok, _ = _verify(U_final, V_final, W_final, n, m, p, tol=1e-9)
            if ok:
                return {"U": U_final, "V": V_final, "W": W_final}
            
            # If rounding broke it, try as float (problem allows half-integers)
            ok, _ = _verify(U, V, W, n, m, p, tol=1e-9)
            if ok:
                return {"U": U, "V": V, "W": W}
        
        # Perturb periodically to escape local minima
        if iteration % 50 == 49 and max_err > 1e-4:
            for row in U:
                for r in range(rank):
                    row[r] += rng.gauss(0, 0.1)
            for row in V:
                for r in range(rank):
                    row[r] += rng.gauss(0, 0.1)
            for row in W:
                for r in range(rank):
                    row[r] += rng.gauss(0, 0.1)
    
    # If we got close, try rounding the best solution
    if best_err < 1e-3 and best_solution:
        U_b, V_b, W_b = best_solution
        # Try rounding to nearest half-integer
        U_r = [[int(round(x*2))/2 for x in row] for row in U_b]
        V_r = [[int(round(x*2))/2 for x in row] for row in V_b]
        W_r = [[int(round(x*2))/2 for x in row] for row in W_b]
        ok, _ = _verify(U_r, V_r, W_r, n, m, p, tol=1e-9)
        if ok:
            return {"U": U_r, "V": V_r, "W": W_r}
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # safety margin
    
    # Try progressively higher ranks (lower is better)
    # Lower bound: max(n*m, m*p, p*n) = 20
    # Try ranks from lower bound upward
    candidates = [20, 21, 22, 23, 24, 25, 26, 28, 30, 32, 36, 40, 48, 56, 60]
    
    for rank in candidates:
        if time.time() > deadline - 30:
            break
        # Try multiple random restarts
        for attempt in range(3):
            if time.time() > deadline - 15:
                break
            result = _try_integer_search(n, m, p, rank, deadline, seed_offset=rank*100+attempt)
            if result is not None:
                return result
    
    # Fallback: schoolbook
    return _schoolbook(n, m, p)
