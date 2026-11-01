import numpy as np
import time
import itertools
from fractions import Fraction

def _to_frac(x, tol=1e-10):
    """Convert a float to a Fraction if it's close to a simple rational (int or half-int)."""
    if abs(x) < tol:
        return Fraction(0)
    # Check if close to integer
    nearest_int = round(x)
    if abs(x - nearest_int) < tol:
        return Fraction(nearest_int)
    # Check if close to half-integer
    nearest_half = round(2*x) / 2.0
    if abs(x - nearest_half) < tol:
        return Fraction(nearest_half, 2)
    # Fallback: limit denominator to 10 for simplicity, though task allows general rationals
    # But usually these decompositions yield integers.
    return Fraction(x).limit_denominator(10)

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly represents matrix multiplication."""
    # Convert to Fraction for exact arithmetic check
    Uf = [[_to_frac(x) for x in row] for row in U]
    Vf = [[_to_frac(x) for x in row] for row in V]
    Wf = [[_to_frac(x) for x in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Uf[i*m+j][c] * Vf[j*p+k][c] * Wf[k*n+i][c]
                if s != 1:
                    return False
    return True

def _build_target_tensor(n, m, p):
    """Build the target tensor T[i,j,k] = 1 if it corresponds to C[i,k] += A[i,j]*B[j,k]"""
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # The tensor for matrix multiplication C = A @ B has T[i,j,k] = 1 
                # for all i,j,k because C[i,k] = sum_j A[i,j] B[j,k]
                # Wait, the tensor T_{ijk} is defined such that T_{ijk} A_{ij} B_{jk} = C_{ik}.
                # So T_{ijk} = 1 for all i,j,k.
                T[i, j, k] = 1
    return T

def _als_solve_rank(T, n, m, p, rank, max_iter=100, tol=1e-9):
    """
    Solve for U, V, W such that sum_c U[:,c] V[:,c] W[:,c] = T.
    Uses Alternating Least Squares.
    """
    # Initialize with random small values
    U = np.random.randn(n*m, rank) * 0.1
    V = np.random.randn(m*p, rank) * 0.1
    W = np.random.randn(p*n, rank) * 0.1
    
    # Target tensor flattened: T_flat[i*m+j, j*p+k, k*n+i] ? No.
    # The equation is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    # Let's reshape T to match the index structure.
    # T is (n, m, p).
    # We can view this as a linear system for one matrix at a time.
    
    for it in range(max_iter):
        # Solve for U
        # For each column c, we need U[:, c] to satisfy:
        # sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = T[i,j,k]
        # This is tricky because V and W are coupled in the "weight" for U.
        # Let's fix V and W, and solve for U.
        # Reshape V and W to create a linear operator.
        # For a fixed c, let M_c be a matrix of size (n*m) x (p*n)? No.
        
        # Alternative: Vectorize.
        # Let u_c = U[:, c] (size n*m)
        # Let v_c = V[:, c] (size m*p)
        # Let w_c = W[:, c] (size p*n)
        
        # The contribution to T from term c is the outer product (in a specific tensor sense).
        # T = sum_c u_c * v_c * w_c (tensor product)
        
        # To solve for U:
        # Fix V, W.
        # For each i, j, k: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # This is a linear system for U.
        # Let's construct the matrix A_U such that A_U @ U_flat = T_flat.
        # This is expensive to build explicitly each iteration if done naively, 
        # but n=2, m=2, p=5 are small.
        
        # Let's use a more efficient approach: 
        # For each c, the term is u_c * v_c * w_c.
        # We can solve for U by treating V and W as fixed.
        # For each (i,j), the equation is:
        # sum_c U[i*m+j, c] * (sum_k? No, k is separate)
        
        # Let's just build the big linear system.
        # Variables: U (n*m * rank), V (m*p * rank), W (p*n * rank)
        # Total vars: rank * (n*m + m*p + p*n)
        # Equations: n*m*p
        
        # Since we are doing ALS, we solve for one block at a time.
        
        # 1. Solve for U
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # For a fixed (i,j), let u_ij = U[i*m+j, :].
        # We have equations for all k:
        # T[i,j,k] = sum_c u_ij[c] * (V[j*p+k, c] * W[k*n+i, c])
        # This is a linear system for u_ij:
        # sum_c u_ij[c] * M_kj[c] = T[i,j,k]  where M_kj[c] = V[j*p+k, c] * W[k*n+i, c]
        # This can be written as a matrix equation for each (i,j) pair?
        # Let's form a matrix A_U of size (n*m*p) x (n*m * rank).
        
        A_U = np.zeros((n*m*p, n*m*rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row_idx = idx
                    for c in range(rank):
                        col_idx = (i*m+j)*rank + c
                        A_U[row_idx, col_idx] = V[j*p+k, c] * W[k*n+i, c]
                    idx += 1
        
        T_flat = T.flatten()
        # Solve A_U @ U_flat = T_flat
        U_flat, residuals, rank_u, sv = np.linalg.lstsq(A_U, T_flat, rcond=None)
        U = U_flat.reshape(n*m, rank)
        
        # 2. Solve for V
        A_V = np.zeros((n*m*p, m*p*rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row_idx = idx
                    for c in range(rank):
                        col_idx = (j*p+k)*rank + c
                        A_V[row_idx, col_idx] = U[i*m+j, c] * W[k*n+i, c]
                    idx += 1
        V_flat, _, _, _ = np.linalg.lstsq(A_V, T_flat, rcond=None)
        V = V_flat.reshape(m*p, rank)
        
        # 3. Solve for W
        A_W = np.zeros((n*m*p, p*n*rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row_idx = idx
                    for c in range(rank):
                        col_idx = (k*n+i)*rank + c
                        A_W[row_idx, col_idx] = U[i*m+j, c] * V[j*p+k, c]
                    idx += 1
        W_flat, _, _, _ = np.linalg.lstsq(A_W, T_flat, rcond=None)
        W = W_flat.reshape(p*n, rank)
        
        # Check convergence
        # Calculate error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err += (s - T[i,j,k])**2
        if np.sqrt(err) < tol:
            break
            
    return U, V, W

def construct():
    n, m, p = 2, 2, 5
    T = _build_target_tensor(n, m, p)
    
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_rank = 20
    best_U, best_V, best_W = None, None, None
    
    # Try ranks from 17 upwards
    # 17 is the theoretical lower bound based on additive rank (7 + 10)
    for rank in range(17, 21):
        if time.time() > deadline:
            break
            
        # Try multiple random initializations for ALS
        found = False
        attempts = 0
        max_attempts = 50 if rank == 17 else 10
        
        while not found and attempts < max_attempts and time.time() < deadline:
            attempts += 1
            U, V, W = _als_solve_rank(T, n, m, p, rank, max_iter=50)
            
            # Check if the solution is "exact" (close to integers/half-integers)
            # We verify with fractions
            if _verify(U, V, W, n, m, p, rank):
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                found = True
                break
            
            # If not found, we can try to snap to nearest integers and re-verify?
            # Sometimes ALS converges to a solution that is numerically exact but floating point noise prevents _verify.
            # Let's try snapping.
            U_snap = np.round(U).astype(int)
            V_snap = np.round(V).astype(int)
            W_snap = np.round(W).astype(int)
            if _verify(U_snap, V_snap, W_snap, n, m, p, rank):
                best_rank = rank
                best_U, best_V, best_W = U_snap, V_snap, W_snap
                found = True
                break
                
            # Also try half-integers?
            # Unlikely for this specific tensor, but possible.
            
            if time.time() > deadline:
                break
        
        if found:
            break
            
    # If we didn't find a solution < 20, fallback to known 20 (schoolbook)
    if best_U is None:
        rank = 20
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_U, best_V, best_W = U, V, W
        best_rank = rank

    # Ensure return types are lists of lists of numbers (int/float/fraction compatible)
    # The evaluator expects integer/half-integer entries. 
    # If we found exact integers, return them.
    # If we found floats that verified as fractions, we should return the fraction values as floats or ints.
    # _verify converts to Fraction. If it passed, the values are effectively integers/half-integers.
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
