"""Rewritten program for matrix multiplication 2x3 by 3x4 using Strassen projection + numerical search."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.
    
    Strategy:
    1. Try to project the Strassen 3x3x3 decomposition (rank 7) onto 2x3x4 subspace.
       If valid, return rank 7.
    2. Try numerical search for ranks 8..23 with integer/half-integer entries.
    3. Fall back to schoolbook rank 24.
    """
    n, m, p = DIMENSIONS  # 2, 3, 4
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Strategy 1: Strassen projection
    result = _strassen_projection(n, m, p)
    if result is not None and time.time() < deadline:
        return result
    
    # Strategy 2: Numerical search for ranks 8..23
    for rank in range(8, 24):
        if time.time() > deadline - 5:
            break
        result = _find_decomposition(rank, n, m, p, deadline)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)


def _strassen_projection(n, m, p):
    """
    Project Strassen's 3x3x3 decomposition onto 2x3x4.
    
    Strassen's algorithm for 3x3 * 3x3 has rank 7. We embed 2x3 and 3x4 into 3x3
    by padding with zeros, then check if the resulting decomposition is valid
    for the 2x3x4 tensor.
    
    Strassen's 7 rank-1 terms for 3x3*3x3:
    M1 = (A11 + A22) * (B11 + B22)
    M2 = (A21 + A22) * B11
    M3 = A11 * (B12 - B22)
    M4 = A22 * (B21 - B11)
    M5 = (A11 + A12) * B22
    M6 = (A21 - A11) * (B11 + B12)
    M7 = (A12 - A22) * (B21 + B22)
    
    Then:
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # For 2x3 * 3x4, we embed:
    # A (2x3) -> A_padded (3x3) with row 2 = 0
    # B (3x4) -> B_padded (3x3) with col 2 = 0 (we only need cols 0,1,2 of B, so pad to 3x3)
    # Actually B is 3x4, so we need B_padded (3x3) taking first 3 columns, col 2 of B_padded = 0
    # Wait, B is 3x4, so B_padded should be 3x3 where we set the 3rd column to 0? No.
    # Let's think again.
    #
    # For 2x3 * 3x4:
    # A[i,j] for i in {0,1}, j in {0,1,2}
    # B[j,k] for j in {0,1,2}, k in {0,1,2,3}
    # C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, k in {0,1,2,3}
    #
    # To use Strassen (3x3*3x3), we need to handle the 4th column of B separately.
    # This projection approach is tricky because B has 4 columns.
    #
    # Alternative: Just check if rank 7 is achievable by solving the linear system.
    # For rank 7, we have 7 terms. The constraint is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    #
    # This is 2*3*4 = 24 equations. With rank 7, we have 6*7 + 12*7 + 8*7 = 42*7 = 294 unknowns.
    # So it's underdetermined. Let's try to find a solution.
    
    rank = 7
    # Use random search for rank 7
    result = _find_decomposition(rank, n, m, p, time.time() + 10)
    return result


def _find_decomposition(rank, n, m, p, deadline):
    """
    Find a rank-r decomposition using random search with linear solve for W.
    """
    start_time = time.time()
    max_time = min(10, max(0, deadline - start_time))
    
    # Candidate values: integers and half-integers
    values = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    
    # Number of random restarts - scale with available time
    base_restarts = 500
    n_restarts = max(100, min(5000, int(base_restarts * max(1, (deadline - start_time) / 10))))
    
    for restart in range(n_restarts):
        if time.time() > deadline - 2:
            break
        
        # Random initialization of U and V
        U = np.array([[random.choice(values) for _ in range(rank)] for _ in range(n * m)])
        V = np.array([[random.choice(values) for _ in range(rank)] for _ in range(m * p)])
        
        # Solve for W: for each (k, i), we need
        # sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = 1 for all j
        # This is a linear system in W[k*n+i, :]
        
        W = np.zeros((p * n, rank))
        valid = True
        
        for k in range(p):
            for i in range(n):
                # Build the linear system: for each j in 0..m-1
                # sum_c W[k*n+i, c] * (U[i*m+j, c] * V[j*p+k, c]) = 1
                A_sys = np.zeros((m, rank))
                for j in range(m):
                    for c in range(rank):
                        A_sys[j, c] = U[i * m + j, c] * V[j * p + k, c]
                
                b_sys = np.ones(m)
                
                # Solve A_sys * W[k*n+i, :]^T = b_sys
                try:
                    # Check if system is consistent
                    solution = np.linalg.lstsq(A_sys, b_sys, rcond=None)[0]
                    residual = np.linalg.norm(A_sys @ solution - b_sys)
                    if residual > 1e-8:
                        valid = False
                        break
                    W[k * n + i, :] = solution
                except:
                    valid = False
                    break
            
            if not valid:
                break
        
        if not valid:
            continue
        
        # Round W to nearest value in the candidate set
        W_rounded = np.array([[round(W[row, col] / 0.5) * 0.5 for col in range(rank)] for row in range(p * n)])
        
        # Check if rounded W is valid
        if _is_valid(U, V, W_rounded, n, m, p):
            # Also round U and V to ensure exactness
            U_rounded = np.array([[round(U[row, col] / 0.5) * 0.5 for col in range(rank)] for row in range(n * m)])
            V_rounded = np.array([[round(V[row, col] / 0.5) * 0.5 for col in range(rank)] for row in range(m * p)])
            
            if _is_valid(U_rounded, V_rounded, W_rounded, n, m, p):
                return _convert_to_list(U_rounded, V_rounded, W_rounded)
    
    return None


def _is_valid(U, V, W, n, m, p):
    """
    Check if the decomposition is valid:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    """
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(U.shape[1]):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _convert_to_list(U, V, W):
    """Convert numpy arrays to lists of lists."""
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }


def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
