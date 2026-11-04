"""Matrix multiplication 4x4x5 - rank minimization via random search + verification."""
import numpy as np
import time
import itertools
import random

DIMENSIONS = [4, 4, 5]
n, m, p = DIMENSIONS
# U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
# Tensor: C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
# Rank-r decomposition: T_{i,j,k} = sum_{r=0}^{R-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]


def verify_decomposition(U, V, W):
    """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == (i==j or not) exactly.
    The matrix multiplication tensor T[i,j,k] = delta(i,k) if we think of it as
    C[i,k] = sum_j A[i,j] B[j,k]. But the tensor has entries T[i,j,k] where
    for each (i,k) the coefficient of A[i,j]*B[j,k] is 1.
    Actually: T_{i,j,k} = 1 always (the bilinear map). Wait no.
    
    The tensor for matmul A(4x4) * B(4x5) -> C(4x5):
    C[i,k] = sum_j A[i,j] * B[j,k]
    So the trilinear tensor has T[i,j,k] = 1 for all valid (i,j,k) since every
    term A[i,j]*B[j,k] contributes to C[i,k] with coefficient 1.
    
    Rank-r decomposition: 1 = sum_{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for all i in [0,3], j in [0,3], k in [0,4].
    """
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(len(U[i*m+j])):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1.0:
                    return False
    return True


def verify_decomposition_fast(U, V, W):
    """Vectorized verification using numpy."""
    R = len(U[0])
    U_arr = np.array(U)  # (n*m, R)
    V_arr = np.array(V)  # (m*p, R)
    W_arr = np.array(W)  # (p*n, R)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def random_half_integer_matrix(rows, cols, scale=3):
    """Generate random matrix with entries from {-scale, ..., scale} * 0.5"""
    # Use values that are integers or half-integers
    vals = np.arange(-2*scale, 2*scale+1) * 0.5
    return np.random.choice(vals, size=(rows, cols)).tolist()


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin before 150s limit
    
    np.random.seed(42)
    random.seed(42)
    
    # Start from schoolbook
    rank_schoolbook = n * m * p
    U_sch = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sch = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sch = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i*m+j][col] = 1
                V_sch[j*p+k][col] = 1
                W_sch[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": U_sch, "V": V_sch, "W": W_sch}
    best_rank = rank_schoolbook
    
    # Try to find lower rank via random search
    # Strategy: for each candidate rank R, try to find U, V, W with random entries
    # such that the tensor equation holds.
    
    # Lower bound: rank >= max(n*p, m*p, n*m) = max(20, 20, 16) = 20
    # Known results: 4x4x4 has rank 49 (Strassen improved to 49), 4x4x5 unknown
    # Let's try various ranks from low to high
    
    candidate_ranks = list(range(20, rank_schoolbook + 1))
    
    # For efficiency, use a structured random approach
    # We need: for all (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # Approach 1: Fix U and V randomly, solve for W
    # For each (k, i), W[k*n+i, :] must satisfy:
    #   sum_r V[j*p+k,r] * W[k*n+i,r] = 1 / U[i*m+j,r]  ... no, that's not linear in W
    # Actually: sum_r (U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r] = 1
    # For fixed (i,j,k), this is one linear equation in W[k*n+i, :].
    # But W[k*n+i, :] appears in equations for all j. So:
    # For fixed (k,i) and varying j: sum_r (U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r] = 1
    # This gives m equations (one per j) in p*n * R unknowns... no, in R unknowns per (k,i).
    # Actually for fixed (k,i): the unknowns are W[k*n+i, 0..R-1] (R unknowns).
    # Constraints: for each j in 0..m-1: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # That's m linear equations in R unknowns. If R >= m, we might have a solution.
    
    # Let's use this approach: random U, V, then solve for W
    attempts_per_rank = 500
    
    for R in candidate_ranks:
        if time.time() > deadline - 10:
            break
        
        found = False
        for attempt in range(attempts_per_rank):
            if time.time() > deadline - 5:
                found = True  # break out
                break
            
            # Random U and V with small integer/half-integer entries
            U = np.random.randint(-2, 3, size=(n*m, R)).astype(float)
            V = np.random.randint(-2, 3, size=(m*p, R)).astype(float)
            
            # Make sure no entire row is zero
            for idx in range(n*m):
                if np.all(U[idx] == 0):
                    U[idx, np.random.randint(R)] = 1
            for idx in range(m*p):
                if np.all(V[idx] == 0):
                    V[idx, np.random.randint(R)] = 1
            
            # For each (k, i), solve for W[k*n+i, :]
            W = np.zeros((p*n, R))
            valid = True
            for k in range(p):
                for i in range(n):
                    # Build system: for each j, sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
                    A_sys = np.zeros((m, R))
                    for j in range(m):
                        A_sys[j] = U[i*m+j] * V[j*p+k]
                    
                    # Solve A_sys @ W_row = [1,1,...,1]
                    try:
                        # Least squares or exact solve
                        W_row, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, np.ones(m), rcond=None)
                        # Check if it's an exact solution
                        if np.allclose(A_sys @ W_row, np.ones(m), atol=1e-8):
                            W[k*n+i] = W_row
                        else:
                            valid = False
                            break
                    except:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Round W to half-integers if close
            W_rounded = np.round(W * 2) / 2.0
            
            # Verify with rounded values
            U_list = U.astype(int).tolist() if np.allclose(U, np.round(U)) else (np.round(U*2)/2).tolist()
            V_list = V.astype(int).tolist() if np.allclose(V, np.round(V)) else (np.round(V*2)/2).tolist()
            W_list = (np.round(W_rounded*2)/2).tolist()
            
            # Verify
            ok = True
            U_arr = np.array(W_list) if False else np.array([list(map(float, x)) for x in W_list])
            U_arr = np.array([list(map(float, x)) for x in U_list])
            V_arr = np.array([list(map(float, x)) for x in V_list])
            W_arr = np.array([list(map(float, x)) for x in W_list])
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                        if abs(s - 1.0) > 1e-6:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                best_rank = R
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                found = True
                # Try even lower
                if R > 20:
                    break
                else:
                    # Found at this rank, try lower ranks
                    break
        
        if found and R < best_rank:
            # Already updated
            pass
    
    return best_result
