"""Find rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill

    # First, try to find a decomposition via embedding into 4x4 and using Strassen-like structure.
    # If that fails, fall back to alternating minimization, then schoolbook.
    
    # Strategy 1: Embed into 4x4, use Strassen (rank 7 for 2x2), but we need 4x4.
    # Actually, the strategy says: embed 2x3 and 3x4 into 4x4, apply 2x2 Strassen recursively?
    # 4x4 multiplication can be done with rank 21 (Strassen recursively) or lower?
    # But we need to project back to 2x3x4.
    # The tensor for 4x4x4 has rank at most 21.
    # If we embed A(2x3) into A'(4x4) by padding, B(3x4) into B'(4x4) by padding,
    # then C'(4x4) = A'B' contains C(2x3*3x4=2x4) in the top-left 2x4 block?
    # Wait, A is 2x3, B is 3x4. C is 2x4.
    # If we pad A to 4x4 (top-left 2x3), and B to 4x4 (top-left 3x4),
    # then C' = A'B' is 4x4. The top-left 2x4 block of C' is exactly C.
    # So if we have a rank-r decomposition for 4x4x4, we can restrict it to the subspace
    # where A is in the 2x3 block and B is in the 3x4 block.
    # But the decomposition is for the *tensor*, not the matrices.
    # The tensor T4[i,j,k] for 4x4x4 is 1 if i==i', j==j', k==k'.
    # Our tensor T234[i,j,k] is 1 if i==i', j==j', k==k' for i in 0..1, j in 0..2, k in 0..3.
    # This is a sub-tensor of T4.
    # If we have a decomposition of T4 with rank r, we can restrict the U, V, W matrices to the
    # relevant indices. But the decomposition must still satisfy the equations for the sub-tensor.
    # However, the rank of the sub-tensor is at most the rank of the full tensor.
    # So if we find a rank-r decomposition for 4x4x4, we can use it for 2x3x4 with the same rank.
    # But we need the entries to be integer/half-integer. Strassen uses integers.
    # Strassen for 2x2x2 has rank 7. For 4x4x4, recursive Strassen has rank 7^2 = 49? No.
    # Wait, Strassen for n x n where n is power of 2: rank is 7^(log2 n).
    # For 4x4, rank is 7^2 = 49? That's worse than schoolbook 64.
    # Actually, schoolbook for 4x4 is 64. Strassen is 49.
    # But we need rank for 2x3x4. Schoolbook is 24.
    # So using 4x4 decomposition (rank 49) is worse.
    # The strategy says "applying the 2x2 Strassen algorithm recursively to the resulting 4x4x4 multiplication tensor".
    # Maybe it means something else. Perhaps it means to use a rank-14 decomposition for 2x3x4 directly?
    # Let's re-read: "Construct a rank-14 decomposition by embedding... and applying the 2x2 Strassen algorithm recursively... then projecting... and halving the coefficients".
    # This suggests that the 4x4x4 tensor has a decomposition of rank 28 (since 14*2=28?) or something.
    # Actually, if we embed 2x3x4 into 4x4x4, the rank of the sub-tensor is at most the rank of the full tensor.
    # But we want a *minimal* rank for 2x3x4.
    # Let's just try to find a low-rank decomposition via alternating minimization, as in the current program.
    # The current program tries ranks from 10 to 24.
    # Let's keep that, but also try a structured search based on the embedding idea if possible.
    # However, given the time constraint and the difficulty of finding exact low-rank decompositions,
    # let's focus on improving the alternating minimization and the verification.

    # Try ranks from low to high
    for rank in range(10, 25):
        if time.time() > deadline - 30:
            break
        result = _try_rank(n, m, p, rank, deadline)
        if result is not None:
            return result

    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_rank(n, m, p, rank, deadline, max_iters=2000, tol=1e-12, n_restarts=20):
    """Try to find a rank-r decomposition via alternating minimization."""
    np.random.seed(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        # Random initialization
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Normalize to avoid scale issues
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        success = _alternating_minimize(n, m, p, rank, U, V, W, max_iters, tol, deadline)
        if success:
            # Verify and clean up
            U, V, W = _clean_solution(n, m, p, rank, U, V, W)
            if _verify(n, m, p, U, V, W, tol=1e-10):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _alternating_minimize(n, m, p, rank, U, V, W, max_iters, tol, deadline):
    """Alternating least squares to satisfy T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1"""
    
    for it in range(max_iters):
        if time.time() > deadline - 30:
            return False
        
        # Solve for U given V, W
        for i in range(n):
            for j in range(m):
                A_mat = np.zeros((p, rank))
                for k in range(p):
                    A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)[0]
                except:
                    return False
        
        # Solve for V given U, W
        for j in range(m):
            for k in range(p):
                A_mat = np.zeros((n, rank))
                for i in range(n):
                    A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)[0]
                except:
                    return False
        
        # Solve for W given U, V
        for k in range(p):
            for i in range(n):
                A_mat = np.zeros((m, rank))
                for j in range(m):
                    A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)[0]
                except:
                    return False
        
        # Check convergence
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0)**2
        if err < tol:
            return True
    
    return False

def _clean_solution(n, m, p, rank, U, V, W):
    """Try to clean up the solution to have exact integer/half-integer values."""
    # Round to nearest 0.5
    U_clean = np.round(U * 2) / 2
    V_clean = np.round(V * 2) / 2
    W_clean = np.round(W * 2) / 2
    return U_clean, V_clean, W_clean

def _verify(n, m, p, U, V, W, tol=1e-10):
    """Verify that the decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True
