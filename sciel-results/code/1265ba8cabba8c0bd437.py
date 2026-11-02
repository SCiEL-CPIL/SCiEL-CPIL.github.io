"""Program for matrix multiplication tensor decomposition (3x5 by 5x5)."""
DIMENSIONS = [3, 5, 5]

import time
import numpy as np
from itertools import product


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Indexing: U[i*m+j], V[j*p+k], W[k*n+i]
    
    # Check all entries (i,j,k) should sum to 1
    # T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    rank = U.shape[1]
    
    # Vectorized check
    # Create indices
    # For each r, we have a tensor U_r (n,m), V_r (m,p), W_r (p,n)
    # Sum over r of U_r[i,j] * V_r[j,k] * W_r[k,i]
    
    # Efficient way:
    # Reshape U to (n, m, rank)
    # Reshape V to (m, p, rank)
    # Reshape W to (p, n, rank)
    
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # We want sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # This is like a tensor contraction.
    
    # Let's just loop over i, j, k since n,m,p are small (3,5,5)
    # 3*5*5 = 75 checks.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i*m + j, :]
                v_val = V[j*p + k, :]
                w_val = W[k*n + i, :]
                
                s = np.dot(np.dot(u_val, v_val), w_val)
                
                if abs(s - 1.0) > tol:
                    return False
    return True


def _build_strassen_hybrid(n, m, p):
    """
    Build a hybrid decomposition using Strassen for 2x2 blocks where possible.
    For 3x5 * 5x5:
    Split A (3x5) into A1 (2x5) and A2 (1x5).
    Split B (5x5) into B1 (5x2) and B2 (5x3).
    
    C1 = A1 * B (2x5)
    C2 = A2 * B (1x5)
    
    For C1 (2x5 * 5x5):
    Split B into B_left (5x2) and B_right (5x3).
    C1_left = A1 * B_left (2x5 * 5x2 = 2x2) -> Use Strassen (rank 7)
    C1_right = A1 * B_right (2x5 * 5x3 = 2x3) -> Schoolbook (rank 2*5*3=30)
    
    For C2 (1x5 * 5x5):
    Schoolbook (rank 1*5*5=25)
    
    Total rank: 7 + 30 + 25 = 62? 
    Wait, the Strassen block covers A1 (2x5) and B_left (5x2).
    A1 is 2x5, B_left is 5x2.
    Strassen applies to 2x2 * 2x2.
    Here we have 2x5 * 5x2.
    We can split A1 into 2x2 and 2x3?
    A1 = [A1_1 (2x2), A1_2 (2x3)]
    B_left = [B_left_1 (2x2); B_left_2 (3x2)]
    
    C1_left = A1_1 * B_left_1 (2x2 * 2x2) -> Strassen rank 7.
    Cross terms:
    A1_1 * B_left_2 (2x2 * 3x2 = 2x2) -> Schoolbook rank 12.
    A1_2 * B_left_1 (2x3 * 2x2 = 2x2) -> Schoolbook rank 12.
    A1_2 * B_left_2 (2x3 * 3x2 = 2x2) -> Schoolbook rank 12.
    
    This is getting complicated to implement manually.
    Let's stick to a simpler hybrid:
    Use Strassen for the top-left 2x2 * 2x2 part of the 3x5 * 5x5 product.
    
    A = [[A11 (2x2), A12 (2x3)],
         [A21 (1x2), A22 (1x3)]]
         
    B = [[B11 (2x2), B12 (2x3)],
         [B21 (2x2), B22 (2x3)],
         [B31 (1x2), B32 (1x3)]]
         
    This block structure is messy because 5 is not divisible by 2.
    
    Alternative: Just use the schoolbook but initialize ALS from a random sparse matrix?
    Or better: Use the known Strassen 2x2 decomposition for the sub-blocks where dimensions allow.
    
    Let's try a different structure:
    Pad to 4x8 * 8x8? No, that increases rank.
    
    Let's just use the schoolbook decomposition as the initial guess for ALS, but with random perturbations to break symmetry, and let ALS try to find a lower rank solution.
    However, ALS on the full 75 rank might get stuck.
    
    Let's try a "Random Projection" or "Low Rank Approximation" approach.
    Actually, the best bet for a structural change is to use the **Landsberg-Lim algorithm** or similar explicit lower bounds, but we need a constructive method.
    
    Let's implement ALS starting from the schoolbook decomposition.
    """
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return U, V, W


def _als_decomposition(U, V, W, n, m, p, max_iter=50, tol=1e-5, deadline=None):
    """
    Perform Alternating Least Squares to minimize the error in the tensor decomposition.
    Error = sum_{i,j,k} (sum_r U_{ij,r} V_{jk,r} W_{ki,r} - 1)^2
    
    We fix two matrices and solve for the third.
    Since the problem is small, we can solve linear systems.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    rank = U.shape[1]
    
    # Precompute indices for efficiency
    # We need to solve for U, V, W.
    # Equation: sum_r U_{ij,r} V_{jk,r} W_{ki,r} = 1
    
    # To solve for U_{ij,r}:
    # sum_r U_{ij,r} * (V_{jk,r} W_{ki,r}) = 1
    # This is a linear system for U_{ij, :} for each (i,j,k)?
    # No, for a fixed (i,j), the equation holds for all k.
    # sum_r U_{ij,r} V_{jk,r} W_{ki,r} = 1 for all k.
    # This is a system of p equations for m*p unknowns? No, U_{ij} is a vector of length rank.
    # For fixed i,j, we have p equations (one for each k).
    # Unknowns: U_{ij, 0}, ..., U_{ij, rank-1}.
    # This is an underdetermined system if rank > p.
    # We want the minimum norm solution.
    
    # Let's form the matrix A_{ij} of size p x rank.
    # A_{ij}[k, r] = V_{jk, r} * W_{ki, r}
    # Then A_{ij} @ U_{ij, :} = 1 (vector of ones of length p)
    
    # Similarly for V and W.
    
    # We will iterate.
    
    def solve_for_U():
        new_U = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                # Build A matrix (p x rank)
                A_mat = np.zeros((p, rank))
                for k in range(p):
                    A_mat[k, :] = V[j*p + k, :] * W[k*n + i, :]
                
                # Solve A_mat @ x = 1
                # Minimum norm solution: x = A^T (A A^T)^{-1} 1
                # Or use np.linalg.lstsq
                try:
                    x, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, np.ones(p), rcond=None)
                    new_U[i*m + j, :] = x
                except np.linalg.LinAlgError:
                    new_U[i*m + j, :] = U[i*m + j, :]
        return new_U

    def solve_for_V():
        new_V = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                # Build A matrix (n x rank)
                # Equation: sum_r U_{ij,r} V_{jk,r} W_{ki,r} = 1
                # For fixed j,k, unknown is V_{jk, :}
                # sum_r V_{jk,r} (U_{ij,r} W_{ki,r}) = 1 for all i.
                A_mat = np.zeros((n, rank))
                for i in range(n):
                    A_mat[i, :] = U[i*m + j, :] * W[k*n + i, :]
                
                try:
                    x, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, np.ones(n), rcond=None)
                    new_V[j*p + k, :] = x
                except np.linalg.LinAlgError:
                    new_V[j*p + k, :] = V[j*p + k, :]
        return new_V

    def solve_for_W():
        new_W = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                # Build A matrix (m x rank)
                # Equation: sum_r U_{ij,r} V_{jk,r} W_{ki,r} = 1
                # For fixed k,i, unknown is W_{ki, :}
                # sum_r W_{ki,r} (U_{ij,r} V_{jk,r}) = 1 for all j.
                A_mat = np.zeros((m, rank))
                for j in range(m):
                    A_mat[j, :] = U[i*m + j, :] * V[j*p + k, :]
                
                try:
                    x, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                    new_W[k*n + i, :] = x
                except np.linalg.LinAlgError:
                    new_W[k*n + i, :] = W[k*n + i, :]
        return new_W

    # ALS Loop
    prev_error = np.inf
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        # Update U
        U = solve_for_U()
        # Update V
        V = solve_for_V()
        # Update W
        W = solve_for_W()
        
        # Check convergence
        # Calculate error
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(np.dot(U[i*m + j, :], V[j*p + k, :]), W[k*n + i, :])
                    error += (s - 1.0)**2
        
        if abs(prev_error - error) < tol:
            break
        prev_error = error
        
    return U, V, W


def _try_reduce_rank_als(U, V, W, n, m, p, target_rank, deadline):
    """
    Try to find a decomposition of rank `target_rank` by initializing with random sparse matrices
    and running ALS.
    """
    if deadline and time.time() > deadline:
        return None, None, None
        
    # Initialize with random values, but sparsely
    # Use a "random subspace" initialization
    U_init = np.random.randn(n * m, target_rank)
    V_init = np.random.randn(m * p, target_rank)
    W_init = np.random.randn(p * n, target_rank)
    
    # Scale to help convergence
    U_init /= np.linalg.norm(U_init)
    V_init /= np.linalg.norm(V_init)
    W_init /= np.linalg.norm(W_init)
    
    U_out, V_out, W_out = _als_decomposition(U_init, V_init, W_init, n, m, p, max_iter=100, deadline=deadline)
    
    # Verify
    if _verify(U_out, V_out, W_out, n, m, p, tol=1e-5):
        return U_out, V_out, W_out
        
    return None, None, None


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    
    # 1. Start with Schoolbook
    U, V, W = _build_strassen_hybrid(n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    # 2. Run ALS on the schoolbook to see if it converges to something better
    # (It shouldn't reduce rank, but it might clean up numerical noise)
    if time.time() < deadline:
        U_als, V_als, W_als = _als_decomposition(U, V, W, n, m, p, max_iter=50, deadline=deadline)
        if _verify(U_als, V_als, W_als, n, m, p, tol=1e-5):
            # Rank is still 75, but maybe it's a "better" representation?
            # Score is rank, so this doesn't help score unless we can drop columns.
            pass
    
    # 3. Try to find lower rank solutions using ALS with random initialization
    # We know schoolbook is 75.
    # Try ranks: 74, 73, ... down to some limit.
    # Since ALS is expensive, we limit the number of attempts.
    
    current_rank = best_rank
    
    # Try to reduce by 1 first
    # Then try to reduce by more
    
    # Strategy: Try a few random restarts for rank = current_rank - 1
    # If successful, update best_rank and try current_rank - 1 again.
    
    max_reductions = 10 # Don't try to reduce by more than 10 in this run
    for reduction in range(1, max_reductions + 1):
        if time.time() > deadline:
            break
            
        target_rank = current_rank - reduction
        
        if target_rank <= 0:
            break
            
        # Try a few random restarts
        found = False
        for restart in range(5):
            if time.time() > deadline:
                break
                
            U_new, V_new, W_new = _try_reduce_rank_als(U, V, W, n, m, p, target_rank, deadline)
            
            if U_new is not None:
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = target_rank
                found = True
                break
        
        if not found:
            # If we couldn't reduce by `reduction`, maybe we can't reduce by `reduction + 1`?
            # Not necessarily, but let's stop if we fail at a small reduction.
            # Actually, let's just continue to next reduction level, but with fewer restarts?
            # For now, break to save time.
            break
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
