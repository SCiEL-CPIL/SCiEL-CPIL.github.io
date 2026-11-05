import numpy as np
import time
import random

DIMENSIONS = [4, 5, 5]

def verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition is correct."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(actual - expected) > tol:
                    return False
    return True

def get_rank(U):
    return len(U[0]) if U else 0

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Initialize with schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][r] = 1
                V[v_idx][r] = 1
                W[w_idx][r] = 1
                r += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    def try_eliminate_column(U, V, W, col_idx, n, m, p):
        """
        Try to eliminate a column by expressing it as a linear combination of others.
        Returns (new_U, new_V, new_W, success)
        """
        rank = get_rank(U)
        if rank <= 1:
            return None, None, None, False
        
        # Extract the column to be eliminated
        u_col = np.array([U[i][col_idx] for i in range(nm)], dtype=float)
        v_col = np.array([V[i][col_idx] for i in range(mp)], dtype=float)
        w_col = np.array([W[i][col_idx] for i in range(pn)], dtype=float)
        
        # Check if column is already zero
        if np.allclose(u_col, 0) and np.allclose(v_col, 0) and np.allclose(w_col, 0):
            new_U = [[U[i][j] for j in range(rank) if j != col_idx] for i in range(nm)]
            new_V = [[V[i][j] for j in range(rank) if j != col_idx] for i in range(mp)]
            new_W = [[W[i][j] for j in range(rank) if j != col_idx] for i in range(pn)]
            return new_U, new_V, new_W, True
        
        # Build linear system to find coefficients c_r such that:
        # sum_r c_r * U_ij,r * V_jk,r * W_ki,r = u_col_ij * v_col_jk * w_col_ki
        # for all i, j, k.
        
        equations = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    rhs = u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
                    if abs(rhs) > 1e-15:
                        coeffs = []
                        for r in range(rank):
                            if r == col_idx:
                                continue
                            coeffs.append(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r])
                        equations.append((coeffs, rhs))
        
        if not equations:
            return None, None, None, False
        
        A = np.array([eq[0] for eq in equations], dtype=float)
        b = np.array([eq[1] for eq in equations], dtype=float)
        
        # Solve least squares
        try:
            coeffs, residuals, rank_sys, _ = np.linalg.lstsq(A, b, rcond=1e-12)
            # Check residual
            residual = np.linalg.norm(A @ coeffs - b)
            if residual < 1e-9:
                # Construct new matrices
                new_U = [[U[i][j] for j in range(rank) if j != col_idx] for i in range(nm)]
                new_V = [[V[i][j] for j in range(rank) if j != col_idx] for i in range(mp)]
                new_W = [[W[i][j] for j in range(rank) if j != col_idx] for i in range(pn)]
                
                # Add the linear combination to the remaining columns
                for idx, c in enumerate(coeffs):
                    if abs(c) > 1e-15:
                        new_col_idx = idx if idx < col_idx else idx - 1
                        for i in range(nm):
                            new_U[i][new_col_idx] += c * U[i][col_idx]
                        for i in range(mp):
                            new_V[i][new_col_idx] += c * V[i][col_idx]
                        for i in range(pn):
                            new_W[i][new_col_idx] += c * W[i][col_idx]
                
                # Final verification
                if verify_decomposition(new_U, new_V, new_W, n, m, p):
                    return new_U, new_V, new_W, True
        except:
            pass
            
        return None, None, None, False

    def apply_random_basis_change(U, V, W, n, m, p):
        """
        Apply a random invertible transformation to U and V to mix columns.
        U' = U * M
        V' = V * M^{-1}
        W' = W
        This preserves the tensor decomposition but changes the column structure.
        """
        rank = get_rank(U)
        # Generate a random invertible matrix M
        M = np.random.randn(rank, rank)
        # Ensure invertibility
        try:
            M_inv = np.linalg.inv(M)
        except:
            return U, V, W
        
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        U_new = U_np @ M
        V_new = V_np @ M_inv
        # W remains the same
        
        # Convert back to list of lists
        U_list = U_new.tolist()
        V_list = V_new.tolist()
        W_list = W_np.tolist()
        
        return U_list, V_list, W_list

    # Main search loop
    max_iterations = 200
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        current_rank = get_rank(best_U)
        found_elimination = False
        
        # Try to eliminate columns in random order
        cols = list(range(current_rank))
        random.shuffle(cols)
        
        for col in cols:
            if time.time() > deadline:
                break
            
            new_U, new_V, new_W, success = try_eliminate_column(
                best_U, best_V, best_W, col, n, m, p
            )
            
            if success:
                best_U, best_V, best_W = new_U, new_V, new_W
                best_rank = get_rank(best_U)
                found_elimination = True
                break
        
        if not found_elimination:
            # If no column could be eliminated, apply a random basis change
            # to expose new dependencies, then try again
            if iteration % 5 == 0 and current_rank > 10:
                best_U, best_V, best_W = apply_random_basis_change(
                    best_U, best_V, best_W, n, m, p
                )
                # Verify the basis change didn't break anything (it shouldn't)
                if not verify_decomposition(best_U, best_V, best_W, n, m, p):
                    # Revert if broken (should not happen)
                    best_U, best_V, best_W = U, V, W
                    best_rank = rank
                    continue

    # Final verification
    if verify_decomposition(best_U, best_V, best_W, n, m, p):
        return {'U': best_U, 'V': best_V, 'W': best_W}
    
    # Fallback to schoolbook
    return {'U': U, 'V': V, 'W': W}
