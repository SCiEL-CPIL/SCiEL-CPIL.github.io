"""
4x4x4 matrix multiplication tensor decomposition.
Uses Kronecker product of Strassen's rank-7 decomposition (rank 49)
with a rank-reduction pass via Gaussian elimination to remove
linearly dependent terms, aiming for rank < 49.
"""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from numpy.linalg import matrix_rank

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W as numpy arrays with shape (4, 7).
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 2: M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # Term 3: M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # Term 4: M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # Term 5: M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # Term 6: M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Term 7: M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_decomp(n, m, p, U, V, W, tol=1e-9):
    """Verify that sum_r U[a,r]*V[b,r]*W[c,r] matches the tensor."""
    T = _build_tensor(n, m, p)
    U_arr = np.asarray(U, dtype=float)
    V_arr = np.asarray(V, dtype=float)
    W_arr = np.asarray(W, dtype=float)
    T_decomp = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    max_diff = np.max(np.abs(T_decomp - T))
    return max_diff < tol, max_diff

def _kronecker_decomposition(n, m, p):
    """
    Construct the rank-49 decomposition for 4x4x4 using Kronecker product
    of the rank-7 Strassen decomposition for 2x2x2.
    Returns numpy arrays U, V, W of shape (16, 49).
    """
    U2, V2, W2 = _strassen_2x2()
    rank = 49
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(2):
                            i = 2 * i1 + i2
                            j = 2 * j1 + j2
                            k = 2 * k1 + k2
                            for r1 in range(7):
                                for r2 in range(7):
                                    r = r1 * 7 + r2
                                    U[i * m + j, r] = U2[i1 * 2 + j1, r1] * U2[i2 * 2 + j2, r2]
                                    V[j * p + k, r] = V2[j1 * 2 + k1, r1] * V2[j2 * 2 + k2, r2]
                                    W[k * n + i, r] = W2[k1 * 2 + i1, r1] * W2[k2 * 2 + i2, r2]
    
    return U, V, W

def _reduce_rank(U, V, W, n, m, p, tol=1e-9, max_iter=50):
    """
    Attempt to reduce the rank by identifying and removing linearly
    dependent terms. A term r is removable if there exist coefficients
    c_s for s != r such that:
        U[:,r] = sum_s c_s * U[:,s]
        V[:,r] = sum_s c_s * V[:,s]
        W[:,r] = sum_s c_s * W[:,s]
    We check this by seeing if the augmented matrix has the same rank
    as the base matrix for each of U, V, W simultaneously.
    
    Returns reduced U, V, W arrays.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    removed = [False] * rank
    removed_count = 0
    
    for iteration in range(max_iter):
        if time.time() > 130:  # Safety margin
            break
        
        found_removal = False
        for r in range(rank):
            if removed[r]:
                continue
            
            # Try to express column r as a linear combination of other non-removed columns
            other_cols = [s for s in range(rank) if not removed[s] and s != r]
            if len(other_cols) < 3:
                continue
            
            U_others = U[:, other_cols]  # (16, len(other_cols))
            V_others = V[:, other_cols]
            W_others = W[:, other_cols]
            
            # Check if U[:,r] is in span of U_others
            U_aug = np.column_stack([U_others, U[:, r]])
            U_rank = matrix_rank(U_others, tol=tol)
            U_aug_rank = matrix_rank(U_aug, tol=tol)
            if U_aug_rank > U_rank:
                continue  # Not in span, can't remove
            
            # Check if V[:,r] is in span of V_others
            V_aug = np.column_stack([V_others, V[:, r]])
            V_rank = matrix_rank(V_others, tol=tol)
            V_aug_rank = matrix_rank(V_aug, tol=tol)
            if V_aug_rank > V_rank:
                continue
            
            # Check if W[:,r] is in span of W_others
            W_aug = np.column_stack([W_others, W[:, r]])
            W_rank = matrix_rank(W_others, tol=tol)
            W_aug_rank = matrix_rank(W_aug, tol=tol)
            if W_aug_rank > W_rank:
                continue
            
            # All three are in their respective spans; find coefficients
            # Solve least squares for U
            try:
                c_U, _, _, _ = np.linalg.lstsq(U_others, U[:, r], rcond=tol)
                c_V, _, _, _ = np.linalg.lstsq(V_others, V[:, r], rcond=tol)
                c_W, _, _, _ = np.linalg.lstsq(W_others, W[:, r], rcond=tol)
            except Exception:
                continue
            
            # Verify the coefficients work for all three
            if (np.max(np.abs(U_others @ c_U - U[:, r])) > 1e-6 or
                np.max(np.abs(V_others @ c_V - V[:, r])) > 1e-6 or
                np.max(np.abs(W_others @ c_W - W[:, r])) > 1e-6):
                continue
            
            # Check if the same coefficients work (they should if the decomposition is valid)
            # Actually, we need the SAME c for all three. Let's check if c_U ≈ c_V ≈ c_W
            if not (np.max(np.abs(c_U - c_V)) < 1e-4 and np.max(np.abs(c_U - c_W)) < 1e-4):
                # Try a joint least squares approach
                # Stack all three equations
                A_stack = np.vstack([U_others, V_others, W_others])
                b_stack = np.concatenate([U[:, r], V[:, r], W[:, r]])
                try:
                    c_joint, _, _, _ = np.linalg.lstsq(A_stack, b_stack, rcond=tol)
                except Exception:
                    continue
                
                if (np.max(np.abs(U_others @ c_joint - U[:, r])) > 1e-5 or
                    np.max(np.abs(V_others @ c_joint - V[:, r])) > 1e-5 or
                    np.max(np.abs(W_others @ c_joint - W[:, r])) > 1e-5):
                    continue
                
                c = c_joint
            else:
                c = c_U
            
            # Remove column r by subtracting its contribution from other columns
            for idx, s in enumerate(other_cols):
                U[:, s] -= c[idx] * U[:, r]
                V[:, s] -= c[idx] * V[:, s]
                W[:, s] -= c[idx] * W[:, r]
            
            removed[r] = True
            removed_count += 1
            found_removal = True
            break  # Restart scan after modification
    
        if not found_removal:
            break
    
    # Build reduced matrices
    keep = [s for s in range(rank) if not removed[s]]
    U_reduced = U[:, keep]
    V_reduced = V[:, keep]
    W_reduced = W[:, keep]
    
    return U_reduced, V_reduced, W_reduced

def _schoolbook_decomposition(n, m, p):
    """Construct the rank-64 schoolbook decomposition for n x m x p."""
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

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s deadline
    
    n, m, p = DIMENSIONS
    
    # Step 1: Build Kronecker-Strassen rank-49 decomposition
    try:
        U, V, W = _kronecker_decomposition(n, m, p)
        
        # Verify the base decomposition
        valid, max_diff = _verify_decomp(n, m, p, U, V, W, tol=1e-9)
        if not valid:
            # Fallback to schoolbook
            result = _schoolbook_decomposition(n, m, p)
            return result
        
        # Step 2: Attempt rank reduction
        if time.time() < deadline - 20:  # Need at least 20s remaining
            U_red, V_red, W_red = _reduce_rank(U, V, W, n, m, p, tol=1e-9, max_iter=50)
            
            # Verify reduced decomposition
            new_rank = U_red.shape[1]
            if new_rank < 49:
                valid_red, max_diff_red = _verify_decomp(n, m, p, U_red, V_red, W_red, tol=1e-9)
                if valid_red:
                    return {
                        "U": U_red.tolist(),
                        "V": V_red.tolist(),
                        "W": W_red.tolist()
                    }
        
        # Return the rank-49 decomposition
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    except Exception:
        pass
    
    # Final fallback: schoolbook
    result = _schoolbook_decomposition(n, m, p)
    return result

# EVOLVE-BLOCK-END
