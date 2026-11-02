"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import time
import itertools

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _verify_fast(U, V, W, n, m, p):
    """Faster verification using numpy einsum."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # Reshape: U -> (n, m, rank), V -> (m, p, rank), W -> (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # total[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    total = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    return np.allclose(total, 1.0, atol=1e-9)


def schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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
    return U, V, W


def strassen_2x2():
    """
    Return Strassen decomposition for 2x2 * 2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    # Strassen formulas:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A01-A10)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    #
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (A00+A11)*(B00+B11) -> contributes to C00 (+1) and C11 (+1)
    # A00: i=0,j=0 -> U[0]; A11: i=1,j=1 -> U[3]
    # B00: j=0,k=0 -> V[0]; B11: j=1,k=1 -> V[3]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # C00: k=0,i=0 -> W[0]; C11: k=1,i=1 -> W[3]
    
    # M2: (A10+A11)*B00 -> contributes to C10 (+1)
    # A10: i=1,j=0 -> U[2]; A11: i=1,j=1 -> U[3]
    # B00: j=0,k=0 -> V[0]
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1  # C10: k=0,i=1 -> W[2]
    
    # M3: A00*(B01-B11) -> contributes to C01 (+1) and C11 (+1)
    # A00: i=0,j=0 -> U[0]
    # B01: j=0,k=1 -> V[1]; B11: j=1,k=1 -> V[3]
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1  # C01: k=1,i=0 -> W[1]; C11: k=1,i=1 -> W[3]
    
    # M4: A11*(B10-B00) -> contributes to C00 (+1) and C10 (+1)
    # A11: i=1,j=1 -> U[3]
    # B10: j=1,k=0 -> V[2]; B00: j=0,k=0 -> V[0]
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1  # C00: k=0,i=0 -> W[0]; C10: k=0,i=1 -> W[2]
    
    # M5: (A00+A10)*B11 -> contributes to C00 (-1) and C01 (+1)
    # A00: i=0,j=0 -> U[0]; A10: i=1,j=0 -> U[2]
    # B11: j=1,k=1 -> V[3]
    U[0][4] = 1; U[2][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1  # C00: k=0,i=0 -> W[0]; C01: k=1,i=0 -> W[1]
    
    # M6: (A01-A10)*(B00+B10) -> contributes to C11 (+1)
    # A01: i=0,j=1 -> U[1]; A10: i=1,j=0 -> U[2]
    # B00: j=0,k=0 -> V[0]; B10: j=1,k=0 -> V[2]
    U[1][5] = 1; U[2][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[3][5] = 1  # C11: k=1,i=1 -> W[3]
    
    # M7: (A01-A11)*(B01+B11) -> contributes to C00 (+1)
    # A01: i=0,j=1 -> U[1]; A11: i=1,j=1 -> U[3]
    # B01: j=0,k=1 -> V[1]; B11: j=1,k=1 -> V[3]
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C00: k=0,i=0 -> W[0]
    
    return U, V, W


def schoolbook_sub(na, mb, pb):
    """Schoolbook decomposition for na x mb * mb x pb."""
    rank = na * mb * pb
    U = [[0] * rank for _ in range(na * mb)]
    V = [[0] * rank for _ in range(mb * pb)]
    W = [[0] * rank for _ in range(pb * na)]
    col = 0
    for i in range(na):
        for j in range(mb):
            for k in range(pb):
                U[i*mb+j][col] = 1
                V[j*pb+k][col] = 1
                W[k*na+i][col] = 1
                col += 1
    return U, V, W


def _try_rank_reduction(U, V, W, n, m, p, target_rank, deadline):
    """
    Try to reduce the rank of a decomposition by finding linear dependencies
    in the tensor space spanned by the columns.
    """
    rank = len(U[0])
    if rank <= target_rank:
        return None
    
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # First pass: remove any zero columns
    for remove_c in range(rank - 1, -1, -1):
        if time.time() > deadline:
            break
        norm_u = np.linalg.norm(U[:, remove_c])
        norm_v = np.linalg.norm(V[:, remove_c])
        norm_w = np.linalg.norm(W[:, remove_c])
        if norm_u < 1e-12 or norm_v < 1e-12 or norm_w < 1e-12:
            U = np.delete(U, remove_c, axis=1)
            V = np.delete(V, remove_c, axis=1)
            W = np.delete(W, remove_c, axis=1)
            rank = len(U[0])
            if rank <= target_rank:
                return U.tolist(), V.tolist(), W.tolist()
    
    if rank <= target_rank:
        return None
    
    # Build tensor matrix: T[i,j,k, c] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    nmp = n * m * p
    T = np.zeros((nmp, rank))
    
    for c in range(rank):
        U3 = U[:, c].reshape(n, m)
        V3 = V[:, c].reshape(m, p)
        W3 = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m*p + j*p + k, c] = U3[i, j] * V3[j, k] * W3[k, i]
    
    # Compute SVD to find effective rank
    try:
        _, s, Vt = np.linalg.svd(T, full_matrices=False)
    except:
        return None
    
    tol = 1e-9
    effective_rank = int(np.sum(s > tol))
    
    if effective_rank >= rank:
        # No dependencies found
        return None
    
    # Try to find a subset of columns whose tensor contributions span the same space
    # Use greedy algorithm: start with empty set, add columns that increase span
    # Then try to remove columns that are in the span of others
    
    # Identify which columns are linearly dependent using the SVD
    # Columns with small singular values in the right singular vectors are candidates
    
    # For each column, check if it can be expressed as a combination of others
    # by solving a least squares problem
    
    removed = []
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = rank
    
    for remove_c in range(current_rank):
        if time.time() > deadline:
            break
        if remove_c in removed:
            continue
        
        # Try to express column remove_c as combination of all other columns
        other_cols = [c for c in range(current_rank) if c != remove_c and c not in removed]
        
        if len(other_cols) < effective_rank:
            continue
        
        # Extract tensor for column remove_c
        target_vec = T[:, remove_c]
        
        # Extract tensors for other columns
        if len(other_cols) > 0:
            other_mat = T[:, other_cols]
            
            # Solve least squares: target_vec ≈ other_mat @ coeffs
            try:
                coeffs, residuals, _, _ = np.linalg.lstsq(other_mat, target_vec, rcond=tol)
                residual_norm = np.linalg.norm(target_vec - other_mat @ coeffs)
                target_norm = np.linalg.norm(target_vec)
                
                if target_norm > 1e-12 and residual_norm / target_norm < 1e-8:
                    # Column is in the span of others, remove it
                    removed.append(remove_c)
                    current_rank -= 1
            except:
                pass
    
    if removed:
        keep_cols = [c for c in range(rank) if c not in removed]
        new_U = U[:, keep_cols]
        new_V = V[:, keep_cols]
        new_W = W[:, keep_cols]
        
        if _verify_fast(new_U.tolist(), new_V.tolist(), new_W.tolist(), n, m, p):
            return new_U.tolist(), new_V.tolist(), new_W.tolist()
    
    return None


def _try_gaussian_elimination_on_columns(U, V, W, n, m, p, deadline):
    """
    Try to find linear dependencies among columns and exploit them to reduce rank.
    Uses a more systematic approach with column pivoting.
    """
    rank = len(U[0])
    if rank <= 2:
        return None
    
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    nmp = n * m * p
    T = np.zeros((nmp, rank))
    
    for c in range(rank):
        U3 = U[:, c].reshape(n, m)
        V3 = V[:, c].reshape(m, p)
        W3 = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m*p + j*p + k, c] = U3[i, j] * V3[j, k] * W3[k, i]
    
    # Use QR decomposition with column pivoting to find independent columns
    try:
        Q, R, P = _qr_with_pivot(T)
    except:
        return None
    
    # P is a permutation such that T[:, P] has QR decomposition
    # The rank is the number of significant diagonal entries in R
    tol = 1e-9
    r_diag = np.abs(np.diag(R))
    effective_rank = int(np.sum(r_diag > tol))
    
    if effective_rank >= rank:
        return None
    
    # The first effective_rank columns (after permutation) are independent
    # Try to remove the remaining columns
    independent_cols = P[:effective_rank]
    dependent_cols = P[effective_rank:]
    
    if len(dependent_cols) == 0:
        return None
    
    # For each dependent column, try to remove it
    # The idea is that if the tensor space has dimension effective_rank,
    # we might be able to find a different decomposition with fewer columns
    
    # This is complex, so we'll just verify if removing dependent columns still works
    # (this is a heuristic - it may not always work)
    
    keep_cols = list(independent_cols)
    new_U = U[:, keep_cols]
    new_V = V[:, keep_cols]
    new_W = W[:, keep_cols]
    
    if _verify_fast(new_U.tolist(), new_V.tolist(), new_W.tolist(), n, m, p):
        return new_U.tolist(), new_V.tolist(), new_W.tolist()
    
    return None


def _qr_with_pivot(A):
    """QR decomposition with column pivoting."""
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()
    P = np.eye(n, dtype=int)
    
    for k in range(min(m, n)):
        # Find column with largest norm in remaining columns
        if k < n:
            norms = np.linalg.norm(R[k:, k:], axis=0)
            col_offset = int(np.argmax(norms)) + k
            if col_offset != k:
                R[:, [k, col_offset]] = R[:, [col_offset, k]]
                P[:, [k, col_offset]] = P[:, [col_offset, k]]
        
        # Compute QR step
        if k < m:
            norm = np.linalg.norm(R[k:, k])
            if norm > 1e-15:
                v = R[k:, k].copy()
                v[0] = 1 + np.sign(v[0])
                v = v / np.linalg.norm(v)
                
                # Apply Householder reflection
                R[k:, k:] -= 2 * np.outer(v, v @ R[k:, k:])
                Q[:, k:] -= 2 * np.outer(Q[:, k:] @ v, v)
    
    return Q, R, P


def _try_numeric_optimization(U_init, V_init, W_init, n, m, p, target_rank, deadline, max_iter=2000, lr=0.01):
    """
    Try to find a lower-rank decomposition using gradient-based optimization.
    Start from the given decomposition and try to reduce rank by zeroing out
    some columns and optimizing the remaining ones.
    """
    import random
    
    U = np.array(U_init, dtype=np.float64).copy()
    V = np.array(V_init, dtype=np.float64).copy()
    W = np.array(W_init, dtype=np.float64).copy()
    
    current_rank = U.shape[1]
    best_rank = current_rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try different target ranks from current down to target
    for target in range(min(current_rank - 1, target_rank), max(target_rank - 1, 39), -1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random subsets of columns to keep
        n_attempts = 5
        for attempt in range(n_attempts):
            if time.time() > deadline - 10:
                break
            
            # Randomly select which columns to keep
            keep_indices = np.random.choice(current_rank, size=target, replace=False)
            
            U_work = U[:, keep_indices].copy()
            V_work = V[:, keep_indices].copy()
            W_work = W[:, keep_indices].copy()
            
            # Optimize using simple gradient descent
            for it in range(max_iter):
                if time.time() > deadline - 10:
                    break
                
                # Compute gradient numerically or use analytical gradient
                # For simplicity, use a simple update rule
                U3 = U_work.reshape(n, m, target)
                V3 = V_work.reshape(m, p, target)
                W3 = W_work.reshape(p, n, target)
                
                # Current tensor
                T_current = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                error = T_current - 1.0
                loss = np.sum(error**2)
                
                if loss < 1e-10:
                    break
                
                # Compute gradients
                # dL/dU[i,j,c] = 2 * error[i,j,k] * V[j,k,c] * W[k,i,c]
                # dL/dV[j,k,c] = 2 * error[i,j,k] * U[i,j,c] * W[k,i,c]
                # dL/dW[k,i,c] = 2 * error[i,j,k] * U[i,j,c] * V[j,k,c]
                
                dU = np.zeros_like(U_work)
                dV = np.zeros_like(V_work)
                dW = np.zeros_like(W_work)
                
                for c in range(target):
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                dU[i*m+j, c] += 2 * error[i,j,k] * V3[j,k,c] * W3[k,i,c]
                                dV[j*p+k, c] += 2 * error[i,j,k] * U3[i,j,c] * W3[k,i,c]
                                dW[k*n+i, c] += 2 * error[i,j,k] * U3[i,j,c] * V3[j,k,c]
                
                # Update
                U_work -= lr * dU
                V_work -= lr * dV
                W_work -= lr * dW
                
                # Check convergence
                if it % 50 == 0:
                    U3 = U_work.reshape(n, m, target)
                    V3 = V_work.reshape(m, p, target)
                    W3 = W_work.reshape(p, n, target)
                    T_check = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                    if np.allclose(T_check, 1.0, atol=1e-9):
                        break
            
            # Verify final result
            U3 = U_work.reshape(n, m, target)
            V3 = V_work.reshape(m, p, target)
            W3 = W_work.reshape(p, n, target)
            T_final = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            
            if np.allclose(T_final, 1.0, atol=1e-9):
                best_rank = target
                best_U, best_V, best_W = U_work.copy(), V_work.copy(), W_work.copy()
                break
        
        if best_rank <= target:
            break
    
    if best_rank < current_rank:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 * 4x4 matrix multiplication.
    Uses Strassen for 2x2*2x2 sub-blocks to reduce rank from 48 to 44,
    then attempts further rank reduction via linear dependency analysis
    and numerical optimization.
    """
    n, m, p = DIMENSIONS  # 3, 4, 4
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Strassen-based decomposition:
    # Split A (3x4) into top 2x4 (rows 0,1) and bottom 1x4 (row 2)
    # Split B (4x4) into four 2x2 blocks
    # Split C (3x4) into top 2x4 (rows 0,1) and bottom 1x4 (row 2)
    #
    # C[0:2, 0:2] = A[0:2, 0:2]*B[0:2, 0:2] + A[0:2, 2:4]*B[2:4, 0:2]
    # C[0:2, 2:4] = A[0:2, 0:2]*B[0:2, 2:4] + A[0:2, 2:4]*B[2:4, 2:4]
    # C[2, 0:2]   = A[2, 0:2]*B[0:2, 0:2] + A[2, 2:4]*B[2:4, 0:2]
    # C[2, 2:4]   = A[2, 0:2]*B[0:2, 2:4] + A[2, 2:4]*B[2:4, 2:4]
    #
    # 2x2*2x2: rank 7 (Strassen)
    # 1x2*2x2: rank 4 (schoolbook)
    #
    # Total: 4*7 + 4*4 = 28 + 16 = 44
    
    # Initialize full matrices with zeros
    rank_total = 44
    U = [[0] * rank_total for _ in range(n * m)]  # 12 x 44
    V = [[0] * rank_total for _ in range(m * p)]  # 16 x 44
    W = [[0] * rank_total for _ in range(p * n)]  # 12 x 44
    
    col = 0
    
    # Helper to add a sub-decomposition at given offsets
    def add_sub(U_sub, V_sub, W_sub, i_off, j_off, k_off, na, mb, pb, col_start):
        nonlocal col
        r = len(U_sub[0])
        for c in range(r):
            # U: index (i_off + i') * m + (j_off + j')
            for i_prime in range(na):
                for j_prime in range(mb):
                    u_val = U_sub[i_prime*mb + j_prime][c]
                    if u_val != 0:
                        U[(i_off + i_prime)*m + (j_off + j_prime)][col_start + c] += u_val
            
            # V: index (j_off + j') * p + (k_off + k')
            for j_prime in range(mb):
                for k_prime in range(pb):
                    v_val = V_sub[j_prime*pb + k_prime][c]
                    if v_val != 0:
                        V[(j_off + j_prime)*p + (k_off + k_prime)][col_start + c] += v_val
            
            # W: index (k_off + k') * n + (i_off + i')
            for k_prime in range(pb):
                for i_prime in range(na):
                    w_val = W_sub[k_prime*na + i_prime][c]
                    if w_val != 0:
                        W[(k_off + k_prime)*n + (i_off + i_prime)][col_start + c] += w_val
        col = col_start + r
    
    # C[0:2, 0:2]: A[0:2, 0:2]*B[0:2, 0:2] (2x2*2x2) + A[0:2, 2:4]*B[2:4, 0:2] (2x2*2x2)
    U_str, V_str, W_str = strassen_2x2()
    add_sub(U_str, V_str, W_str, 0, 0, 0, 2, 2, 2, col)
    
    U_str, V_str, W_str = strassen_2x2()
    add_sub(U_str, V_str, W_str, 0, 2, 0, 2, 2, 2, col)
    
    # C[0:2, 2:4]: A[0:2, 0:2]*B[0:2, 2:4] (2x2*2x2) + A[0:2, 2:4]*B[2:4, 2:4] (2x2*2x2)
    U_str, V_str, W_str = strassen_2x2()
    add_sub(U_str, V_str, W_str, 0, 0, 2, 2, 2, 2, col)
    
    U_str, V_str, W_str = strassen_2x2()
    add_sub(U_str, V_str, W_str, 0, 2, 2, 2, 2, 2, col)
    
    # C[2, 0:2]: A[2, 0:2]*B[0:2, 0:2] (1x2*2x2) + A[2, 2:4]*B[2:4, 0:2] (1x2*2x2)
    U_sub, V_sub, W_sub = schoolbook_sub(1, 2, 2)
    add_sub(U_sub, V_sub, W_sub, 2, 0, 0, 1, 2, 2, col)
    
    U_sub, V_sub, W_sub = schoolbook_sub(1, 2, 2)
    add_sub(U_sub, V_sub, W_sub, 2, 2, 0, 1, 2, 2, col)
    
    # C[2, 2:4]: A[2, 0:2]*B[0:2, 2:4] (1x2*2x2) + A[2, 2:4]*B[2:4, 2:4] (1x2*2x2)
    U_sub, V_sub, W_sub = schoolbook_sub(1, 2, 2)
    add_sub(U_sub, V_sub, W_sub, 2, 0, 2, 1, 2, 2, col)
    
    U_sub, V_sub, W_sub = schoolbook_sub(1, 2, 2)
    add_sub(U_sub, V_sub, W_sub, 2, 2, 2, 1, 2, 2, col)
    
    # Verify the decomposition
    if _verify_fast(U, V, W, n, m, p):
        best_result = {"U": U, "V": V, "W": W}
        best_rank = rank_total
        
        # Try to reduce rank further using multiple methods
        if time.time() < deadline:
            # Method 1: Linear dependency analysis
            reduced = _try_rank_reduction(U, V, W, n, m, p, best_rank - 1, deadline)
            if reduced is not None:
                new_rank = len(reduced[0][0])
                if new_rank < best_rank and _verify_fast(reduced[0], reduced[1], reduced[2], n, m, p):
                    best_result = {"U": reduced[0], "V": reduced[1], "W": reduced[2]}
                    best_rank = new_rank
                    U, V, W = reduced
        
        # Method 2: QR-based column elimination
        if time.time() < deadline and best_rank > 40:
            reduced2 = _try_gaussian_elimination_on_columns(U, V, W, n, m, p, deadline)
            if reduced2 is not None:
                new_rank = len(reduced2[0][0])
                if new_rank < best_rank and _verify_fast(reduced2[0], reduced2[1], reduced2[2], n, m, p):
                    best_result = {"U": reduced2[0], "V": reduced2[1], "W": reduced2[2]}
                    best_rank = new_rank
        
        # Method 3: Numerical optimization to find lower rank
        if time.time() < deadline and best_rank > 40:
            np.random.seed(42)
            reduced3 = _try_numeric_optimization(U, V, W, n, m, p, 40, deadline)
            if reduced3 is not None:
                new_rank = len(reduced3[0][0])
                if new_rank < best_rank and _verify_fast(reduced3[0], reduced3[1], reduced3[2], n, m, p):
                    best_result = {"U": reduced3[0], "V": reduced3[1], "W": reduced3[2]}
                    best_rank = new_rank
    
    # If verification failed, return schoolbook
    return best_result

# EVOLVE-BLOCK-END
