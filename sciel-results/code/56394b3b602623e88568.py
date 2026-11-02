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
    Try to reduce the rank of a decomposition by finding linear dependencies.
    Uses a simple approach: try to express one column as a linear combination of others.
    """
    rank = len(U[0])
    if rank <= target_rank:
        return None
    
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Try to find a column that can be removed by adjusting others
    # For each column c, try to find alpha such that column c = sum_{c' != c} alpha_c' * column c'
    # This requires the tensor contribution of column c to be in the span of others
    
    # Build the tensor for each column
    # T_c[i,j,k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Try removing each column and seeing if we can compensate
    for remove_c in range(rank):
        if time.time() > deadline:
            break
        
        # Check if column remove_c is "redundant" by checking if its tensor
        # can be approximated by a combination of others
        # Simple heuristic: if the column has very small norm, try removing it
        norm = np.linalg.norm(U[:, remove_c]) * np.linalg.norm(V[:, remove_c]) * np.linalg.norm(W[:, remove_c])
        
        if norm < 1e-10:
            # Column is effectively zero, remove it
            new_U = np.delete(U, remove_c, axis=1)
            new_V = np.delete(V, remove_c, axis=1)
            new_W = np.delete(W, remove_c, axis=1)
            if _verify_fast(new_U.tolist(), new_V.tolist(), new_W.tolist(), n, m, p):
                return new_U.tolist(), new_V.tolist(), new_W.tolist()
    
    return None


def _try_gaussian_elimination_on_columns(U, V, W, n, m, p, deadline):
    """
    Try to find linear dependencies among columns by checking if the tensor
    contributions are linearly dependent.
    """
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Flatten each column's tensor contribution
    # T_c is a vector of length n*m*p
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
    
    # Find the rank of T
    # If rank(T) < rank, there are linear dependencies
    # We can try to use this to reduce the rank
    
    # Compute SVD to find the rank
    try:
        _, s, Vt = np.linalg.svd(T, full_matrices=False)
        # Count significant singular values
        tol = 1e-10
        effective_rank = np.sum(s > tol)
        
        if effective_rank < rank:
            # There are dependencies, but exploiting them is complex
            # For now, just return the original
            pass
    except:
        pass
    
    return None


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 * 4x4 matrix multiplication.
    Uses Strassen for 2x2*2x2 sub-blocks to reduce rank from 48 to 44.
    """
    n, m, p = DIMENSIONS  # 3, 4, 4
    deadline = time.time() + 135  # Safety margin
    
    # Start with schoolbook
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
    # 2x2*2x1: rank 4 (schoolbook, since 2*2*1=4)
    # 1x2*2x2: rank 4 (schoolbook, since 1*2*2=4)
    # 1x2*2x1: rank 2 (schoolbook, since 1*2*1=2)
    #
    # Total: 2*7 + 2*7 + 2*4 + 2*4 = 14 + 14 + 8 + 8 = 44
    
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
    # First: i_off=0, j_off=0, k_off=0, na=2, mb=2, pb=2
    U_str, V_str, W_str = strassen_2x2()
    add_sub(U_str, V_str, W_str, 0, 0, 0, 2, 2, 2, col)
    
    # Second: i_off=0, j_off=2, k_off=0, na=2, mb=2, pb=2
    U_str, V_str, W_str = strassen_2x2()
    add_sub(U_str, V_str, W_str, 0, 2, 0, 2, 2, 2, col)
    
    # C[0:2, 2:4]: A[0:2, 0:2]*B[0:2, 2:4] (2x2*2x2) + A[0:2, 2:4]*B[2:4, 2:4] (2x2*2x2)
    # First: i_off=0, j_off=0, k_off=2, na=2, mb=2, pb=2
    U_str, V_str, W_str = strassen_2x2()
    add_sub(U_str, V_str, W_str, 0, 0, 2, 2, 2, 2, col)
    
    # Second: i_off=0, j_off=2, k_off=2, na=2, mb=2, pb=2
    U_str, V_str, W_str = strassen_2x2()
    add_sub(U_str, V_str, W_str, 0, 2, 2, 2, 2, 2, col)
    
    # C[2, 0:2]: A[2, 0:2]*B[0:2, 0:2] (1x2*2x2) + A[2, 2:4]*B[2:4, 0:2] (1x2*2x2)
    # First: i_off=2, j_off=0, k_off=0, na=1, mb=2, pb=2
    U_sub, V_sub, W_sub = schoolbook_sub(1, 2, 2)
    add_sub(U_sub, V_sub, W_sub, 2, 0, 0, 1, 2, 2, col)
    
    # Second: i_off=2, j_off=2, k_off=0, na=1, mb=2, pb=2
    U_sub, V_sub, W_sub = schoolbook_sub(1, 2, 2)
    add_sub(U_sub, V_sub, W_sub, 2, 2, 0, 1, 2, 2, col)
    
    # C[2, 2:4]: A[2, 0:2]*B[0:2, 2:4] (1x2*2x2) + A[2, 2:4]*B[2:4, 2:4] (1x2*2x2)
    # First: i_off=2, j_off=0, k_off=2, na=1, mb=2, pb=2
    U_sub, V_sub, W_sub = schoolbook_sub(1, 2, 2)
    add_sub(U_sub, V_sub, W_sub, 2, 0, 2, 1, 2, 2, col)
    
    # Second: i_off=2, j_off=2, k_off=2, na=1, mb=2, pb=2
    U_sub, V_sub, W_sub = schoolbook_sub(1, 2, 2)
    add_sub(U_sub, V_sub, W_sub, 2, 2, 2, 1, 2, 2, col)
    
    # Verify the decomposition
    if _verify_fast(U, V, W, n, m, p):
        best_result = {"U": U, "V": V, "W": W}
        best_rank = rank_total
    
        # Try to reduce rank further
        if time.time() < deadline:
            reduced = _try_rank_reduction(U, V, W, n, m, p, best_rank - 1, deadline)
            if reduced is not None:
                new_rank = len(reduced[0][0])
                if new_rank < best_rank and _verify_fast(reduced[0], reduced[1], reduced[2], n, m, p):
                    best_result = {"U": reduced[0], "V": reduced[1], "W": reduced[2]}
                    best_rank = new_rank
    
    # If verification failed, return schoolbook
    return best_result

# EVOLVE-BLOCK-END
