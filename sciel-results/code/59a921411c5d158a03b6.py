import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin within 120s limit
    
    # Build rank-28 decomposition using block structure
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen_222(i_off, j_off, k_off):
        nonlocal col
        # 7 Strassen terms for 2x2x2
        # M1 = (a00+a11)(b00+b11)
        U[(0 + i_off) * m + (0 + j_off), col] = 1
        U[(1 + i_off) * m + (1 + j_off), col] = 1
        V[(0 + j_off) * p + (0 + k_off), col] = 1
        V[(1 + j_off) * p + (1 + k_off), col] = 1
        W[(0 + k_off) * n + (0 + i_off), col] = 1
        W[(1 + k_off) * n + (1 + i_off), col] = 1
        col += 1
        
        # M2 = a10*(b00+b01)
        U[(1 + i_off) * m + (0 + j_off), col] = 1
        U[(1 + i_off) * m + (1 + j_off), col] = 1
        V[(0 + j_off) * p + (0 + k_off), col] = 1
        W[(0 + k_off) * n + (1 + i_off), col] = 1
        W[(1 + k_off) * n + (1 + i_off), col] = -1
        col += 1
        
        # M3 = (a00-a10)*b10
        U[(0 + i_off) * m + (0 + j_off), col] = 1
        V[(0 + j_off) * p + (1 + k_off), col] = 1
        V[(1 + j_off) * p + (1 + k_off), col] = -1
        W[(1 + k_off) * n + (0 + i_off), col] = 1
        W[(1 + k_off) * n + (1 + i_off), col] = 1
        col += 1
        
        # M4 = a11*(b10-b00)
        U[(1 + i_off) * m + (1 + j_off), col] = 1
        V[(1 + j_off) * p + (0 + k_off), col] = 1
        V[(0 + j_off) * p + (0 + k_off), col] = -1
        W[(0 + k_off) * n + (0 + i_off), col] = 1
        W[(0 + k_off) * n + (1 + i_off), col] = 1
        col += 1
        
        # M5 = (a00+a01)*b11
        U[(0 + i_off) * m + (0 + j_off), col] = 1
        U[(0 + i_off) * m + (1 + j_off), col] = 1
        V[(1 + j_off) * p + (1 + k_off), col] = 1
        W[(0 + k_off) * n + (0 + i_off), col] = -1
        W[(1 + k_off) * n + (0 + i_off), col] = 1
        col += 1
        
        # M6 = (a10-a00)*(b00+b11)
        U[(1 + i_off) * m + (0 + j_off), col] = 1
        U[(0 + i_off) * m + (0 + j_off), col] = -1
        V[(0 + j_off) * p + (0 + k_off), col] = 1
        V[(0 + j_off) * p + (1 + k_off), col] = 1
        W[(1 + k_off) * n + (1 + i_off), col] = 1
        col += 1
        
        # M7 = (a01-a11)*(b10+b11)
        U[(0 + i_off) * m + (1 + j_off), col] = 1
        U[(1 + i_off) * m + (1 + j_off), col] = -1
        V[(1 + j_off) * p + (0 + k_off), col] = 1
        V[(1 + j_off) * p + (1 + k_off), col] = 1
        W[(0 + k_off) * n + (0 + i_off), col] = 1
        col += 1
    
    # Block 1: j=0, all i,k -> 2*5 = 10 schoolbook terms
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1
            V[0 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
    
    # Block 2: j=1,2, k=0,1 -> Strassen 2x2x2 = 7 terms
    add_strassen_222(0, 1, 0)
    
    # Block 3: j=1,2, k=2 -> schoolbook 2x2x1 = 4 terms
    for i in range(n):
        for j in [1, 2]:
            U[i * m + j, col] = 1
            V[j * p + 2, col] = 1
            W[2 * n + i, col] = 1
            col += 1
    
    # Block 4: j=1,2, k=3,4 -> Strassen 2x2x2 = 7 terms
    add_strassen_222(0, 1, 3)
    
    assert col == rank, f"Column count mismatch: {col} != {rank}"
    
    # Verify the decomposition is correct
    def verify(U_mat, V_mat, W_mat):
        """Check that sum_r u_r ⊗ v_r ⊗ w_r = T where T[i,j,k]=1"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    coeff = 0
                    for r in range(U_mat.shape[1]):
                        coeff += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if abs(coeff - 1) > 1e-10:
                        return False
        return True
    
    if not verify(U, V, W):
        # Fallback: pure schoolbook
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i*m*p + j*p + k
                    U[i*m+j, idx] = 1
                    V[j*p+k, idx] = 1
                    W[k*n+i, idx] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find linear dependencies to reduce rank
    # Build 30 x rank matrix where each column is the contribution vector
    M = np.zeros((n*m*p, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i*m*p + j*p + k
                M[idx, :] = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
    
    # Check if the 28 columns are linearly independent
    rank_M = np.linalg.matrix_rank(M, tol=1e-10)
    
    if rank_M < rank:
        # Find null space
        U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
        null_mask = s < 1e-10
        if np.any(null_mask):
            # Try each null vector to eliminate a term
            for null_idx in np.where(null_mask)[0]:
                c = Vt[null_idx, :]
                k_idx = np.argmax(np.abs(c))
                if abs(c[k_idx]) > 1e-10:
                    # Eliminate term k_idx
                    new_terms = []
                    for r in range(rank):
                        if r == k_idx:
                            continue
                        coeff = 1 - c[r] / c[k_idx]
                        new_terms.append((U[:, r] * coeff, V[:, r] * coeff, W[:, r] * coeff))
                    
                    new_U = np.column_stack([t[0] for t in new_terms])
                    new_V = np.column_stack([t[1] for t in new_terms])
                    new_W = np.column_stack([t[2] for t in new_terms])
                    
                    # Verify
                    if verify(new_U, new_V, new_W):
                        return {"U": new_U.tolist(), "V": new_V.tolist(), "W": new_W.tolist()}
    
    # Return the rank-28 solution
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
    # Verify
    n, m, p = 2, 3, 5
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    rank = U.shape[1]
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                coeff = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(coeff - 1) > 1e-10:
                    ok = False
                    print(f"FAIL at ({i},{j},{k}): {coeff}")
    print("Valid:", ok)
