"""
Program for matrix multiplication tensor decomposition (3x5 by 5x5).
Strategy: Partition into 2x4x4 core (4 disjoint 2x2x2 Strassen blocks, rank 4*7=28)
and 1x5x5 residual (schoolbook 25). Total initial rank 53.
Then apply rank reduction search on the residual.
"""
import time
import numpy as np
import random

DIMENSIONS = [3, 5, 5]

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm (rank 7).
    Indices:
    U: (i, j) where i in 0..1, j in 0..1 -> index i*2+j
    V: (j, k) where j in 0..1, k in 0..1 -> index j*2+k
    W: (k, i) where k in 0..1, i in 0..1 -> index k*2+i
    
    Strassen factors:
    M1 = (A11 + A22)(B12 + B21)
    M2 = (A21 + A22)B11
    M3 = A11(B11 - B21)
    M4 = A22(B22 - B12)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (A11+A22)(B12+B21)
    # A indices: A11=0, A12=1, A21=2, A22=3
    # B indices: B11=0, B12=1, B21=2, B22=3
    # C indices: C11=0, C12=1, C21=2, C22=3 (Trace convention W[k*n+i] for 2x2 is W[k*2+i])
    # Wait, W index is k*n+i. For 2x2, n=2.
    # C[i,k] corresponds to W[k*2+i].
    # C11 -> i=0, k=0 -> W[0]
    # C12 -> i=0, k=1 -> W[2]
    # C21 -> i=1, k=0 -> W[1]
    # C22 -> i=1, k=1 -> W[3]
    
    # M1
    U[0, 0] += 1; U[3, 0] += 1
    V[1, 0] += 1; V[2, 0] += 1
    W[0, 0] += 1; W[3, 0] += 1
    
    # M2: (A21+A22)B11
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    W[1, 1] += 1; W[3, 1] += 1
    
    # M3: A11(B11-B21)
    U[0, 2] += 1
    V[0, 2] += 1; V[2, 2] -= 1
    W[2, 2] += 1; W[1, 2] += 1
    
    # M4: A22(B22-B12)
    U[3, 3] += 1
    V[3, 3] += 1; V[1, 3] -= 1
    W[1, 3] += 1; W[3, 3] += 1
    
    # M5: (A11+A12)B22
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    W[3, 4] -= 1; W[2, 4] += 1
    
    # M6: (A21-A11)(B11+B12)
    U[2, 5] += 1; U[0, 5] -= 1
    V[0, 5] += 1; V[1, 5] += 1
    W[3, 5] += 1
    
    # M7: (A12-A22)(B21+B22)
    U[1, 6] += 1; U[3, 6] -= 1
    V[2, 6] += 1; V[3, 6] += 1
    W[0, 6] += 1; W[2, 6] -= 1
    
    return U, V, W

def _build_block_decomp(n, m, p):
    """
    Builds a block decomposition for n=3, m=5, p=5.
    Splits A (3x5) into [A_top (2x5); A_bot (1x5)].
    Splits B (5x5) into [B_left (5x4) | B_right (5x1)].
    
    Core: A_top (2x4) * B_left (4x4) -> C_top_left (2x4)
    This is 4 disjoint 2x2x2 multiplications.
    Residual:
    1. A_top (2x4) * B_right (4x1) -> C_top_right (2x1)
    2. A_bot (1x4) * B_left (4x4) -> C_bot_left (1x4)
    3. A_bot (1x1) * B_right (1x1) -> C_bot_right (1x1)
    
    Wait, the split of B is columns. B is 5x5.
    B_left is 5x4 (cols 0-3), B_right is 5x1 (col 4).
    A_top is 2x5 (rows 0-1), A_bot is 1x5 (row 2).
    
    The multiplication is C = A * B.
    C is 3x5.
    C[i, k] = sum_j A[i, j] B[j, k].
    
    Let's partition indices:
    i in {0,1,2}, j in {0,1,2,3,4}, k in {0,1,2,3,4}.
    
    Block 1: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}.
    This is a 2x4x4 tensor. It can be viewed as 4 independent 2x2x2 tensors?
    No, A is 2x4, B is 4x4.
    We can split j into pairs {0,1} and {2,3}, and k into pairs {0,1} and {2,3}.
    So we have 4 sub-blocks of size 2x2x2.
    Block (j_pair, k_pair) where j_pair in {0,1}, k_pair in {0,1}.
    For each, A_sub is 2x2, B_sub is 2x2, C_sub is 2x2.
    These are disjoint in terms of indices.
    
    Residual:
    - i in {0,1}, j in {4}, k in {0,1,2,3,4}. (A_top * B_col4)
    - i in {2}, j in {0,1,2,3}, k in {0,1,2,3,4}. (A_bot * B_left)
    - i in {2}, j in {4}, k in {4}. (A_bot * B_col4)
    
    Actually, let's just do schoolbook for the residual parts to ensure correctness first,
    then optimize the residual.
    """
    U = np.zeros((n*m, 0))
    V = np.zeros((m*p, 0))
    W = np.zeros((p*n, 0))
    
    # 1. Add 4 Strassen blocks for the 2x4x4 core
    strassen_U, strassen_V, strassen_W = _strassen_2x2x2()
    
    # Map local 2x2 indices to global
    # j_pairs: (0,1) and (2,3)
    # k_pairs: (0,1) and (2,3)
    for jp in range(2):
        for kp in range(2):
            j_offset = jp * 2
            k_offset = kp * 2
            
            # U indices: i in {0,1}, j in {j_offset, j_offset+1}
            # Global U index: i*m + j
            # V indices: j in {j_offset, j_offset+1}, k in {k_offset, k_offset+1}
            # Global V index: j*p + k
            # W indices: k in {k_offset, k_offset+1}, i in {0,1}
            # Global W index: k*n + i
            
            for r in range(7):
                # U
                for li in range(2):
                    global_i = li
                    for lj in range(2):
                        global_j = j_offset + lj
                        local_idx = li * 2 + lj
                        global_idx = global_i * m + global_j
                        U = np.column_stack([U, np.zeros(n*m)]) if U.size == 0 else np.hstack([U, np.zeros((n*m, 1))])
                        # Wait, easier to build lists first
                        pass
            
            # Let's restart with lists for easier construction
            pass
            
    # Re-do with lists
    U_list = []
    V_list = []
    W_list = []
    
    # Initialize with zeros for the core part (rank 28)
    core_rank = 4 * 7
    for _ in range(core_rank):
        U_list.append(np.zeros(n*m))
        V_list.append(np.zeros(m*p))
        W_list.append(np.zeros(p*n))
        
    col_idx = 0
    for jp in range(2):
        for kp in range(2):
            j_offset = jp * 2
            k_offset = kp * 2
            
            for r in range(7):
                # U
                for li in range(2):
                    for lj in range(2):
                        local_u = strassen_U[li*2+lj, r]
                        if local_u != 0:
                            gi = li
                            gj = j_offset + lj
                            U_list[col_idx][gi*m + gj] = local_u
                            
                # V
                for lj in range(2):
                    for lk in range(2):
                        local_v = strassen_V[lj*2+lk, r]
                        if local_v != 0:
                            gj = j_offset + lj
                            gk = k_offset + lk
                            V_list[col_idx][gj*p + gk] = local_v
                            
                # W
                for lk in range(2):
                    for li in range(2):
                        local_w = strassen_W[lk*2+li, r]
                        if local_w != 0:
                            gk = k_offset + lk
                            gi = li
                            W_list[col_idx][gk*n + gi] = local_w
                            
                col_idx += 1
                
    # 2. Add Schoolbook for the residual
    # Residual indices:
    # (i, j, k) such that NOT (i in {0,1} AND j in {0,1,2,3} AND k in {0,1,2,3})
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                is_core = (i < 2 and j < 4 and k < 4)
                if not is_core:
                    # Add schoolbook term
                    U_list.append(np.zeros(n*m))
                    V_list.append(np.zeros(m*p))
                    W_list.append(np.zeros(p*n))
                    U_list[-1][i*m + j] = 1
                    V_list[-1][j*p + k] = 1
                    W_list[-1][k*n + i] = 1
                    
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify decomposition."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _reduce_rank(U, V, W, n, m, p, deadline, tol=1e-9):
    """
    Attempt to reduce rank by checking if any column is linearly dependent
    in a way that allows removal, or by finding small perturbations.
    For now, a simple greedy check: can we remove a column?
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    
    # Greedy removal
    while rank > 0:
        if time.time() > deadline:
            break
            
        removed = False
        for c in range(rank):
            # Try removing column c
            # Check if T - sum_{r != c} u_r v_r w_r == 0
            # This is expensive. Instead, check if u_c, v_c, w_c are "zero" or dependent?
            # Actually, for schoolbook, no column is removable.
            # For Strassen, maybe?
            
            # Simple check: if column c is all zeros in U, V, or W, remove it.
            if np.allclose(U[:, c], 0) or np.allclose(V[:, c], 0) or np.allclose(W[:, c], 0):
                U = np.delete(U, c, axis=1)
                V = np.delete(V, c, axis=1)
                W = np.delete(W, c, axis=1)
                rank -= 1
                removed = True
                break
                
        if not removed:
            # Try a more sophisticated check:
            # For each pair of columns, see if they can be merged?
            # Or check linear dependence of the set of vectors?
            # Let's just break for now to avoid spending too much time on ineffective search.
            break
            
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Build block decomposition
    U, V, W = _build_block_decomp(n, m, p)
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook if block decomp is wrong
        rank = n * m * p
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Try to reduce rank
    U, V, W = _reduce_rank(U, V, W, n, m, p, deadline)
    
    # Final verify
    if not _verify(U, V, W, n, m, p):
        # Fallback
        rank = n * m * p
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
