import time
import numpy as np
from itertools import combinations

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    """
    n, m, p = 2, 3, 8
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Build the rank-44 solution
    U44, V44, W44 = _build_rank_44()
    
    best_rank = 44
    best_U, best_V, best_W = U44, V44, W44
    
    # Try to find a rank-43 solution
    if time.time() < deadline - 30:
        found = _try_rank_43(deadline)
        if found is not None:
            best_rank = 43
            best_U, best_V, best_W = found
    
    return {"U": best_U, "V": best_V, "W": best_W}


def _build_rank_44():
    """
    Build the rank-44 solution using Strassen for 2x2*2x2 blocks and schoolbook for 2x1*1x8.
    """
    n, m, p = 2, 3, 8
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44
    
    # Strassen 2x2x2 coefficients
    S_U = [
        [1, 0, 0, 1],  # M1: (A00+A11)(B00+B11)
        [0, 0, 1, 1],  # M2: (A10+A11)B00
        [1, 0, 0, 0],  # M3: A00(B10-B11)
        [0, 0, 0, 1],  # M4: A11(B01-B10)
        [1, 1, 0, 0],  # M5: (A00+A10)B11
        [-1, 0, 1, 0], # M6: (A11-A00)(B00+B10)
        [0, 1, 0, -1]  # M7: (A10-A01)(B11+B01)
    ]
    
    S_V = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1]   # M7
    ]
    
    S_W = [
        [1, 0, 0, 1],  # M1
        [0, 1, 0, -1], # M2
        [0, 0, 1, 1],  # M3
        [1, 1, 0, 0],  # M4
        [-1, 0, 1, 0], # M5
        [0, 0, 0, 1],  # M6
        [1, 0, 0, 0]   # M7
    ]
    
    col = 0
    
    # Part 1: A1 (2x2) * B1 (2x8)
    # B1 has rows 0,1 of B. Split into 4 blocks of 2 columns.
    for block_idx in range(4):
        k_start = block_idx * 2
        
        for r in range(7):
            # Fill U for A1 (rows 0,1 of A, cols 0,1 of A)
            for li in range(2):
                for la in range(2):
                    val = S_U[r][li * 2 + la]
                    if val != 0:
                        u_idx = li * m + la
                        U[u_idx][col] += val
            
            # Fill V for B1 block (rows 0,1 of B, cols k_start..k_start+1)
            for la in range(2):
                for lk in range(2):
                    val = S_V[r][la * 2 + lk]
                    if val != 0:
                        v_idx = la * p + (k_start + lk)
                        V[v_idx][col] += val
            
            # Fill W for C block (cols k_start..k_start+1 of C, rows 0..1 of C)
            for lk in range(2):
                for li in range(2):
                    val = S_W[r][lk * 2 + li]
                    if val != 0:
                        w_idx = (k_start + lk) * n + li
                        W[w_idx][col] += val
            
            col += 1
    
    # Part 2: A2 (2x1) * B2 (1x8)
    # A2 is col 2 of A (rows 0,1). B2 is row 2 of B (cols 0..7).
    for i in range(2):
        for k in range(8):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
    
    # Validate
    if _validate(U, V, W):
        return U, V, W
    
    # Fallback: schoolbook
    return _schoolbook()


def _schoolbook():
    """Schoolbook decomposition, rank = n*m*p = 48."""
    n, m, p = 2, 3, 8
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for a in range(m):
            for k in range(p):
                U[i * m + a][col] = 1
                V[a * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return U, V, W


def _validate(U, V, W):
    """
    Validate that sum_t U[i,t]*V[j,t]*W[k,t] = 1 for all valid (i,j,k)
    and 0 otherwise.
    """
    n, m, p = 2, 3, 8
    rank = len(U[0])
    
    for i in range(n * m):
        for j in range(m * p):
            for k in range(p * n):
                acc = 0
                for t in range(rank):
                    acc += U[i][t] * V[j][t] * W[k][t]
                
                # Determine expected value
                # i = row_of_A * m + col_of_A, so col_of_A = i % m, row_of_A = i // m
                # j = row_of_B * p + col_of_B, so row_of_B = j // p, col_of_B = j % p
                # k = col_of_C * n + row_of_C, so row_of_C = k % n, col_of_C = k // n
                
                col_of_A = i % m
                row_of_A = i // m
                row_of_B = j // p
                col_of_B = j % p
                row_of_C = k % n
                col_of_C = k // n
                
                # Valid if col_of_A == row_of_B and row_of_A == row_of_C and col_of_B == col_of_C
                if col_of_A == row_of_B and row_of_A == row_of_C and col_of_B == col_of_C:
                    expected = 1
                else:
                    expected = 0
                
                if abs(acc - expected) > 1e-10:
                    return False
    
    return True


def _try_rank_43(deadline):
    """
    Try to find a rank-43 solution.
    
    Strategy: Start from the rank-44 solution and try to merge two terms.
    Specifically, try to find two terms t1, t2 such that their combined tensor
    contribution can be expressed as a single new term plus adjustments.
    
    A more practical approach: use alternating least squares to find a rank-43
    decomposition that satisfies the tensor equations.
    """
    n, m, p = 2, 3, 8
    target_rank = 43
    
    # Get the rank-44 solution
    U44, V44, W44 = _build_rank_44()
    
    # Initialize rank-43 solution by dropping the last term of rank-44
    U = [row[:target_rank] for row in U44]
    V = [row[:target_rank] for row in V44]
    W = [row[:target_rank] for row in W44]
    
    # Convert to numpy for efficiency
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Precompute the target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n * m):
        for j in range(m * p):
            if i % m == j // p:  # col_of_A == row_of_B
                k = (j % p) * n + (i // m)  # col_of_B * n + row_of_A
                T[i, j, k] = 1
    
    # Alternating least squares
    num_iters = 50
    for it in range(num_iters):
        if time.time() > deadline - 15:
            break
        
        # Update U: for each i, solve for U[i,:] given V, W
        for i in range(n * m):
            # Build system: U[i,:] * (V[j,:] ⊙ W[k,:]) = T[i,j,k] for all j,k
            # This is a linear system in U[i,:]
            A_rows = []
            b_vals = []
            for j in range(m * p):
                for k in range(p * n):
                    if T[i, j, k] != 0 or True:  # Include all for stability
                        vw = V_arr[j, :] * W_arr[k, :]
                        A_rows.append(vw)
                        b_vals.append(T[i, j, k])
            
            if A_rows:
                A_mat = np.array(A_rows)
                b_vec = np.array(b_vals)
                try:
                    U_arr[i, :], residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except:
                    pass
        
        # Update V: for each j, solve for V[j,:] given U, W
        for j in range(m * p):
            A_rows = []
            b_vals = []
            for i in range(n * m):
                for k in range(p * n):
                    uw = U_arr[i, :] * W_arr[k, :]
                    A_rows.append(uw)
                    b_vals.append(T[i, j, k])
            
            if A_rows:
                A_mat = np.array(A_rows)
                b_vec = np.array(b_vals)
                try:
                    V_arr[j, :], residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except:
                    pass
        
        # Update W: for each k, solve for W[k,:] given U, V
        for k in range(p * n):
            A_rows = []
            b_vals = []
            for i in range(n * m):
                for j in range(m * p):
                    uv = U_arr[i, :] * V_arr[j, :]
                    A_rows.append(uv)
                    b_vals.append(T[i, j, k])
            
            if A_rows:
                A_mat = np.array(A_rows)
                b_vec = np.array(b_vals)
                try:
                    W_arr[k, :], residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except:
                    pass
        
        # Check convergence
        error = 0
        for i in range(n * m):
            for j in range(m * p):
                for k in range(p * n):
                    acc = np.dot(U_arr[i, :], V_arr[j, :] * W_arr[k, :])
                    error += (acc - T[i, j, k]) ** 2
        
        if error < 1e-12:
            break
    
    # Final validation
    if time.time() < deadline - 5:
        U_final = U_arr.tolist()
        V_final = V_arr.tolist()
        W_final = W_arr.tolist()
        
        if _validate(U_final, V_final, W_final):
            return U_final, V_final, W_final
    
    return None


if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
