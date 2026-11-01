import time
import numpy as np

DIMENSIONS = [2, 2, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS

    # Target: 2x2 * 2x9 -> 2x9
    # Decomposition strategy:
    # Split the 2x9 matrix B into four 2x2 blocks (cols 0-7) and one 2x1 block (col 8).
    # Use Strassen (rank 7) for each 2x2 block multiplication.
    # Use trivial multiplication (rank 4) for the 2x1 block.
    # Total rank = 4 * 7 + 4 = 32.

    rank = 32
    # U: (n*m) x rank = 4 x 32
    # V: (m*p) x rank = 18 x 32
    # W: (p*n) x rank = 18 x 32
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    current_col = 0

    def add_strassen_block(k_start, k_end):
        """
        Adds a Strassen decomposition for a 2x2 block of B (columns k_start, k_end)
        multiplied by A (2x2).
        Indices for B: rows 0,1. Cols k_start, k_end.
        Indices for C: rows 0,1. Cols k_start, k_end.
        """
        nonlocal current_col
        k0 = k_start
        k1 = k_end
        
        # Strassen formulas for 2x2 * 2x2
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6

        # 1. M1
        U[0][current_col] = 1
        U[3][current_col] = 1
        V[0*p + k0][current_col] = 1
        V[1*p + k1][current_col] = 1
        W[k0*n + 0][current_col] = 1
        W[k1*n + 1][current_col] = 1
        current_col += 1

        # 2. M2
        U[2][current_col] = 1
        U[3][current_col] = 1
        V[0*p + k0][current_col] = 1
        W[k0*n + 1][current_col] = 1
        W[k1*n + 1][current_col] = -1
        current_col += 1

        # 3. M3
        U[0][current_col] = 1
        V[0*p + k1][current_col] = 1
        V[1*p + k1][current_col] = -1
        W[k1*n + 0][current_col] = 1
        W[k1*n + 1][current_col] = 1
        current_col += 1

        # 4. M4
        U[3][current_col] = 1
        V[1*p + k0][current_col] = 1
        V[0*p + k0][current_col] = -1
        W[k0*n + 0][current_col] = 1
        W[k0*n + 1][current_col] = 1
        current_col += 1

        # 5. M5
        U[0][current_col] = 1
        U[1][current_col] = 1
        V[1*p + k1][current_col] = 1
        W[k0*n + 0][current_col] = -1
        W[k1*n + 0][current_col] = 1
        current_col += 1

        # 6. M6
        U[2][current_col] = 1
        U[0][current_col] = -1
        V[0*p + k0][current_col] = 1
        V[0*p + k1][current_col] = 1
        W[k1*n + 1][current_col] = 1
        current_col += 1

        # 7. M7
        U[1][current_col] = 1
        U[3][current_col] = -1
        V[1*p + k0][current_col] = 1
        V[1*p + k1][current_col] = 1
        W[k0*n + 0][current_col] = 1
        current_col += 1

    def add_trivial_col(k):
        """
        Adds trivial decomposition for a 2x1 column of B (column k)
        multiplied by A (2x2).
        """
        nonlocal current_col
        # C0k = A00*B0k + A01*B1k
        # C1k = A10*B0k + A11*B1k
        
        # Term 1: A00 * B0k -> C0k
        U[0][current_col] = 1
        V[0*p + k][current_col] = 1
        W[k*n + 0][current_col] = 1
        current_col += 1

        # Term 2: A01 * B1k -> C0k
        U[1][current_col] = 1
        V[1*p + k][current_col] = 1
        W[k*n + 0][current_col] = 1
        current_col += 1

        # Term 3: A10 * B0k -> C1k
        U[2][current_col] = 1
        V[0*p + k][current_col] = 1
        W[k*n + 1][current_col] = 1
        current_col += 1

        # Term 4: A11 * B1k -> C1k
        U[3][current_col] = 1
        V[1*p + k][current_col] = 1
        W[k*n + 1][current_col] = 1
        current_col += 1

    # Apply Strassen to blocks (0,1), (2,3), (4,5), (6,7)
    add_strassen_block(0, 1)
    add_strassen_block(2, 3)
    add_strassen_block(4, 5)
    add_strassen_block(6, 7)
    
    # Apply trivial to column 8
    add_trivial_col(8)

    # Verification (optional but good for safety)
    def verify(U_list, V_list, W_list):
        U_arr = np.array(U_list, dtype=np.float64)
        V_arr = np.array(V_list, dtype=np.float64)
        W_arr = np.array(W_list, dtype=np.float64)
        max_err = 0.0
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    # T[i,j,k] should be 1
                    # Sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
                    u_vec = U_arr[i*m + k]
                    v_vec = V_arr[k*p + j]
                    w_val = W_arr[j*n + i]
                    # Note: W is a vector of length rank, but in the tensor definition,
                    # the third index is fixed for a specific output element (i,j).
                    # The decomposition is T[i,j,k] = sum_r U[i,k,r] V[k,j,r] W[j,i,r]
                    # Here W[j*n+i] is a scalar for a fixed (j,i)?
                    # No, W is a matrix of size (p*n) x rank.
                    # So W[j*n+i, r] is the r-th entry.
                    # The term is U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
                    # Sum over r.
                    prod = u_vec * v_vec * w_val # w_val is a row of W?
                    # Wait. W is (p*n) x rank.
                    # w_val = W_arr[j*n + i] is a vector of length rank.
                    # So we need element-wise product of three vectors, then sum.
                    term = u_vec * v_vec * w_val
                    val = np.sum(term)
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    # Check if time allows verification
    if time.time() < deadline - 5.0:
        err = verify(U, V, W)
        if err > 1e-9:
            # Fallback to trivial if Strassen implementation has a bug
            # This is highly unlikely but safe.
            rank_t = n * m * p
            U_t = [[0] * rank_t for _ in range(n * m)]
            V_t = [[0] * rank_t for _ in range(m * p)]
            W_t = [[0] * rank_t for _ in range(p * n)]
            c = 0
            for ii in range(n):
                for jj in range(m):
                    for kk in range(p):
                        U_t[ii*m + jj][c] = 1
                        V_t[jj*p + kk][c] = 1
                        W_t[kk*n + ii][c] = 1
                        c += 1
            return {"U": U_t, "V": V_t, "W": W_t}

    return {"U": U, "V": V, "W": W}
