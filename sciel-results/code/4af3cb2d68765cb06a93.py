import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe
    
    # Rank-7 (Strassen) decomposition of the 3x3x3 multiplication tensor.
    # Indices: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # For a 3x3x3 slice, let local indices be i, j, k in 0..2
    # The tensor T[i,j,k] = 1 if the indices form a valid multiplication step.
    # Specifically, for the sub-problem, we are computing C_ik = sum_j A_ij B_jk.
    # The Strassen decomposition provides 7 terms such that:
    # sum_{r=0}^6 A_r[i,j] B_r[j,k] C_r[i,k] = delta_{ik, computed} ... wait.
    # The standard Strassen formula is for the *result* matrix C.
    # C_ik = sum_j A_ij B_jk.
    # The tensor we are decomposing is T_{(i,j),(j,k),(k,i)} = 1.
    # Note the indices in the task contract:
    # U has shape (n*m, rank), indexed by (i, j).
    # V has shape (m*p, rank), indexed by (j, k).
    # W has shape (p*n, rank), indexed by (k, i).
    # The condition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k).
    # And 0 otherwise.
    #
    # For a 3x3x3 block (say columns 0-2 of B), we map local indices i, j, k_global = k_local.
    # We need a decomposition of the 3x3x3 multiplication tensor.
    # Let's verify the Strassen coefficients provided in inspiration 2.
    # The inspiration 2 code had _A, _B, _C.
    # _A[r][i][j] corresponds to U[i*m+j, r] for the local block.
    # _B[r][j][k] corresponds to V[j*p+k, r] for the local block.
    # _C[r][i][k] corresponds to W[k*n+i, r] for the local block.
    #
    # Let's check if sum_r _A[r][i][j] * _B[r][j][k] * _C[r][i][k] equals 1 for all i,j,k?
    # No, the Strassen algorithm computes C = A * B.
    # The "tensor" of matrix multiplication M_{n,n,n} has entries 1 at (i, j, k) where the triple is valid?
    # Actually, the tensor T is defined on indices (i, j, k) such that T[i, j, k] = 1 if it contributes to C[i, k] via A[i, j] B[j, k].
    # In the standard definition, T[i, j, k] = 1 for all i, j, k.
    # The decomposition is: T[i, j, k] = sum_r U_r[i, j] V_r[j, k] W_r[i, k].
    # Note: W is indexed by (i, k) in standard literature, but here W is indexed by (k, i).
    # So W_task[k, i] = W_strassen[i, k].
    
    # Let's use the coefficients from inspiration 2, which were claimed to work for rank 14.
    # We will verify them at runtime. If they fail, we fall back to schoolbook.
    
    _A = [
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[0, 0, 0], [1, 0, 0], [1, 1, 1]],
        [[0, 0, 0], [1, 0, 0], [1, 0, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 1, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 1, 1]],
        [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[1, 1, 1], [0, 0, 0], [0, 0, 0]],
    ]
    _B = [
        [[0, 0, 0], [0, 1, 0], [0, 1, 0]],
        [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
        [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[1, 0, 0], [0, 0, 0], [0, -1, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 1, 0]],
        [[0, 0, 0], [0, 0, 0], [1, 1, 1]],
        [[0, 0, 0], [0, 0, 0], [1, 1, 1]],
    ]
    _C = [
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[0, 0, 0], [0, 1, 0], [0, 1, 0]],
        [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 1], [0, 0, 1]],
        [[0, 0, 0], [0, 0, 0], [1, 1, 0]],
    ]

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def get_rank14():
        # Split the 6 columns of B into two groups of 3.
        # Group 0: k in [0, 2], Group 1: k in [3, 5]
        # For each group, use the rank-7 Strassen decomposition.
        # Total rank = 7 + 7 = 14.
        rank = 14
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for g in range(2):
            base = g * 7
            k_offset = g * 3
            for r in range(7):
                # U: A_r[i, j]
                for i in range(n):
                    for j in range(m):
                        U[i * m + j, base + r] = _A[r][i][j]
                        
                # V: B_r[j, k_local] -> V[j, k_global]
                for j in range(m):
                    for k_local in range(3):
                        k_global = k_offset + k_local
                        V[j * p + k_global, base + r] = _B[r][j][k_local]
                        
                # W: C_r[i, k_local] -> W[k_global, i]
                # Note: _C[r][i][k_local] is the coefficient for the result C[i, k_local].
                # In our tensor definition, W is indexed by (k, i).
                # So W[k_global, i] should be _C[r][i][k_local].
                for i in range(n):
                    for k_local in range(3):
                        k_global = k_offset + k_local
                        W[k_global * n + i, base + r] = _C[r][i][k_local]
                        
        return U, V, W

    def verify(U, V, W, tol=1e-9):
        # Check 1-entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > tol:
                        return False
        
        # Check 0-entries
        # We iterate over all (a, b, c) and check if they form a valid (i, j, k).
        # a = i*m + j
        # b = j*p + k
        # c = k*n + i
        # Valid if:
        # j_a == j_b
        # k_b == k_c
        # i_c == i_a
        # where i_a = a // m, j_a = a % m
        # j_b = b // p, k_b = b % p
        # k_c = c // n, i_c = c % n
        
        for a in range(n * m):
            i_a = a // m
            j_a = a % m
            for b in range(m * p):
                j_b = b // p
                k_b = b % p
                if j_a != j_b:
                    # This is a 0-entry for any c
                    # We still need to check all c?
                    # Yes, because the sum is over r.
                    # But if j_a != j_b, the indices don't match the pattern.
                    for c in range(p * n):
                        val = 0.0
                        for r in range(U.shape[1]):
                            val += U[a, r] * V[b, r] * W[c, r]
                        if abs(val) > tol:
                            return False
                else:
                    # j_a == j_b
                    for c in range(p * n):
                        k_c = c // n
                        i_c = c % n
                        # Check if it's a 1-entry
                        if i_a == i_c and k_b == k_c:
                            continue # Already checked in 1-entries loop
                        # It's a 0-entry
                        val = 0.0
                        for r in range(U.shape[1]):
                            val += U[a, r] * V[b, r] * W[c, r]
                        if abs(val) > tol:
                            return False
        return True

    # Try rank 14 first
    U14, V14, W14 = get_rank14()
    if verify(U14, V14, W14):
        return {"U": U14.tolist(), "V": V14.tolist(), "W": W14.tolist()}
        
    # Fallback to schoolbook
    Us, Vs, Ws = get_schoolbook()
    # Verify schoolbook just in case (should always be true)
    if not verify(Us, Vs, Ws):
         raise Exception("Schoolbook failed")
         
    return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}
# EVOLVE-BLOCK-END
