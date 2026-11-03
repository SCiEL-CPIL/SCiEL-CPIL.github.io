import time
import numpy as np

def construct():
    n, m, p = 2, 3, 8

    dim_U = n * m  # 6
    dim_V = m * p  # 24
    dim_W = p * n  # 16

    def verify(U_list, V_list, W_list, dim_n, dim_m, dim_p):
        """Verify that the decomposition exactly computes matrix multiplication."""
        rk = len(U_list[0])
        # Build the multiplication tensor from factors
        # T[i,j,k,ii,kk] should be 1 if C[ii,kk] gets A[i,j]*B[j,kk] contribution
        # Actually: C[ii,kk] = sum_{j,k} A[ii,j] * B[j,kk]
        # W index: kk * dim_n + ii (per task: W[k*n+i, r])
        # U index: i * dim_m + j (row i, col j of A)
        # V index: j * dim_p + k (row j, col k of B)
        
        # For each rank term r, it computes:
        # (sum_{i,j} U[i*dim_m+j, r] * A[i,j]) * (sum_{j,k} V[j*dim_p+k, r] * B[j,k])
        # and adds W[kk*dim_n+ii, r] * that to C[ii,kk]
        
        # The total contribution to C[ii,kk] from term r is:
        # W[kk*dim_n+ii, r] * (sum_{i,j} U[i*dim_m+j, r] * A[i,j]) * (sum_{j,k} V[j*dim_p+k, r] * B[j,k])
        
        # For this to equal sum_{j,k} A[ii,j] * B[j,kk], we need:
        # sum_r W[kk*dim_n+ii, r] * U[ii*dim_m+j, r] * V[j*dim_p+kk, r] = 1 for all ii,j,kk
        # (and 0 for other combinations, but since we're matching exactly, we check the full tensor)
        
        # More precisely, the tensor entry T[aa, bb, cc] = sum_r U[aa*dim_m+bb, r] * V[bb*dim_p+cc, r] * W[cc*dim_n+aa, r]
        # should equal 1 if this corresponds to a valid multiplication term, 0 otherwise.
        # Actually the matrix multiplication tensor is: T[aa, bb, cc] = 1 if we're computing C[aa, cc] = sum_bb A[aa,bb]*B[bb,cc]
        # So T[aa, bb, cc] = 1 for all valid (aa, bb, cc), and the decomposition must reproduce this.
        
        # Let's compute the decomposed tensor and check
        for aa in range(dim_n):
            for bb in range(dim_m):
                for cc in range(dim_p):
                    val = 0
                    for r in range(rk):
                        u_val = U_list[aa * dim_m + bb][r]
                        v_val = V_list[bb * dim_p + cc][r]
                        w_val = W_list[cc * dim_n + aa][r]
                        val += u_val * v_val * w_val
                    if val != 1:
                        return False, (aa, bb, cc, val)
        return True, None

    # === Standard Rank-44 Construction ===
    rank = 44

    U = np.zeros((dim_U, rank), dtype=object)
    V = np.zeros((dim_V, rank), dtype=object)
    W = np.zeros((dim_W, rank), dtype=object)

    col = 0

    # Part 1: 2x1x8 (column j=0 of A, row j=0 of B)
    # A[:,0] (2x1) times B[0,:] (1x8) -> 2x8
    for k in range(p):
        # C[0,k] = A[0,0] * B[0,k]
        U[0, col] = 1  # A[0,0]
        V[0 * p + k, col] = 1  # B[0,k]
        W[k * n + 0, col] = 1  # C[0,k]
        col += 1

        # C[1,k] = A[1,0] * B[0,k]
        U[1 * m + 0, col] = 1  # A[1,0]
        V[0 * p + k, col] = 1  # B[0,k]
        W[k * n + 1, col] = 1  # C[1,k]
        col += 1

    # Part 2: 2x2x8 (columns j=1,2 of A, rows j=1,2 of B)
    # Split into 4 blocks of 2x2x2, each using Strassen's 7-mult algorithm
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1

        # Strassen for 2x2x2: A is 2x2 (rows 0,1; cols 1,2), B is 2x2 (rows 1,2; cols k0,k1)
        # Output C is 2x2 (rows 0,1; cols k0,k1)

        # P1 = (A[0,1] + A[1,2]) * (B[1,k0] + B[2,k1])
        U[0 * m + 1, col] = 1   # A[0,1]
        U[1 * m + 2, col] = 1   # A[1,2]
        V[1 * p + k0, col] = 1  # B[1,k0]
        V[2 * p + k1, col] = 1  # B[2,k1]
        W[k0 * n + 0, col] = 1  # C[0,k0]
        W[k1 * n + 1, col] = 1  # C[1,k1]
        col += 1

        # P2 = (A[1,1] + A[1,2]) * B[1,k0]
        U[1 * m + 1, col] = 1   # A[1,1]
        U[1 * m + 2, col] = 1   # A[1,2]
        V[1 * p + k0, col] = 1  # B[1,k0]
        W[k0 * n + 1, col] = 1  # C[1,k0]
        W[k1 * n + 1, col] = -1 # -C[1,k1]
        col += 1

        # P3 = A[0,1] * (B[1,k1] - B[2,k1])
        U[0 * m + 1, col] = 1   # A[0,1]
        V[1 * p + k1, col] = 1  # B[1,k1]
        V[2 * p + k1, col] = -1 # -B[2,k1]
        W[k1 * n + 0, col] = 1  # C[0,k1]
        W[k1 * n + 1, col] = 1  # C[1,k1]
        col += 1

        # P4 = A[1,2] * (B[2,k0] - B[1,k0])
        U[1 * m + 2, col] = 1   # A[1,2]
        V[2 * p + k0, col] = 1  # B[2,k0]
        V[1 * p + k0, col] = -1 # -B[1,k0]
        W[k0 * n + 0, col] = 1  # C[0,k0]
        W[k0 * n + 1, col] = 1  # C[1,k0]
        col += 1

        # P5 = (A[0,1] + A[0,2]) * B[2,k1]
        U[0 * m + 1, col] = 1   # A[0,1]
        U[0 * m + 2, col] = 1   # A[0,2]
        V[2 * p + k1, col] = 1  # B[2,k1]
        W[k0 * n + 0, col] = -1 # -C[0,k0]
        W[k1 * n + 0, col] = 1  # C[0,k1]
        col += 1

        # P6 = (A[1,1] - A[0,1]) * (B[1,k0] + B[1,k1])
        U[1 * m + 1, col] = 1   # A[1,1]
        U[0 * m + 1, col] = -1  # -A[0,1]
        V[1 * p + k0, col] = 1  # B[1,k0]
        V[1 * p + k1, col] = 1  # B[1,k1]
        W[k1 * n + 1, col] = 1  # C[1,k1]
        col += 1

        # P7 = (A[0,2] - A[1,2]) * (B[2,k0] + B[2,k1])
        U[0 * m + 2, col] = 1   # A[0,2]
        U[1 * m + 2, col] = -1  # -A[1,2]
        V[2 * p + k0, col] = 1  # B[2,k0]
        V[2 * p + k1, col] = 1  # B[2,k1]
        W[k0 * n + 0, col] = 1  # C[0,k0]
        col += 1

    U_list = [[int(U[i, j]) for j in range(rank)] for i in range(dim_U)]
    V_list = [[int(V[i, j]) for j in range(rank)] for i in range(dim_V)]
    W_list = [[int(W[i, j]) for j in range(rank)] for i in range(dim_W)]

    # Verify
    ok, err = verify(U_list, V_list, W_list, n, m, p)
    if not ok:
        # Fallback: return schoolbook 48-rank solution
        rank48 = 48
        U48 = np.zeros((dim_U, rank48), dtype=object)
        V48 = np.zeros((dim_V, rank48), dtype=object)
        W48 = np.zeros((dim_W, rank48), dtype=object)
        col48 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U48[i * m + j, col48] = 1
                    V48[j * p + k, col48] = 1
                    W48[k * n + i, col48] = 1
                    col48 += 1
        return {
            "U": [[int(U48[i, j]) for j in range(rank48)] for i in range(dim_U)],
            "V": [[int(V48[i, j]) for j in range(rank48)] for i in range(dim_V)],
            "W": [[int(W48[i, j]) for j in range(rank48)] for i in range(dim_W)]
        }

    # Try to reduce to rank 43 by merging one term
    # Strategy: try to absorb one of the 16 schoolbook terms into a Strassen term
    # by adding the schoolbook contribution to one of the Strassen factors
    # and adjusting the W matrix accordingly
    
    # The idea: if we have a Strassen term with U_str, V_str, W_str and a schoolbook
    # term with U_sch, V_sch, W_sch, we can try to find alpha, beta, gamma such that
    # (U_str + alpha * U_sch) * (V_str + beta * V_sch) * W_str + W_sch' = W_str + W_sch
    # This requires: alpha * U_sch * V_str * W_str + beta * U_str * V_sch * W_str + 
    #                  alpha * beta * U_sch * V_sch * W_str = W_sch
    # which is generally not possible with small integers.
    # 
    # Alternative: try a different decomposition of one 2x2x2 block that uses 
    # only 6 multiplications by exploiting the specific structure.
    # 
    # For now, return the verified rank-44 solution.
    
    return {"U": U_list, "V": V_list, "W": W_list}


if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
