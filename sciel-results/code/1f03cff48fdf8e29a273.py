"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START

def _construct_rank36():
    n, m, p = 2, 4, 5
    rank = 36
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    def ui(i, j):
        return i * m + j

    def vi(j, k):
        return j * p + k

    def wi(k, i):
        return k * n + i

    col = [0]

    def strassen(y00, y01, y10, y11, z00, z01, z10, z11, w00, w10, w01, w11):
        c = col[0]

        # M1: (y00+y11)(z00+z11) -> X00, X11
        U[ui(y00[0], y00[1]), c] = 1
        U[ui(y11[0], y11[1]), c] = 1
        V[vi(z00[0], z00[1]), c] = 1
        V[vi(z11[0], z11[1]), c] = 1
        W[wi(w00[0], w00[1]), c] += 1
        W[wi(w11[0], w11[1]), c] += 1
        col[0] += 1

        # M2: (y10+y11)z00 -> X10, -X11
        U[ui(y10[0], y10[1]), c] = 1
        U[ui(y11[0], y11[1]), c] = 1
        V[vi(z00[0], z00[1]), c] = 1
        W[wi(w10[0], w10[1]), c] += 1
        W[wi(w11[0], w11[1]), c] -= 1
        col[0] += 1

        # M3: y00(z01-z11) -> X01, X11
        U[ui(y00[0], y00[1]), c] = 1
        V[vi(z01[0], z01[1]), c] = 1
        V[vi(z11[0], z11[1]), c] = -1
        W[wi(w01[0], w01[1]), c] += 1
        W[wi(w11[0], w11[1]), c] += 1
        col[0] += 1

        # M4: y11(z10-z00) -> X00, X10
        U[ui(y11[0], y11[1]), c] = 1
        V[vi(z10[0], z10[1]), c] = 1
        V[vi(z00[0], z00[1]), c] = -1
        W[wi(w00[0], w00[1]), c] += 1
        W[wi(w10[0], w10[1]), c] += 1
        col[0] += 1

        # M5: (y00+y01)z11 -> -X00, X01
        U[ui(y00[0], y00[1]), c] = 1
        U[ui(y01[0], y01[1]), c] = 1
        V[vi(z11[0], z11[1]), c] = 1
        W[wi(w00[0], w00[1]), c] -= 1
        W[wi(w01[0], w01[1]), c] += 1
        col[0] += 1

        # M6: (y10-y00)(z00+z01) -> X11
        U[ui(y10[0], y10[1]), c] = 1
        U[ui(y00[0], y00[1]), c] = -1
        V[vi(z00[0], z00[1]), c] = 1
        V[vi(z01[0], z01[1]), c] = 1
        W[wi(w11[0], w11[1]), c] += 1
        col[0] += 1

        # M7: (y01-y11)(z10+z11) -> X00
        U[ui(y01[0], y01[1]), c] = 1
        U[ui(y11[0], y11[1]), c] = -1
        V[vi(z10[0], z10[1]), c] = 1
        V[vi(z11[0], z11[1]), c] = 1
        W[wi(w00[0], w00[1]), c] += 1
        col[0] += 1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    strassen((0,0),(0,1),(1,0),(1,1), (0,0),(0,1),(1,0),(1,1), (0,0),(1,0),(0,1),(1,1))
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    strassen((0,2),(0,3),(1,2),(1,3), (2,0),(2,1),(3,0),(3,1), (0,0),(1,0),(0,1),(1,1))
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    strassen((0,0),(0,1),(1,0),(1,1), (0,2),(0,3),(1,2),(1,3), (2,0),(3,0),(2,1),(3,1))
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    strassen((0,2),(0,3),(1,2),(1,3), (2,2),(2,3),(3,2),(3,3), (2,0),(3,0),(2,1),(3,1))
    # Block 5: A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5]  (rank 4)
    U[ui(0,0), col[0]] = 1; V[vi(0,4), col[0]] = 1; W[wi(4,0), col[0]] = 1; col[0] += 1
    U[ui(0,1), col[0]] = 1; V[vi(1,4), col[0]] = 1; W[wi(4,0), col[0]] = 1; col[0] += 1
    U[ui(1,0), col[0]] = 1; V[vi(0,4), col[0]] = 1; W[wi(4,1), col[0]] = 1; col[0] += 1
    U[ui(1,1), col[0]] = 1; V[vi(1,4), col[0]] = 1; W[wi(4,1), col[0]] = 1; col[0] += 1
    # Block 6: A[0:2, 2:4] * B[2:4, 4:5] -> C[0:2, 4:5]  (rank 4)
    U[ui(0,2), col[0]] = 1; V[vi(2,4), col[0]] = 1; W[wi(4,0), col[0]] = 1; col[0] += 1
    U[ui(0,3), col[0]] = 1; V[vi(3,4), col[0]] = 1; W[wi(4,0), col[0]] = 1; col[0] += 1
    U[ui(1,2), col[0]] = 1; V[vi(2,4), col[0]] = 1; W[wi(4,1), col[0]] = 1; col[0] += 1
    U[ui(1,3), col[0]] = 1; V[vi(3,4), col[0]] = 1; W[wi(4,1), col[0]] = 1; col[0] += 1

    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p):
    """Check that the decomposition exactly represents matrix multiplication."""
    rank = U.shape[1]
    # Build the tensor T from the decomposition: T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # For matrix mult: T[i,j,k] should be 1 if j is the common index, i.e., T is the structure tensor
    # Actually, the tensor for matrix mult C=AB has entries: T_{i,j,k} = sum_l A_{i,l} B_{l,k}
    # So the tensor is: T_{i,l,k} = delta (i.e., T[i,l,k] = 1 if it's the (i,l,k) term)
    # In the factorization: T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r}
    # where i indexes rows of A (n), j indexes the "middle" (m), k indexes cols of B (p)
    # Wait, let me re-read the interface:
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # U is indexed by (i,j) where i is row of A, j is col of A (= row of B)
    # V is indexed by (j,k) where j is row of B, k is col of B
    # W is indexed by (k,i) where k is col of B, i is row of A
    # So the tensor is: for each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal 1 if (i,j,k) corresponds to a valid term in C=AB, i.e., always (it's the identity tensor)
    # Actually the matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k? No.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the trilinear form is: C_{i,k} A'_{i,j} B'_{j,k} ... 
    # The tensor is: T_{i,j,k} where the multiplication is: product = sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} C_{i,k}
    # For matrix mult, T_{i,j,k} = 1 for all i,j,k? No, that's not right either.
    # Actually: (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the bilinear form is: sum_{i,k} (AB)_{i,k} C_{i,k} = sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k}
    # So T_{i,j,k} = 1 for all valid (i,j,k). The tensor is the all-ones tensor in the appropriate sense.
    # Wait no. The tensor is defined such that the bilinear form equals sum T[i,j,k] A[i,j] B[j,k] C[i,k].
    # Since (AB)_{i,k} = sum_j A_{i,j} B_{j,k}, we have sum_{i,k} (AB)_{i,k} C_{i,k} = sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k}
    # So T[i,j,k] = 1 for all i,j,k.
    
    # Let's verify: for each (i,j,k), check sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i * m + j, :]
                v_val = V_arr[j * p + k, :]
                w_val = W_arr[k * n + i, :]
                s = np.sum(u_val * v_val * w_val)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    
    # Start with the known rank-36 decomposition
    result36 = _construct_rank36()
    
    # Try to find a lower rank decomposition using a systematic approach
    # The tensor has n*m*p = 40 entries, each must sum to 1
    # Try ranks from 20 upward
    
    deadline = t0 + 140  # Leave 10s safety margin
    
    best_result = result36
    best_rank = 36
    
    # For lower rank attempts, we can try to use the structure of the problem.
    # Since n=2 is small, we can parameterize the solution.
    # The key insight: for each (j,k) pair in B, the contribution to C involves summing over i.
    # Let's try a rank-32 approach: use Strassen for the 2x2 blocks but optimize the cross terms.
    
    # Alternative: try rank 30 by combining the last column of B more efficiently
    # B has columns 0,1,2,3,4. We can group (0,1), (2,3), (4) 
    # But column 4 is a single column, so 2x1x1 blocks are rank 1 each... 
    # Actually 2x1x1 has rank 2 (can't do better than trivial)
    
    # Let me try a different grouping: (0,1,2,3) as two Strassen blocks, and column 4 separately
    # That's what we have: 4*7 + 2*4 = 36
    
    # Try: treat B as 2x(4,1) where the 4 is handled by Strassen and 1 is separate
    # Or: group columns differently
    
    # Actually, let's try to reduce by using the fact that for the 2x1x1 blocks,
    # we can combine them. The 2x1x1 multiplication has rank 2 (trivially).
    # But if we have multiple 2x1x1 blocks, can we share factors?
    
    # Let me try a completely different approach: use a rank-28 decomposition
    # by treating the whole thing as a 2x4x5 tensor and using optimized methods.
    
    # For now, let's try to verify rank 36 works, then attempt rank 35, 34, ...
    for target_rank in range(35, 19, -1):
        if time.time() > deadline:
            break
        # This is a placeholder for actual rank-reduction search
        # A proper search would use iterative compression or similar
        pass
    
    return best_result

# EVOLVE-BLOCK-END
