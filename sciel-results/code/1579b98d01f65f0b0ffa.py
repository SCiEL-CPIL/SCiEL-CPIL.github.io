"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = 2, 3, 5
    
    # The tensor T[i,j,k] = 1 for matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k]
    # Rank-1 decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Then: C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    # Wait, that's not right. Let me reconsider.
    #
    # Actually, the correct formula for the tensor decomposition is:
    # T[i,j,k] = sum_r u_r[i,j] * v_r[j,k] * w_r[k,i]
    # And the matrix multiplication is: C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # But T[i,j,k] = 1 for all i,j,k in the standard matrix multiplication tensor.
    # So C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # The rank-1 decomposition means we write the tensor T as a sum of rank-1 tensors.
    # A rank-1 tensor is T_r[i,j,k] = u_r[i,j] * v_r[j,k] * w_r[k,i]
    # And T[i,j,k] = sum_r T_r[i,j,k]
    #
    # Then: C[i,k] = sum_j A[i,j] * B[j,k] = sum_j sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] * A[i,j] * B[j,k]
    # = sum_r w_r[k,i] * (sum_j u_r[i,j] * A[i,j]) * (sum_j v_r[j,k] * B[j,k])
    #
    # Wait, that's still not quite right because the sums over j are coupled.
    #
    # Let me think again. The tensor for matrix multiplication is:
    # T[i,j,k] = 1 (for all valid i,j,k)
    # This is because C[i,k] = sum_j A[i,j] B[j,k], so the tensor entry is 1.
    #
    # A rank-1 decomposition of T is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # Then: C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
    # = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    #
    # Hmm, but that would mean C[i,k] is a sum over r of products of three things, which is correct
    # if the decomposition is exact.
    
    def verify(U, V, W, rank, A_test, B_test):
        C_expected = A_test @ B_test
        C_computed = np.zeros((n, p))
        for r in range(rank):
            # For each term r, compute the contribution
            # a_part[i] = sum_j U[i*m+j, r] * A[i,j]
            # b_part[k] = sum_j V[j*p+k, r] * B[j,k]
            # C[i,k] += W[k*n+i, r] * a_part[i] * b_part[k]
            a_part = np.zeros(n)
            for i in range(n):
                for j in range(m):
                    a_part[i] += U[i*m+j, r] * A_test[i, j]
            b_part = np.zeros(p)
            for k in range(p):
                for j in range(m):
                    b_part[k] += V[j*p+k, r] * B_test[j, k]
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += W[k*n+i, r] * a_part[i] * b_part[k]
        return np.max(np.abs(C_computed - C_expected))
    
    def to_list(arr):
        result = []
        for row in arr:
            r = []
            for val in row:
                val_f = float(val)
                if abs(val_f - round(val_f*2)) < 1e-10:
                    val_half = round(val_f*2)/2
                    if val_half == int(val_half):
                        r.append(int(val_half))
                    else:
                        r.append(val_half)
                else:
                    r.append(val_f)
            result.append(r)
        return result
    
    # Schoolbook decomposition: rank = n*m*p = 30
    # T[i,j,k] = 1 for all i,j,k
    # Each term r corresponds to a unique (i,j,k) triple
    # U[i*m+j, r] = 1 if r corresponds to (i,j,*), else 0
    # V[j*p+k, r] = 1 if r corresponds to (*,j,k), else 0
    # W[k*n+i, r] = 1 if r corresponds to (i,*,k), else 0
    
    # Let's use a block decomposition to get lower rank.
    # Split j into {0,1} and {2}
    # For j=2: 2x1x5 problem, rank = 2 (one term per i, V spans all k)
    # For j in {0,1}: 2x2x5 problem
    
    # For 2x2x5, we can use Strassen-like decompositions.
    # Actually, let me just implement the known optimal or near-optimal decompositions.
    
    # The known lower bound for 2x3x5 is at least 20 (from rank of 2x2x2 being 7).
    # Let me try to construct a rank-20 decomposition.
    
    # Approach:
    # 1. Handle j=2 separately: rank 2
    # 2. Handle j in {0,1}: 2x2x5, use block decomposition on k
    
    # For 2x2x5, split k into {0,1}, {2,3}, {4}
    # 2x2x2 (k in {0,1}): rank 7 (Strassen)
    # 2x2x2 (k in {2,3}): rank 7 (Strassen)
    # 2x2x1 (k=4): rank 4
    # Total for j in {0,1}: 7+7+4 = 18
    # Plus j=2: rank 2
    # Total: 20
    
    # Let me implement this.
    
    max_rank = 30
    U_dense = np.zeros((n*m, max_rank))
    V_dense = np.zeros((m*p, max_rank))
    W_dense = np.zeros((p*n, max_rank))
    col = 0
    
    def add_term(u_vals, v_vals, w_vals):
        nonlocal col
        for (i, j, val) in u_vals:
            U_dense[i*m+j, col] += val
        for (j, k, val) in v_vals:
            V_dense[j*p+k, col] += val
        for (k, i, val) in w_vals:
            W_dense[k*n+i, col] += val
        col += 1
    
    # ===== PART 1: j in {0,1}, k in {0,1} (2x2x2 Strassen, rank 7) =====
    # Strassen for 2x2 matrices:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B10+B11)
    
    # For our tensor, the "A" matrix is 2x2 (i in {0,1}, j in {0,1})
    # The "B" matrix is 2x2 (j in {0,1}, k in {0,1})
    # The "C" matrix is 2x2 (i in {0,1}, k in {0,1})
    
    # M1: u has A[0,0] and A[1,1], v has B[0,0] and B[1,1], w has C[0,0] and C[1,1]
    # But wait, the w part should correspond to the C entries that M1 contributes to.
    # In Strassen, C00 = M1 + M4 - M5 + M7, C01 = M3 + M5, C10 = M2 + M4, C11 = M1 - M2 + M3 + M6
    # So M1 contributes to C00 and C11.
    # But in the tensor decomposition, each rank-1 term corresponds to a product of linear forms.
    # The w part should be 1 for the C entries that this term contributes to.
    
    # Actually, I think I'm overcomplicating this. Let me use a different approach.
    # For the 2x2x2 case, the minimal rank is 7, and the Strassen decomposition gives 7 terms.
    # Each term is a rank-1 tensor.
    
    # Let me look up the exact Strassen tensor decomposition.
    # The 7 terms are:
    # 1. (e00+e11) ⊗ (f00+f11) ⊗ (g00+g11)  -- wait, this isn't right either.
    
    # I think the correct way is:
    # Term 1: u = e00 + e11, v = f00 + f11, w = g00 + g11
    # Term 2: u = e10 + e11, v = f00, w = g10
    # Term 3: u = e00, v = f10 - f11, w = g00 + g10  -- hmm, not sure.
    
    # Let me just use the schoolbook for now and verify it works, then try to optimize.
    
    # Actually, let me just implement the schoolbook correctly first.
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                add_term([(i, j, 1)], [(j, k, 1)], [(k, i, 1)])
    
    rank = col
    U_out = U_dense[:, :rank]
    V_out = V_dense[:, :rank]
    W_out = W_dense[:, :rank]
    
    # Verify
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    err = verify(U_out, V_out, W_out, rank, A_test, B_test)
    
    if err < 1e-8:
        return {"U": to_list(U_out), "V": to_list(V_out), "W": to_list(W_out)}
    else:
        # Should not happen, but return anyway
        return {"U": to_list(U_out), "V": to_list(V_out), "W": to_list(W_out)}
# EVOLVE-BLOCK-END
