"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # We want to decompose C = A * B where A is 4x4, B is 4x5
    # C is 4x5
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    # The decomposition is: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, that's not quite right. Let me think about the correct convention.
    
    # Actually, the standard bilinear form for matrix multiplication is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # A rank-1 term in tensor decomposition has the form:
    # C[i,k] += (sum_j U[i,j,r] * A[i,j]) * (sum_j V[j,k,r] * B[j,k]) * W[k,i,r]
    
    # But that would require U and V to be functions of i and k respectively, 
    # which doesn't match the shape constraints.
    
    # Let me re-read the problem. The shapes are:
    # U: (n*m, rank) = (16, rank), indexed by (i,j)
    # V: (m*p, rank) = (20, rank), indexed by (j,k)  
    # W: (p*n, rank) = (20, rank), indexed by (k,i)
    
    # The correct interpretation is:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
    
    # But wait, that would mean for each r, we compute:
    # f_r(A) = sum_j U[i*m+j, r] * A[i,j]  (depends on i)
    # g_r(B) = sum_j V[j*p+k, r] * B[j,k]  (depends on k)
    # C[i,k] += W[k*n+i, r] * f_r(A) * g_r(B)
    
    # For this to equal sum_j A[i,j] * B[j,k], we need specific structures.
    
    # Let me use the simplest correct decomposition first: rank 20 (4*5)
    # For each (i,k) pair, use one term that directly computes C[i,k]
    
    rank = 20  # 4 * 5
    U = [[0] * rank for _ in range(n * m)]  # 16 x 20
    V = [[0] * rank for _ in range(m * p)]  # 20 x 20
    W = [[0] * rank for _ in range(p * n)]  # 20 x 20
    
    col = 0
    for i in range(n):
        for k in range(p):
            # For C[i,k] = sum_j A[i,j] * B[j,k]
            # We need: C[i,k] = W[k*n+i, col] * (sum_j U[i*m+j, col] * A[i,j]) * (sum_j V[j*p+k, col] * B[j,k])
            # This doesn't work directly because we're multiplying two sums.
            
            # Actually, the correct form for a rank-1 term is:
            # C[i,k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # summed over j and r? No, that's not standard either.
            
            # Let me think differently. The tensor T[i,j,k] for matmul satisfies:
            # C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]... no.
            
            # Standard: C[i,k] = sum_j A[i,j] B[j,k]
            # So the "tensor" has one index for each of i, j, k, and the value is 1.
            # A rank-1 decomposition: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            # But that's not quite right dimensionally.
            
            # Let me just use the schoolbook algorithm directly:
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # This is rank 4*5 = 20 if we use one term per (i,k) pair... 
            # but actually each term handles one (i,j,k) triple, so it's rank 4*4*5 = 80.
            
            # Wait, no. Let me re-examine.
            
            # The bilinear form is: C[i,k] = sum_j A[i,j] B[j,k]
            # This can be written as: C[i,k] = sum_{i,j,k} delta_{ik}... no.
            
            # Let me use the correct tensor decomposition convention:
            # The multiplication tensor T has indices (i, j, k) where:
            # T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
            # But C[i,k] = sum_j A[i,j] B[j,k], so for fixed (i,k), we sum over j.
            
            # A rank-1 term is: U[i,j] * V[j,k] * W[k,i] (with appropriate indexing)
            # such that sum_r U_r[i,j] V_r[j,k] W_r[k,i] = T[i,j,k]
            
            # For the schoolbook: T[i,j,k] = 1 for all valid (i,j,k)
            # So we can use rank-1 terms where for a specific (i,j,k):
            # U[i,j] = 1, V[j,k] = 1, W[k,i] = 1, and all other entries 0.
            # This gives rank = 4*4*5 = 80.
            
            # But we can do better! For each fixed j, we can combine:
            # C[i,k] += A[i,j] * B[j,k]
            # This is already a product of two linear forms in A and B.
            # So for each (i,j) pair and each (j,k) pair, we have a term.
            # But that's still 4*4*4*5 = 320... no.
            
            # Let me just use the simplest correct approach: rank 20.
            # For each (i,k) pair, we want C[i,k] = sum_j A[i,j] B[j,k]
            # We can achieve this with 4 terms (one for each j) per (i,k) pair.
            # Total: 4 * 4 * 5 = 80 terms.
            
            # OR, we can use rank 20 by being clever:
            # For term r corresponding to (i_r, k_r):
            # U[i_r*m + j, r] = 1 for all j
            # V[j*p + k_r, r] = 1 for all j
            # W[k_r*n + i_r, r] = 1
            # Then: C[i_r, k_r] += sum_j (1 * A[i_r,j]) * (1 * B[j,k_r]) * 1 = sum_j A[i_r,j] B[j,k_r]
            # But this only works if U and V don't depend on the other indices...
            
            # Actually, in the decomposition, U[i*m+j, r] is a coefficient for A[i,j] in term r.
            # If we set U[i_r*m+j, r] = 1 for all j, and 0 otherwise,
            # and V[j*p+k_r, r] = 1 for all j, and 0 otherwise,
            # and W[k_r*n+i_r, r] = 1,
            # then the contribution to C[i,k] from term r is:
            # sum_{j} U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r]... 
            # This doesn't match the standard form.
            
            # I think the correct form is:
            # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
            # But this requires the inner sums to be scalars, which they are,
            # but then we're multiplying three scalars, which gives the wrong result.
            
            # Let me just use rank 80 (schoolbook) to be safe.
            pass
    
    # Actually, let me use the simplest correct decomposition: rank 80
    rank = 80
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Verify with random test
    if time.time() < deadline - 10:
        try:
            np.random.seed(42)
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            C_expected = A_test @ B_test
            
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            
            C_computed = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for r in range(rank):
                            C_computed[i, k] += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
            
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                # Decomposition is incorrect; return zeros to indicate failure
                U = [[0] * rank for _ in range(n * m)]
                V = [[0] * rank for _ in range(m * p)]
                W = [[0] * rank for _ in range(p * n)]
        except Exception:
            pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
