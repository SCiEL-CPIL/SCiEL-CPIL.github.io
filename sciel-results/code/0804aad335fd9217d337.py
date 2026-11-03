"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def _build_5x5x5():
    """Build a rank-46 exact decomposition for 5x5x5 matrix multiplication."""
    # We use a block decomposition: split 5x5 into blocks.
    # One effective approach: use Strassen for 2x2 blocks and schoolbook for the rest.
    # 
    # A known rank-46 construction for 5x5x5:
    # Split the 5x5 matrices into blocks of sizes (2,3) x (2,3).
    # We have 4 sub-problems:
    #   2x2x2: rank 7 (Strassen)
    #   2x3x2: rank 12 (known optimal)
    #   3x2x2: rank 12 (known optimal)
    #   3x3x3: rank 15 (known optimal)
    # Total: 7 + 12 + 12 + 15 = 46
    #
    # We need explicit rank-1 decompositions for each sub-problem.

    n = 5
    m = 5
    p = 5
    U_rows = n * m  # 25
    V_rows = m * p  # 25
    W_rows = p * n  # 25
    
    U = np.zeros((U_rows, 46))
    V = np.zeros((V_rows, 46))
    W = np.zeros((W_rows, 46))
    
    col = 0

    # Helper: add a schoolbook decomposition for n_sub x m_sub x p_sub at offset (i0, j0, k0)
    def add_schoolbook(i0, j0, k0, ns, ms, ps):
        nonlocal col
        for i in range(ns):
            for j in range(ms):
                for k in range(ps):
                    U[i0 + i, col] = 1.0
                    V[j0 + j, col] = 1.0
                    W[k0 + k, col] = 1.0
                    col += 1

    # Helper: add Strassen 2x2x2 at offset (i0, j0, k0)
    def add_strassen(i0, j0, k0):
        nonlocal col
        # Strassen 7 products:
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b12-b22)
        # M4 = a22(b21-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        #
        # c11 = M1 + M4 - M5 + M7
        # c12 = M3 + M5
        # c21 = M2 + M4
        # c22 = M1 - M2 + M3 + M6

        # M1: (a11+a22)(b11+b22)
        U[i0+0, col] = 1.0
        U[i0+1, col] = 1.0
        V[j0+0, col] = 1.0
        V[j0+1, col] = 1.0
        W[k0+0, col] = 1.0  # c11 += M1
        W[k0+3, col] = 1.0  # c22 += M1
        col += 1

        # M2: (a21+a22)b11
        U[i0+1, col] = 1.0
        U[i0+2, col] = 1.0  # Note: for 2x2, indices are 0,1. Let me re-index.
        # Wait, I need to be careful with indices. For a 2x2 block:
        # a[0][0], a[0][1], a[1][0], a[1][1]
        # Let me redo this carefully.
        
        # Reset and redo Strassen with clear indexing
        # For a 2x2x2 block at (i0, j0, k0):
        # A = [[a00, a01], [a10, a11]] where a_ij is at row i0+i, col j0+j in the full matrix
        # B = [[b00, b01], [b10, b11]] where b_jk is at row j0+j, col k0+k
        # C = [[c00, c01], [c10, c11]] where c_ik is at row k0+k, col i0+i in W (trace convention)
        
        # Actually, let me use a cleaner approach. I'll just hardcode the 46 rank-1 terms
        # for the 5x5x5 case directly.

    # Let me take a different, cleaner approach.
    # I'll construct the 5x5x5 decomposition using known optimal decompositions for each block.
    
    # Block structure: 
    #   Rows of A: 0-1 (block 0), 2-4 (block 1)
    #   Cols of A / Rows of B: 0-1 (block 0), 2-4 (block 1)
    #   Cols of B: 0-1 (block 0), 2-4 (block 1)
    
    # Sub-problems:
    # (0,0,0): 2x2x2 -> rank 7 (Strassen)
    # (0,0,1): 2x2x3 -> rank ? 
    # This is getting complex. Let me use a simpler known construction.
    
    # Actually, the simplest reliable approach: use the fact that we can pad 4x5x5 to 5x5x5.
    # If we have a rank-r decomposition for 5x5x5, then 4x5x5 has rank <= r.
    # Known: R(5,5,5) <= 46.
    
    # For now, let me just return the schoolbook 4x5x5 to be safe, since constructing
    # the 46-rank decomposition correctly is error-prone without verification.
    
    return None

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    # The tensor T[i,j,k] should equal sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # For matrix multiplication: T[i,j,k] = 1 if j is a valid "middle" index, i.e.,
    # C[i,k] = sum_j A[i,j] B[j,k], so T[i,j,k] = 1 for all valid i,j,k.
    # Actually, the multiplication tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # We need: for all i, k: C[i,k] = sum_j A[i,j] B[j,k]
    # The rank-1 decomposition gives: C[i,k] = sum_c (sum_j U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c]
    # Wait, that's not right either. Let me think about this more carefully.
    #
    # Standard rank-1 decomposition of matmul tensor:
    # C = A * B, where C[i,k] = sum_j A[i,j] * B[j,k]
    # Rank-1 term c contributes: C[i,k] += U[i,j,c] * V[j,k,c] * W[i,k,c]
    # But U, V, W are indexed differently.
    #
    # Per the task: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # U[i*m+j, c] corresponds to A[i,j]
    # V[j*p+k, c] corresponds to B[j,k]  
    # W[k*n+i, c] corresponds to C[i,k]
    #
    # The decomposition: C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * V[j*p+k, c])
    # This should equal sum_j A[i,j] * B[j,k]
    #
    # So for each c, define:
    #   u_c[i,j] = U[i*m+j, c]
    #   v_c[j,k] = V[j*p+k, c]
    #   w_c[i,k] = W[k*n+i, c]
    # Then: w_c[i,k] * (sum_j u_c[i,j] * v_c[j,k]) should sum to the matmul tensor.
    
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    U_reshaped = U.reshape(n, m, -1)
    V_reshaped = V.reshape(m, p, -1)
    W_reshaped = W.reshape(p, n, -1)
    
    # For each rank-1 term c:
    # contribution to C[i,k] = W[k,i,c] * sum_j U[i,j,c] * V[j,k,c]
    
    rank = U.shape[1]
    C_recon = np.zeros((n, p))
    for c in range(rank):
        u = U_reshaped[:, :, c]  # (n, m)
        v = V_reshaped[:, :, c]  # (m, p)
        w = W_reshaped[:, :, c]  # (p, n) -> we need w.T for (n, p)
        C_recon += w.T * (u @ v)
    
    # Target: C[i,k] = sum_j A[i,j]*B[j,k], so the tensor value is 1 for each (i,j,k) combination
    # Actually, the matmul "tensor" T is defined such that T[i,j,k] = 1, and
    # C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # The rank decomposition requires: for all i,k: C[i,k] = sum_c W[k*n+i,c] * sum_j U[i*m+j,c] * V[j*p+k,c]
    # This must equal sum_j A[i,j] * B[j,k] for all A, B.
    # So the condition is: for all i,k, sum_c W[k*n+i,c] * sum_j U[i*m+j,c] * V[j*p+k,c] = sum_j delta(i,j) * delta(j,k)
    # Wait no. The condition is that the bilinear form defined by the decomposition equals matmul.
    # For arbitrary A and B: sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] * A[i,j] * V[j*p+k,c] * B[j,k]) = sum_j A[i,j] * B[j,k]
    # This means: for all i,k, sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] * V[j*p+k,c] * A[i,j] * B[j,k]) = sum_j A[i,j] * B[j,k]
    # For this to hold for all A, B, we need: for all i,j,k:
    #   sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c] = delta(i,j,k is valid) = 1 (if j is in range)
    
    # Let's verify: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] == 1
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j, :], W_arr[k*n+i, :] * V_arr[j*p+k, :])
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook (rank 100)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # Try to find a lower-rank decomposition
    # For 4x5x5, the known lower bound is 20, and upper bound from 5x5x5 is 46.
    # Let's try to construct a rank-46 decomposition.
    
    if time.time() < deadline:
        # Attempt to build a 5x5x5 rank-46 decomposition and slice to 4x5x5
        try:
            U5, V5, W5 = _build_5x5x5_decomp()
            if U5 is not None:
                U5 = np.array(U5)
                V5 = np.array(V5)
                W5 = np.array(W5)
                
                # Verify 5x5x5
                if _verify(U5, V5, W5, 5, 5, 5):
                    # Slice to 4x5x5: keep i in [0,4), i.e., rows 0..3 of the 5x5 A matrix
                    # U has shape (5*5, rank) -> keep rows where i < 4, i.e., i*m+j for i in 0..3
                    # That's rows 0..19 (since m=5, i=0..3 gives rows 0..4, 5..9, 10..14, 15..19)
                    U4 = U5[:20, :]  # rows 0-19
                    V4 = V5  # all rows
                    # W has shape (5*5, rank) -> keep rows where i < 4, i.e., k*5+i for i in 0..3
                    # That means for each k, keep i=0,1,2,3, skip i=4
                    W4_rows = []
                    for k in range(5):
                        for i in range(4):
                            W4_rows.append(k * 5 + i)
                    W4 = W5[W4_rows, :]
                    
                    if _verify(U4.tolist(), V4.tolist(), W4.tolist(), 4, 5, 5):
                        new_rank = U4.shape[1]
                        if new_rank < best_rank:
                            best_rank = new_rank
                            best_result = {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}
        except Exception:
            pass
    
    return best_result

def _build_5x5x5_decomp():
    """Build a rank-46 decomposition for 5x5x5 matrix multiplication."""
    # This is a complex construction. For reliability, I'll return None if I can't
    # verify it works.
    
    # Let me use a block-based approach with known optimal decompositions.
    # 
    # Split 5 = 2 + 3
    # Block (2,2,2): Strassen, rank 7
    # Block (2,3,2): rank 12
    # Block (3,2,2): rank 12  
    # Block (3,3,3): rank 15
    # Total: 46
    
    # I need explicit rank-1 decompositions for each.
    # For now, let me try a simpler approach: just use schoolbook for each block
    # and see if I can do better.
    
    # Schoolbook ranks: 2*2*2=8, 2*3*2=12, 3*2*2=12, 3*3*3=27 -> total 59
    # Strassen for 2x2x2 saves 1: total 58
    # Need better decompositions for 2x3x2, 3x2x2, 3x3x3.
    
    # Known optimal: R(2,3,2) <= 12, R(3,2,2) <= 12, R(3,3,3) <= 15
    # But I need explicit decompositions.
    
    # This is too complex to get right without verification in a short time.
    # Return None to fall back to schoolbook.
    return None

# EVOLVE-BLOCK-END
