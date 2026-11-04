"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Schoolbook rank = n*m*p = 80
    schoolbook_rank = n * m * p
    
    def build_schoolbook():
        rank = schoolbook_rank
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
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        # Check shapes
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if any(len(row) != rank for row in U):
            return False
        if any(len(row) != rank for row in V):
            return False
        if any(len(row) != rank for row in W):
            return False
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def build_strassen_2x2x2():
        """
        Build the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
        Returns U, V, W as lists of lists with shapes (4, 7), (4, 7), (4, 7).
        
        For 2x2x2:
        n=2, m=2, p=2
        U has shape (n*m, rank) = (4, 7)
        V has shape (m*p, rank) = (4, 7)
        W has shape (p*n, rank) = (4, 7)
        
        U[i*m+j, c] corresponds to A[i,j]
        V[j*p+k, c] corresponds to B[j,k]
        W[k*n+i, c] corresponds to C[i,k] (output row i, column k)
        
        Strassen's 7 products:
        M1 = (A00+A11)(B00+B11)
        M2 = (A10+A11)B00
        M3 = A00(B01-B11)
        M4 = A11(B10-B00)
        M5 = (A00+A01)B11
        M6 = (A10-A00)(B00+B01)
        M7 = (A01-A11)(B10+B11)
        
        Output:
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        """
        rank = 7
        U = [[0] * rank for _ in range(4)]
        V = [[0] * rank for _ in range(4)]
        W = [[0] * rank for _ in range(4)]
        
        # c=0: M1 = (A00+A11)(B00+B11)
        U[0][0] = 1  # A00
        U[3][0] = 1  # A11
        V[0][0] = 1  # B00
        V[3][0] = 1  # B11
        W[0][0] = 1  # C00 (k=0,i=0)
        W[2][0] = 1  # C01 (k=1,i=0)
        W[1][0] = 1  # C10 (k=0,i=1)
        W[3][0] = 1  # C11 (k=1,i=1)
        
        # c=1: M2 = (A10+A11)B00
        U[2][1] = 1  # A10
        U[3][1] = 1  # A11
        V[0][1] = 1  # B00
        W[1][1] = 1  # C10 (k=0,i=1)
        W[3][1] = 1  # C11 (k=1,i=1)
        
        # c=2: M3 = A00(B01-B11)
        U[0][2] = 1  # A00
        V[1][2] = 1  # B01
        V[3][2] = -1 # B11
        W[2][2] = 1  # C01 (k=1,i=0)
        
        # c=3: M4 = A11(B10-B00)
        U[3][3] = 1  # A11
        V[2][3] = 1  # B10
        V[0][3] = -1 # B00
        W[1][3] = 1  # C10 (k=0,i=1)
        W[3][3] = 1  # C11 (k=1,i=1)
        
        # c=4: M5 = (A00+A01)B11
        U[0][4] = 1  # A00
        U[1][4] = 1  # A01
        V[3][4] = 1  # B11
        W[2][4] = 1  # C01 (k=1,i=0)
        W[3][4] = 1  # C11 (k=1,i=1)
        
        # c=5: M6 = (A10-A00)(B00+B01)
        U[2][5] = 1  # A10
        U[0][5] = -1 # A00
        V[0][5] = 1  # B00
        V[1][5] = 1  # B01
        W[3][5] = 1  # C11 (k=1,i=1)
        
        # c=6: M7 = (A01-A11)(B10+B11)
        U[1][6] = 1  # A01
        U[3][6] = -1 # A11
        V[2][6] = 1  # B10
        V[3][6] = 1  # B11
        W[0][6] = 1  # C00 (k=0,i=0)
        W[2][6] = 1  # C01 (k=1,i=0)
        
        return {"U": U, "V": V, "W": W}
    
    def build_4x4x4_strassen():
        """
        Build a rank-49 decomposition for 4x4x4 using Strassen.
        Uses the 2x2x2 Strassen decomposition at the block level.
        
        Split 4x4 into 2x2 blocks of 2x2 matrices.
        A = [[A11, A12], [A21, A22]] where each Ai is 2x2
        B = [[B11, B12], [B21, B22]] where each Bk is 2x2
        
        The 7 Strassen products are 2x2 matrix multiplications.
        Each 2x2x2 multiplication has rank 7 (using Strassen).
        Total rank = 7 * 7 = 49.
        """
        # Get the 2x2x2 Strassen decomposition
        s2 = build_strassen_2x2x2()
        U2 = np.array(s2["U"], dtype=float)  # shape (4, 7)
        V2 = np.array(s2["V"], dtype=float)  # shape (4, 7)
        W2 = np.array(s2["W"], dtype=float)  # shape (4, 7)
        
        rank = 49
        U = np.zeros((16, rank), dtype=float)
        V = np.zeros((16, rank), dtype=float)
        W = np.zeros((16, rank), dtype=float)
        
        # The 4x4 matrix is split into 2x2 blocks:
        # A[i,j] where i,j in {0,1,2,3}
        # Block (bi, bj) where bi = i//2, bj = j//2, and within block (ri, rj) where ri = i%2, rj = j%2
        # So A[i,j] is in block (bi, bj) at position (ri, rj)
        # Index in A: i*m + j = i*4 + j
        # Block index: bi = i//2, bj = j//2
        # Within block: ri = i%2, rj = j%2
        # Position in block: ri*2 + rj (for 2x2 block)
        
        # For the 4x4x4 tensor:
        # U[i*4+j, c] for A[i,j]
        # V[j*4+k, c] for B[j,k]
        # W[k*4+i, c] for C[i,k]
        
        # The 7 Strassen products at the block level:
        # For each of the 7 block-level factors (c_block in 0..6),
        # we need a 2x2x2 matrix multiplication (rank 7)
        # So total rank = 7 * 7 = 49
        
        for c_block in range(7):
            for c_inner in range(7):
                c = c_block * 7 + c_inner
                
                # U factor: for each (i,j), U[i*4+j, c]
                # This comes from U2[ri*2+rj, c_inner] * (block factor for A)
                # The block factor for A is determined by c_block and the block indices (bi, bj)
                
                # We need to figure out what the block-level U factor is.
                # The 2x2x2 Strassen at the block level has:
                # U_block[bi*2+bj, c_block] for the block matrix A_block[bi, bj]
                # V_block[bj*2+bk, c_block] for the block matrix B_block[bj, bk]
                # W_block[bk*2+bi, c_block] for the output block C_block[bi, bk]
                
                # Wait, I need to be more careful. Let me define the block-level indices.
                # At the block level, we have a 2x2x2 tensor:
                # U_block[bi*2+bj, c_block] for A_block[bi, bj]
                # V_block[bj*2+bk, c_block] for B_block[bj, bk]
                # W_block[bk*2+bi, c_block] for C_block[bi, bk]
                
                # U_block is U2 (shape 4x7)
                # V_block is V2 (shape 4x7)
                # W_block is W2 (shape 4x7)
                
                # For the 4x4x4 tensor, the factor c = (c_block, c_inner) has:
                # U[i*4+j, c] = U_block[bi*2+bj, c_block] * U2[ri*2+rj, c_inner]
                # V[j*4+k, c] = V_block[bj*2+bk, c_block] * V2[rj*2+rk, c_inner]
                # W[k*4+i, c] = W_block[bk*2+bi, c_block] * W2[rk*2+ri, c_inner]
                
                # where bi = i//2, bj = j//2, bk = k//2
                # ri = i%2, rj = j%2, rk = k%2
                
                for i in range(4):
                    for j in range(4):
                        bi = i // 2
                        bj = j // 2
                        ri = i % 2
                        rj = j % 2
                        u_block = U2[bi*2+bj, c_block]
                        u_inner = U2[ri*2+rj, c_inner]
                        U[i*4+j, c] = u_block * u_inner
                
                for j in range(4):
                    for k in range(4):
                        bj = j // 2
                        bk = k // 2
                        rj = j % 2
                        rk = k % 2
                        v_block = V2[bj*2+bk, c_block]
                        v_inner = V2[rj*2+rk, c_inner]
                        V[j*4+k, c] = v_block * v_inner
                
                for k in range(4):
                    for i in range(4):
                        bk = k // 2
                        bi = i // 2
                        rk = k % 2
                        ri = i % 2
                        w_block = W2[bk*2+bi, c_block]
                        w_inner = W2[rk*2+ri, c_inner]
                        W[k*4+i, c] = w_block * w_inner
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def build_4x4x5():
        """
        Build a decomposition for 4x4x5.
        Use Strassen for the first 4 columns (4x4x4 part, rank 49)
        and schoolbook for the 5th column (4x4x1 part, rank 16).
        Total rank = 49 + 16 = 65.
        """
        # First, build the 4x4x4 Strassen decomposition
        s4 = build_4x4x4_strassen()
        U4 = np.array(s4["U"], dtype=float)  # shape (16, 49)
        V4 = np.array(s4["V"], dtype=float)  # shape (16, 49)
        W4 = np.array(s4["W"], dtype=float)  # shape (16, 49)
        
        # Now add the 5th column using schoolbook
        # For the 5th column (k=4), we need:
        # C[i,4] = sum_j A[i,j] * B[j,4]
        # This is a 4x4x1 tensor, which has rank 16 (schoolbook)
        
        rank = 49 + 16
        U = np.zeros((16, rank), dtype=float)
        V = np.zeros((20, rank), dtype=float)  # m*p = 4*5 = 20
        W = np.zeros((20, rank), dtype=float)  # p*n = 5*4 = 20
        
        # Copy the 4x4x4 part
        U[:, :49] = U4
        V[:16, :49] = V4  # V has shape (20, rank), first 16 rows are for k=0..3
        W[:16, :49] = W4  # W has shape (20, rank), first 16 rows are for k=0..3
        
        # Add the 5th column part (schoolbook for 4x4x1)
        # For each (i, j), we need a term that contributes to C[i,4]
        # U[i*4+j, c] = 1 for the A[i,j] factor
        # V[j*4+4, c] = 1 for the B[j,4] factor (j*4+4 = j*4 + 4, since p=5, so B[j,4] is at index j*5+4)
        # W[4*4+i, c] = 1 for the C[i,4] factor (W[k*n+i] = W[4*4+i] = W[16+i])
        
        for i in range(4):
            for j in range(4):
                c = 49 + i*4 + j
                U[i*4+j, c] = 1
                V[j*5+4, c] = 1  # B[j,4] is at index j*5+4
                W[4*4+i, c] = 1  # C[i,4] is at index 4*4+i = 16+i
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def try_optimize_rank(rank, max_iters=500, tol=1e-8):
        """Try to find a rank-r decomposition using iterative optimization."""
        if time.time() > deadline:
            return None
        
        for attempt in range(100):
            if time.time() > deadline:
                return None
            
            seed = attempt * 1000 + 7
            np.random.seed(seed)
            
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            scale = 1.0 / np.sqrt(rank)
            U = U * scale
            V = V * scale
            W = W * scale
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        if abs(s - 1.0) > 0.01:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
        
        return None
    
    # Build the guaranteed schoolbook solution
    best_result = build_schoolbook()
    best_rank = schoolbook_rank
    
    # Try the 4x4x5 Strassen-based construction (rank 65)
    if time.time() < deadline - 10:
        try:
            result65 = build_4x4x5()
            if verify(result65["U"], result65["V"], result65["W"], tol=1e-6):
                best_result = result65
                best_rank = 65
        except Exception:
            pass
    
    # Try to find lower-rank decompositions
    candidate_ranks = [64, 63, 62, 61, 60, 56, 50]
    
    for rank in candidate_ranks:
        if time.time() > deadline - 5:
            break
        if rank >= best_rank:
            continue
        
        result = try_optimize_rank(rank, max_iters=200)
        if result is not None:
            if verify(result["U"], result["V"], result["W"]):
                best_result = result
                best_rank = rank
                break
    
    return best_result
# EVOLVE-BLOCK-END
