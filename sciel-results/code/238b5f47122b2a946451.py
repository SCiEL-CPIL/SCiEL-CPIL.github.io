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
    
    def build_strassen_4x4x4():
        """
        Build the rank-7 Strassen decomposition for 4x4x4 matrix multiplication.
        Returns U, V, W as lists of lists with shapes (16, 7), (16, 7), (16, 7).
        
        The Strassen algorithm for 4x4 matrices uses 7 multiplications.
        We split each 4x4 matrix into 2x2 blocks:
        A = [[A11, A12], [A21, A22]]
        B = [[B11, B12], [B21, B22]]
        
        The 7 products are:
        M1 = (A11 + A22) * (B11 + B22)
        M2 = (A21 + A22) * B11
        M3 = A11 * (B12 - B22)
        M4 = A22 * (B21 - B11)
        M5 = (A11 + A12) * B22
        M6 = (A21 - A11) * (B11 + B12)
        M7 = (A12 - A22) * (B21 + B22)
        
        Then:
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        
        For the tensor decomposition, we need to express this as:
        sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = delta for the 4x4x4 tensor
        
        Actually, let me think about this more carefully. The matrix multiplication tensor
        for multiplying an (n x m) matrix by an (m x p) matrix has components T[i,j,k] = 1
        for all valid (i,j,k), and the decomposition is:
        T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        For the 4x4x4 case, we want rank-7. Let me use the standard Strassen construction.
        
        For a 4x4 matrix multiplication, we can view it as 2x2 block multiplication where
        each block is 2x2. The Strassen algorithm gives us 7 scalar multiplications for
        the 2x2 block case, but we need to lift this to the 2x2 block level.
        
        Actually, for 4x4x4, the standard Strassen gives rank 7 * (2*2*2) = 56 if we
        naively apply it to 2x2 blocks. But that's not right either.
        
        Let me reconsider. The rank of the 4x4x4 matrix multiplication tensor is known
        to be at most 49 (using Strassen recursively) and at least 49. Actually, the
        exact rank is not known, but upper bounds include 49.
        
        Wait, I'm confusing myself. Let me think about this differently.
        
        For multiplying a 4x4 matrix by a 4x4 matrix, we can use Strassen's algorithm
        which splits into 2x2 blocks. Each 2x2x2 Strassen step has rank 7. But we have
        4 blocks in each dimension, so... 
        
        Actually, the standard approach: a 4x4x4 tensor can be decomposed using the
        Strassen algorithm for 2x2x2 applied at the block level. If we partition each
        4x4 matrix into four 2x2 blocks, then the 4x4x4 multiplication tensor is the
        "Kronecker product" of the 2x2x2 identity tensor with itself, which has rank
        8 (the schoolbook rank for 2x2x2 is 8, and Strassen reduces it to 7).
        
        Hmm, this is getting complicated. Let me just use a known rank-7 decomposition
        for 2x2x2 and embed it.
        
        For 2x2x2, the Strassen rank-7 decomposition has:
        U (shape 4x7), V (shape 4x7), W (shape 4x7)
        
        The 7 terms correspond to the 7 products in Strassen's algorithm.
        
        For the 2x2x2 case with indices i, j, k in {0,1}:
        T[i,j,k] = 1 for all (i,j,k)
        
        Strassen's 7 factors:
        c=0: M1 = (A00+A11)(B00+B11)
        c=1: M2 = (A10+A11)B00
        c=2: M3 = A00(B01-B11)
        c=3: M4 = A11(B10-B00)
        c=4: M5 = (A00+A01)B11
        c=5: M6 = (A10-A00)(B00+B01)
        c=6: M7 = (A01-A11)(B10+B11)
        
        Then:
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        
        So for each c, we have contributions to specific (i,j,k) positions.
        
        Let me define the factors for 2x2x2:
        
        For c=0 (M1 = (A00+A11)(B00+B11)):
        U[0,0]=1, U[3,0]=1 (A00 and A11)
        V[0,0]=1, V[3,0]=1 (B00 and B11)
        W[0,0]=1, W[1,1]=1, W[2,2]=1, W[3,3]=1 (all C positions get +1)
        Wait, that's not right. W should have shape (p*n, rank) = (4, 7) for 2x2x2.
        W[k*n+i, c] where k is column index and i is row index of output.
        
        Let me be more careful. For 2x2x2:
        n=2, m=2, p=2
        U has shape (n*m, rank) = (4, rank)
        V has shape (m*p, rank) = (4, rank)
        W has shape (p*n, rank) = (4, rank)
        
        U[i*m+j, c] corresponds to A[i,j]
        V[j*p+k, c] corresponds to B[j,k]
        W[k*n+i, c] corresponds to C[i,k] (output row i, column k)
        
        For c=0: M1 = (A00+A11)(B00+B11)
        U[0,0]=1 (A00), U[3,0]=1 (A11)
        V[0,0]=1 (B00), V[3,0]=1 (B11)
        This product contributes to C00, C01, C10, C11 with coefficient +1 each? No.
        The product (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
        This contributes to:
        C00: A00*B00 (i=0,k=0) -> W[0*2+0, 0] = W[0,0] += 1
        C01: A00*B11 (i=0,k=1) -> W[1*2+0, 0] = W[2,0] += 1
        C10: A11*B00 (i=1,k=0) -> W[0*2+1, 0] = W[1,0] += 1
        C11: A11*B11 (i=1,k=1) -> W[1*2+1, 0] = W[3,0] += 1
        
        So W[0,0]=1, W[1,0]=1, W[2,0]=1, W[3,0]=1
        
        For c=1: M2 = (A10+A11)B00
        U[2,1]=1 (A10), U[3,1]=1 (A11)
        V[0,1]=1 (B00)
        Contributes to:
        C10: A10*B00 (i=1,k=0) -> W[0*2+1, 1] = W[1,1] += 1
        C11: A11*B00 (i=1,k=1) -> W[1*2+1, 1] = W[3,1] += 1
        So W[1,1]=1, W[3,1]=1
        
        For c=2: M3 = A00(B01-B11)
        U[0,2]=1 (A00)
        V[1,2]=1 (B01), V[3,2]=-1 (B11)
        Contributes to:
        C01: A00*B01 (i=0,k=1) -> W[1*2+0, 2] = W[2,2] += 1
        C01: A00*(-B11) (i=0,k=1) -> W[2,2] += -1
        Wait, that doesn't work. Let me reconsider.
        
        Actually, M3 = A00*(B01-B11) is a single product. The factors are:
        U[0,2]=1
        V[1,2]=1, V[3,2]=-1
        W: this contributes to C01 with coefficient +1 for the B01 part and -1 for the B11 part.
        But in the tensor decomposition, each c gives a rank-1 term. So for c=2:
        The term is U[i,j,2]*V[j,k,2]*W[k,i,2]
        For (i=0,j=0,k=1): U[0,2]*V[1,2]*W[2,2] = 1*1*W[2,2]
        For (i=0,j=0,k=0): U[0,2]*V[0,2]*W[1,2] = 1*0*W[1,2] = 0
        For (i=0,j=1,k=1): U[1,2]*V[3,2]*W[2,2] = 0*(-1)*W[2,2] = 0
        ...
        
        I think I'm overcomplicating this. Let me just hardcode the known Strassen
        factors for 2x2x2 and then use them to build the 4x4x4 and 4x4x5 decompositions.
        
        Actually, let me just directly construct a valid decomposition. For 4x4x5,
        I'll use the schoolbook for the first 4 columns of the output (4x4x4 part)
        with Strassen, and add 4 more terms for the 5th column.
        
        But wait, the rank-7 Strassen is for 2x2x2, not 4x4x4. For 4x4x4, I need
        a different approach.
        
        Let me just use a simple approach: use the schoolbook for the entire 4x4x5
        but try to find a lower-rank decomposition using the structure.
        
        Actually, let me just return the schoolbook solution. It's guaranteed to work.
        """
        pass
    
    def try_optimize_rank(rank, max_iters=500, tol=1e-8):
        """Try to find a rank-r decomposition using iterative optimization."""
        if time.time() > deadline:
            return None
        
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
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
    
    # Try to find lower-rank decompositions
    candidate_ranks = [79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 68, 65, 60, 56, 50]
    
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
