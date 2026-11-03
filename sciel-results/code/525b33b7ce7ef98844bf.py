import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10-second safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
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
    
    # Try to construct rank-40 decomposition using rank-20 for [4,4,4]
    # The [4,4,4] rank-20 decomposition
    # U4: (16, 20), V4: (16, 20), W4: (16, 20)
    
    # Define the 20 terms for [4,4,4] MM
    # Each term: (u, v, w) as 16-element lists (4x4 matrices, row-major)
    # Using the known optimal rank-20 decomposition
    
    # I'll use the construction based on the following:
    # The [4,4,4] tensor can be decomposed using 20 rank-1 tensors.
    # Here's an explicit construction:
    
    terms_444 = []
    
    # Define helper: create a 4x4 matrix as a 16-element list
    def mat4(row0, row1, row2, row3):
        return list(row0) + list(row1) + list(row2) + list(row3)
    
    # The 20 terms for [4,4,4] MM (rank 20)
    # Based on the standard optimal decomposition
    
    # Term construction using the block structure:
    # Split 4x4 into 2x2 blocks of 2x2 matrices
    # Use the following 20 rank-1 components:
    
    # I'll use a verified construction. The 20 terms are:
    
    # Let me define them using the "subcube" approach
    # where we enumerate specific patterns
    
    # Actually, let me use a different strategy: 
    # Use the fact that [4,4,4] = [2,2,2] x [2,2,2] in tensor product sense
    # but with rank reduction via the "shared" structure
    
    # The correct rank-20 decomposition uses the following 20 terms:
    # I'll define them explicitly
    
    # Using the construction from the literature:
    # The 20 terms correspond to the following (u,v,w) triples
    
    # For clarity, I'll define each as a 16-element vector
    
    # Term 0: e00 * e00 * e00
    terms_444.append((
        mat4([1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]),
        mat4([1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]),
        mat4([1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0])
    ))
    
    # This approach of defining 20 terms manually is error-prone.
    # Let me use a different strategy: construct the rank-20 decomposition
    # using the known algorithm structure.
    
    # Alternative: Use the "Strassen-like" decomposition for [4,4,4]
    # that achieves rank 20.
    
    # The [4,4,4] MM with rank 20 can be constructed as follows:
    # We have 4x4 matrices A, B, C where C = A*B
    # Partition into 2x2 blocks of 2x2 matrices:
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    # where each block is 2x2
    
    # The 20 multiplications are:
    # M0  = (A00 + A11) * (B00 + B11)
    # M1  = (A10 + A11) * B00
    # M2  = A00 * (B01 - B11)
    # M3  = A11 * (B10 - B00)
    # M4  = (A00 + A01) * B11
    # M5  = (A10 - A00) * (B00 + B01)
    # M6  = (A01 - A11) * (B10 + B11)
    # M7  = A00 * B00
    # M8  = A01 * B01
    # M9  = A10 * B10
    # M10 = A11 * B11
    # M11 = A00 * B11
    # M12 = A11 * B00
    # M13 = A01 * B10
    # M14 = A10 * B01
    # M15 = (A00 + A01 + A10 + A11) * (B00 + B01 + B10 + B11)
    # ... this is getting complicated and I'm not sure this gives exactly 20
    
    # Let me try a completely different approach: use a rank-28 decomposition
    # (which is 7*4, using Strassen for the 2x2 blocks with schoolbook for inner)
    # or better yet, let me just try to find ANY decomposition better than 128.
    
    # Actually, the simplest improvement: use Strassen for 2x2x2 (rank 7)
    # applied to the block structure.
    
    # For [4,4,8]: view as [4,4,4] x 2 (two independent 4x4x4 problems)
    # For each [4,4,4]: use 2x2 blocks of 2x2 matrices
    # Each 2x2x2 block multiplication: rank 7 (Strassen)
    # There are 4 block positions in the output, but they share computations
    
    # Actually, for [4,4,4] with 2x2 blocks:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Each 2x2x2 multiplication has rank 7.
    # If we do each independently: 4 * 7 = 28 for [4,4,4]
    # For [4,4,8]: 8 * 7 = 56
    
    # But we can do better by sharing. The optimal is 20 for [4,4,4].
    
    # Let me just use rank 28 for [4,4,4] (4 independent Strassen computations)
    # giving rank 56 for [4,4,8]. That's better than 128.
    
    # Actually, let me use the rank-7 Strassen for 2x2x2 and apply it
    # to each of the 4 block multiplications in [4,4,4].
    
    # For [4,4,4]:
    # 4 block positions, each needs a 2x2x2 multiplication (rank 7)
    # Total: 4*7 = 28
    
    # For [4,4,8]:
    # 8 block positions (4x2 blocks of 2x2), each needs rank 7
    # Total: 8*7 = 56
    
    # Let me construct the rank-56 decomposition.
    
    # Strassen's 7 terms for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each 2x2x2 block multiplication, I need to define the U, V, W
    # contributions for the 7 Strassen terms.
    
    # Let me define the Strassen decomposition for a single 2x2x2:
    # A is 2x2: indices (i,j) with i,j in {0,1}
    # B is 2x2: indices (j,k) with j,k in {0,1}
    # C is 2x2: indices (k,i) with k,i in {0,1}
    
    # Strassen terms (u, v, w) for 2x2x2:
    def strassen_222():
        """Return 7 (u,v,w) triples for 2x2x2 MM, each is 4-element list."""
        terms = []
        # M1 = (A00+A11)(B00+B11)
        terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,-1]))  # C00+=M1, C11-=M1
        # M2 = (A10+A11)B00
        terms.append(([0,0,1,1], [1,0,0,0], [0,0,0,1]))    # C10+=M2
        # M3 = A00(B01-B11)
        terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))    # C01+=M3
        # M4 = A11(B10-B00)
        terms.append(([0,0,0,1], [0,0,1,-1], [1,0,0,0]))    # C00+=M4, C10+=M4
        # M5 = (A00+A01)B11
        terms.append(([1,1,0,0], [0,0,0,1], [0,-1,0,0]))    # C01-=M5
        # M6 = (A10-A00)(B00+B01)
        terms.append(([0,0,1,-1], [1,1,0,0], [0,0,0,1]))    # C11+=M6
        # M7 = (A01-A11)(B10+B11)
        terms.append(([0,1,0,-1], [0,0,1,1], [0,0,1,0]))    # C10+=M7, C00+=M7
        
        # Wait, I need to be more careful. Let me re-derive.
        # C00 = M1 - M5 + M4 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So for each M term, I need to specify its contribution to each C element.
        # M1: C00 += 1, C11 += 1
        # M2: C10 += 1
        # M3: C01 += 1
        # M4: C00 += 1, C10 += 1
        # M5: C01 += 1, C00 -= 1
        # M6: C11 += 1
        # M7: C00 += 1, C10 += 1
        
        # Let me redo:
        terms = []
        # M1 = (A00+A11)(B00+B11): contributes +1 to C00, +1 to C11
        terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
        # M2 = (A10+A11)B00: contributes +1 to C10
        terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
        # M3 = A00(B01-B11): contributes +1 to C01
        terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))
        # M4 = A11(B10-B00): contributes +1 to C00, +1 to C10
        terms.append(([0,0,0,1], [0,0,1,-1], [1,0,1,0]))
        # M5 = (A00+A01)B11: contributes +1 to C01, -1 to C00
        terms.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
        # M6 = (A10-A00)(B00+B01): contributes +1 to C11
        terms.append(([0,0,1,-1], [1,1,0,0], [0,0,0,1]))
        # M7 = (A01-A11)(B10+B11): contributes +1 to C00, +1 to C10
        terms.append(([0,1,0,-1], [0,0,1,1], [1,0,1,0]))
        
        return terms
    
    # Now let me verify the Strassen decomposition
    def verify_strassen():
        terms = strassen_222()
        # Build the full decomposition
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        for t, (u, v, w) in enumerate(terms):
            for idx in range(4):
                U[idx][t] = u[idx]
                V[idx][t] = v[idx]
                W[idx][t] = w[idx]
        
        # Verify: for all i,j,k: sum_t U[i,j,t] * V[j,k,t] == W[k,i,t]... 
        # Actually the convention is: C[k,i] = sum_j A[i,j] B[j,k]
        # And the decomposition is: C[k*n+i, r] pattern
        # U[i*m+j, t], V[j*p+k, t], W[k*n+i, t]
        # So: sum_t U[i*m+j, t] * V[j*p+k, t] should give the coefficient for C[k,i]
        # But W[k*n+i, t] is the output coefficient
        
        # For 2x2x2: n=m=p=2
        # U[i*2+j, t], V[j*2+k, t], W[k*2+i, t]
        # The MM tensor: T[i,j,k] = delta (i.e., C[k,i] = sum_j A[i,j]*B[j,k])
        # Decomposition: T[i,j,k] = sum_t U[i*2+j, t] * V[j*2+k, t] * W[k*2+i, t]
        # But this is a 3-way decomposition. For MM, it's:
        # C[k,i] = sum_j A[i,j] B[j,k]
        # = sum_j sum_t U[i*2+j,t] * A[i,j] * V[j*2+k,t] * B[j,k] * W[k*2+i,t]
        # This doesn't quite work as a standard tensor decomposition.
        
        # The correct interpretation: the MM tensor T has T[i,j,k] = 1 if the 
        # multiplication A[i,j]*B[j,k] contributes to C[k,i].
        # The rank-r decomposition: T[i,j,k] = sum_{t=0}^{r-1} U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        
        # For 2x2x2: T[i,j,k] = 1 for all i,j,k (since every A[i,j]*B[j,k] contributes to C[k,i])
        # Wait no. T[i,j,k] = 1 means the term A[i,j]*B[j,k] appears in C[k,i].
        # In standard MM: C[k,i] = sum_j A[i,j]*B[j,k], so T[i,j,k] = 1 for all valid i,j,k.
        
        # So we need: sum_t U[i*2+j, t] * V[j*2+k, t] * W[k*2+i, t] = 1 for all i,j,k
        
        # Let me verify:
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0
                    for t in range(7):
                        s += U[i*2+j][t] * V[j*2+k][t] * W[k*2+i][t]
                    if s != 1:
                        return False, (i,j,k,s)
        return True, None
    
    # Let me verify
    ok, info = verify_strassen()
    if not ok:
        # My Strassen decomposition is wrong, let me debug
        pass
    
    # Given the complexity and time constraints, let me use a simpler approach:
    # Use the rank-28 decomposition for [4,4,4] (4 independent 2x2x2 Strassen)
    # giving rank 56 for [4,4,8].
    
    # For [4,4,8]:
    # A is 4x4, B is 4x8, C is 8x4
    # View B as [B_left, B_right] where each is 4x4
    # C = [C_left, C_right] where each is 4x4
    # C_left = A * B_left (4x4 x 4x4 -> 4x4)
    # C_right = A * B_right (4x4 x 4x4 -> 4x4)
    
    # For each 4x4x4: use 2x2 blocks of 2x2 matrices
    # 4 block multiplications, each rank 7 (Strassen)
    # Total per 4x4x4: 28
    # Total for 4x4x8: 56
    
    # Let me construct this.
    
    # First, let me properly define the Strassen decomposition for 2x2x2.
    # I'll verify it numerically.
    
    def get_strassen_222():
        """
        Return (U, V, W) for 2x2x2 MM with rank 7.
        U: (4, 7), V: (4, 7), W: (4, 7)
        Convention: U[i*2+j, t], V[j*2+k, t], W[k*2+i, t]
        """
        # Strassen's 7 products:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)(B00)
        # M3 = (A00)(B01-B11)
        # M4 = (A11)(B10-B00)
        # M5 = (A00+A01)(B11)
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        U = [[0]*7 for _ in range(4)]  # U[i*2+j, t]
        V = [[0]*7 for _ in range(4)]  # V[j*2+k, t]
        W = [[0]*7 for _ in range(4)]  # W[k*2+i, t]
        
        # M1: A part = A00 + A11, B part = B00 + B11, C: C00 += 1, C11 += 1
        U[0][0] = 1; U[3][0] = 1  # A00, A11
        V[0][0] = 1; V[3][0] = 1  # B00, B11
        W[0][0] = 1; W[3][0] = 1  # C00 (k=0,i=0), C11 (k=1,i=1)
        
        # M2: A part = A10 + A11, B part = B00, C: C10 += 1
        U[2][1] = 1; U[3][1] = 1  # A10, A11
        V[0][1] = 1  # B00
        W[2][1] = 1  # C10 (k=1,i=0)
        
        # M3: A part = A00, B part = B01 - B11, C: C01 += 1
        U[0][2] = 1  # A00
        V[1][2] = 1; V[3][2] = -1  # B01, -B11
        W[1][2] = 1  # C01 (k=0,i=1)
        
        # M4: A part = A11, B part = B10 - B00, C: C00 += 1, C10 += 1
        U[3][3] = 1  # A11
        V[2][3] = 1; V[0][3] = -1  # B10, -B00
        W[0][3] = 1; W[2][3] = 1  # C00, C10
        
        # M5: A part = A00 + A01, B part = B11, C: C01 += 1, C00 -= 1
        U[0][4] = 1; U[1][4] = 1  # A00, A01
        V[3][4] = 1  # B11
        W[1][4] = 1; W[0][4] = -1  # C01, -C00
        
        # M6: A part = A10 - A00, B part = B00 + B01, C: C11 += 1
        U[2][5] = 1; U[0][5] = -1  # A10, -A00
        V[0][5] = 1; V[1][5] = 1  # B00, B01
        W[3][5] = 1  # C11 (k=1,i=1)
        
        # M7: A part = A01 - A11, B part = B10 + B11, C: C00 += 1, C10 += 1
        U[1][6] = 1; U[3][6] = -1  # A01, -A11
        V[2][6] = 1; V[3][6] = 1  # B10, B11
        W[0][6] = 1; W[2][6] = 1  # C00, C10
        
        return U, V, W
    
    # Verify Strassen 2x2x2
    def verify_222(U, V, W):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = sum(U[i*2+j][t] * V[j*2+k][t] * W[k*2+i][t] for t in range(7))
                    if s != 1:
                        return False
        return True
    
    U222, V222, W222 = get_strassen_222()
    if not verify_222(U222, V222, W222):
        # Fallback to schoolbook
        return schoolbook()
    
    # Now construct [4,4,4] with rank 28 (4 independent 2x2x2 Strassen)
    # A is 4x4, partitioned into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # Block (bi, bj) of A occupies rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # Similarly for B and C.
    
    # C = A * B (4x4 * 4x4 -> 4x4)
    # C[bi,bj] = sum_bk A[bi,bk] * B[bk,bj]
    
    # For each output block (bi,bj), we need sum over bk of A[bi,bk]*B[bk,bj]
    # Each A[bi,bk]*B[bk,bj] is a 2x2*2x2->2x2 multiplication, rank 7
    # There are 4 output blocks, each with 2 input block pairs -> 8 2x2x2 multiplications
    # Total rank: 8 * 7 = 56 for [4,4,4]
    
    # Wait, that's for [4,4,4]. For [4,4,8], B is 4x8 = [B0, B1] where each is 4x4
    # C is 8x4 = [C0, C1] where each is 4x4
    # C0 = A * B0, C1 = A * B1
    # Each 4x4x4 has 8 2x2x2 multiplications (rank 7 each) = rank 56
    # Total for [4,4,8]: 2 * 56 = 112
    
    # That's still better than 128! Let me construct rank 112.
    
    # Actually wait. For [4,4,4]:
    # C[bi,bj] = sum_{bk=0}^{1} A[bi,bk] * B[bk,bj]
    # For fixed (bi,bj), there are 2 terms (bk=0,1), each a 2x2x2 MM
    # So 4 output blocks * 2 = 8 2x2x2 MMs, rank 7 each = 56
    
    # For [4,4,8]:
    # B is 4x8. Partition into B_left (4x4) and B_right (4x4).
    # B_left has 2x2 blocks B[bi,bj] for bi,bj in {0,1}
    # B_right has 2x2 blocks B[bi, 2+bj] for bi in {0,1}, bj in {0,1}
    
    # C is 8x4. C_top (4x4) = A * B_left, C_bottom (4x4) = A * B_right
    # C_top[bi,bj] = sum_{bk} A[bi,bk] * B_left[bk,bj]
    # C_bottom[bi,bj] = sum_{bk} A[bi,bk] * B_right[bk,bj]
    
    # Total 2x2x2 MMs: 2 (left/right) * 4 (output blocks) * 2 (sum over bk) = 16
    # Rank: 16 * 7 = 112
    
    # Let me construct this.
    
    # For [4,4,8]:
    # A: 4x4, index (i,j) with i,j in 0..3
    # B: 4x8, index (j,k) with j in 0..3, k in 0..7
    # C: 8x4, index (k,i) with k in 0..7, i in 0..3
    # Convention: W[k*n+i, t] = W[k*4+i, t]
    
    # Partition:
    # A: blocks A[bi,bj] where bi = i//2, bj = j//2, local (i%2, j%2)
    # B: blocks B[bi,bk] where bi = j//2, bk = k//2, local (j%2, k%2)
    #   For B: j in 0..3 (row), k in 0..7 (col)
    #   Block row: j//2 (0 or 1), block col: k//2 (0,1,2,3)
    # C: blocks C[bk,bi] where bk = k//2, bi = i//2, local (k%2, i%2)
    #   For C: k in 0..7 (row), i in 0..3 (col)
    #   Block row: k//2 (0,1,2,3), block col: i//2 (0 or 1)
    
    # C[bk,bi] = sum_{bj} A[bi,bj] * B[bj,bk]
    # where A[bi,bj] is 2x2, B[bj,bk] is 2x2, C[bk,bi] is 2x2
    
    # For each (bk, bi, bj): one 2x2x2 MM
    # bk in 0..3, bi in 0..1, bj in 0..1: 4*2*2 = 16 terms
    # Each rank 7: total rank 112
    
    # Let me construct the U, V, W matrices.
    
    rank = 112
    U = [[0]*rank for _ in range(16)]  # n*m = 4*4 = 16
    V = [[0]*rank for _ in range(32)]  # m*p = 4*8 = 32
    W = [[0]*rank for _ in range(32)]  # p*n = 8*4 = 32
    
    # For each (bk, bi, bj) and each Strassen term t (0..6):
    # The 2x2x2 MM: A[bi,bj] * B[bj,bk] -> C[bk,bi]
    # A[bi,bj]: rows 2*bi..2*bi+1, cols 2*bj..2*bj+1 in A
    #   Global A index: i = 2*bi + li, j = 2*bj + lj -> A index = i*4+j = (2*bi+li)*4 + (2*bj+lj)
    # B[bj,bk]: rows 2*bj..2*bj+1, cols 2*bk..2*bk+1 in B
    #   Global B index: j = 2*bj + lj, k = 2*bk + lk -> B index = j*8+k = (2*bj+lj)*8 + (2*bk+lk)
    # C[bk,bi]: rows 2*bk..2*bk+1, cols 2*bi..2*bi+1 in C
    #   Global C index: k = 2*bk + lk, i = 2*bi + li -> C index = k*4+i = (2*bk+lk)*4 + (2*bi+li)
    
    col = 0
    for bk in range(4):
        for bi in range(2):
            for bj in range(2):
                for t in range(7):
                    # U contribution: A[bi,bj] local (li, lj) -> U222[li*2+lj, t]
                    for li in range(2):
                        for lj in range(2):
                            global_i = 2*bi + li
                            global_j = 2*bj + lj
                            U[global_i*4 + global_j][col] = U222[li*2+lj][t]
                    
                    # V contribution: B[bj,bk] local (lj, lk) -> V222[lj*2+lk, t]
                    for lj in range(2):
                        for lk in range(2):
                            global_j = 2*bj + lj
                            global_k = 2*bk + lk
                            V[global_j*8 + global_k][col] = V222[lj*2+lk][t]
                    
                    # W contribution: C[bk,bi] local (lk, li) -> W222[lk*2+li, t]
                    for lk in range(2):
                        for li in range(2):
                            global_k = 2*bk + lk
                            global_i = 2*bi + li
                            W[global_k*4 + global_i][col] = W222[lk*2+li][t]
                    
                    col += 1
    
    # Verify the decomposition
    def verify_decomposition(U, V, W, n, m, p):
        # Check: for all i,j,k: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for t in range(len(U[0])):
                        s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                    if s != 1:
                        return False
        return True
    
    if verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        return schoolbook()
