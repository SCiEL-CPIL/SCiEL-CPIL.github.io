import numpy as np
from fractions import Fraction
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = 2, 3, 10
    
    # Try to find a low-rank decomposition
    # First, let's try a block-based approach
    # Group columns of B into pairs: 10 = 5 * 2
    # For each 2x3x2 block, try to find a low-rank decomposition
    
    # The 2x3x2 tensor T has T[i,j,k] = 1 if (i,j,k) corresponds to A[i,j]*B[j,k]
    # Let's build the 2x3x2 tensor and try to find its rank
    
    # Actually, let me try a different approach: use known decompositions
    # For 2x2x2, Strassen gives rank 7
    # Can I adapt this to 2x3x2?
    
    # Let me try to construct a decomposition directly
    # For the full 2x3x10 problem, I'll try various rank values
    
    # First, let's try to see if we can use a Kronecker-like structure
    # 2x3x10 = 2x3x(2*5), so if I have a 2x3x2 decomposition of rank r,
    # I can get 2x3x10 of rank 5*r
    
    # Let me try to find a 2x3x2 decomposition
    # Build the tensor
    # T[i, j, k] for i in {0,1}, j in {0,1,2}, k in {0,1}
    # The multiplication tensor has T[i,j,k] = 1 for the (i,j,k) triple
    # representing A[i,j] * B[j,k]
    
    # In the U,V,W format for 2x3x2:
    # U has shape (n*m, rank) = (6, rank)
    # V has shape (m*p, rank) = (6, rank)  
    # W has shape (p*n, rank) = (4, rank)
    # W[k*n+i] = W[2k+i]
    
    # Let me try to use a rank-11 decomposition for 2x3x2
    # This would give rank 55 for 2x3x10
    
    # Actually, let me try a more aggressive approach
    # Use numerical optimization to find low-rank decompositions
    
    def verify_decomposition(U, V, W, n, m, p, tol=1e-10):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor"""
        rank = len(U[0]) if U else 0
        for c in range(rank):
            uc = [U[i*m+j][c] for i in range(n) for j in range(m)]
            vc = [V[j*p+k][c] for j in range(m) for k in range(p)]
            wc = [W[k*n+i][c] for k in range(p) for i in range(n)]
            # This rank-1 term contributes uc[i*m+j] * vc[j*p+k] to W[k*n+i]
            # For all i, j, k: sum_c U[i*m+j,c]*V[j*p+k,c] should equal 1 if it's the multiplication tensor
        # Better: check the full product
        U_mat = np.array(U)
        V_mat = np.array(V)
        W_mat = np.array(W)
        # The tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c]
        # Should equal 1 for valid (i,j,k) and 0 otherwise... 
        # Actually no. The multiplication tensor has T[i,j,k] = delta (it's 1 for each (i,j,k) triple)
        # The decomposition is: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * ... 
        # Wait, I need to be more careful.
        
        # The matrix multiplication tensor is:
        # T[i, j, k] = 1 for all i in {0..n-1}, j in {0..m-1}, k in {0..p-1}
        # No wait, that's not right either.
        
        # The tensor represents: C[i,k] = sum_j A[i,j] * B[j,k]
        # As a trilinear form: <C^T, A^T, B^T> = sum_{i,j,k} C[i,k] * A[i,j] * B[j,k]
        # = sum_{i,j,k} (sum_j' A[i,j'] * B[j',k]) * A[i,j] * B[j,k]
        
        # The rank-1 decomposition is:
        # T = sum_c U_c ⊗ V_c ⊗ W_c
        # where U_c is n*m, V_c is m*p, W_c is p*n
        # T[i,j,k] = sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i]
        
        # For matrix multiplication: T[i,j,k] = 1 for all i,j,k (since the coefficient of A[i,j]*B[j,k] in C[i,k] is 1)
        # Wait no, T[i,j,k] should be 1 if j is the "middle" index that connects i and k.
        # Actually, the standard convention: the tensor T has T[i,j,k] = 1 for all valid (i,j,k),
        # and the decomposition gives C[i,k] = sum_j A[i,j] B[j,k] = sum_{i,j,k} A[i,j] B[j,k] T[i,j,k]
        # where T[i,j,k] = delta (it's 1 for each triple, and the sum over j gives the product)
        
        # So T[i,j,k] = 1 for all i in {0..n-1}, j in {0..m-1}, k in {0..p-1}
        # And we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
        
        # Let me verify this
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Let me try to find a decomposition by starting from schoolbook and trying to reduce
    # Use a greedy/local search approach
    
    # First, generate the schoolbook decomposition
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W
    
    # Try to find lower rank by using random initialization and optimization
    # This is complex, so let me try a simpler approach first
    
    # Known: for 2x2x2, Strassen gives rank 7
    # For 2x3x2, let me try to find a good decomposition
    
    # Let me try rank 11 for 2x3x2, which would give 55 for 2x3x10
    # Or even better, try to find the actual minimum
    
    # Alternative approach: use the fact that we can combine Strassen-like tricks
    # Split A into [A1 | A2] where A1 is 2x2 and A2 is 2x1
    # Split B into [B1; B2] where B1 is 2x10 and B2 is 1x10
    # C = A1*B1 + A2*B2
    
    # A1*B1 is 2x2x10, A2*B2 is 2x1x10
    # R(2,2,10) and R(2,1,10) = 2*1*10 = 20 (since 1x10 is just a vector)
    # R(2,1,10) = 20 (schoolbook, likely optimal)
    # R(2,2,10): can use Strassen-like. 2x2x2 is rank 7, so 2x2x10 might be 7*5=35? 
    # But that's not quite right because Strassen works on 2x2x2 blocks.
    
    # Actually, for 2x2xp, I can use:
    # C = A*B where A is 2x2, B is 2xp
    # Split B into pairs of columns: B = [B1 B2 ... B5] where each Bj is 2x2
    # Then C = [A*B1 A*B2 ... A*B5]
    # Each A*Bj can be done with rank 7 (Strassen)
    # Total rank: 5*7 = 35 for 2x2x10
    
    # So 2x3x10: R <= R(2,2,10) + R(2,1,10) <= 35 + 20 = 55
    
    # Let me construct this!
    # A = [A_left | A_right] where A_left is 2x2, A_right is 2x1
    # B = [B_top; B_bottom] where B_top is 2x10, B_bottom is 1x10
    # C = A_left * B_top + A_right * B_bottom
    
    # For A_left * B_top (2x2x10):
    # Split B_top into 5 blocks of 2x2: B_top = [B0 B1 B2 B3 B4]
    # For each block Bj, use Strassen
    # Strassen for 2x2x2:
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # M3 = A[0,0]*(B[0,1]-B[1,1])
    # M4 = A[1,1]*(B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # Each M involves products of (linear combinations of A entries) * (linear combinations of B entries)
    # So each M is a rank-1 term in the tensor decomposition
    
    # For 2x2x10 with 5 blocks:
    # Each block uses 7 rank-1 terms
    # Total: 35 terms
    
    # For A_right * B_bottom (2x1x10):
    # This is just: C[i,k] += A[1,k_offset] * B[2,k] (since A_right is column 2, B_bottom is row 2)
    # Wait, A is 2x3, so A_right is the third column (index 2), shape 2x1
    # B is 3x10, so B_bottom is the third row (index 2), shape 1x10
    # A_right * B_bottom is 2x10
    # C[i,k] += A[i,2] * B[2,k]
    # This is rank 2*10 = 20 in schoolbook
    
    # Total: 35 + 20 = 55
    
    # Let me construct this explicitly
    
    # First, let me build the 2x2x10 Strassen-based decomposition
    # Then add the 2x1x10 schoolbook part
    
    # Let me define the Strassen decomposition for a single 2x2x2 block
    # Input: A is 2x2 (indices a00, a01, a10, a11), B is 2x2 (b00, b01, b10, b11)
    # Output: C is 2x2 (c00, c01, c10, c11)
    
    # In the tensor decomposition framework:
    # U entries correspond to A coefficients, V entries to B coefficients, W entries to C coefficients
    # For a rank-1 term: U * V * W where U is a vector over A indices, V over B indices, W over C indices
    
    # Strassen terms for 2x2x2:
    # Term 1: M1 = (a00+a11)*(b00+b11)
    #   U: a00=1, a11=1; V: b00=1, b11=1; W: contributes to c00(+1), c11(-1) wait let me recheck
    
    # Standard Strassen:
    # p1 = (a00+a11)(b00+b11)
    # p2 = (a10+a11)b00
    # p3 = a00(b01-b11)
    # p4 = a11(b10-b00)
    # p5 = (a00+a01)b11
    # p6 = (a10-a00)(b00+b01)
    # p7 = (a01-a11)(b10+b11)
    
    # c00 = p1 + p4 - p5 + p7
    # c01 = p3 + p5
    # c10 = p2 + p4
    # c11 = p1 - p2 + p3 + p6
    
    # So for term 1 (p1):
    # U: [1, 0, 0, 1] for [a00, a01, a10, a11]
    # V: [1, 0, 0, 1] for [b00, b01, b10, b11]
    # W: [1, 0, 0, 1] for [c00, c01, c10, c11] (c00 gets +1, c11 gets +1... wait)
    # c00 = ... + p1 ... so W[0] = 1
    # c11 = p1 - ... so W[3] = 1
    # Actually: c00 += p1, c11 += p1, so W = [1, 0, 0, 1]
    
    # Let me code this up properly
    
    def build_2x2x2_strassen():
        """Return U, V, W for 2x2x2 Strassen decomposition, rank 7"""
        # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # W index: k*n+i where n=2, so (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        
        U = []
        V = []
        W = []
        
        # p1 = (a00+a11)(b00+b11)
        U.append([1, 0, 0, 1])
        V.append([1, 0, 0, 1])
        W.append([1, 0, 0, 1])  # c00 += p1, c11 += p1
        
        # p2 = (a10+a11)b00
        U.append([0, 0, 1, 1])
        V.append([1, 0, 0, 0])
        W.append([0, 0, 1, -1])  # c10 += p2, c11 -= p2
        
        # p3 = a00(b01-b11)
        U.append([1, 0, 0, 0])
        V.append([0, 1, 0, -1])
        W.append([0, 1, 0, 1])  # c01 += p3, c11 += p3
        
        # p4 = a11(b10-b00)
        U.append([0, 0, 0, 1])
        V.append([0, 0, 1, -1])
        W.append([1, 0, 1, 0])  # c00 += p4, c10 += p4
        
        # p5 = (a00+a01)b11
        U.append([1, 1, 0, 0])
        V.append([0, 0, 0, 1])
        W.append([-1, 1, 0, 0])  # c00 -= p5, c01 += p5
        
        # p6 = (a10-a00)(b00+b01)
        U.append([-1, 0, 1, 0])
        V.append([1, 1, 0, 0])
        W.append([0, 0, 0, 1])  # c11 += p6
        
        # p7 = (a01-a11)(b10+b11)
        U.append([0, 1, 0, -1])
        V.append([0, 0, 1, 1])
        W.append([1, 0, 0, 1])  # c00 += p7, c11 += p7
        
        # Convert to the required format
        # U should be (n*m, rank) = (4, 7)
        # V should be (m*p, rank) = (4, 7)
        # W should be (p*n, rank) = (4, 7)
        
        U_mat = [[U[c][i] for c in range(7)] for i in range(4)]
        V_mat = [[V[c][i] for c in range(7)] for i in range(4)]
        W_mat = [[W[c][i] for c in range(7)] for i in range(4)]
        
        return U_mat, V_mat, W_mat
    
    # Verify the Strassen decomposition
    U_s, V_s, W_s = build_2x2x2_strassen()
    
    # Verify: sum_c U[i,c]*V[j,c]*W[k,c] should equal 1 for all i,j,k in 0..3
    # where i indexes A, j indexes B, k indexes C (with W[k*n+i] convention, but for 2x2x2, n=2)
    # Wait, the W index is k*n+i where n=2, so W[0]=W[0*2+0], W[1]=W[0*2+1], W[2]=W[1*2+0], W[3]=W[1*2+1]
    # In my Strassen, W[0]=c00, W[1]=c01, W[2]=c10, W[3]=c11, which matches k*n+i with (k=0,i=0), (k=0,i=1), (k=1,i=0), (k=1,i=1)
    
    # But in the tensor decomposition, the index structure is:
    # U[i*m+j] for A[i,j], V[j*p+k] for B[j,k], W[k*n+i] for C[i,k]
    # For 2x2x2: U[i*2+j], V[j*2+k], W[k*2+i]
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C indices via W[k*2+i]: (k=0,i=0)=0, (k=0,i=1)=1, (k=1,i=0)=2, (k=1,i=1)=3
    # This matches c00, c01, c10, c11
    
    # Verify
    n2, m2, p2 = 2, 2, 2
    valid = True
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                s = 0
                for c in range(7):
                    s += U_s[i*m2+j][c] * V_s[j*p2+k][c] * W_s[k*n2+i][c]
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Strassen verification failed, fall back to schoolbook
        U, V, W = schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    # Now build the 2x3x10 decomposition
    # A is 2x3: columns 0,1 form A_left (2x2), column 2 is A_right (2x1)
    # B is 3x10: rows 0,1 form B_top (2x10), row 2 is B_bottom (1x10)
    # C = A_left * B_top + A_right * B_bottom
    
    # Part 1: A_left * B_top (2x2x10)
    # Split B_top into 5 blocks of 2 columns each: B_top[:, 2*b:2*b+2] for b=0..4
    # Each block uses Strassen (rank 7)
    
    # Part 2: A_right * B_bottom (2x1x10)
    # C[i,k] += A[i,2] * B[2,k] for i=0,1; k=0..9
    # This is 20 rank-1 terms (schoolbook)
    
    total_rank = 5 * 7 + 20  # 35 + 20 = 55
    
    U = [[0]*total_rank for _ in range(n*m)]  # (6, 55)
    V = [[0]*total_rank for _ in range(m*p)]  # (30, 55)
    W = [[0]*total_rank for _ in range(p*n)]  # (20, 55)
    
    # Build Part 1: 5 Strassen blocks for A_left * B_top
    col_offset = 0
    for b in range(5):
        # Block b covers B columns 2*b and 2*b+1
        # A_left uses A columns 0 and 1
        # For this block, the local 2x2x2 decomposition:
        # A_local[i,j] = A[i,j] for i in {0,1}, j in {0,1}
        # B_local[j,k] = B[j, 2*b+k] for j in {0,1}, k in {0,1}
        # C_local[i,k] = C[i, 2*b+k]
        
        # U indices for A_left: A[i, j] -> U[i*3 + j] (since m=3)
        #   (0,0)=0, (0,1)=1, (1,0)=3, (1,1)=4
        # V indices for B_top block: B[j, 2*b+k] -> V[j*10 + (2*b+k)]
        #   (0,2b)=2b, (0,2b+1)=2b+1, (1,2b)=10+2b, (1,2b+1)=10+2b+1
        # W indices for C: C[i, 2*b+k] -> W[(2*b+k)*2 + i]
        #   (i=0,k=2b)=2b*2+0=4b, (i=1,k=2b)=4b+1, (i=0,k=2b+1)=4b+2, (i=1,k=2b+1)=4b+3
        
        for c_local in range(7):
            c = col_offset + c_local
            
            # U contributions: A[i, j] for i in {0,1}, j in {0,1}
            # U_s[i*2+j][c_local] is the coefficient
            for i in range(2):
                for j in range(2):
                    val = U_s[i*2+j][c_local]
                    if val != 0:
                        U[i*m+j][c] += val  # A[i,j] index is i*3+j
            
            # V contributions: B[j, 2*b+k] for j in {0,1}, k in {0,1}
            # V_s[j*2+k][c_local] is the coefficient
            for j in range(2):
                for k in range(2):
                    val = V_s[j*2+k][c_local]
                    if val != 0:
                        V[j*p + (2*b+k)][c] += val  # B[j, 2b+k] index is j*10 + (2b+k)
            
            # W contributions: C[i, 2*b+k] for i in {0,1}, k in {0,1}
            # W_s[k*2+i][c_local] is the coefficient
            for k in range(2):
                for i in range(2):
                    val = W_s[k*2+i][c_local]
                    if val != 0:
                        W[(2*b+k)*n + i][c] += val  # C[i, 2b+k] index is (2b+k)*2+i
    
    col_offset += 35
    
    # Build Part 2: A_right * B_bottom (2x1x10)
    # C[i,k] += A[i,2] * B[2,k]
    # For each i in {0,1}, k in {0..9}:
    #   U[i*3+2][c] = 1 (A[i,2])
    #   V[2*10+k][c] = 1 (B[2,k])
    #   W[k*2+i][c] = 1 (C[i,k])
    
    for i in range(2):
        for k in range(10):
            c = col_offset + i*10 + k
            U[i*m+2][c] = 1  # A[i, 2]
            V[2*p+k][c] = 1  # B[2, k]
            W[k*n+i][c] = 1  # C[i, k]
    
    # Verify the full decomposition
    if time.time() < deadline - 10:
        # Verify
        U_mat = np.array(U)
        V_mat = np.array(V)
        W_mat = np.array(W)
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(total_rank):
                        s += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
                    if abs(s - 1.0) > 1e-10:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback to schoolbook
            U, V, W = schoolbook(n, m, p)
            total_rank = n*m*p
    
    return {"U": U, "V": V, "W": W}
