import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # We decompose M_{4,4,9} into M_{4,4,4} + M_{4,4,4} + M_{4,4,1}
    # Rank 16 + 16 + 4 = 36
    
    U = []
    V = []
    W = []
    
    # Helper to add a block
    def add_block(U_block, V_block, W_block):
        for i in range(len(U_block)):
            U.append(U_block[i])
        for i in range(len(V_block)):
            V.append(V_block[i])
        for i in range(len(W_block)):
            W.append(W_block[i])
            
    # 1. M_{4,4,4} Schoolbook (Rank 16)
    # Indices: A[i,j], B[j,k], C[i,k]
    # U index: i*m+j
    # V index: j*p+k (here p=4)
    # W index: k*n+i (here n=4)
    
    U1 = [[0]*16 for _ in range(16)]
    V1 = [[0]*16 for _ in range(16)]
    W1 = [[0]*16 for _ in range(16)]
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U1[i*4+j][col] = 1
                V1[j*4+k][col] = 1
                W1[k*4+i][col] = 1
                col += 1
    add_block(U1, V1, W1)
    
    # 2. M_{4,4,4} Schoolbook (Rank 16)
    # This corresponds to the next 4 columns of B, i.e., B[:, 4:8]
    # The structure is the same, but we need to map the indices correctly.
    # In the full problem, B is 4x9.
    # The second block uses B[:, 4:8].
    # The tensor decomposition is local to the block.
    # So we just reuse the same U, V, W structure?
    # No, V indices are global.
    # In the first block, V indices were j*4+k for k in 0..3.
    # In the second block, V indices should be j*9 + (4+k).
    # So we cannot just copy V1. We need to construct V2 with shifted indices.
    
    U2 = [[0]*16 for _ in range(16)]
    V2 = [[0]*16 for _ in range(16)]
    W2 = [[0]*16 for _ in range(16)]
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U2[i*4+j][col] = 1
                # V index: j*9 + (4+k)
                V2[j*9 + (4+k)][col] = 1
                W2[k*4+i][col] = 1
                col += 1
    add_block(U2, V2, W2)
    
    # 3. M_{4,4,1} (Rank 4)
    # Uses B[:, 8].
    # C[i, 8] = sum_j A[i,j] B[j, 8]
    # We need rank 4 decomposition for this.
    # As derived, Rank(M_{4,4,1}) = 4.
    # One such decomposition:
    # We need sum_k u_k(A) v_k(B) w_k(C) = sum_{i,j} A[i,j] B[j,8] C[i,8]
    # Let's try to construct it.
    # We can use the fact that it's a linear combination.
    # Actually, a simple rank 4 decomposition for M_{n,m,1} is not trivial to write down generally without knowing the specific basis.
    # However, for n=m=4, we can use the identity:
    # sum_{i,j} A_{i,j} B_{j,1} C_{i,1}
    # This is the trace of A B C^T? No.
    # It is simply the bilinear form.
    # Let's use the standard basis vectors.
    # We can set:
    # For k=0..3:
    # u_k(A) = sum_j A_{k,j}  (Row k of A)
    # v_k(B) = B_{k,8}        (Element k of column 8 of B)
    # w_k(C) = C_{k,8}        (Element k of column 8 of C)
    # Sum_k (sum_j A_{k,j}) B_{k,8} C_{k,8}
    # This equals sum_k sum_j A_{k,j} B_{k,8} C_{k,8}.
    # This is NOT equal to sum_{i,j} A_{i,j} B_{j,8} C_{i,8}.
    # The indices don't match. We need B_{j,8} with A_{i,j}.
    
    # Correct decomposition for M_{4,4,1}:
    # T = sum_{i,j} A_{i,j} B_{j,1} C_{i,1}
    # This tensor has rank 4?
    # Let's check rank of M_{2,2,1} again.
    # T = A11 B11 C11 + A12 B21 C11 + A21 B11 C21 + A22 B21 C21
    # = B11 (A11 C11 + A21 C21) + B21 (A12 C11 + A22 C21)
    # This is 2 terms. Rank 2.
    # The terms are:
    # 1. u1(A) = A11 C11 + A21 C21 ? No, u1 must be linear in A.
    # The decomposition is:
    # Term 1: u1(A) = A11 + A21? No.
    # Let's look at the structure:
    # B11 * (A11 C11 + A21 C21) is not a product of 3 linear forms.
    # It is a product of B11 and a bilinear form in A, C.
    # So we need to decompose the bilinear form.
    
    # Actually, Rank(M_{n,m,1}) is n*m?
    # If Rank(M_{2,2,1}) = 4, then Rank(M_{4,4,1}) = 16.
    # Total rank = 16 + 16 + 16 = 48.
    # This is still better than 144.
    
    # Let's assume Rank(M_{4,4,1}) = 16 (Schoolbook).
    # It is safer to use schoolbook for the last part if we are not sure about the rank 4 decomposition.
    
    U3 = [[0]*16 for _ in range(16)]
    V3 = [[0]*16 for _ in range(16)]
    W3 = [[0]*16 for _ in range(16)]
    
    col = 0
    for i in range(4):
        for j in range(4):
            # k is fixed at 8 (index 8 in 0..8)
            k = 8
            U3[i*4+j][col] = 1
            V3[j*9 + k][col] = 1
            W3[k*4+i][col] = 1
            col += 1
    add_block(U3, V3, W3)
    
    # Total rank = 16 + 16 + 16 = 48.
    
    return {"U": U, "V": V, "W": W}
