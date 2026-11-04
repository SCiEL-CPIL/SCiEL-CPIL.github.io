import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    # A is n x m = 4x4, B is m x p = 4x5, C is n x p = 4x5
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # U has shape (n*m, rank) = (16, rank)
    # V has shape (m*p, rank) = (20, rank)  
    # W has shape (p*n, rank) = (20, rank)
    # W index: k*n + i
    
    # First, define Strassen 2x2x2 rank-7 decomposition
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B
    # A index: i*2+j (row-major): a=0, b=1, c=2, d=3
    # B index: j*2+k (row-major): e=0, f=1, g=2, h=3
    # C index: k*2+i: c00=0, c10=1, c01=2, c11=3
    
    # Strassen multiplications:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(g-h)
    # m4 = d(f-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    
    # C00 = m1 + m4 - m5 - m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m7
    
    # Note: m6 does NOT appear in the output formulas. This is correct -
    # the 7 terms form a valid rank-7 decomposition even though m6 has zero
    # coefficient in all C entries. Wait, that can't be right either.
    
    # Let me re-derive. Actually, I think the issue is that m6 IS used but I
    # have the wrong C formulas. Let me just verify numerically.
    
    strassen_2x2 = [
        # (u, v, w) where u is 4-vec for A, v is 4-vec for B, w is 4-vec for C
        # m1 = (a+d)(e+h)
        (np.array([1,0,0,1]), np.array([1,0,0,1]), np.array([1,0,0,1])),
        # m2 = (c+d)e
        (np.array([0,0,1,1]), np.array([1,0,0,0]), np.array([0,1,0,-1])),
        # m3 = a(g-h)
        (np.array([1,0,0,0]), np.array([0,0,1,-1]), np.array([0,0,1,1])),
        # m4 = d(f-e)
        (np.array([0,0,0,1]), np.array([0,1,-1,0]), np.array([1,1,0,0])),
        # m5 = (a+b)h
        (np.array([1,1,0,0]), np.array([0,0,0,1]), np.array([-1,0,1,0])),
        # m6 = (c-a)(e+f)
        (np.array([0,0,1,-1]), np.array([1,1,0,0]), np.array([0,1,0,-1])),
        # m7 = (b-d)(g+h)
        (np.array([0,1,0,-1]), np.array([0,0,1,1]), np.array([-1,0,-1,0])),
    ]
    
    # Verify Strassen numerically
    def verify_strassen():
        rng = np.random.RandomState(42)
        for _ in range(10):
            A_t = rng.randint(-3, 4, (2,2)).astype(float)
            B_t = rng.randint(-3, 4, (2,2)).astype(float)
            C_expected = A_t @ B_t
            # C in our order: k*2+i -> [c00, c10, c01, c11]
            C_exp = np.array([C_expected[0,0], C_expected[1,0], C_expected[0,1], C_expected[1,1]])
            
            C_comp = np.zeros(4)
            for u, v, w in strassen_2x2:
                A_vec = A_t.flatten()  # row-major: [a, b, c, d]
                B_vec = B_t.flatten()  # row-major: [e, f, g, h]
                prod = np.dot(A_vec, u) * np.dot(B_vec, v)
                C_comp += prod * w
            
            if not np.allclose(C_comp, C_exp, atol=1e-10):
                return False
        return True
    
    strassen_ok = verify_strassen()
    
    if not strassen_ok:
        # Fall back: use schoolbook for everything, rank = 4*4*5 = 80
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
    
    # Build 4x4x4 decomposition using one level of Strassen (7 blocks of 2x2x2)
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # Each block is 2x2.
    # 
    # Block Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # For the 4x4x4 part, each M_i is a 2x2x2 multiplication, so we use
    # the rank-7 Strassen decomposition for each.
    #
    # Total terms: 7 (block level) * 7 (scalar level) = 49
    
    # Let me build this carefully.
    # A is 4x4, indexed as A[i,j] with i,j in 0..3
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    # Similarly for B (4x4 part) and C (4x4 part)
    
    # For the block-level Strassen, I need to express each M_i as a product
    # of linear combinations of 2x2 blocks, then expand each 2x2x2 product
    # using the scalar Strassen decomposition.
    
    # This is complex. Let me instead directly enumerate all 49 terms.
    
    # Actually, let me think about this differently. The 4x4x4 tensor can be
    # decomposed as follows:
    # 
    # The 49 terms come from: for each of the 7 block-level Strassen terms,
    # we have a 2x2x2 sub-problem. Each 2x2x2 sub-problem has 7 scalar terms.
    #
    # For block term M_i, the U vector is a 16-dim vector (for the 4x4 A),
    # the V vector is a 16-dim vector (for the 4x4 B), and the W vector is
    # a 16-dim vector (for the 4x4 C).
    #
    # Let me define the block-level structure:
    # 
    # M1 = (A00+A11)(B00+B11)
    #   U_block: A00 gets +1, A11 gets +1 -> in 4x4 A: positions (0,0),(0,1),(1,0),(1,1) get +1,
    #            positions (2,2),(2,3),(3,2),(3,3) get +1
    #   V_block: B00 gets +1, B11 gets +1 -> in 4x4 B: positions (0,0),(0,1),(1,0),(1,1) get +1,
    #            positions (2,2),(2,3),(3,2),(3,3) get +1
    #   W_block: C00 gets +1, C11 gets +1 -> in 4x4 C: positions (0,0),(0,1),(1,0),(1,1) get +1,
    #            positions (2,2),(2,3),(3,2),(3,3) get +1
    #
    # But wait, M1 is a 2x2x2 multiplication, not a scalar. So I need to expand it
    # using the scalar Strassen decomposition.
    
    # This is getting very complex to do by hand. Let me use a different approach:
    # directly construct the 49 terms programmatically.
    
    # For each block-level Strassen term i (0..6), and each scalar Strassen term j (0..6),
    # I can compute the U, V, W vectors.
    
    # Block-level Strassen definitions (which blocks get which coefficients):
    # M1: A: {00:+1, 11:+1}, B: {00:+1, 11:+1}, C: {00:+1, 11:+1}
    # M2: A: {10:+1, 11:+1}, B: {00:+1}, C: {10:+1, 11:-1}
    # M3: A: {00:+1}, B: {10:+1, 11:-1}, C: {01:+1, 11:+1}
    # M4: A: {11:+1}, B: {01:+1, 00:-1}, C: {00:+1, 10:+1}
    # M5: A: {00:+1, 01:+1}, B: {11:+1}, C: {00:-1, 01:+1}
    # M6: A: {10:+1, 00:-1}, B: {00:+1, 01:+1}, C: {10:+1, 11:-1}
    # M7: A: {01:+1, 11:-1}, B: {10:+1, 11:+1}, C: {00:-1, 01:-1}
    
    # For each block term, the U vector for the 4x4 A is determined by which
    # 2x2 blocks get which coefficients. Similarly for V and W.
    
    # But the scalar Strassen decomposition within each 2x2x2 sub-problem means
    # that the actual U, V, W vectors are more complex.
    
    # Let me think about this more carefully. When we write:
    # M1 = (A00+A11)(B00+B11)
    # This means: for each element (i,j) of A00+A11 and each element (j',k) of B00+B11,
    # we compute the product and add it to the appropriate C entry.
    #
    # (A00+A11) is a 2x2 matrix where element (a,b) = A00[a,b] + A11[a,b]
    # (B00+B11) is a 2x2 matrix where element (c,d) = B00[c,d] + B11[c,d]
    # The product (A00+A11)(B00+B11) is a 2x2 matrix.
    #
    # To decompose this 2x2x2 product using Strassen, I apply the scalar Strassen
    # to the 2x2 matrices (A00+A11) and (B00+B11).
    
    # So for M1, the effective 2x2 A matrix is A00+A11, and the effective 2x2 B matrix
    # is B00+B11. The scalar Strassen terms for this 2x2x2 problem give me 7 scalar
    # multiplications.
    
    # For scalar Strassen term j with vectors (u_j, v_j, w_j):
    # The scalar product is: (sum over (a,b) of (A00+A11)[a,b] * u_j[a*2+b]) *
    #                         (sum over (c,d) of (B00+B11)[c,d] * v_j[c*2+d])
    # = (sum over (a,b) of (A[0+a,0+b] + A[2+a,2+b]) * u_j[a*2+b]) *
    #   (sum over (c,d) of (B[0+c,0+d] + B[2+c,2+d]) * v_j[c*2+d])
    
    # So the U vector for the 4x4 A has:
    #   U[a*4+b] += u_j[a*2+b] for a,b in 0..1 (from A00)
    #   U[(2+a)*4+(2+b)] += u_j[a*2+b] for a,b in 0..1 (from A11)
    
    # Similarly for V and W.
    
    # Let me code this up.
    
    # Block-level Strassen: for each term i, define the block coefficients
    # block_A[i]: dict mapping (block_row, block_col) -> coefficient
    # block_B[i]: dict mapping (block_row, block_col) -> coefficient  
    # block_C[i]: dict mapping (block_row, block_col) -> coefficient
    
    block_strassen = [
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = A00(B10-B11) -> C01 += M3, C11 += M3
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = A11(B01-B00) -> C00 += M4, C10 += M4
        ({(1,1): 1}, {(0,1): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6 = (A10-A00)(B00+B01) -> C10 += M6, C11 -= M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,0): 1, (1,1): -1}),
        # M7 = (A01-A11)(B10+B11) -> C00 -= M7, C01 -= M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): -1, (0,1): -1}),
    ]
    
    # Now build the 49 terms for 4x4x4
    U_4x4x4 = []
    V_4x4x4 = []
    W_4x4x4 = []
    
    for i in range(7):
        block_A, block_B, block_C = block_strassen[i]
        for j in range(7):
            u_2x2, v_2x2, w_2x2 = strassen_2x2[j]
            
            # Build 16-dim U vector for 4x4 A
            U_vec = np.zeros(16)
            for (br, bc), coeff in block_A.items():
                for a in range(2):
                    for b in range(2):
                        idx_A = (br*2 + a) * 4 + (bc*2 + b)  # row = br*2+a, col = bc*2+b
                        idx_2x2 = a * 2 + b
                        U_vec[idx_A] += coeff * u_2x2[idx_2x2]
            
            # Build 16-dim V vector for 4x4 B
            V_vec = np.zeros(16)
            for (br, bc), coeff in block_B.items():
                for c in range(2):
                    for d in range(2):
                        idx_B = (br*2 + c) * 4 + (bc*2 + d)  # row = br*2+c, col = bc*2+d
                        idx_2x2 = c * 2 + d
                        V_vec[idx_B] += coeff * v_2x2[idx_2x2]
            
            # Build 16-dim W vector for 4x4 C
            W_vec = np.zeros(16)
            for (br, bc), coeff in block_C.items():
                for a in range(2):
                    for b in range(2):
                        # C index in our convention: k*4+i where k=col, i=row
                        # Block (br, bc) means C rows br*2..br*2+1, cols bc*2..bc*2+1
                        # For element (a,b) in the block: row = br*2+a, col = bc*2+b
                        # W index = col * n + row = (bc*2+b) * 4 + (br*2+a)
                        idx_W = (bc*2 + b) * 4 + (br*2 + a)
                        idx_2x2 = b * 2 + a  # In 2x2 C, index = k*2+i = b*2+a
                        W_vec[idx_W] += coeff * w_2x2[idx_2x2]
            
            U_4x4x4.append(U_vec)
            V_4x4x4.append(V_vec)
            W_4x4x4.append(W_vec)
    
    # Verify the 4x4x4 decomposition
    def verify_4x4x4():
        rng = np.random.RandomState(123)
        for _ in range(5):
            A_t = rng.randint(-3, 4, (4,4)).astype(float)
            B_t = rng.randint(-3, 4, (4,4)).astype(float)
            C_expected = A_t @ B_t
            
            C_comp = np.zeros((4,4))
            for idx in range(49):
                U_vec = U_4x4x4[idx]
                V_vec = V_4x4x4[idx]
                W_vec = W_4x4x4[idx]
                
                A_flat = A_t.flatten()  # row-major: A[i,j] at index i*4+j
                B_flat = B_t.flatten()  # row-major: B[j,k] at index j*4+k
                prod = np.dot(A_flat, U_vec) * np.dot(B_flat, V_vec)
                
                # W_vec is in order k*4+i
                for k in range(4):
                    for i in range(4):
                        C_comp[i, k] += prod * W_vec[k * 4 + i]
            
            if not np.allclose(C_comp, C_expected, atol=1e-10):
                return False
        return True
    
    four_4_4_ok = verify_4x4x4()
    
    if not four_4_4_ok:
        # Fall back to schoolbook
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
    
    # Now add the 5th column of B (column index 4)
    # C[i, 4] = sum_j A[i,j] * B[j, 4]
    # This is a 4x4x1 problem, which has rank 4 (schoolbook is optimal for this)
    
    # Total rank = 49 + 4 = 53
    total_rank = 53
    
    U_full = []
    V_full = []
    W_full = []
    
    # First 49 terms: 4x4x4 part
    # U is (16, 53), V is (20, 53), W is (20, 53)
    for idx in range(49):
        U_vec = U_4x4x4[idx]
        V_vec = V_4x4x4[idx]
        W_vec = W_4x4x4[idx]
        
        # U: 16-dim, extend to 16 (already 16)
        U_row = U_vec.tolist()
        # V: 16-dim for 4x4 B, extend to 20-dim for 4x5 B (last 4 entries are 0)
        V_row = list(V_vec) + [0.0, 0.0, 0.0, 0.0]
        # W: 16-dim for 4x4 C (k*4+i for k=0..3), extend to 20-dim for 4x5 C
        # W index k*n+i: for k=0..3, i=0..3 -> indices 0..15
        # For k=4, i=0..3 -> indices 16..19
        W_row = list(W_vec) + [0.0, 0.0, 0.0, 0.0]
        
        U_full.append(U_row)
        V_full.append(V_row)
        W_full.append(W_row)
    
    # Last 4 terms: 4x4x1 part (5th column)
    for i in range(4):
        U_row = [0.0] * 16
        U_row[i * 4 + j] = 0.0  # Will be set per term
        # Actually, for each term r = 0..3, we handle one row i of A
        # Term for row i: U has 1s at positions i*4+0, i*4+1, i*4+2, i*4+3
        # Wait, no. For the 4x4x1 schoolbook:
        # C[i,4] = A[i,0]*B[0,4] + A[i,1]*B[1,4] + A[i,2]*B[2,4] + A[i,3]*B[3,4]
        # This can be decomposed into 4 rank-1 terms:
        # Term j: U has 1 at position i*4+j, V has 1 at position j*5+4, W has 1 at position 4*4+i
        # But that's 16 terms total (4 i's * 4 j's).
        #
        # Actually, the rank of the 4x4x1 tensor is 4, not 16.
        # The 4x4x1 tensor is: T[i,j,k] = delta_{k,4} * (A[i,j] * B[j,4])
        # Wait, the tensor for matrix multiplication 4x4x5 has entries:
        # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
        # So T[i,j,k] = 1 for all i,j,k (it's the identity tensor).
        #
        # For the 5th column only: we want terms that contribute to C[i,4] for all i.
        # C[i,4] = sum_j A[i,j] * B[j,4]
        #
        # A rank-1 term is: U[i*4+j] * V[j*5+4] * W[4*4+i]
        # For a single rank-1 term to contribute to C[i,4] for a specific (i,j) pair:
        # U has 1 at (i,j), V has 1 at (j,4), W has 1 at (4,i)
        #
        # To get the full sum over j, we need 4 terms (one for each j):
        # Term j: U[i*4+j] = 1 for all i, V[j*5+4] = 1, W[4*4+i] = 1 for all i
        # But that's not rank-1 in the right sense.
        #
        # Actually, the correct decomposition:
        # For each j in 0..3:
        #   U has 1s at positions (0,j), (1,j), (2,j), (3,j) -> i.e., column j of A
        #   V has 1 at position (j, 4) -> i.e., B[j,4]
        #   W has 1s at positions (4,0), (4,1), (4,2), (4,3) -> i.e., W[16+i] for i=0..3
        #
        # Each such term is rank-1. Total: 4 terms.
        
        U_row = [0.0] * 16
        for i in range(4):
            U_row[i * 4 + j] = 1.0  # Column j of A
        
        V_row = [0.0] * 20
        V_row[j * 5 + 4] = 1.0  # B[j, 4]
        
        W_row = [0.0] * 20
        for i in range(4):
            W_row[4 * 4 + i] = 1.0  # W[16+i] = 1
        
        U_full.append(U_row)
        V_full.append(V_row)
        W_full.append(W_row)
    
    # Verify the full 4x4x5 decomposition
    def verify_full():
        rng = np.random.RandomState(777)
        for _ in range(5):
            A_t = rng.randint(-3, 4, (4,4)).astype(float)
            B_t = rng.randint(-3, 4, (4,5)).astype(float)
            C_expected = A_t @ B_t
            
            C_comp = np.zeros((4,5))
            for idx in range(total_rank):
                U_vec = np.array(U_full[idx])
                V_vec = np.array(V_full[idx])
                W_vec = np.array(W_full[idx])
                
                A_flat = A_t.flatten()  # 16 elements, row-major
                B_flat = B_t.flatten()  # 20 elements, row-major
                prod = np.dot(A_flat, U_vec) * np.dot(B_flat, V_vec)
                
                # W_vec is in order k*4+i, length 20
                for k in range(5):
                    for i in range(4):
                        C_comp[i, k] += prod * W_vec[k * 4 + i]
            
            if not np.allclose(C_comp, C_expected, atol=1e-10):
                return False
        return True
    
    full_ok = verify_full()
    
    if not full_ok:
        # Fall back to schoolbook
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
    
    return {"U": U_full, "V": V_full, "W": W_full}
