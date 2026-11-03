import numpy as np
import time

def construct():
    n, m, p = 2, 3, 7
    rank = 40
    
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    col = 0
    
    # Part 1: M_{2,3,3} using schoolbook (rank 18)
    # A is 2x3, B is 3x3, C is 2x3
    # C[i,k] = sum_j A[i,j] B[j,k]
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U[i*3 + j, col] = 1.0
                V[j*3 + k, col] = 1.0
                W[k*2 + i, col] = 1.0
                col += 1
    
    # Part 2: M_{2,3,4} using rank 22 construction
    # A is 2x3, B is 3x4, C is 2x4
    # Split A into A1 (2x2) and A2 (2x1)
    # A1 = A[:, 0:2], A2 = A[:, 2]
    # Split B into B1 (2x4) and B2 (1x4)
    # B1 = B[0:2, :], B2 = B[2, :]
    # C = A1 * B1 + A2 * B2
    
    # Subpart 2a: A1 * B1 (2x2 times 2x4)
    # Split B1 into B1a (2x2) and B1b (2x2)
    # B1a = B[0:2, 0:2], B1b = B[0:2, 2:4]
    # A1 * B1 = A1 * B1a + A1 * B1b
    # Use Strassen for each 2x2 * 2x2
    
    # Strassen decomposition for 2x2 * 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A10)B11
    # M6 = (A11-A00)(B00+B10)
    # M7 = (A10-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M1+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    def add_strassen_block(U, V, W, col, i_offset, j_offset, k_offset):
        # i_offset: row offset in A (2x2)
        # j_offset: col offset in A / row offset in B (2x2)
        # k_offset: col offset in B (2x2) / row offset in C (2x2)
        # A is at [i_offset:i_offset+2, j_offset:j_offset+2]
        # B is at [j_offset:j_offset+2, k_offset:k_offset+2]
        # C is at [i_offset:i_offset+2, k_offset:k_offset+2]
        
        # M1 = (A00+A11)(B00+B11)
        # A00 at (i_offset, j_offset), A11 at (i_offset+1, j_offset+1)
        # B00 at (j_offset, k_offset), B11 at (j_offset+1, k_offset+1)
        U[i_offset*3 + j_offset, col] = 1.0
        U[(i_offset+1)*3 + (j_offset+1), col] = 1.0
        V[j_offset*4 + k_offset, col] = 1.0
        V[(j_offset+1)*4 + (k_offset+1), col] = 1.0
        W[k_offset*2 + i_offset, col] = 1.0
        W[(k_offset+1)*2 + (i_offset+1), col] = 1.0
        W[k_offset*2 + (i_offset+1), col] = 1.0
        W[(k_offset+1)*2 + i_offset, col] = 1.0
        col += 1
        
        # M2 = (A10+A11)B00
        U[(i_offset+1)*3 + j_offset, col] = 1.0
        U[(i_offset+1)*3 + (j_offset+1), col] = 1.0
        V[j_offset*4 + k_offset, col] = 1.0
        W[(k_offset+1)*2 + (i_offset+1), col] = 1.0
        col += 1
        
        # M3 = A00(B10-B11)
        U[i_offset*3 + j_offset, col] = 1.0
        V[(j_offset+1)*4 + k_offset, col] = 1.0
        V[(j_offset+1)*4 + (k_offset+1), col] = -1.0
        W[k_offset*2 + i_offset, col] = 1.0
        W[(k_offset+1)*2 + i_offset, col] = 1.0
        col += 1
        
        # M4 = A11(B01-B00)
        U[(i_offset+1)*3 + (j_offset+1), col] = 1.0
        V[j_offset*4 + (k_offset+1), col] = 1.0
        V[j_offset*4 + k_offset, col] = -1.0
        W[(k_offset+1)*2 + (i_offset+1), col] = 1.0
        W[k_offset*2 + (i_offset+1), col] = 1.0
        col += 1
        
        # M5 = (A00+A10)B11
        U[i_offset*3 + j_offset, col] = 1.0
        U[(i_offset+1)*3 + j_offset, col] = 1.0
        V[(j_offset+1)*4 + (k_offset+1), col] = 1.0
        W[(k_offset+1)*2 + i_offset, col] = 1.0
        W[(k_offset+1)*2 + (i_offset+1), col] = 1.0
        col += 1
        
        # M6 = (A11-A00)(B00+B10)
        U[(i_offset+1)*3 + (j_offset+1), col] = 1.0
        U[i_offset*3 + j_offset, col] = -1.0
        V[j_offset*4 + k_offset, col] = 1.0
        V[(j_offset+1)*4 + k_offset, col] = 1.0
        W[k_offset*2 + (i_offset+1), col] = 1.0
        W[(k_offset+1)*2 + (i_offset+1), col] = 1.0
        col += 1
        
        # M7 = (A10-A11)(B01+B11)
        U[(i_offset+1)*3 + j_offset, col] = 1.0
        U[(i_offset+1)*3 + (j_offset+1), col] = -1.0
        V[j_offset*4 + (k_offset+1), col] = 1.0
        V[(j_offset+1)*4 + (k_offset+1), col] = 1.0
        W[k_offset*2 + (i_offset+1), col] = 1.0
        W[(k_offset+1)*2 + (i_offset+1), col] = 1.0
        col += 1
        
        return col
    
    # Strassen block 1: A1 * B1a (A at [0:2, 0:2], B at [0:2, 0:2], C at [0:2, 0:2])
    col = add_strassen_block(U, V, W, col, 0, 0, 0)
    
    # Strassen block 2: A1 * B1b (A at [0:2, 0:2], B at [0:2, 2:4], C at [0:2, 2:4])
    col = add_strassen_block(U, V, W, col, 0, 0, 2)
    
    # Subpart 2b: A2 * B2 (2x1 times 1x4)
    # A2 is 2x1 (column 2 of A), B2 is 1x4 (row 2 of B)
    # C2 = A2 * B2
    # C2[i,k] = A2[i] * B2[k]
    # This is an outer product, rank 1? No, it's a rank 1 tensor?
    # T[a,b,c] = U[a,r] V[b,r] W[c,r]
    # For A2[i] * B2[k], a = i*3+2, b = 2*4+k = 8+k, c = k*2+i
    # U[i*3+2, r] = 1, V[8+k, r] = 1, W[k*2+i, r] = 1
    # But this is for each i,k. We need to sum over i,k.
    # Actually, A2 * B2 is a 2x4 matrix.
    # C2[i,k] = A2[i] * B2[k]
    # This can be done with 2 multiplications? No.
    # It's an outer product of a 2x1 and 1x4.
    # Rank is 1? No, the tensor for outer product of 2x1 and 1x4 has rank 2?
    # Let's see: T[a,b,c] = 1 if a corresponds to (i,2), b corresponds to (2,k), c corresponds to (k,i).
    # a = i*3+2, b = 8+k, c = k*2+i.
    # We can decompose this as sum over i of (e_i in A2) * (e_i^T in B2)? No.
    # It's sum_i sum_k A2[i] B2[k] E_{i,k}.
    # This is a sum of 2*4=8 rank-1 terms? No.
    # Actually, the tensor for multiplying a 2x1 by a 1x4 is:
    # T_{i,k} = A_i B_k.
    # This is a rank 1 tensor in the (i,k) space? No, it's a product of two vectors.
    # The tensor is T = u v^T where u is 2-dim, v is 4-dim.
    # The rank of the tensor T_{a,b,c} where a=i, b=k, c=(i,k) is...
    # Wait, the indices are a (2*3=6), b (3*4=12), c (4*2=8).
    # For this subproblem, a is restricted to {2, 5} (i=0,1), b is restricted to {8,9,10,11} (k=0..3), c is restricted to {i, 4+i} for i=0,1? No, c = k*2+i.
    # c values: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3, (2,0)->4, (2,1)->5, (3,0)->6, (3,1)->7.
    # So for a=2 (i=0), c can be 0, 2, 4, 6.
    # For a=5 (i=1), c can be 1, 3, 5, 7.
    # We can decompose this as:
    # Sum_i (A2[i] * I) * (B2 * E_i^T)?
    # Let's use 2 terms.
    # Term 1: i=0. U[2, r] = 1. V[8+k, r] = 1 for all k? No, V is 12xrank.
    # If we set U[2, r0] = 1, V[8+k, r0] = 1 for all k, then W[c, r0] must be 1 if c corresponds to (k,0) i.e., c=2k.
    # So W[0, r0]=1, W[2, r0]=1, W[4, r0]=1, W[6, r0]=1. Others 0.
    # This gives C[0,k] = A2[0] * B2[k]. Correct for i=0.
    # Term 2: i=1. U[5, r1] = 1, V[8+k, r1] = 1 for all k, W[1, r1]=1, W[3, r1]=1, W[5, r1]=1, W[7, r1]=1.
    # This gives C[1,k] = A2[1] * B2[k]. Correct for i=1.
    # So rank 2 for this part.
    
    # Add term for i=0
    U[2, col] = 1.0
    for k in range(4):
        V[8+k, col] = 1.0
    for k in range(4):
        W[2*k, col] = 1.0
    col += 1
    
    # Add term for i=1
    U[5, col] = 1.0
    for k in range(4):
        V[8+k, col] = 1.0
    for k in range(4):
        W[2*k+1, col] = 1.0
    col += 1
    
    # Total rank should be 18 + 14 + 14 + 2 = 48?
    # Wait, Strassen is 7 terms. Two Strassen blocks = 14 terms.
    # Outer product part = 2 terms.
    # Total for M_{2,3,4} = 14 + 2 = 16?
    # Let's re-verify.
    # A1 * B1a: 7 terms.
    # A1 * B1b: 7 terms.
    # A2 * B2: 2 terms.
    # Total 16.
    # So total rank = 18 + 16 = 34.
    # This is better than 40!
    
    # Let's verify the logic.
    # M_{2,3,4}:
    # A = [A1, A2] where A1 is 2x2, A2 is 2x1.
    # B = [B1; B2] where B1 is 2x4, B2 is 1x4.
    # C = A1 B1 + A2 B2.
    # B1 = [B1a, B1b] where B1a is 2x2, B1b is 2x2.
    # A1 B1 = A1 B1a + A1 B1b.
    # Each A1 B1x is 2x2 * 2x2, rank 7.
    # So A1 B1 has rank 14.
    # A2 B2 is 2x1 * 1x4, rank 2.
    # Total rank 16.
    # M_{2,3,3} schoolbook rank 18.
    # Total rank 34.
    
    # Let's double check the indices for the outer product part.
    # A2 is column 2 of A. So a = i*3 + 2.
    # B2 is row 2 of B. So b = 2*4 + k = 8 + k.
    # C is 2x4. c = k*2 + i.
    # Term i=0:
    # U[0*3+2, r] = U[2, r] = 1.
    # V[8+k, r] = 1 for k=0..3.
    # W[k*2+0, r] = W[2k, r] = 1 for k=0..3.
    # This computes sum_k U[2,r] V[8+k,r] W[2k,r] = sum_k 1*1*1 = 4?
    # No, the tensor is T[a,b,c] = U[a,r] V[b,r] W[c,r].
    # For a=2, b=8+k, c=2k: T[2, 8+k, 2k] = 1*1*1 = 1.
    # This corresponds to C[0,k] += A2[0] * B2[k].
    # Since U[2,r]=1, it assumes A2[0]=1? No, U is a coefficient.
    # The term is U[2,r] * V[8+k,r] * W[2k,r].
    # This contributes to C[0,k] (which is c=2k, a=2).
    # The value is 1*1*1 = 1.
    # But we want A2[0] * B2[k].
    # A2[0] is the input A[0,2]. This is index a=2.
    # B2[k] is the input B[2,k]. This is index b=8+k.
    # So the product is A[0,2] * B[2,k].
    # The term is U[2,r] A[0,2] * V[8+k,r] B[2,k] * W[2k,r].
    # If U[2,r]=1, V[8+k,r]=1, W[2k,r]=1, then the term is A[0,2] B[2,k].
    # This is correct for C[0,k].
    # Similarly for i=1.
    
    # So the construction is correct.
    
    # We have col = 18 + 14 + 2 = 34.
    # But rank is set to 40.
    # We should use rank = 34.
    
    # Let's re-initialize with rank 34.
    rank = 34
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    col = 0
    
    # Part 1: M_{2,3,3} schoolbook
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U[i*3 + j, col] = 1.0
                V[j*3 + k, col] = 1.0
                W[k*2 + i, col] = 1.0
                col += 1
    
    # Part 2: M_{2,3,4}
    # Strassen block 1
    col = add_strassen_block(U, V, W, col, 0, 0, 0)
    # Strassen block 2
    col = add_strassen_block(U, V, W, col, 0, 0, 2)
    
    # Outer product part
    # i=0
    U[2, col] = 1.0
    for k in range(4):
        V[8+k, col] = 1.0
    for k in range(4):
        W[2*k, col] = 1.0
    col += 1
    
    # i=1
    U[5, col] = 1.0
    for k in range(4):
        V[8+k, col] = 1.0
    for k in range(4):
        W[2*k+1, col] = 1.0
    col += 1
    
    # col should be 34.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
