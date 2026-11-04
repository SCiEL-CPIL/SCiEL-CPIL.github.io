"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will use a structured decomposition based on Strassen's algorithm
    # for 2x2x2 sub-problems and simple rank-1 terms for the remainders.
    
    # Dimensions: n=3, m=4, p=5
    # We partition the indices:
    # n: {0,1}, {2}
    # m: {0,1}, {2,3}
    # p: {0,1}, {2,3}, {4}
    
    # 4 disjoint 2x2x2 blocks:
    # 1. i in {0,1}, j in {0,1}, k in {0,1}
    # 2. i in {0,1}, j in {2,3}, k in {0,1}
    # 3. i in {0,1}, j in {0,1}, k in {2,3}
    # 4. i in {0,1}, j in {2,3}, k in {2,3}
    
    # Remainders:
    # 5. i=2, j in {0,1,2,3}, k in {0,1,2,3,4} -> Rank 1
    # 6. i in {0,1}, j in {0,1,2,3}, k=4 -> Rank 1
    
    # Total Rank = 4 * 7 + 1 + 1 = 30
    
    rank = 30
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def add_strassen_block(i1, i2, j1, j2, k1, k2, offset):
        # Strassen terms for 2x2x2 block
        # Local indices 0,1 map to (i1,j1,k1) and (i2,j2,k2)
        # A = [[A00, A01], [A10, A11]] -> A[i1,j1], A[i1,j2], A[i2,j1], A[i2,j2]
        # B = [[B00, B01], [B10, B11]] -> B[j1,k1], B[j1,k2], B[j2,k1], B[j2,k2]
        
        # P1 = (A00 + A11)(B00 + B11)
        # U[i1,j1] += 1, U[i2,j2] += 1
        # V[j1,k1] += 1, V[j2,k2] += 1
        # W[k1,i1] += 1, W[k2,i2] += 1
        # But wait, Strassen combines these.
        # The rank-1 tensors are:
        # 1. (A00+A11) * (B00+B11) * (C00+C11)
        # This is not a single rank-1 tensor in the original space, it's a sum.
        # The 7 products are:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Each Mk is a scalar.
        # The contribution to C is:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to represent this as sum of rank-1 tensors.
        # A rank-1 tensor is defined by vectors u, v, w.
        # u[i,j] v[j,k] w[k,i]
        
        # For M1:
        # u1[i1,j1]=1, u1[i2,j2]=1
        # v1[j1,k1]=1, v1[j2,k2]=1
        # w1[k1,i1]=1, w1[k2,i2]=1
        # This term contributes to C[i1,k1], C[i1,k2], C[i2,k1], C[i2,k2]
        # Specifically:
        # C[i1,k1] += u1[i1,j1]v1[j1,k1]w1[k1,i1] + ... (sums over j)
        # Actually, for a fixed term c, the contribution to T[i,j,k] is U[i,j,c]V[j,k,c]W[k,i,c].
        # We want to sum over c.
        
        # Let's define the 7 terms explicitly.
        
        # Term 1: M1
        # U: u[i1,j1]=1, u[i2,j2]=1
        # V: v[j1,k1]=1, v[j2,k2]=1
        # W: w[k1,i1]=1, w[k2,i2]=1
        # This generates non-zero values at:
        # (i1,j1,k1): 1*1*1 = 1
        # (i1,j2,k2): 0*1*1 = 0 (U[i1,j2] is 0) -> Wait.
        # The tensor is T[i,j,k].
        # If we set U[i1,j1]=1, U[i2,j2]=1, and others 0.
        # V[j1,k1]=1, V[j2,k2]=1, and others 0.
        # W[k1,i1]=1, W[k2,i2]=1, and others 0.
        
        # Contribution to T[i,j,k]:
        # If i=i1, j=j1, k=k1: U[i1,j1]V[j1,k1]W[k1,i1] = 1.
        # If i=i1, j=j2, k=k2: U[i1,j2] is 0. So 0.
        # If i=i2, j=j1, k=k1: U[i2,j1] is 0. So 0.
        # If i=i2, j=j2, k=k2: U[i2,j2]V[j2,k2]W[k2,i2] = 1.
        
        # So this term only contributes to (i1,j1,k1) and (i2,j2,k2).
        # But Strassen M1 should contribute to all 4 outputs?
        # No, M1 is an intermediate product.
        # The final C is a linear combination of M1..M7.
        # So we need to construct the U, V, W such that the sum of the rank-1 tensors
        # equals the desired T[i,j,k]=1 for all i,j,k in the block.
        
        # Actually, the standard Strassen decomposition of the 2x2x2 tensor
        # (which is all ones) into 7 rank-1 tensors is known.
        
        # The 7 rank-1 tensors correspond to the 7 multiplications.
        # Let's list them:
        
        # 1. (A00+A11) * (B00+B11)
        # u: [1,0; 0,1] (for A part) -> u[i1,j1]=1, u[i2,j2]=1
        # v: [1,0; 0,1] (for B part) -> v[j1,k1]=1, v[j2,k2]=1
        # w: [1,0; 0,1] (for C part) -> w[k1,i1]=1, w[k2,i2]=1
        # This term is T1.
        
        # 2. (A10+A11) * B00
        # u: [0,0; 1,1] -> u[i2,j1]=1, u[i2,j2]=1
        # v: [1,0; 0,0] -> v[j1,k1]=1
        # w: [1,0; 0,0] -> w[k1,i1]=1
        # This term is T2.
        
        # 3. A00 * (B01-B11)
        # u: [1,0; 0,0] -> u[i1,j1]=1
        # v: [0,1; 0,-1] -> v[j1,k2]=1, v[j2,k2]=-1
        # w: [0,1; 0,0] -> w[k2,i1]=1
        # This term is T3.
        
        # 4. A11 * (B10-B00)
        # u: [0,0; 0,1] -> u[i2,j2]=1
        # v: [0,0; 1,-1] -> v[j2,k1]=1, v[j1,k1]=-1
        # w: [1,0; 0,0] -> w[k1,i2]=1
        # This term is T4.
        
        # 5. (A00+A01) * B11
        # u: [1,1; 0,0] -> u[i1,j1]=1, u[i1,j2]=1
        # v: [0,0; 0,1] -> v[j2,k2]=1
        # w: [0,0; 1,0] -> w[k2,i2]=1
        # This term is T5.
        
        # 6. (A10-A00) * (B00+B01)
        # u: [0,0; 1,-1] -> u[i2,j1]=1, u[i1,j1]=-1
        # v: [1,1; 0,0] -> v[j1,k1]=1, v[j1,k2]=1
        # w: [1,1; 0,0] -> w[k1,i1]=1, w[k2,i1]=1
        # This term is T6.
        
        # 7. (A01-A11) * (B10+B11)
        # u: [0,1; 0,-1] -> u[i1,j2]=1, u[i2,j2]=-1
        # v: [0,0; 1,1] -> v[j2,k1]=1, v[j2,k2]=1
        # w: [0,1; 0,1] -> w[k1,i2]=1, w[k2,i2]=1
        # This term is T7.
        
        # We add these to U, V, W at the given offset.
        
        # T1
        U[i1*m+j1, offset] += 1
        U[i2*m+j2, offset] += 1
        V[j1*p+k1, offset] += 1
        V[j2*p+k2, offset] += 1
        W[k1*n+i1, offset] += 1
        W[k2*n+i2, offset] += 1
        
        # T2
        U[i2*m+j1, offset+1] += 1
        U[i2*m+j2, offset+1] += 1
        V[j1*p+k1, offset+1] += 1
        W[k1*n+i1, offset+1] += 1
        
        # T3
        U[i1*m+j1, offset+2] += 1
        V[j1*p+k2, offset+2] += 1
        V[j2*p+k2, offset+2] -= 1
        W[k2*n+i1, offset+2] += 1
        
        # T4
        U[i2*m+j2, offset+3] += 1
        V[j2*p+k1, offset+3] += 1
        V[j1*p+k1, offset+3] -= 1
        W[k1*n+i2, offset+3] += 1
        
        # T5
        U[i1*m+j1, offset+4] += 1
        U[i1*m+j2, offset+4] += 1
        V[j2*p+k2, offset+4] += 1
        W[k2*n+i2, offset+4] += 1
        
        # T6
        U[i2*m+j1, offset+5] += 1
        U[i1*m+j1, offset+5] -= 1
        V[j1*p+k1, offset+5] += 1
        V[j1*p+k2, offset+5] += 1
        W[k1*n+i1, offset+5] += 1
        W[k2*n+i1, offset+5] += 1
        
        # T7
        U[i1*m+j2, offset+6] += 1
        U[i2*m+j2, offset+6] -= 1
        V[j2*p+k1, offset+6] += 1
        V[j2*p+k2, offset+6] += 1
        W[k1*n+i2, offset+6] += 1
        W[k2*n+i2, offset+6] += 1

    # Apply Strassen blocks
    add_strassen_block(0, 1, 0, 1, 0, 1, 0)
    add_strassen_block(0, 1, 2, 3, 0, 1, 7)
    add_strassen_block(0, 1, 0, 1, 2, 3, 14)
    add_strassen_block(0, 1, 2, 3, 2, 3, 21)
    
    # Remainder 1: i=2, j in {0,1,2,3}, k in {0,1,2,3,4}
    # Term 28
    for j in range(m):
        U[2*m+j, 28] = 1
    for j in range(m):
        V[j*p, 28] = 1 # k=0? No, k is any.
    # Wait, V is indexed by (j,k).
    # We want V[j,k] = 1 for all k?
    # If V[j,k] = 1 for all k, and W[k,2] = 1 for all k,
    # and U[2,j] = 1 for all j.
    # Then T[2,j,k] = sum_c U[2,j,c] V[j,k,c] W[k,2,c]
    # For c=28: 1 * 1 * 1 = 1.
    # We need V[j,k,28] = 1 for all j,k.
    for j in range(m):
        for k in range(p):
            V[j*p+k, 28] = 1
    for k in range(p):
        W[k*n+2, 28] = 1
        
    # Remainder 2: i in {0,1}, j in {0,1,2,3}, k=4
    # Term 29
    for i in range(2):
        for j in range(m):
            U[i*m+j, 29] = 1
    for j in range(m):
        V[j*p+4, 29] = 1 # Only k=4
    for i in range(2):
        W[4*n+i, 29] = 1
        
    # Verify
    def verify(U, V, W):
        n, m, p = DIMENSIONS
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1.0) > 1e-8:
                        return False
        return True
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    if not verify(U_list, V_list, W_list):
        # Fallback to schoolbook
        rank_sb = n*m*p
        U_sb = np.zeros((n*m, rank_sb))
        V_sb = np.zeros((m*p, rank_sb))
        W_sb = np.zeros((p*n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
        
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
