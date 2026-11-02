import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    # We use a Kronecker product based approach.
    # First, decompose M_{2,4,7} using the fact that M_{n,m,p} can be built from smaller blocks.
    # Specifically, we can use the "doubling" trick or simply the standard tensor decomposition.
    # However, the standard rank is n*m*p = 56.
    # Can we do better?
    # For 2x4x7, we can split the 4 dimension into 2+2.
    # M_{2,4,7} is isomorphic to two M_{2,2,7} blocks in parallel (with respect to the middle dimension).
    # So rank(M_{2,4,7}) <= 2 * rank(M_{2,2,7}).
    # Now, M_{2,2,7}. We can split 7 into 2+2+2+1.
    # rank(M_{2,2,7}) <= rank(M_{2,2,2}) + rank(M_{2,2,2}) + rank(M_{2,2,2}) + rank(M_{2,2,1}).
    # rank(M_{2,2,2}) = 7 (Strassen).
    # rank(M_{2,2,1}) = 4 (Schoolbook, since it's just dot products).
    # So rank(M_{2,2,7}) <= 7+7+7+4 = 25.
    # Then rank(M_{2,4,7}) <= 2 * 25 = 50.
    
    # Let's verify if we can do even better.
    # Is there a known bound?
    # rank(M_{2,2,p}) is known to be roughly 3p - 2 for large p? No, Strassen gives 7 for p=2.
    # For p=3, rank(M_{2,2,3}) is 10? (3*3-2 = 7? No. 2x2x3 tensor rank is 10? Actually, it's 10 or 11?
    # Let's check small cases.
    # M_{2,2,1}: rank 4.
    # M_{2,2,2}: rank 7.
    # M_{2,2,3}: rank 10? (Using 2+1 split: 7+4=11. But maybe 10 is possible? 
    # Actually, rank(M_{2,2,3}) = 10 is known? No, I think it's 10 or 11. 
    # Let's stick to the safe upper bound of 11 for M_{2,2,3} if we use 2+1 split.
    # Wait, 7+4=11.
    # So M_{2,2,7} = 7+7+7+4 = 25.
    # M_{2,4,7} = 2 * 25 = 50.
    
    # Let's construct the matrices explicitly.
    
    # Helper to get Strassen matrices for 2x2x2
    # U, V, W are 4x7 matrices.
    # Indices: i,j in 0..1 for A, j,k in 0..1 for B, k,i in 0..1 for C.
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r].
    # For 2x2x2, n=2, m=2, p=2.
    # U rows: 0,1,2,3 (i*2+j)
    # V rows: 0,1,2,3 (j*2+k)
    # W rows: 0,1,2,3 (k*2+i)
    
    def strassen_222():
        # Standard Strassen decomposition
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(f-h)
        # m4 = d(g-e)
        # m5 = (a+b)h
        # m6 = (c-d)(f+h)
        # m7 = (b-d)(e+g)
        # C11 = m1 + m4 - m5 + m7
        # C12 = m3 + m5
        # C21 = m2 + m4
        # C22 = m1 - m2 + m3 + m6
        
        # Map to tensor decomposition
        # U[r, l] corresponds to coefficient of A[row] in l-th term
        # V[r, l] corresponds to coefficient of B[row] in l-th term
        # W[r, l] corresponds to coefficient of C[row] in l-th term
        
        # A rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # B rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # C rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # m1: (a+d)(e+h)
        # a is A[0,0] -> U[0,0]=1
        # d is A[1,1] -> U[3,0]=1
        # e is B[0,0] -> V[0,0]=1
        # h is B[1,1] -> V[3,0]=1
        # C11: m1 -> W[0,0]=1
        # C22: m1 -> W[3,0]=1
        
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # m2: (c+d)e
        # c is A[1,0] -> U[2,1]=1
        # d is A[1,1] -> U[3,1]=1
        # e is B[0,0] -> V[0,1]=1
        # C21: m2 -> W[2,1]=1
        # C22: -m2 -> W[3,1]=-1
        
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # m3: a(f-h)
        # a is A[0,0] -> U[0,2]=1
        # f is B[0,1] -> V[1,2]=1
        # h is B[1,1] -> V[3,2]=-1
        # C12: m3 -> W[1,2]=1
        # C22: m3 -> W[3,2]=1
        
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # m4: d(g-e)
        # d is A[1,1] -> U[3,3]=1
        # g is B[1,0] -> V[2,3]=1
        # e is B[0,0] -> V[0,3]=-1
        # C11: m4 -> W[0,3]=1
        # C21: m4 -> W[2,3]=1
        
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # m5: (a+b)h
        # a is A[0,0] -> U[0,4]=1
        # b is A[0,1] -> U[1,4]=1
        # h is B[1,1] -> V[3,4]=1
        # C11: -m5 -> W[0,4]=-1
        # C12: m5 -> W[1,4]=1
        
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # m6: (c-d)(f+h)
        # c is A[1,0] -> U[2,5]=1
        # d is A[1,1] -> U[3,5]=-1
        # f is B[0,1] -> V[1,5]=1
        # h is B[1,1] -> V[3,5]=1
        # C22: m6 -> W[3,5]=1
        
        U[2, 5] = 1; U[3, 5] = -1
        V[1, 5] = 1; V[3, 5] = 1
        W[3, 5] = 1
        
        # m7: (b-d)(e+g)
        # b is A[0,1] -> U[1,6]=1
        # d is A[1,1] -> U[3,6]=-1
        # e is B[0,0] -> V[0,6]=1
        # g is B[1,0] -> V[2,6]=1
        # C11: m7 -> W[0,6]=1
        
        U[1, 6] = 1; U[3, 6] = -1
        V[0, 6] = 1; V[2, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def schoolbook_221():
        # 2x2x1 matrix multiplication
        # A is 2x2, B is 2x1. C is 2x1.
        # C[0] = A[0,0]B[0,0] + A[0,1]B[1,0]
        # C[1] = A[1,0]B[0,0] + A[1,1]B[1,0]
        # n=2, m=2, p=1.
        # U shape: (4, 4)
        # V shape: (2, 4)
        # W shape: (2, 4)
        
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        
        # Term 0: A[0,0] * B[0,0] -> C[0]
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1
        
        # Term 1: A[0,1] * B[1,0] -> C[0]
        U[1, 1] = 1
        V[1, 1] = 1
        W[0, 1] = 1
        
        # Term 2: A[1,0] * B[0,0] -> C[1]
        U[2, 2] = 1
        V[0, 2] = 1
        W[1, 2] = 1
        
        # Term 3: A[1,1] * B[1,0] -> C[1]
        U[3, 3] = 1
        V[1, 3] = 1
        W[1, 3] = 1
        
        return U, V, W

    # Build M_{2,2,7}
    # Split p=7 into 2,2,2,1
    # We have 4 blocks: 3 of type 2x2x2, 1 of type 2x2x1.
    # U_{2,2,7} shape: (4, 25)
    # V_{2,2,7} shape: (14, 25)  (m*p = 2*7 = 14)
    # W_{2,2,7} shape: (14, 25)  (p*n = 7*2 = 14)
    
    U_227 = np.zeros((4, 25))
    V_227 = np.zeros((14, 25))
    W_227 = np.zeros((14, 25))
    
    col_offset = 0
    # Block 1: k=0,1 (2x2x2)
    U_s, V_s, W_s = strassen_222()
    U_227[:, col_offset:col_offset+7] = U_s
    # V_s is 4x7. Rows correspond to j*p+k.
    # In M_{2,2,2}, j in 0..1, k in 0..1.
    # In M_{2,2,7}, j in 0..1, k in 0..6.
    # For block 1 (k=0,1), V rows are j*7+k.
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=1, k=0 -> 7
    # j=1, k=1 -> 8
    # V_s rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # So V_s[0] goes to V_227[0]
    # V_s[1] goes to V_227[1]
    # V_s[2] goes to V_227[7]
    # V_s[3] goes to V_227[8]
    V_227[0, col_offset:col_offset+7] = V_s[0]
    V_227[1, col_offset:col_offset+7] = V_s[1]
    V_227[7, col_offset:col_offset+7] = V_s[2]
    V_227[8, col_offset:col_offset+7] = V_s[3]
    
    # W_s is 4x7. Rows correspond to k*n+i.
    # In M_{2,2,2}, k in 0..1, i in 0..1.
    # In M_{2,2,7}, k in 0..6, i in 0..1.
    # For block 1 (k=0,1), W rows are k*2+i.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    # W_s rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    W_227[0, col_offset:col_offset+7] = W_s[0]
    W_227[1, col_offset:col_offset+7] = W_s[1]
    W_227[2, col_offset:col_offset+7] = W_s[2]
    W_227[3, col_offset:col_offset+7] = W_s[3]
    
    col_offset += 7
    
    # Block 2: k=2,3 (2x2x2)
    U_s, V_s, W_s = strassen_222()
    U_227[:, col_offset:col_offset+7] = U_s
    # V rows:
    # j=0, k=2 -> 2
    # j=0, k=3 -> 3
    # j=1, k=2 -> 9
    # j=1, k=3 -> 10
    V_227[2, col_offset:col_offset+7] = V_s[0]
    V_227[3, col_offset:col_offset+7] = V_s[1]
    V_227[9, col_offset:col_offset+7] = V_s[2]
    V_227[10, col_offset:col_offset+7] = V_s[3]
    
    # W rows:
    # k=2, i=0 -> 4
    # k=2, i=1 -> 5
    # k=3, i=0 -> 6
    # k=3, i=1 -> 7
    W_227[4, col_offset:col_offset+7] = W_s[0]
    W_227[5, col_offset:col_offset+7] = W_s[1]
    W_227[6, col_offset:col_offset+7] = W_s[2]
    W_227[7, col_offset:col_offset+7] = W_s[3]
    
    col_offset += 7
    
    # Block 3: k=4,5 (2x2x2)
    U_s, V_s, W_s = strassen_222()
    U_227[:, col_offset:col_offset+7] = U_s
    # V rows:
    # j=0, k=4 -> 4
    # j=0, k=5 -> 5
    # j=1, k=4 -> 11
    # j=1, k=5 -> 12
    V_227[4, col_offset:col_offset+7] = V_s[0]
    V_227[5, col_offset:col_offset+7] = V_s[1]
    V_227[11, col_offset:col_offset+7] = V_s[2]
    V_227[12, col_offset:col_offset+7] = V_s[3]
    
    # W rows:
    # k=4, i=0 -> 8
    # k=4, i=1 -> 9
    # k=5, i=0 -> 10
    # k=5, i=1 -> 11
    W_227[8, col_offset:col_offset+7] = W_s[0]
    W_227[9, col_offset:col_offset+7] = W_s[1]
    W_227[10, col_offset:col_offset+7] = W_s[2]
    W_227[11, col_offset:col_offset+7] = W_s[3]
    
    col_offset += 7
    
    # Block 4: k=6 (2x2x1)
    U_s, V_s, W_s = schoolbook_221()
    U_227[:, col_offset:col_offset+4] = U_s
    # V rows:
    # j=0, k=6 -> 6
    # j=1, k=6 -> 13
    V_227[6, col_offset:col_offset+4] = V_s[0]
    V_227[13, col_offset:col_offset+4] = V_s[1]
    
    # W rows:
    # k=6, i=0 -> 12
    # k=6, i=1 -> 13
    W_227[12, col_offset:col_offset+4] = W_s[0]
    W_227[13, col_offset:col_offset+4] = W_s[1]
    
    # Now build M_{2,4,7} from two M_{2,2,7} blocks.
    # Split m=4 into 2+2.
    # U_{2,4,7} shape: (8, 50)
    # V_{2,4,7} shape: (28, 50)
    # W_{2,4,7} shape: (14, 50)
    
    U_247 = np.zeros((8, 50))
    V_247 = np.zeros((28, 50))
    W_247 = np.zeros((14, 50))
    
    # Block 1: j=0,1
    # U rows: i*m+j.
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=1, j=0 -> 2
    # i=1, j=1 -> 3
    U_247[0:4, 0:25] = U_227
    
    # V rows: j*p+k.
    # j=0, k=0..6 -> 0..6
    # j=1, k=0..6 -> 7..13
    # V_227 rows 0..6 correspond to j=0, k=0..6
    # V_227 rows 7..13 correspond to j=1, k=0..6
    V_247[0:7, 0:25] = V_227[0:7, :]
    V_247[7:14, 0:25] = V_227[7:14, :]
    
    # W rows: k*n+i.
    # Same as W_227.
    W_247[:, 0:25] = W_227
    
    # Block 2: j=2,3
    # U rows:
    # i=0, j=2 -> 2
    # i=0, j=3 -> 3
    # i=1, j=2 -> 4
    # i=1, j=3 -> 5
    # Wait, U index is i*m+j.
    # i=0, j=2 -> 2
    # i=0, j=3 -> 3
    # i=1, j=2 -> 4
    # i=1, j=3 -> 5
    # But U_227 rows are 0,1,2,3 corresponding to i=0,j=0; i=0,j=1; i=1,j=0; i=1,j=1.
    # So for block 2, we map U_227[0] -> U_247[2], U_227[1] -> U_247[3], U_227[2] -> U_247[4], U_227[3] -> U_247[5].
    U_247[2, 25:50] = U_227[0]
    U_247[3, 25:50] = U_227[1]
    U_247[4, 25:50] = U_227[2]
    U_247[5, 25:50] = U_227[3]
    
    # V rows:
    # j=2, k=0..6 -> 14..20
    # j=3, k=0..6 -> 21..27
    # V_227 rows 0..6 -> j=0, k=0..6. Here we need j=2, so rows 14..20.
    # V_227 rows 7..13 -> j=1, k=0..6. Here we need j=3, so rows 21..27.
    V_247[14:21, 25:50] = V_227[0:7, :]
    V_247[21:28, 25:50] = V_227[7:14, :]
    
    # W rows:
    # Same as W_227.
    W_247[:, 25:50] = W_227
    
    return {"U": U_247.tolist(), "V": V_247.tolist(), "W": W_247.tolist()}
