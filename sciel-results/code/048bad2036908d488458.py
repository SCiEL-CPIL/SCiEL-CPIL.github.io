import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def _verify_decomposition(U, V, W, n, m, p):
    """
    Fast verification using einsum.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    r = U.shape[1]
    if r == 0:
        return True
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # T[a, b, c, d, i, k] = sum_l U[a*m+b, l] * V[c*p+d, l] * W[k*n+i, l]
    # We want to check if this equals delta_{a,i} delta_{b,d} delta_{c,k}
    
    # Reshape for efficient computation
    # U is (n*m, r). Reshape to (n, m, r)
    U3 = U_np.reshape(n, m, r)
    # V is (m*p, r). Reshape to (m, p, r)
    V3 = V_np.reshape(m, p, r)
    # W is (p*n, r). Reshape to (p, n, r)
    W3 = W_np.reshape(p, n, r)
    
    # Compute T[a, b, c, d, i, k] = sum_l U3[a,b,l] V3[c,d,l] W3[k,i,l]
    # This is a contraction over l.
    # Let's compute it in steps to avoid huge intermediate arrays if possible, 
    # but n=2, m=3, p=6, r<=36. Total size 2*3*3*6*2*6 = 1296. Very small.
    
    # T[a, b, c, d, i, k]
    # Sum over l: U3[a,b,l] * V3[c,d,l] -> (n, m, m, p)
    # Then multiply by W3[k,i,l] and sum over l? No, l is summed in the first step.
    # Wait, the sum is over l for the product of all three.
    # T[a, b, c, d, i, k] = sum_l (U3[a,b,l] * V3[c,d,l] * W3[k,i,l])
    
    # This can be computed as:
    # Temp1[a, b, c, d, l] = U3[a,b,l] * V3[c,d,l]  (broadcasting)
    # T = np.einsum('abcl,cdl,ikl->abcdik', U3, V3, W3) ? No, indices don't match.
    
    # Let's use einsum directly on the flattened or reshaped arrays.
    # U3: a, b, l
    # V3: c, d, l
    # W3: k, i, l
    
    # T[a, b, c, d, i, k] = sum_l U3[a,b,l] V3[c,d,l] W3[k,i,l]
    T = np.einsum('abl,cdl,kil->abcdik', U3, V3, W3)
    
    # Target tensor
    # Target[a, b, c, d, i, k] = 1 if a==i, b==d, c==k else 0
    # We can construct this or check element-wise.
    # Since the size is small, let's construct the target.
    
    # Identity matrices for the deltas
    # delta_{a,i}: shape (n, n)
    # delta_{b,d}: shape (m, p) -- Note: b is 0..m-1, d is 0..p-1.
    #   Wait, b is column of A (0..m-1), d is column of B (0..p-1).
    #   The condition is b == d. But b ranges 0..2 and d ranges 0..5.
    #   So delta_{b,d} is 1 if b==d, else 0.
    # delta_{c,k}: shape (m, p) -- c is row of B (0..m-1), k is row of C (0..p-1).
    #   Condition c == k. c ranges 0..2, k ranges 0..5.
    
    # Let's build the target tensor T_target of shape (n, m, m, p, n, p)
    T_target = np.zeros((n, m, m, p, n, p))
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for i in range(n):
                        for k in range(p):
                            if a == i and b == d and c == k:
                                T_target[a, b, c, d, i, k] = 1.0
    
    if not np.allclose(T, T_target, atol=1e-9):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin

    n, m, p = DIMENSIONS
    
    # Strategy: Structural Block Decomposition with Correct Strassen Implementation
    # The previous program failed likely due to incorrect Strassen coefficients or indexing.
    # We will construct a valid Rank 24 decomposition by combining 3 blocks of Rank 8 for M_{2,3,2}.
    # Rank 8 for M_{2,3,2} comes from Rank 7 (Strassen for 2x2x2) + Rank 1 (extra term).
    
    # Verify the Rank 8 sub-decomposition first.
    # Dimensions for sub-problem: n_sub=2, m_sub=3, p_sub=2
    # U_sub: (2*3, 8) = (6, 8)
    # V_sub: (3*2, 8) = (6, 8)
    # W_sub: (2*2, 8) = (4, 8)
    
    U_sub = np.zeros((6, 8))
    V_sub = np.zeros((6, 8))
    W_sub = np.zeros((4, 8))
    
    # Indices in sub-problem:
    # A is 2x3. A[i, j]. Index in U: i*3 + j.
    #   A[0,0]=0, A[0,1]=1, A[0,2]=2
    #   A[1,0]=3, A[1,1]=4, A[1,2]=5
    # B is 3x2. B[j, k]. Index in V: j*2 + k.
    #   B[0,0]=0, B[0,1]=1
    #   B[1,0]=2, B[1,1]=3
    #   B[2,0]=4, B[2,1]=5
    # C is 2x2. C[i, k]. Index in W: k*2 + i (per task convention W[k*n+i]).
    #   C[0,0]=0, C[0,1]=1
    #   C[1,0]=2, C[1,1]=3
    
    # Strassen Terms for A[:, 0:2] and B[0:2, :]
    # These involve A[0,0], A[0,1], A[1,0], A[1,1] and B[0,0], B[0,1], B[1,0], B[1,1]
    # U indices: 0, 1, 3, 4
    # V indices: 0, 1, 2, 3
    
    # T0: (A00+A11)(B00+B11)
    # U: A[0,0]=0, A[1,1]=4
    U_sub[0, 0] = 1; U_sub[4, 0] = 1
    # V: B[0,0]=0, B[1,1]=3
    V_sub[0, 0] = 1; V_sub[3, 0] = 1
    # W: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    W_sub[0, 0] = 1; W_sub[1, 0] = 1; W_sub[2, 0] = 1; W_sub[3, 0] = 1
    
    # T1: (A10+A11)B00
    # U: A[1,0]=3, A[1,1]=4
    U_sub[3, 1] = 1; U_sub[4, 1] = 1
    # V: B[0,0]=0
    V_sub[0, 1] = 1
    # W: C[1,0]=2 (+), C[1,1]=3 (-)
    W_sub[2, 1] = 1; W_sub[3, 1] = -1
    
    # T2: A00(B10-B11)
    # U: A[0,0]=0
    U_sub[0, 2] = 1
    # V: B[1,0]=2, B[1,1]=3
    V_sub[2, 2] = 1; V_sub[3, 2] = -1
    # W: C[0,1]=1 (+), C[1,1]=3 (+)
    W_sub[1, 2] = 1; W_sub[3, 2] = 1
    
    # T3: A11(B01-B00)
    # U: A[1,1]=4
    U_sub[4, 3] = 1
    # V: B[0,1]=1, B[0,0]=0
    V_sub[1, 3] = 1; V_sub[0, 3] = -1
    # W: C[0,0]=0 (+), C[1,0]=2 (+)
    W_sub[0, 3] = 1; W_sub[2, 3] = 1
    
    # T4: (A00+A01)B11
    # U: A[0,0]=0, A[0,1]=1
    U_sub[0, 4] = 1; U_sub[1, 4] = 1
    # V: B[1,1]=3
    V_sub[3, 4] = 1
    # W: C[0,0]=0 (-), C[0,1]=1 (+)
    W_sub[0, 4] = -1; W_sub[1, 4] = 1
    
    # T5: (A10-A00)(B00+B01)
    # U: A[1,0]=3, A[0,0]=0
    U_sub[3, 5] = 1; U_sub[0, 5] = -1
    # V: B[0,0]=0, B[0,1]=1
    V_sub[0, 5] = 1; V_sub[1, 5] = 1
    # W: C[1,0]=2 (+), C[1,1]=3 (+)
    W_sub[2, 5] = 1; W_sub[3, 5] = 1
    
    # T6: (A01-A11)(B10+B11)
    # U: A[0,1]=1, A[1,1]=4
    U_sub[1, 6] = 1; U_sub[4, 6] = -1
    # V: B[1,0]=2, B[1,1]=3
    V_sub[2, 6] = 1; V_sub[3, 6] = 1
    # W: C[0,0]=0 (+), C[1,1]=3 (+)
    W_sub[0, 6] = 1; W_sub[3, 6] = 1
    
    # T7: Extra term for A[:, 2] and B[2, :]
    # U: A[0,2]=2, A[1,2]=5
    U_sub[2, 7] = 1; U_sub[5, 7] = 1
    # V: B[2,0]=4, B[2,1]=5
    V_sub[4, 7] = 1; V_sub[5, 7] = 1
    # W: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    W_sub[0, 7] = 1; W_sub[1, 7] = 1; W_sub[2, 7] = 1; W_sub[3, 7] = 1
    
    # Verify the sub-decomposition
    if _verify_decomposition(U_sub, V_sub, W_sub, 2, 3, 2):
        # Construct the full M_{2,3,6} decomposition using 3 blocks
        # Block b (b=0,1,2) handles B columns 2b, 2b+1 and C columns 2b, 2b+1
        # A is shared.
        
        U_full = np.zeros((6, 24))
        V_full = np.zeros((18, 24))
        W_full = np.zeros((12, 24))
        
        for b in range(3):
            # U is the same for all blocks
            U_full[:, b*8:(b+1)*8] = U_sub
            
            # V mapping
            # Full V index: j*6 + k (j=0..2, k=0..5)
            # Sub V index: j*2 + k_local (j=0..2, k_local=0..1)
            # k = 2*b + k_local
            for j in range(3):
                for k_local in range(2):
                    k_global = 2*b + k_local
                    V_full[j*6 + k_global, b*8:(b+1)*8] = V_sub[j*2 + k_local, :]
            
            # W mapping
            # Full W index: k*2 + i (k=0..5, i=0..1)
            # Sub W index: k_local*2 + i (k_local=0..1, i=0..1)
            # k = 2*b + k_local
            for k_local in range(2):
                for i in range(2):
                    k_global = 2*b + k_local
                    W_full[k_global*2 + i, b*8:(b+1)*8] = W_sub[k_local*2 + i, :]
        
        # Verify the full decomposition
        if _verify_decomposition(U_full, V_full, W_full, n, m, p):
            return {
                "U": U_full.tolist(),
                "V": V_full.tolist(),
                "W": W_full.tolist()
            }
    
    # Fallback: Schoolbook Rank 36
    U_school = np.zeros((6, 36))
    V_school = np.zeros((18, 36))
    W_school = np.zeros((12, 36))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, idx] = 1.0
                V_school[j*p + k, idx] = 1.0
                W_school[k*n + i, idx] = 1.0
                idx += 1
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }

if __name__ == "__main__":
    construct()
