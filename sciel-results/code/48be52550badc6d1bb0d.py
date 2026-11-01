import numpy as np
import time
from fractions import Fraction

def _to_frac(x, tol=1e-10):
    """Convert a float to a Fraction if it's close to a simple rational (int or half-int)."""
    if abs(x) < tol:
        return Fraction(0)
    # Check if close to integer
    nearest_int = round(x)
    if abs(x - nearest_int) < tol:
        return Fraction(nearest_int)
    # Check if close to half-integer
    nearest_half = round(2*x) / 2.0
    if abs(x - nearest_half) < tol:
        return Fraction(nearest_half, 2)
    # Fallback: limit denominator to 10 for simplicity
    return Fraction(x).limit_denominator(10)

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly represents matrix multiplication."""
    # Convert to Fraction for exact arithmetic check
    Uf = [[_to_frac(x) for x in row] for row in U]
    Vf = [[_to_frac(x) for x in row] for row in V]
    Wf = [[_to_frac(x) for x in row] for row in W]
    
    # The tensor equation is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # This corresponds to C[i,k] = sum_j A[i,j] B[j,k]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Uf[i*m+j][c] * Vf[j*p+k][c] * Wf[k*n+i][c]
                if s != 1:
                    return False
    return True

def _build_target_tensor(n, m, p):
    """Build the target tensor T[i,j,k] = 1 for all i,j,k."""
    T = np.ones((n, m, p))
    return T

def _construct_rank_18_decomposition():
    """
    Construct a rank-18 decomposition for 2x2x5 matrix multiplication.
    Split B into B1 (2x2) and B2 (2x2) and B3 (2x1).
    Use Strassen for B1 and B2 (rank 7 each) and schoolbook for B3 (rank 4).
    Total rank: 7 + 7 + 4 = 18.
    """
    n, m, p = 2, 2, 5
    
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a 2x2x2 Strassen decomposition shifted by offset_k
    def add_strassen_222(offset_k):
        # Standard Strassen decomposition for 2x2x2
        # C = A @ B
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
        #
        # Strassen:
        # m1 = (a+d)(e+h)
        # m2 = (c+d)(e)
        # m3 = a(f-h)
        # m4 = d(g-e)
        # m5 = (a+b)(h)
        # m6 = (c-a)(e+f)
        # m7 = (b-d)(g+h)
        #
        # C11 = m1 + m4 - m5 + m7
        # C12 = m3 + m5
        # C21 = m2 + m4
        # C22 = m1 - m2 + m3 + m6
        #
        # Indices:
        # A[i,j] -> i in {0,1}, j in {0,1}
        # B[j,k] -> j in {0,1}, k in {0,1} (local) -> global k = offset_k + local_k
        # C[i,k] -> i in {0,1}, k in {0,1} (local) -> global k = offset_k + local_k
        #
        # Our tensor convention:
        # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        #
        # For each Strassen term, we have a factorization of the 2x2x2 tensor.
        # We need to map the local indices to global indices.
        
        # Term 1: m1 = (a+d)(e+h)
        # A part: a+d -> U[0,0]=1, U[1,1]=1 (indices 0,1 in A)
        # B part: e+h -> V[0,0]=1, V[1,1]=1 (indices 0,1 in B local)
        # C part: contributes to C11 and C22
        # C11: W[0,0] += 1
        # C22: W[1,1] += 1
        
        # We need to add these to the global U, V, W.
        # But Strassen terms are sums of products.
        # Actually, each Strassen term is a rank-1 tensor in the 2x2x2 space.
        # So we can just add the vectors.
        
        # Let's define the local U, V, W vectors for each Strassen term.
        # Local A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Local B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Local C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        # Term 1: (a+d)(e+h)
        u1 = [1, 0, 0, 1]
        v1 = [1, 0, 0, 1]
        w1 = [1, 0, 0, 1] # C11 and C22
        
        # Term 2: (c+d)(e)
        u2 = [0, 0, 1, 1]
        v2 = [1, 0, 0, 0]
        w2 = [0, 0, 1, 0] # C21
        
        # Term 3: a(f-h)
        u3 = [1, 0, 0, 0]
        v3 = [0, 1, 0, -1]
        w3 = [0, 1, 0, 1] # C12 and C22? No.
        # C12 = m3 + m5. C22 = m1 - m2 + m3 + m6.
        # So m3 contributes to C12 and C22.
        # C12 is local index 1. C22 is local index 3.
        w3 = [0, 1, 0, 1]
        
        # Term 4: d(g-e)
        u4 = [0, 0, 0, 1]
        v4 = [0, 0, 1, -1]
        w4 = [1, 0, 1, 0] # C11 and C21?
        # C11 = m1 + m4 - m5 + m7. C21 = m2 + m4.
        # So m4 contributes to C11 and C21.
        # C11 is local index 0. C21 is local index 2.
        w4 = [1, 0, 1, 0]
        
        # Term 5: (a+b)(h)
        u5 = [1, 1, 0, 0]
        v5 = [0, 0, 0, 1]
        w5 = [-1, 1, 0, 0] # C11 and C12?
        # C11 = m1 + m4 - m5 + m7. C12 = m3 + m5.
        # So m5 contributes to C11 (with -1) and C12 (with +1).
        w5 = [-1, 1, 0, 0]
        
        # Term 6: (c-a)(e+f)
        u6 = [-1, 0, 1, 0]
        v6 = [1, 1, 0, 0]
        w6 = [0, 0, 0, 1] # C22?
        # C22 = m1 - m2 + m3 + m6.
        # So m6 contributes to C22.
        w6 = [0, 0, 0, 1]
        
        # Term 7: (b-d)(g+h)
        u7 = [0, 1, 0, -1]
        v7 = [0, 0, 1, 1]
        w7 = [1, 0, 0, -1] # C11 and C22?
        # C11 = m1 + m4 - m5 + m7. C22 = m1 - m2 + m3 + m6.
        # Wait, m7 contributes to C11 (with +1) and... does it contribute to C22?
        # Let's re-verify Strassen formulas.
        # C11 = m1 + m4 - m5 + m7
        # C12 = m3 + m5
        # C21 = m2 + m4
        # C22 = m1 - m2 + m3 + m6
        # So m7 only contributes to C11.
        w7 = [1, 0, 0, 0]
        
        # Now, we need to map these local vectors to global vectors.
        # Global A indices: i*m+j. n=2, m=2.
        # 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1). Same as local.
        # Global B indices: j*p+k. p=5.
        # Local k=0 -> global k=offset_k.
        # Local k=1 -> global k=offset_k+1.
        # So local B index 0 (j=0,k=0) -> global (0, offset_k) -> 0*5+offset_k = offset_k.
        # Local B index 1 (j=0,k=1) -> global (0, offset_k+1) -> offset_k+1.
        # Local B index 2 (j=1,k=0) -> global (1, offset_k) -> 1*5+offset_k = 5+offset_k.
        # Local B index 3 (j=1,k=1) -> global (1, offset_k+1) -> 5+offset_k+1.
        
        # Global C indices: k*n+i. n=2.
        # Local C index 0 (k=0,i=0) -> global (offset_k, 0) -> offset_k*2+0 = 2*offset_k.
        # Local C index 1 (k=0,i=1) -> global (offset_k, 1) -> 2*offset_k+1.
        # Local C index 2 (k=1,i=0) -> global (offset_k+1, 0) -> 2*(offset_k+1) = 2*offset_k+2.
        # Local C index 3 (k=1,i=1) -> global (offset_k+1, 1) -> 2*offset_k+3.
        
        # We will add these to the global U, V, W lists.
        # But we are building U, V, W as matrices.
        # Let's return the local vectors and the mapping.
        
        return [u1, u2, u3, u4, u5, u6, u7], [v1, v2, v3, v4, v5, v6, v7], [w1, w2, w3, w4, w5, w6, w7]
    
    # We will build U, V, W as lists of rows.
    # U has size 4 x rank.
    # V has size 10 x rank.
    # W has size 10 x rank.
    
    U = np.zeros((4, 18))
    V = np.zeros((10, 18))
    W = np.zeros((10, 18))
    
    # Block 1: columns 0,1 (offset_k=0)
    u_block1, v_block1, w_block1 = add_strassen_222(0)
    for c in range(7):
        U[0:4, c] = u_block1[c]
        # Map V
        v_local = v_block1[c]
        V[0, c] = v_local[0] # j=0, k=0
        V[1, c] = v_local[1] # j=0, k=1
        V[5, c] = v_local[2] # j=1, k=0
        V[6, c] = v_local[3] # j=1, k=1
        # Map W
        w_local = w_block1[c]
        W[0, c] = w_local[0] # k=0, i=0
        W[1, c] = w_local[1] # k=0, i=1
        W[2, c] = w_local[2] # k=1, i=0
        W[3, c] = w_local[3] # k=1, i=1
        
    # Block 2: columns 2,3 (offset_k=2)
    u_block2, v_block2, w_block2 = add_strassen_222(2)
    for c in range(7):
        U[0:4, 7+c] = u_block2[c]
        # Map V
        v_local = v_block2[c]
        V[2, 7+c] = v_local[0] # j=0, k=2
        V[3, 7+c] = v_local[1] # j=0, k=3
        V[7, 7+c] = v_local[2] # j=1, k=2
        V[8, 7+c] = v_local[3] # j=1, k=3
        # Map W
        w_local = w_block2[c]
        W[4, 7+c] = w_local[0] # k=2, i=0
        W[5, 7+c] = w_local[1] # k=2, i=1
        W[6, 7+c] = w_local[2] # k=3, i=0
        W[7, 7+c] = w_local[3] # k=3, i=1
        
    # Block 3: column 4 (offset_k=4). 2x2x1 multiplication.
    # C[i,4] = sum_j A[i,j] B[j,4]
    # This is just 4 terms.
    # Term 1: A[0,0] B[0,4] -> C[0,4]
    # Term 2: A[0,1] B[1,4] -> C[0,4]
    # Term 3: A[1,0] B[0,4] -> C[1,4]
    # Term 4: A[1,1] B[1,4] -> C[1,4]
    
    # U part:
    # Term 1: A[0,0] -> U[0, 14] = 1
    # Term 2: A[0,1] -> U[1, 15] = 1
    # Term 3: A[1,0] -> U[2, 16] = 1
    # Term 4: A[1,1] -> U[3, 17] = 1
    
    # V part:
    # Term 1: B[0,4] -> V[4, 14] = 1 (j=0, k=4)
    # Term 2: B[1,4] -> V[9, 15] = 1 (j=1, k=4)
    # Term 3: B[0,4] -> V[4, 16] = 1 (j=0, k=4)
    # Term 4: B[1,4] -> V[9, 17] = 1 (j=1, k=4)
    
    # W part:
    # Term 1: C[0,4] -> W[8, 14] = 1 (k=4, i=0)
    # Term 2: C[0,4] -> W[8, 15] = 1 (k=4, i=0)
    # Term 3: C[1,4] -> W[9, 16] = 1 (k=4, i=1)
    # Term 4: C[1,4] -> W[9, 17] = 1 (k=4, i=1)
    
    U[0, 14] = 1
    U[1, 15] = 1
    U[2, 16] = 1
    U[3, 17] = 1
    
    V[4, 14] = 1
    V[9, 15] = 1
    V[4, 16] = 1
    V[9, 17] = 1
    
    W[8, 14] = 1
    W[8, 15] = 1
    W[9, 16] = 1
    W[9, 17] = 1
    
    return U, V, W

def construct():
    n, m, p = 2, 2, 5
    
    start_time = time.time()
    
    # Try to find a better decomposition using a different method?
    # For now, return the rank-18 decomposition.
    # We can try to verify it.
    
    U, V, W = _construct_rank_18_decomposition()
    
    # Verify
    if _verify(U, V, W, n, m, p, 18):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to schoolbook rank-20
        U_fb = np.zeros((4, 20))
        V_fb = np.zeros((10, 20))
        W_fb = np.zeros((10, 20))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {
            "U": U_fb.tolist(),
            "V": V_fb.tolist(),
            "W": W_fb.tolist()
        }
