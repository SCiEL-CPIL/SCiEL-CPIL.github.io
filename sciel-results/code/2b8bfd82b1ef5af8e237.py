import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    def verify_decomposition(U, V, W):
        """Verify that the decomposition correctly computes matrix multiplication."""
        R = U.shape[1]
        n, m, p = 2, 4, 8
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if s != 1:
                        return False, (i, j, k, s)
        return True, None
    
    def build_strassen_56():
        """Build the standard 56-rank Strassen decomposition using 8 independent 2x2x2 blocks."""
        U_local = np.zeros((4, 7), dtype=int)
        V_local = np.zeros((4, 7), dtype=int)
        W_local = np.zeros((4, 7), dtype=int)
        
        # r=0 (M1)
        U_local[0,0] = 1; U_local[3,0] = 1
        V_local[0,0] = 1; V_local[3,0] = 1
        W_local[0,0] = 1; W_local[3,0] = 1
        
        # r=1 (M2)
        U_local[2,1] = 1; U_local[3,1] = 1
        V_local[0,1] = 1
        W_local[2,1] = 1; W_local[3,1] = -1
        
        # r=2 (M3)
        U_local[0,2] = 1
        V_local[1,2] = 1; V_local[3,2] = -1
        W_local[1,2] = 1; W_local[3,2] = 1
        
        # r=3 (M4)
        U_local[3,3] = 1
        V_local[2,3] = 1; V_local[0,3] = -1
        W_local[0,3] = 1; W_local[2,3] = 1
        
        # r=4 (M5)
        U_local[0,4] = 1; U_local[1,4] = 1
        V_local[3,4] = 1
        W_local[0,4] = -1; W_local[1,4] = 1
        
        # r=5 (M6)
        U_local[2,5] = 1; U_local[0,5] = -1
        V_local[0,5] = 1; V_local[1,5] = 1
        W_local[3,5] = 1
        
        # r=6 (M7)
        U_local[1,6] = 1; U_local[3,6] = -1
        V_local[2,6] = 1; V_local[3,6] = 1
        W_local[0,6] = 1
        
        total_rank = 56
        U = np.zeros((8, total_rank), dtype=int)
        V = np.zeros((32, total_rank), dtype=int)
        W = np.zeros((16, total_rank), dtype=int)
        
        for j in range(4):
            for s in range(2):
                base_r = (j * 2 + s) * 7
                
                a_map = [0, 1, 4, 5] if s == 0 else [2, 3, 6, 7]
                b_map = [0*8+2*j, 0*8+2*j+1, 1*8+2*j, 1*8+2*j+1] if s == 0 else \
                        [2*8+2*j, 2*8+2*j+1, 3*8+2*j, 3*8+2*j+1]
                c_map = [2*j*2+0, (2*j+1)*2+0, 2*j*2+1, (2*j+1)*2+1]
                
                for l in range(7):
                    r = base_r + l
                    for la in range(4):
                        val = U_local[la, l]
                        if val != 0:
                            U[a_map[la], r] += val
                    for lb in range(4):
                        val = V_local[lb, l]
                        if val != 0:
                            V[b_map[lb], r] += val
                    for lc in range(4):
                        val = W_local[lc, l]
                        if val != 0:
                            W[c_map[lc], r] += val
        
        return U, V, W
    
    def try_rank_52():
        """
        Try to find a rank-52 decomposition by finding a rank-13 scheme for 2x4x2.
        
        For 2x4x2: C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k]
        We need U: (8, R), V: (8, R), W: (4, R) such that
        sum_r U[i*4+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1 for all i,j,k
        
        Strategy: Start from the 14-term Strassen (two 2x2x2 blocks) and try to
        find a linear dependency among the 14 terms that allows rank reduction.
        Alternatively, do a direct search for rank-13.
        """
        if time.time() > deadline:
            return None
        
        # First, build the 14-term 2x4x2 Strassen decomposition
        # Block 1: j in {0,1}, standard 2x2x2 Strassen
        # Block 2: j in {2,3}, standard 2x2x2 Strassen
        
        U14 = np.zeros((8, 14), dtype=int)
        V14 = np.zeros((8, 14), dtype=int)
        W14 = np.zeros((4, 14), dtype=int)
        
        # Standard 2x2x2 Strassen for indices (a, b, c) where a,b,c in {0,1}
        # M1 = (a0+a1)(b0+b1) -> c0 += M1, c1 += M1
        # M2 = (a1)(b0)        -> c0 += M2, c1 -= M2  (wait, let me be careful)
        # Actually for C = A*B: C[0,0]=A[0,0]*B[0,0]+A[0,1]*B[1,0]
        # C[0,1]=A[0,0]*B[0,1]+A[0,1]*B[1,1]
        # C[1,0]=A[1,0]*B[0,0]+A[1,1]*B[1,0]
        # C[1,1]=A[1,0]*B[0,1]+A[1,1]*B[1,1]
        
        # Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B00)  -- wait, let me use the standard formulation
        # Actually standard Strassen for 2x2:
        # M1 = (A11-A00)(B00+B01)  -- this varies by convention
        # Let me just use the one that works:
        
        # Using the convention from the 2x2x2 block:
        # For A = [[a0,a1],[a2,a3]], B = [[b0,b1],[b2,b3]]
        # C[0,0] = a0*b0 + a1*b2
        # C[0,1] = a0*b1 + a1*b3
        # C[1,0] = a2*b0 + a3*b2
        # C[1,1] = a2*b1 + a3*b3
        
        # Strassen:
        # M1 = (a0+a3)(b0+b3)
        # M2 = (a2+a3)*b0
        # M3 = a0*(b2-b0)
        # M4 = a3*(b1-b0)  -- hmm, let me just verify
        # Actually, let me use a known correct 2x2x2 rank-7:
        
        # r=0: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1]  -> (a0+a3)(b0+b3) contributes to c00,c11
        # r=1: U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,-1] -> (a2+a3)*b0 contributes to c10, c11 with sign
        # r=2: U=[1,0,0,0], V=[0,1,0,-1], W=[0,1,0,1] -> a0*(b1-b3) contributes to c01, c11
        # r=3: U=[0,0,0,1], V=[0,0,1,-1], W=[1,0,1,0] -> a3*(b2-b0) contributes to c00, c10
        # r=4: U=[1,1,0,0], V=[0,0,0,1], W=[-1,1,0,0] -> (a0+a1)*b3 contributes to c00, c01
        # r=5: U=[-1,0,1,0], V=[1,1,0,0], W=[0,0,0,1] -> (a2-a0)*(b0+b1) contributes to c11
        # r=6: U=[0,1,0,-1], V=[0,0,1,1], W=[1,0,0,0] -> (a1-a3)*(b2+b3) contributes to c00
        
        # Let me verify this is correct by checking each output:
        # c00: r0:1*1*1 + r3:1*1*1 + r4:1*1*(-1) + r5:0 + r6:1*1*1 = 1+1-1+0+1 = 2? That's wrong.
        
        # Let me just use a verified 2x2x2 Strassen. I'll construct it carefully.
        
        # For 2x2x2, the tensor is T[a,b,c] = 1 for all a,b,c in {0,1}
        # We need sum_r U[a,r]*V[b,r]*W[c,r] = 1 for all a,b,c
        
        # Known rank-7 decomposition (one standard form):
        # r=0: U=[1,1], V=[1,1], W=[1,1]  -> (a0+a1)(b0+b1) -> c0+=1, c1+=1... 
        # This doesn't directly work. Let me use a different approach.
        
        # I'll just verify the 56-rank solution and return it if valid.
        # The search for rank 52 is complex and may not succeed in the time budget.
        
        # Alternative approach: try to find if the 14-term 2x4x2 has a dependency.
        # Build the 14-term solution explicitly.
        
        # For block j={0,1}:
        # A indices: 0,1 (for i=0) and 4,5 (for i=1) -> A[i,j] where i in {0,1}, j in {0,1}
        # B indices: 0,1 (for k=0) and 2,3 (for k=1) -> B[j,k] where j in {0,1}, k in {0,1}
        # C indices: 0,1 (for i=0,k=0 and i=0,k=1) and 2,3 (for i=1,k=0 and i=1,k=1)
        # C[k*n+i] = C[k*2+i]
        
        # So for block 0 (j=0,1):
        # A: [A00, A01, A10, A11] = [U[0], U[1], U[4], U[5]]
        # B: [B00, B01, B10, B11] = [V[0], V[1], V[2], V[3]]  (B[j,k]: j=0->k=0,1; j=1->k=0,1)
        #    B[0,0]=V[0], B[0,1]=V[1], B[1,0]=V[2], B[1,1]=V[3]
        # C: [C00, C01, C10, C11] = [W[0], W[1], W[2], W[3]]
        #    C[k*2+i]: C[0*2+0]=C00=W[0], C[0*2+1]=C01=W[1], C[1*2+0]=C10=W[2], C[1*2+1]=C11=W[3]
        
        # For block 1 (j=2,3):
        # A: [A00', A01', A10', A11'] = [U[2], U[3], U[6], U[7]]  (A[i,j] with j=2,3)
        # B: [B00', B01', B10', B11'] = [V[4], V[5], V[6], V[7]]
        # C: [C00', C01', C10', C11'] = [W[4], W[5], W[6], W[7]]
        
        # Use standard 2x2x2 Strassen for each block.
        # I'll use a verified rank-7 2x2x2 decomposition.
        
        # Let me define the 2x2x2 Strassen properly:
        # For matrices A=[[a0,a1],[a2,a3]], B=[[b0,b1],[b2,b3]], C=A*B
        # C00 = a0*b0 + a1*b2
        # C01 = a0*b1 + a1*b3
        # C10 = a2*b0 + a3*b2
        # C11 = a2*b1 + a3*b3
        
        # Strassen decomposition (one valid form):
        # M1 = (a0+a3)(b0+b3)
        # M2 = (a2+a3)*b0
        # M3 = a0*(b2-b0)
        # M4 = a3*(b1-b0)
        # M5 = (a0+a1)*b3
        # M6 = (a2-a0)*(b0+b1)
        # M7 = (a1-a3)*(b2+b3)
        
        # Check C00 = M1 - M4 - M5 + M6 + M7? Let me compute:
        # M1 = a0*b0 + a0*b3 + a3*b0 + a3*b3
        # M2 = a2*b0 + a3*b0
        # M3 = a0*b2 - a0*b0
        # M4 = a3*b1 - a3*b0
        # M5 = a0*b3 + a1*b3
        # M6 = a2*b0 + a2*b1 - a0*b0 - a0*b1
        # M7 = a1*b2 + a1*b3 - a3*b2 - a3*b3
        
        # C00 = a0*b0 + a1*b2
        # Let's try: M1 - M4 - M5 + M6 + M7
        # = (a0*b0 + a0*b3 + a3*b0 + a3*b3) - (a3*b1 - a3*b0) - (a0*b3 + a1*b3) + (a2*b0 + a2*b1 - a0*b0 - a0*b1) + (a1*b2 + a1*b3 - a3*b2 - a3*b3)
        # = a0*b0 + a0*b3 + a3*b0 + a3*b3 - a3*b1 + a3*b0 - a0*b3 - a1*b3 + a2*b0 + a2*b1 - a0*b0 - a0*b1 + a1*b2 + a1*b3 - a3*b2 - a3*b3
        # = (a0*b0 - a0*b0) + (a0*b3 - a0*b3) + (a3*b3 - a3*b3) + a3*b0 + a3*b0 - a3*b1 + a2*b0 + a2*b1 - a0*b1 - a1*b3 + a1*b3 + a1*b2 - a3*b2
        # = 2*a3*b0 - a3*b1 + a2*b0 + a2*b1 - a0*b1 + a1*b2 - a3*b2
        # This doesn't simplify to a0*b0 + a1*b2. So this Strassen form is wrong.
        
        # Let me just use a known correct one. I'll look up the standard Strassen.
        # Standard Strassen:
        # M1 = (A11 - A00)(B00 + B01)
        # M2 = (A00 + A11)B11
        # M3 = (A01 - A11)(B10 + B11)
        # M4 = A00(B01 - B11)
        # M5 = A11(B00 - B01)
        # M6 = (A00 + A01)B10
        # M7 = (A10 - A00)(B00 + B10)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M2 + M3
        # C10 = M6 + M5
        # C11 = M2 - M1 + M3 + M6
        
        # Let me verify C00:
        # M1 = (A11-A00)(B00+B01) = A11*B00 + A11*B01 - A00*B00 - A00*B01
        # M4 = A00*(B01-B11) = A00*B01 - A00*B11
        # M5 = A11*(B00-B01) = A11*B00 - A11*B01
        # M7 = (A10-A00)(B00+B10) = A10*B00 + A10*B10 - A00*B00 - A00*B10
        
        # M1 + M4 - M5 + M7 = A11*B00 + A11*B01 - A00*B00 - A00*B01 + A00*B01 - A00*B11 - A11*B00 + A11*B01 + A10*B00 + A10*B10 - A00*B00 - A00*B10
        # = A11*B01 + A11*B01 - A00*B00 - A00*B11 + A10*B00 + A10*B10 - A00*B10
        # = 2*A11*B01 - 2*A00*B00 - A00*B11 + A10*B00 + A10*B10 - A00*B10
        
        # This should equal A00*B00 + A01*B10. That's clearly not matching. I'm confusing myself.
        
        # Let me just use a simple approach: I'll verify the existing 56-rank solution and return it.
        # The search for lower rank is not making progress.
        
        return None
    
    U56, V56, W56 = build_strassen_56()
    valid, err = verify_decomposition(U56, V56, W56)
    
    if not valid:
        # Fallback to naive 64-rank if Strassen is broken
        U = np.zeros((8, 64), dtype=int)
        V = np.zeros((32, 64), dtype=int)
        W = np.zeros((16, 64), dtype=int)
        for i in range(2):
            for j in range(4):
                for k in range(8):
                    r = i*32 + j*8 + k
                    U[i*4+j, r] = 1
                    V[j*8+k, r] = 1
                    W[k*2+i, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    result_52 = try_rank_52()
    if result_52 is not None:
        return result_52
    
    # Return the verified 56-rank solution
    return {"U": U56.tolist(), "V": V56.tolist(), "W": W56.tolist()}
# EVOLVE-BLOCK-END
