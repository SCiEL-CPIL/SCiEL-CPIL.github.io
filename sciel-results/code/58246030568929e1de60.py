"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8
    
    def naive_decomp():
        rank = n * m * p  # 16
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_einsum(U, V, W):
        """Verify the tensor decomposition is exact using einsum."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        R = U_arr.shape[1]
        
        # Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Using einsum: sum over r
        result = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
        
        # Check that result[i,j,k] = 1 for all i,j,k
        # But we need to map indices correctly
        # U index: i*m+j where i in [0,n), j in [0,m)
        # V index: j*p+k where j in [0,m), k in [0,p)
        # W index: k*n+i where k in [0,p), i in [0,n)
        
        # Let's build the expected tensor
        expected = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected[i,j,k] = 1.0
        
        # The einsum gives us T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
        # but U is indexed by (i*m+j), V by (j*p+k), W by (k*n+i)
        # So we need to reshape
        U_reshaped = U_arr.reshape(n, m, R)
        V_reshaped = V_arr.reshape(m, p, R)
        W_reshaped = W_arr.reshape(p, n, R)
        
        result2 = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        if not np.allclose(result2, expected, atol=1e-9):
            return False
        return True
    
    def verify(U, V, W):
        """Verify the tensor decomposition is exact."""
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def try_strassen_rectangular():
        """
        Rank-14 decomposition for 2x2 * 2x4 using two Strassen 2x2 blocks.
        """
        R = 14
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # Left block (cols 0,1 of B):
        # r=0: M1 = (a00+a11)(b00+b11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[5][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # r=1: M2 = (a10+a11)b00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        # r=2: M3 = a00(b01-b11)
        U[0][2] = 1
        V[1][2] = 1; V[5][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        # r=3: M4 = a11(b10-b00)
        U[3][3] = 1
        V[4][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        # r=4: M5 = (a00+a01)b11
        U[0][4] = 1; U[1][4] = 1
        V[5][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        # r=5: M6 = (a10-a00)(b00+b01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # r=6: M7 = (a01-a11)(b10+b11)
        U[1][6] = 1; U[3][6] = -1
        V[4][6] = 1; V[5][6] = 1
        W[0][6] = 1
        
        # Right block (cols 2,3 of B):
        # r=7: N1 = (a00+a11)(b02+b13)
        U[0][7] = 1; U[3][7] = 1
        V[2][7] = 1; V[7][7] = 1
        W[4][7] = 1; W[7][7] = 1
        
        # r=8: N2 = (a10+a11)b02
        U[2][8] = 1; U[3][8] = 1
        V[2][8] = 1
        W[5][8] = 1; W[7][8] = -1
        
        # r=9: N3 = a00(b03-b13)
        U[0][9] = 1
        V[3][9] = 1; V[7][9] = -1
        W[6][9] = 1; W[7][9] = 1
        
        # r=10: N4 = a11(b12-b02)
        U[3][10] = 1
        V[6][10] = 1; V[2][10] = -1
        W[4][10] = 1; W[5][10] = 1
        
        # r=11: N5 = (a00+a01)b13
        U[0][11] = 1; U[1][11] = 1
        V[7][11] = 1
        W[4][11] = -1; W[6][11] = 1
        
        # r=12: N6 = (a10-a00)(b02+b03)
        U[2][12] = 1; U[0][12] = -1
        V[2][12] = 1; V[3][12] = 1
        W[7][12] = 1
        
        # r=13: N7 = (a01-a11)(b12+b13)
        U[1][13] = 1; U[3][13] = -1
        V[6][13] = 1; V[7][13] = 1
        W[4][13] = 1
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        else:
            return None
    
    def try_rank13_shared():
        """
        Try to find a rank-13 decomposition by sharing one term between blocks.
        The idea: M1 and N1 both have (a00+a11) in U. We can try to combine
        (b00+b11) and (b02+b13) into a single rank-1 term if possible, but they
        are different. However, we might be able to share if we adjust.
        
        Actually, let's try a different approach: use the known lower bound.
        For 2x2 * 2x4, the rank is at least 13 (by Lysikov). Let's try to
        construct a rank-13 solution by modifying the Strassen decomposition.
        
        One approach: replace N1 = (a00+a11)(b02+b13) with a combination that
        reuses M1's U-vector. But M1 has V = (b00+b11), not (b02+b13).
        
        Alternative: Try to find if there's a rank-13 by solving a system.
        We'll use a greedy approach with random perturbations.
        """
        if time.time() > deadline:
            return None
        
        # Try a structured rank-13: share the (a00+a11) term
        # M1 = (a00+a11)(b00+b11) contributes to C[0][0], C[1][1]
        # N1 = (a00+a11)(b02+b13) contributes to C[0][2], C[1][3]
        # If we could have a single term (a00+a11)(b00+b11+b02+b13) that splits...
        # But that would give cross terms.
        
        # Let's try: Can we express the right block using only 6 terms if we
        # borrow one from the left? This is complex. Let's try a numerical search.
        
        # Start with rank-14 and try to zero out one column
        U14 = [[0]*14 for _ in range(4)]
        V14 = [[0]*14 for _ in range(8)]
        W14 = [[0]*14 for _ in range(8)]
        
        # Copy the rank-14 solution
        U14[0][0] = 1; U14[3][0] = 1
        V14[0][0] = 1; V14[5][0] = 1
        W14[0][0] = 1; W14[3][0] = 1
        
        U14[2][1] = 1; U14[3][1] = 1
        V14[0][1] = 1
        W14[1][1] = 1; W14[3][1] = -1
        
        U14[0][2] = 1
        V14[1][2] = 1; V14[5][2] = -1
        W14[2][2] = 1; W14[3][2] = 1
        
        U14[3][3] = 1
        V14[4][3] = 1; V14[0][3] = -1
        W14[0][3] = 1; W14[1][3] = 1
        
        U14[0][4] = 1; U14[1][4] = 1
        V14[5][4] = 1
        W14[0][4] = -1; W14[2][4] = 1
        
        U14[2][5] = 1; U14[0][5] = -1
        V14[0][5] = 1; V14[1][5] = 1
        W14[3][5] = 1
        
        U14[1][6] = 1; U14[3][6] = -1
        V14[4][6] = 1; V14[5][6] = 1
        W14[0][6] = 1
        
        U14[0][7] = 1; U14[3][7] = 1
        V14[2][7] = 1; V14[7][7] = 1
        W14[4][7] = 1; W14[7][7] = 1
        
        U14[2][8] = 1; U14[3][8] = 1
        V14[2][8] = 1
        W14[5][8] = 1; W14[7][8] = -1
        
        U14[0][9] = 1
        V14[3][9] = 1; V14[7][9] = -1
        W14[6][9] = 1; W14[7][9] = 1
        
        U14[3][10] = 1
        V14[6][10] = 1; V14[2][10] = -1
        W14[4][10] = 1; W14[5][10] = 1
        
        U14[0][11] = 1; U14[1][11] = 1
        V14[7][11] = 1
        W14[4][11] = -1; W14[6][11] = 1
        
        U14[2][12] = 1; U14[0][12] = -1
        V14[2][12] = 1; V14[3][12] = 1
        W14[7][12] = 1
        
        U14[1][13] = 1; U14[3][13] = -1
        V14[6][13] = 1; V14[7][13] = 1
        W14[4][13] = 1
        
        # Try to eliminate one column by absorbing it into others
        # This is a research problem; for now, just verify rank-14 works
        if verify(U14, V14, W14):
            # Return rank-14 as fallback within this function
            return None  # Let the caller use the rank-14
        
        return None
    
    # Try the structured rank-14 decomposition first
    result = try_strassen_rectangular()
    if result is not None:
        return result
    
    # Fallback to naive rank 16
    return naive_decomp()
# EVOLVE-BLOCK-END
