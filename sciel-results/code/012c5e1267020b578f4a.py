"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5
    deadline = time.time() + 145  # Safety margin before 150s kill
    
    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        target = 1.0 if j == jp else 0.0
                        val = np.dot(U[i*m + j, :], V[jp*p + k, :] * W[k*n + i, :])
                        if abs(val - target) > tol:
                            return False
        return True
    
    # Try to find a rank-17 solution by sharing one term between Strassen blocks
    # Strategy: Use Strassen for cols 0-1 (7 terms), then a modified 2x3 block for cols 2-4 (10 terms)
    # Total: 7 + 10 = 17
    
    def try_rank_17():
        """Attempt rank-17 decomposition: 7 (Strassen) + 10 (2x3 block)"""
        rank = 17
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Strassen block for cols 0,1 (terms 0-6)
        # M1 = (a+d)(e+g)
        r = 0
        U[0, r] = 1; U[3, r] = 1
        V[0, r] = 1; V[p + 0, r] = 1
        W[0 * n, r] = 1; W[1 * n + 1, r] = 1
        
        # M2 = (c+d)e
        r = 1
        U[2, r] = 1; U[3, r] = 1
        V[0, r] = 1
        W[0 * n + 1, r] = 1; W[1 * n + 1, r] = -1
        
        # M3 = a(f-g)
        r = 2
        U[0, r] = 1
        V[1, r] = 1; V[p + 1, r] = -1
        W[1 * n, r] = 1; W[1 * n + 1, r] = 1
        
        # M4 = d(h-f)
        r = 3
        U[3, r] = 1
        V[p + 0, r] = 1; V[0, r] = -1
        W[0 * n, r] = 1; W[0 * n + 1, r] = 1
        
        # M5 = (a+b)h
        r = 4
        U[0, r] = 1; U[1, r] = 1
        V[p + 1, r] = 1
        W[0 * n, r] = -1; W[1 * n, r] = 1
        
        # M6 = (c-a)(e+f)
        r = 5
        U[2, r] = 1; U[0, r] = -1
        V[0, r] = 1; V[1, r] = 1
        W[1 * n + 1, r] = 1
        
        # M7 = (b-d)(g+h)
        r = 6
        U[1, r] = 1; U[3, r] = -1
        V[p + 0, r] = 1; V[p + 1, r] = 1
        W[0 * n, r] = 1
        
        # Now for cols 2,3,4: we need to compute C[:,2:5] = A @ B[:,2:5]
        # This is a 2x2 times 2x3 multiplication
        # We'll use a 10-term decomposition for this block
        # Let's denote B2 = B[:,2:5] with entries:
        # B[0,2]=e2, B[0,3]=f2, B[0,4]=g2
        # B[1,2]=h2, B[1,3]=i2, B[1,4]=j2
        
        # We can use Strassen-like formulas adapted for 2x3
        # Terms 7-16 (10 terms)
        
        # T1 = (a+d)(e2+h2)
        r = 7
        U[0, r] = 1; U[3, r] = 1
        V[2, r] = 1; V[p + 2, r] = 1
        W[2 * n, r] = 1; W[2 * n + 1, r] = 1
        
        # T2 = (c+d)e2
        r = 8
        U[2, r] = 1; U[3, r] = 1
        V[2, r] = 1
        W[2 * n + 1, r] = 1
        
        # T3 = a(f2-h2)
        r = 9
        U[0, r] = 1
        V[3, r] = 1; V[p + 2, r] = -1
        W[3 * n, r] = 1; W[3 * n + 1, r] = 1
        
        # T4 = d(i2-f2)
        r = 10
        U[3, r] = 1
        V[p + 3, r] = 1; V[3, r] = -1
        W[3 * n, r] = 1; W[3 * n + 1, r] = 1
        
        # T5 = (a+b)i2
        r = 11
        U[0, r] = 1; U[1, r] = 1
        V[p + 3, r] = 1
        W[3 * n, r] = 1
        
        # T6 = (c-a)(e2+f2)
        r = 12
        U[2, r] = 1; U[0, r] = -1
        V[2, r] = 1; V[3, r] = 1
        W[3 * n + 1, r] = 1
        
        # T7 = (b-d)(h2+i2)
        r = 13
        U[1, r] = 1; U[3, r] = -1
        V[p + 2, r] = 1; V[p + 3, r] = 1
        W[2 * n, r] = 1
        
        # For col 4 (g2, j2):
        # C[0,4] = a*g2 + b*j2
        # C[1,4] = c*g2 + d*j2
        
        # T8 = a*g2
        r = 14
        U[0, r] = 1
        V[4, r] = 1
        W[4 * n, r] = 1
        
        # T9 = b*j2
        r = 15
        U[1, r] = 1
        V[p + 4, r] = 1
        W[4 * n, r] = 1
        
        # T10 = c*g2
        r = 16
        U[2, r] = 1
        V[4, r] = 1
        W[4 * n + 1, r] = 1
        
        # We need d*j2 for C[1,4], but we only have 10 terms (7-16)
        # This means we need 11 terms for the 2x3 block, giving total 18
        # Let's return None to indicate this approach doesn't work with 17
        
        # Actually, we can combine: C[1,4] = c*g2 + d*j2
        # If we make T10 = (c+d)*g2 and add another term for d*j2 - d*g2... no, that's getting complex
        
        # Let's try a different approach: use 9 terms for cols 2,3 and 3 for col 4
        # But that's still 7+9+3=19 or 7+10+1=18
        
        # For now, return None
        return None
    
    # Try rank 18 (known to work)
    def try_rank_18():
        """Rank-18 decomposition: two Strassen blocks + schoolbook for last column"""
        rank = 18
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        def add_strassen_block(col_offset, base_rank):
            r = base_rank
            U[0, r] = 1; U[3, r] = 1
            V[col_offset, r] = 1; V[p + col_offset, r] = 1
            W[col_offset * n, r] = 1; W[(col_offset + 1) * n + 1, r] = 1
            
            r = base_rank + 1
            U[2, r] = 1; U[3, r] = 1
            V[col_offset, r] = 1
            W[col_offset * n + 1, r] = 1; W[(col_offset + 1) * n + 1, r] = -1
            
            r = base_rank + 2
            U[0, r] = 1
            V[col_offset + 1, r] = 1; V[p + col_offset + 1, r] = -1
            W[(col_offset + 1) * n, r] = 1; W[(col_offset + 1) * n + 1, r] = 1
            
            r = base_rank + 3
            U[3, r] = 1
            V[p + col_offset, r] = 1; V[col_offset, r] = -1
            W[col_offset * n, r] = 1; W[col_offset * n + 1, r] = 1
            
            r = base_rank + 4
            U[0, r] = 1; U[1, r] = 1
            V[p + col_offset + 1, r] = 1
            W[col_offset * n, r] = -1; W[(col_offset + 1) * n, r] = 1
            
            r = base_rank + 5
            U[2, r] = 1; U[0, r] = -1
            V[col_offset, r] = 1; V[col_offset + 1, r] = 1
            W[(col_offset + 1) * n + 1, r] = 1
            
            r = base_rank + 6
            U[1, r] = 1; U[3, r] = -1
            V[p + col_offset, r] = 1; V[p + col_offset + 1, r] = 1
            W[col_offset * n, r] = 1
        
        add_strassen_block(0, 0)
        add_strassen_block(2, 7)
        
        # Schoolbook for col 4
        r = 14
        U[0, r] = 1
        V[4, r] = 1
        W[8, r] = 1
        
        r = 15
        U[1, r] = 1
        V[9, r] = 1
        W[8, r] = 1
        
        r = 16
        U[2, r] = 1
        V[4, r] = 1
        W[9, r] = 1
        
        r = 17
        U[3, r] = 1
        V[9, r] = 1
        W[9, r] = 1
        
        if verify(U, V, W):
            return U, V, W
        return None
    
    # Try rank 17 first
    result_17 = try_rank_17()
    if result_17 is not None:
        U, V, W = result_17
        if verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to rank 18
    result_18 = try_rank_18()
    if result_18 is not None:
        U, V, W = result_18
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Final fallback: schoolbook rank 20
    rank = 20
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
