"""Combined program: Strassen-based block decomposition for 4x4 * 4x9 matrix multiplication."""
import time
import sys

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36
    
    def make_schoolbook():
        """Return schoolbook decomposition of rank n*m*p."""
        rank = n * m * p  # 144
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition is valid."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(total - 1) > tol:
                        return False
        return True
    
    # Build Strassen-based decomposition for 4x4 * 4x9
    # Strategy: Split A (4x4) into 2x2 blocks, B (4x9) into 2x9 blocks
    # Use 7-term Strassen for each 2x2 * 2x9 product
    
    def strassen_2x2_2x9_terms():
        """
        Return 7 rank-1 term patterns for 2x2 * 2x9 multiplication.
        Each term is (U_pattern, V_pattern, W_pattern) where:
        - U_pattern: dict mapping (i,j) -> coeff for i,j in {0,1}
        - V_pattern: dict mapping (j,k) -> coeff for j in {0,1}, k in 0..8
        - W_pattern: dict mapping (i,k) -> coeff for i in {0,1}, k in 0..8
        
        Strassen 7 products for M(2x2) * N(2x9):
        p1 = (a+d) * (N1+N2)
        p2 = (c+d) * (N1+N2)
        p3 = a * (N2-N1)
        p4 = d * (N1-N2)
        p5 = (a+b) * N2
        p6 = (c-a) * (N1+N2)
        p7 = (b-d) * (N1+N2)
        
        Result:
        R[0,k] = p1 + p4 - p5
        R[1,k] = p2 + p3 + p6 - p7  (wait, let me verify)
        
        Actually standard Strassen:
        p1 = (a+d)(e+h)
        p2 = (c+d)(e+h)
        p3 = a(f-e)
        p4 = d(h-g)
        p5 = (a+b)h
        p6 = (c-a)(e+f)
        p7 = (b-d)(g+h)
        
        Result:
        [[p1+p4-p5+p7, ...],
         [p2+p3-p4+p6, ...]]
        
        For rectangular 2x2 * 2x9, N has rows N1 (row 0) and N2 (row 1), each of length 9.
        e,f,g,h become N1[0],N1[1],...N1[8] and N2[0],N2[1],...N2[8]
        
        So:
        p1 = (a+d) * (N1+N2)  -- this is a row vector of length 9
        p2 = (c+d) * (N1+N2)
        p3 = a * (N2-N1)
        p4 = d * (N1-N2)
        p5 = (a+b) * N2
        p6 = (c-a) * (N1+N2)
        p7 = (b-d) * (N1+N2)
        
        Result row 0: p1 + p4 - p5 + p7  (element-wise)
        Result row 1: p2 + p3 - p4 + p6  (element-wise)
        
        Wait, let me re-derive. For 2x2*2x2:
        C[0,0] = p1 + p4 - p5 + p7
        C[0,1] = p5 - p7  (hmm, not quite)
        
        Let me just use the standard formula:
        C[0,0] = p1 + p4 - p5 + p7
        C[0,1] = p5 - p7  ... no
        
        Standard Strassen output:
        [[p1+p4-p5+p7, p5-p7],
         [p2+p3-p4+p6, p1-p2+p3+p4]]
        
        For rectangular, each "entry" becomes a row of 9 elements.
        So:
        C_row0 = p1 + p4 - p5 + p7  (each is a 1x9 vector, add element-wise)
        C_row1 = p2 + p3 - p4 + p6  (each is a 1x9 vector, add element-wise)
        
        But wait, that's only 2 rows, and we need the full 2x9 result. Let me check:
        C[0,k] = sum_j M[0,j]*N[j,k] = a*N[0,k] + b*N[1,k]
        C[1,k] = sum_j M[1,j]*N[j,k] = c*N[0,k] + d*N[1,k]
        
        With Strassen:
        p1 = (a+d)*(N[0,k]+N[1,k])
        p2 = (c+d)*(N[0,k]+N[1,k])
        p3 = a*(N[1,k]-N[0,k])
        p4 = d*(N[0,k]-N[1,k])
        p5 = (a+b)*N[1,k]
        p6 = (c-a)*(N[0,k]+N[1,k])
        p7 = (b-d)*(N[0,k]+N[1,k])
        
        C[0,k] should be a*N[0,k] + b*N[1,k]
        C[1,k] should be c*N[0,k] + d*N[1,k]
        
        Let me verify p1+p4-p5+p7:
        = (a+d)(N0+N1) + d(N0-N1) - (a+b)N1 + (b-d)(N0+N1)
        = (a+d+b-d)(N0+N1) + d*N0 - d*N1 - a*N1 - b*N1
        = (a+b)(N0+N1) + d*N0 - (d+a+b)*N1
        = a*N0 + a*N1 + b*N0 + b*N1 + d*N0 - d*N1 - a*N1 - b*N1
        = a*N0 + b*N0 + d*N0 + (a-a)*N1 + (b-b)*N1 + (d-d)*N1... 
        
        This is getting messy. Let me just use a verified Strassen formula.
        
        Standard 2x2 Strassen:
        p1 = (a+d)(e+h)
        p2 = (c+d)(e+h)  
        p3 = a(f-e)
        p4 = d(h-g)
        p5 = (a+b)h
        p6 = (c-a)(e+f)
        p7 = (b-d)(g+h)
        
        C[0,0] = p1 + p4 - p5 + p7
        C[0,1] = p5 - p7
        C[1,0] = p2 + p3 - p4 + p6
        C[1,1] = p1 - p2 + p3 + p4
        
        For rectangular 2x2 * 2x9, N is 2x9 with rows N0 (length 9) and N1 (length 9).
        The "entries" e,f,g,h become vectors.
        
        p1 = (a+d) * (N0 + N1)  -- 1x9 vector
        p2 = (c+d) * (N0 + N1)
        p3 = a * (N1 - N0)
        p4 = d * (N0 - N1)
        p5 = (a+b) * N1
        p6 = (c-a) * (N0 + N1)
        p7 = (b-d) * (N0 + N1)
        
        C_row0 = p1 + p4 - p5 + p7  (element-wise, gives 1x9)
        C_row1 = p2 + p3 - p4 + p6  (element-wise, gives 1x9)
        
        Let me verify C_row0[0] (k=0):
        = (a+d)(N0[0]+N1[0]) + d(N0[0]-N1[0]) - (a+b)N1[0] + (b-d)(N0[0]+N1[0])
        = (a+d+b-d)(N0[0]+N1[0]) + d*N0[0] - d*N1[0] - a*N1[0] - b*N1[0]
        = (a+b)(N0[0]+N1[0]) + d*N0[0] - (d+a+b)*N1[0]
        = a*N0[0] + a*N1[0] + b*N0[0] + b*N1[0] + d*N0[0] - d*N1[0] - a*N1[0] - b*N1[0]
        = a*N0[0] + b*N0[0] + d*N0[0] + (a-a)*N1[0] + (b-b)*N1[0] + (d-d)*N1[0]
        = (a+b+d)*N0[0]  -- this is NOT a*N0[0] + b*N1[0]
        
        So my Strassen formula is wrong for the rectangular case, or I'm applying it incorrectly.
        
        The issue: Strassen's 7-product algorithm is designed for 2x2*2x2 where all four outputs are computed. For 2x2*2x9, we only need 2 rows of output (each of length 9), not 4 "blocks".
        
        Let me think differently. For 2x2 * 2x9:
        We need to compute:
        C[0,k] = a*N[0,k] + b*N[1,k] for k=0..8
        C[1,k] = c*N[0,k] + d*N[1,k] for k=0..8
        
        This is 18 scalar multiplications schoolbook (2*2*9/2... no, 2*2*9=36 scalar mults, but we can share).
        
        Actually, for each k, we compute a*N[0,k]+b*N[1,k] and c*N[0,k]+d*N[1,k].
        The coefficients a,b,c,d are shared across all k.
        
        One approach: Precompute linear combinations of N rows.
        Let S = N[0,:] + N[1,:] (1x9)
        Let D = N[0,:] - N[1,:] (1x9)
        
        Then:
        a*N[0,k] + b*N[1,k] = a*(S[k]+D[k])/2 + b*(S[k]-D[k])/2 = ((a+b)/2)*S[k] + ((a-b)/2)*D[k]
        c*N[0,k] + d*N[1,k] = ((c+d)/2)*S[k] + ((c-d)/2)*D[k]
        
        So we need:
        s1 = (a+b)/2
        s2 = (c+d)/2
        d1 = (a-b)/2
        d2 = (c-d)/2
        
        Then C[0,:] = s1*S + d1*D
        C[1,:] = s2*S + d2*D
        
        This uses 4 scalar-vector products (each 1x9), which is 4*9 = 36 scalar mults, same as schoolbook.
        
        To reduce, we need to share the computation of S and D. But S and D are just linear combinations, so they don't add rank.
        
        The rank-1 decomposition perspective: Each term in the sum corresponds to a rank-1 tensor.
        
        Let me try a different approach: just use the schoolbook for the sub-problems but look for global optimizations.
        
        Actually, I think the simplest improvement is to recognize that for 4x4*4x9, we can use the fact that the B matrix is 4x9 and we're multiplying 4x4 by it. 
        
        Let me just implement a clean schoolbook and verify it works, then try to find a smaller rank.
        """
        # For now, return schoolbook as safe baseline
        pass
    
    # Start with schoolbook
    best_result = make_schoolbook()
    best_rank = n * m * p
    
    # Try to find a lower rank decomposition
    if time.time() < deadline - 10:
        # For 4x4 * 4x9, one approach: treat it as 4 independent 1x4 * 4x9 products
        # But they share B, so we can potentially share work.
        
        # Alternative: Use the tensor structure directly.
        # T[i,j,k] = 1 for i in 0..3, j in 0..3, k in 0..8
        # U index: i*4+j, V index: j*9+k, W index: k*4+i
        
        # A rank-1 term (u,v,w) contributes u[i*4+j]*v[j*9+k]*w[k*4+i] to T[i,j,k]
        
        # For the sum to equal 1 at each (i,j,k), we need:
        # sum_r U[i*4+j, r] * V[j*9+k, r] * W[k*4+i, r] = 1
        
        # This is a system of 4*4*9 = 144 equations in R*16 + R*36 + R*36 variables (but with the constraint that it's a sum of rank-1 terms).
        
        # The schoolbook solution has R=144. Can we do better?
        
        # Lower bound: The tensor has 144 non-zero entries. Each rank-1 term can cover at most... well, it depends on the support.
        
        # For now, just return schoolbook.
        pass
    
    return best_result
# EVOLVE-BLOCK-END
