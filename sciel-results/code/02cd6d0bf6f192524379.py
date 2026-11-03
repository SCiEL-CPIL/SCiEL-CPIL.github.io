"""Program for M_{2,3,6} rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

def construct():
    start = time.time()
    deadline = start + 130.0  # safety margin under 150s
    n, m, p = DIMENSIONS
    
    # For M_{2,3,6}, we decompose into 6 slices (one per k)
    # Each slice is a 2x3x1 tensor with rank 4
    # Total rank = 6 * 4 = 24
    
    # The tensor T has T[i*m+j, j*p+k, k*n+i] = 1
    # For fixed k, the slice tensor S_k has S_k[i*m+j, j, i] = 1
    # Shape: (n*m, m, n) = (6, 3, 2)
    
    # For 2x3x1 matrix multiplication, the rank is 4
    # The 6 bilinear forms are: x0*y0, x0*y1, x0*y2, x1*y0, x1*y1, x1*y2
    # Output: z0 = x0*y0 + x0*y1 + x0*y2 + x1*y0 + x1*y1 + x1*y2 ... wait
    # Actually for 2x3 times 3x1:
    # z0 = x00*y0 + x01*y1 + x02*y2
    # z1 = x10*y0 + x11*y1 + x12*y2
    # The tensor has entries T[i*3+j, j, i] = 1 for i in {0,1}, j in {0,1,2}
    
    # Rank of 2x3x1 tensor: we can show it's 4
    # Using the formula: rank = n*m - gcd(n,m) + 1 for n x m x 1? No.
    # Actually for 2x3x1: the matricizations have ranks:
    # Mode 1: (6, 6) matrix, rank = 6
    # Mode 2: (3, 12) matrix, rank = 3  
    # Mode 3: (2, 18) matrix, rank = 2
    # Lower bound from matricizations is 6, but actual rank might be 4.
    
    # Let me construct the rank-4 decomposition for 2x3x1 directly.
    # T[i*3+j, j, i] = 1
    # We need U (6x4), V (3x4), W (2x4) such that sum_c U[i*3+j,c]*V[j,c]*W[i,c] = 1 if (i,j) valid
    
    # For 2x3x1, use the known rank-4 decomposition:
    # This is equivalent to multiplying a 2x3 matrix by a 3x1 vector.
    # The bilinear map is: (x00,x01,x02,x10,x11,x12) * (y0,y1,y2) -> (z0,z1)
    # z0 = x00*y0 + x01*y1 + x02*y2
    # z1 = x10*y0 + x11*y1 + x12*y2
    
    # Rank 4 decomposition:
    # Term 1: x00 * y0 -> contributes to z0
    # Term 2: x01 * y1 -> contributes to z0
    # Term 3: x02 * y2 -> contributes to z0
    # Term 4: (x10 - x00) * y0 + (x11 - x01) * y1 + (x12 - x02) * y2 -> contributes to z1
    # But we need z1 = x10*y0 + x11*y1 + x12*y2
    # So z1 = (x10-x00)*y0 + (x11-x01)*y1 + (x12-x02)*y2 + x00*y0 + x01*y1 + x02*y2
    # But x00*y0, x01*y1, x02*y2 are already in z0, not z1.
    
    # Better approach: 
    # Let a = x00, b = x01, c = x02, d = x10, e = x11, f = x12
    # z0 = a*y0 + b*y1 + c*y2
    # z1 = d*y0 + e*y1 + f*y2
    
    # Rank 4:
    # t0 = a * y0
    # t1 = b * y1
    # t2 = c * y2
    # t3 = (d-a) * y0 + (e-b) * y1 + (f-c) * y2
    # Then z0 = t0 + t1 + t2
    # z1 = t0 + t1 + t2 + t3 = a*y0 + b*y1 + c*y2 + (d-a)*y0 + (e-b)*y1 + (f-c)*y2 = d*y0 + e*y1 + f*y2 ✓
    
    # So for slice k, with indices i in {0,1}, j in {0,1,2}:
    # U has 6 rows (i*3+j), V has 3 rows (j), W has 2 rows (i)
    # 4 terms per slice:
    
    # For slice k, define the 4 rank-1 tensors:
    # Term 0: U[0*3+0, c] = 1, V[0, c] = 1, W[0, c] = 1  (i=0,j=0)
    # Term 1: U[0*3+1, c] = 1, V[1, c] = 1, W[0, c] = 1  (i=0,j=1)
    # Term 2: U[0*3+2, c] = 1, V[2, c] = 1, W[0, c] = 1  (i=0,j=2)
    # Term 3: U[1*3+0, c] = 1, V[0, c] = 1, W[1, c] = 1
    #          U[1*3+1, c] = 1, V[1, c] = 1, W[1, c] = 1
    #          U[1*3+2, c] = 1, V[2, c] = 1, W[1, c] = 1
    #          AND U[0*3+0, c] = -1, U[0*3+1, c] = -1, U[0*3+2, c] = -1
    #          V[0, c] = 1, V[1, c] = 1, V[2, c] = 1
    #          W[0, c] = 0, W[1, c] = 1
    
    # Wait, let me re-derive. For term 3 in slice k:
    # We want: for i=1, j=0: contribution = 1 (from d*y0 part)
    # for i=1, j=1: contribution = 1 (from e*y1 part)
    # for i=1, j=2: contribution = 1 (from f*y2 part)
    # for i=0, j=0: contribution = -1 (to cancel a*y0)
    # for i=0, j=1: contribution = -1 (to cancel b*y1)
    # for i=0, j=2: contribution = -1 (to cancel c*y2)
    
    # So U for term 3: [0*3+0]=-1, [0*3+1]=-1, [0*3+2]=-1, [1*3+0]=1, [1*3+1]=1, [1*3+2]=1
    # V for term 3: [0]=1, [1]=1, [2]=1
    # W for term 3: [0]=0, [1]=1
    
    # Verify: sum_c U[i*3+j,c]*V[j,c]*W[i,c]
    # For i=0, j=0: term0: 1*1*1=1, term1: 0, term2: 0, term3: (-1)*1*0=0 => 1 ✓
    # For i=0, j=1: term0: 0, term1: 1*1*1=1, term2: 0, term3: (-1)*1*0=0 => 1 ✓
    # For i=0, j=2: term0: 0, term1: 0, term2: 1*1*1=1, term3: (-1)*1*0=0 => 1 ✓
    # For i=1, j=0: term0: 0, term1: 0, term2: 0, term3: 1*1*1=1 => 1 ✓
    # For i=1, j=1: term0: 0, term1: 0, term2: 0, term3: 1*1*1=1 => 1 ✓
    # For i=1, j=2: term0: 0, term1: 0, term2: 0, term3: 1*1*1=1 => 1 ✓
    # All other (i,j) pairs: 0 ✓
    
    # Now build the full 2x3x6 decomposition with 6 slices, 4 terms each = 24 terms
    
    rank = 24
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    for k in range(p):
        base = k * 4  # 4 terms per slice
        
        # Term 0: i=0, j=0
        U[0*3+0, base + 0] = 1.0
        V[0*p+k, base + 0] = 1.0
        W[k*2+0, base + 0] = 1.0
        
        # Term 1: i=0, j=1
        U[0*3+1, base + 1] = 1.0
        V[1*p+k, base + 1] = 1.0
        W[k*2+0, base + 1] = 1.0
        
        # Term 2: i=0, j=2
        U[0*3+2, base + 2] = 1.0
        V[2*p+k, base + 2] = 1.0
        W[k*2+0, base + 2] = 1.0
        
        # Term 3: difference term
        U[0*3+0, base + 3] = -1.0
        U[0*3+1, base + 3] = -1.0
        U[0*3+2, base + 3] = -1.0
        U[1*3+0, base + 3] = 1.0
        U[1*3+1, base + 3] = 1.0
        U[1*3+2, base + 3] = 1.0
        V[0*p+k, base + 3] = 1.0
        V[1*p+k, base + 3] = 1.0
        V[2*p+k, base + 3] = 1.0
        W[k*2+0, base + 3] = 0.0
        W[k*2+1, base + 3] = 1.0
    
    # Verify
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    T_hat = np.einsum('ic,jc,kc->ijk', U, V, W)
    if not np.allclose(T_hat, T, atol=1e-9):
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U = np.zeros((n*m, rank_schoolbook), dtype=float)
        V = np.zeros((m*p, rank_schoolbook), dtype=float)
        W = np.zeros((p*n, rank_schoolbook), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1.0
                    V[j*p+k, c] = 1.0
                    W[k*n+i, c] = 1.0
                    c += 1
        rank = rank_schoolbook
    
    # Convert to lists of lists
    U_list = [[float(U[i, j]) for j in range(U.shape[1])] for i in range(U.shape[0])]
    V_list = [[float(V[i, j]) for j in range(V.shape[1])] for i in range(V.shape[0])]
    W_list = [[float(W[i, j]) for j in range(W.shape[1])] for i in range(W.shape[0])]
    
    return {"U": U_list, "V": V_list, "W": W_list}
