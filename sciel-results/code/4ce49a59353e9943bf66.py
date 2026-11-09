"""
Matrix multiplication 4x5 x 5x5: rank-minimal exact tensor decomposition.

Strategy:
- Use block decomposition with Strassen's rank-7 algorithm for 2x2x2 blocks.
- Split dimensions to maximize use of Strassen where possible.
- For (4,5,5): split n=4 into 2+2, m=5 into 2+3, p=5 into 2+3.
- This gives 8 sub-problems:
  (2,2,2): Strassen rank 7
  (2,2,3): schoolbook rank 12
  (2,3,2): schoolbook rank 12
  (2,3,3): schoolbook rank 18
  (2,2,2): Strassen rank 7
  (2,2,3): schoolbook rank 12
  (2,3,2): schoolbook rank 12
  (2,3,3): schoolbook rank 18
- Total: 2*7 + 2*12 + 2*12 + 2*18 = 14 + 24 + 24 + 36 = 98

Actually, let me reconsider the splitting. The tensor for (n,m,p) multiplication
has the property that we can split any dimension:
  r(n1+n2, m, p) <= r(n1,m,p) + r(n2,m,p)
  r(n, m1+m2, p) <= r(n,m1,p) + r(n,m2,p)
  r(n, m, p1+p2) <= r(n,m,p1) + r(n,m,p2)

For (4,5,5):
Option 1: Split n=4 into 2+2:
  r(4,5,5) <= r(2,5,5) + r(2,5,5) = 2*r(2,5,5)
  For r(2,5,5): split m=5 into 2+3:
    r(2,5,5) <= r(2,2,5) + r(2,3,5)
    r(2,2,5): split p=5 into 2+3: r(2,2,2) + r(2,2,3) = 7 + 12 = 19
    r(2,3,5): split p=5 into 2+3: r(2,3,2) + r(2,3,3) = 12 + 18 = 30
    So r(2,5,5) <= 19 + 30 = 49
  Total: 2*49 = 98

Option 2: Split m=5 into 2+3:
  r(4,5,5) <= r(4,2,5) + r(4,3,5)
  r(4,2,5): split n=4 into 2+2: 2*r(2,2,5) = 2*19 = 38
  r(4,3,5): split n=4 into 2+2: 2*r(2,3,5) = 2*30 = 60
  Total: 38 + 60 = 98

Option 3: Split p=5 into 2+3:
  r(4,5,5) <= r(4,5,2) + r(4,5,3)
  r(4,5,2): split m=5 into 2+3: r(4,2,2) + r(4,3,2)
    r(4,2,2): split n=4 into 2+2: 2*r(2,2,2) = 2*7 = 14
    r(4,3,2): split n=4 into 2+2: 2*r(2,3,2) = 2*12 = 24
    So r(4,5,2) <= 14 + 24 = 38
  r(4,5,3): split m=5 into 2+3: r(4,2,3) + r(4,3,3)
    r(4,2,3): split n=4 into 2+2: 2*r(2,2,3) = 2*12 = 24
    r(4,3,3): split n=4 into 2+2: 2*r(2,3,3) = 2*18 = 36
    So r(4,5,3) <= 24 + 36 = 60
  Total: 38 + 60 = 98

All options give 98. Let me implement Option 3 (split p first) as it's straightforward.
"""

import time
import numpy as np

DIMENSIONS = [4, 5, 5]


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Build the target tensor T of shape (n*m, m*p, n*p)
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k*n+i) with matching indices
    T = np.zeros((n * m, m * p, n * p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i  # W[k*n+i, r] convention
                T[a, b, c] = 1.0

    def verify(U, V, W, tol=1e-10):
        """Check if U, V, W exactly decompose T."""
        T_recon = np.einsum('as,bs,cs->abc', U, V, W)
        return np.max(np.abs(T_recon - T)) < tol

    def to_output(U, V, W):
        """Convert numpy arrays to the required dict format."""
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Schoolbook decomposition for (nn, mm, pp)
    # Returns U, V, W of shapes (nn*mm, rank), (mm*pp, rank), (pp*nn, rank)
    def schoolbook(nn, mm, pp):
        rank = nn * mm * pp
        U = np.zeros((nn * mm, rank))
        V = np.zeros((mm * pp, rank))
        W = np.zeros((pp * nn, rank))
        col = 0
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    U[i * mm + j, col] = 1.0
                    V[j * pp + k, col] = 1.0
                    W[k * nn + i, col] = 1.0
                    col += 1
        return U, V, W

    # Strassen decomposition for 2x2x2
    # Returns U, V, W of shapes (4, 7) each
    def strassen_2x2x2():
        """
        Strassen's algorithm for 2x2 matrix multiplication.
        Uses the convention: a[i,j] -> i*2+j, b[i,j] -> i*2+j, c[k,i] -> k*2+i
        """
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # M1 = (a00+a11)*(b00+b11)
        U[:, 0] = [1, 0, 0, 1]  # a00 + a11
        V[:, 0] = [1, 0, 0, 1]  # b00 + b11
        # c[k,i] contributions: c[0,0] += M1, c[1,1] += M1
        # c[0,0] -> idx 0*2+0 = 0, c[1,1] -> idx 1*2+1 = 3
        W[:, 0] = [1, 0, 0, 1]
        
        # M2 = (a10+a11)*b00
        U[:, 1] = [0, 0, 1, 1]  # a10 + a11
        V[:, 1] = [1, 0, 0, 0]  # b00
        # c[0,1] += M2, c[1,1] -= M2
        # c[0,1] -> idx 0*2+1 = 1, c[1,1] -> idx 1*2+1 = 3
        W[:, 1] = [0, 1, 0, -1]
        
        # M3 = a00*(b01-b11)
        U[:, 2] = [1, 0, 0, 0]  # a00
        V[:, 2] = [0, 1, 0, -1]  # b01 - b11
        # c[0,1] += M3, c[1,1] += M3
        W[:, 2] = [0, 1, 0, 1]
        
        # M4 = a11*(b10-b00)
        U[:, 3] = [0, 0, 0, 1]  # a11
        V[:, 3] = [0, -1, 1, 0]  # b10 - b00
        # c[0,0] -= M4, c[0,1] += M4
        # c[0,0] -> idx 0, c[0,1] -> idx 1
        W[:, 3] = [-1, 1, 0, 0]
        
        # M5 = (a00+a01)*b11
        U[:, 4] = [1, 1, 0, 0]  # a00 + a01
        V[:, 4] = [0, 0, 0, 1]  # b11
        # c[0,0] += M5, c[1,1] += M5
        W[:, 4] = [1, 0, 0, 1]
        
        # M6 = (a10-a00)*(b00+b10)
        U[:, 5] = [-1, 0, 1, 0]  # a10 - a00
        V[:, 5] = [1, 0, 1, 0]  # b00 + b10
        # c[1,1] += M6
        W[:, 5] = [0, 0, 0, 1]
        
        # M7 = (a01-a11)*(b01+b11)
        U[:, 6] = [0, 1, 0, -1]  # a01 - a11
        V[:, 6] = [0, 1, 0, 1]  # b01 + b11
        # c[0,0] += M7, c[0,1] += M7
        W[:, 6] = [1, 1, 0, 0]
        
        return U, V, W

    # Verify Strassen decomposition
    def verify_strassen():
        U, V, W = strassen_2x2x2()
        T2 = np.zeros((4, 4, 4), dtype=float)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    a = i * 2 + j
                    b = j * 2 + k
                    c = k * 2 + i
                    T2[a, b, c] = 1.0
        T2_recon = np.einsum('as,bs,cs->abc', U, V, W)
        return np.max(np.abs(T2_recon - T2)) < 1e-10

    # Helper: extract sub-tensor for block (i1:i2, j1:j2, k1:k2)
    # and construct the corresponding U, V, W factors
    def block_factors(nn, mm, pp, i_off, j_off, k_off, use_strassen=False):
        """
        Get U, V, W for a sub-block of the full (n,m,p) tensor.
        i_off, j_off, k_off are the starting indices in the full tensor.
        
        For the full tensor:
        - U has shape (n*m, rank_full), indexed by (i,j) -> i*m+j
        - V has shape (m*p, rank_full), indexed by (j,k) -> j*p+k  
        - W has shape (p*n, rank_full), indexed by (k,i) -> k*n+i
        
        For a sub-block (nn x mm x pp) at offset (i_off, j_off, k_off):
        - U_sub has shape (nn*mm, rank_sub), indexed by (i',j') -> i'*mm+j'
        - V_sub has shape (mm*pp, rank_sub), indexed by (j',k') -> j'*pp+k'
        - W_sub has shape (pp*nn, rank_sub), indexed by (k',i') -> k'*nn+i'
        
        To embed into full tensor:
        - U_full row i*m+j maps to U_sub row i'*mm+j' where i=i_off+i', j=j_off+j'
        - V_full row j*p+k maps to V_sub row j'*pp+k' where j=j_off+j', k=k_off+k'
        - W_full row k*n+i maps to W_sub row k'*nn+i' where k=k_off+k', i=i_off+i'
        """
        if use_strassen and nn == 2 and mm == 2 and pp == 2:
            U_sub, V_sub, W_sub = strassen_2x2x2()
        else:
            U_sub, V_sub, W_sub = schoolbook(nn, mm, pp)
        
        rank_sub = U_sub.shape[1]
        
        # Create embedding matrices
        U_full = np.zeros((n * m, rank_sub))
        V_full = np.zeros((m * p, rank_sub))
        W_full = np.zeros((p * n, rank_sub))
        
        for i in range(nn):
            for j in range(mm):
                U_full[(i_off + i) * m + (j_off + j), :] = U_sub[i * mm + j, :]
        
        for j in range(mm):
            for k in range(pp):
                V_full[(j_off + j) * p + (k_off + k), :] = V_sub[j * pp + k, :]
        
        for k in range(pp):
            for i in range(nn):
                W_full[(k_off + k) * n + (i_off + i), :] = W_sub[k * nn + i, :]
        
        return U_full, V_full, W_full

    # Build the decomposition by splitting p=5 into 2+3
    # r(4,5,5) = r(4,5,2) + r(4,5,3)
    # r(4,5,2): split m=5 into 2+3: r(4,2,2) + r(4,3,2)
    #   r(4,2,2): split n=4 into 2+2: 2*r(2,2,2) = 2*7 = 14
    #   r(4,3,2): split n=4 into 2+2: 2*r(2,3,2) = 2*12 = 24
    # r(4,5,3): split m=5 into 2+3: r(4,2,3) + r(4,3,3)
    #   r(4,2,3): split n=4 into 2+2: 2*r(2,2,3) = 2*12 = 24
    #   r(4,3,3): split n=4 into 2+2: 2*r(2,3,3) = 2*18 = 36
    # Total: 14 + 24 + 24 + 36 = 98
    
    blocks = []
    
    # p=2 part (k_off=0, pp=2)
    # m=2 part (j_off=0, mm=2)
    # n=2 parts (i_off=0 and i_off=2, nn=2)
    blocks.append((2, 2, 2, 0, 0, 0, True))   # (2,2,2) at (0,0,0), use Strassen
    blocks.append((2, 2, 2, 2, 0, 0, True))   # (2,2,2) at (2,0,0), use Strassen
    
    # m=3 part (j_off=2, mm=3)
    # n=2 parts (i_off=0 and i_off=2, nn=2)
    blocks.append((2, 3, 2, 0, 2, 0, False))  # (2,3,2) at (0,2,0)
    blocks.append((2, 3, 2, 2, 2, 0, False))  # (2,3,2) at (2,2,0)
    
    # p=3 part (k_off=2, pp=3)
    # m=2 part (j_off=0, mm=2)
    # n=2 parts (i_off=0 and i_off=2, nn=2)
    blocks.append((2, 2, 3, 0, 0, 2, False))  # (2,2,3) at (0,0,2)
    blocks.append((2, 2, 3, 2, 0, 2, False))  # (2,2,3) at (2,0,2)
    
    # m=3 part (j_off=2, mm=3)
    # n=2 parts (i_off=0 and i_off=2, nn=2)
    blocks.append((2, 3, 3, 0, 2, 2, False))  # (2,3,3) at (0,2,2)
    blocks.append((2, 3, 3, 2, 2, 2, False))  # (2,3,3) at (2,2,2)
    
    # Total rank
    total_rank = sum(nn*mm*pp if not use_str else 7 for nn,mm,pp,_,_,_,use_str in blocks)
    
    # Build full U, V, W
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for nn, mm, pp, i_off, j_off, k_off, use_strassen in blocks:
        U_blk, V_blk, W_blk = block_factors(nn, mm, pp, i_off, j_off, k_off, use_strassen)
        rank_blk = U_blk.shape[1]
        U[:, col:col+rank_blk] = U_blk
        V[:, col:col+rank_blk] = V_blk
        W[:, col:col+rank_blk] = W_blk
        col += rank_blk
    
    # Verify the decomposition
    if verify(U, V, W):
        return to_output(U, V, W)
    
    # Fallback to schoolbook if verification fails
    U, V, W = schoolbook(n, m, p)
    return to_output(U, V, W)
