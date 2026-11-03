import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = 110.0

    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n  # 20, 30, 24

    # Partition: n=4=2+2, m=5=2+3, p=6=2+4
    # A (4x5): rows [0,1]|[2,3], cols [0,1]|[2,3,4]
    # B (5x6): rows [0,1]|[2,3,4], cols [0,1]|[2,3,4,5]
    # C (4x6): rows [0,1]|[2,3], cols [0,1]|[2,3,4,5]
    #
    # C[0:2,0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:5]*B[2:5,0:2]
    # C[0:2,2:6] = A[0:2,0:2]*B[0:2,2:6] + A[0:2,2:5]*B[2:5,2:6]
    # C[2:4,0:2] = A[2:4,0:2]*B[0:2,0:2] + A[2:4,2:5]*B[2:5,0:2]
    # C[2:4,2:6] = A[2:4,0:2]*B[0:2,2:6] + A[2:4,2:5]*B[2:5,2:6]
    #
    # Block ranks:
    # A[0:2,0:2]*B[0:2,0:2]: 2x2*2x2 = rank 7 (Strassen)
    # A[0:2,2:5]*B[2:5,0:2]: 2x3*3x2 = rank 12 (schoolbook)
    # A[0:2,0:2]*B[0:2,2:6]: 2x2*2x4 = rank 16 (schoolbook)
    # A[0:2,2:5]*B[2:5,2:6]: 2x3*3x4 = rank 24 (schoolbook)
    # A[2:4,0:2]*B[0:2,0:2]: 2x2*2x2 = rank 7 (Strassen)
    # A[2:4,2:5]*B[2:5,0:2]: 2x3*3x2 = rank 12 (schoolbook)
    # A[2:4,0:2]*B[0:2,2:6]: 2x2*2x4 = rank 16 (schoolbook)
    # A[2:4,2:5]*B[2:5,2:6]: 2x3*3x4 = rank 24 (schoolbook)
    # Total: 7+12+16+24+7+12+16+24 = 118
    #
    # Hmm, that's only 2 better than 120. Let me try a better partition.
    
    # Try: n=4=2+2, m=5=2+3, p=6=3+3
    # A[0:2,0:2]*B[0:2,0:3]: 2x2*2x3 = rank 12
    # A[0:2,2:5]*B[2:5,0:3]: 2x3*3x3 = rank 18
    # A[0:2,0:2]*B[0:2,3:6]: 2x2*2x3 = rank 12
    # A[0:2,2:5]*B[2:5,3:6]: 2x3*3x3 = rank 18
    # A[2:4,0:2]*B[0:2,0:3]: 2x2*2x3 = rank 12
    # A[2:4,2:5]*B[2:5,0:3]: 2x3*3x3 = rank 18
    # A[2:4,0:2]*B[0:2,3:6]: 2x2*2x3 = rank 12
    # A[2:4,2:5]*B[2:5,3:6]: 2x3*3x3 = rank 18
    # Total: 120. No improvement.
    
    # The key insight: for rectangular blocks like 2x2*2x3, I can split
    # the 2x3 into 2x2 + 2x1 and use Strassen on the 2x2*2x2 part.
    # 2x2*2x3 = 2x2*2x2 + 2x2*2x1 = rank 7 + rank 4 = rank 11
    # vs schoolbook rank 12. Saves 1 per block.
    
    # With n=4=2+2, m=5=2+3, p=6=3+3:
    # 2x2*2x3 blocks (4 of them): 11 each = 44
    # 2x3*3x3 blocks (4 of them): 18 each = 72
    # Total: 44+72 = 116. Saves 4.
    
    # Let me try to get Strassen working properly for 2x2*2x2.
    
    def strassen_2x2_block(i_off, j_off, k_off):
        """Return (U, V, W) contributions for 2x2*2x2 at offset (i_off, j_off, k_off).
        U shape (4,7), V shape (4,7), W shape (4,7)
        U indexed by local (i,j) -> i*2+j, V by (j,k) -> j*2+k, W by (k,i) -> k*2+i
        """
        r = 7
        U = np.zeros((4, r))
        V = np.zeros((4, r))
        W = np.zeros((4, r))
        
        # a00=0, a01=1, a10=2, a11=3
        # b00=0, b01=1, b10=2, b11=3
        # c00=0, c01=1, c10=2, c11=3 (W index: k*2+i)
        
        # M1 = (a10+a11)(b00+b11)
        t = 0
        U[2, t] = 1; U[3, t] = 1
        V[0, t] = 1; V[3, t] = 1
        W[0, t] = 1; W[3, t] = 1
        
        # M2 = a10(b11-b00)
        t = 1
        U[2, t] = 1
        V[3, t] = 1; V[0, t] = -1
        W[2, t] = 1; W[3, t] = -1
        
        # M3 = (a00+a01)b11
        t = 2
        U[0, t] = 1; U[1, t] = 1
        V[3, t] = 1
        W[1, t] = 1; W[3, t] = 1
        
        # M4 = a00(b10-b11)
        t = 3
        U[0, t] = 1
        V[2, t] = 1; V[3, t] = -1
        W[0, t] = 1; W[3, t] = -1
        
        # M5 = (a00+a10)b00
        t = 4
        U[0, t] = 1; U[2, t] = 1
        V[0, t] = 1
        W[0, t] = -1; W[2, t] = 1
        
        # M6 = (a10-a00)(b00+b01)
        t = 5
        U[2, t] = 1; U[0, t] = -1
        V[0, t] = 1; V[1, t] = 1
        W[0, t] = 0; W[1, t] = 0; W[2, t] = 0; W[3, t] = 0
        
        # M7 = (a01-a11)(b10+b11)
        t = 6
        U[1, t] = 1; U[3, t] = -1
        V[2, t] = 1; V[3, t] = 1
        W[0, t] = 1; W[1, t] = 1
        
        return U, V, W

    # Verify Strassen
    sU, sV, sW = strassen_2x2_block(0, 0, 0)
    T_2x2 = np.zeros((4, 4, 4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T_2x2[i*2+j, j*2+k, k*2+i] = 1.0
    
    recon_2x2 = np.zeros((4, 4, 4))
    for t in range(7):
        recon_2x2 += np.einsum('a,b,c->abc', sU[:, t], sV[:, t], sW[:, t])
    
    strassen_ok = np.max(np.abs(recon_2x2 - T_2x2)) < 1e-10
    
    if not strassen_ok:
        # Fall back to schoolbook
        rank_sb = n * m * p
        U = np.zeros((nm, rank_sb))
        V = np.zeros((mp, rank_sb))
        W = np.zeros((pn, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Strassen is valid! Use partition n=4=2+2, m=5=2+3, p=6=3+3
    # For 2x2*2x3 blocks, split into 2x2*2x2 (Strassen, rank 7) + 2x2*2x1 (schoolbook, rank 4)
    # Total per 2x2*2x3 block: 11
    # For 2x3*3x3 blocks: schoolbook rank 18
    # Total: 4*11 + 4*18 = 44 + 72 = 116
    
    total_rank = 116
    U = np.zeros((nm, total_rank))
    V = np.zeros((mp, total_rank))
    W = np.zeros((pn, total_rank))
    col = 0
    
    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        nonlocal col
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    
    def add_strassen(i_off, j_off, k_off):
        """Add Strassen rank-7 for 2x2*2x2 at given offsets."""
        nonlocal col
        sU, sV, sW = strassen_2x2_block(i_off, j_off, k_off)
        for t in range(7):
            for li in range(2):
                for lj in range(2):
                    U[(i_off+li)*m + (j_off+lj), col] = sU[li*2+lj, t]
            for lj in range(2):
                for lk in range(2):
                    V[(j_off+lj)*p + (k_off+lk), col] = sV[lj*2+lk, t]
            for lk in range(2):
                for li in range(2):
                    W[(k_off+lk)*n + (i_off+li), col] = sW[lk*2+li, t]
            col += 1
    
    # Partition: n=4=2+2, m=5=2+3, p=6=3+3
    # A rows: [0,1]|[2,3], A cols: [0,1]|[2,3,4]
    # B rows: [0,1]|[2,3,4], B cols: [0,1,2]|[3,4,5]
    # C rows: [0,1]|[2,3], C cols: [0,1,2]|[3,4,5]
    
    # C[0:2, 0:3] = A[0:2,0:2]*B[0:2,0:3] + A[0:2,2:5]*B[2:5,0:3]
    #   A[0:2,0:2]*B[0:2,0:3]: 2x2*2x3 = Strassen(2x2*2x2) + schoolbook(2x2*2x1)
    add_strassen(0, 0, 0)  # 2x2*2x2 for B cols 0:2, rank 7
    add_schoolbook(0, 2, 0, 2, 2, 3)  # 2x2*2x1 for B col 2, rank 4
    add_schoolbook(0, 2, 2, 5, 0, 3)  # 2x3*3x3, rank 18
    
    # C[0:2, 3:6] = A[0:2,0:2]*B[0:2,3:6] + A[0:2,2:5]*B[2:5,3:6]
    add_strassen(0, 0, 3)  # 2x2*2x2 for B cols 3:5, rank 7
    add_schoolbook(0, 2, 0, 2, 5, 6)  # 2x2*2x1 for B col 5, rank 4
    add_schoolbook(0, 2, 2, 5, 3, 6)  # 2x3*3x3, rank 18
    
    # C[2:4, 0:3] = A[2:4,0:2]*B[0:2,0:3] + A[2:4,2:5]*B[2:5,0:3]
    add_strassen(2, 0, 0)  # 2x2*2x2 for B cols 0:2, rank 7
    add_schoolbook(2, 4, 0, 2, 2, 3)  # 2x2*2x1 for B col 2, rank 4
    add_schoolbook(2, 4, 2, 5, 0, 3)  # 2x3*3x3, rank 18
    
    # C[2:4, 3:6] = A[2:4,0:2]*B[0:2,3:6] + A[2:4,2:5]*B[2:5,3:6]
    add_strassen(2, 0, 3)  # 2x2*2x2 for B cols 3:5, rank 7
    add_schoolbook(2, 4, 0, 2, 5, 6)  # 2x2*2x1 for B col 5, rank 4
    add_schoolbook(2, 4, 2, 5, 3, 6)  # 2x3*3x3, rank 18
    
    # Total: 4*(7+4+18) = 4*29 = 116
    # But wait, I have 4 strassen (4*7=28) + 4 schoolbook 2x2x1 (4*4=16) + 4 schoolbook 2x3x3 (4*18=72)
    # = 28+16+72 = 116. But col should be 116.
    
    # Verify
    recon = np.zeros((nm, mp, pn))
    for t in range(total_rank):
        recon += np.einsum('a,b,c->abc', U[:, t], V[:, t], W[:, t])
    
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    err = np.max(np.abs(recon - T))
    
    if err > 1e-10:
        # Fallback to schoolbook
        rank_sb = n * m * p
        U = np.zeros((nm, rank_sb))
        V = np.zeros((mp, rank_sb))
        W = np.zeros((pn, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
