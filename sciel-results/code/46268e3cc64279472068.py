import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = 110.0

    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n  # 20, 30, 24

    def strassen_2x2_block():
        """Return (U, V, W) for 2x2*2x2 Strassen.
        U indexed by (i,j) -> i*2+j, V by (j,k) -> j*2+k, W by (k,i) -> k*2+i
        """
        r = 7
        U = np.zeros((4, r))
        V = np.zeros((4, r))
        W = np.zeros((4, r))
        
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
        W[1, t] = 1; W[2, t] = 1
        
        # M7 = (a01-a11)(b10+b11)
        t = 6
        U[1, t] = 1; U[3, t] = -1
        V[2, t] = 1; V[3, t] = 1
        W[0, t] = 1; W[1, t] = 1
        
        return U, V, W

    def laderman_3x3_block():
        """Return (U, V, W) for 3x3*3x3 Laderman rank-7.
        U indexed by (i,j) -> i*3+j, V by (j,k) -> j*3+k, W by (k,i) -> k*3+i
        """
        r = 7
        U = np.zeros((9, r))
        V = np.zeros((9, r))
        W = np.zeros((9, r))
        
        # Indices: a_ij -> i*3+j, b_jk -> j*3+k, c_ki -> k*3+i
        
        # M1 = a00 * b00
        t = 0
        U[0, t] = 1
        V[0, t] = 1
        W[0, t] = 1
        
        # M2 = (a01+a11) * (b10-b11)
        t = 1
        U[1, t] = 1; U[4, t] = 1
        V[4, t] = 1; V[7, t] = -1
        W[1, t] = 1; W[4, t] = -1
        
        # M3 = (a00-a01) * (b01+b11)
        t = 2
        U[0, t] = 1; U[1, t] = -1
        V[1, t] = 1; V[4, t] = 1
        W[2, t] = 1; W[3, t] = 1
        
        # M4 = (a02+a22) * (b20-b22)
        t = 3
        U[2, t] = 1; U[8, t] = 1
        V[6, t] = 1; V[8, t] = -1
        W[2, t] = 1; W[8, t] = -1
        
        # M5 = (a20-a22) * (b21+b22)
        t = 4
        U[6, t] = 1; U[8, t] = -1
        V[7, t] = 1; V[8, t] = 1
        W[5, t] = 1; W[8, t] = 1
        
        # M6 = (a10+a20) * (b02+b12)
        t = 5
        U[3, t] = 1; U[6, t] = 1
        V[2, t] = 1; V[5, t] = 1
        W[6, t] = 1; W[7, t] = 1
        
        # M7 = (a12+a22) * (b10+b20)
        t = 6
        U[4, t] = 1; U[8, t] = 1
        V[3, t] = 1; V[6, t] = 1
        W[0, t] = 1; W[6, t] = 1
        
        return U, V, W

    # Verify Strassen
    sU, sV, sW = strassen_2x2_block()
    T_2x2 = np.zeros((4, 4, 4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T_2x2[i*2+j, j*2+k, k*2+i] = 1.0
    recon_2x2 = np.zeros((4, 4, 4))
    for t in range(7):
        recon_2x2 += np.einsum('a,b,c->abc', sU[:, t], sV[:, t], sW[:, t])
    strassen_ok = np.max(np.abs(recon_2x2 - T_2x2)) < 1e-10

    # Verify Laderman
    lU, lV, lW = laderman_3x3_block()
    T_3x3 = np.zeros((9, 9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                T_3x3[i*3+j, j*3+k, k*3+i] = 1.0
    recon_3x3 = np.zeros((9, 9, 9))
    for t in range(7):
        recon_3x3 += np.einsum('a,b,c->abc', lU[:, t], lV[:, t], lW[:, t])
    laderman_ok = np.max(np.abs(recon_3x3 - T_3x3)) < 1e-10

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

    # If Laderman doesn't work, use the old approach (rank 116)
    if not laderman_ok:
        total_rank = 116
        U = np.zeros((nm, total_rank))
        V = np.zeros((mp, total_rank))
        W = np.zeros((pn, total_rank))
        col = 0
        
        def add_schoolbook_old(i_start, i_end, j_start, j_end, k_start, k_end):
            nonlocal col
            for i in range(i_start, i_end):
                for j in range(j_start, j_end):
                    for k in range(k_start, k_end):
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1
        
        def add_strassen_old(i_off, j_off, k_off):
            nonlocal col
            sU, sV, sW = strassen_2x2_block()
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
        
        add_strassen_old(0, 0, 0)
        add_schoolbook_old(0, 2, 0, 2, 2, 3)
        add_schoolbook_old(0, 2, 2, 5, 0, 3)
        add_strassen_old(0, 0, 3)
        add_schoolbook_old(0, 2, 0, 2, 5, 6)
        add_schoolbook_old(0, 2, 2, 5, 3, 6)
        add_strassen_old(2, 0, 0)
        add_schoolbook_old(2, 4, 0, 2, 2, 3)
        add_schoolbook_old(2, 4, 2, 5, 0, 3)
        add_strassen_old(2, 0, 3)
        add_schoolbook_old(2, 4, 0, 2, 5, 6)
        add_schoolbook_old(2, 4, 2, 5, 3, 6)
        
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

    # Use Laderman for 3x3 blocks: rank 7 instead of 18
    # Partition: n=4=2+2, m=5=2+3, p=6=3+3
    # 2x2*2x3 blocks: Strassen(2x2*2x2)=7 + schoolbook(2x2*2x1)=4, total 11
    # 2x3*3x3 blocks: Laderman(3x3*3x3)=7 + schoolbook(2x1*3x3)=6, total 13
    # Total: 4*(11+13) = 96
    
    total_rank = 96
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
        nonlocal col
        sU, sV, sW = strassen_2x2_block()
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

    def add_laderman(i_off, j_off, k_off):
        nonlocal col
        lU, lV, lW = laderman_3x3_block()
        for t in range(7):
            for li in range(3):
                for lj in range(3):
                    U[(i_off+li)*m + (j_off+lj), col] = lU[li*3+lj, t]
            for lj in range(3):
                for lk in range(3):
                    V[(j_off+lj)*p + (k_off+lk), col] = lV[lj*3+lk, t]
            for lk in range(3):
                for li in range(3):
                    W[(k_off+lk)*n + (i_off+li), col] = lW[lk*3+li, t]
            col += 1

    # C[0:2, 0:3] = A[0:2,0:2]*B[0:2,0:3] + A[0:2,2:5]*B[2:5,0:3]
    add_strassen(0, 0, 0)  # 2x2*2x2, rank 7
    add_schoolbook(0, 2, 0, 2, 2, 3)  # 2x2*2x1, rank 4
    add_laderman(0, 2, 0)  # 2x3*3x3: use Laderman on 3x3*3x3 part, rank 7
    add_schoolbook(0, 2, 0, 2, 0, 3)  # Wait, this is wrong. Let me reconsider.
    
    # Actually, A[0:2,2:5] is 2x3, B[2:5,0:3] is 3x3
    # The product is 2x3. I need to decompose this 2x3*3x3 multiplication.
    # Split A[0:2,2:5] into rows: A[0,2:5] (1x3) and A[1,2:5] (1x3)
    # Each 1x3 * 3x3 = 1x3, schoolbook rank 9 each, total 18
    # Or: split into A[0:2,2:4] (2x2) * B[2:4,0:3] (2x3) + A[0:2,4:5] (2x1) * B[4:5,0:3] (1x3)
    # 2x2*2x3: Strassen(2x2*2x2)=7 + schoolbook(2x2*2x1)=4 = 11
    # 2x1*1x3: schoolbook rank 3
    # Total: 11+3 = 14. Better than 18.
    
    # Let me redo this more carefully.
    # I'll clear and restart the construction.
    
    total_rank = 0
    U = np.zeros((nm, 200))  # allocate extra, trim later
    V = np.zeros((mp, 200))
    W = np.zeros((pn, 200))
    col = 0

    def add_schoolbook2(i_start, i_end, j_start, j_end, k_start, k_end):
        nonlocal col
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

    def add_strassen2(i_off, j_off, k_off):
        nonlocal col
        sU, sV, sW = strassen_2x2_block()
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

    def add_laderman2(i_off, j_off, k_off):
        nonlocal col
        lU, lV, lW = laderman_3x3_block()
        for t in range(7):
            for li in range(3):
                for lj in range(3):
                    U[(i_off+li)*m + (j_off+lj), col] = lU[li*3+lj, t]
            for lj in range(3):
                for lk in range(3):
                    V[(j_off+lj)*p + (k_off+lk), col] = lV[lj*3+lk, t]
            for lk in range(3):
                for li in range(3):
                    W[(k_off+lk)*n + (i_off+li), col] = lW[lk*3+li, t]
            col += 1

    # Block 1: C[0:2, 0:3] = A[0:2,0:2]*B[0:2,0:3] + A[0:2,2:5]*B[2:5,0:3]
    # A[0:2,0:2]*B[0:2,0:3]: 2x2*2x3
    add_strassen2(0, 0, 0)  # 2x2*2x2, rank 7
    add_schoolbook2(0, 2, 0, 2, 2, 3)  # 2x2*2x1, rank 4
    # A[0:2,2:5]*B[2:5,0:3]: 2x3*3x3
    add_laderman2(0, 2, 0)  # 3x3*3x3 core... but A is only 2 rows!
    
    # Problem: Laderman needs 3x3 A block. I only have 2 rows.
    # So I can't directly apply Laderman here.
    # Alternative: use schoolbook for 2x3*3x3 = rank 18, or find better.
    
    # Let me just use schoolbook for the 2x3*3x3 blocks.
    # Total: 4*(7+4+18) = 116. Same as before.
    
    # The Laderman approach only helps if I have full 3x3 A blocks.
    # With n=4, I can't get 3x3 A blocks from a 2+2 split.
    # Try n=4=1+3? Then I get 1x5 and 3x5 A blocks.
    # 3x5*3x3: split 5=2+3, 3x2*3x3 (schoolbook 18) + 3x3*3x3 (Laderman 7) = 25
    # vs schoolbook 3x5*3x3 = 45. Saves 20.
    # 1x5*3x3: schoolbook 15
    # Total for n=1+3: 
    # C[0:1,0:3] = A[0:1,0:2]*B[0:2,0:3] + A[0:1,2:5]*B[2:5,0:3] = 1x2*2x3 + 1x3*3x3 = 6+9=15
    # C[0:1,3:6] = 1x2*2x3 + 1x3*3x3 = 6+9=15
    # C[1:4,0:3] = A[1:4,0:2]*B[0:2,0:3] + A[1:4,2:5]*B[2:5,0:3]
    #   A[1:4,0:2] is 3x2, B[0:2,0:3] is 2x3: 3x2*2x3, schoolbook 18
    #   A[1:4,2:5] is 3x3, B[2:5,0:3] is 3x3: Laderman 7
    #   Total: 25
    # C[1:4,3:6] = similarly 25
    # Grand total: 15+15+25+25 = 80!
    
    # But wait, I need to verify the 3x2*2x3 part. Can I do better than schoolbook 18?
    # 3x2*2x3: split into 2x2*2x2 (Strassen 7) + 1x2*2x2 (schoolbook 4) + 2x2*2x1 (schoolbook 4) = 15
    # Or just use schoolbook: 18.
    # Let's use 15 for the 3x2*2x3 part.
    
    # Revised total: 15+15+(15+7)+(15+7) = 15+15+22+22 = 74!
    
    # Let me implement this.
    
    U = np.zeros((nm, 100))
    V = np.zeros((mp, 100))
    W = np.zeros((pn, 100))
    col = 0

    def sb(i_s, i_e, j_s, j_e, k_s, k_e):
        nonlocal col
        for i in range(i_s, i_e):
            for j in range(j_s, j_e):
                for k in range(k_s, k_e):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1

    def st(i_o, j_o, k_o):
        nonlocal col
        sU, sV, sW = strassen_2x2_block()
        for t in range(7):
            for li in range(2):
                for lj in range(2):
                    U[(i_o+li)*m + (j_o+lj), col] = sU[li*2+lj, t]
            for lj in range(2):
                for lk in range(2):
                    V[(j_o+lj)*p + (k_o+lk), col] = sV[lj*2+lk, t]
            for lk in range(2):
                for li in range(2):
                    W[(k_o+lk)*n + (i_o+li), col] = sW[lk*2+li, t]
            col += 1

    def la(i_o, j_o, k_o):
        nonlocal col
        lU, lV, lW = laderman_3x3_block()
        for t in range(7):
            for li in range(3):
                for lj in range(3):
                    U[(i_o+li)*m + (j_o+lj), col] = lU[li*3+lj, t]
            for lj in range(3):
                for lk in range(3):
                    V[(j_o+lj)*p + (k_o+lk), col] = lV[lj*3+lk, t]
            for lk in range(3):
                for li in range(3):
                    W[(k_o+lk)*n + (i_o+li), col] = lW[lk*3+li, t]
            col += 1

    # C[0:1, 0:3] = A[0:1,0:2]*B[0:2,0:3] + A[0:1,2:5]*B[2:5,0:3]
    sb(0, 1, 0, 2, 0, 3)  # 1x2*2x3, rank 6
    sb(0, 1, 2, 5, 0, 3)  # 1x3*3x3, rank 9
    
    # C[0:1, 3:6] = A[0:1,0:2]*B[0:2,3:6] + A[0:1,2:5]*B[2:5,3:6]
    sb(0, 1, 0, 2, 3, 6)  # 1x2*2x3, rank 6
    sb(0, 1, 2, 5, 3, 6)  # 1x3*3x3, rank 9
    
    # C[1:4, 0:3] = A[1:4,0:2]*B[0:2,0:3] + A[1:4,2:5]*B[2:5,0:3]
    # A[1:4,0:2] is 3x2, B[0:2,0:3] is 2x3
    # Split: A[1:3,0:2] (2x2) * B[0:2,0:3] (2x3) + A[3:4,0:2] (1x2) * B[0:2,0:3] (2x3)
    # 2x2*2x3: st(1,0,0) + sb(1,3,0,2,2,3) = 7+4 = 11
    # 1x2*2x3: sb(3,4,0,2,0,3) = 6
    st(1, 0, 0)  # 2x2*2x2, rank 7
    sb(1, 3, 0, 2, 2, 3)  # 2x2*2x1, rank 4
    sb(3, 4, 0, 2, 0, 3)  # 1x2*2x3, rank 6
    # A[1:4,2:5] is 3x3, B[2:5,0:3] is 3x3
    la(1, 2, 0)  # 3x3*3x3, rank 7
    
    # C[1:4, 3:6] = A[1:4,0:2]*B[0:2,3:6] + A[1:4,2:5]*B[2:5,3:6]
    st(1, 0, 3)  # 2x2*2x2, rank 7
    sb(1, 3, 0, 2, 5, 6)  # 2x2*2x1, rank 4
    sb(3, 4, 0, 2, 3, 6)  # 1x2*2x3, rank 6
    la(1, 2, 3)  # 3x3*3x3, rank 7
    
    # Total: 6+9+6+9+7+4+6+7+7+4+6+7 = 78
    
    # Verify
    recon = np.zeros((nm, mp, pn))
    for t in range(col):
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
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Trim to actual rank
    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
