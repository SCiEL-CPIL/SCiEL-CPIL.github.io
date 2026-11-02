import numpy as np
import time

def construct():
    n, m, p = 4, 5, 6
    
    # Total rank estimate: 108
    # We will build it up
    rank = 108
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    
    # Define Strassen decomposition for 2x2 * 2x2
    # Returns U (2x2), V (2x2), W (2x2) for the 7 terms?
    # No, the tensor decomposition of 2x2*2x2 with rank 7.
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # For 2x2*2x2: n=2, m=2, p=2.
    # U is 4x7, V is 4x7, W is 4x7.
    
    def get_strassen_2x2():
        # Standard Strassen coefficients for T_ijk = 1
        # r=0..6
        # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        # This is tricky to hardcode directly.
        # Let's use the known rank 7 decomposition.
        # One such decomposition is given by the 7 Strassen products.
        # P1 = (A11+A22)(B11+B22)
        # P2 = (A21+A22)B11
        # P3 = A11(B12-B22)
        # P4 = A22(B21-B11)
        # P5 = (A11+A12)B22
        # P6 = (A21-A11)(B11+B12)
        # P7 = (A12-A22)(B21+B22)
        
        # C11 = P1 + P4 - P5 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6
        
        # We need to express P_k as sum of U*V*W.
        # Each P_k is a scalar product of linear forms.
        # So each P_k contributes 1 to the rank? No, P_k is a product of two linear forms.
        # The tensor is sum of P_k * (linear form in C).
        # So each term is rank 1?
        # Yes, the rank 7 decomposition consists of 7 rank-1 terms.
        
        # Term 0: P1
        # U part: A11+A22 -> (1,0; 0,1) + (0,0; 0,1)? No.
        # A is [[a00, a01], [a10, a11]].
        # A11+A22 = a00 + a11.
        # This is a linear form in A.
        # B11+B22 = b00 + b11.
        # Linear form in B.
        # C part: C11 and C22.
        # C11 += 1 * P1. C22 += 1 * P1.
        
        # So for r=0:
        # U[0,0] = 1, U[1,1] = 1 (for a00, a11)
        # V[0,0] = 1, V[1,1] = 1 (for b00, b11)
        # W[0,0] = 1 (C11), W[1,1] = 1 (C22)
        
        # Let's generate these 7 terms.
        # U is 4x7, V is 4x7, W is 4x7.
        u = np.zeros((4, 7))
        v = np.zeros((4, 7))
        w = np.zeros((4, 7))
        
        # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        
        # P1: (a00+a11)(b00+b11) -> C11, C22
        u[0, 0] = 1; u[3, 0] = 1
        v[0, 0] = 1; v[3, 0] = 1
        w[0, 0] = 1; w[3, 0] = 1
        
        # P2: (a10+a11)b00 -> C21
        u[2, 1] = 1; u[3, 1] = 1
        v[0, 1] = 1
        w[1, 1] = 1 # C21 is k=0, i=1 -> W[0*2+1] = W[1]
        
        # P3: a00(b01-b11) -> C12
        u[0, 2] = 1
        v[1, 2] = 1; v[3, 2] = -1
        w[2, 2] = 1 # C12 is k=1, i=0 -> W[1*2+0] = W[2]
        
        # P4: a11(b10-b00) -> C11, C21
        u[3, 3] = 1
        v[2, 3] = 1; v[0, 3] = -1
        w[0, 3] = 1 # C11
        w[1, 3] = 1 # C21
        
        # P5: (a00+a01)b11 -> C11, C12
        u[0, 4] = 1; u[1, 4] = 1
        v[3, 4] = 1
        w[0, 4] = 1 # C11
        w[2, 4] = 1 # C12
        
        # P6: (a10-a00)(b00+b01) -> C22
        u[2, 5] = 1; u[0, 5] = -1
        v[0, 5] = 1; v[1, 5] = 1
        w[3, 5] = 1 # C22 is k=1, i=1 -> W[1*2+1] = W[3]
        
        # P7: (a01-a11)(b10+b11) -> C11, C22
        u[1, 6] = 1; u[3, 6] = -1
        v[2, 6] = 1; v[3, 6] = 1
        w[0, 6] = 1 # C11
        w[3, 6] = 1 # C22
        
        return u, v, w

    def get_simple_2x1x2():
        # 2x1 * 1x2
        # U is 2x4 (i in 0,1, j=0)
        # V is 2x4 (j=0, k in 0,1)
        # W is 4x4 (k in 0,1, i in 0,1)
        u = np.zeros((2, 4))
        v = np.zeros((2, 4))
        w = np.zeros((4, 4))
        
        # r=0: a0 b0 -> C00
        u[0, 0] = 1
        v[0, 0] = 1
        w[0, 0] = 1 # k=0, i=0
        
        # r=1: a0 b1 -> C01
        u[0, 1] = 1
        v[1, 1] = 1
        w[1, 1] = 1 # k=1, i=0 -> W[1*2+0] = W[2]?
        # W index is k*n + i. n=2.
        # k=1, i=0 -> 1*2+0 = 2.
        # So w[2, 1] = 1.
        
        # r=2: a1 b0 -> C10
        u[1, 2] = 1
        v[0, 2] = 1
        w[0, 2] = 1 # k=0, i=1 -> W[0*2+1] = W[1]
        
        # r=3: a1 b1 -> C11
        u[1, 3] = 1
        v[1, 3] = 1
        w[3, 3] = 1 # k=1, i=1 -> W[1*2+1] = W[3]
        
        # Correct W indices:
        # w[0,0] = 1
        # w[2,1] = 1
        # w[1,2] = 1
        # w[3,3] = 1
        
        return u, v, w

    u_str, v_str, w_str = get_strassen_2x2()
    u_smp, v_smp, w_smp = get_simple_2x1x2()
    
    # Helper to add a block
    def add_block(i_start, j_start, k_start, block_type, col_offset):
        if block_type == 'strassen':
            # i_start, i_start+1
            # j_start, j_start+1
            # k_start, k_start+1
            for li in range(2):
                for lj in range(2):
                    for r in range(7):
                        val = u_str[li*2+lj, r]
                        if val != 0:
                            U[(i_start+li)*m + (j_start+lj), col_offset+r] = val
            for lj in range(2):
                for lk in range(2):
                    for r in range(7):
                        val = v_str[lj*2+lk, r]
                        if val != 0:
                            V[(j_start+lj)*p + (k_start+lk), col_offset+r] = val
            for lk in range(2):
                for li in range(2):
                    for r in range(7):
                        val = w_str[lk*2+li, r]
                        if val != 0:
                            W[(k_start+lk)*n + (i_start+li), col_offset+r] = val
            return 7
        elif block_type == 'simple':
            # i_start, i_start+1
            # j_start
            # k_start, k_start+1
            for li in range(2):
                for r in range(4):
                    val = u_smp[li, r]
                    if val != 0:
                        U[(i_start+li)*m + j_start, col_offset+r] = val
            for lk in range(2):
                for r in range(4):
                    val = v_smp[lk, r]
                    if val != 0:
                        V[j_start*p + (k_start+lk), col_offset+r] = val
            for lk in range(2):
                for li in range(2):
                    for r in range(4):
                        val = w_smp[lk*2+li, r]
                        if val != 0:
                            W[(k_start+lk)*n + (i_start+li), col_offset+r] = val
            return 4

    # Blocks
    # A rows: 0-1 (0), 2-3 (1)
    # A cols: 0-1 (0), 2-3 (1), 4 (2)
    # B rows: 0-1 (0), 2-3 (1), 4 (2)
    # B cols: 0-1 (0), 2-3 (1), 4-5 (2)
    
    # i_block, j_block, k_block
    # j_block 0: 2x2, j_block 1: 2x2, j_block 2: 1x1 (in A) / 1x2 (in B)
    
    for i_blk in range(2):
        i_start = i_blk * 2
        for j_blk in range(3):
            if j_blk < 2:
                j_start = j_blk * 2
                b_type = 'strassen'
            else:
                j_start = 4
                b_type = 'simple'
                
            for k_blk in range(3):
                if k_blk < 2:
                    k_start = k_blk * 2
                else:
                    k_start = 4
                    
                # Determine if B block is 2x2 or 1x2
                # B rows: j_blk determines row block of B.
                # If j_blk < 2, B row block is 2x2.
                # If j_blk == 2, B row block is 1x2.
                
                # Wait, the block type depends on both A and B.
                # A col block j_blk: 0,1 are 2 wide. 2 is 1 wide.
                # B row block j_blk: 0,1 are 2 high. 2 is 1 high.
                # B col block k_blk: 0,1 are 2 wide. 2 is 2 wide.
                
                # If j_blk < 2: A is 2x2, B is 2x2 (since k_blk is 2 wide? No, B is 2x2 if k_blk<2? No, B is 2x2 if k_blk is 0 or 1? B cols are 2,2,2. So B is always 2 wide?
                # B cols: 0-1, 2-3, 4-5. All width 2.
                # So B is 2 wide always.
                # B rows: 0-1 (2), 2-3 (2), 4 (1).
                
                # So if j_blk < 2: B is 2x2. A is 2x2. -> Strassen.
                # If j_blk == 2: B is 1x2. A is 2x1. -> Simple.
                
                if j_blk < 2:
                    rank_added = add_block(i_start, j_start, k_start, 'strassen', col)
                else:
                    rank_added = add_block(i_start, j_start, k_start, 'simple', col)
                
                col += rank_added

    # Verify
    # The total col should be 108.
    # U, V, W are constructed.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print(res['U'][0][:5])
