"""Rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110
    
    # Verify helper
    def verify(U, V, W, n, m, p, rank):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Schoolbook decomposition (rank = 96)
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, rank
    
    # Strassen for 2x2x2 blocks
    # A (2x2) * B (2x2) = C (2x2)
    # Indices: A[r,c], B[c,d], C[r,d] with r,c,d in {0,1}
    # U index: r*2+c, V index: c*2+d, W index: d*2+r
    def strassen_222(i0, j0, k0, U, V, W, col):
        """Add Strassen decomposition for block at (i0,j0,k0). Returns new col."""
        # M1 = (A00+A11)(B00+B11): U terms: (0,0)=1,(1,1)=1; V: (0,0)=1,(1,1)=1; W: (0,0)=1,(1,1)=1
        U[(i0+0)*m+(j0+0)][col] += 1; V[(j0+0)*p+(k0+0)][col] += 1; W[(k0+0)*n+(i0+0)][col] += 1
        U[(i0+1)*m+(j0+1)][col] += 1; V[(j0+1)*p+(k0+1)][col] += 1; W[(k0+1)*n+(i0+1)][col] += 1
        col += 1
        
        # M2 = (A10+A11)B00: U: (1,0)=1,(1,1)=1; V: (0,0)=1; W: (0,1)=1
        U[(i0+1)*m+(j0+0)][col] += 1; V[(j0+0)*p+(k0+0)][col] += 1; W[(k0+0)*n+(i0+1)][col] += 1
        U[(i0+1)*m+(j0+1)][col] += 1; V[(j0+1)*p+(k0+0)][col] += 1; W[(k0+1)*n+(i0+1)][col] += 1
        col += 1
        
        # M3 = A00(B10-B11): U: (0,0)=1; V: (0,1)=1,(1,1)=-1; W: (1,0)=1
        U[(i0+0)*m+(j0+0)][col] += 1; V[(j0+0)*p+(k0+1)][col] += 1; W[(k0+1)*n+(i0+0)][col] += 1
        U[(i0+1)*m+(j0+1)][col] += 1; V[(j0+1)*p+(k0+1)][col] += 1; W[(k0+1)*n+(i0+1)][col] += 1
        # Wait, M3 should only have U=(0,0), V=(0,1) and (1,1), W=(1,0)
        # Let me redo: M3 = A[0,0] * (B[1,0] - B[1,1])
        # U: (0,0) -> 1
        # V: (0,1) -> 1, (1,1) -> -1  
        # W: (1,0) -> 1
        # But we need separate terms for the -1 part
        # Actually in tensor form, M3 is ONE rank-1 term: U*M3 * V*M3 * W*M3
        # where U has 1 at (0,0), V has 1 at (0,1) and -1 at (1,1), W has 1 at (1,0)
        # But that's not a single rank-1 term in the tensor sense.
        
        # I think I'm overcomplicating this. Let me use the correct Strassen terms.
        # Each Mi is a product of a linear form in A and a linear form in B,
        # contributing to a linear combination in C.
        
        # Let me restart with the proper structure.
        # For term r, we have:
        #   U_r[i*m+j] = a_{ij} (coefficient in the A-factor)
        #   V_r[j*p+k] = b_{jk} (coefficient in the B-factor)  
        #   W_r[k*n+i] = c_{ki} (coefficient in the C-factor)
        # And the product U_r * V_r * W_r should contribute to T.
        
        # Strassen for 2x2x2:
        # M1: A-factor: A00+A11, B-factor: B00+B11, C: C00+C11
        # M2: A-factor: A10+A11, B-factor: B00, C: C10-C11
        # M3: A-factor: A00, B-factor: B10-B11, C: C01+C11  
        # M4: A-factor: A11, B-factor: B01-B10, C: C00+C10
        # M5: A-factor: A00+A01, B-factor: B11, C: -C00+C01
        # M6: A-factor: A10-A00, B-factor: B00+B10, C: C00
        # M7: A-factor: A01-A11, B-factor: B01+B11, C: -C00+C11
        
        # Let me verify: C00 = M1 + M4 - M5 - M7 + M6? 
        # Actually standard Strassen:
        # p1 = (A00+A11)(B00+B11)
        # p2 = (A10+A11)(B00)
        # p3 = (A00)(B10-B11)
        # p4 = (A11)(B01-B10)
        # p5 = (A00+A01)(B11)
        # p6 = (A10-A00)(B00+B10)
        # p7 = (A01-A11)(B01+B11)
        # C00 = p1 - p5 + p6 - p7
        # C01 = p3 + p5
        # C10 = p2 + p4
        # C11 = p1 - p3 + p4 + p7
        
        # Let me just implement this correctly.
        pass
    
    # Let me take a cleaner approach: directly construct Strassen terms
    def build_strassen_block(i0, j0, k0, U, V, W, col):
        """Add 7 Strassen terms for the 2x2x2 block starting at (i0,j0,k0)."""
        # Term 1: (A00+A11)(B00+B11) -> contributes to C00 and C11
        U[(i0+0)*m+(j0+0)][col] = 1
        U[(i0+1)*m+(j0+1)][col] = 1
        V[(j0+0)*p+(k0+0)][col] = 1
        V[(j0+1)*p+(k0+1)][col] = 1
        W[(k0+0)*n+(i0+0)][col] = 1
        W[(k0+1)*n+(i0+1)][col] = 1
        col += 1
        
        # Term 2: (A10+A11)B00 -> contributes to C10 (coeff +1) and C11 (coeff -1)
        U[(i0+1)*m+(j0+0)][col] = 1
        U[(i0+1)*m+(j0+1)][col] = 1
        V[(j0+0)*p+(k0+0)][col] = 1
        W[(k0+0)*n+(i0+1)][col] = 1
        W[(k0+1)*n+(i0+1)][col] = -1
        col += 1
        
        # Term 3: A00(B10-B11) -> contributes to C01 (coeff +1) and C11 (coeff +1)
        U[(i0+0)*m+(j0+0)][col] = 1
        V[(j0+0)*p+(k0+1)][col] = 1
        V[(j0+1)*p+(k0+1)][col] = -1
        W[(k0+1)*n+(i0+0)][col] = 1
        W[(k0+1)*n+(i0+1)][col] = 1
        col += 1
        
        # Term 4: A11(B01-B10) -> contributes to C00 (coeff +1) and C10 (coeff +1)
        U[(i0+1)*m+(j0+1)][col] = 1
        V[(j0+0)*p+(k0+1)][col] = 1
        V[(j0+1)*p+(k0+0)][col] = -1
        W[(k0+0)*n+(i0+0)][col] = 1
        W[(k0+0)*n+(i0+1)][col] = 1
        col += 1
        
        # Term 5: (A00+A01)B11 -> contributes to C00 (coeff -1) and C01 (coeff +1)
        U[(i0+0)*m+(j0+0)][col] = 1
        U[(i0+0)*m+(j0+1)][col] = 1
        V[(j0+1)*p+(k0+1)][col] = 1
        W[(k0+1)*n+(i0+0)][col] = -1
        W[(k0+1)*n+(i0+0)][col] = 1  # This overwrites, need to handle properly
        # Actually C00 gets -1, C01 gets +1
        W[(k0+1)*n+(i0+0)][col] = -1
        W[(k0+1)*n+(i0+0)][col] = 1
        # I'm making errors. Let me think more carefully.
        # W index is k*n+i. For C00: k=0,i=0 -> W[0*n+0]. For C01: k=1,i=0 -> W[1*n+0]
        # So:
        W[(k0+0)*n+(i0+0)][col] = -1
        W[(k0+1)*n+(i0+0)][col] = 1
        col += 1
        
        # Term 6: (A10-A00)(B00+B10) -> contributes to C00 (coeff +1)
        U[(i0+1)*m+(j0+0)][col] = 1
        U[(i0+0)*m+(j0+0)][col] = -1
        V[(j0+0)*p+(k0+0)][col] = 1
        V[(j0+1)*p+(k0+0)][col] = 1
        W[(k0+0)*n+(i0+0)][col] = 1
        col += 1
        
        # Term 7: (A01-A11)(B01+B11) -> contributes to C00 (coeff -1) and C11 (coeff +1)
        U[(i0+0)*m+(j0+1)][col] = 1
        U[(i0+1)*m+(j0+1)][col] = -1
        V[(j0+0)*p+(k0+1)][col] = 1
        V[(j0+1)*p+(k0+1)][col] = 1
        W[(k0+0)*n+(i0+0)][col] = -1
        W[(k0+1)*n+(i0+1)][col] = 1
        col += 1
        
        return col
    
    # Verify Strassen block
    def test_strassen():
        U = np.zeros((n*m, 7))
        V = np.zeros((m*p, 7))
        W = np.zeros((p*n, 7))
        col = build_strassen_block(0, 0, 0, U.tolist(), V.tolist(), W.tolist(), 0)
        U = np.array(U); V = np.array(V); W = np.array(W)
        # Check: for i in {0,1}, j in {0,1}, k in {0,1}: sum_r U[i*4+j, r] * V[j*8+k, r] * W[k*3+i, r] should be 1
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = sum(U[i*4+j, r] * V[j*8+k, r] * W[k*3+i, r] for r in range(7))
                    if abs(s - 1.0) > 1e-9:
                        print(f"FAIL at ({i},{j},{k}): {s}")
                        return False
        print("Strassen block verified")
        return True
    
    # Build full decomposition using block Strassen where possible
    def build_block_decomposition():
        # Split n=3 into [0,2) and [2,3)
        # Split m=4 into [0,2) and [2,4)
        # Split p=8 into [0,2), [2,4), [4,6), [6,8)
        
        i_blocks = [(0, 2), (2, 3)]
        m_blocks = [(0, 2), (2, 4)]
        p_blocks = [(0, 2), (2, 4), (4, 6), (6, 8)]
        
        # Calculate total rank
        total_rank = 0
        blocks_info = []
        for i0, i1 in i_blocks:
            for j0, j1 in m_blocks:
                for k0, k1 in p_blocks:
                    di = i1 - i0
                    dj = j1 - j0
                    dk = k1 - k0
                    if di == 2 and dj == 2 and dk == 2:
                        r = 7
                        method = 'strassen'
                    else:
                        r = di * dj * dk
                        method = 'schoolbook'
                    total_rank += r
                    blocks_info.append((i0, i1, j0, j1, k0, k1, method, r))
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        for (i0, i1, j0, j1, k0, k1, method, r) in blocks_info:
            di = i1 - i0
            dj = j1 - j0
            dk = k1 - k0
            
            if method == 'strassen':
                col = build_strassen_block(i0, j0, k0, U, V, W, col)
            else:
                for di_idx in range(di):
                    for dj_idx in range(dj):
                        for dk_idx in range(dk):
                            U[(i0+di_idx)*m + (j0+dj_idx)][col] = 1
                            V[(j0+dj_idx)*p + (k0+dk_idx)][col] = 1
                            W[(k0+dk_idx)*n + (i0+di_idx)][col] = 1
                            col += 1
        
        if col != total_rank:
            return None, None, None, 0
        
        return U, V, W, total_rank
    
    # Try block decomposition
    U, V, W, rank = build_block_decomposition()
    
    if U is not None and verify(U, V, W, n, m, p, rank):
        print(f"Block decomposition verified with rank {rank}")
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W, rank = schoolbook()
    if verify(U, V, W, n, m, p, rank):
        print(f"Schoolbook decomposition verified with rank {rank}")
        return {"U": U, "V": V, "W": W}
    
    # Should not reach here
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
