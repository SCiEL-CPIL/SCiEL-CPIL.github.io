import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n  # 20, 30, 24
    
    # Build the target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Strassen-based recursive decomposition
    # Partition: A (4x5) = [A1(4x4) | A2(4x1)]
    #              B (5x6) = [B1(4x6); B2(1x6)]
    # C = A1*B1 + A2*B2
    # 
    # A1*B1: 4x4 times 4x4 -> use Strassen (rank 7)
    # A2*B2: 4x1 times 1x6 -> schoolbook (rank 4)
    # Total rank: 7 + 4 = 11
    #
    # Strassen decomposition for 4x4 * 4x4:
    # We need to find rank-7 decomposition.
    # Standard approach: use the fact that 4x4 mult can be done with 7 multiplications
    # via recursive Strassen on 2x2 blocks, or use known rank-7 scheme.
    #
    # Let's construct it explicitly. For 4x4 matrices X, Y:
    # Partition into 2x2 blocks: X = [[X11,X12],[X21,X22]], Y = [[Y11,Y12],[Y21,Y22]]
    # Each 2x2*2x2 uses Strassen with 7 mults, so 4 sub-multiplications * 7 = 28... 
    # That's not rank 7 total. 
    #
    # Actually, the rank of 4x4 matrix multiplication is known to be 49 (schoolbook) 
    # but can be reduced. The Strassen algorithm gives rank 28 for 4x4 (4 sub-blocks each rank 7).
    # 
    # Let me reconsider. The known best upper bound for 4x4 is 49, with Strassen giving 28.
    # For our problem, we need 4x4 * 4x4 which has rank at most 28 via Strassen.
    #
    # Let me build the Strassen decomposition for 4x4 * 4x4 explicitly.
    # 
    # 4x4 matrices partitioned into 2x2 blocks of 2x2:
    # X = [[X00, X01], [X10, X11]], Y = [[Y00, Y01], [Y10, Y11]]
    # 
    # Each 2x2 * 2x2 uses Strassen:
    # M1 = (X11 + X10) * (Y00 + Y11)
    # M2 = X10 * (Y11 - Y00)  
    # M3 = (X00 + X01) * Y11
    # M4 = X00 * (Y10 - Y11)
    # M5 = (X00 + X11) * Y00
    # M6 = (X10 - X00) * (Y00 + Y01)
    # M7 = (X01 - X11) * (Y10 + Y11)
    # 
    # Result blocks:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M7
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 - M4
    # 
    # For 4x4 * 4x4, we have 4 such sub-multiplications (for each pair of 2x2 blocks).
    # Total rank = 4 * 7 = 28.
    
    # Build Strassen rank-28 decomposition for 4x4 * 4x4
    # Indices: X[i,j] where i,j in 0..3, Y[j,k] where j,k in 0..3
    # X row index: i (0-3), X col index: j (0-3)  
    # Y row index: j (0-3), Y col index: k (0-3)
    # Result: C[i,k]
    
    def build_strassen_4x4():
        """Build rank-28 Strassen decomposition for 4x4 * 4x4.
        Returns U (16, 28), V (16, 28), W (16, 28) such that
        sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] = delta for the 4x4*4x4 tensor.
        Here i,j,k are in 0..3, so U is indexed by (i,j) -> i*4+j, etc.
        """
        r = 28
        U = np.zeros((16, r))
        V = np.zeros((16, r))
        W = np.zeros((16, r))
        
        # 4 sub-multiplications: (row_block, col_block) in {(0,0), (0,1), (1,0), (1,1)}
        # row_block rb: X rows 2*rb to 2*rb+1
        # col_block cb: X cols 2*cb to 2*cb+1, Y rows 2*cb to 2*cb+1
        # Y col_block: 2*cb to 2*cb+1
        
        for rb in range(2):
            for cb in range(2):
                base = (rb * 2 + cb) * 7
                # X block: rows 2*rb..2*rb+1, cols 2*cb..2*cb+1
                # X11 = X[2*rb+1, 2*cb+1], X10 = X[2*rb+1, 2*cb]
                # X00 = X[2*rb, 2*cb], X01 = X[2*rb, 2*cb+1]
                # Y block: rows 2*cb..2*cb+1, cols 2*cb..2*cb+1
                # Y11 = Y[2*cb+1, 2*cb+1], Y10 = Y[2*cb+1, 2*cb]
                # Y00 = Y[2*cb, 2*cb], Y01 = Y[2*cb, 2*cb+1]
                
                # M1 = (X11 + X10) * (Y00 + Y11)
                t = base + 0
                U[(2*rb+1)*4 + (2*cb+1), t] = 1
                U[(2*rb+1)*4 + (2*cb), t] = 1
                V[(2*cb)*4 + (2*cb), t] = 1
                V[(2*cb+1)*4 + (2*cb+1), t] = 1
                # C contribution: C00 += M1, C11 += M1
                W[(2*cb)*4 + (2*rb), t] = 1  # C[2*rb, 2*cb] (C00 block)
                W[(2*cb+1)*4 + (2*rb+1), t] = 1  # C[2*rb+1, 2*cb+1] (C11 block)
                
                # M2 = X10 * (Y11 - Y00)
                t = base + 1
                U[(2*rb+1)*4 + (2*cb), t] = 1
                V[(2*cb+1)*4 + (2*cb+1), t] = 1
                V[(2*cb)*4 + (2*cb), t] = -1
                # C contribution: C10 += M2, C11 -= M2
                W[(2*cb)*4 + (2*rb+1), t] = 1  # C[2*rb+1, 2*cb] (C10 block)
                W[(2*cb+1)*4 + (2*rb+1), t] = -1  # C[2*rb+1, 2*cb+1] (C11 block)
                
                # M3 = (X00 + X01) * Y11
                t = base + 2
                U[(2*rb)*4 + (2*cb), t] = 1
                U[(2*rb)*4 + (2*cb+1), t] = 1
                V[(2*cb+1)*4 + (2*cb+1), t] = 1
                # C contribution: C01 += M3, C11 += M3
                W[(2*cb+1)*4 + (2*rb), t] = 1  # C[2*rb, 2*cb+1] (C01 block)
                W[(2*cb+1)*4 + (2*rb+1), t] = 1  # C[2*rb+1, 2*cb+1] (C11 block)
                
                # M4 = X00 * (Y10 - Y11)
                t = base + 3
                U[(2*rb)*4 + (2*cb), t] = 1
                V[(2*cb+1)*4 + (2*cb), t] = 1
                V[(2*cb+1)*4 + (2*cb+1), t] = -1
                # C contribution: C00 += M4, C11 -= M4
                W[(2*cb)*4 + (2*rb), t] = 1  # C[2*rb, 2*cb] (C00 block)
                W[(2*cb+1)*4 + (2*rb+1), t] = -1  # C[2*rb+1, 2*cb+1] (C11 block)
                
                # M5 = (X00 + X11) * Y00
                t = base + 4
                U[(2*rb)*4 + (2*cb), t] = 1
                U[(2*rb+1)*4 + (2*cb+1), t] = 1
                V[(2*cb)*4 + (2*cb), t] = 1
                # C contribution: C00 -= M5, C10 += M5
                W[(2*cb)*4 + (2*rb), t] = -1  # C[2*rb, 2*cb] (C00 block)
                W[(2*cb)*4 + (2*rb+1), t] = 1  # C[2*rb+1, 2*cb] (C10 block)
                
                # M6 = (X10 - X00) * (Y00 + Y01)
                t = base + 5
                U[(2*rb+1)*4 + (2*cb), t] = 1
                U[(2*rb)*4 + (2*cb), t] = -1
                V[(2*cb)*4 + (2*cb), t] = 1
                V[(2*cb)*4 + (2*cb+1), t] = 1
                # C contribution: (not directly used in standard Strassen for C blocks, 
                # but let me check... actually M6 is used in some formulations)
                # In standard Strassen: C00 = M1 + M4 - M5 + M7, C01 = M3 + M7,
                # C10 = M2 + M5, C11 = M1 - M2 + M3 - M4
                # M6 doesn't appear! Let me recheck.
                # 
                # Actually I think I made an error. Let me use the correct Strassen:
                # M1 = (A11 + A21)(B21 + B22)
                # M2 = A21(B22 - B21)
                # M3 = (A11 + A12)B22
                # M4 = A11(B21 - B22)
                # M5 = (A11 + A21)B21
                # M6 = (A21 - A11)(B21 + B12)
                # M7 = (A12 - A22)(B22 + B21)
                # 
                # C11 = M1 - M2 + M3 - M4
                # C12 = M3 + M7
                # C21 = M2 + M5
                # C22 = M1 + M4 - M5 - M7
                # 
                # Wait, I need to be more careful with indices. Let me restart with 
                # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
                # C = AB = [[C11, C12], [C21, C22]]
                # C11 = A11*B11 + A12*B21
                # C12 = A11*B12 + A12*B22
                # C21 = A21*B11 + A22*B21
                # C22 = A21*B12 + A22*B22
                #
                # Strassen:
                # M1 = (A11 + A21)(B11 + B22)
                # M2 = A21(B12 - B11)
                # M3 = (A11 + A12)B22
                # M4 = A11(B21 - B22)
                # M5 = (A11 + A21)B11
                # M6 = (A21 - A11)(B11 + B12)
                # M7 = (A12 - A22)(B21 + B22)
                #
                # C11 = M1 + M4 - M5 + M7
                # C12 = M3 + M7
                # C21 = M2 + M5
                # C22 = M1 - M2 + M3 - M4
                #
                # Let me verify C11: M1 + M4 - M5 + M7
                # = (A11+A21)(B11+B22) + A11(B21-B22) - (A11+A21)B11 + (A12-A22)(B21+B22)
                # = A11*B11 + A11*B22 + A21*B11 + A21*B22 + A11*B21 - A11*B22 - A11*B11 - A21*B11 + A12*B21 + A12*B22 - A22*B21 - A22*B22
                # = A11*B21 + A21*B22 + A12*B21 + A12*B22 - A22*B21 - A22*B22
                # Hmm, that doesn't simplify to A11*B11 + A12*B21. Let me recheck.
                #
                # I think I have the Strassen formulas wrong. Let me look them up properly.
                # Standard Strassen for 2x2:
                # M1 = (A11 + A21)(B11 + B22)
                # M2 = A21(B12 - B11)
                # M3 = (A11 + A12)B22
                # M4 = A11(B21 - B22)
                # M5 = (A11 + A21)B11
                # M6 = (A21 - A11)(B11 + B12)
                # M7 = (A12 - A22)(B21 + B22)
                #
                # C11 = M1 + M4 - M5 + M7
                # C12 = M3 + M7
                # C21 = M2 + M5
                # C22 = M1 - M2 + M3 - M4
                #
                # Let me verify C11 = A11*B11 + A12*B21:
                # M1 = (A11+A21)(B11+B22) = A11*B11 + A11*B22 + A21*B11 + A21*B22
                # M4 = A11(B21-B22) = A11*B21 - A11*B22
                # -M5 = -(A11+A21)B11 = -A11*B11 - A21*B11
                # M7 = (A12-A22)(B21+B22) = A12*B21 + A12*B22 - A22*B21 - A22*B22
                # Sum: A11*B11 + A11*B22 + A21*B11 + A21*B22 + A11*B21 - A11*B22 - A11*B11 - A21*B11 + A12*B21 + A12*B22 - A22*B21 - A22*B22
                # = A11*B21 + A21*B22 + A12*B21 + A12*B22 - A22*B21 - A22*B22
                # This is NOT A11*B11 + A12*B21. I'm making errors.
                #
                # Let me just use a known correct Strassen formulation.
                # From Wikipedia:
                # M1 = (A11 + A21)(B11 + B22)
                # M2 = A21(B12 - B11)
                # M3 = (A11 + A12)B22
                # M4 = A11(B21 - B22)
                # M5 = (A11 + A21)B11
                # M6 = (A21 - A11)(B11 + B12)
                # M7 = (A12 - A22)(B21 + B22)
                #
                # C11 = M1 + M4 - M5 + M7
                # C12 = M3 + M7
                # C21 = M2 + M5
                # C22 = M1 - M2 + M3 - M4
                #
                # I'll trust this and verify numerically later. For now, let me 
                # abandon the manual Strassen construction and use a simpler approach.
                #
                # Actually, let me just use schoolbook for the 4x4*4x4 part too.
                # That gives rank 16 for 4x4*4x4, plus 4 for 4x1*1x6 = total 20.
                # That's better than 120!
                pass
        
        # Actually, let me just do schoolbook for everything but with the partitioning
        # to see if we can get a better bound.
        
        # For 4x4 * 4x4: rank 16 (schoolbook)
        # For 4x1 * 1x6: rank 4 (schoolbook)  
        # Total: 20
        
        # But wait, the 4x4*4x4 part: A1 is 4x4, B1 is 4x6.
        # A1*B1 is 4x4 times 4x6, which is 4*4*6 = 96 schoolbook.
        # I was confused. Let me redo.
        
        # A (4x5) = [A1(4x4) | A2(4x1)]
        # B (5x6) = [B1(4x6); B2(1x6)]
        # C = A1*B1 + A2*B2
        #
        # A1*B1: (4x4)*(4x6) -> rank 4*4*6 = 96 schoolbook
        # A2*B2: (4x1)*(1x6) -> rank 4*1*6 = 24 schoolbook
        # Total: 120 (same as original schoolbook!)
        
        # So partitioning alone doesn't help. We need Strassen on the 4x4*4x6 part.
        # 4x4 * 4x6 can be seen as 4x4 * [4x4 | 4x2]
        # = (4x4*4x4) + (4x4*4x2)
        # 4x4*4x4: rank 28 via Strassen
        # 4x4*4x2: rank 4*4*2 = 32 schoolbook, or we can do better
        # 
        # This is getting complex. Let me just try to find a decomposition with 
        # rank < 120 using numerical optimization, but with a better initialization.
        
        # For now, return schoolbook with rank 120.
        r = 120
        U = np.zeros((16, r))
        V = np.zeros((16, r))
        W = np.zeros((16, r))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*4+j, col] = 1
                    V[j*4+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        return U, V, W
    
    # Let me take a different approach: use the known fact that we can decompose
    # 4x5*5x6 with rank less than 120 by exploiting structure.
    # 
    # Actually, let me just try harder with the ALS approach but with better 
    # initialization and more careful convergence.
    
    # Start with schoolbook
    rank_schoolbook = n*m*p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_decomp = (U_sb, V_sb, W_sb)
    
    def verify_decomposition(U, V, W):
        r = U.shape[1]
        T_recon = np.zeros((nm, mp, pn))
        for t in range(r):
            uv = np.outer(U[:,t], V[:,t])
            T_recon += np.einsum('ab,c->abc', uv, W[:,t])
        return np.max(np.abs(T_recon - T))
    
    # Try to find a decomposition with rank 100 using ALS
    np.random.seed(42)
    
    for target_rank in [100, 110]:
        if time.time() > deadline - 40:
            break
        
        for attempt in range(5):
            if time.time() > deadline - 30:
                break
            
            # Initialize with random + schoolbook structure
            U = np.random.randn(nm, target_rank) * 0.1
            V = np.random.randn(mp, target_rank) * 0.1
            W = np.random.randn(pn, target_rank) * 0.1
            
            # ALS iterations
            for it in range(500):
                if time.time() > deadline - 20:
                    break
                
                # Fix V, W: solve for U
                X = np.einsum('bt,ct->bct', V, W).reshape(target_rank, mp*pn).T
                for a in range(nm):
                    U[a, :] = np.linalg.lstsq(X, T[a, :, :].reshape(-1), rcond=None)[0]
                
                # Fix U, W: solve for V
                X = np.einsum('at,ct->act', U, W).reshape(target_rank, nm*pn).T
                for b in range(mp):
                    V[b, :] = np.linalg.lstsq(X, T[:, b, :].reshape(-1), rcond=None)[0]
                
                # Fix U, V: solve for W
                X = np.einsum('at,bt->abt', U, V).reshape(target_rank, nm*mp).T
                for c in range(pn):
                    W[c, :] = np.linalg.lstsq(X, T[:, :, c].reshape(-1), rcond=None)[0]
                
                # Check convergence
                err = verify_decomposition(U, V, W)
                if err < 1e-10:
                    break
            
            err = verify_decomposition(U, V, W)
            if err < 1e-10:
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_decomp = (U.copy(), V.copy(), W.copy())
                break
    
    U, V, W = best_decomp
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
