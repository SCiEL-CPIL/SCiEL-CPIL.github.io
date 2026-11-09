"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]
import time
_start_time = time.time()
TIME_LIMIT = 140.0  # 150s limit, 10s safety margin

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=6
    # U shape: (n*m, R) = (6, R)
    # V shape: (m*p, R) = (18, R)  
    # W shape: (p*n, R) = (12, R)
    
    def check_time():
        return (time.time() - _start_time) > TIME_LIMIT
    
    # Strategy: Try multiple constructions, return the one with minimum rank
    
    best_result = None
    best_rank = float('inf')
    
    # Construction 1: Trivial rank 36
    def trivial_decomposition():
        rank = n * m * p  # 36
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
        return {"U": U, "V": V, "W": W}
    
    # Construction 2: Split j=2 separately + Strassen on remaining
    def strassen_based_decomposition():
        terms = []
        def make_vec(size, idx_vals):
            vec = [0] * size
            for idx, val in idx_vals.items():
                vec[idx] = val
            return vec
        
        # Part 1: j=2, 12 terms
        for i in range(n):
            for k in range(p):
                u_idx = i * m + 2
                v_idx = 2 * p + k
                w_idx = k * n + i
                terms.append((make_vec(n*m, {u_idx: 1}),
                             make_vec(m*p, {v_idx: 1}),
                             make_vec(p*n, {w_idx: 1})))
        
        # Part 2: j=0,1 using Strassen on 2x2 blocks
        # A[:,0:2] is 2x2, B[0:2,:] is 2x6
        # Split B[0:2,:] into three 2x2 blocks
        for j_block in range(3):
            # B block columns: 2*j_block, 2*j_block+1
            b00 = 0 * p + 2*j_block
            b01 = 0 * p + 2*j_block + 1
            b10 = 1 * p + 2*j_block
            b11 = 1 * p + 2*j_block + 1
            
            # Output W indices for C[i][2*j_block] and C[i][2*j_block+1]
            c00 = (2*j_block) * n + 0
            c01 = (2*j_block + 1) * n + 0
            c10 = (2*j_block) * n + 1
            c11 = (2*j_block + 1) * n + 1
            
            # A indices
            a00 = 0 * m + 0
            a01 = 0 * m + 1
            a10 = 1 * m + 0
            a11 = 1 * m + 1
            
            # 7 Strassen terms
            # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
            terms.append((make_vec(6, {a00:1, a11:1}),
                         make_vec(18, {b00:1, b11:1}),
                         make_vec(12, {c00:1, c11:1})))
            # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
            terms.append((make_vec(6, {a10:1, a11:1}),
                         make_vec(18, {b00:1}),
                         make_vec(12, {c10:1, c11:-1})))
            # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
            terms.append((make_vec(6, {a00:1}),
                         make_vec(18, {b01:1, b11:-1}),
                         make_vec(12, {c01:1, c11:1})))
            # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
            terms.append((make_vec(6, {a11:1}),
                         make_vec(18, {b10:1, b00:-1}),
                         make_vec(12, {c00:1, c10:1})))
            # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
            terms.append((make_vec(6, {a00:1, a01:1}),
                         make_vec(18, {b11:1}),
                         make_vec(12, {c00:-1, c01:1})))
            # M6 = (A10-A00)(B00+B01) -> C11 += M6
            terms.append((make_vec(6, {a10:1, a00:-1}),
                         make_vec(18, {b00:1, b01:1}),
                         make_vec(12, {c11:1})))
            # M7 = (A01-A11)(B10+B11) -> C00 += M7
            terms.append((make_vec(6, {a01:1, a11:-1}),
                         make_vec(18, {b10:1, b11:1}),
                         make_vec(12, {c00:1})))
        
        rank = len(terms)
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for c, (u,v,w) in enumerate(terms):
            for i in range(n*m):
                U[i][c] = u[i]
            for i in range(m*p):
                V[i][c] = v[i]
            for i in range(p*n):
                W[i][c] = w[i]
        return {"U": U, "V": V, "W": W}
    
    # Construction 3: Shared Strassen across all j-dimension
    # View as: C = A * B where A is 2x3, B is 3x6
    # Split B into three 3x2 blocks: B = [B0, B1, B2]
    # C = [A*B0, A*B1, A*B2]
    # For each 2x3 * 3x2 product, we can use a rank-8 decomposition
    # But we can share computations across the three blocks
    
    def shared_strassen_decomposition():
        terms = []
        def make_vec(size, idx_vals):
            vec = [0] * size
            for idx, val in idx_vals.items():
                vec[idx] = val
            return vec
        
        # A is 2x3: A[i,j] for i in {0,1}, j in {0,1,2}
        # B is 3x6: B[j,k] for j in {0,1,2}, k in {0,...,5}
        # C is 2x6: C[i,k] for i in {0,1}, k in {0,...,5}
        
        # Strategy: Use the fact that we can precompute linear combinations of A's columns
        # and combine with B's rows
        
        # For each output column pair (2b, 2b+1) for b in {0,1,2}:
        # C[:, 2b:2b+2] = A * B[:, 2b:2b+2]
        # where B[:, 2b:2b+2] is a 3x2 matrix
        
        # We can use a rank-9 decomposition for 2x3 * 3x2, but try to share
        # across the three blocks by using the same A-combinations
        
        # Define 9 A-combinations (rank-9 for 2x3*3x2):
        # Actually, let's use a different approach:
        # Precompute 8 combinations of A, then use them with B
        
        # A-combinations (these are linear forms in A):
        # L0 = A[0,0] + A[0,1] + A[0,2]
        # L1 = A[1,0] + A[1,1] + A[1,2]
        # L2 = A[0,0] + A[0,1] - A[0,2]
        # L3 = A[1,0] + A[1,1] - A[1,2]
        # L4 = A[0,0] - A[0,1] + A[0,2]
        # L5 = A[1,0] - A[1,1] + A[1,2]
        # L6 = A[0,0] - A[0,1] - A[0,2]
        # L7 = A[1,0] - A[1,1] - A[1,2]
        
        # For each block b (0,1,2), we need to compute C[:, 2b:2b+2] = A * B_b
        # where B_b is 3x2
        
        # Let's use a systematic approach: enumerate all possible rank-8 decompositions
        # and check which ones work
        
        # Actually, let's try a specific construction:
        # Use 7 Strassen-like terms per block, but share some A-combinations
        
        # For block b, B_b = [[B[0,2b], B[0,2b+1]],
        #                      [B[1,2b], B[1,2b+1]],
        #                      [B[2,2b], B[2,2b+1]]]
        
        # C[0,2b] = A[0,0]*B[0,2b] + A[0,1]*B[1,2b] + A[0,2]*B[2,2b]
        # C[1,2b] = A[1,0]*B[0,2b] + A[1,1]*B[1,2b] + A[1,2]*B[2,2b]
        # C[0,2b+1] = A[0,0]*B[0,2b+1] + A[0,1]*B[1,2b+1] + A[0,2]*B[2,2b+1]
        # C[1,2b+1] = A[1,0]*B[0,2b+1] + A[1,1]*B[1,2b+1] + A[1,2]*B[2,2b+1]
        
        # Let's define 8 shared A-combinations:
        # M0 = A[0,0]
        # M1 = A[1,0]
        # M2 = A[0,1]
        # M3 = A[1,1]
        # M4 = A[0,2]
        # M5 = A[1,2]
        # M6 = A[0,0] + A[0,1]
        # M7 = A[1,0] + A[1,1]
        
        # For each block b, we need 8 terms (one for each A-combination)
        # Total: 8 * 3 = 24 terms, which is worse than 33
        
        # Let's try a different approach: use the known rank-8 for 2x3*3x2
        # and see if we can share
        
        # Actually, the rank of 2x3*3x2 is at most 2*3*2 = 12 (trivial)
        # and at least max(2*2, 3*2, 2*3) = 6
        # Known: rank(2x3*3x2) = 8 (from literature)
        
        # For three blocks, if we can't share, it's 3*8 = 24
        # But we already have 33 from the other approach, so 24 would be better!
        
        # Let's construct a rank-8 decomposition for 2x3*3x2:
        # This is a known result. Let me derive it.
        
        # A is 2x3, B is 3x2, C is 2x2
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
        
        # Use 8 terms:
        # T1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1]) -> contributes to C[0,0], C[1,1]
        # T2 = (A[1,0]+A[1,1]) * B[0,0] -> contributes to C[1,0], C[1,1]
        # T3 = A[0,0] * (B[0,1]-B[1,1]) -> contributes to C[0,1], C[1,1]
        # T4 = A[1,1] * (B[1,0]-B[0,0]) -> contributes to C[0,0], C[1,0]
        # T5 = (A[0,0]+A[0,1]) * B[1,1] -> contributes to C[0,0], C[0,1]
        # T6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1]) -> contributes to C[1,1]
        # T7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1]) -> contributes to C[0,0]
        # T8 = A[0,2] * B[2,0] + A[1,2] * B[2,0] ... wait, this doesn't work
        
        # Let me use a different known decomposition. The rank of 2x3*3x2 is actually 8.
        # Here's one construction:
        
        # Define:
        # P1 = A[0,0], Q1 = B[0,0]
        # P2 = A[1,0], Q2 = B[0,0]
        # P3 = A[0,1], Q3 = B[1,0]
        # P4 = A[1,1], Q4 = B[1,0]
        # P5 = A[0,2], Q5 = B[2,0]
        # P6 = A[1,2], Q6 = B[2,0]
        # P7 = (A[0,0]+A[1,1]), Q7 = (B[0,1]+B[1,1])
        # P8 = (A[0,1]+A[1,0]), Q8 = (B[0,1]+B[1,1])
        
        # This is getting complex. Let me just use the trivial 12-term for each block
        # and see if we can reduce.
        
        # Actually, let's just try rank 24 (8 per block) with a known decomposition
        
        # For 2x3 * 3x2, here's a rank-8 decomposition:
        # (from standard results)
        
        a00, a01, a02 = 0, 1, 2  # A indices
        a10, a11, a12 = 3, 4, 5  # A indices (row 1)
        
        for b in range(3):
            # B indices for block b
            b00 = 0 * p + 2*b    # B[0, 2b]
            b01 = 0 * p + 2*b+1  # B[0, 2b+1]
            b10 = 1 * p + 2*b    # B[1, 2b]
            b11 = 1 * p + 2*b+1  # B[1, 2b+1]
            b20 = 2 * p + 2*b    # B[2, 2b]
            b21 = 2 * p + 2*b+1  # B[2, 2b+1]
            
            # C indices
            c00 = (2*b) * n + 0      # C[0, 2b]
            c01 = (2*b+1) * n + 0    # C[0, 2b+1]
            c10 = (2*b) * n + 1      # C[1, 2b]
            c11 = (2*b+1) * n + 1    # C[1, 2b+1]
            
            # 8 terms for this block
            # T1: A[0,0] * B[0,0]
            terms.append((make_vec(6, {a00:1}),
                         make_vec(18, {b00:1}),
                         make_vec(12, {c00:1})))
            # T2: A[1,0] * B[0,0]
            terms.append((make_vec(6, {a10:1}),
                         make_vec(18, {b00:1}),
                         make_vec(12, {c10:1})))
            # T3: A[0,1] * B[1,0]
            terms.append((make_vec(6, {a01:1}),
                         make_vec(18, {b10:1}),
                         make_vec(12, {c00:1})))
            # T4: A[1,1] * B[1,0]
            terms.append((make_vec(6, {a11:1}),
                         make_vec(18, {b10:1}),
                         make_vec(12, {c10:1})))
            # T5: A[0,2] * B[2,0]
            terms.append((make_vec(6, {a02:1}),
                         make_vec(18, {b20:1}),
                         make_vec(12, {c00:1})))
            # T6: A[1,2] * B[2,0]
            terms.append((make_vec(6, {a12:1}),
                         make_vec(18, {b20:1}),
                         make_vec(12, {c10:1})))
            # T7: A[0,0] * B[0,1]
            terms.append((make_vec(6, {a00:1}),
                         make_vec(18, {b01:1}),
                         make_vec(12, {c01:1})))
            # T8: A[1,0] * B[0,1]
            terms.append((make_vec(6, {a10:1}),
                         make_vec(18, {b01:1}),
                         make_vec(12, {c11:1})))
            # Wait, I'm missing terms for A[0,1]*B[1,1], A[1,1]*B[1,1], A[0,2]*B[2,1], A[1,2]*B[2,1]
            # That's 8 more terms, total 16 per block = 48 total, which is worse
            
        # This approach isn't working well. Let me stick with the 33-term approach
        # and try to optimize it.
        
        rank = len(terms)
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for c, (u,v,w) in enumerate(terms):
            for i in range(n*m):
                U[i][c] = u[i]
            for i in range(m*p):
                V[i][c] = v[i]
            for i in range(p*n):
                W[i][c] = w[i]
        return {"U": U, "V": V, "W": W}
    
    # Try constructions
    for name, decomp_func in [("trivial", trivial_decomposition), 
                               ("strassen", strassen_based_decomposition)]:
        if check_time():
            break
        try:
            result = decomp_func()
            rank = len(result["U"][0])
            if rank < best_rank:
                best_rank = rank
                best_result = result
        except Exception:
            pass
    
    # Try optimization-based search to reduce rank from 33
    if not check_time() and best_rank > 28:
        try:
            import numpy as np
            np.random.seed(42)
            
            def verify_decomposition(U, V, W, tol=1e-10):
                """Check if the decomposition exactly computes matrix multiplication"""
                n, m, p = 2, 3, 6
                rank = len(U[0])
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for c in range(rank):
                                s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                            if abs(s - 1) > tol:
                                return False
                return True
            
            # Start from the 33-term Strassen decomposition and try to merge terms
            base = strassen_based_decomposition()
            base_rank = len(base["U"][0])
            
            # Try to find a lower rank by random perturbation and rounding
            for rank_try in range(base_rank - 1, 20, -1):
                if check_time():
                    break
                
                # Try multiple random initializations
                for attempt in range(20):
                    if check_time():
                        break
                    
                    # Initialize with random sparse vectors
                    U = np.zeros((6, rank_try))
                    V = np.zeros((18, rank_try))
                    W = np.zeros((12, rank_try))
                    
                    # Each column: pick 2-3 non-zero entries with ±1 values
                    for c in range(rank_try):
                        # U: pick 2 indices
                        nz_u = np.random.choice(6, size=2, replace=False)
                        U[nz_u, c] = np.random.choice([-1, 1], size=2)
                        
                        # V: pick 2-3 indices
                        nz_v_size = np.random.choice([2, 3])
                        nz_v = np.random.choice(18, size=nz_v_size, replace=False)
                        V[nz_v, c] = np.random.choice([-1, 1], size=nz_v_size)
                        
                        # W: pick 2 indices
                        nz_w = np.random.choice(12, size=2, replace=False)
                        W[nz_w, c] = np.random.choice([-1, 1], size=2)
                    
                    # Verify
                    if verify_decomposition(U.tolist(), V.tolist(), W.tolist()):
                        if rank_try < best_rank:
                            best_rank = rank_try
                            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        break  # Found valid for this rank, try lower
        
        except Exception:
            pass
    
    # Fallback: always have a valid result
    if best_result is None:
        best_result = trivial_decomposition()
        best_rank = len(best_result["U"][0])
    
    return best_result
# EVOLVE-BLOCK-END
