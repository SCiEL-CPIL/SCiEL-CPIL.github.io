"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    start_time = time.time()
    time_limit = 130  # leave margin before 150s kill
    
    def time_check():
        return (time.time() - start_time) < time_limit
    
    # Schoolbook decomposition (rank 63)
    rank_fb = n * m * p  # 63
    U_fb = np.zeros((n * m, rank_fb), dtype=float)
    V_fb = np.zeros((m * p, rank_fb), dtype=float)
    W_fb = np.zeros((p * n, rank_fb), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, c] = 1.0
                V_fb[j * p + k, c] = 1.0
                W_fb[k * n + i, c] = 1.0
                c += 1
    
    # Verify function
    def verify(U, V, W):
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(r))
                    if abs(actual - expected) > 1e-8:
                        return False
        # Check zero entries
        for i2 in range(n):
            for j2 in range(m):
                for k2 in range(p):
                    if i2 == i and j2 == j and k2 == k:
                        continue
                    expected2 = 0.0
                    actual2 = sum(U[i2*m+j2][c] * V[j2*p+k2][c] * W[k2*n+i2][c] for c in range(r))
                    if abs(actual2 - expected2) > 1e-8:
                        return False
        return True
    
    # Known construction for M_{3,3,1}: rank 7
    # C = A (3x3) times b (3x1): C_i = sum_j A_{ij} * b_j
    # This is a special case where we can use a known rank-7 decomposition
    
    def build_m331():
        """Build rank-7 decomposition for M_{3,3,1} (3x3 matrix times 3x1 vector)"""
        # For A(3x3) * b(3x1) -> C(3x1)
        # U: 9x7, V: 3x7, W: 3x7
        # Using known construction with entries in {0, 1, -1, 0.5}
        
        # One known rank-7 construction:
        # Terms:
        # 1. a11*b1 -> c1
        # 2. a22*b2 -> c2  
        # 3. a33*b3 -> c3
        # 4. (a12+a21)*b1 -> c1+c2
        # 5. (a13+a31)*b1 -> c1+c3
        # 6. (a23+a32)*b2 -> c2+c3
        # 7. (a12+a21+a13+a31+a23+a32)*b3 -> c1+c2+c3 (adjusted)
        
        # Actually, let me use a verified construction:
        # For computing c_i = sum_j a_ij b_j for i=0,1,2
        # With 9 inputs a_00..a_22 and 3 inputs b_0,b_1,b_2
        
        # Rank-7 construction (Strassen-type):
        U = np.zeros((9, 7))
        V = np.zeros((3, 7))
        W = np.zeros((3, 7))
        
        # Term 1: a00*b0 -> c0
        U[0, 0] = 1.0; V[0, 0] = 1.0; W[0, 0] = 1.0
        # Term 2: a11*b1 -> c1
        U[4, 1] = 1.0; V[1, 1] = 1.0; W[1, 1] = 1.0
        # Term 3: a22*b2 -> c2
        U[8, 2] = 1.0; V[2, 2] = 1.0; W[2, 2] = 1.0
        # Term 4: (a01+a10)*b0 -> c0+c1
        U[1, 3] = 1.0; U[3, 3] = 1.0; V[0, 3] = 1.0; W[0, 3] = 1.0; W[1, 3] = 1.0
        # Term 5: (a02+a20)*b0 -> c0+c2
        U[2, 4] = 1.0; U[6, 4] = 1.0; V[0, 4] = 1.0; W[0, 4] = 1.0; W[2, 4] = 1.0
        # Term 6: (a12+a21)*b1 -> c1+c2
        U[5, 5] = 1.0; U[7, 5] = 1.0; V[1, 5] = 1.0; W[1, 5] = 1.0; W[2, 5] = 1.0
        # Term 7: (a01+a10+a02+a20+a12+a21)*b2 -> c0+c1+c2
        U[1, 6] = 1.0; U[3, 6] = 1.0; U[2, 6] = 1.0; U[6, 6] = 1.0; U[5, 6] = 1.0; U[7, 6] = 1.0
        V[2, 6] = 1.0; W[0, 6] = 1.0; W[1, 6] = 1.0; W[2, 6] = 1.0
        
        # Verify this construction
        if not verify(U.tolist(), V.tolist(), W.tolist()):
            # Fallback: use schoolbook rank 9
            U_fb = np.zeros((9, 9))
            V_fb = np.zeros((3, 9))
            W_fb = np.zeros((3, 9))
            c = 0
            for i in range(3):
                for j in range(3):
                    U_fb[i*3+j, c] = 1.0
                    V_fb[j, c] = 1.0
                    W_fb[i, c] = 1.0
                    c += 1
            return U_fb, V_fb, W_fb, 9
        
        return U, V, W, 7
    
    # Build M_{3,3,3} using schoolbook (rank 27) for now
    def build_m333():
        """Build rank-27 schoolbook decomposition for M_{3,3,3}"""
        U = np.zeros((9, 27))
        V = np.zeros((9, 27))
        W = np.zeros((9, 27))
        c = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*3+j, c] = 1.0
                    V[j*3+k, c] = 1.0
                    W[k*3+i, c] = 1.0
                    c += 1
        return U, V, W, 27
    
    # Try to find a better M_{3,3,3} via random search
    def try_m333_random(max_rank=25, max_iters=100):
        """Try random search for M_{3,3,3} with rank <= max_rank"""
        import random
        random.seed(123)
        
        for r in range(max_rank, 26):
            for it in range(max_iters):
                if not time_check():
                    return None
                
                U = np.zeros((9, r))
                V = np.zeros((9, r))
                W = np.zeros((9, r))
                
                for col in range(r):
                    # Random sparse pattern
                    for idx in range(9):
                        if random.random() < 0.3:
                            U[idx, col] = random.choice([0.5, 1.0, -0.5, -1.0])
                        if random.random() < 0.3:
                            V[idx, col] = random.choice([0.5, 1.0, -0.5, -1.0])
                        if random.random() < 0.3:
                            W[idx, col] = random.choice([0.5, 1.0, -0.5, -1.0])
                
                # Quick verify
                valid = True
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            s = sum(U[i*3+j,c]*V[j*3+k,c]*W[k*3+i,c] for c in range(r))
                            if abs(s - 1.0) > 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    full_ok = True
                    for i2 in range(3):
                        for j2 in range(3):
                            for k2 in range(3):
                                if i2==i and j2==j and k2==k:
                                    continue
                                s2 = sum(U[i2*3+j2,c]*V[j2*3+k2,c]*W[k2*3+i2,c] for c in range(r))
                                if abs(s2) > 1e-6:
                                    full_ok = False
                                    break
                            if not full_ok:
                                break
                        if not full_ok:
                            break
                    if full_ok:
                        return U, V, W, r
            if not time_check():
                return None
        return None
    
    # Strategy: Split p=7 into 3+3+1
    # M_{3,3,7} = M_{3,3,3} + M_{3,3,3} + M_{3,3,1}
    # Use best available constructions for each part
    
    # Build M_{3,3,1}
    U1, V1, W1, r1 = build_m331()
    
    # Try to find better M_{3,3,3}
    m333_result = try_m333_random(max_rank=24, max_iters=50)
    if m333_result is not None:
        U2, V2, W2, r2 = m333_result
    else:
        U2, V2, W2, r2 = build_m333()
    
    # Build full M_{3,3,7} by combining
    # For M_{3,3,7}: U is (3*3)=9 x rank, V is (3*7)=21 x rank, W is (7*3)=21 x rank
    # 
    # Part 1 (k=0,1,2): uses U1 (9xr1), V1[:, 0:3] (3xr1), W1[:, 0:3] (3xr1)
    #   But V1 has 3 rows (for b_0,b_1,b_2), and in full problem V has 21 rows (j=0,1,2; k=0..6)
    #   So V1 row j corresponds to full V row j*7+k for k=0,1,2
    #   W1 row i corresponds to full W row k*3+i for k=0,1,2
    #
    # Let me redo this more carefully.
    #
    # Full problem: A is 3x3 (rows i=0,1,2; cols j=0,1,2), B is 3x7 (rows j=0,1,2; cols k=0..6)
    # Output C is 3x7 (rows i=0,1,2; cols k=0..6)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Tensor entries: T[i,j,k] = 1 for all i,j,k (we need to compute all products)
    # Decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    #
    # U: (n*m) x rank = 9 x rank, indexed by (i,j)
    # V: (m*p) x rank = 21 x rank, indexed by (j,k)
    # W: (p*n) x rank = 21 x rank, indexed by (k,i)
    
    # For M_{3,3,1} part (k=0,1,2 only, but this isn't quite right...)
    # Actually, splitting by k doesn't work directly because A is shared.
    
    # Better approach: Think of it as computing C = A * B where B has 7 columns.
    # Split B into B1 (cols 0-2) and B2 (cols 3-5) and B3 (col 6).
    # C1 = A * B1 (3x3 result), C2 = A * B2 (3x3 result), C3 = A * B3 (3x1 result)
    # Total rank = rank(M_{3,3,3}) + rank(M_{3,3,3}) + rank(M_{3,3,1})
    
    # Build full decomposition
    total_rank = r1 + r2 + r2  # r1 for M_{3,3,1}, two copies of r2 for M_{3,3,3}
    
    U_full = np.zeros((9, total_rank))
    V_full = np.zeros((21, total_rank))
    W_full = np.zeros((21, total_rank))
    
    # Part 1: k=0,1,2 (M_{3,3,3})
    for c in range(r2):
        for i in range(3):
            for j in range(3):
                U_full[i*3+j, c] = U2[i*3+j, c]
        for j in range(3):
            for k in range(3):
                V_full[j*7+k, c] = V2[j*3+k, c]
        for k in range(3):
            for i in range(3):
                W_full[k*3+i, c] = W2[k*3+i, c]
    
    # Part 2: k=3,4,5 (M_{3,3,3})
    offset = r2
    for c in range(r2):
        for i in range(3):
            for j in range(3):
                U_full[i*3+j, offset+c] = U2[i*3+j, c]
        for j in range(3):
            for k in range(3):
                V_full[j*7+(k+3), offset+c] = V2[j*3+k, c]
        for k in range(3):
            for i in range(3):
                W_full[(k+3)*3+i, offset+c] = W2[k*3+i, c]
    
    # Part 3: k=6 (M_{3,3,1})
    offset = 2 * r2
    for c in range(r1):
        for i in range(3):
            for j in range(3):
                U_full[i*3+j, offset+c] = U1[i*3+j, c]
        for j in range(3):
            V_full[j*7+6, offset+c] = V1[j, c]
        for i in range(3):
            W_full[6*3+i, offset+c] = W1[i, c]
    
    # Verify full decomposition
    if verify(U_full.tolist(), V_full.tolist(), W_full.tolist()):
        rank_found = total_rank
        U_final = U_full.tolist()
        V_final = V_full.tolist()
        W_final = W_full.tolist()
        
        # Convert to half-integers if possible
        def to_half_integer(arr):
            result = []
            for row in arr:
                new_row = []
                for x in row:
                    if abs(x * 2 - round(x * 2)) < 1e-9:
                        new_row.append(round(x * 2) / 2.0)
                    else:
                        new_row.append(float(x))
                result.append(new_row)
            return result
        
        return {"U": to_half_integer(U_final), "V": to_half_integer(V_final), "W": to_half_integer(W_final)}
    
    # If verification failed, try a more direct approach: random search for full M_{3,3,7}
    import random
    random.seed(42)
    
    best_result = None
    for r in range(40, 63):
        if not time_check():
            break
        max_iters = max(20, 500 // (63-r))
        for it in range(max_iters):
            if not time_check():
                break
            
            U_r = np.zeros((9, r))
            V_r = np.zeros((21, r))
            W_r = np.zeros((21, r))
            
            for col in range(r):
                # Structured random: each column covers a small set of (i,j,k)
                ni = 1 + (random.random() < 0.4)
                nj = 1 + (random.random() < 0.4)
                nk = 1 + (random.random() < 0.3)
                
                idx_i = [random.randint(0,2)] if ni==1 else random.sample(range(3),2)
                idx_j = [random.randint(0,2)] if nj==1 else random.sample(range(3),2)
                idx_k = [random.randint(0,6)] if nk==1 else random.sample(range(7),2)
                
                for ii in idx_i:
                    for jj in idx_j:
                        U_r[ii*3+jj, col] = random.choice([0.5, 1.0, -0.5, -1.0])
                for jj in idx_j:
                    for kk in idx_k:
                        V_r[jj*7+kk, col] = random.choice([0.5, 1.0, -0.5, -1.0])
                for kk in idx_k:
                    for ii in idx_i:
                        W_r[kk*3+ii, col] = random.choice([0.5, 1.0, -0.5, -1.0])
            
            # Quick verify
            valid = True
            for i in range(3):
                for j in range(3):
                    for k in range(7):
                        s = sum(U_r[i*3+j,c]*V_r[j*7+k,c]*W_r[k*3+i,c] for c in range(r))
                        if abs(s-1.0) > 1e-6:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            if valid:
                full_ok = True
                for i2 in range(3):
                    for j2 in range(3):
                        for k2 in range(7):
                            if i2==i and j2==j and k2==k: continue
                            s2 = sum(U_r[i2*3+j2,c]*V_r[j2*7+k2,c]*W_r[k2*3+i2,c] for c in range(r))
                            if abs(s2) > 1e-6:
                                full_ok = False
                                break
                        if not full_ok: break
                    if not full_ok: break
                if full_ok:
                    best_result = (U_r, V_r, W_r, r)
                    break
            if best_result is not None:
                break
        
        if best_result is not None:
            break
    
    if best_result is not None:
        U_found, V_found, W_found, r_found = best_result
        U_final = U_found.tolist()
        V_final = V_found.tolist()
        W_final = W_found.tolist()
        
        def to_half_integer(arr):
            result = []
            for row in arr:
                new_row = []
                for x in row:
                    if abs(x * 2 - round(x * 2)) < 1e-9:
                        new_row.append(round(x * 2) / 2.0)
                    else:
                        new_row.append(float(x))
                result.append(new_row)
            return result
        
        return {"U": to_half_integer(U_final), "V": to_half_integer(V_final), "W": to_half_integer(W_final)}
    
    # Fallback to schoolbook
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

# EVOLVE-BLOCK-END
