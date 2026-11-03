"""Matrix multiplication 4,4,6 rank-minimal decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_schoolbook(n, m, p):
    """Build the trivial schoolbook decomposition with rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, trials=3, seed=42):
    """Verify decomposition correctness."""
    rank = len(U[0])
    rng = np.random.RandomState(seed)
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    for _ in range(trials):
        A = rng.rand(n, m)
        B = rng.rand(m, p)
        expected = A @ B
        result = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    s += np.dot(Ua[i*m+j] * Va[j*p+k], Wa[k*n+i])
                result[i, k] = s
        if not np.allclose(result, expected, atol=1e-9):
            return False
    return True

def _strassen_4x4x4():
    """
    Construct a rank-28 decomposition for 4x4 x 4x4 using Strassen.
    Strassen: 7 block mults of 2x2, each 2x2x2 = 4 scalar mults.
    """
    n = m = p = 4
    # We'll build the decomposition by composing 2x2 Strassen with scalar 2x2x2.
    # 2x2x2 schoolbook has rank 4.
    # Strassen for 2x2x2 has rank 7.
    # For 4x4x4: split into 2x2 blocks.
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C11 = M1 - M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 - M2 + M6
    # where M1..M7 are 2x2 multiplications.
    
    # Each 2x2x2 multiplication uses 4 scalar multiplications (schoolbook).
    # Total: 7 * 4 = 28.
    
    # Let's build this explicitly.
    # Index convention: U row = i*m+j, V row = j*p+k, W row = k*n+i
    # For 4x4x4: n=m=p=4
    
    rank = 28
    U = [[0]*rank for _ in range(n*m)]  # 16 x 28
    V = [[0]*rank for _ in range(m*p)]  # 16 x 28
    W = [[0]*rank for _ in range(p*n)]  # 16 x 28
    
    # Strassen M factors for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A11+A12)(B21+B22)  -- wait, let me use standard Strassen
    
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 - M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 - M2 + M6
    
    # For each M, we need a 2x2x2 multiplication decomposition (rank 4 schoolbook).
    # M is computed as: C_M[i,k] = sum_j A_M[i,j] * B_M[j,k]
    # where A_M and B_M are 2x2 matrices (linear combos of blocks).
    
    # For each M factor, the 2x2x2 schoolbook uses 4 terms.
    # Term (a,b,c) in the 2x2x2: A_M[a,b] * B_M[b,c] contributes to C_M[a,c].
    
    # Let me define the Strassen factors.
    # For M1: A_M = A11+A22, B_M = B11+B22
    # For M2: A_M = A21+A22, B_M = B11
    # For M3: A_M = A11, B_M = B12-B22
    # For M4: A_M = A22, B_M = B21-B11
    # For M5: A_M = A11+A12, B_M = B22
    # For M6: A_M = A21-A11, B_M = B11+B12
    # For M7: A_M = A12-A22, B_M = B21+B22
    
    # A blocks: A11 = A[0:2,0:2], A12 = A[0:2,2:4], A21 = A[2:4,0:2], A22 = A[2:4,2:4]
    # B blocks: B11 = B[0:2,0:2], B12 = B[0:2,2:4], B21 = B[2:4,0:2], B22 = B[2:4,2:4]
    # C blocks: C11 = C[0:2,0:2], C12 = C[0:2,2:4], C21 = C[2:4,0:2], C22 = C[2:4,2:4]
    
    # A_M linear combos (coefficients for A[0,0], A[0,1], A[0,2], A[0,3], A[1,0], A[1,1], A[1,2], A[1,3], A[2,0], ...):
    # A11: rows 0,1 cols 0,1
    # A12: rows 0,1 cols 2,3
    # A21: rows 2,3 cols 0,1
    # A22: rows 2,3 cols 2,3
    
    # For each Strassen M, define A_M as a 2x2 matrix with entries that are linear combos of A.
    # Then for each of the 4 scalar mults in the 2x2x2 schoolbook for M,
    # we create 4 terms in the global decomposition.
    
    # Let me define helper: for a 2x2x2 schoolbook, term (a,b,c):
    # U[A_row(a,b)] = 1, V[B_row(b,c)] = 1, W[C_row(a,c)] = 1
    # where A_row(a,b) is the global row index for A[a,b] in the 2x2 block,
    # B_row(b,c) is the global row index for B[b,c],
    # C_row(a,c) is the global row index for C[a,c].
    
    # But A_M entries are linear combos, so U values will be the coefficients.
    
    # Let me define the Strassen M factors more carefully.
    # For M1: A_M[i,j] = A11[i,j] + A22[i,j], B_M[i,j] = B11[i,j] + B22[i,j]
    # A11[i,j] = A[2*0+i, 2*0+j] = A[i,j] for i,j in {0,1}
    # A22[i,j] = A[2*1+i, 2*1+j] = A[2+i, 2+j]
    # So A_M[i,j] = A[i,j] + A[2+i, 2+j]
    # B11[i,j] = B[i,j], B22[i,j] = B[2+i, 2+j]
    # B_M[i,j] = B[i,j] + B[2+i, 2+j]
    
    # For the global decomposition:
    # U row index for A[i,j] is i*4+j (since m=4)
    # V row index for B[j,k] is j*4+k
    # W row index for C[k,i] is k*4+i
    
    # For M1, term (a,b,c) in 2x2x2:
    # A_M[a,b] * B_M[b,c] contributes to C_M[a,c]
    # A_M[a,b] = A[a,b] + A[2+a,2+b]
    # B_M[b,c] = B[b,c] + B[2+b,2+c]
    # Product = A[a,b]*B[b,c] + A[a,b]*B[2+b,2+c] + A[2+a,2+b]*B[b,c] + A[2+a,2+b]*B[2+b,2+c]
    # This is 4 scalar multiplications, each contributing to C_M[a,c] = C[a,c] (since M1 contributes to C11 and C22)
    
    # Wait, this is getting complicated. Let me just build it directly.
    
    # For each Strassen M factor (7 total), and for each of the 4 scalar mults in 2x2x2 schoolbook,
    # we create 4 global terms (because each "scalar" in the 2x2 block is actually a sum of 2x2 = 4 elements).
    # No wait, that's not right. Each entry of A_M is a linear combo of 2 A entries, each entry of B_M is a linear combo of 2 B entries.
    # So A_M[a,b]*B_M[b,c] expands to 4 scalar products.
    # Each scalar product A[i,j]*B[j',k] is a separate term in the global decomposition.
    
    # Actually, I think the cleanest way is:
    # For each Strassen M, for each (a,b,c) in {0,1}^3:
    #   A_M[a,b] = sum of coeff * A[i,j]
    #   B_M[b,c] = sum of coeff * B[j,k]
    #   The product contributes to C_M[a,c] = sum of coeff * C[i,k]
    #   For each pair of (i,j) in A_M[a,b] and (j,k) in B_M[b,c]:
    #     Create a term: U[A_idx(i,j)] += coeff_A, V[B_idx(j,k)] += coeff_B, W[C_idx(k,i)] += coeff_C
    #     where coeff_A * coeff_B * coeff_C should give the right contribution.
    
    # Hmm, this is getting messy. Let me just use a simpler approach:
    # Build the 2x2x2 Strassen decomposition (rank 7) and then "expand" it to 4x4x4.
    
    # 2x2x2 Strassen:
    # U2, V2, W2 of shape (4,7), (4,7), (4,7) for 2x2x2.
    # Then for 4x4x4, each "element" of the 2x2 matrices is a 2x2 block.
    # So U4 has shape (16, 28), where each of the 4 rows of U2 corresponds to 4 rows of U4,
    # and each of the 7 columns of U2 corresponds to 4 columns of U4.
    
    # Actually, the expansion is: if U2[a,c] = u, then for the 4x4x4,
    # U4[2*block_i + a, 4*block_j + c'] = u for appropriate mappings.
    
    # This is getting too complex to do by hand. Let me just try a different approach:
    # Use a numerical search to find a low-rank decomposition.
    
    return None  # Placeholder

def _try_numerical_search(n, m, p, deadline):
    """Try to find a low-rank decomposition using numerical methods."""
    best = None
    best_rank = n * m * p
    
    # Try ranks from lower bound up
    lower_bound = max(n*m, m*p, p*n)
    
    for target_rank in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        
        # Try a few random initializations
        for seed in range(10):
            if time.time() > deadline:
                break
            rng = np.random.RandomState(seed + target_rank * 1000)
            
            U = rng.randn(n*m, target_rank) * 0.1
            V = rng.randn(m*p, target_rank) * 0.1
            W = rng.randn(p*n, target_rank) * 0.1
            
            # Try a few iterations of alternating minimization
            for it in range(50):
                if time.time() > deadline:
                    break
                
                # Fix V and W, optimize U
                # For each row i of U, we want to minimize error.
                # This is a least squares problem.
                
                # Actually, let's just check if the current solution is close to exact.
                # Compute error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(target_rank):
                                s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            err += (s - 1.0)**2
                
                if err < 1e-10:
                    # Found a solution!
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    if _verify(U_list, V_list, W_list, n, m, p, trials=5, seed=999):
                        best = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = target_rank
                        break
                
                # Alternating minimization: fix V, W, update U
                # For each (i,j), U[i*m+j, :] should satisfy:
                # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                # This is a linear system in U[i*m+j, :]
                
                # Build the linear system for U[i*m+j, :]
                # For each k: sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
                # Let M[k,c] = V[j*p+k,c] * W[k*n+i,c]
                # Then M @ U[i*m+j, :] = 1 (vector of ones)
                
                # Solve least squares
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((p, target_rank))
                        for k in range(p):
                            M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        # Solve M @ u = 1
                        try:
                            u, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                            U[i*m+j, :] = u
                        except:
                            pass
                
                # Similarly for V and W
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((n, target_rank))
                        for i in range(n):
                            M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        try:
                            v, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                            V[j*p+k, :] = v
                        except:
                            pass
                
                for k in range(p):
                    for i in range(n):
                        M = np.zeros((m, target_rank))
                        for j in range(m):
                            M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        try:
                            w, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                            W[k*n+i, :] = w
                        except:
                            pass
            
            if best is not None:
                break
        
        if best is not None:
            break
    
    return best

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Leave 20s margin
    
    # Start with schoolbook as baseline
    baseline = _build_schoolbook(n, m, p)
    
    # Try to find a lower rank decomposition
    result = _try_numerical_search(n, m, p, deadline)
    
    if result is not None:
        return result
    
    return baseline
# EVOLVE-BLOCK-END
