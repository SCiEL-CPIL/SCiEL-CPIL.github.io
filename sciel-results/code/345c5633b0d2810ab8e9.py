import numpy as np
import time
import itertools

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _build_strassen_222():
    """
    Returns (U, V, W) for 2x2x2 matrix multiplication using Strassen's algorithm.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Convention: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    # Strassen constants
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # A indices: A11=0, A12=1, A21=2, A22=3
    # B indices: B11=0, B12=1, B21=2, B22=3
    # C indices: C11=0, C12=1, C21=2, C22=3
    
    # We need to define U and V such that for each intermediate product M_t,
    # U[:, t] and V[:, t] represent the linear combinations.
    # The product is sum_t U[a, t] V[b, t] M_t? No.
    # The decomposition is C = sum_t (U[:, t] * A) * (V[:, t] * B) ? No.
    # Tensor decomposition: T_{ijk} = sum_t U_{it} V_{jt} W_{kt}
    # Here i is row of A (0,1), j is col of A/row of B (0,1), k is col of B (0,1).
    # Wait, standard notation:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # Tensor T_{ijk} where i is A row, j is A col/B row, k is B col.
    # U is (n*m, r) -> (4, 7). Index i*m+j.
    # V is (m*p, r) -> (4, 7). Index j*p+k.
    # W is (p*n, r) -> (4, 7). Index k*n+i.
    
    # Let's map Strassen terms to this.
    # M1 = (A11+A22)(B11+B22) -> contributes to C11, C12, C21, C22?
    # Actually, let's just use the known rank-7 decomposition.
    
    # U rows: A11(0), A12(1), A21(2), A22(3)
    # V rows: B11(0), B12(1), B21(2), B22(3)
    # W rows: C11(0), C12(1), C21(2), C22(3)
    
    # Term 1: M1 = (A11+A22)(B11+B22)
    # U1: [1, 0, 0, 1]
    # V1: [1, 0, 0, 1]
    # W1: [1, 1, 1, 1] -> C11 += M1, C12 += M1, C21 += M1, C22 += M1
    # Wait, Strassen:
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M7
    # C21 = M2 - M5 + M6
    # C22 = M1 - M2 + M3 + M4
    
    # Let's build U, V, W columns corresponding to M1..M7.
    
    # M1 = (A11+A22)(B11+B22)
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    w1 = [1, 0, 0, 1] # C11 gets M1. C12, C21, C22?
    # Let's re-verify Strassen formulas carefully.
    # C = [[M1 - M4 + M5 - M7, M3 + M7], [M2 - M5 + M6, M1 - M2 + M3 + M4]]
    
    # So:
    # W for M1: [1, 0, 0, 1] (C11, C22)
    # W for M2: [0, 0, 1, -1] (C21, C22)
    # W for M3: [0, 1, 0, 1] (C12, C22)
    # W for M4: [-1, 0, 0, 1] (C11, C22)
    # W for M5: [1, 0, -1, 0] (C11, C21)
    # W for M6: [0, 0, 1, 0] (C21)
    # W for M7: [-1, 1, 0, 0] (C11, C12)
    
    # U vectors:
    # M1: A11+A22 -> [1, 0, 0, 1]
    # M2: (A21+A22)B11 -> A part: [0, 0, 1, 1]
    # M3: A11(B12-B22) -> A part: [1, 0, 0, 0]
    # M4: A22(B21-B11) -> A part: [0, 0, 0, 1]
    # M5: (A11+A12)B22 -> A part: [1, 1, 0, 0]
    # M6: (A21-A11)(B11+B12) -> A part: [-1, 0, 1, 0]
    # M7: (A12-A22)(B21+B22) -> A part: [0, 1, 0, -1]
    
    # V vectors:
    # M1: B11+B22 -> [1, 0, 0, 1]
    # M2: B11 -> [1, 0, 0, 0]
    # M3: B12-B22 -> [0, 1, 0, -1]
    # M4: B21-B11 -> [-1, 0, 1, 0]
    # M5: B22 -> [0, 0, 0, 1]
    # M6: B11+B12 -> [1, 1, 0, 0]
    # M7: B21+B22 -> [0, 0, 1, 1]
    
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0], # A11
        [0, 0, 0, 0, 1, 0, 1],  # A12
        [0, 1, 0, 0, 0, 1, 0],  # A21
        [1, 1, 0, 1, 0, 0, -1]  # A22
    ], dtype=float)
    
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0], # B11
        [0, 0, 1, 0, 0, 1, 0],  # B12
        [0, 0, 0, 1, 0, 0, 1],  # B21
        [1, 0, -1, 0, 1, 0, 1]  # B22
    ], dtype=float)
    
    W = np.array([
        [1, 0, 0, -1, 1, 0, -1], # C11
        [0, 0, 1, 0, 0, 0, 1],   # C12
        [0, 1, 0, 0, -1, 1, 0],  # C21
        [1, -1, 1, 1, 0, 0, 0]   # C22
    ], dtype=float)
    
    return U, V, W

def _embed_strassen(n, m, p):
    """
    Embeds 2x2x2 Strassen into n x m x p.
    We can partition A (n x m), B (m x p) into blocks.
    Since n=2, we can split A into two 1x5 blocks? No, Strassen is 2x2.
    We can try to cover the 2x5x6 multiplication with blocks of size 2x2x2.
    However, 5 and 6 are not multiples of 2.
    
    Alternative: Use the 2x2x2 decomposition for the "core" and handle boundaries?
    Or, use a known higher rank decomposition for 2x5x6.
    
    Actually, we can try to construct a rank-30 decomposition using the structure.
    Lower bound for 2x5x6:
    R(2,5,6) >= max(10, 12, 30) ? No.
    R(n,m,p) >= nm, mp, np.
    R(2,5,6) >= 10, 30, 12. So >= 30.
    
    If we can find a rank-30 decomposition, that would be optimal.
    One way is to use the "border rank" or specific algebraic structures.
    However, finding it is hard.
    
    Let's try a different approach:
    Use the fact that R(2, n, m) <= 2nm - 2n - 2m + 4? No.
    
    Let's stick to the search strategy but improve it.
    The previous search used random U, V with +/-1 and solved for W.
    This is a "fix U, V, solve W" strategy.
    The constraint is that W must be half-integers.
    
    Improvement:
    1. Try smaller ranks first (30).
    2. Use a smarter initialization for U and V.
       Instead of random +/-1, maybe use structured matrices?
       Or, use the Strassen 2x2x2 as a building block for a larger decomposition?
       
    Let's try to construct a rank-30 decomposition by combining 2x2x2 blocks.
    We have A (2x5), B (5x6).
    We can view A as [A1 A2] where A1 is 2x2, A2 is 2x3.
    B as [B1; B2] where B1 is 2x6, B2 is 3x6.
    C = A1 B1 + A2 B2.
    
    This doesn't directly help with rank reduction unless we can share terms.
    
    Let's go back to the search but make it more efficient.
    We can fix U to be a random matrix with +/-1 entries.
    Then for each (i, k), we solve for W_ik.
    We require W_ik to be half-integers.
    
    To increase the chance of finding a solution, we can:
    1. Iterate over ranks 30, 31, ...
    2. For each rank, try many random U, V.
    3. When solving for W, check if the solution is close to half-integers.
    4. If yes, snap to half-integers and verify.
    
    To speed up:
    - Use `np.linalg.lstsq` efficiently.
    - Early exit if residuals are too large or W is not close to half-integers.
    
    Let's implement this robustly.
    """
    pass

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    min_rank = 30
    max_rank = n * m * p  # 60
    
    # Pre-compute indices for V rows
    V_rows = [[j * p + k for j in range(m)] for k in range(p)]
    ones_m = np.ones(m)
    
    for r in range(min_rank, max_rank + 1):
        if time.time() > deadline:
            break
            
        # Try multiple random initializations for U and V
        # We want U and V to be "generic" enough to allow a solution W
        # but structured enough to yield half-integer W.
        # Using entries from {-1, 1} is a good start.
        
        # Adjust trials based on rank: lower ranks are harder, try more
        # But also, lower ranks are more likely to be infeasible.
        # Let's try 200 trials for r=30, decreasing for higher r?
        # Actually, if r=30 is feasible, it's the best.
        # Let's give r=30 a good chance.
        num_trials = 1000 if r == 30 else 500
        
        for trial in range(num_trials):
            if time.time() > deadline:
                break
                
            # Generate random U and V with entries -1, 1
            # Using bit operations for speed
            U = np.random.choice([-1, 1], size=(n * m, r))
            V = np.random.choice([-1, 1], size=(m * p, r))
            
            W = np.zeros((p * n, r))
            valid = True
            
            # Solve for W for each (i, k) pair
            for i in range(n):
                for k in range(p):
                    # Construct M_ik of shape (m, r)
                    # M_ik[j, alpha] = U[i*m + j, alpha] * V[j*p + k, alpha]
                    U_part = U[i*m : (i+1)*m, :]
                    V_part = V[V_rows[k], :]
                    
                    M_ik = U_part * V_part
                    # Solve M_ik @ w = ones(m)
                    # Use least squares
                    try:
                        w, residuals, rank_m, s = np.linalg.lstsq(M_ik, ones_m, rcond=None)
                    except:
                        valid = False
                        break
                    
                    # Check if w is close to half-integers
                    # 2*w should be close to integers
                    two_w = 2 * w
                    rounded_two_w = np.round(two_w)
                    if not np.allclose(two_w, rounded_two_w, atol=1e-8):
                        valid = False
                        break
                    
                    W[k * n + i, :] = w
            
            if valid:
                # Final verification
                # Reconstruct C and check
                # C_ik = sum_j A_ij B_jk
                # Our decomposition: sum_t U_it V_jt W_kt ?
                # Wait, the tensor is T_{ijk} = sum_t U_{i,j,t} V_{j,k,t} W_{k,i,t}
                # Here U is (nm, r), V is (mp, r), W is (pn, r).
                # Index of U: i*m + j. Index of V: j*p + k. Index of W: k*n + i.
                # Let's verify one random case.
                
                # Create random A, B
                A = np.random.randint(0, 10, size=(n, m))
                B = np.random.randint(0, 10, size=(m, p))
                C_true = A @ B
                
                C_calc = np.zeros((n, p))
                for i in range(n):
                    for k in range(p):
                        for t in range(r):
                            s = 0
                            for j in range(m):
                                s += U[i*m+j][t] * V[j*p+k][t]
                            C_calc[i, k] += s * W[k*n+i][t]
                            
                if not np.allclose(C_true, C_calc, atol=1e-8):
                    valid = False
                    # This should not happen if W was solved correctly,
                    # but snapping to half-integers might introduce error
                    # if the tolerance was too loose.
                    # If it fails, we can try to re-solve with the snapped W?
                    # Or just discard this trial.
                    continue
                
                if valid:
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
        
    # Fallback to schoolbook if no better rank found in time
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
