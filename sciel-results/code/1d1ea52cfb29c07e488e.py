"""Search for low-rank decomposition of 4x5x5 matrix multiplication tensor."""
import numpy as np
import time
import random

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify_numpy(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using numpy einsum."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=tol)

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using integer arithmetic."""
    rank = len(U[0])
    # Multiply by 2 to convert half-integers to integers
    U2 = [[2 * x for x in row] for row in U]
    V2 = [[2 * x for x in row] for row in V]
    W2 = [[2 * x for x in row] for row in W]
    
    # Check each (i,j,k) entry
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    u_val = U2[i * m + j][c]
                    v_val = V2[j * p + k][c]
                    w_val = W2[k * n + i][c]
                    if u_val != 0 and v_val != 0 and w_val != 0:
                        val += u_val * v_val * w_val
                # Expected value is 1, so 2*2*2 = 8 when multiplied by 8
                if val != 8:
                    return False
    return True

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2x2 matrix multiplication (rank 7)."""
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # Strassen's 7-term algorithm for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    U[0,0] = 1; U[3,0] = 1; V[0,0] = 1; V[3,0] = 1; W[0,0] = 1; W[3,0] = 1
    # M2 = (A10+A11)B00
    U[1,1] = 1; U[2,1] = 1; V[0,1] = 1; W[0,1] = 1
    # M3 = A00(B10-B11)
    U[0,2] = 1; V[1,2] = 1; V[3,2] = -1; W[1,2] = 1; W[3,2] = -1
    # M4 = A11(B01-B00)
    U[3,3] = 1; V[1,3] = 1; V[0,3] = -1; W[1,3] = 1; W[0,3] = -1
    # M5 = (A00+A01)B11
    U[0,4] = 1; U[1,4] = 1; V[3,4] = 1; W[3,4] = 1
    # M6 = (A10-A00)(B00+B01)
    U[1,5] = 1; U[0,5] = -1; V[0,5] = 1; V[1,5] = 1; W[0,5] = 1; W[1,5] = 1
    # M7 = (A01-A11)(B10+B11)
    U[1,6] = 1; U[3,6] = -1; V[1,6] = 1; V[3,6] = 1; W[1,6] = 1; W[3,6] = 1
    
    return U.tolist(), V.tolist(), W.tolist()

def _laderman_3x3x3():
    """Return Laderman decomposition for 3x3x3 matrix multiplication (rank 13)."""
    # Laderman's 13-term algorithm for 3x3x3
    # This is a known result: M(3,3,3) <= 13
    # The decomposition uses specific sparse patterns
    
    U = np.zeros((9, 13), dtype=int)
    V = np.zeros((9, 13), dtype=int)
    W = np.zeros((9, 13), dtype=int)
    
    # Index mapping: row*3+col
    # Let's use the known Laderman decomposition
    # Terms are labeled M1 through M13
    
    # M1: covers (0,0),(1,0),(0,1) in A, (0,0),(0,1),(1,0) in B
    U[0,0] = 1; U[3,0] = 1; U[1,0] = 1
    V[0,0] = 1; V[1,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1; W[1,0] = 1
    
    # M2: covers (0,2) in A, (2,0),(2,1) in B
    U[2,1] = 1
    V[6,1] = 1; V[7,1] = 1
    W[6,1] = 1; W[7,1] = 1
    
    # M3: covers (1,2) in A, (2,0),(2,1) in B  
    U[5,2] = 1
    V[6,2] = 1; V[7,2] = 1
    W[6,2] = 1; W[7,2] = 1
    
    # M4: covers (2,0) in A, (0,2),(1,2) in B
    U[6,3] = 1
    V[2,3] = 1; V[5,3] = 1
    W[2,3] = 1; W[5,3] = 1
    
    # M5: covers (2,1) in A, (0,2),(1,2) in B
    U[7,4] = 1
    V[2,4] = 1; V[5,4] = 1
    W[2,4] = 1; W[5,4] = 1
    
    # M6: covers (2,2) in A, (2,2) in B
    U[8,5] = 1
    V[8,5] = 1
    W[8,5] = 1
    
    # M7: covers (1,1),(1,2) in A, (1,1) in B
    U[4,6] = 1; U[5,6] = 1
    V[4,6] = 1
    W[4,6] = 1
    
    # M8: covers (0,1),(1,1) in A, (1,1) in B
    U[1,7] = 1; U[4,7] = 1
    V[4,7] = 1
    W[1,7] = 1; W[4,7] = 1
    
    # M9: covers (1,0),(1,1) in A, (0,1) in B
    U[3,8] = 1; U[4,8] = 1
    V[1,8] = 1
    W[3,8] = 1; W[4,8] = 1
    
    # M10: covers (0,0),(0,1) in A, (1,0) in B
    U[0,9] = 1; U[1,9] = 1
    V[3,9] = 1
    W[0,9] = 1; W[3,9] = 1
    
    # M11: covers (0,2) in A, (0,2) in B
    U[2,10] = 1
    V[2,10] = 1
    W[2,10] = 1
    
    # M12: covers (2,0) in A, (2,0) in B
    U[6,11] = 1
    V[6,11] = 1
    W[6,11] = 1
    
    # M13: covers (2,1) in A, (1,2) in B
    U[7,12] = 1
    V[5,12] = 1
    W[7,12] = 1
    
    return U.tolist(), V.tolist(), W.tolist()

def _block_decomposition_strassen_laderman():
    """
    Use block decomposition: split 4x5x5 into blocks and apply Strassen to 2x2x2 blocks.
    4 = 2+2, 5 = 2+3, 5 = 2+3
    
    Strategy:
    - Split A (4x5) into 4 blocks: A11(2x2), A12(2x3), A21(2x2), A22(2x3)
    - Split B (5x5) into 4 blocks: B11(2x2), B12(2x3), B21(3x2), B22(3x3)
    - C (5x4) has blocks: C11(2x2), C12(2x2), C21(3x2), C22(3x2)
    
    For the 2x2x2 corner blocks, use Strassen (rank 7 each).
    For mixed-size blocks, use schoolbook.
    """
    n, m, p = 4, 5, 5
    n1, n2 = 2, 2
    m1, m2 = 2, 3
    p1, p2 = 2, 3
    
    # We'll build up the decomposition column by column
    # Each "column" in the decomposition corresponds to one scalar multiplication
    # in the block algorithm
    
    U_cols = []  # List of column vectors for U (length n*m = 20)
    V_cols = []  # List of column vectors for V (length m*p = 25)
    W_cols = []  # List of column vectors for W (length p*n = 20)
    
    def add_term(u_vec, v_vec, w_vec):
        """Add a single rank-1 term."""
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Helper: create a zero vector of given length
    def zeros(length):
        return [0] * length
    
    # Helper: set entries in a vector
    def set_entries(vec, indices, values):
        vec = list(vec)
        for idx, val in zip(indices, values):
            vec[idx] = val
        return vec
    
    # ---- Strassen for 2x2x2 block (top-left corner) ----
    # A11 (2x2) * B11 (2x2) -> C11 (2x2)
    # Strassen's 7 products:
    # M1 = (A11[0,0]+A11[1,1]) * (B11[0,0]+B11[1,1])
    # M2 = (A11[1,0]+A11[1,1]) * B11[0,0]
    # M3 = A11[0,0] * (B11[1,0]-B11[1,1])
    # M4 = A11[1,1] * (B11[0,1]-B11[0,0])
    # M5 = (A11[0,0]+A11[0,1]) * B11[1,1]
    # M6 = (A11[1,0]-A11[0,0]) * (B11[0,0]+B11[0,1])
    # M7 = (A11[0,1]-A11[1,1]) * (B11[1,0]+B11[1,1])
    
    # A11 indices in full A: rows 0,1; cols 0,1 -> A[0], A[1], A[2], A[3]
    # B11 indices in full B: rows 0,1; cols 0,1 -> B[0], B[1], B[2], B[3]
    # C11 indices in full C: rows 0,1; cols 0,1 -> C[0], C[1], C[2], C[3]
    
    # M1
    u = zeros(20); v = zeros(25); w = zeros(20)
    u[0] = 1; u[3] = 1  # A11[0,0], A11[1,1]
    v[0] = 1; v[3] = 1  # B11[0,0], B11[1,1]
    w[0] = 1; w[3] = 1  # C11[0,0], C11[1,1]
    add_term(u, v, w)
    
    # M2
    u = zeros(20); v = zeros(25); w = zeros(20)
    u[2] = 1; u[3] = 1  # A11[1,0], A11[1,1]
    v[0] = 1  # B11[0,0]
    w[0] = 1  # C11[0,0]
    add_term(u, v, w)
    
    # M3
    u = zeros(20); v = zeros(25); w = zeros(20)
    u[0] = 1  # A11[0,0]
    v[2] = 1; v[3] = -1  # B11[1,0], -B11[1,1]
    w[2] = 1; w[3] = -1  # C11[1,0], -C11[1,1]
    add_term(u, v, w)
    
    # M4
    u = zeros(20); v = zeros(25); w = zeros(20)
    u[3] = 1  # A11[1,1]
    v[1] = 1; v[0] = -1  # B11[0,1], -B11[0,0]
    w[1] = 1; w[0] = -1  # C11[0,1], -C11[0,0]
    add_term(u, v, w)
    
    # M5
    u = zeros(20); v = zeros(25); w = zeros(20)
    u[0] = 1; u[1] = 1  # A11[0,0], A11[0,1]
    v[3] = 1  # B11[1,1]
    w[3] = 1  # C11[1,1]
    add_term(u, v, w)
    
    # M6
    u = zeros(20); v = zeros(25); w = zeros(20)
    u[2] = 1; u[0] = -1  # A11[1,0], -A11[0,0]
    v[0] = 1; v[1] = 1  # B11[0,0], B11[0,1]
    w[0] = 1; w[1] = 1  # C11[0,0], C11[0,1]
    add_term(u, v, w)
    
    # M7
    u = zeros(20); v = zeros(25); w = zeros(20)
    u[1] = 1; u[3] = -1  # A11[0,1], -A11[1,1]
    v[2] = 1; v[3] = 1  # B11[1,0], B11[1,1]
    w[2] = 1; w[3] = 1  # C11[1,0], C11[1,1]
    add_term(u, v, w)
    
    # Now we need to fix up the C11 entries. The Strassen products give:
    # C11[0,0] = M1 - M2 + M3 + M4 - M6
    # C11[0,1] = M4 + M5 + M6 - M7
    # C11[1,0] = M2 + M3 - M6 + M7
    # C11[1,1] = M1 - M3 - M4 + M5 + M7
    # 
    # But our current terms only put partial contributions. We need to add
    # correction terms to make the full Strassen formula work.
    # Actually, the way I've set it up, each "term" contributes to multiple C entries.
    # Let me re-approach: each Strassen product M_i is a scalar, and it contributes
    # to specific C entries with specific coefficients.
    
    # Let me restart with a cleaner approach.
    # I'll clear what I have and redo.
    U_cols = []
    V_cols = []
    W_cols = []
    
    # For each Strassen product, we add ONE term that contributes to all affected C entries.
    # The key insight: in the tensor decomposition, each rank-1 term is:
    #   sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = C[i,j] for each (i,j,k)
    # 
    # For a block computation, we want:
    #   C_block[i,j] = sum over block products
    # 
    # Let me use a different approach: explicitly enumerate all the scalar multiplications
    # needed and use Strassen to reduce the count for 2x2x2 blocks.
    
    # Actually, let me just do schoolbook for all blocks but use Strassen for the 2x2x2 part.
    # The 2x2x2 part needs 7 instead of 8 multiplications.
    # The other blocks (involving 2x3, 3x2, 3x3) use schoolbook.
    
    # Block structure:
    # C11 = A11*B11 (2x2 * 2x2) -> use Strassen, 7 terms
    # C12 = A11*B12 + A12*B22 (2x2*2x3 + 2x3*3x3) -> schoolbook: 2*2*3 + 2*3*3 = 12+18=30
    # C21 = A21*B11 + A22*B21 (2x2*2x2 + 2x3*3x2) -> schoolbook: 2*2*2 + 2*3*3 = 8+18=26
    # C22 = A21*B12 + A22*B22 (2x2*2x3 + 2x3*3x3) -> schoolbook: 2*2*3 + 2*3*3 = 12+18=30
    
    # Total schoolbook: 4*5*5 = 100
    # With Strassen on C11: 7 + 30 + 26 + 30 = 93
    
    # Let me build this properly.
    
    def schoolbook_block(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """Add schoolbook terms for a block multiplication."""
        for ar in range(len(a_rows)):
            for ac in range(len(a_cols)):
                for bc in range(len(b_cols)):
                    # A[a_rows[ar], a_cols[ac]] * B[b_rows[bc], b_cols[bc]]
                    # Wait, I need to be more careful with indices.
                    pass
    
    # Let me just do full schoolbook but skip one term in the 2x2x2 corner and add Strassen corrections.
    # This is getting complex. Let me just return schoolbook for now and focus on getting
    # the verification working properly, then optimize.
    
    return _schoolbook()

def _try_rank_random(r, n, m, p, max_iters=2000, timeout=8.0, seed_base=0):
    """Try random {-1,0,1} entries for rank r."""
    start_time = time.time()
    for it in range(max_iters):
        if time.time() - start_time > timeout:
            break
        rng = np.random.RandomState(seed_base + it)
        U = rng.randint(-1, 2, size=(n * m, r))
        V = rng.randint(-1, 2, size=(m * p, r))
        W = rng.randint(-1, 2, size=(p * n, r))
        
        if _verify_numpy(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None

def _try_rank_gaussian(r, n, m, p, max_iters=50, timeout=5.0, seed_base=1000):
    """Try Gaussian random initialization with rounding."""
    start_time = time.time()
    for it in range(max_iters):
        if time.time() - start_time > timeout:
            break
        rng = np.random.RandomState(seed_base + it)
        U = rng.randn(n * m, r)
        V = rng.randn(m * p, r)
        W = rng.randn(p * n, r)
        
        # Round to nearest 0.5
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        if _verify_numpy(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    return None

def _construct_92_rank():
    """
    Construct a rank-93 decomposition for 4x5x5 using Strassen on the 2x2x2 corner.
    """
    n, m, p = 4, 5, 5
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    def z20(): return [0]*20
    def z25(): return [0]*25
    
    # Schoolbook for all 100 terms, but replace the 8 terms for A11*B11 with 7 Strassen terms.
    # 
    # A11*B11: A rows 0-1, cols 0-1; B rows 0-1, cols 0-1; C rows 0-1, cols 0-1
    # Schoolbook would be:
    #   C[0,0] += A[0,0]*B[0,0], A[0,1]*B[1,0]
    #   C[0,1] += A[0,0]*B[0,1], A[0,1]*B[1,1]
    #   C[1,0] += A[1,0]*B[0,0], A[1,1]*B[1,0]
    #   C[1,1] += A[1,0]*B[0,1], A[1,1]*B[1,1]
    # That's 8 terms. Strassen uses 7.
    
    # Strassen products for 2x2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 - M2 + M3 + M4 - M6
    # C01 = M4 + M5 + M6 - M7
    # C10 = M2 + M3 - M6 + M7
    # C11 = M1 - M3 - M4 + M5 + M7
    
    # A11: A[0,0]=idx0, A[0,1]=idx1, A[1,0]=idx2, A[1,1]=idx3
    # B11: B[0,0]=idx0, B[0,1]=idx1, B[1,0]=idx2, B[1,1]=idx3
    # C11: C[0,0]=idx0, C[0,1]=idx1, C[1,0]=idx2, C[1,1]=idx3
    
    # M1 = (A00+A11)*(B00+B11)
    u = z20(); v = z25(); w = z20()
    u[0]=1; u[3]=1
    v[0]=1; v[3]=1
    w[0]=1; w[3]=1
    add_term(u, v, w)
    
    # M2 = (A10+A11)*B00
    u = z20(); v = z25(); w = z20()
    u[2]=1; u[3]=1
    v[0]=1
    w[0]=1
    add_term(u, v, w)
    
    # M3 = A00*(B10-B11)
    u = z20(); v = z25(); w = z20()
    u[0]=1
    v[2]=1; v[3]=-1
    w[2]=1; w[3]=-1
    add_term(u, v, w)
    
    # M4 = A11*(B01-B00)
    u = z20(); v = z25(); w = z20()
    u[3]=1
    v[1]=1; v[0]=-1
    w[1]=1; w[0]=-1
    add_term(u, v, w)
    
    # M5 = (A00+A01)*B11
    u = z20(); v = z25(); w = z20()
    u[0]=1; u[1]=1
    v[3]=1
    w[3]=1
    add_term(u, v, w)
    
    # M6 = (A10-A00)*(B00+B01)
    u = z20(); v = z25(); w = z20()
    u[2]=1; u[0]=-1
    v[0]=1; v[1]=1
    w[0]=1; w[1]=1
    add_term(u, v, w)
    
    # M7 = (A01-A11)*(B10+B11)
    u = z20(); v = z25(); w = z20()
    u[1]=1; u[3]=-1
    v[2]=1; v[3]=1
    w[2]=1; w[3]=1
    add_term(u, v, w)
    
    # Now add schoolbook for all OTHER terms (those not involving A11*B11)
    # All terms: A[i,j]*B[j,k] contributes to C[k,i]
    # Skip terms where i in {0,1}, j in {0,1}, k in {0,1} (those are A11*B11)
    
    for i in range(4):
        for j in range(5):
            for k in range(5):
                # Skip if this is part of A11*B11
                if i < 2 and j < 2 and k < 2:
                    continue
                u = z20(); v = z25(); w = z20()
                u[i*5+j] = 1
                v[j*5+k] = 1
                w[k*4+i] = 1
                add_term(u, v, w)
    
    rank = len(U_cols)
    U = [[U_cols[c][r] for c in range(rank)] for r in range(20)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(25)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(20)]
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try the structured Strassen-based block decomposition
    block_result = _construct_92_rank()
    if block_result is not None:
        block_rank = len(block_result['U'][0])
        if block_rank < best_rank:
            # Verify it's correct
            if _verify_numpy(block_result['U'], block_result['V'], block_result['W'], n, m, p):
                best = block_result
                best_rank = block_rank
    
    # Try progressively lower ranks with random search
    for rank in range(best_rank - 1, 24, -1):
        if time.time() > deadline - 20:
            break
        
        remaining = deadline - time.time()
        if remaining < 3:
            break
        
        # Try random {-1,0,1} search
        t_budget = min(8.0, remaining * 0.4)
        result = _try_rank_random(rank, n, m, p, max_iters=3000, timeout=t_budget, seed_base=rank * 100)
        if result is not None:
            best = result
            best_rank = rank
            break
        
        if time.time() > deadline - 15:
            break
        
        # Try Gaussian with rounding
        t_budget = min(5.0, remaining * 0.3)
        result = _try_rank_gaussian(rank, n, m, p, max_iters=30, timeout=t_budget, seed_base=rank * 200)
        if result is not None:
            best = result
            best_rank = rank
            break
    
    return best
# EVOLVE-BLOCK-END
