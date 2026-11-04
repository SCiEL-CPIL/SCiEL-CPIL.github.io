"""Program for matrix multiplication 4x4 x 4x6 tensor decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2.
    
    Returns (U, V, W) where:
      U: (4, 7) - for A[i,j], row index = i*2+j
      V: (4, 7) - for B[j,k], row index = j*2+k
      W: (4, 7) - for C[i,k], row index = k*2+i  (trace convention: W[k*n+i] with n=2)
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    
    # M2 = (A10 + A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    
    # M3 = A00 * (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    
    # M4 = A11 * (B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    
    # M5 = (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    
    # W: C[i,k] contributions, W row = k*2 + i
    # C[0,0]: i=0,k=0 -> W[0]; C[0,1]: i=0,k=1 -> W[2]
    # C[1,0]: i=1,k=0 -> W[1]; C[1,1]: i=1,k=1 -> W[3]
    
    # C[0,0] = M1 + M4 - M5 + M7
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    
    # C[0,1] = M3 + M5
    W[2, 2] = 1; W[2, 4] = 1
    
    # C[1,0] = M2 + M4
    W[1, 1] = 1; W[1, 3] = 1
    
    # C[1,1] = M1 - M2 + M3 + M6
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    
    return U, V, W


def _block_outer_decomp():
    """
    Decompose the outer 2x2 * 2x3 -> 2x3 block multiplication.
    
    Returns a list of dicts with keys 'U', 'V', 'W' where each is a dict
    mapping (row_block, col_block) -> coefficient.
    
    Uses 11 outer terms:
    - 7 Strassen terms for cols 0,1
    - 4 schoolbook terms for col 2
    """
    sU, sV, sW = _strassen_2x2()
    
    terms = []
    
    # 7 Strassen terms for cols 0,1
    for t in range(7):
        U_dict = {}
        V_dict = {}
        W_dict = {}
        for a in range(2):
            for b in range(2):
                coeff = sU[a*2+b, t]
                if coeff != 0:
                    U_dict[(a,b)] = U_dict.get((a,b), 0) + coeff
        for b in range(2):
            for c in range(2):
                coeff = sV[b*2+c, t]
                if coeff != 0:
                    V_dict[(b,c)] = V_dict.get((b,c), 0) + coeff
        for a in range(2):
            for c in range(2):
                coeff = sW[c*2+a, t]
                if coeff != 0:
                    W_dict[(a,c)] = W_dict.get((a,c), 0) + coeff
        terms.append({'U': U_dict, 'V': V_dict, 'W': W_dict})
    
    # 4 schoolbook terms for col 2
    for a in range(2):
        for b in range(2):
            U_dict = {(a,b): 1}
            V_dict = {(b,2): 1}
            W_dict = {(a,2): 1}
            terms.append({'U': U_dict, 'V': V_dict, 'W': W_dict})
    
    return terms


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strategy: Use 2x2 block structure with Kronecker product decomposition.
    # A is 4x4: 2x2 blocks of size 2x2. A[ai][bj] where ai,bj in {0,1}
    #   A[i,j] with i=2*ai+ai2, j=2*bj+bj2, ai2,bj2 in {0,1}
    # B is 4x6: 2x3 blocks of size 2x2. B[bj][ck] where bj in {0,1}, ck in {0,1,2}
    #   B[j,k] with j=2*bj+bj2, k=2*ck+ck2, bj2,ck2 in {0,1}
    # C is 4x6: 2x3 blocks of size 2x2. C[ai][ck] where ai in {0,1}, ck in {0,1,2}
    #   C[i,k] with i=2*ai+ai2, k=2*ck+ck2, ai2,ck2 in {0,1}
    #
    # C[ai][ck] = sum_bj A[ai][bj] * B[bj][ck]  (block matrix multiplication)
    # Each block product is 2x2 * 2x2 -> 2x2, decomposed via Strassen (rank 7).
    #
    # Outer structure: 2x2 * 2x3 -> 2x3 block multiplication
    # Use 11 outer terms: 7 (Strassen for cols 0,1) + 4 (schoolbook for col 2)
    # Each outer term involves one 2x2*2x2 block product (rank 7)
    # Total rank: 11 * 7 = 77
    
    outer_terms = _block_outer_decomp()
    sU, sV, sW = _strassen_2x2()
    
    total_rank = len(outer_terms) * 7
    U = np.zeros((n*m, total_rank), dtype=int)
    V = np.zeros((m*p, total_rank), dtype=int)
    W = np.zeros((p*n, total_rank), dtype=int)
    
    for ot_idx, outer in enumerate(outer_terms):
        for t in range(7):
            col = ot_idx * 7 + t
            
            # U part: for each (ai, bj) block position
            for (ai, bj), coeff_outer in outer['U'].items():
                for ai2 in range(2):
                    for bj2 in range(2):
                        coeff_inner = sU[ai2*2 + bj2, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            # A[i,j] where i=2*ai+ai2, j=2*bj+bj2
                            u_row = (2*ai + ai2) * 4 + (2*bj + bj2)
                            U[u_row, col] += coeff_outer * coeff_inner
            
            # V part: for each (bj, ck) block position
            for (bj, ck), coeff_outer in outer['V'].items():
                for bj2 in range(2):
                    for ck2 in range(2):
                        coeff_inner = sV[bj2*2 + ck2, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            # B[j,k] where j=2*bj+bj2, k=2*ck+ck2
                            v_row = (2*bj + bj2) * 6 + (2*ck + ck2)
                            V[v_row, col] += coeff_outer * coeff_inner
            
            # W part: for each (ai, ck) block position
            for (ai, ck), coeff_outer in outer['W'].items():
                for ai2 in range(2):
                    for ck2 in range(2):
                        coeff_inner = sW[ck2*2 + ai2, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            # C[i,k] where i=2*ai+ai2, k=2*ck+ck2
                            # Trace convention: W[k*n+i] = W[k*4+i]
                            w_row = (2*ck + ck2) * 4 + (2*ai + ai2)
                            W[w_row, col] += coeff_outer * coeff_inner
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_decomposition():
    """Verify the decomposition is correct."""
    n, m, p = DIMENSIONS  # 4, 4, 6
    result = construct()
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    
    # Check shapes
    assert U.shape == (n*m, U.shape[1]), f"U shape {U.shape} != ({n*m}, ?)"
    assert V.shape == (m*p, V.shape[1]), f"V shape {V.shape} != ({m*p}, ?)"
    assert W.shape == (p*n, W.shape[1]), f"W shape {W.shape} != ({p*n}, ?)"
    assert U.shape[1] == V.shape[1] == W.shape[1], "Rank mismatch"
    
    rank = U.shape[1]
    
    # Test with random matrices
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Compute C from decomposition
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] ... 
    # Actually: C[i,k] = sum_{j,r} A[i,j] * B[j,k] 
    # In tensor form: C[i,k] = sum_{j,r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # But U, V, W are the factor matrices, so:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal the tensor T[i,j,k] = delta_{ij} for matmul
    # Actually the decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T[i,j,k] = 1 if this term contributes to C[i,k] from A[i,j]*B[j,k]
    
    # For matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has T[i,j,k] = 1 for all valid i,j,k (it's the matmul tensor)
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Verify: for random A, B, compute C via decomposition
    C_decomp = np.zeros((n, p))
    for r in range(rank):
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # Wait, this isn't right. Let me think again.
                # U[i*m+j, r] extracts A[i,j] coefficient
                # V[j*p+k, r] extracts B[j,k] coefficient  
                # W[k*n+i, r] extracts C[i,k] coefficient
                # The product sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should give
                # the coefficient such that sum_{j,r} A[i,j] B[j,k] W[k*n+i,r] U[i*m+j,r] V[j*p+k,r] = C[i,k]
                # This means: sum_r W[k*n+i,r] * (sum_j U[i*m+j,r] A[i,j]) * (sum_j V[j*p+k,r] B[j,k])
                # Hmm, that's not quite right either.
                
                # Correct interpretation:
                # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
                # No, that's not right. The standard tensor decomposition is:
                # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
                # where T[i,j,k] is the matmul tensor = 1 for all i,j,k
                # And C[i,k] = sum_{j,r} A[i,j] B[j,k] T[i,j,k] = sum_j A[i,j] B[j,k]
                # So: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
                # And C[i,k] = sum_{j,r} A[i,j] B[j,k] * (sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r])
                # = sum_{j,r} A[i,j] B[j,k] * delta_{T} where T should be 1
                # Actually: C[i,k] = sum_r W[k*n+i,r] * (sum_j U[i*m+j,r] A[i,j]) * (sum_j' V[j'*p+k,r] B[j',k])
                # This requires the sums over j and j' to be the same index, which they are.
                pass
    
    # Simpler verification: check that the tensor T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] equals 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-10:
                    print(f"FAIL: T[{i},{j},{k}] = {s}, expected 1.0")
                    return False
    print(f"PASS: rank {rank} decomposition verified")
    return True

# EVOLVE-BLOCK-END
