"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W):
    """Verify that the tensor decomposition is exact.
    
    The matrix multiplication tensor T satisfies:
    T[i, j, k] = 1 if and only if the output C[i,k] = sum_j A[i,j]*B[j,k]
    
    In the standard tensor decomposition:
    T = sum_r u_r ⊗ v_r ⊗ w_r
    
    where u_r is indexed by (i,j), v_r by (j,k), w_r by (k,i) [with W[k*n+i, r]]
    
    The bilinear form for computing C[i,k] is:
    C[i,k] = sum_r (sum_j U[i*m+j, r]) * (sum_j V[j*p+k, r]) * W[k*n+i, r]
    
    But actually, the standard formulation is:
    C[i,k] = sum_r U[i*m+?, r] * V[?, r] * W[k*n+i, r]
    
    Let me think more carefully. The standard tensor decomposition for matrix multiplication:
    We want C = A * B where A is n×m, B is m×p, C is n×p.
    
    The multiplication tensor is a 3-tensor T of shape (n*m, m*p, p*n) where:
    T[i*m+j, j*p+k, k*n+i] = 1 for all valid i, j, k.
    
    A rank-r decomposition gives:
    T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    For the schoolbook, each term c corresponds to one (i,j,k) triple,
    and U, V, W have 1s at the corresponding positions.
    
    To verify, we need to check that for all i, j, k:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    And for all (i,j,k') where k' ≠ k (same i,j):
    sum_c U[i*m+j, c] * V[j*p+k', c] * W[k'*n+i, c] = 0
    
    More generally, we need the full tensor to match.
    
    Let's construct the full tensor and compare.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    # Build the tensor from the decomposition
    # T[i*m+j, j*p+k, k*n+i]
    T_comp = np.zeros((n*m, m*p, p*n))
    for c in range(rank):
        uc = np.array([U[i][c] for i in range(n*m)])
        vc = np.array([V[j][c] for j in range(m*p)])
        wc = np.array([W[k][c] for k in range(p*n)])
        T_comp += np.einsum('a,b,c->abc', uc, vc, wc)
    
    # Build the true tensor
    T_true = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_true[i*m+j, j*p+k, k*n+i] = 1
    
    # Check exactness (allowing for half-integers)
    diff = np.abs(T_comp - T_true)
    max_diff = np.max(diff)
    return max_diff < 1e-9

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = len(best["U"][0])
    
    # Try to find a lower rank decomposition using a simple approach:
    # We know the rank of 2x2x9 matrix multiplication.
    # The schoolbook gives rank 36.
    # 
    # Key insight: For multiplying a 2×2 by 2×p, we can potentially exploit
    # structure. The 2×2×2 case has rank 7 (Strassen), not 8.
    # 
    # For 2×2×p, one approach: decompose the p columns.
    # For each pair of consecutive columns, use a Strassen-like trick.
    # 
    # Actually, let me think about this differently.
    # The tensor for 2×2×p matrix multiplication can be viewed as:
    # We have 4×p = 4p products in schoolbook.
    # 
    # Strassen's algorithm for 2×2×2 gives rank 7 instead of 8.
    # The 7 Strassen products are:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)h
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)e
    # m6 = (c+d)(h-i)  -- wait, let me be more careful
    # 
    # Standard Strassen for 2x2:
    # m1 = (a00+a11)*(b00+b11)
    # m2 = (a10+a11)*b00
    # m3 = a00*(b01-b11)
    # m4 = a11*(b10-b00)
    # m5 = (a00+a01)*b11
    # m6 = (a10-a00)*(b00+b10)
    # m7 = (a01-a11)*(b01+b11)
    # 
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    # 
    # Now for 2×2×9: we have 9 columns of B.
    # For each column, we need 4 products in schoolbook.
    # Using Strassen per 2 columns: for 2 columns, we need 7 products instead of 8.
    # 
    # But the Strassen products mix rows of A with combinations of B columns.
    # For column k of B (which is a 2-vector [B0k, B1k]), Strassen gives us
    # the 2×1 result using 7 "virtual" products that involve both rows of A.
    # 
    # Wait, Strassen is for 2×2 × 2×2 → 2×2. For 2×2 × 2×1, the rank is actually 4 (it's just 4 scalar multiplications).
    # 
    # The saving comes from processing multiple columns together.
    # 
    # For 2×2 × 2×p: think of it as 2×2 × (2×p). 
    # We can group columns into pairs and use Strassen on each pair.
    # For 2 columns: rank 7 instead of 8.
    # For 9 columns: 4 pairs (8 columns) + 1 single column.
    # 4 pairs × 7 + 1 single × 4 = 28 + 4 = 32? 
    # But the single column still needs 4 products.
    # Hmm, but can we do better for the last column?
    # 
    # Actually, the rank of the 2×2×2p tensor... let me think about it as a block.
    # 
    # Alternative: use the fact that rank(2×2×p) ≤ 4p - something.
    # 
    # Known results: The tensor rank of multiplying m×n by n×p matrices.
    # For 2×2×2: rank = 7
    # For 2×2×3: rank = ? 
    # 
    # Let me try a direct construction.
    # 
    # For 2×2×p, represent A as [[a, b], [c, d]] and B as a 2×p matrix.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # 
    # C[0,k] = a*B[0,k] + b*B[1,k]
    # C[1,k] = c*B[0,k] + d*B[1,k]
    # 
    # Schoolbook: 4p products.
    # 
    # Strassen-style: introduce auxiliary variables.
    # For pairs of columns (2k, 2k+1):
    # Let B0k = B[0, 2k], B1k = B[1, 2k], B0(k+1) = B[0, 2k+1], B1(k+1) = B[1, 2k+1]
    # 
    # Using Strassen for the 2×2 block [[B0k, B0(k+1)], [B1k, B1(k+1)]]:
    # m1 = (a+d)*(B0k+B1(k+1))  -- wait, Strassen uses specific combinations
    # 
    # Actually, let me just use the known Strassen formulas directly.
    # For the 2×2 subproblem with B_block = [[B0k, B0k1], [B1k, B1k1]]:
    # m1 = (a+d)(B0k+B1k1)
    # m2 = (c+d)B1k
    # m3 = a(B0k1-B1k1)
    # m4 = d(B1k-B0k)
    # m5 = (a+b)B1k1
    # m6 = (c-a)(B0k+B1k)
    # m7 = (b-d)(B0k1+B1k1)
    # 
    # C[0,2k] = m1 + m4 - m5 + m7
    # C[0,2k+1] = m3 + m5
    # C[1,2k] = m2 + m4
    # C[1,2k+1] = m1 - m2 + m3 + m6
    # 
    # Each of these is a product of a linear form in {a,b,c,d} and a linear form in the B entries.
    # So each m_i is a valid tensor term.
    # 
    # For 9 columns: 4 pairs (columns 0-1, 2-3, 4-5, 6-7) + column 8.
    # 4 pairs × 7 products = 28 products.
    # Column 8: 4 products (schoolbook).
    # Total: 32 products.
    # 
    # Wait, but the W matrix has entries W[k*n+i, r] = W[k*2+i, r] for the output index.
    # And the output C has indices (i, k) where i is row (0,1) and k is column (0..8).
    # So W index is k*2 + i.
    # 
    # Let me construct this.
    # 
    # For each Strassen product m_r in pair g (processing columns 2g, 2g+1):
    # The U vector has entries for (i,j) pairs: (0,0), (0,1), (1,0), (1,1)
    # i.e., a, b, c, d.
    # The V vector has entries for (j,k) pairs: j in {0,1}, k in {0..8}
    # i.e., B[0,0], B[1,0], B[0,1], B[1,1], ..., B[0,8], B[1,8]
    # The W vector has entries for (k,i) pairs: k in {0..8}, i in {0,1}
    # i.e., W index = k*2 + i, so W[0], W[1], W[2], W[3], ..., W[16], W[17]
    # 
    # For m1 = (a+d)(B0k+B1k1):
    # U: a→1, d→1, so U[0,0]=1 (i=0,j=0), U[1,1]=1 (i=1,j=1)
    # V: B0k→1 (j=0,k=2g), B1k1→1 (j=1,k=2g+1), so V[0*9+2g]=1, V[1*9+2g+1]=1
    # W: This product contributes to C[0,2k] (W[2k*2+0]=W[4k]), C[1,2k] (W[2k*2+1]=W[4k+1]),
    #     C[1,2k+1] (W[(2k+1)*2+1]=W[4k+3])
    # With coefficients: C[0,2k] gets +1, C[1,2k] gets +1, C[1,2k+1] gets +1
    # So W[4k]=1, W[4k+1]=1, W[4k+3]=1
    # 
    # Let me verify with the formula:
    # C[0,2k] = m1 + m4 - m5 + m7
    # C[1,2k] = m2 + m4
    # C[0,2k+1] = m3 + m5
    # C[1,2k+1] = m1 - m2 + m3 + m6
    # 
    # So m1 contributes +1 to C[0,2k], +1 to C[1,2k], +1 to C[1,2k+1]
    # 
    # Let me code this up.
    
    n, m, p = DIMENSIONS
    # n=2, m=2, p=9
    
    # We'll build the decomposition.
    # U has shape (n*m, rank) = (4, rank)
    # V has shape (m*p, rank) = (18, rank)
    # W has shape (p*n, rank) = (18, rank)
    
    terms = []  # Each term is (U_vec, V_vec, W_vec) as dicts or sparse representations
    
    # Process pairs: (0,1), (2,3), (4,5), (6,7) and single column 8
    for pair_idx in range(4):
        k0 = 2 * pair_idx      # first column
        k1 = 2 * pair_idx + 1  # second column
        
        # m1 = (a+d)(B[0,k0] + B[1,k1])
        # U: a=1, d=1 → U[0]=1 (i=0,j=0), U[3]=1 (i=1,j=1) [using index i*m+j = i*2+j]
        # V: B[0,k0]=1, B[1,k1]=1 → V[0*9+k0]=1, V[1*9+k1]=1
        # W: contributes +1 to C[0,k0], +1 to C[1,k0], +1 to C[1,k1]
        # W indices: k*2+i → C[0,k0]: W[k0*2+0]=W[2k0], C[1,k0]: W[2k0+1], C[1,k1]: W[2k1+1]
        terms.append({
            'U': {0: 1, 3: 1},
            'V': {k0: 1, 9+k1: 1},
            'W': {2*k0: 1, 2*k0+1: 1, 2*k1+1: 1}
        })
        
        # m2 = (c+d)*B[1,k0]
        # U: c=1, d=1 → U[2]=1 (i=1,j=0), U[3]=1 (i=1,j=1)
        # V: B[1,k0]=1 → V[1*9+k0]=1
        # W: contributes +1 to C[1,k0] → W[2k0+1]
        terms.append({
            'U': {2: 1, 3: 1},
            'V': {9+k0: 1},
            'W': {2*k0+1: 1}
        })
        
        # m3 = a*(B[0,k1] - B[1,k1])
        # U: a=1 → U[0]=1
        # V: B[0,k1]=1, B[1,k1]=-1 → V[k1]=1, V[9+k1]=-1
        # W: contributes +1 to C[0,k1] → W[2k1+0]
        terms.append({
            'U': {0: 1},
            'V': {k1: 1, 9+k1: -1},
            'W': {2*k1: 1}
        })
        
        # m4 = d*(B[1,k0] - B[0,k0])
        # U: d=1 → U[3]=1
        # V: B[1,k0]=1, B[0,k0]=-1 → V[9+k0]=1, V[k0]=-1
        # W: contributes +1 to C[0,k0] and +1 to C[1,k0]
        terms.append({
            'U': {3: 1},
            'V': {9+k0: 1, k0: -1},
            'W': {2*k0: 1, 2*k0+1: 1}
        })
        
        # m5 = (a+b)*B[1,k1]
        # U: a=1, b=1 → U[0]=1 (i=0,j=0), U[1]=1 (i=0,j=1)
        # V: B[1,k1]=1 → V[9+k1]=1
        # W: contributes -1 to C[0,k0] and +1 to C[0,k1]
        terms.append({
            'U': {0: 1, 1: 1},
            'V': {9+k1: 1},
            'W': {2*k0: -1, 2*k1: 1}
        })
        
        # m6 = (c-a)*(B[0,k0] + B[1,k0])
        # U: c=1, a=-1 → U[2]=1 (i=1,j=0), U[0]=-1 (i=0,j=0)
        # V: B[0,k0]=1, B[1,k0]=1 → V[k0]=1, V[9+k0]=1
        # W: contributes +1 to C[1,k1]
        terms.append({
            'U': {2: 1, 0: -1},
            'V': {k0: 1, 9+k0: 1},
            'W': {2*k1+1: 1}
        })
        
        # m7 = (b-d)*(B[0,k1] + B[1,k1])
        # U: b=1, d=-1 → U[1]=1 (i=0,j=1), U[3]=-1 (i=1,j=1)
        # V: B[0,k1]=1, B[1,k1]=1 → V[k1]=1, V[9+k1]=1
        # W: contributes +1 to C[0,k0]
        terms.append({
            'U': {1: 1, 3: -1},
            'V': {k1: 1, 9+k1: 1},
            'W': {2*k0: 1}
        })
    
    # Single column 8 (k=8): schoolbook 4 products
    k = 8
    # C[0,8] = a*B[0,8] + b*B[1,8]
    # C[1,8] = c*B[0,8] + d*B[1,8]
    
    # a*B[0,8]
    terms.append({
        'U': {0: 1},
        'V': {k: 1},
        'W': {2*k: 1}
    })
    # b*B[1,8]
    terms.append({
        'U': {1: 1},
        'V': {9+k: 1},
        'W': {2*k: 1}
    })
    # c*B[0,8]
    terms.append({
        'U': {2: 1},
        'V': {k: 1},
        'W': {2*k+1: 1}
    })
    # d*B[1,8]
    terms.append({
        'U': {3: 1},
        'V': {9+k: 1},
        'W': {2*k+1: 1}
    })
    
    # Now convert to matrix form
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, term in enumerate(terms):
        for idx, val in term['U'].items():
            U[idx][r] = val
        for idx, val in term['V'].items():
            V[idx][r] = val
        for idx, val in term['W'].items():
            W[idx][r] = val
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify
    if _verify(U, V, W):
        return result
    else:
        # Fall back to schoolbook
        return _schoolbook()

# EVOLVE-BLOCK-END
