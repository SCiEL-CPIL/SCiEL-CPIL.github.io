"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def _verify_exact(U, V, W, n, m, p, rank, tol=1e-6):
    """Verify decomposition exactly by checking all valid and invalid triplets."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check all valid (i,j,k) triplets sum to 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    
    # Check all invalid triplets sum to 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if i == i2 and j == j2 and k == k2:
                                continue
                            s = 0.0
                            for c in range(rank):
                                s += U[i*m+j, c] * V[j2*p+k, c] * W[k*n+i2, c]
                            if abs(s) > tol:
                                return False
    return True

def _build_strassen_2x2x2(offset_u, offset_v, offset_w, i_base, j_base, k_base):
    """
    Build Strassen decomposition for a 2x2x2 sub-block.
    A[i,j] with i in {i_base, i_base+1}, j in {j_base, j_base+1}
    B[j,k] with j in {j_base, j_base+1}, k in {k_base, k_base+1}
    C[i,k] with i in {i_base, i_base+1}, k in {k_base, k_base+1}
    
    Returns list of 7 (u_idx, v_idx, w_idx, coeff_u, coeff_v, coeff_w) tuples
    where u_idx = i*m+j, v_idx = j*p+k, w_idx = k*n+i
    """
    n, m, p = 2, 3, 4  # global dims for index computation
    
    # Strassen multipliers for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each multiplier is a sum of products of A-entries and B-entries.
    # For the tensor decomposition, each "multiplication" corresponds to one rank-1 term.
    # But Strassen uses linear combinations, so we need to distribute.
    
    # The key insight: each Strassen multiplier M_r = sum of (A-linear-comb) * (B-linear-comb)
    # In the tensor decomposition, we can't have linear combinations in a single rank-1 term.
    # Instead, we need to expand each multiplier into its constituent rank-1 terms.
    
    # Actually, for tensor rank decomposition, each term is U[a,c]*V[b,c]*W[d,c].
    # The Strassen algorithm says C = sum_r M_r * (linear combo of C-entries)
    # where M_r = (linear combo of A-entries) * (linear combo of B-entries)
    
    # To convert to tensor rank decomposition:
    # M_r = sum_i alpha_{r,i} * A_i  (linear combo of A entries)
    #      * sum_j beta_{r,j} * B_j  (linear combo of B entries)
    #      = sum_{i,j} alpha_{r,i} * beta_{r,j} * A_i * B_j
    #
    # So each M_r contributes |alpha_r| * |beta_r| rank-1 terms.
    # But that would give us back the schoolbook rank!
    
    # The trick: in the tensor decomposition, the "output" side also has linear combos.
    # C_k = sum_r gamma_{r,k} * M_r
    # So the full decomposition is:
    # C_k = sum_r gamma_{r,k} * sum_{i,j} alpha_{r,i} * beta_{r,j} * A_i * B_j
    #     = sum_{r,i,j} (gamma_{r,k} * alpha_{r,i} * beta_{r,j}) * A_i * B_j
    #
    # But this is still many terms. The key is that for the TENSOR decomposition,
    # we're decomposing the tensor T_{a,b,c} = sum_r U_{a,r} V_{b,r} W_{c,r}
    #
    # For the 2x2x2 matmul tensor, T[(i,j),(j',k),(k',i')] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    # Strassen shows this tensor has rank 7.
    
    # The 7 rank-1 terms for 2x2x2 Strassen:
    # Term r has:
    #   U part: linear combo of A-entries (indices (i,j) for i,j in {0,1})
    #   V part: linear combo of B-entries (indices (j',k) for j',k in {0,1})  
    #   W part: linear combo of C-entries (indices (k',i') for k',i' in {0,1})
    
    # Strassen's 7 terms (for local indices 0,1):
    # Term 1: U = A[0,0]+A[1,1], V = B[0,0]+B[1,1], W = C[0,0]+C[1,1] - C[0,1] + C[1,0]
    #         Actually let me use the standard form:
    # M1 = (A11+A22)(B11+B22), contributes to: C11 += M1, C12 -= M1... no.
    
    # Let me use the explicit decomposition. For 2x2x2, the tensor has 7 rank-1 terms.
    # Using the standard Strassen with local indices (0,0),(0,1),(1,0),(1,1):
    
    # The 7 rank-1 components (U_coeffs, V_coeffs, W_coeffs) where each is a dict from local index to coefficient:
    strassen_terms = [
        # M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
        # C[0,0] += M1, C[0,1] += 0, C[1,0] += 0, C[1,1] -= M1... 
        # Standard: C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6
        # So M1 contributes: C[0,0] += 1, C[1,1] += 1 (with sign in M1 definition)
        # Actually: C11 gets +M1, C22 gets +M1
        # U: A[0,0]=1, A[1,1]=1
        # V: B[0,0]=1, B[1,1]=1
        # W: C[0,0]=1, C[1,1]=1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (A[1,0]+A[1,1])B[0,0]
        # C21 += M2, C22 -= M2
        # U: A[1,0]=1, A[1,1]=1
        # V: B[0,0]=1
        # W: C[1,0]=1, C[1,1]=-1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = A[0,0](B[1,0]-B[1,1])
        # C12 += M3, C22 += M3
        # U: A[0,0]=1
        # V: B[1,0]=1, B[1,1]=-1
        # W: C[0,1]=1, C[1,1]=1
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = A[1,1](B[0,1]-B[0,0])
        # C11 += M4, C21 += M4
        # U: A[1,1]=1
        # V: B[0,1]=1, B[0,0]=-1
        # W: C[0,0]=1, C[1,0]=1
        ({(1,1): 1}, {(0,1): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (A[0,0]+A[0,1])B[1,1]
        # C11 -= M5, C12 += M5
        # U: A[0,0]=1, A[0,1]=1
        # V: B[1,1]=1
        # W: C[0,0]=-1, C[0,1]=1
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6 = (A[1,0]-A[0,0])(B[0,0]+B[0,1])
        # C22 += M6
        # U: A[1,0]=1, A[0,0]=-1
        # V: B[0,0]=1, B[0,1]=1
        # W: C[1,1]=1
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (A[0,1]-A[1,1])(B[1,0]+B[1,1])
        # C11 += M7
        # U: A[0,1]=1, A[1,1]=-1
        # V: B[1,0]=1, B[1,1]=1
        # W: C[0,0]=1
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    terms = []
    for u_coeff, v_coeff, w_coeff in strassen_terms:
        u_entries = []
        for (li, lj), c in u_coeff.items():
            gi = i_base + li
            gj = j_base + lj
            u_entries.append((gi * m + gj, c))
        
        v_entries = []
        for (lj, lk), c in v_coeff.items():
            gj = j_base + lj
            gk = k_base + lk
            v_entries.append((gj * p + gk, c))
        
        w_entries = []
        for (lk, li), c in w_coeff.items():
            gk = k_base + lk
            gi = i_base + li
            w_entries.append((gk * n + gi, c))
        
        terms.append((u_entries, v_entries, w_entries))
    
    return terms

def _build_block_decomposition():
    """
    Build the block decomposition for 2x3 x 3x4 using Strassen on 2x2x2 blocks.
    
    Split j (middle dim, size 3) into {0,1} and {2}
    Split k (output dim, size 4) into {0,1} and {2,3}
    
    This gives 4 sub-problems:
    1. i in {0,1}, j in {0,1}, k in {0,1}: 2x2x2 -> Strassen rank 7
    2. i in {0,1}, j in {0,1}, k in {2,3}: 2x2x2 -> Strassen rank 7
    3. i in {0,1}, j in {2}, k in {0,1}: 2x1x2 -> rank 2
    4. i in {0,1}, j in {2}, k in {2,3}: 2x1x2 -> rank 2
    
    Total rank: 7+7+2+2 = 18
    """
    n, m, p = 2, 3, 4
    
    # Collect all terms: list of (u_entries, v_entries, w_entries)
    # where each _entries is list of (index, coefficient)
    all_terms = []
    
    # Sub-problem 1: i_base=0, j_base=0, k_base=0 (2x2x2)
    all_terms.extend(_build_strassen_2x2x2(None, None, None, 0, 0, 0))
    
    # Sub-problem 2: i_base=0, j_base=0, k_base=2 (2x2x2)
    all_terms.extend(_build_strassen_2x2x2(None, None, None, 0, 0, 2))
    
    # Sub-problem 3: i in {0,1}, j=2, k in {0,1} (2x1x2)
    # T[(i,2),(2,k),(k',i')] = delta_{i,i'} delta_{k,k'}
    # Rank 2 decomposition:
    # Term 3a: i=0, k=0: U[0*m+2]=1, V[2*p+0]=1, W[0*n+0]=1
    # Term 3b: i=1, k=1: U[1*m+2]=1, V[2*p+1]=1, W[1*n+1]=1
    all_terms.append(
        ([(0 * m + 2, 1)], [(2 * p + 0, 1)], [(0 * n + 0, 1)])
    )
    all_terms.append(
        ([(1 * m + 2, 1)], [(2 * p + 1, 1)], [(1 * n + 1, 1)])
    )
    
    # Sub-problem 4: i in {0,1}, j=2, k in {2,3} (2x1x2)
    # T[(i,2),(2,k),(k',i')] = delta_{i,i'} delta_{k,k'} for k in {2,3}
    # Term 4a: i=0, k=2: U[0*m+2]=1, V[2*p+2]=1, W[2*n+0]=1
    # Term 4b: i=1, k=3: U[1*m+2]=1, V[2*p+3]=1, W[3*n+1]=1
    all_terms.append(
        ([(0 * m + 2, 1)], [(2 * p + 2, 1)], [(2 * n + 0, 1)])
    )
    all_terms.append(
        ([(1 * m + 2, 1)], [(2 * p + 3, 1)], [(3 * n + 1, 1)])
    )
    
    # Now build U, V, W matrices
    rank = len(all_terms)
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    
    for c, (u_entries, v_entries, w_entries) in enumerate(all_terms):
        for idx, coeff in u_entries:
            U[idx][c] += Fraction(coeff)
        for idx, coeff in v_entries:
            V[idx][c] += Fraction(coeff)
        for idx, coeff in w_entries:
            W[idx][c] += Fraction(coeff)
    
    return {"U": U, "V": V, "W": W}

def _to_rational_array(mat, tol=1e-8):
    """Convert a numeric matrix to a matrix of Fractions (exact rationals)."""
    rows, cols = mat.shape
    result = [[Fraction(0) for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            val = mat[i, j]
            if abs(val) < tol:
                result[i][j] = Fraction(0)
                continue
            found = False
            for den in range(1, 201):
                num = round(val * den)
                if abs(val - num / den) < tol:
                    result[i][j] = Fraction(num, den)
                    found = True
                    break
            if not found:
                result[i][j] = Fraction(val).limit_denominator(1000000)
    return result

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140.0
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 24
    
    # Build the schoolbook solution as fallback
    def make_schoolbook():
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = make_schoolbook()
    best_rank = schoolbook_rank
    
    # Try the block decomposition with Strassen
    try:
        block_result = _build_block_decomposition()
        rank = len(block_result["U"][0])
        
        # Verify
        if _verify_exact(block_result["U"], block_result["V"], block_result["W"], n, m, p, rank, tol=1e-6):
            best_result = block_result
            best_rank = rank
        else:
            # Verification failed, try to debug
            pass
    except Exception as e:
        pass
    
    return best_result
# EVOLVE-BLOCK-END
