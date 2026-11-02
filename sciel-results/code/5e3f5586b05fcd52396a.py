"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _strassen_2x2x2_terms():
    """Return rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Returns list of (u_idx, v_idx, w_idx, coeff) where:
      U row index: i*2+j for A[i,j]  (0..3)
      V row index: j*2+k for B[j,k]  (0..3)
      W row index: k*2+i for C[k,i]  (0..3)
    """
    # Strassen: 7 products, each expanded into rank-1 terms
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    #
    # W index: W[k*2+i]
    # C[0,0]: k=0,i=0 -> W[0]
    # C[1,0]: k=0,i=1 -> W[1]
    # C[0,1]: k=1,i=0 -> W[2]
    # C[1,1]: k=1,i=1 -> W[3]
    
    # Each Strassen product is a product of two linear forms.
    # We represent each as a set of (u,v,w,coeff) rank-1 contributions.
    # But Strassen gives 7 scalar products, each contributing to multiple C entries.
    # The standard way: each M_t is one "virtual" product, but in tensor rank terms,
    # we need to expand each into its rank-1 components.
    
    # Actually, the cleanest way: the Strassen algorithm defines 7 products where
    # each product is (linear form in A) * (linear form in B).
    # Each such product contributes to C entries with specific coefficients.
    # In the tensor decomposition, each "product" of two linear forms becomes
    # a sum of rank-1 tensors.
    
    # Let me define the 7 Strassen multiplications and their contributions to C:
    # M1 = (A00+A11)*(B00+B11) -> contributes to C[0,0](+1), C[1,1](+1)
    # M2 = (A10+A11)*B00 -> contributes to C[1,0](+1), C[1,1](-1)
    # M3 = A00*(B01-B11) -> contributes to C[0,1](+1), C[1,1](+1)
    # M4 = A11*(B10-B00) -> contributes to C[0,0](+1), C[1,0](+1)
    # M5 = (A00+A01)*B11 -> contributes to C[0,0](-1), C[0,1](+1)
    # M6 = (A10-A00)*(B00+B01) -> contributes to C[1,1](+1)
    # M7 = (A01-A11)*(B10+B11) -> contributes to C[0,0](+1)
    
    # For each M_t = (sum of a_i * A[i,j]) * (sum of b_j * B[j,k]),
    # we expand into rank-1 terms.
    
    terms = []
    
    # M1 = (A00+A11)*(B00+B11)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # A00*B00: i=0,j=0,k=0 -> U[0],V[0],W[0], coeff for C[0,0]=+1, C[1,1]=+1
    terms.append((0, 0, 0, 1))    # C[0,0]
    terms.append((0, 3, 2, 0))    # placeholder, will fix
    # Let me redo this more carefully by tracking which C entries each term hits.
    
    # Better approach: for each of the 7 Strassen products, expand into rank-1 terms
    # and track the coefficient for each C entry.
    
    # M1 = (A00+A11)*(B00+B11)
    # Rank-1 terms: (A00,B00), (A00,B11), (A11,B00), (A11,B11)
    # C[0,0] gets +1 from M1, C[1,1] gets +1 from M1
    # So:
    # (A00,B00) -> C[0,0]+=1, C[1,1]+=1: W[0]+=1, W[3]+=1
    terms.append((0, 0, 0, 1))   # A00*B00, C[0,0]
    terms.append((0, 0, 3, 1))   # A00*B00, C[1,1]
    terms.append((0, 3, 2, 0))   # A00*B11, C[0,1]... wait
    
    # I'm overcomplicating this. Let me use a direct approach.
    # The Strassen decomposition for 2x2x2 has rank 7.
    # Each of the 7 "products" is actually a product of two linear combinations.
    # In the tensor rank decomposition, each such product expands to multiple rank-1 terms.
    # The total rank is NOT 7; it's the number of rank-1 terms after expansion.
    # 
    # Wait - no. In tensor rank, the Strassen decomposition IS rank 7.
    # Each of the 7 terms is a rank-1 tensor in the appropriate sense.
    # The key insight: in the tensor M_{ijk} = sum_r U_{ir} V_{jr} W_{kr},
    # each r corresponds to one rank-1 term.
    # For Strassen, we have 7 such terms, but each involves linear combinations
    # of the input entries, not single entries.
    #
    # So U, V, W matrices have entries that are 0, 1, or -1 (linear combinations).
    
    # Let me define the 7 Strassen terms properly.
    # For 2x2*2x2->2x2:
    # A is 2x2, B is 2x2, C is 2x2
    # U has 4 rows (A entries), V has 4 rows (B entries), W has 4 rows (C entries)
    # W index: k*2+i
    
    # Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # C[0,0] = M1 + M4 - M5 + M7
    # C[1,0] = M2 + M4
    # C[0,1] = M3 + M5
    # C[1,1] = M1 - M2 + M3 + M6
    
    # For each M_t, define u_t (4-vector), v_t (4-vector), and c_t (4-vector for C entries)
    # Then the contribution to the tensor is: for each C entry, sum over t of c_t[entry] * u_t * v_t
    
    # Actually, the standard formulation:
    # C[i,k] = sum_t (sum_j u_t[i,j] * A[i,j]) * (sum_j v_t[j,k] * B[j,k]) * w_t[i,k]
    # No, that's not right either.
    
    # The correct formulation for tensor rank:
    # M[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # where M[i,j,k] = 1 if i,j,k form a valid multiplication term, else 0
    # For matrix mult: M[i,j,k] = 1 for all i,j,k (it's the full tensor)
    # Wait no. The matrix multiplication tensor is T[i,j,k] = delta_{j,j} which is just 1.
    # No: T is defined such that C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all i,j,k in the appropriate ranges.
    # The rank decomposition: T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    
    # For Strassen on 2x2:
    # U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    # where indices are: U indexed by (i,j) flattened, V by (j,k) flattened, W by (k,i) flattened
    
    # Let me just hardcode the known Strassen rank-7 decomposition.
    # From literature, the 7 terms for 2x2x2 are:
    
    # Term 1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]  -> (A00+A11)(B00+B11)
    # Term 2: u=[0,0,1,1], v=[1,0,0,0], w=[0,1,0,0]  -> (A10+A11)B00
    # Term 3: u=[1,0,0,0], v=[0,1,0,1], w=[0,0,1,0]  -> A00(B01-B11)... 
    
    # This is getting complicated. Let me just use the schoolbook approach but
    # try to find a lower rank by search.
    
    return terms

def _schoolbook_3x3x6():
    """Schoolbook decomposition for 3x3 * 3x6 -> 3x6. Rank = 54."""
    n, m, p = 3, 3, 6
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
    return U, V, W

def _verify_decomposition(U, V, W, n=3, m=3, p=6):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # T[i,j,k] = 1 for all valid i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Start with schoolbook
    U, V, W = _schoolbook_3x3x6()
    best_rank = n * m * p  # 54
    
    # Verify schoolbook
    if not _verify_decomposition(U, V, W, n, m, p):
        # Should not happen, but fallback
        pass
    
    # Try to improve by attempting to reduce rank through linear algebra
    # The idea: if some columns of [U;V;W] are linearly dependent in a way
    # that preserves the tensor, we can reduce rank.
    # This is hard to do efficiently, so we'll stick with schoolbook for now.
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 5:
        pass  # Safety
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
