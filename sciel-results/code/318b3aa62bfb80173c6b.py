"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _verify_fast(U, V, W, n, m, p, T=None):
    """Fast verification using numpy."""
    if T is None:
        T = _build_tensor(n, m, p)
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    T_decomp = np.zeros_like(T)
    for c in range(rank):
        T_decomp += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
    return np.allclose(T_decomp, T, atol=1e-9)

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _strassen_2x2(U_base, V_base, W_base, i_off, j_off, k_off, n, m, p, rank_offset=0):
    """
    Apply Strassen's algorithm to a 2x2 block multiplication.
    This generates 7 rank-1 terms instead of 8 for the 2x2x2 sub-problem.
    """
    # We are multiplying a 2x2 matrix by a 2x2 matrix.
    # The blocks are defined by offsets in the global indices.
    # U indices: (i, j) -> i_off + i, j_off + j
    # V indices: (j, k) -> j_off + j, k_off + k
    # W indices: (k, i) -> k_off + k, i_off + i
    
    # Strassen formulas for C = A * B:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each "M" is a scalar product of two linear combinations.
    # In the tensor decomposition, each M corresponds to a set of rank-1 terms.
    # Since we are at the leaf level (2x2x2), each M is actually a sum of 1 or more rank-1 terms?
    # No, at the leaf, the "multiplication" is the bilinear form.
    # For a 1x1x1 multiplication, it's 1 term.
    # For a 2x2x2 multiplication using Strassen, we have 7 terms.
    # But wait, the "blocks" A11 etc are 1x1 matrices here.
    # So M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # This is a product of a linear form in A and a linear form in B.
    # This corresponds to a rank-1 term in the tensor if we consider the coefficients.
    # However, the output C is also a linear combination.
    # The rank-1 tensor term is u ⊗ v ⊗ w where:
    # u has non-zero entries at the A-indices involved in the linear form
    # v has non-zero entries at the B-indices involved in the linear form
    # w has non-zero entries at the C-indices involved in the linear form
    
    # Let's generate the 7 terms.
    terms = []
    
    # Helper to add a term
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        # u_coeffs: dict {global_u_idx: coeff}
        # v_coeffs: dict {global_v_idx: coeff}
        # w_coeffs: dict {global_w_idx: coeff}
        terms.append((u_coeffs, v_coeffs, w_coeffs))

    # Indices for the 2x2 block
    # A indices: (i_off+0, j_off+0), (i_off+0, j_off+1), (i_off+1, j_off+0), (i_off+1, j_off+1)
    # B indices: (j_off+0, k_off+0), (j_off+0, k_off+1), (j_off+1, k_off+0), (j_off+1, k_off+1)
    # C indices: (k_off+0, i_off+0), (k_off+0, i_off+1), (k_off+1, i_off+0), (k_off+1, i_off+1)
    
    A11 = (i_off+0, j_off+0)
    A12 = (i_off+0, j_off+1)
    A21 = (i_off+1, j_off+0)
    A22 = (i_off+1, j_off+1)
    
    B11 = (j_off+0, k_off+0)
    B12 = (j_off+0, k_off+1)
    B21 = (j_off+1, k_off+0)
    B22 = (j_off+1, k_off+1)
    
    C11 = (k_off+0, i_off+0)
    C12 = (k_off+0, i_off+1)
    C21 = (k_off+1, i_off+0)
    C22 = (k_off+1, i_off+1)
    
    # M1 = (A11 + A22) * (B11 + B22)
    # Contributes to C11 (+1), C22 (+1)
    add_term({A11: 1, A22: 1}, {B11: 1, B22: 1}, {C11: 1, C22: 1})
    
    # M2 = (A21 + A22) * B11
    # Contributes to C21 (+1), C22 (-1)
    add_term({A21: 1, A22: 1}, {B11: 1}, {C21: 1, C22: -1})
    
    # M3 = A11 * (B12 - B22)
    # Contributes to C12 (+1), C22 (+1)
    add_term({A11: 1}, {B12: 1, B22: -1}, {C12: 1, C22: 1})
    
    # M4 = A22 * (B21 - B11)
    # Contributes to C11 (+1), C21 (+1)
    add_term({A22: 1}, {B21: 1, B11: -1}, {C11: 1, C21: 1})
    
    # M5 = (A11 + A12) * B22
    # Contributes to C11 (-1), C12 (+1)
    add_term({A11: 1, A12: 1}, {B22: 1}, {C11: -1, C12: 1})
    
    # M6 = (A21 - A11) * (B11 + B12)
    # Contributes to C22 (+1)
    add_term({A21: 1, A11: -1}, {B11: 1, B12: 1}, {C22: 1})
    
    # M7 = (A12 - A22) * (B21 + B22)
    # Contributes to C11 (+1)
    add_term({A12: 1, A22: -1}, {B21: 1, B22: 1}, {C11: 1})
    
    return terms

def _construct_strassen_recursive(n, m, p, depth=0, max_depth=3):
    """
    Recursively apply Strassen to compute n x m * m x p.
    We pad dimensions to powers of 2.
    """
    # Pad to next power of 2
    def next_pow2(x):
        if x <= 1: return 1
        return 1 << (x - 1).bit_length()
    
    n_pad = next_pow2(n)
    m_pad = next_pow2(m)
    p_pad = next_pow2(p)
    
    # If we are at the base case (1x1x1), return 1 term
    if n_pad == 1 and m_pad == 1 and p_pad == 1:
        # Single element multiplication
        # U index: 0, V index: 0, W index: 0
        return [({(0,0): 1}, {(0,0): 1}, {(0,0): 1})]
    
    # If depth > max_depth, fall back to schoolbook for the padded block
    if depth > max_depth:
        terms = []
        for i in range(n_pad):
            for j in range(m_pad):
                for k in range(p_pad):
                    terms.append(({(i,j): 1}, {(j,k): 1}, {(k,i): 1}))
        return terms
    
    # Split into 2x2 blocks
    n1 = n_pad // 2
    m1 = m_pad // 2
    p1 = p_pad // 2
    
    # We need to generate terms for the 4 sub-problems:
    # 1. Top-Left: (n1 x m1) * (m1 x p1) -> (n1 x p1)
    # 2. Top-Right: (n1 x m1) * (m1 x p1) -> (n1 x p1)
    # 3. Bottom-Left: (n1 x m1) * (m1 x p1) -> (n1 x p1)
    # 4. Bottom-Right: (n1 x m1) * (m1 x p1) -> (n1 x p1)
    
    # Wait, Strassen operates on 2x2 blocks of matrices.
    # A is n_pad x m_pad. B is m_pad x p_pad.
    # We split A into 4 blocks: A11 (n1xm1), A12 (n1xm1), A21 (n1xm1), A22 (n1xm1)
    # We split B into 4 blocks: B11 (m1xp1), B12 (m1xp1), B21 (m1xp1), B22 (m1xp1)
    # We compute C = A*B, which is n_pad x p_pad.
    # C is split into 4 blocks: C11 (n1xp1), C12 (n1xp1), C21 (n1xp1), C22 (n1xp1)
    
    # The 7 Strassen products are:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # Each Mi is a matrix multiplication of size n1 x m1 * m1 x p1.
    # We can recursively compute these.
    
    # However, the "sums and differences" in the inputs mean we cannot just call
    # the recursive function on the sub-blocks directly, because the recursive function
    # assumes standard multiplication.
    
    # Instead, we should generate the terms for the sub-multiplications and then
    # combine them according to the Strassen formulas.
    
    # Let's define a helper to get the terms for a sub-multiplication with given offsets.
    def get_sub_terms(i_off, j_off, k_off, depth):
        return _construct_strassen_recursive(n1, m1, p1, depth+1, max_depth)
    
    # But the offsets are different for each part.
    # A11 is at (0,0), A12 at (0,m1), A21 at (n1,0), A22 at (n1,m1)
    # B11 is at (0,0), B12 at (0,p1), B21 at (m1,0), B22 at (m1,p1)
    # C11 is at (0,0), C12 at (0,p1), C21 at (n1,0), C22 at (n1,p1)
    
    # This approach is getting complicated because the recursive function returns
    # terms relative to the local coordinates of the sub-problem.
    
    # Let's try a different approach: generate the full set of terms for the padded
    # problem using a recursive Strassen algorithm that tracks the global indices.
    
    # Actually, a simpler way:
    # 1. Generate terms for M1..M7 recursively.
    # 2. Each term in Mi is (u, v, w).
    # 3. We need to map the local indices of Mi to the global indices of the main problem.
    # 4. And we need to apply the linear combinations (sums/differences) to the u, v, w vectors.
    
    # Let's implement a function that takes a list of terms (u, v, w) and a mapping
    # function for u, v, w, and returns the transformed terms.
    
    def transform_terms(terms, u_map, v_map, w_map):
        # u_map: dict {local_u_idx: global_u_idx} or None
        # Actually, u_map should be a function or a dict that maps local (i,j) to global (i,j)
        # And also applies coefficients? No, coefficients are handled by the Strassen formulas.
        # Wait, the Strassen formulas involve sums like A11 + A22.
        # This means the u-vector for M1 has non-zero entries at both A11 and A22 positions.
        # So we can't just transform terms from A11 and A22 separately.
        
        # We need to construct the u, v, w vectors for each Mi directly.
        # This means we need to know the structure of the sub-problems.
        
        # Let's re-evaluate.
        # For M1 = (A11 + A22) * (B11 + B22):
        # The u-vector is the sum of the u-vectors of A11 and A22.
        # The v-vector is the sum of the v-vectors of B11 and B22.
        # The w-vector is the sum of the w-vectors of C11 and C22? No, M1 contributes to C11 and C22.
        # The w-vector for M1 has +1 at C11 positions and +1 at C22 positions.
        
        # This implies that we cannot just "multiply" the terms. We have to construct
        # the linear forms explicitly.
        
        # This is getting too complex for a direct recursive term transformation.
        
        # Alternative: Use the known result that Strassen reduces the rank.
        # For 2x2x2, rank is 7.
        # For 4x4x4, rank is 7^2 = 49.
        # For 8x8x8, rank is 7^3 = 343.
        
        # Our problem is 3x5 x 5x7.
        # Pad to 4x8 x 8x8.
        # We can split 4x8 x 8x8 into 2x4 x 4x4 + 2x4 x 4x4? No.
        
        # Let's use the known construction for 4x4x4 with rank 49.
        # And for 8x8x8 with rank 343.
        
        # But we don't need 8x8x8. We need 4x8 x 8x8.
        
        # Let's try to construct the 4x8 x 8x8 multiplication using Strassen.
        # Split 4x8 into 2x4 + 2x4.
        # Split 8x8 into 4x4 + 4x4.
        # This doesn't form a 2x2 block structure directly.
        
        # Let's pad to 8x8 x 8x8 x 8x8?
        # 3->4, 5->8, 7->8.
        # 4x8 x 8x8.
        # Pad 4 to 8? 8x8 x 8x8.
        # Rank of 8x8x8 Strassen is 343.
        # Schoolbook is 512.
        # 343 < 512.
        
        # So if we pad 3x5 x 5x7 to 8x8 x 8x8, we get rank 343.
        # 343 < 105? No, 343 > 105.
        # So padding to 8x8 is worse than schoolbook.
        
        # What if we pad to 4x4 x 4x4?
        # 3->4, 5->4 (lose data!), 7->4 (lose data!).
        # We can't lose data.
        
        # So we must pad to at least 4x5 x 5x7?
        # 4 is next pow2 of 3.
        # 5 is not pow2. Next is 8.
        # 7 is not pow2. Next is 8.
        
        # So we must pad to 4x8 x 8x8.
        # How to compute 4x8 x 8x8 with low rank?
        
        # We can split 4x8 into 2x4 + 2x4.
        # And 8x8 into 4x4 + 4x4.
        # C = A1*B1 + A2*B2 where A1 is 2x4, A2 is 2x4, B1 is 4x4, B2 is 4x4.
        # Rank(2x4 x 4x4) + Rank(2x4 x 4x4).
        
        # How to compute 2x4 x 4x4?
        # Pad 2 to 4? 4x4 x 4x4. Rank 49 (Strassen).
        # But we lose data? No, 2x4 can be embedded in 4x4.
        # So Rank(2x4 x 4x4) <= 49.
        # Total rank <= 49 + 49 = 98.
        
        # 98 < 105.
        # So this should work!
        
        # Let's implement this.
        
        pass

def _construct_2x4_4x4_strassen():
    """
    Construct a rank-49 decomposition for 2x4 x 4x4.
    We pad 2x4 to 4x4.
    Then use Strassen 4x4x4.
    """
    # Strassen 4x4x4 has rank 49.
    # We need to generate the 49 terms.
    
    # This is complex to generate manually.
    # Let's use a known list or generate it recursively.
    
    # For now, let's return a placeholder that will be filled by a recursive generator.
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # Strategy:
    # 1. Pad n=3 to 4, m=5 to 8, p=7 to 8.
    # 2. Split 4x8 into two 2x4 blocks.
    # 3. Split 8x8 into two 4x4 blocks.
    # 4. Compute 2x4 * 4x4 using padded 4x4 * 4x4 (Strassen, rank 49).
    # 5. Total rank = 49 + 49 = 98.
    
    # We need a generator for Strassen 4x4x4 terms.
    # Let's implement a recursive Strassen term generator.
    
    def generate_strassen_terms(n, m, p, i_off=0, j_off=0, k_off=0, depth=0):
        """
        Generate terms for n x m * m x p multiplication using Strassen.
        n, m, p must be powers of 2.
        Returns a list of (u_dict, v_dict, w_dict).
        """
        if n == 1 and m == 1 and p == 1:
            return [({(i_off, j_off): 1}, {(j_off, k_off): 1}, {(k_off, i_off): 1})]
        
        n1 = n // 2
        m1 = m // 2
        p1 = p // 2
        
        # Offsets for the 4 blocks
        # A: (0,0), (0,m1), (n1,0), (n1,m1)
        # B: (0,0), (0,p1), (m1,0), (m1,p1)
        # C: (0,0), (0,p1), (n1,0), (n1,p1)
        
        # We need to generate terms for the 7 Strassen products.
        # Each product is a multiplication of size n1 x m1 * m1 x p1.
        
        # But the inputs are linear combinations.
        # This means we cannot just call generate_strassen_terms for the sub-blocks
        # and combine them, because the u, v, w vectors are summed.
        
        # Instead, we observe that the set of terms for (A11+A22)*(B11+B22) is the
        # "sum" of the terms for A11*B11, A11*B22, A22*B11, A22*B22, with signs.
        # This is not true for rank-1 terms.
        
        # Correct approach:
        # The rank of the tensor T is the minimal number of rank-1 terms.
        # Strassen's algorithm provides a specific set of 7^d rank-1 terms for 2^d x 2^d x 2^d.
        
        # Let's hardcode the 49 terms for 4x4x4 or generate them via a known method.
        
        # For simplicity, let's return the schoolbook terms for now if we can't generate Strassen.
        if depth > 2:
            terms = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        terms.append(({(i_off+i, j_off+j): 1}, {(j_off+j, k_off+k): 1}, {(k_off+k, i_off+i): 1}))
            return terms
        
        # This recursive generation of Strassen terms is non-trivial.
        # Let's fall back to a known construction for 4x4x4.
        
        # Actually, let's just return the schoolbook for the sub-blocks and see if we can merge.
        # No, that won't reduce rank.
        
        # Let's try a different strategy:
        # Use the known fact that M(4,4,4) <= 49.
        # We can find a 49-term decomposition online or derive it.
        
        # For now, let's return the schoolbook for the whole problem to ensure correctness.
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms.append(({(i_off+i, j_off+j): 1}, {(j_off+j, k_off+k): 1}, {(k_off+k, i_off+i): 1}))
        return terms
    
    # Since generating Strassen terms is complex, let's try a simpler approach:
    # Use the 2x2x2 Strassen construction directly on the 3x5x5x7 problem
    # by splitting into 2x2 blocks where possible.
    
    # 3x5 x 5x7.
    # Split 3 into 2+1.
    # Split 5 into 2+2+1.
    # Split 7 into 2+2+2+1.
    
    # This is getting too complex.
    
    # Let's just return the schoolbook.
    result = _construct_schoolbook()
    return result

def _construct_schoolbook():
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

# EVOLVE-BLOCK-END
