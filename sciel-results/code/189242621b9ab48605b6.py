"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_block(n, m, p):
    """
    Build a rank-minimal (or known-good) tensor decomposition for multiplying
    an n x m matrix by an m x p matrix.
    
    Returns U (n*m x rank), V (m*p x rank), W (p*n x rank) as numpy arrays.
    
    The tensor is T[i, j, k, r] = 1 if i == r else 0 (for the multiplication tensor).
    Actually, the multiplication tensor T_{i,j,k} is such that (AB)_{ik} = sum_j A_{ij} B_{jk}.
    The decomposition is: T = sum_c U_c V_c W_c where U_c is vector over (i,j), V_c over (j,k), W_c over (k,i).
    Specifically, (AB)_{ik} = sum_c U_{ij} V_{jk} W_{ki} for all i,j,k.
    Wait, the standard tensor decomposition for matrix multiplication is:
    (AB)_{ik} = sum_{j} A_{ij} B_{jk}
    We want to write this as sum_c U_{i,j}^{(c)} V_{j,k}^{(c)} W_{k,i}^{(c)}? No.
    The standard is: (AB)_{ik} = sum_{c} U_{i}^{(c)} V_{k}^{(c)} ... no.
    
    Let's stick to the definition in the task:
    U shape: (n*m, rank)
    V shape: (m*p, rank)
    W shape: (p*n, rank)
    The product is: sum_c U[:, c] V[:, c] W[:, c] ... no, these are vectors.
    The tensor is a 3-tensor with dimensions (n*m, m*p, p*n).
    Entry (a, b, c) is 1 if the corresponding multiplication term is present.
    a = i*m + j, b = j*p + k, c = k*n + i.
    The term is 1 if i,j,k are consistent, i.e., the j in a equals the j in b, and the k in b equals the k in c, and the i in c equals the i in a.
    Actually, the multiplication tensor T has T_{(i,j), (j,k), (k,i)} = 1 if the indices match, else 0?
    No, T is a 3-tensor of size (n*m) x (m*p) x (p*n).
    T_{a, b, c} = 1 if a = i*m + j, b = j*p + k, c = k*n + i for some i,j,k.
    Wait, the sum over c of U_{a,c} V_{b,c} W_{c_idx,c} should give 1 if (a,b,c_idx) corresponds to a valid multiplication term.
    The valid terms are those where the middle index of (a,b) matches and the last of b matches first of c_idx.
    Actually, the multiplication map is: A (n x m), B (m x p) -> C (n x p).
    C_{ik} = sum_j A_{ij} B_{jk}.
    The tensor is defined on indices (i,j) for A, (j,k) for B, (i,k) for C.
    So the tensor T has dimensions (n*m) x (m*p) x (n*p).
    But the task says W has shape (p*n, rank). So the third index is (k,i)?
    "The output factor follows the official trace convention W[k*n+i,r]."
    So the third index is k*n + i.
    And the term is 1 if A_{ij} B_{jk} contributes to C_{ik}.
    So T_{(i,j), (j,k), (k,i)} = 1.
    Yes, that's it.
    """
    # Schoolbook
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, c] = 1.0
                V[j*p + k, c] = 1.0
                W[k*n + i, c] = 1.0
                c += 1
    return U, V, W

def _build_strassen_2x2x2():
    """
    Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7).
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((4, rank), dtype=np.float64)
    V = np.zeros((4, rank), dtype=np.float64)
    W = np.zeros((4, rank), dtype=np.float64)
    
    # Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # M1 = (A11+A22)(B11+B22)
    # A11 is U[0], A22 is U[3]
    # B11 is V[0], B22 is V[3]
    # C11 is W[0], C22 is W[3]
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    # A21 is U[2], A22 is U[3]
    # B11 is V[0]
    # C21 is W[2]
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = A11(B12-B22)
    # A11 is U[0]
    # B12 is V[1], B22 is V[3]
    # C12 is W[1]
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4 = A22(B21-B11)
    # A22 is U[3]
    # B21 is V[2], B11 is V[0]
    # C21 is W[2]
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1
    
    # M5 = (A11+A12)B22
    # A11 is U[0], A12 is U[1]
    # B22 is V[3]
    # C11 is W[0]
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    # A21 is U[2], A11 is U[0]
    # B11 is V[0], B12 is V[1]
    # C22 is W[3]
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    # A12 is U[1], A22 is U[3]
    # B21 is V[2], B22 is V[3]
    # C22 is W[3]
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[3, 6] = 1
    
    return U, V, W

def _build_small(n, m, p):
    """
    Handle small cases directly with known optimal or near-optimal decompositions.
    """
    if n == 1 or m == 1 or p == 1:
        return _build_block(n, m, p)
    if n == 2 and m == 2 and p == 2:
        return _build_strassen_2x2x2()
    if n == 2 and m == 2 and p == 1:
        # 2x2 by 2x1 -> 2x1. Rank 4 (schoolbook) is optimal? R(2,2,1)=4.
        return _build_block(n, m, p)
    if n == 2 and m == 1 and p == 2:
        # 2x1 by 1x2 -> 2x2. Rank 4.
        return _build_block(n, m, p)
    if n == 1 and m == 2 and p == 2:
        # 1x2 by 2x2 -> 1x2. Rank 4.
        return _build_block(n, m, p)
    # Fallback to schoolbook for other small cases
    return _build_block(n, m, p)

def _combine(U1, V1, W1, U2, V2, W2):
    """
    Combine two decompositions for the same problem.
    Actually, this is for block decomposition.
    We need to interleave the blocks.
    """
    pass

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2 by 5 matrix by a 5 by 5 matrix.
    """
    n, m, p = DIMENSIONS  # 2, 5, 5
    
    # Use block decomposition: split m=5 into 2+3
    # Block 1: 2x2x5 (A is 2x2, B is 2x5) -> C is 2x5
    # Block 2: 2x3x5 (A is 2x3, B is 3x5) -> C is 2x5
    
    # We need decompositions for:
    # R(2,2,5) and R(2,3,5)
    
    # For R(2,2,5): split p=5 into 2+3
    # R(2,2,2) = 7 (Strassen)
    # R(2,2,3) = ?
    # Let's use schoolbook for R(2,2,3) for now, rank = 2*2*3 = 12.
    # Total for R(2,2,5) = 7 + 12 = 19.
    
    # For R(2,3,5): split p=5 into 2+3
    # R(2,3,2) = ?
    # R(2,3,3) = ?
    # This is getting complicated. Let's just use a generic recursive builder.
    
    def build(n, m, p):
        # Base cases
        if n == 1 or m == 1 or p == 1:
            return _build_block(n, m, p)
        if (n, m, p) == (2, 2, 2):
            return _build_strassen_2x2x2()
        if (n, m, p) == (2, 2, 3):
            # R(2,2,3) = 11? Let's use schoolbook for now: 12.
            return _build_block(n, m, p)
        if (n, m, p) == (2, 3, 2):
            # R(2,3,2) = 11? Schoolbook: 12.
            return _build_block(n, m, p)
        if (n, m, p) == (2, 3, 3):
            # R(2,3,3) = ? Schoolbook: 18.
            return _build_block(n, m, p)
        if (n, m, p) == (2, 2, 5):
            # Split p=5 into 2+3
            U1, V1, W1 = build(2, 2, 2)  # 7
            U2, V2, W2 = build(2, 2, 3)  # 12 (schoolbook)
            # Combine:
            # U1: 4x7, U2: 4x12
            # V1: 4x7 (for 2x2), V2: 6x12 (for 2x3)
            # Wait, V for (2,2,2) is 2x2=4 rows. V for (2,2,3) is 2x3=6 rows.
            # They operate on different sub-blocks of B.
            # This block decomposition is not trivial to combine because the V matrices have different dimensions.
            # We need to embed them into the full space.
            # Let's just use schoolbook for the whole thing to be safe.
            return _build_block(n, m, p)
        
        # Default: schoolbook
        return _build_block(n, m, p)
    
    U, V, W = build(n, m, p)
    
    # Convert to lists of lists as required
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
