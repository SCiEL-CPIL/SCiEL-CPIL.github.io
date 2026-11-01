import time
import numpy as np
from itertools import product

_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0

def _time_remaining():
    return _DEADLINE - time.time()

def _build_target_tensor():
    """Build the (2,2,7) matrix multiplication tensor T[i,j,k] where C[i,k] = sum_j A[i,j]B[j,k]."""
    n, m, p = 2, 2, 7
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T=None):
    """Verify U, V, W decompose the target tensor exactly."""
    if T is None:
        T = _build_target_tensor()
    T_rec = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_rec, T, atol=1e-10)

def _schoolbook_decomposition():
    """Fallback rank-28 schoolbook decomposition."""
    n, m, p = 2, 2, 7
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, idx] = 1
                V[j*p + k, idx] = 1
                W[k*n + i, idx] = 1
                idx += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _strassen_2x2_decomposition():
    """
    Strassen decomposition for 2x2 matrix multiplication.
    Returns rank-7 decomposition for multiplying 2x2 by 2x2.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    
    Indices:
    A: a00=0, a01=1, a10=2, a11=3
    B: b00=4, b01=5, b10=6, b11=7
    C: c00=8, c01=9, c10=10, c11=11
    
    Strassen:
    M1 = (a00+a11)*(b00+b11)
    M2 = (a10+a11)*b00
    M3 = a00*(b01-b11)
    M4 = a11*(b10-b00)
    M5 = (a00+a01)*b11
    M6 = (a10-a00)*(b00+b01)
    M7 = (a01-a11)*(b10+b11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # For 2x2 * 2x2 = 2x2
    # U: A factors (4, 7)
    # V: B factors (4, 7)
    # W: C factors (4, 7)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a00+a11)*(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[4, 0] = 1; V[7, 0] = 1
    W[8, 0] = 1; W[11, 0] = 1
    
    # M2 = (a10+a11)*b00
    U[2, 1] = 1; U[3, 1] = 1
    V[4, 1] = 1
    W[10, 1] = 1; W[11, 1] = -1
    
    # M3 = a00*(b01-b11)
    U[0, 2] = 1
    V[5, 2] = 1; V[7, 2] = -1
    W[9, 2] = 1; W[11, 2] = 1
    
    # M4 = a11*(b10-b00)
    U[3, 3] = 1
    V[6, 3] = 1; V[4, 3] = -1
    W[8, 3] = 1; W[10, 3] = 1
    
    # M5 = (a00+a01)*b11
    U[0, 4] = 1; U[1, 4] = 1
    V[7, 4] = 1
    W[8, 4] = -1; W[9, 4] = 1
    
    # M6 = (a10-a00)*(b00+b01)
    U[2, 5] = 1; U[0, 5] = -1
    V[4, 5] = 1; V[5, 5] = 1
    W[11, 5] = 1
    
    # M7 = (a01-a11)*(b10+b11)
    U[1, 6] = 1; U[3, 6] = -1
    V[6, 6] = 1; V[7, 6] = 1
    W[8, 6] = 1
    
    return U, V, W

def _construct_2x2_2x7_decomposition():
    """
    Construct a low-rank decomposition for 2x2 * 2x7 = 2x7.
    
    Key insight: We can decompose the 2x7 matrix B into blocks and use Strassen
    for the 2x2 part, combined with linear combinations for the 7 columns.
    
    Another approach: Use the fact that rank(2,2,p) is known to be low.
    For p=7, we can try to find a decomposition by combining:
    - Strassen for the 2x2 core (rank 7)
    - Additional terms for the extra columns
    
    Actually, let's think more carefully. The tensor for 2x2 * 2x7 has:
    - 4 inputs from A
    - 14 inputs from B
    - 14 outputs from C
    
    We can view this as: for each column k of B (k=0..6), we compute
    C[:,k] = A @ B[:,k]
    
    So we have 7 independent 2x2 * 2x1 multiplications, but they share the same A.
    
    A better structural approach: 
    C = A @ B where A is 2x2, B is 2x7.
    
    We can write B as [B1 | B2 | ... | B7] where each Bj is 2x1.
    Then C = [A@B1 | A@B2 | ... | A@B7].
    
    Using Strassen for 2x2 multiplication:
    A = [[a00, a01], [a10, a11]]
    Bj = [bj0, bj1]^T
    
    A @ Bj = [a00*bj0 + a01*bj1, a10*bj0 + a11*bj1]^T
    
    This is a bilinear map. For each column, we need 2 multiplications (schoolbook).
    With Strassen, we can do 2x2 * 2x1 with rank... hmm, that's not directly Strassen.
    
    Let me try a different approach: direct tensor decomposition search.
    
    Actually, the known result is that rank(2,2,7) = 11 or similar.
    Let me try to construct it using the "wedge" or "subspace" method.
    
    Alternative: Use the fact that the multiplication tensor for (2,2,7) can be
    decomposed by considering the 7 columns of B.
    
    Let me try a structured decomposition:
    
    For 2x2 * 2x7, we can use:
    - 7 terms for the "diagonal" (a00*bj0, a11*bj1 for each j)
    - Additional terms for off-diagonal (a01*bj1, a10*bj0)
    
    But that gives 28 terms. We need to do better.
    
    Let me try using Strassen-like combinations across columns.
    
    Actually, I recall that for (n,m,p) matrix multiplication, the tensor rank
    can be bounded using the "border rank" or specific constructions.
    
    For (2,2,7), one known approach:
    - Use 7 Strassen decompositions, one for each pair of columns
    - But share the A-factors
    
    Let me try a concrete construction. Define the following rank terms:
    
    Term 0: (a00+a11) * (b00+b10+...+b60) -> contributes to c0k for all k
    This doesn't work cleanly.
    
    Let me try a different structural idea:
    
    We can write the 2x7 multiplication as:
    C[0,k] = a00*B[0,k] + a01*B[1,k]
    C[1,k] = a10*B[0,k] + a11*B[1,k]
    
    So for each k, we have a 2D vector [C[0,k], C[1,k]] = A @ [B[0,k], B[1,k]]
    
    This is 7 independent applications of the 2x2 linear map A.
    
    The tensor rank of 2x2 * 2x7 is at most 7 * rank(2x2 * 2x1) = 7 * 2 = 14 (schoolbook per column)
    Or 7 * 7 = 49 if we use full Strassen per column (wasteful).
    
    For 2x2 * 2x1, the rank is 2 (just two multiplications: a00*b0, a01*b1 for top, etc.)
    Wait, 2x2 * 2x1:
    C[0] = a00*b0 + a01*b1
    C[1] = a10*b0 + a11*b1
    
    This has tensor rank 4 (schoolbook: a00*b0, a01*b1, a10*b0, a11*b1)
    But can it be done with rank 3? Using Strassen for 2x2*2x1?
    
    Actually, 2x2 * 2x1 is a special case. The tensor has:
    Inputs: a00, a01, a10, a11, b0, b1
    Outputs: c0, c1
    
    c0 = a00*b0 + a01*b1
    c1 = a10*b0 + a11*b1
    
    This can be decomposed as:
    Term 0: a00 * b0 -> c0
    Term 1: a01 * b1 -> c0
    Term 2: a10 * b0 -> c1
    Term 3: a11 * b1 -> c1
    
    Rank 4. Can we do rank 3?
    
    Using Strassen-like: 
    Let M1 = (a00+a11) * b0
    Let M2 = a01 * b1
    Let M3 = a10 * b0
    
    c0 = M1 + M2 - a11*b0 ... no, this doesn't close.
    
    Actually, for 2x2 * 2x1, the tensor rank is known to be 4.
    
    So for 2x2 * 2x7, if we do it column by column, rank = 7*4 = 28.
    
    But we can share structure across columns!
    
    Key insight: The A-factors are shared across all columns.
    
    Let me try this construction:
    
    For each column k, we have:
    C[0,k] = a00*B[0,k] + a01*B[1,k]
    C[1,k] = a10*B[0,k] + a11*B[1,k]
    
    Define:
    U0 = a00 + a11
    U1 = a00
    U2 = a01
    U3 = a10
    U4 = a11
    
    Then for column k:
    C[0,k] = U1*B[0,k] + U2*B[1,k]
    C[1,k] = U3*B[0,k] + U4*B[1,k]
    
    This is still 4 per column = 28 total.
    
    Let me try Strassen across columns:
    
    Group columns into pairs: (0,1), (2,3), (4,5), and column 6 alone.
    
    For columns (2k, 2k+1):
    C[0,2k] = a00*B[0,2k] + a01*B[1,2k]
    C[0,2k+1] = a00*B[0,2k+1] + a01*B[1,2k+1]
    C[1,2k] = a10*B[0,2k] + a11*B[1,2k]
    C[1,2k+1] = a10*B[0,2k+1] + a11*B[1,2k+1]
    
    This is like multiplying A (2x2) by B_block (2x2).
    Using Strassen for this 2x2 * 2x2 block, we get rank 7 per block.
    3 blocks = 21, plus 1 column = 4, total = 25.
    
    Better than 28!
    
    Let me implement this:
    - 3 Strassen decompositions for column pairs (0,1), (2,3), (4,5)
    - 1 schoolbook decomposition for column 6
    - Total rank: 3*7 + 4 = 25
    """
    n, m, p = 2, 2, 7
    rank = 25
    U = np.zeros((n*m, rank))  # (4, 25)
    V = np.zeros((m*p, rank))  # (14, 25)
    W = np.zeros((p*n, rank))  # (14, 25)
    
    def _add_strassen_block(col_start, base_idx):
        """Add a Strassen decomposition for columns col_start and col_start+1."""
        # A indices: a00=0, a01=1, a10=2, a11=3
        # B indices for columns col_start, col_start+1:
        #   b0_col_start = 0*p + col_start = col_start
        #   b1_col_start = p + col_start = 7 + col_start
        #   b0_col_start+1 = col_start + 1
        #   b1_col_start+1 = 7 + col_start + 1
        # C indices:
        #   c0_col_start = col_start * n + 0 = col_start * 2
        #   c1_col_start = col_start * n + 1 = col_start * 2 + 1
        #   c0_col_start+1 = (col_start+1) * 2
        #   c1_col_start+1 = (col_start+1) * 2 + 1
        
        b00 = col_start
        b01 = col_start + 1
        b10 = 7 + col_start
        b11 = 7 + col_start + 1
        c00 = col_start * 2
        c01 = (col_start + 1) * 2
        c10 = col_start * 2 + 1
        c11 = (col_start + 1) * 2 + 1
        
        # M1 = (a00+a11)*(b00+b11)
        U[0, base_idx+0] = 1; U[3, base_idx+0] = 1
        V[b00, base_idx+0] = 1; V[b11, base_idx+0] = 1
        W[c00, base_idx+0] = 1; W[c11, base_idx+0] = 1
        
        # M2 = (a10+a11)*b00
        U[2, base_idx+1] = 1; U[3, base_idx+1] = 1
        V[b00, base_idx+1] = 1
        W[c10, base_idx+1] = 1; W[c11, base_idx+1] = -1
        
        # M3 = a00*(b01-b11)
        U[0, base_idx+2] = 1
        V[b01, base_idx+2] = 1; V[b11, base_idx+2] = -1
        W[c01, base_idx+2] = 1; W[c11, base_idx+2] = 1
        
        # M4 = a11*(b10-b00)
        U[3, base_idx+3] = 1
        V[b10, base_idx+3] = 1; V[b00, base_idx+3] = -1
        W[c00, base_idx+3] = 1; W[c10, base_idx+3] = 1
        
        # M5 = (a00+a01)*b11
        U[0, base_idx+4] = 1; U[1, base_idx+4] = 1
        V[b11, base_idx+4] = 1
        W[c00, base_idx+4] = -1; W[c01, base_idx+4] = 1
        
        # M6 = (a10-a00)*(b00+b01)
        U[2, base_idx+5] = 1; U[0, base_idx+5] = -1
        V[b00, base_idx+5] = 1; V[b01, base_idx+5] = 1
        W[c11, base_idx+5] = 1
        
        # M7 = (a01-a11)*(b10+b11)
        U[1, base_idx+6] = 1; U[3, base_idx+6] = -1
        V[b10, base_idx+6] = 1; V[b11, base_idx+6] = 1
        W[c00, base_idx+6] = 1
    
    # Add Strassen blocks for column pairs (0,1), (2,3), (4,5)
    _add_strassen_block(0, 0)
    _add_strassen_block(2, 7)
    _add_strassen_block(4, 14)
    
    # Schoolbook for column 6
    # C[0,6] = a00*B[0,6] + a01*B[1,6]
    # C[1,6] = a10*B[0,6] + a11*B[1,6]
    b0_6 = 6
    b1_6 = 13
    c0_6 = 12
    c1_6 = 13
    
    # Term 21: a00 * b0_6 -> c0_6
    U[0, 21] = 1
    V[b0_6, 21] = 1
    W[c0_6, 21] = 1
    
    # Term 22: a01 * b1_6 -> c0_6
    U[1, 22] = 1
    V[b1_6, 22] = 1
    W[c0_6, 22] = 1
    
    # Term 23: a10 * b0_6 -> c1_6
    U[2, 23] = 1
    V[b0_6, 23] = 1
    W[c1_6, 23] = 1
    
    # Term 24: a11 * b1_6 -> c1_6
    U[3, 24] = 1
    V[b1_6, 24] = 1
    W[c1_6, 24] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _construct_2x2_2x7_rank21():
    """
    Try to get rank 21 by using Strassen for all 7 columns with shared structure.
    
    Idea: Use 3 Strassen blocks for pairs (0,1), (2,3), (4,5) = 21 terms,
    and fold column 6 into one of the blocks.
    
    Actually, let me try a different grouping:
    - Block 1: columns 0,1 -> Strassen (7 terms)
    - Block 2: columns 2,3,4 -> ? 
    - Block 3: columns 5,6 -> Strassen (7 terms)
    
    For 2x2 * 2x3, what's the rank?
    C[0,0] = a00*B[0,0] + a01*B[1,0]
    C[0,1] = a00*B[0,1] + a01*B[1,1]
    C[0,2] = a00*B[0,2] + a01*B[1,2]
    C[1,0] = a10*B[0,0] + a11*B[1,0]
    C[1,1] = a10*B[0,1] + a11*B[1,1]
    C[1,2] = a10*B[0,2] + a11*B[1,2]
    
    This is 2x2 * 2x3. Rank is at most 12 (schoolbook).
    Can we do better?
    
    Using Strassen-like:
    Let M1 = (a00+a11)*(B[0,0]+B[1,2])
    Let M2 = (a10+a11)*B[0,0]
    Let M3 = a00*(B[0,1]-B[1,2])
    Let M4 = a11*(B[1,0]-B[0,0])
    Let M5 = (a00+a01)*B[1,1]
    Let M6 = (a10-a00)*(B[0,0]+B[0,1])
    Let M7 = (a01-a11)*(B[1,0]+B[1,1])
    
    This doesn't cleanly cover all 6 outputs.
    
    Let me just go with rank 25 for now and try to improve.
    """
    pass

def _search_rank(T, rank, time_limit=5.0):
    """Try to find a rank-`rank` decomposition of T using ALS with better initialization."""
    n, m, p = 2, 2, 7
    dim_a, dim_b, dim_c = n*m, m*p, p*n  # 4, 14, 14
    
    start = time.time()
    best_residual = np.inf
    
    # Try multiple random initializations
    for trial in range(30):
        if time.time() - start > time_limit:
            break
        
        # Random initialization
        U = np.random.randn(dim_a, rank) * 0.5
        V = np.random.randn(dim_b, rank) * 0.5
        W = np.random.randn(dim_c, rank) * 0.5
        
        # ALS iterations
        for it in range(300):
            if time.time() - start > time_limit:
                break
            
            # Update U
            for r in range(rank):
                VW = np.outer(V[:, r], W[:, r])
                denom = np.sum(VW ** 2)
                if denom < 1e-12:
                    continue
                U[:, r] = np.einsum('abc,bc->a', T, VW) / denom
            
            # Update V
            for r in range(rank):
                UW = np.outer(U[:, r], W[:, r])
                denom = np.sum(UW ** 2)
                if denom < 1e-12:
                    continue
                V[:, r] = np.einsum('abc,ac->b', T, UW) / denom
            
            # Update W
            for r in range(rank):
                UV = np.outer(U[:, r], V[:, r])
                denom = np.sum(UV ** 2)
                if denom < 1e-12:
                    continue
                W[:, r] = np.einsum('abc,ab->c', T, UV) / denom
            
            # Check convergence
            T_rec = np.einsum('ar,br,cr->abc', U, V, W)
            residual = np.linalg.norm(T_rec - T)
            best_residual = min(best_residual, residual)
            
            if residual < 1e-10:
                if np.allclose(T_rec, T, atol=1e-10):
                    return U, V, W
        
        # Check final residual
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        if np.allclose(T_rec, T, atol=1e-10):
            return U, V, W
    
    return None

def construct():
    """
    Search for a minimal-rank exact tensor decomposition for 2x2 * 2x7.
    Uses Strassen block decomposition for rank 25, then tries to improve.
    """
    if _time_remaining() < 10.0:
        return _schoolbook_decomposition()
    
    T = _build_target_tensor()
    
    # First try the Strassen block decomposition (rank 25)
    strassen_result = _construct_2x2_2x7_decomposition()
    U_s = np.array(strassen_result['U'])
    V_s = np.array(strassen_result['V'])
    W_s = np.array(strassen_result['W'])
    
    if _verify_decomposition(U_s, V_s, W_s, T):
        best_rank = 25
        best_result = strassen_result
    else:
        best_rank = 28
        best_result = _schoolbook_decomposition()
    
    # Try to improve via search
    search_start = time.time()
    max_search_time = _time_remaining() - 15.0
    
    # Try ranks from best_rank-1 down to 11
    for rank in range(best_rank - 1, 10, -1):
        if time.time() - search_start > max_search_time:
            break
        
        # Allocate time per rank attempt
        time_per_rank = max(1.0, max_search_time / (best_rank - rank))
        
        result = _search_rank(T, rank, time_limit=time_per_rank)
        if result is not None:
            U, V, W = result
            T_rec = np.einsum('ar,br,cr->abc', U, V, W)
            if np.allclose(T_rec, T, atol=1e-10):
                best_rank = rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return best_result

if __name__ == "__main__":
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    A = np.random.rand(2, 2)
    B = np.random.rand(2, 7)
    C_expected = A @ B
    
    C_W = np.zeros(14)
    for r in range(U.shape[1]):
        SA = np.dot(U[:, r], A.flatten())
        SB = np.dot(V[:, r], B.flatten())
        for idx in range(14):
            C_W[idx] += W[idx, r] * SA * SB
            
    C_mat = np.zeros((2,7))
    for k in range(7):
        for i in range(2):
            C_mat[i, k] = C_W[2*k+i]
            
    if np.allclose(C_mat, C_expected):
        print("Verification PASSED")
    else:
        print("Verification FAILED")
        print(C_mat)
        print(C_expected)
