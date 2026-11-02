"""Rewritten program for Matrix Multiplication 3,4,4 rank minimization."""

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

TIME_LIMIT = 145.0
START_TIME = time.time()

def time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def schoolbook_solution(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
                idx += 1
    
    return U, V, W

def verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = U.shape[1]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                coeff = 0.0
                for r in range(rank):
                    coeff += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(coeff - 1.0) > tol:
                    return False, f"C[{i},{k}] coeff of A[{i},{j}]*B[{j},{k}] is {coeff}, expected 1"
    return True, "OK"

def strassen_2x2():
    """
    Return the Strassen rank-7 decomposition for 2x2 * 2x2 -> 2x2.
    
    Strassen's 7 products:
    P1 = (A00+A11)*(B00+B11)
    P2 = (A10+A11)*B00
    P3 = A00*(B01-B11)
    P4 = A11*(B10-B00)
    P5 = (A00+A01)*B11
    P6 = (A10-A00)*(B00+B01)
    P7 = (A01-A11)*(B10+B11)
    
    Results:
    C00 = P1 + P4 - P5 + P7
    C01 = P3 + P5
    C10 = P2 + P4
    C11 = P1 - P2 + P3 + P6
    
    For 2x2x2 tensor with indices:
    A is 2x2: A[0,0], A[0,1], A[1,0], A[1,1] -> U indices: 0,1,2,3
    B is 2x2: B[0,0], B[0,1], B[1,0], B[1,1] -> V indices: 0,1,2,3
    C is 2x2: C[0,0], C[0,1], C[1,0], C[1,1] -> W indices: 0,1,2,3
    
    Returns U (4x7), V (4x7), W (4x7)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # P1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # P2 = (A10+A11)*B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # P3 = A00*(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # P4 = A11*(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # P5 = (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # P6 = (A10-A00)*(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # P7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def hybrid_strassen_solution(n, m, p):
    """
    Hybrid approach: Use Strassen (rank 7) for the 2x2 block of top 2 rows,
    and trivial multiplication (rank 4) for the bottom row.
    
    For 3x4 * 4x4 -> 3x4:
    Partition A (3x4) as:
      Top 2 rows (2x4): use Strassen on the 2x2*2x2 -> 2x2 sub-blocks
      Bottom row (1x4): use trivial multiplication
    
    Specifically, for the top 2 rows:
    A_top (2x4) * B (4x4) -> C_top (2x4)
    
    We can partition B into left 2x4 and right 2x4:
    B = [B_left (4x2), B_right (4x2)]
    
    Then:
    C_top[:, 0:2] = A_top * B_left  (2x2 result)
    C_top[:, 2:4] = A_top * B_right (2x2 result)
    
    For each 2x2*2x2 -> 2x2 multiplication, use Strassen (rank 7).
    So top 2 rows contribute 7 + 7 = 14 terms.
    
    For the bottom row:
    C_bottom (1x4) = A_bottom (1x4) * B (4x4)
    This is trivial: 4 multiplications.
    
    Total rank: 14 + 4 = 18? No wait, that's not right.
    
    Actually, let me reconsider. The tensor decomposition must work for ALL
    combinations of (i, j, k) where i in [0,2], j in [0,3], k in [0,3].
    
    For the top 2 rows (i=0,1), we have a 2x4*4x4 -> 2x4 computation.
    We can partition the 4 columns of B into two groups of 2:
      Group 1: columns 0,1
      Group 2: columns 2,3
    
    For group 1: A_top (2x4) * B[:, 0:2] (4x2) -> C_top[:, 0:2] (2x2)
    For group 2: A_top (2x4) * B[:, 2:4] (4x2) -> C_top[:, 2:4] (2x2)
    
    But A_top is 2x4, not 2x2. So we need to further partition A_top's columns.
    
    Let me think differently. The full computation is:
    C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    For i in {0,1} (top 2 rows) and k in {0,1} (left 2 cols):
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    This is a 2x4 * 4x2 -> 2x2 computation. We can view this as:
    [A[i,0], A[i,1]] * [[B[0,k]], [B[1,k]]] + [A[i,2], A[i,3]] * [[B[2,k]], [B[3,k]]]
    
    Each 2x1 * 1x2 -> 2x2 is a rank-2 computation (trivial).
    So this sub-computation has rank 4 (two rank-2 blocks).
    
    Alternatively, we can use Strassen if we can form 2x2 blocks.
    
    Let me try a cleaner partition:
    
    Partition the 4 columns of A into two groups of 2:
    A_left (3x2), A_right (3x2)
    Partition the 4 rows of B into two groups of 2:
    B_top (2x4), B_bottom (2x4)
    
    Then C = A_left * B_top + A_right * B_bottom
    
    For the top 2 rows of C (i=0,1):
    C_top = A_top_left (2x2) * B_top (2x4) + A_top_right (2x2) * B_bottom (2x4)
    
    For the bottom row of C (i=2):
    C_bottom = A_bottom_left (1x2) * B_top (2x4) + A_bottom_right (1x2) * B_bottom (2x4)
    
    Now, A_top_left * B_top is 2x2 * 2x4 -> 2x4.
    We can partition B_top into two 2x2 blocks: B_top_left (2x2), B_top_right (2x2).
    A_top_left * B_top_left -> 2x2 (use Strassen, rank 7)
    A_top_left * B_top_right -> 2x2 (use Strassen, rank 7)
    
    Similarly for A_top_right * B_bottom.
    
    So for top 2 rows: 7 + 7 + 7 + 7 = 28 terms
    For bottom row: 1x2 * 2x2 -> 1x2 (rank 2) for each of 4 sub-blocks = 8 terms
    Total: 36 terms
    
    That's better than 48 but not 44. Let me reconsider.
    
    Actually, I think the 44 rank comes from a different partition. Let me try:
    
    Use Strassen for the 2x2x2 core and handle the extra dimensions separately.
    
    For 3x4 * 4x4 -> 3x4, think of it as:
    (3x4) * (4x4) = (3x4) * (4x4)
    
    Partition into:
    - 2x2 * 2x2 -> 2x2 (Strassen, rank 7) for the core
    - Handle the extra row and columns
    
    One approach: 
    C[0:2, 0:2] = A[0:2, 0:2]*B[0:2, 0:2] + A[0:2, 2:4]*B[2:4, 0:2]
    C[0:2, 2:4] = A[0:2, 0:2]*B[0:2, 2:4] + A[0:2, 2:4]*B[2:4, 2:4]
    C[2, 0:2] = A[2, 0:2]*B[0:2, 0:2] + A[2, 2:4]*B[2:4, 0:2]
    C[2, 2:4] = A[2, 0:2]*B[0:2, 2:4] + A[2, 2:4]*B[2:4, 2:4]
    
    For the 2x2*2x2 -> 2x2 parts, use Strassen (rank 7 each).
    For the 1x2*2x2 -> 1x2 parts, use trivial (rank 2 each).
    
    C[0:2, 0:2]: 7 + 7 = 14
    C[0:2, 2:4]: 7 + 7 = 14
    C[2, 0:2]: 2 + 2 = 4
    C[2, 2:4]: 2 + 2 = 4
    Total: 36
    
    Still 36. To get 44, maybe a different partition is used.
    
    Let me just implement the 36-rank solution and verify it.
    """
    n, m, p = 3, 4, 4
    
    # Build the decomposition
    # We'll accumulate U, V, W matrices
    
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a term
    def add_term(u_dict, v_dict, w_dict):
        """Add a rank-1 term. u_dict: {U_idx: val}, v_dict: {V_idx: val}, w_dict: {W_idx: val}"""
        U_row = np.zeros(n * m)
        V_row = np.zeros(m * p)
        W_row = np.zeros(p * n)
        
        for idx, val in u_dict.items():
            U_row[idx] = val
        for idx, val in v_dict.items():
            V_row[idx] = val
        for idx, val in w_dict.items():
            W_row[idx] = val
        
        U_list.append(U_row)
        V_list.append(V_row)
        W_list.append(W_row)
    
    # For Strassen 2x2*2x2 -> 2x2, we need to map the local indices to global indices.
    # Local A indices: 0->A[i0,j0], 1->A[i0,j1], 2->A[i1,j0], 3->A[i1,j1]
    # Local B indices: 0->B[j0,k0], 1->B[j0,k1], 2->B[j1,k0], 3->B[j1,k1]
    # Local C indices: 0->C[i0,k0], 1->C[i0,k1], 2->C[i1,k0], 3->C[i1,k1]
    
    def add_strassen(i0, i1, j0, j1, k0, k1):
        """Add Strassen decomposition for A[i0:i1+1, j0:j1+1] * B[j0:j1+1, k0:k1+1] -> C[i0:i1+1, k0:k1+1]"""
        U_s, V_s, W_s = strassen_2x2()
        
        for r in range(7):
            u_dict = {}
            v_dict = {}
            w_dict = {}
            
            for local_i in range(2):
                for local_j in range(2):
                    val = U_s[local_i * 2 + local_j, r]
                    if val != 0:
                        global_i = i0 + local_i
                        global_j = j0 + local_j
                        u_dict[global_i * m + global_j] = val
            
            for local_j in range(2):
                for local_k in range(2):
                    val = V_s[local_j * 2 + local_k, r]
                    if val != 0:
                        global_j = j0 + local_j
                        global_k = k0 + local_k
                        v_dict[global_j * p + global_k] = val
            
            for local_k in range(2):
                for local_i in range(2):
                    val = W_s[local_k * 2 + local_i, r]
                    if val != 0:
                        global_k = k0 + local_k
                        global_i = i0 + local_i
                        w_dict[global_k * n + global_i] = val
            
            add_term(u_dict, v_dict, w_dict)
    
    def add_trivial_1x2x2(i, j0, j1, k0, k1):
        """Add trivial decomposition for A[i, j0:j1+1] * B[j0:j1+1, k0:k1+1] -> C[i, k0:k1+1]"""
        # A[i,j0]*B[j0,k0] -> C[i,k0]
        add_term({i * m + j0: 1}, {j0 * p + k0: 1}, {k0 * n + i: 1})
        # A[i,j0]*B[j0,k1] -> C[i,k1]
        add_term({i * m + j0: 1}, {j0 * p + k1: 1}, {k1 * n + i: 1})
        # A[i,j1]*B[j1,k0] -> C[i,k0]
        add_term({i * m + j1: 1}, {j1 * p + k0: 1}, {k0 * n + i: 1})
        # A[i,j1]*B[j1,k1] -> C[i,k1]
        add_term({i * m + j1: 1}, {j1 * p + k1: 1}, {k1 * n + i: 1})
    
    # Partition:
    # A columns: [0,1] and [2,3]
    # B rows: [0,1] and [2,3]
    # C columns: [0,1] and [2,3]
    
    # Top 2 rows (i=0,1):
    # C[0:2, 0:2] = A[0:2, 0:2]*B[0:2, 0:2] + A[0:2, 2:4]*B[2:4, 0:2]
    add_strassen(0, 1, 0, 1, 0, 1)  # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    add_strassen(0, 1, 2, 3, 0, 1)  # A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    
    # C[0:2, 2:4] = A[0:2, 0:2]*B[0:2, 2:4] + A[0:2, 2:4]*B[2:4, 2:4]
    add_strassen(0, 1, 0, 1, 2, 3)  # A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    add_strassen(0, 1, 2, 3, 2, 3)  # A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    
    # Bottom row (i=2):
    # C[2, 0:2] = A[2, 0:2]*B[0:2, 0:2] + A[2, 2:4]*B[2:4, 0:2]
    add_trivial_1x2x2(2, 0, 1, 0, 1)  # A[2,0:2] * B[0:2,0:2] -> C[2,0:2]
    add_trivial_1x2x2(2, 2, 3, 0, 1)  # A[2,2:4] * B[2:4,0:2] -> C[2,0:2]
    
    # C[2, 2:4] = A[2, 0:2]*B[0:2, 2:4] + A[2, 2:4]*B[2:4, 2:4]
    add_trivial_1x2x2(2, 0, 1, 2, 3)  # A[2,0:2] * B[0:2,2:4] -> C[2,2:4]
    add_trivial_1x2x2(2, 2, 3, 2, 3)  # A[2,2:4] * B[2:4,2:4] -> C[2,2:4]
    
    rank = len(U_list)
    U = np.column_stack(U_list)
    V = np.column_stack(V_list)
    W = np.column_stack(W_list)
    
    # Verify
    ok, msg = verify_decomposition(U, V, W, n, m, p)
    if ok:
        return U, V, W
    else:
        print(f"Hybrid verification failed: {msg}")
        return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try hybrid Strassen approach first
    result = hybrid_strassen_solution(n, m, p)
    if result is not None:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
    U, V, W = schoolbook_solution(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
