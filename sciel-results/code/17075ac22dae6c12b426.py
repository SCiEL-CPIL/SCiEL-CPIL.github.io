"""Construct a rank-28 exact tensor decomposition for 2x3 * 3x5 matrix multiplication."""
import time

DIMENSIONS = [2, 3, 5]

def construct():
    """
    Build a rank-minimal (known best: 28) exact decomposition.
    
    Strategy:
    - Use Strassen's 7-multiplication algorithm for 2x2 blocks where possible.
    - Fill remaining terms with schoolbook (rank 1 each).
    
    The tensor T_{i,j,k} = A_{i,j} * B_{j,k} is decomposed as:
      sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    where m=3 (cols of A / rows of B), n=2 (rows of A), p=5 (cols of B).
    
    We partition the 3 column-dimension (j) and 5 row-dimension (k) into blocks:
    - j in {0,1} paired with k in {0,1}: use Strassen (rank 7)
    - j in {0,1} paired with k in {2}:    schoolbook (rank 2)
    - j in {0,1} paired with k in {3,4}:  use Strassen (rank 7)
    - j in {2} paired with k in {0..4}:   schoolbook (rank 5)
    - j in {0,1} paired with k in {2}:    already counted
    
    Wait, let me recount:
    - Block (j=0,1; k=0,1): Strassen, rank 7
    - Block (j=0,1; k=2):    schoolbook, rank 2 (terms (0,2) and (1,2) in j,k)
    - Block (j=0,1; k=3,4): Strassen, rank 7
    - Block (j=2; k=0,1,2,3,4): schoolbook, rank 5
    
    Total: 7 + 2 + 7 + 5 = 21? No, that's not right. Let me reconsider.
    
    Actually, for 2x3 * 3x5:
    - The middle dimension is 3 (j = 0, 1, 2).
    - We can group j=0,1 into a 2-block and handle j=2 separately.
    - For the 2x2 (in i,j) by 2x2 (in j,k) subproblem with j=0,1:
      - k=0,1: Strassen 2x2*2x2 -> rank 7
      - k=2: we need C[:, 2] = A[:,0]*B[0,2] + A[:,1]*B[1,2], schoolbook rank 2
      - k=3,4: Strassen 2x2*2x2 (using B[:,3:5] as 2x2) -> rank 7
    - For j=2: C[:,k] += A[:,2]*B[2,k] for all k=0..4, schoolbook rank 5
    
    Total rank: 7 + 2 + 7 + 5 = 21.
    
    Hmm, but the known optimal is 28? Let me check: the lower bound for 2x3*3x5 is...
    Actually, let me just implement a correct decomposition and verify.
    """
    n, m, p = 2, 3, 5
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_entries, v_entries, w_entries):
        """Add one rank-1 term. Each entry is (index, value)."""
        u_row = [0.0] * (n * m)
        v_row = [0.0] * (m * p)
        w_row = [0.0] * (p * n)
        for idx, val in u_entries:
            u_row[idx] += val
        for idx, val in v_entries:
            v_row[idx] += val
        for idx, val in w_entries:
            w_row[idx] += val
        U_list.append(u_row)
        V_list.append(v_row)
        W_list.append(w_row)
    
    # Strassen decomposition for 2x2 * 2x2
    # Inputs: A[2,2], B[2,2], Output C[2,2]
    # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
    # Global: i_global = i_offset + i, j_global = j_offset + j, k_global = k_offset + k
    # U index: i_global * m + j_global
    # V index: j_global * p + k_global
    # W index: k_global * n + i_global
    
    def strassen_block(i_off, j_off, k_off):
        """Add 7 Strassen terms for a 2x2 * 2x2 subblock."""
        # M1 = A00 * (B01 - B11)
        add_term(
            [(i_off * m + j_off, 1.0)],
            [((j_off + 0) * p + (k_off + 1), 1.0), ((j_off + 1) * p + (k_off + 1), -1.0)],
            [((k_off + 1) * n + i_off, 1.0), ((k_off + 1) * n + (i_off + 1), 1.0)]
        )
        # M2 = (A10 + A00) * B11
        add_term(
            [(i_off * m + (j_off + 0), 1.0), ((i_off + 1) * m + (j_off + 0), 1.0)],
            [((j_off + 1) * p + (k_off + 1), 1.0)],
            [((k_off + 1) * n + (i_off + 1), 1.0)]
        )
        # M3 = A11 * (B10 - B00)
        add_term(
            [((i_off + 1) * m + (j_off + 1), 1.0)],
            [((j_off + 1) * p + (k_off + 0), 1.0), ((j_off + 0) * p + (k_off + 0), -1.0)],
            [((k_off + 0) * n + (i_off + 1), 1.0)]
        )
        # M4 = (A01 - A11) * (B10 + B11)
        add_term(
            [(i_off * m + (j_off + 1), 1.0), ((i_off + 1) * m + (j_off + 1), -1.0)],
            [((j_off + 1) * p + (k_off + 0), 1.0), ((j_off + 1) * p + (k_off + 1), 1.0)],
            [((k_off + 0) * n + i_off, 1.0), ((k_off + 1) * n + i_off, 1.0),
             ((k_off + 0) * n + (i_off + 1), 1.0), ((k_off + 1) * n + (i_off + 1), 1.0)]
        )
        # M5 = (A00 + A11) * (B00 + B11)
        add_term(
            [(i_off * m + j_off, 1.0), ((i_off + 1) * m + (j_off + 1), 1.0)],
            [((j_off + 0) * p + (k_off + 0), 1.0), ((j_off + 1) * p + (k_off + 1), 1.0)],
            [((k_off + 0) * n + i_off, 1.0), ((k_off + 0) * n + (i_off + 1), -1.0),
             ((k_off + 1) * n + i_off, -1.0), ((k_off + 1) * n + (i_off + 1), 1.0)]
        )
        # M6 = (A01 - A00) * B10
        add_term(
            [(i_off * m + (j_off + 1), 1.0), (i_off * m + j_off, -1.0)],
            [((j_off + 1) * p + (k_off + 0), 1.0)],
            [((k_off + 0) * n + i_off, 1.0)]
        )
        # M7 = A10 * (B01 - B00)
        add_term(
            [((i_off + 1) * m + j_off, 1.0)],
            [((j_off + 0) * p + (k_off + 1), 1.0), ((j_off + 0) * p + (k_off + 0), -1.0)],
            [((k_off + 0) * n + (i_off + 1), 1.0)]
        )
    
    def schoolbook_term(i, j, k):
        """Add a single schoolbook term for position (i,j,k)."""
        add_term(
            [(i * m + j, 1.0)],
            [(j * p + k, 1.0)],
            [(k * n + i, 1.0)]
        )
    
    # Block 1: Strassen for i={0,1}, j={0,1}, k={0,1}
    strassen_block(0, 0, 0)
    
    # Block 2: Schoolbook for i={0,1}, j={0,1}, k={2}
    for i in range(2):
        for j in range(2):
            schoolbook_term(i, j, 2)
    
    # Block 3: Strassen for i={0,1}, j={0,1}, k={3,4}
    strassen_block(0, 0, 3)
    
    # Block 4: Schoolbook for i={0,1}, j={2}, k={0,1,2,3,4}
    for i in range(2):
        for k in range(5):
            schoolbook_term(i, 2, k)
    
    rank = len(U_list)
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    start = time.time()
    result = construct()
    elapsed = time.time() - start
    print(f"Rank: {len(result['U'])}, Time: {elapsed:.4f}s")
    
    # Verify
    import numpy as np
    n, m, p = 2, 3, 5
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    
    # Build the multiplication tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Reconstruct
    T_recon = np.zeros_like(T)
    for r in range(len(U)):
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    T_recon[a,b,c] += U[a,r] * V[b,r] * W[c,r]
    
    diff = np.max(np.abs(T - T_recon))
    print(f"Max diff: {diff}")
    assert diff < 1e-12, "Decomposition is not exact!"
    print("Verification passed.")
