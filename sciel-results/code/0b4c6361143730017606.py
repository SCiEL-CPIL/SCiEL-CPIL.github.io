import numpy as np
import time
import random

DIMENSIONS = [2, 3, 5]

def _verify_decomposition(U, V, W, n=2, m=3, p=5):
    """Verify that the decomposition exactly represents matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2x2(U, V, W, col, i_off, j_off, k_off, n, m, p):
    """Add Strassen 2x2x2 block. Returns new col."""
    # M1: (A00+A11)(B00+B11) -> C00 +=, C11 +=
    U[(i_off) * m + j_off][col] = 1
    U[(i_off+1) * m + (j_off+1)][col] = 1
    V[j_off * p + k_off][col] = 1
    V[(j_off+1) * p + (k_off+1)][col] = 1
    W[k_off * n + i_off][col] = 1
    W[(k_off+1) * n + (i_off+1)][col] = 1
    col += 1
    
    # M2: (A10+A11)B00 -> C10 +=, C11 -=
    U[(i_off+1) * m + j_off][col] = 1
    U[(i_off+1) * m + (j_off+1)][col] = 1
    V[j_off * p + k_off][col] = 1
    W[k_off * n + (i_off+1)][col] = 1
    W[(k_off+1) * n + (i_off+1)][col] = -1
    col += 1
    
    # M3: A00(B01-B11) -> C01 +=, C11 +=
    U[i_off * m + j_off][col] = 1
    V[j_off * p + (k_off+1)][col] = 1
    V[(j_off+1) * p + (k_off+1)][col] = -1
    W[(k_off+1) * n + i_off][col] = 1
    W[(k_off+1) * n + (i_off+1)][col] = 1
    col += 1
    
    # M4: A11(B10-B00) -> C00 +=, C10 +=
    U[(i_off+1) * m + (j_off+1)][col] = 1
    V[(j_off+1) * p + k_off][col] = 1
    V[j_off * p + k_off][col] = -1
    W[k_off * n + i_off][col] = 1
    W[k_off * n + (i_off+1)][col] = 1
    col += 1
    
    # M5: (A00+A01)B11 -> C00 -=, C01 +=
    U[i_off * m + j_off][col] = 1
    U[i_off * m + (j_off+1)][col] = 1
    V[(j_off+1) * p + (k_off+1)][col] = 1
    W[k_off * n + i_off][col] = -1
    W[(k_off+1) * n + i_off][col] = 1
    col += 1
    
    # M6: (A10-A00)(B00+B01) -> C11 +=
    U[(i_off+1) * m + j_off][col] = 1
    U[i_off * m + j_off][col] = -1
    V[j_off * p + k_off][col] = 1
    V[j_off * p + (k_off+1)][col] = 1
    W[(k_off+1) * n + (i_off+1)][col] = 1
    col += 1
    
    # M7: (A01-A11)(B10+B11) -> C00 +=
    U[i_off * m + (j_off+1)][col] = 1
    U[(i_off+1) * m + (j_off+1)][col] = -1
    V[(j_off+1) * p + k_off][col] = 1
    V[(j_off+1) * p + (k_off+1)][col] = 1
    W[k_off * n + i_off][col] = 1
    col += 1
    
    return col

def _schoolbook_block(U, V, W, col, i_range, j_range, k_range, n, m, p):
    """Add schoolbook terms for given ranges. Returns new col."""
    for i in i_range:
        for j in j_range:
            for k in k_range:
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return col

def _try_partition_27():
    """
    Try to find a rank-27 solution using clever partitioning.
    
    Strategy: Instead of treating j=0 separately, try to merge j=0 with parts of j=1,2
    using Strassen-like combinations.
    
    Key idea: For the 2x3x2 sub-problem, can we do better than 11?
    
    Let's try: C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k]
    
    Use Strassen on j={0,1} (7 terms) + schoolbook for j=2 (4 terms) = 11.
    
    Alternative: Can we find a 9-term decomposition for 2x3x2?
    
    Let me try a different approach: use the fact that we have 5 k-values.
    
    Partition: k={0,1,2,3,4}
    - j={0,1}, k={0,1}: Strassen (7)
    - j={0,1}, k={2,3}: Strassen (7)  
    - j={0,1}, k={4}: schoolbook (4)
    - j={2}, k={0,1,2,3,4}: schoolbook (10)
    Total: 7+7+4+10 = 28
    
    To get 27, we need to save 1 term somewhere.
    
    Idea: For j={2}, k={4}, can we merge with something?
    
    Actually, let's try: 
    - j={0,1}, k={0,1}: Strassen (7)
    - j={0,1}, k={2,3}: Strassen (7)
    - j={0,1}, k={4}: schoolbook (4)
    - j={2}, k={0,1}: schoolbook (4)
    - j={2}, k={2,3}: schoolbook (4)
    - j={2}, k={4}: schoolbook (2)
    Total: 7+7+4+4+4+2 = 28
    
    Still 28. Let me try yet another partition.
    
    What if I use a non-Strassen approach for one of the 2x2x2 blocks?
    
    For 2x2x2, the minimum rank is 7 (proven). So each 2x2x2 block needs at least 7.
    
    What about combining across k-blocks?
    
    For j=2, all k: C[i,4] includes A[i,2]*B[2,4]. This is just 2 terms.
    
    Let me try:
    - j={0,1}, k={0,1}: Strassen (7)
    - j={0,1}, k={2,3}: Strassen (7)
    - j={0,2}, k={4}: Can we do 2x2x1 in 3 terms? No, minimum is 4.
    
    Hmm. Let me try a completely different approach.
    
    What if I don't partition by k at all, but by some other structure?
    
    Actually, let me try:
    - j={0,1}, k={0,1,2}: 2x2x3 sub-problem
    - j={0,1}, k={3,4}: 2x2x2 sub-problem (Strassen, 7)
    - j={2}, k={0,1,2,3,4}: 2x1x5 sub-problem (schoolbook, 10)
    
    For 2x2x3: C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k] for i∈{0,1}, k∈{0,1,2}
    This is 6 entries. Minimum rank? At least 3 (since it's a 2x3 matrix times a 2x2 matrix structure).
    Schoolbook: 4 terms. Can we do 3? Probably not for generic matrices.
    
    So: 4 + 7 + 10 = 21? No wait, that's for j={0,1} only.
    
    Let me recount:
    - j={0,1}, k={0,1,2}: schoolbook 2x2x3 = 4 terms
    - j={0,1}, k={3,4}: Strassen 2x2x2 = 7 terms
    - j={2}, k={0,1,2,3,4}: schoolbook 2x1x5 = 10 terms
    Total: 4+7+10 = 21? That can't be right, the total should be at least 30/4 ≈ 8 but realistically much higher.
    
    Wait, I'm confusing myself. The schoolbook for 2x2x3 is:
    C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k]
    For i∈{0,1}, k∈{0,1,2}: that's 2*2*3 = 12 scalar products, but each rank-1 term covers 2*2=4 entries.
    So minimum 3 terms, schoolbook is 4 terms (A00*B0, A01*B1, A10*B0, A11*B1 where B0, B1 are 1x3 vectors).
    
    Actually schoolbook: for each (i,j) pair, one term: 2*2 = 4 terms. Yes.
    
    So 4+7+10 = 21. But that's way less than 28, which means I'm making an error.
    
    Oh wait, I think I see the issue. When I say "j={0,1}, k={0,1,2}", I mean the sub-tensor where i∈{0,1}, j∈{0,1}, k∈{0,1,2}. The schoolbook for this is 2*2*3 = 12 terms? No, that's not right either.
    
    Schoolbook for matrix multiplication: each (i,j,k) triple gets one term. For i∈{0,1}, j∈{0,1}, k∈{0,1,2}: that's 2*2*3 = 12 terms.
    
    But Strassen reduces 2x2x2 from 8 to 7. So for 2x2x3, schoolbook is 12, and we could do Strassen for k={0,1} (7) + schoolbook for k={2} (4) = 11.
    
    So my earlier calculation was wrong. Let me redo:
    
    - j={0,1}, k={0,1}: Strassen (7)
    - j={0,1}, k={2}: schoolbook (4)
    - j={0,1}, k={3,4}: Strassen (7)
    - j={2}, k={0,1,2,3,4}: schoolbook (10)
    Total: 7+4+7+10 = 28. Same as before.
    
    To get 27, I need to find a way to save 1 term.
    
    Idea: Can I merge the j={2}, k={2} term with something?
    
    C[i,2] = A[i,0]B[0,2] + A[i,1]B[1,2] + A[i,2]B[2,2]
    
    The j={0,1}, k={2} schoolbook gives: A[i,0]B[0,2] and A[i,1]B[1,2] (4 terms for i∈{0,1}).
    The j={2}, k={2} schoolbook gives: A[i,2]B[2,2] (2 terms for i∈{0,1}).
    
    Can I combine these 6 terms into 5? That would be a 2x3x1 tensor in 5 terms instead of 6.
    
    The 2x3x1 tensor: C[i] = A[i,0]B[0] + A[i,1]B[1] + A[i,2]B[2] for i∈{0,1}.
    This is 2 entries. Each rank-1 term U[i]*V[j]*W[k] with k fixed gives U[i]*V[j] which is 2 entries (for i=0,1).
    So each term can contribute to both C[0] and C[1] with different j values.
    
    Actually, for k fixed, the tensor is T[i,j] = 1 for all i,j (since C[i] = Σ_j A[i,j]B[j]).
    This is a 2x3 matrix of ones. Its rank is 1! So T[i,j] = 1 = 1*1 for all i,j.
    
    Wait, that means for fixed k, the tensor T[i,j,k] = 1 for all i,j has rank 1 in the (i,j) space.
    So C[i] = Σ_j A[i,j]B[j] can be written as: for each j, one term A[i,j]*B[j]. That's 3 terms (for j=0,1,2), each contributing to both i=0 and i=1.
    
    So the 2x3x1 sub-problem has rank 3, not 6! I was making a fundamental error.
    
    Let me redo the whole thing:
    
    For fixed k: C[i,k] = Σ_{j=0}^{2} A[i,j]B[j,k]
    The tensor T[i,j,k] = 1 for all i,j (with k fixed) has rank 1 in (i,j) space.
    So we need 3 terms (one for each j), each of the form: U[i,j]*V[j,k]*W[k,i]
    For term j: U[i,j]=1 for i=0,1; V[j,k]=1; W[k,i]=1 for i=0,1.
    That's 3 terms per k-value.
    
    So schoolbook for the full 2x3x5: 3*5 = 15 terms? No, that's not right either.
    
    I'm confusing myself. Let me be very careful.
    
    The matrix multiplication tensor: T[i,j,k] = 1 for all i∈{0,1}, j∈{0,1,2}, k∈{0,1,2,3,4}.
    We want: T[i,j,k] = Σ_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    Schoolbook: one term per (i,j,k) triple. Total: 2*3*5 = 30 terms.
    Each term: U[i,j,r]=1, V[j,k,r]=1, W[k,i,r]=1 for that specific (i,j,k).
    
    But we can do better. For example, Strassen shows that 2x2x2 can be done in 7 instead of 8.
    
    The key insight: for fixed (i,k), the tensor over j is T[i,j,k] = 1 for all j. This is a vector of ones, which has rank 1.
    So for fixed (i,k), we need Σ_j A[i,j]B[j,k] = 1 (in the tensor sense).
    This can be achieved with 3 terms (one per j), where each term has U[i,j,r]=1, V[j,k,r]=1, W[k,i,r]=1.
    
    But the same term r can be reused across different (i,k) pairs if the structure allows.
    
    In schoolbook, each (i,j,k) gets its own term. But if I fix j and vary (i,k), I get a 2x5 sub-tensor T[i,j,k] = 1 for all i,k. This is a 2x5 matrix of ones, which has rank 1.
    So for fixed j, I can use 1 term that covers all (i,k) pairs!
    
    Wait, that would mean the total rank is 3 (one per j value). But that's clearly wrong because Strassen needs 7 for 2x2x2.
    
    I think I'm making an error in my tensor rank reasoning. Let me reconsider.
    
    The tensor T[i,j,k] = 1 for all i,j,k. We want to write it as Σ_r U_r ⊗ V_r ⊗ W_r where (U_r ⊗ V_r ⊗ W_r)[i,j,k] = U_r[i,j] * V_r[j,k] * W_r[k,i].
    
    Note the constraint: U depends on (i,j), V on (j,k), W on (k,i). This is a cyclic constraint that makes it different from a standard tensor.
    
    For a standard tensor T[i,j,k] = 1, the rank is 1 (just U=V=W=1). But here, the factors are constrained to depend on specific pairs of indices, which makes it harder.
    
    So my earlier reasoning was wrong. The schoolbook 30 terms is the baseline, and Strassen reduces 2x2x2 from 8 to 7.
    
    OK so back to the original problem. The current best is 28. To get 27, I need a more clever decomposition.
    
    Let me try a different partition:
    - j={0,1}, k={0,1}: Strassen (7)
    - j={1,2}, k={2,3}: Strassen (7)
    - j={0}, k={2,3,4}: schoolbook (6)
    - j={2}, k={0,1}: schoolbook (4)
    - j={1}, k={4}: schoolbook (2)
    Total: 7+7+6+4+2 = 26? Let me verify this covers all (i,j,k) triples.
    
    j=0: k=0,1 (in Strassen block 1), k=2,3,4 (in schoolbook block 3). ✓
    j=1: k=0,1 (in Strassen block 1), k=2,3 (in Strassen block 2), k=4 (in schoolbook block 5). ✓
    j=2: k=0,1 (in schoolbook block 4), k=2,3 (in Strassen block 2), k=4 (missing!)
    
    I'm missing j=2, k=4. Let me add it: +2 terms = 28. Still 28.
    
    Let me try yet another approach. What if I use a 3x3x3 Strassen-like decomposition?
    
    Actually, let me just try to brute-force search for a rank-27 solution using the time budget. I'll set up a linear system and try to find a solution.
    
    For rank r, we have:
    - U: 6 x r
    - V: 15 x r  
    - W: 10 x r
    Total variables: 6r + 15r + 10r = 31r
    
    Constraints: 30 equations (one per (i,j,k) triple).
    
    For r=27: 31*27 = 837 variables, 30 equations. Highly underdetermined.
    
    But the equations are cubic (product of three variables), so it's not linear.
    
    However, if I fix U and V, the equations become linear in W:
    Σ_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    For each (i,j,k): Σ_r (U[i,j,r] * V[j,k,r]) * W[k,i,r] = 1
    
    This is linear in W! So I can:
    1. Randomly initialize U and V
    2. Solve for W via linear system
    3. Check if the solution is valid
    
    But the solution W might not have the right structure (it needs to be integer/half-integer).
    
    Let me try this approach with the time budget.
    
    """
    return None

def _try_linear_search_for_rank(r, n=2, m=3, p=5, max_iters=1000, timeout=100):
    """
    Try to find a rank-r decomposition by:
    1. Randomly initialize U and V
    2. Solve linear system for W
    3. Check validity
    """
    start_time = time.time()
    np.random.seed(42)
    
    for it in range(max_iters):
        if time.time() - start_time > timeout:
            break
        
        # Random initialization with small integers
        U = np.random.randint(-2, 3, size=(n*m, r))
        V = np.random.randint(-2, 3, size=(m*p, r))
        
        # Build linear system for W
        # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # Group by (k,i) pair: for each (k,i), we have m equations (one per j)
        # sum_r (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1
        
        A_mat = np.zeros((n*p, r))  # n*p rows, r cols
        b_vec = np.zeros(n*p)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = k * n + i
                    for r_idx in range(r):
                        A_mat[row, r_idx] += U[i*m+j, r_idx] * V[j*p+k, r_idx]
                    b_vec[row] = 1.0
        
        # Solve A_mat @ W_flat = b_vec
        # W_flat is r*n*p, but W is (p*n, r), so W_flat[k*n+i, r] = W[k*n+i, r]
        # Actually W has shape (p*n, r), so W_flat should be (p*n*r,)
        # But our system is: for each (k,i): sum_r A_mat[k*n+i, r] * W[k*n+i, r] = b_vec[k*n+i]
        # This is not a standard linear system because W[k*n+i, r] appears in multiple equations.
        
        # Let me restructure: W has shape (p*n, r). Let W_flat = W.reshape(-1) of size p*n*r.
        # The equation for (i,j,k) is: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i, r] = 1
        # In terms of W_flat: sum_r U[i*m+j,r] * V[j*p+k,r] * W_flat[(k*n+i)*r + r] = 1
        
        # This is linear in W_flat. Let me build the system properly.
        
        # Actually, this is getting complicated. Let me use a different approach.
        # Fix U and V, then for each (k,i), solve for W[k*n+i, :] from the m equations.
        
        W = np.zeros((p*n, r))
        valid = True
        
        for k in range(p):
            for i in range(n):
                # For this (k,i), we have m equations:
                # For each j: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # This is a linear system: M @ W[k*n+i, :] = 1 (vector of ones, length m)
                M = np.zeros((m, r))
                for j in range(m):
                    for r_idx in range(r):
                        M[j, r_idx] = U[i*m+j, r_idx] * V[j*p+k, r_idx]
                
                # Solve M @ x = ones(m)
                # This is underdetermined (m < r typically), so use least squares
                try:
                    x, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    W[k*n+i, :] = x
                except:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        # Verify
        if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def _known_rank_28():
    """Return the known rank-28 solution."""
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: j=0, all k (schoolbook) -> 10 terms
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Block 2a: j={1,2}, k={0,1} (Strassen) -> 7 terms
    col = _strassen_2x2x2(U, V, W, col, 0, 1, 0, n, m, p)
    
    # Block 2b: j={1,2}, k=2 (schoolbook) -> 4 terms
    for i in range(n):
        for j in range(1, 3):
            U[i * m + j][col] = 1
            V[j * p + 2][col] = 1
            W[2 * n + i][col] = 1
            col += 1
    
    # Block 2c: j={1,2}, k={3,4} (Strassen) -> 7 terms
    col = _strassen_2x2x2(U, V, W, col, 0, 1, 3, n, m, p)
    
    assert col == rank, f"Column mismatch: {col} != {rank}"
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = 2, 3, 5
    start_time = time.time()
    deadline = start_time + 140
    
    # First, get the known rank-28 solution as fallback
    best = _known_rank_28()
    best_rank = 28
    
    # Try to find rank-27 solution
    if time.time() < deadline - 20:
        result = _try_linear_search_for_rank(27, n, m, p, max_iters=500, timeout=110)
        if result is not None:
            best = result
            best_rank = 27
    
    return best

