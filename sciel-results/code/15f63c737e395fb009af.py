"""Matrix multiplication 4x4x5 rank-minimal decomposition search."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 5]
n, m, p = DIMENSIONS  # 4, 4, 5
NM, MP, PN = n * m, m * p, p * n  # 16, 20, 20

# Strassen 2x2x2 basis: 7 rank-1 terms
# Each term: (u_vec, v_vec, w_vec) where u is 4-dim (2x2), v is 4-dim, w is 4-dim
_STRASSEN_2X2 = [
    # (u, v, w) as flat (row-major) vectors of length 4
    # M1: (a+d)*(e+h)
    ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
    # M2: (c+d)*e
    ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
    # M3: a*(f-h)
    ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),
    # M4: d*(g-e)
    ([0,0,0,1], [0,0,1,-1], [1,0,1,0]),
    # M5: (a+b)*h
    ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
    # M6: (c-a)*(e+f)
    ([1,-1,0,0], [1,1,0,0], [0,0,0,1]),
    # M7: (b-d)*(g+h)
    ([0,1,-1,0], [0,0,1,1], [1,0,0,0]),
]


def _build_tensor(U, V, W):
    """Compute the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]."""
    r = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    tensor = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                tensor[i, j, k] = np.dot(U_np[idx_u], V_np[idx_v] * W_np[idx_w])
    return tensor


def _construct_strassen_kron_rank40():
    """
    Block-diagonal decomposition: partition A (4x4) into four 2x4 row blocks,
    B (4x5) into four 2x5 column blocks. Each block computes a 2x4 * 2x5 product.
    
    For each 2x4 * 2x5 sub-problem, we use Kronecker structure:
    - The 2x2x2 Strassen gives 7 terms for the 2x2 part
    - We need to handle the 4 columns of A-block and 5 columns of B-block
    
    Strategy: For 2x4 * 2x5:
    - Split the 4 columns of A into two pairs: cols (0,1) and cols (2,3)
    - Split the 5 columns of B into: cols (0,1,2,3) as two 2-col groups + col 4
    - For each (A_pair, B_group) combination, use Strassen (7 terms)
    - Plus direct terms for the 5th column
    
    Actually, let me think more carefully.
    
    A is 4x4, B is 4x5. C = A*B is 4x5.
    
    Block partition:
    - A rows 0,1 (top half), A rows 2,3 (bottom half)
    - B rows 0,1 (top half), B rows 2,3 (bottom half)
    
    C[0:2, :] = A[0:2,:] * B  (2x4 * 4x5 = 2x5)
    C[2:4, :] = A[2:4,:] * B  (2x4 * 4x5 = 2x5)
    
    For C[0:2, :]:
    C[0:2, 0:4] = A[0:2, 0:2] * B[0:2, 0:4] + A[0:2, 2:4] * B[2:4, 0:4]
    C[0:2, 4]   = A[0:2, 0:2] * B[0:2, 4:5]   + A[0:2, 2:4] * B[2:4, 4:5]
    
    So for the top-left 2x5 block of C:
    - A_top_left (2x2) * B_top (2x4): use Strassen-like for 2x2*2x4
    - A_top_right (2x2) * B_bottom (2x4): use Strassen-like for 2x2*2x4
    - A_top_left (2x2) * B_top_5th (2x1): direct
    - A_top_right (2x2) * B_bottom_5th (2x1): direct
    
    For 2x2 * 2x4: we can view this as computing 4 separate 2x2*2x1 products,
    but with shared A. Use Kronecker: Strassen_2x2 ⊗ identity_2 gives 7*2=14 terms?
    No, that's not right either.
    
    Let me think differently. For 2x2 * 2x4:
    Result is 2x4. Each column of result is 2x2 * 2x1.
    If we use Strassen for the 2x2*2x1 part, we get 7 terms per column = 28 total.
    But we can share: the A factor is the same for all 4 columns.
    
    Actually, the Kronecker product approach:
    2x2*2x4 tensor = (2x2*2x2 tensor) ⊗ (1x1*2x2 tensor)
    Rank of 2x2*2x2 is 7 (Strassen)
    Rank of 1x1*2x2 is 2 (direct)
    So rank of Kronecker is 7*2 = 14.
    
    Similarly 2x2*2x1: rank = 7*1 = 7.
    
    So for C[0:2, 0:4]:
    - A_top_left * B_top: rank 14
    - A_top_right * B_bottom: rank 14
    For C[0:2, 4]:
    - A_top_left * B_top_5: rank 7
    - A_top_right * B_bottom_5: rank 7
    
    Total for top half: 14+14+7+7 = 42
    Total for both halves: 84. That's worse than 72.
    
    Let me try a different partition. What if we split B's 5 columns differently?
    
    Alternative: Use the fact that 4x5 = 4x(4+1).
    C = A * B[:, 0:4] + A * B[:, 4:5]
    
    A * B[:, 0:4]: 4x4 * 4x4. This is a square 4x4x4 problem.
    Best known rank for 4x4x4 is 49 (or maybe lower with newer results).
    Using 2x2 Strassen blocks: 4 blocks * 7 = 28? No, that's not how it works.
    
    Actually for 4x4*4x4, we can use:
    - 4 Strassen 2x2 blocks on the diagonal: 4*7 = 28
    - But we also need off-diagonal interactions...
    
    Hmm, let me reconsider. The standard block approach for 4x4*4x4:
    Partition into 2x2 blocks. Each block multiplication is 2x2*2x2.
    There are 4 block positions in C, each requiring sums of products of 2x2 blocks of A and B.
    
    C[0:2,0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    C[0:2,2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]
    C[2:4,0:2] = A[2:4,0:2]*B[0:2,0:2] + A[2:4,2:4]*B[2:4,0:2]
    C[2:4,2:4] = A[2:4,0:2]*B[0:2,2:4] + A[2:4,2:4]*B[2:4,2:4]
    
    Each of these is a sum of two 2x2*2x2 products. If we do each independently with Strassen:
    8 * 7 = 56 terms.
    
    Plus A * B[:, 4:5]: 4x4 * 4x1.
    C[:, 4] = A[:, 0:2] * B[0:2, 4] + A[:, 2:4] * B[2:4, 4]
    Each 2x4 * 2x1... wait, 4x4 * 4x1:
    C[0:2, 4] = A[0:2, 0:2] * B[0:2, 4] + A[0:2, 2:4] * B[2:4, 4]
    C[2:4, 4] = A[2:4, 0:2] * B[0:2, 4] + A[2:4, 2:4] * B[2:4, 4]
    
    Each 2x2 * 2x1: rank 7 (Strassen for 2x2*2x1, since 2x1 is like 2x2 with last column zero)
    Actually 2x2 * 2x1: the result is 2x1. We can use 7 Strassen terms.
    So 4 * 7 = 28 more terms.
    
    Total: 56 + 28 = 84. Still worse.
    
    The current Strassen approach gives 72. Let me just keep that and try to improve the search.
    
    Actually, re-reading the directive: "Apply a structured block-diagonal decomposition that partitions the 4x4x5 multiplication tensor into four independent 2x4x5 sub-tensors, each decomposed via a custom rank-10 algorithm"
    
    Four 2x4x5 sub-tensors, each rank 10, total rank 40.
    
    How to get rank 10 for 2x4*4x5? That seems very aggressive. The schoolbook rank is 40, and Strassen-like would be at least 7*something.
    
    Wait, maybe the partition is different. Let me re-read: "partitions the 4x4x5 multiplication tensor into four independent 2x4x5 sub-tensors"
    
    If we partition A into 4 row groups of size 1, and B into... no, that doesn't make 2x4x5.
    
    Maybe: partition A's 4 rows into 2 groups of 2, and B's 4 rows into 2 groups of 2, giving 4 sub-problems of 2x4 * 2x5.
    
    Sub-problem: A_sub (2x4) * B_sub (2x5) = C_sub (2x5).
    
    For 2x4 * 2x5:
    Schoolbook rank = 2*4*5 = 40.
    Can we do better? 
    
    2x4 * 2x5 = (2x2 * 2x5) + (2x2 * 2x5) where we split A's 4 cols into two 2-col groups.
    Each 2x2 * 2x5:
    2x2 * 2x5 = (2x2*2x2) + (2x2*2x2) + (2x2*2x1)
    = 7 + 7 + 7 = 21 per 2x2*2x5.
    Two of those: 42. Worse than schoolbook 40.
    
    Hmm, this doesn't seem to work for rank 10 per sub-tensor.
    
    Let me just keep the existing rank-72 Strassen solution and improve the random search to be more effective, perhaps using a smarter search strategy with the time saved from faster verification.
    """
    # This function is a placeholder; the actual improvement is in the search strategy.
    return None


def _is_valid(U, V, W, tol=1e-9):
    """Check if factors exactly decompose the all-ones tensor, with early exit."""
    try:
        r = len(U[0])
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        # Check a few entries first for quick rejection
        test_indices = [(0,0,0), (0,0,1), (1,1,1), (3,3,4), (0,3,0), (2,1,3)]
        for (i,j,k) in test_indices:
            idx_u = i * m + j
            idx_v = j * p + k
            idx_w = k * n + i
            val = np.dot(U_np[idx_u], V_np[idx_v] * W_np[idx_w])
            if abs(val - 1.0) > tol:
                return False
        # Full check
        tensor = np.zeros((n, m, p), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    tensor[i, j, k] = np.dot(U_np[idx_u], V_np[idx_v] * W_np[idx_w])
        if np.allclose(tensor, np.ones((n, m, p)), atol=tol):
            return True
    except Exception:
        pass
    return False


def _construct_strassen_rank72():
    """Deterministic Strassen-based decomposition with rank 72."""
    rank_total = 72
    U = np.zeros((NM, rank_total))
    V = np.zeros((MP, rank_total))
    W = np.zeros((PN, rank_total))
    state = [0]

    def add_strassen_2x2(ra, ca, rb, cb):
        col = state[0]
        # M1: (a+d)*(e+h)
        U[2*ra * m + 2*ca, col] = 1.0
        U[(2*ra+1) * m + (2*ca+1), col] = 1.0
        V[2*rb * p + 2*cb, col] = 1.0
        V[(2*rb+1) * p + (2*cb+1), col] = 1.0
        W[2*cb * n + 2*ra, col] = 1.0
        W[(2*cb+1) * n + (2*ra+1), col] = 1.0
        state[0] += 1

        col = state[0]
        # M2: (c+d)*e
        U[(2*ra+1) * m + 2*ca, col] = 1.0
        U[(2*ra+1) * m + (2*ca+1), col] = 1.0
        V[2*rb * p + 2*cb, col] = 1.0
        W[2*cb * n + (2*ra+1), col] = 1.0
        W[(2*cb+1) * n + (2*ra+1), col] = -1.0
        state[0] += 1

        col = state[0]
        # M3: a*(f-h)
        U[2*ra * m + 2*ca, col] = 1.0
        V[2*rb * p + (2*cb+1), col] = 1.0
        V[(2*rb+1) * p + (2*cb+1), col] = -1.0
        W[(2*cb+1) * n + 2*ra, col] = 1.0
        W[(2*cb+1) * n + (2*ra+1), col] = 1.0
        state[0] += 1

        col = state[0]
        # M4: d*(g-e)
        U[(2*ra+1) * m + (2*ca+1), col] = 1.0
        V[(2*rb+1) * p + 2*cb, col] = 1.0
        V[2*rb * p + 2*cb, col] = -1.0
        W[2*cb * n + 2*ra, col] = 1.0
        W[2*cb * n + (2*ra+1), col] = 1.0
        state[0] += 1

        col = state[0]
        # M5: (a+b)*h
        U[2*ra * m + 2*ca, col] = 1.0
        U[2*ra * m + (2*ca+1), col] = 1.0
        V[(2*rb+1) * p + (2*cb+1), col] = 1.0
        W[2*cb * n + 2*ra, col] = -1.0
        W[(2*cb+1) * n + 2*ra, col] = 1.0
        state[0] += 1

        col = state[0]
        # M6: (c-a)*(e+f)
        U[(2*ra+1) * m + 2*ca, col] = 1.0
        U[2*ra * m + 2*ca, col] = -1.0
        V[2*rb * p + 2*cb, col] = 1.0
        V[2*rb * p + (2*cb+1), col] = 1.0
        W[(2*cb+1) * n + (2*ra+1), col] = 1.0
        state[0] += 1

        col = state[0]
        # M7: (b-d)*(g+h)
        U[2*ra * m + (2*ca+1), col] = 1.0
        U[(2*ra+1) * m + (2*ca+1), col] = -1.0
        V[(2*rb+1) * p + 2*cb, col] = 1.0
        V[(2*rb+1) * p + (2*cb+1), col] = 1.0
        W[2*cb * n + 2*ra, col] = 1.0
        state[0] += 1

    def add_direct_2x2_2x1(ra, ca, rb, cb_col):
        for i in range(2):
            for j in range(2):
                col = state[0]
                U[(2*ra + i) * m + (2*ca + j), col] = 1.0
                V[(2*rb + j) * p + cb_col, col] = 1.0
                W[cb_col * n + (2*ra + i), col] = 1.0
                state[0] += 1

    # Block 0: B cols 0,1 (cb=0)
    add_strassen_2x2(0, 0, 0, 0)
    add_strassen_2x2(0, 1, 1, 0)
    add_strassen_2x2(1, 0, 0, 0)
    add_strassen_2x2(1, 1, 1, 0)

    # Block 1: B cols 2,3 (cb=1)
    add_strassen_2x2(0, 0, 0, 1)
    add_strassen_2x2(0, 1, 1, 1)
    add_strassen_2x2(1, 0, 0, 1)
    add_strassen_2x2(1, 1, 1, 1)

    # Block 2: B col 4 (cb_col=4)
    add_direct_2x2_2x1(0, 0, 0, 4)
    add_direct_2x2_2x1(0, 1, 1, 4)
    add_direct_2x2_2x1(1, 0, 0, 4)
    add_direct_2x2_2x1(1, 1, 1, 4)

    used_rank = state[0]
    U = U[:, :used_rank].tolist()
    V = V[:, :used_rank].tolist()
    W = W[:, :used_rank].tolist()
    return U, V, W


def _construct_schoolbook():
    """Schoolbook decomposition with rank 80."""
    rank = n * m * p
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _random_candidate(rank, entry_set, seed=None):
    """Generate a random candidate of given rank."""
    if seed is not None:
        random.seed(seed)
    U = [[random.choice(entry_set) for _ in range(rank)] for _ in range(NM)]
    V = [[random.choice(entry_set) for _ in range(rank)] for _ in range(MP)]
    W = [[random.choice(entry_set) for _ in range(rank)] for _ in range(PN)]
    return U, V, W


def _perturb_solution(U, V, W, entry_set, num_perturbations=3):
    """Perturb a solution slightly to explore nearby configurations."""
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0])
    for _ in range(num_perturbations):
        # Perturb a random entry in a random matrix
        mat_choice = random.choice(['U', 'V', 'W'])
        if mat_choice == 'U':
            r_idx = random.randint(0, NM - 1)
            c_idx = random.randint(0, rank - 1)
            U[r_idx][c_idx] = random.choice(entry_set)
        elif mat_choice == 'V':
            r_idx = random.randint(0, MP - 1)
            c_idx = random.randint(0, rank - 1)
            V[r_idx][c_idx] = random.choice(entry_set)
        else:
            r_idx = random.randint(0, PN - 1)
            c_idx = random.randint(0, rank - 1)
            W[r_idx][c_idx] = random.choice(entry_set)
    return U, V, W


def _structured_low_rank_attempt(rank, deadline):
    """Try structured approaches for lower rank using local search from Strassen."""
    entry_sets = [
        [-1, 0, 1],
        [-1, -0.5, 0, 0.5, 1],
        [-2, -1, -0.5, 0, 0.5, 1, 2],
    ]
    
    # Start from the Strassen rank-72 solution and try to reduce rank
    # by zeroing out columns and checking validity
    U_str, V_str, W_str = _construct_strassen_rank72()
    
    for entry_set in entry_sets:
        if time.time() > deadline - 5:
            break
        
        # Try to find a subset of Strassen columns that works at lower rank
        # This is a combinatorial problem, so use random subset selection
        for trial in range(100):
            if time.time() > deadline - 5:
                break
            
            # Randomly select `rank` columns from the 72 Strassen columns
            col_indices = random.sample(range(72), min(rank, 72))
            if len(col_indices) < rank:
                continue
                
            U_sub = [[U_str[i][c] for c in col_indices] for i in range(NM)]
            V_sub = [[V_str[i][c] for c in col_indices] for i in range(MP)]
            W_sub = [[W_str[i][c] for c in col_indices] for i in range(PN)]
            
            # Pad with random entries if needed
            if len(col_indices) < rank:
                pad_cols = rank - len(col_indices)
                for i in range(NM):
                    U_sub[i].extend([random.choice(entry_set) for _ in range(pad_cols)])
                for i in range(MP):
                    V_sub[i].extend([random.choice(entry_set) for _ in range(pad_cols)])
                for i in range(PN):
                    W_sub[i].extend([random.choice(entry_set) for _ in range(pad_cols)])
            
            if _is_valid(U_sub, V_sub, W_sub):
                return U_sub, V_sub, W_sub
            
            # Try local search from this starting point
            for local_trial in range(20):
                if time.time() > deadline - 5:
                    break
                U_p, V_p, W_p = _perturb_solution(U_sub, V_sub, W_sub, entry_set, num_perturbations=1)
                if _is_valid(U_p, V_p, W_p):
                    return U_p, V_p, W_p
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    # Start with deterministic Strassen rank-72 solution
    U_str, V_str, W_str = _construct_strassen_rank72()
    best_rank = 72
    best_solution = {"U": U_str, "V": V_str, "W": W_str}

    if not _is_valid(U_str, V_str, W_str):
        # Fallback to schoolbook if Strassen fails
        U_sb, V_sb, W_sb = _construct_schoolbook()
        best_rank = 80
        best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}

    entry_sets = [
        [-1, 0, 1],
        [-1, -0.5, 0, 0.5, 1],
    ]

    # Phase 1: Try to reduce rank from Strassen using structured subset search
    # Focus on ranks 50-71 first (closer to Strassen, more likely to succeed)
    for r in range(71, 49, -1):
        if time.time() > deadline - 30:
            break
        remaining = deadline - time.time() - 30
        if remaining <= 0:
            break
        
        result = _structured_low_rank_attempt(r, deadline - 25)
        if result is not None:
            U_r, V_r, W_r = result
            best_rank = r
            best_solution = {"U": U_r, "V": V_r, "W": W_r}
            break

    # Phase 2: Random search for lower ranks
    for r in range(40, max(25, best_rank - 1), 2):  # Step by 2 to save time
        if time.time() > deadline - 15:
            break
        remaining = deadline - time.time() - 15
        if remaining <= 0:
            break
        
        num_trials = min(200, max(30, int(remaining * 20)))
        found = False
        for entry_set in entry_sets:
            if time.time() > deadline - 15:
                break
            for trial in range(num_trials // len(entry_sets)):
                if time.time() > deadline - 15:
                    break
                U, V, W = _random_candidate(r, entry_set)
                if _is_valid(U, V, W):
                    best_rank = r
                    best_solution = {"U": U, "V": V, "W": W}
                    found = True
                    break
            if found:
                break

    # Phase 3: If we found something, try to shave off a few more ranks
    if best_rank < 72 and time.time() < deadline - 10:
        for r in range(best_rank - 1, max(20, best_rank - 5), -1):
            if time.time() > deadline - 10:
                break
            for entry_set in entry_sets:
                for trial in range(50):
                    if time.time() > deadline - 10:
                        break
                    U, V, W = _random_candidate(r, entry_set)
                    if _is_valid(U, V, W):
                        best_rank = r
                        best_solution = {"U": U, "V": V, "W": W}
                        break

    return best_solution
