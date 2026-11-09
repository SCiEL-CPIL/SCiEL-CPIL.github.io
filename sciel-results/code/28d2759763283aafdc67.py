"""Matrix multiplication 3x4x6: slice-and-sum decomposition with rank minimization.

Strategy:
  1. Build the schoolbook (rank 72) decomposition as a guaranteed baseline.
  2. Use the "slice-and-sum" method: partition the middle index j into groups,
     and for each group construct a sub-decomposition. The total rank is the
     sum of the sub-ranks. If we can find sub-decompositions with rank less
     than the schoolbook sub-rank, we get an overall improvement.
  3. For each partition of j into groups, try to find low-rank decompositions
     of the resulting sub-tensors using random search with verification.
  4. Also try a direct global random search at ranks below 72.

The tensor T for multiplying (3x4) by (4x6) has shape (12, 24, 18) with:
  T[i*m+j, j*p+k, k*n+i] = 1  for all i in [0,3), j in [0,4), k in [0,6)

A rank-r decomposition: T[a,b,c] = sum_s U[a,s]*V[b,s]*W[c,s]
"""

import time
import numpy as np

DIMENSIONS = [3, 4, 6]


def _build_tensor(n, m, p):
    """Build the full multiplication tensor T of shape (n*m, m*p, n*p)."""
    T = np.zeros((n * m, m * p, n * p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _schoolbook(n, m, p):
    """Return schoolbook decomposition (rank n*m*p)."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((n * p, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify(U, V, W, T):
    """Check if U, V, W exactly decompose T."""
    n_m, r = U.shape
    m_p = V.shape[0]
    n_p = W.shape[0]
    T_rec = np.zeros((n_m, m_p, n_p))
    for s in range(r):
        T_rec += np.einsum('a,b,c->abc', U[:, s], V[:, s], W[:, s])
    return np.allclose(T_rec, T, atol=1e-9)


def _verify_exact(U, V, W, T):
    """Check exact integer/half-integer match."""
    n_m, r = U.shape
    m_p = V.shape[0]
    n_p = W.shape[0]
    T_rec = np.zeros((n_m, m_p, n_p))
    for s in range(r):
        T_rec += np.einsum('a,b,c->abc', U[:, s], V[:, s], W[:, s])
    # Multiply by 2 to handle half-integers, then check integer match
    T_rec_2 = np.round(T_rec * 2)
    T_2 = np.round(T * 2)
    return np.array_equal(T_rec_2, T_2)


def _slice_and_sum_decomp(n, m, p, j_groups):
    """
    Slice-and-sum: partition the middle index j into groups.
    For each group G of j-values, we need to compute:
      C[i,k] += sum_{j in G} A[i,j] * B[j,k]
    
    The sub-tensor for group G has:
      - U rows: i*m + j for j in G  (|G| * n rows, but actually n*|G| rows out of n*m)
      - V rows: j*p + k for j in G  (|G| * p rows out of m*p)
      - W rows: k*n + i             (all n*p rows)
    
    Actually, the sub-tensor is sparse. Let's think differently.
    
    For a group G, the contribution to T is:
      T_sub[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,k and j in G
    
    This is equivalent to a matrix multiplication of size n x |G| times |G| x p.
    Its schoolbook rank is n * |G| * p.
    
    We try to find a lower-rank decomposition for each sub-problem.
    """
    U_parts = []
    V_parts = []
    W_parts = []
    
    for G in j_groups:
        g_size = len(G)
        # Sub-problem: n x g_size times g_size x p
        # Schoolbook rank for this sub-problem: n * g_size * p
        sub_rank = n * g_size * p
        
        # Build sub-tensor indices
        # U sub-indices: [i*m + j for i in range(n) for j in G]
        # V sub-indices: [j*p + k for j in G for k in range(p)]
        # W sub-indices: [k*n + i for k in range(p) for i in range(n)]  (all of them)
        
        u_idx = [i * m + j for i in range(n) for j in G]
        v_idx = [j * p + k for j in G for k in range(p)]
        w_idx = list(range(n * p))
        
        # Build sub-tensor
        T_sub = np.zeros((len(u_idx), len(v_idx), len(w_idx)))
        for ii, i in enumerate(range(n)):
            for jj, j in enumerate(G):
                for kk, k in enumerate(range(p)):
                    # u_idx position: i*g_size + jj
                    # v_idx position: jj*p + kk
                    # w_idx position: k*n + i
                    T_sub[i * g_size + jj, jj * p + kk, k * n + i] = 1.0
        
        # Try to find a low-rank decomposition for T_sub
        # Start with schoolbook
        U_sub = np.zeros((len(u_idx), sub_rank))
        V_sub = np.zeros((len(v_idx), sub_rank))
        W_sub = np.zeros((len(w_idx), sub_rank))
        col = 0
        for i in range(n):
            for jj, j in enumerate(G):
                for k in range(p):
                    U_sub[i * g_size + jj, col] = 1.0
                    V_sub[jj * p + k, col] = 1.0
                    W_sub[k * n + i, col] = 1.0
                    col += 1
        
        # Try to reduce rank of sub-problem
        # For small sub-problems, try random search
        best_sub_rank = sub_rank
        best_U_sub = U_sub
        best_V_sub = V_sub
        best_W_sub = W_sub
        
        # Try ranks from sub_rank-1 down
        for target_rank in range(sub_rank - 1, max(n * p, 1) - 1, -1):
            found = False
            for attempt in range(50):
                U_try = np.random.randint(-1, 2, size=(len(u_idx), target_rank))
                V_try = np.random.randint(-1, 2, size=(len(v_idx), target_rank))
                W_try = np.random.randint(-1, 2, size=(len(w_idx), target_rank))
                if _verify(U_try, V_try, W_sub if False else W_try, T_sub):
                    best_sub_rank = target_rank
                    best_U_sub = U_try
                    best_V_sub = V_try
                    best_W_sub = W_try
                    found = True
                    break
            if found:
                break
        
        # Map back to full indices
        U_full = np.zeros((n * m, best_sub_rank))
        V_full = np.zeros((m * p, best_sub_rank))
        W_full = np.zeros((n * p, best_sub_rank))
        
        for idx_u, val in enumerate(best_U_sub):
            U_full[u_idx[idx_u]] = val
        for idx_v, val in enumerate(best_V_sub):
            V_full[v_idx[idx_v]] = val
        for idx_w, val in enumerate(best_W_sub):
            W_full[w_idx[idx_w]] = val
        
        U_parts.append(U_full)
        V_parts.append(V_full)
        W_parts.append(W_full)
    
    # Concatenate all parts
    U_total = np.hstack(U_parts)
    V_total = np.hstack(V_parts)
    W_total = np.hstack(W_parts)
    
    return U_total, V_total, W_total, U_total.shape[1]


def construct():
    """Return the lowest-rank exact decomposition found within the time budget."""
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    n, m, p = DIMENSIONS
    n_m = n * m      # 12
    m_p = m * p      # 24
    n_p = p * n      # 18
    schoolbook_rank = n * m * p  # 72

    # Build the full tensor for verification
    T = _build_tensor(n, m, p)

    # Start with schoolbook (guaranteed correct)
    best_U, best_V, best_W = _schoolbook(n, m, p)
    best_rank = schoolbook_rank

    # Verify schoolbook
    assert _verify(best_U, best_V, best_W, T), "Schoolbook verification failed"

    # Strategy 1: Slice-and-sum with different partitions of j
    # j ranges over {0, 1, 2, 3}
    partitions = [
        # All in one group (same as schoolbook)
        [[0, 1, 2, 3]],
        # Two groups of 2
        [[0, 1], [2, 3]],
        [[0, 2], [1, 3]],
        [[0, 3], [1, 2]],
        # One group of 1, one group of 3
        [[0], [1, 2, 3]],
        [[1], [0, 2, 3]],
        [[2], [0, 1, 3]],
        [[3], [0, 1, 2]],
        # Three groups: 1+1+2
        [[0], [1], [2, 3]],
        [[0], [2], [1, 3]],
        [[0], [3], [1, 2]],
        [[1], [2], [0, 3]],
        [[1], [3], [0, 2]],
        [[2], [3], [0, 1]],
        # Four groups of 1 (same as schoolbook)
        [[0], [1], [2], [3]],
        # Two groups: 1+3 variations already covered
        # Three groups: 1+1+2 variations already covered
    ]

    for partition in partitions:
        if time.time() > deadline - 20:
            break
        
        try:
            U_try, V_try, W_try, rank_try = _slice_and_sum_decomp(n, m, p, partition)
            if rank_try < best_rank and _verify(U_try, V_try, W_try, T):
                best_rank = rank_try
                best_U = U_try
                best_V = V_try
                best_W = W_try
        except Exception:
            continue

    # Strategy 2: Direct global random search at ranks below best_rank
    # Try a few random attempts at various ranks
    for target_rank in range(best_rank - 1, max(n_p, m_p, n_m) - 1, -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        # Number of attempts depends on remaining time
        remaining = deadline - time.time()
        n_attempts = min(100, max(10, int(remaining * 5)))
        
        for attempt in range(n_attempts):
            if time.time() > deadline - 5:
                break
            
            # Random matrices with small integer entries
            U_try = np.random.randint(-1, 2, size=(n_m, target_rank))
            V_try = np.random.randint(-1, 2, size=(m_p, target_rank))
            W_try = np.random.randint(-1, 2, size=(n_p, target_rank))
            
            if _verify(U_try, V_try, W_try, T):
                best_rank = target_rank
                best_U = U_try
                best_V = V_try
                best_W = W_try
                found = True
                break
        
        if found:
            break

    # Strategy 3: Try structured low-rank constructions
    # For 3x4x6, we can try to exploit the fact that m=4=2+2
    # and use Strassen-like ideas on 2x2 blocks
    
    # View A as [A1 (3x2), A2 (3x2)] and B as [B1 (2x6); B2 (2x6)]
    # C = A1*B1 + A2*B2
    # Each product is 3x2 * 2x6, schoolbook rank 36
    # Total: 72 (same as before)
    # But if we can do 3x2*2x6 in less than 36, we save
    
    # For 3x2*2x6: this is equivalent to multiplying a 3x2 by 2x6
    # The tensor has shape (6, 12, 18) with T[i*2+j, j*6+k, k*3+i] = 1
    # Schoolbook rank: 3*2*6 = 36
    # Lower bound: max(6, 12, 18) = 18
    
    # Try to find a decomposition of 3x2*2x6 with rank < 36
    # This is a smaller problem, so random search might work better
    n2, m2, p2 = 3, 2, 6
    T_sub = _build_tensor(n2, m2, p2)
    sub_schoolbook = n2 * m2 * p2  # 36
    
    best_sub_rank = sub_schoolbook
    best_sub_U, best_sub_V, best_sub_W = _schoolbook(n2, m2, p2)
    
    for target_rank in range(sub_schoolbook - 1, max(n2*m2, m2*p2, n2*p2) - 1, -1):
        if time.time() > deadline - 5:
            break
        
        found = False
        remaining = deadline - time.time()
        n_attempts = min(200, max(20, int(remaining * 10)))
        
        for attempt in range(n_attempts):
            if time.time() > deadline - 2:
                break
            
            U_try = np.random.randint(-1, 2, size=(n2*m2, target_rank))
            V_try = np.random.randint(-1, 2, size=(m2*p2, target_rank))
            W_try = np.random.randint(-1, 2, size=(n2*p2, target_rank))
            
            if _verify(U_try, V_try, W_try, T_sub):
                best_sub_rank = target_rank
                best_sub_U = U_try
                best_sub_V = V_try
                best_sub_W = W_try
                found = True
                break
        
        if found:
            break
    
    # If we found a better 3x2*2x6, use it for both halves
    if best_sub_rank < sub_schoolbook:
        # Construct full decomposition using the improved sub-decomposition
        # A = [A1, A2] where A1 is columns 0,1 and A2 is columns 2,3
        # B = [B1; B2] where B1 is rows 0,1 and B2 is rows 2,3
        
        U_full = np.zeros((n_m, 2 * best_sub_rank))
        V_full = np.zeros((m_p, 2 * best_sub_rank))
        W_full = np.zeros((n_p, 2 * best_sub_rank))
        
        # First half: A1 (cols 0,1) * B1 (rows 0,1)
        # U1[i*2+j, s] = best_sub_U[i*2+j, s] for j in {0,1}
        # V1[j*6+k, s] = best_sub_V[j*6+k, s] for j in {0,1}
        # W1[k*3+i, s] = best_sub_W[k*3+i, s]
        
        for i in range(n):
            for j in range(m2):
                for s in range(best_sub_rank):
                    U_full[i * m + j, s] = best_sub_U[i * m2 + j, s]
        
        for j in range(m2):
            for k in range(p):
                for s in range(best_sub_rank):
                    V_full[j * p + k, s] = best_sub_V[j * p2 + k, s]
        
        for k in range(p):
            for i in range(n):
                for s in range(best_sub_rank):
                    W_full[k * n + i, s] = best_sub_W[k * n2 + i, s]
        
        # Second half: A2 (cols 2,3) * B2 (rows 2,3)
        offset = best_sub_rank
        for i in range(n):
            for j in range(m2):
                for s in range(best_sub_rank):
                    U_full[i * m + (m2 + j), offset + s] = best_sub_U[i * m2 + j, s]
        
        for j in range(m2):
            for k in range(p):
                for s in range(best_sub_rank):
                    V_full[(m2 + j) * p + k, offset + s] = best_sub_V[j * p2 + k, s]
        
        for k in range(p):
            for i in range(n):
                for s in range(best_sub_rank):
                    W_full[k * n + i, offset + s] = best_sub_W[k * n2 + i, s]
        
        total_rank = 2 * best_sub_rank
        if total_rank < best_rank and _verify(U_full, V_full, W_full, T):
            best_rank = total_rank
            best_U = U_full
            best_V = V_full
            best_W = W_full

    # Final verification
    if not _verify(best_U, best_V, best_W, T):
        # Fall back to schoolbook
        best_U, best_V, best_W = _schoolbook(n, m, p)
        best_rank = schoolbook_rank

    # Convert to lists for return
    U_ret = best_U.tolist()
    V_ret = best_V.tolist()
    W_ret = best_W.tolist()

    return {"U": U_ret, "V": V_ret, "W": W_ret}
