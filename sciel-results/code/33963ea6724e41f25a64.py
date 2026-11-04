import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def _schoolbook_decomposition(n, m, p):
    """Return the standard n*m*p rank decomposition for matrix multiplication."""
    U_cols = []
    V_cols = []
    W_cols = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                
                U_cols.append(u)
                V_cols.append(v)
                W_cols.append(w)
                
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    return U, V, W


def _strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is (U, V, W) where U, V, W are 4-element lists (for 2x2 matrices flattened).
    U[i*2+j], V[j*2+k], W[k*2+i].
    """
    terms = []
    
    # M1 = (a+d)(e+h)
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    
    # M2 = (c+d)(e)
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
    
    # M3 = a(f-h)
    terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))
    
    # M4 = d(g-e)
    terms.append(([0,0,0,1], [0,0,1,-1], [-1,0,1,0]))
    
    # M5 = (a+b)(h)
    terms.append(([1,1,0,0], [0,0,0,1], [1,1,0,0]))
    
    # M6 = (c-a)(e+f)
    terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    
    # M7 = (b-d)(g+h)
    terms.append(([0,1,0,-1], [0,0,1,1], [-1,0,0,-1]))
    
    return terms


def _embed_2x2_to_4x4(vec_4, row_start, col_start, size=4):
    """Embed a 4-element vector (for 2x2) into a 16-element vector (for 4x4) at given position."""
    result = [0] * (size * size)
    for idx, val in enumerate(vec_4):
        if val == 0:
            continue
        r = idx // 2
        c = idx % 2
        result[(row_start + r) * size + (col_start + c)] = val
    return result


def _strassen_4x4_4x4_terms():
    """Generate 49 Strassen terms for 4x4x4 matrix multiplication using 2x2 blocks.
    
    The 4x4x4 tensor is the Kronecker product of the 2x2x2 tensor with itself.
    If T_2 has rank 7, then T_4 = T_2 ⊗ T_2 has rank at most 7*7 = 49.
    """
    base_terms = _strassen_2x2_terms()
    
    terms_4x4 = []
    for (u1, v1, w1) in base_terms:
        for (u2, v2, w2) in base_terms:
            # Kronecker product of the terms
            u = np.array(u1) * np.array(u2)  # element-wise for Kronecker-like
            # Actually, we need to embed each 2x2 term into the 4x4 space
            # The Kronecker product of two 2x2 matrices gives a 4x4 matrix
            # For the tensor decomposition, we take the tensor product of the vectors
            
            # u1 is length 4 (2x2 flattened), u2 is length 4 (2x2 flattened)
            # The Kronecker product u1 ⊗ u2 has length 16
            u_kron = np.kron(np.array(u1), np.array(u2))
            v_kron = np.kron(np.array(v1), np.array(v2))
            w_kron = np.kron(np.array(w1), np.array(w2))
            
            terms_4x4.append((u_kron.tolist(), v_kron.tolist(), w_kron.tolist()))
    
    return terms_4x4


def _strassen_4x4_4x7_terms():
    """Generate terms for 4x4 x 4x7 matrix multiplication.
    
    We decompose the 4x7 matrix into four 2x2 blocks and one 2x1 block (or similar).
    Actually, 4x7 = 4 rows, 7 cols. We can think of it as 2x2 blocks in rows and 
    columns, but 7 is not divisible by 2. 
    
    Alternative approach: Use the 4x4x4 Strassen decomposition and extend to 4x7.
    The 4x7 matrix can be seen as [4x4 | 4x3]. We can handle the 4x4 part with 
    Strassen and the 4x3 part separately, but that's not clean.
    
    Better approach: Use a block decomposition where we treat the 4x7 as having
    2x2 blocks in the row dimension and handle columns appropriately.
    
    Actually, let's use a simpler structural approach:
    - For the 4x4 x 4x4 part, use Strassen (49 terms)
    - For the 4x4 x 4x3 part, we need additional terms
    
    But this gets complicated. Let's try a different structural idea:
    
    Use the fact that 4x7 multiplication can be decomposed by:
    1. Split A (4x4) into 2x2 blocks: A = [[A11, A12], [A21, A22]]
    2. Split B (4x7) into 2x2 blocks in rows, but 7 cols is tricky.
    
    Let's use a simpler approach: just use schoolbook for now as a baseline,
    and try to find a better decomposition through search.
    """
    # For now, return schoolbook as a safe fallback
    n, m, p = 4, 4, 7
    U, V, W = _schoolbook_decomposition(n, m, p)
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the tensor decomposition is exact."""
    rank = U.shape[1]
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication, T[i,j,k] should be 1 for all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _random_decomposition(n, m, p, rank, scale=2.0):
    """Generate a random decomposition with integer/half-integer entries."""
    # Use entries from {-1, 0, 1} or {-1, -0.5, 0, 0.5, 1}
    values = [-1, -0.5, 0, 0.5, 1]
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for c in range(rank):
        for i in range(n*m):
            if random.random() < 0.3:  # sparse
                U[i, c] = random.choice(values)
        for j in range(m*p):
            if random.random() < 0.3:
                V[j, c] = random.choice(values)
        for k in range(p*n):
            if random.random() < 0.3:
                W[k, c] = random.choice(values)
    
    return U, V, W


def _local_search(U, V, W, n, m, p, max_iters=1000, time_limit=10.0):
    """Local search to improve a decomposition by flipping entries."""
    start_time = time.time()
    rank = U.shape[1]
    
    # Compute initial error
    def compute_error(U, V, W):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err += (s - 1.0) ** 2
        return err
    
    current_err = compute_error(U, V, W)
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    best_err = current_err
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Randomly pick a matrix and entry to flip
        mat = random.choice(['U', 'V', 'W'])
        if mat == 'U':
            idx = random.randrange(n*m)
            c = random.randrange(rank)
            old_val = U[idx, c]
            new_val = -old_val if old_val != 0 else (1 if random.random() < 0.5 else -1)
            U[idx, c] = new_val
            new_err = compute_error(U, V, W)
            U[idx, c] = old_val
        elif mat == 'V':
            idx = random.randrange(m*p)
            c = random.randrange(rank)
            old_val = V[idx, c]
            new_val = -old_val if old_val != 0 else (1 if random.random() < 0.5 else -1)
            V[idx, c] = new_val
            new_err = compute_error(U, V, W)
            V[idx, c] = old_val
        else:
            idx = random.randrange(p*n)
            c = random.randrange(rank)
            old_val = W[idx, c]
            new_val = -old_val if old_val != 0 else (1 if random.random() < 0.5 else -1)
            W[idx, c] = new_val
            new_err = compute_error(U, V, W)
            W[idx, c] = old_val
        
        if new_err < current_err - 1e-15:
            current_err = new_err
            if current_err < best_err:
                best_err = current_err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                if best_err < 1e-10:
                    break
        else:
            # Revert
            pass
    
    return best_U, best_V, best_W, best_err


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Strategy: Try to find a decomposition with rank less than n*m*p = 112
    # Start with schoolbook (rank 112) and try to reduce
    
    # First, get schoolbook as baseline
    U_school, V_school, W_school = _schoolbook_decomposition(n, m, p)
    best_rank = U_school.shape[1]
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Try Strassen-based approaches for the 4x4x4 part
    if time.time() < deadline:
        # For 4x4x4, Strassen gives rank 49
        # For 4x4x7, we need to extend this
        # One approach: 4x7 = 4x4 + 4x3
        # The 4x4 part can use Strassen (49 terms)
        # The 4x3 part needs additional terms
        
        # Let's try a block approach:
        # A (4x4) split into 2x2 blocks
        # B (4x7) split into 2x2 blocks for first 6 cols, plus 2x1 for last col
        # This is getting complex. Let's try a different structural idea.
        
        # Use the 4x4x4 Strassen decomposition and extend to 4x7
        # by adding terms for the extra 3 columns
        
        strassen_4x4 = _strassen_4x4_4x4_terms()
        # This gives 49 terms for 4x4x4
        # For 4x4x7, we need to handle 3 more columns
        
        # For the extra columns (cols 4,5,6 of B), we can use schoolbook-like terms
        # A[i,j] * B[j,k] for k in 4,5,6
        extra_terms = []
        for i in range(n):
            for j in range(m):
                for k in range(4, p):
                    u = [0] * (n*m)
                    v = [0] * (m*p)
                    w = [0] * (p*n)
                    u[i*m+j] = 1
                    v[j*p+k] = 1
                    w[k*n+i] = 1
                    extra_terms.append((u, v, w))
        
        # Combine Strassen 4x4x4 terms with extra terms
        # But the Strassen terms are for 4x4x4, so W has size 16, not 28
        # We need to extend W to size 28 (p*n = 7*4 = 28)
        
        all_U = []
        all_V = []
        all_W = []
        
        for u, v, w in strassen_4x4:
            # u is length 16 (n*m), v is length 16 (m*p for 4x4), w is length 16 (p*n for 4x4)
            # For 4x4x7, v should be length 28 (m*p = 4*7), w should be length 28 (p*n = 7*4)
            v_ext = [0] * 28
            w_ext = [0] * 28
            for idx, val in enumerate(v):
                v_ext[idx] = val  # First 16 entries correspond to k=0..3
            for idx, val in enumerate(w):
                w_ext[idx] = val  # First 16 entries correspond to k=0..3
            all_U.append(u)
            all_V.append(v_ext)
            all_W.append(w_ext)
        
        for u, v, w in extra_terms:
            all_U.append(u)
            all_V.append(v)
            all_W.append(w)
        
        U_combined = np.array(all_U).T
        V_combined = np.array(all_V).T
        W_combined = np.array(all_W).T
        
        rank_combined = U_combined.shape[1]
        
        # Verify
        if _verify_decomposition(U_combined, V_combined, W_combined, n, m, p):
            if rank_combined < best_rank:
                best_rank = rank_combined
                best_result = {"U": U_combined.tolist(), "V": V_combined.tolist(), "W": W_combined.tolist()}
    
    # Try random search for better decompositions
    if time.time() < deadline:
        for attempt in range(5):
            if time.time() > deadline - 20:
                break
            
            # Try different ranks
            for target_rank in [80, 90, 100]:
                if time.time() > deadline - 15:
                    break
                
                U_rand, V_rand, W_rand = _random_decomposition(n, m, p, target_rank)
                
                # Local search
                U_opt, V_opt, W_opt, err = _local_search(
                    U_rand, V_rand, W_rand, n, m, p,
                    max_iters=200, time_limit=15.0
                )
                
                if err < 1e-10:
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_result = {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
                    break
    
    return best_result
# EVOLVE-BLOCK-END
