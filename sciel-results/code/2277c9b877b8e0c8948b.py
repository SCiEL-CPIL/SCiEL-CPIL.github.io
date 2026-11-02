import numpy as np
import time
import itertools

DIMENSIONS = [3, 4, 6]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is correct."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.shape != (n*m, W.shape[1]):
        return False
    if V.shape != (m*p, W.shape[1]):
        return False
    if W.shape != (p*n, U.shape[1]):
        return False
    
    rank = U.shape[1]
    np.random.seed(42)
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        AB = A @ B
        AB_decomp = np.zeros((n, p))
        for r in range(rank):
            L_r = np.sum(U[:, r] * A.flatten())
            R_r = np.sum(V[:, r] * B.flatten())
            AB_decomp += W[:, r] * L_r * R_r
            # W is (p*n, rank), so W[:, r] is (p*n,). 
            # We need to reshape W[:, r] to (n, p) or (p, n)?
            # Task says: W[k*n+i, r]. So index is k*n+i.
            # So W[:, r] is a vector of length n*p.
            # We need to map index k*n+i to (i, k).
            # Let's reshape carefully.
            # W_vec = W[:, r]. 
            # W_mat = W_vec.reshape(n, p) ? 
            # If index is k*n + i, then row k, col i?
            # Standard C-order: index = row * cols + col.
            # If k is row, n is cols? No, n is rows of A, p is cols of B.
            # Output is n x p.
            # Index k*n + i. If k goes 0..p-1 and i goes 0..n-1.
            # This implies the vector is ordered by k (outer), i (inner).
            # So reshape(n, p) would put i in rows?
            # Let's check: index 0 -> k=0, i=0.
            # index 1 -> k=0, i=1.
            # ...
            # index n-1 -> k=0, i=n-1.
            # index n -> k=1, i=0.
            # This is column-major for (k, i) if k is row?
            # If we reshape to (n, p) in C-order, index i*p + k.
            # Here index is k*n + i.
            # So we should reshape to (p, n) and then transpose?
            # Or just construct the matrix manually.
            
        # Let's re-verify the W indexing convention.
        # "W[k*n+i,r]"
        # Let's build the matrix properly.
        
    # Re-do verification with correct W indexing
    np.random.seed(42)
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        AB = A @ B
        AB_decomp = np.zeros((n, p))
        for r in range(rank):
            L_r = np.sum(U[:, r] * A.flatten())
            R_r = np.sum(V[:, r] * B.flatten())
            # Construct W matrix from vector W[:, r] with index k*n+i
            W_mat = np.zeros((n, p))
            for k in range(p):
                for i in range(n):
                    W_mat[i, k] = W[k*n + i, r]
            AB_decomp += W_mat * L_r * R_r
        if not np.allclose(AB, AB_decomp, atol=tol):
            return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

def _solve_3x2x3(time_limit=5.0):
    """Find a low-rank decomposition for 3x2 * 2x3."""
    n, m, p = 3, 2, 3
    start = time.time()
    best_rank = n * m * p
    best_U, best_V, best_W = _schoolbook(n, m, p)
    
    # Try ranks lower than schoolbook
    for rank in range(12, best_rank):
        if time.time() - start > time_limit:
            break
        # Simple random search with a few restarts
        for _ in range(10):
            if time.time() - start > time_limit:
                break
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
            if _verify(U, V, W, n, m, p):
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                return best_U, best_V, best_W, rank
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110 # Safety margin
    
    # 1. Solve the base case 3x2x3
    U_base, V_base, W_base, rank_base = _solve_3x2x3(time_limit=10.0)
    
    # 2. Construct 3x4x3 by splitting 4 into 2+2
    # A (3x4) = [A1 (3x2), A2 (3x2)]
    # B (4x3) = [B1 (2x3); B2 (2x3)]
    # AB = A1*B1 + A2*B2
    # We need to map the base decomposition indices to the full indices.
    
    # Base: n_b=3, m_b=2, p_b=3
    # Full: n_f=3, m_f=4, p_f=3
    # U_base is (6, rank_base). Indices: i*2 + j (i in 0..2, j in 0..1)
    # V_base is (6, rank_base). Indices: j*3 + k (j in 0..1, k in 0..2)
    # W_base is (9, rank_base). Indices: k*3 + i (k in 0..2, i in 0..2)
    
    # For 3x4x3:
    # U_full (12, 2*rank_base). 
    # For A1 (j=0,1), use U_base. For A2 (j=2,3), use U_base (shifted j).
    # V_full (12, 2*rank_base).
    # For B1 (j=0,1), use V_base. For B2 (j=2,3), use V_base (shifted j).
    # W_full (9, 2*rank_base).
    # Sum of two W_base.
    
    rank_343 = 2 * rank_base
    U_343 = np.zeros((3*4, rank_343))
    V_343 = np.zeros((4*3, rank_343))
    W_343 = np.zeros((3*3, rank_343))
    
    for r in range(rank_base):
        # Block 1: A1*B1
        # U: i in 0..2, j in 0..1. Index in U_full: i*4 + j.
        for i in range(3):
            for j in range(2):
                U_343[i*4 + j, r] = U_base[i*2 + j, r]
        # V: j in 0..1, k in 0..2. Index in V_full: j*3 + k.
        for j in range(2):
            for k in range(3):
                V_343[j*3 + k, r] = V_base[j*3 + k, r]
        # W: k in 0..2, i in 0..2. Index in W_full: k*3 + i.
        for k in range(3):
            for i in range(3):
                W_343[k*3 + i, r] += W_base[k*3 + i, r]
        
        # Block 2: A2*B2
        # U: i in 0..2, j in 2..3. Map j_full=2,3 to j_base=0,1.
        for i in range(3):
            for j_base in range(2):
                j_full = 2 + j_base
                U_343[i*4 + j_full, rank_base + r] = U_base[i*2 + j_base, r]
        # V: j in 2..3, k in 0..2. Map j_full=2,3 to j_base=0,1.
        for j_base in range(2):
            j_full = 2 + j_base
            for k in range(3):
                V_343[j_full*3 + k, rank_base + r] = V_base[j_base*3 + k, r]
        # W: Same as before, add to the same W slots?
        # Wait, the multiplication is A1B1 + A2B2.
        # The W part is the same for both terms because the output indices (i,k) are the same.
        # So we add the W contributions from both blocks.
        for k in range(3):
            for i in range(3):
                W_343[k*3 + i, rank_base + r] += W_base[k*3 + i, r]
                
    # Verify 3x4x3
    if not _verify(U_343, V_343, W_343, 3, 4, 3):
        # Fallback to schoolbook for 3x4x3 if construction failed (shouldn't happen)
        U_343, V_343, W_343 = _schoolbook(3, 4, 3)
        rank_343 = 36

    # 3. Construct 3x4x6 by splitting 6 into 3+3
    # B (4x6) = [B_col1 (4x3), B_col2 (4x3)]
    # AB = A*B_col1 + A*B_col2
    # We use the 3x4x3 decomposition twice.
    
    rank_final = 2 * rank_343
    U_final = np.zeros((3*4, rank_final))
    V_final = np.zeros((4*6, rank_final))
    W_final = np.zeros((6*3, rank_final))
    
    for r in range(rank_343):
        # Block 1: B_col1 (k=0..2)
        # U: Same as U_343.
        U_final[:, r] = U_343[:, r]
        # V: j in 0..3, k in 0..2. Index in V_final: j*6 + k.
        for j in range(4):
            for k in range(3):
                V_final[j*6 + k, r] = V_343[j*3 + k, r]
        # W: k in 0..2, i in 0..2. Index in W_final: k*3 + i.
        for k in range(3):
            for i in range(3):
                W_final[k*3 + i, r] = W_343[k*3 + i, r]
                
        # Block 2: B_col2 (k=3..5)
        # U: Same as U_343.
        U_final[:, rank_343 + r] = U_343[:, r]
        # V: j in 0..3, k in 3..5. Map k_full=3..5 to k_base=0..2.
        for j in range(4):
            for k_base in range(3):
                k_full = 3 + k_base
                V_final[j*6 + k_full, rank_343 + r] = V_343[j*3 + k_base, r]
        # W: k in 3..5, i in 0..2. Map k_full=3..5 to k_base=0..2.
        for k_base in range(3):
            k_full = 3 + k_base
            for i in range(3):
                W_final[k_full*3 + i, rank_343 + r] = W_343[k_base*3 + i, r]
                
    # Verify final
    if _verify(U_final, V_final, W_final, 3, 4, 6):
        return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
    else:
        # Fallback to schoolbook
        U_s, V_s, W_s = _schoolbook(3, 4, 6)
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
