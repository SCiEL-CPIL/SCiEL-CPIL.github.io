"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """
    Verify tensor decomposition exactly for integer/half-integer entries.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Target T[i, j, k] = 1 if i=a*m+b, j=b*p+c, k=c*n+a else 0
    """
    # Convert to integers by scaling by 2 (handles half-integers)
    def scale2(mat):
        # Assume input is list of lists or array
        arr = np.array(mat, dtype=np.float64)
        arr2 = arr * 2
        # Check if close to integer
        if np.max(np.abs(arr2 - np.round(arr2))) > 1e-9:
            return None
        return np.round(arr2).astype(np.int64)
    
    U2 = scale2(U)
    V2 = scale2(V)
    W2 = scale2(W)
    
    if U2 is None or V2 is None or W2 is None:
        return False
        
    rank = U2.shape[1]
    
    # Build target tensor
    # Indices:
    # U index: i in [0, n*m) -> a = i // m, b = i % m
    # V index: j in [0, m*p) -> b = j // p, c = j % p
    # W index: k in [0, p*n) -> c = k // n, a = k % n
    # Note: The problem statement says W[k*n+i]. 
    # In the seed code: W[k*n+i] where k is the col of B (which is row of C? No, C is n x p).
    # Let's re-read the seed code carefully.
    # Seed:
    # for i in range(n):
    #   for j in range(m):
    #     for k in range(p):
    #       U[i*m+j][col] = 1  # Row index of A (n x m). i is row, j is col.
    #       V[j*p+k][col] = 1  # Row index of B (m x p). j is row, k is col.
    #       W[k*n+i][col] = 1  # Row index of C? C is n x p.
    # Wait, standard matrix mult C = A B. C is n x p.
    # C[a, c] = sum_b A[a, b] B[b, c].
    # The seed uses W[k*n+i]. Here k is in range(p) (col of B / col of C), i is in range(n) (row of A / row of C).
    # So W index = c * n + a.
    # This corresponds to flattening C as row-major? No, c*n+a means c is the "major" index, a is "minor".
    # If C is n x p, usually flat index is a * p + c.
    # But the seed uses k*n + i. Here k (col of C) is the major index, i (row of C) is minor.
    # This is effectively C^T flattened row-major, or C flattened column-major.
    # The task says: "output factor follows the official trace convention W[k*n+i,r]".
    # So we must stick to this convention.
    
    # Target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                u_idx = a * m + b
                v_idx = b * p + c
                w_idx = c * n + a  # Matches seed: k=c, i=a
                T[u_idx, v_idx, w_idx] = 1
                
    # Compute T_computed[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
    # Using scaled integers: U2, V2, W2 are 2*U, 2*V, 2*W.
    # Product is 8 * U V W.
    # So T_computed_scaled = einsum(...) should equal 8 * T.
    
    # Efficient computation:
    # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
    # Let M = U2.T @ V2  (shape rank x n*m x m*p? No, rank x (n*m) and (m*p) x rank -> rank x n*m x m*p is not standard matmul)
    # We can do:
    # tmp[i, j, r] = U2[i, r] * V2[j, r] -> (n*m, m*p, rank)
    # Then T_scaled[i, j, k] = sum_r tmp[i, j, r] * W2[k, r]
    
    # Using einsum is easiest and optimized in numpy
    T_computed = np.einsum('ir,jr,kr->ijk', U2, V2, W2, optimize=True)
    T_target_scaled = T * 8
    
    if not np.allclose(T_computed, T_target_scaled, atol=0):
        return False
        
    return True

def construct():
    deadline = time.time() + 140.0  # Safety margin
    n, m, p = DIMENSIONS
    
    # 1. Schoolbook Baseline (Rank 125)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_rank = rank_sb
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # 2. Attempt Block Decomposition (Kronecker Sum Strategy)
    # Split 5x5 into 2x2 + 3x3 blocks.
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A22 is 3x3.
    # B = [[B11, B12], [B21, B22]]
    # C = A B
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    
    # This approach is complex to map to the global index convention W[k*n+i].
    # The global indices mix the blocks.
    # Instead, we will use a "Random Search with Structure" approach.
    # We know rank 100 is achievable.
    # We will try to find a rank < 125 decomposition using random initialization and alternating minimization,
    # followed by integer quantization.
    
    # However, finding exact integer decompositions via continuous optimization is hard.
    # A more robust strategy for this specific task (5x5x5) is to use the known fact that
    # rank(5x5x5) <= 100.
    
    # Let's try a targeted search for rank 100 using the structure of Strassen/Laderman blocks if possible,
    # or just random search with verification.
    
    # Since random search for rank 100 with integers in {-1, 0, 1} is unlikely to hit exactly,
    # we will rely on the schoolbook solution as a safe fallback, but try to improve it slightly
    # using known low-rank identities for small sub-blocks if we can embed them.
    
    # Actually, there is a known rank 100 algorithm for 5x5.
    # It is often constructed by taking a 4x4 Strassen (rank 49) and adding the 5th dimension naively?
    # 4x4x4 rank 49.
    # If we split 5 = 4 + 1:
    # The cross terms with the 5th row/col require additional rank.
    # This is non-trivial to code from scratch correctly within time limits without a reference implementation.
    
    # Alternative: Use the "Kronecker Product" property if the matrix mult tensor had one, but it doesn't directly.
    
    # Let's try a simple local search from the schoolbook solution? No, that increases rank.
    
    # Strategy:
    # 1. Keep schoolbook as best.
    # 2. Try to construct a rank 100 solution using a heuristic search over a small subset of parameters
    #    derived from the 4x4 Strassen and 1x1 naive.
    
    # Given the difficulty and time constraint, and the fact that the inspiration suggests a Kronecker sum
    # of 2x2 Strassen (7) and 3x3 Laderman (15) + corrections, let's analyze that.
    # 2x2 Strassen rank 7.
    # 3x3 Laderman rank 15.
    # Direct sum rank = 7 + 15 = 22.
    # This handles the diagonal blocks A11B11 and A22B22.
    # We still need to compute:
    # A12 B21, A11 B12, A12 B22, A21 B11, A21 B12, A22 B21.
    # These are 2x3, 2x2, 2x3, 3x2, 3x2, 3x3 multiplications.
    # This is effectively computing the full 5x5 product.
    # The "correction terms" are not just rank-1 updates; they are full matrix multiplications of mixed sizes.
    # The rank of the full tensor is not simply the sum of diagonal block ranks.
    
    # Therefore, the "Kronecker sum" strategy as described is likely flawed or incomplete for the full 5x5 case
    # unless the "correction terms" are handled by a complex joint decomposition.
    
    # Let's stick to a robust random search for small integer entries, but limit the scope to be fast.
    # We will search for ranks 110, 115, 120, 124.
    # If we find one, we return it. Otherwise, schoolbook.
    
    # To make random search more effective, we can use the fact that the tensor is sparse.
    # But generating random sparse matrices that exactly match the tensor is hard.
    
    # Let's try one specific known construction:
    # The 5x5 matrix multiplication can be done in 100 multiplications.
    # One such algorithm is derived from the 4x4 Strassen (49) + 51 extra.
    # Without the explicit coefficients, we cannot hardcode it reliably.
    
    # Let's try a "Greedy" approach:
    # Start with random U, V, W of rank R.
    # Compute error.
    # Adjust W to minimize error (least squares).
    # Quantize W to nearest half-integer.
    # Verify.
    # If success, try R-1.
    
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    # Build target tensor T (unscaled, float for LS)
    T = np.zeros((n_m, m_p, p_n))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                T[a * m + b, b * p + c, c * n + a] = 1.0
    
    # Flatten T for LS
    # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
    # For fixed k, T[:, :, k] = sum_r W[k, r] * (U[:, r] * V[:, r])_outer
    # Let B_r = U[:, r] * V[:, r] (outer product)
    # We want to solve for W[k, :] such that sum_r W[k, r] B_r ~= T[:, :, k]
    
    def try_rank(rank):
        if time.time() > deadline - 20:
            return None
            
        # Random init
        U = np.random.randn(n_m, rank)
        V = np.random.randn(m_p, rank)
        
        # Precompute B_flat: (rank, n_m * m_p)
        B_flat = np.zeros((rank, n_m * m_p))
        for r in range(rank):
            B_flat[r] = np.outer(U[:, r], V[:, r]).flatten()
        
        # T_flat: (p_n, n_m * m_p)
        T_flat = T.transpose(2, 0, 1).reshape(p_n, n_m * m_p)
        
        # Solve B_flat.T @ W.T = T_flat.T
        # B_flat.T is (n_m * m_p, rank)
        # W.T is (rank, p_n)
        # T_flat.T is (n_m * m_p, p_n)
        
        # Least Squares
        try:
            W_solution, residuals, rank_ls, sv = np.linalg.lstsq(B_flat.T, T_flat.T, rcond=None)
        except:
            return None
            
        W = W_solution.T # (p_n, rank)
        
        # Quantize to half-integers
        W_q = np.round(W * 2) / 2.0
        
        # Verify with quantized W
        # We should also quantize U and V? 
        # The problem allows integer/half-integer entries for ALL factors.
        # If we only quantize W, U and V might be irrational, which is not allowed.
        # We must quantize all three.
        # This makes the search much harder.
        
        # Strategy: Quantize U, V, W to nearest half-integer and verify.
        U_q = np.round(U * 2) / 2.0
        V_q = np.round(V * 2) / 2.0
        W_q = np.round(W * 2) / 2.0
        
        if _verify(U_q.tolist(), V_q.tolist(), W_q.tolist(), n, m, p):
            return {"U": U_q.tolist(), "V": V_q.tolist(), "W": W_q.tolist()}
            
        return None

    # Try ranks from 124 down to 100
    # This is a long shot, but better than nothing if it works.
    # To save time, we limit the number of attempts per rank.
    
    for rank in range(124, 99, -1):
        if time.time() > deadline - 10:
            break
            
        found = False
        for attempt in range(20): # 20 attempts per rank
            if time.time() > deadline - 10:
                break
            res = try_rank(rank)
            if res:
                best_rank = rank
                best_result = res
                found = True
                break
        if found:
            break
            
    return best_result

# EVOLVE-BLOCK-END
