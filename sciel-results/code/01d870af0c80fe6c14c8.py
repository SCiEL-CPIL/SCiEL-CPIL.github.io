import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 9]

# Pre-compute Strassen 2x2x2 terms
_STRASSEN_222 = [
    ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
    ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
    ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
    ([0, 0, 0, 1], [0, -1, 1, 0], [1, 0, 1, 0]),
    ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
    ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
    ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
]

def _get_strassen_444_terms():
    """Generate the 49 rank-1 terms for 4x4x4 using Strassen."""
    terms = []
    # We need to map the 2x2x2 Strassen to 4x4x4.
    # This is done by applying Strassen to the 4x4 blocks.
    # U, V, W are 4-vectors.
    # We iterate over the 4x4 grid of blocks? No, Strassen is recursive.
    # For 4x4x4, we use 7 products of 2x2x2.
    # Each 2x2x2 product is rank 7. Total 49.
    
    # Let's just hardcode the 49 terms or generate them via the block method.
    # The current program uses _add_444 which effectively does this.
    # We will reuse the logic from _add_444 to generate the list of (u,v,w) tuples.
    
    U_sums = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    V_sums = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [0, -1, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    W_sums = [
        [1, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 1, 0],
        [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]
    
    # These are for a single 4x4x4 block where indices are 0..3.
    # To get the full 4x4x4 tensor terms, we need to expand these into the 4x4 grid.
    # Wait, the U_sums above are for the 4 elements of the 2x2 block?
    # No, in _add_444, the loop `for s in range(4)` iterates over the 4 elements of the 2x2 block.
    # So `u[s]` is the coefficient for the s-th element of the 2x2 block.
    # The full 4x4x4 term is a sum of these 4 sub-terms?
    # No, the rank-1 term is a single vector of length 4 (for the 2x2 block).
    # When we flatten the 4x4 matrix, the indices are 0..15.
    # The mapping from 2x2 block index (0..3) to 4x4 index (0..15) depends on the block position.
    
    # Let's generate the 49 terms for the first 4x4x4 block (cols 0-3 of B, rows 0-3 of C).
    # We will do this for all 3 blocks of B (width 3? No, width 4, 4, 1).
    
    # This is getting complicated to generate from scratch.
    # I will just use the existing _build_rank114 logic to get the 114 terms,
    # then add random candidates.
    pass

def _build_rank114_terms(n, m, p):
    """Returns list of (u, v, w) tuples for the 114 rank decomposition."""
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def _add_444(U, V, W, col, a_base, b_base, c_base):
        U_sums = [
            [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
            [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
        ]
        V_sums = [
            [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [0, -1, 1, 0],
            [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
        ]
        W_sums = [
            [1, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 1, 0],
            [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
        ]
        for t in range(7):
            u, v, w = U_sums[t], V_sums[t], W_sums[t]
            for s in range(4):
                if u[s]:
                    U[a_base + s][col] += u[s]
                if v[s]:
                    V[b_base + s][col] += v[s]
                if w[s]:
                    W[c_base + s][col] += w[s]
            col += 1
        return col

    def _schoolbook_block(U, V, W, col, a_base, b_base, c_base, ar, ac, bc):
        for i in range(ar):
            for j in range(ac):
                for k in range(bc):
                    U[a_base + i * ac + j][col] += 1
                    V[b_base + j * bc + k][col] += 1
                    W[c_base + i * bc + k][col] += 1
                    col += 1
        return col

    col = _add_444(U, V, W, col, 0, 0, 0)
    col = _add_444(U, V, W, col, 0, 4, 4)
    col = _schoolbook_block(U, V, W, col, 0, 8, 8, 4, 4, 1)
    
    terms = []
    for r in range(col):
        u = [U[i][r] for i in range(n*m)]
        v = [V[i][r] for i in range(m*p)]
        w = [W[i][r] for i in range(p*n)]
        terms.append((u, v, w))
    return terms

def _generate_random_candidates(n, m, p, num_candidates):
    """Generate random sparse rank-1 tensors with entries in {-1, 0, 1}."""
    candidates = []
    np.random.seed(42)
    for _ in range(num_candidates):
        u = np.zeros(n*m, dtype=int)
        v = np.zeros(m*p, dtype=int)
        w = np.zeros(p*n, dtype=int)
        
        # Randomly set a few entries to +/- 1
        for _ in range(3):
            idx = np.random.randint(n*m)
            u[idx] = np.random.choice([-1, 1])
        for _ in range(3):
            idx = np.random.randint(m*p)
            v[idx] = np.random.choice([-1, 1])
        for _ in range(3):
            idx = np.random.randint(p*n)
            w[idx] = np.random.choice([-1, 1])
            
        # Ensure it's not zero
        if np.any(u) and np.any(v) and np.any(w):
            candidates.append((list(u), list(v), list(w)))
    return candidates

def _verify(U, V, W, n, m, p):
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    T = np.einsum('ar,br,cr->abc', Ua, Va, Wa)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i * m + j, j * p + k, k * n + i] - 1.0) > 1e-9:
                    return False
    T2 = T.copy()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T2[i * m + j, j * p + k, k * n + i] = 0.0
    if np.max(np.abs(T2)) > 1e-9:
        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # 1. Get the base 114 terms
    base_terms = _build_rank114_terms(n, m, p)
    
    # 2. Generate random candidates
    random_terms = _generate_random_candidates(n, m, p, 50)
    
    # 3. Combine
    all_terms = base_terms + random_terms
    
    # 4. Greedy Sparse Decomposition
    # Target Tensor T
    T_flat = np.zeros(n*m * m*p * p*n)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_flat[(i*m+j) * (m*p) * (p*n) + (j*p+k) * (p*n) + (k*n+i)] = 1.0
                
    Residual = T_flat.copy()
    Selected_Coeffs = []
    Selected_Terms = []
    
    # Pre-flatten all terms
    flat_terms = []
    for u, v, w in all_terms:
        ua = np.array(u)
        va = np.array(v)
        wa = np.array(w)
        tensor = np.einsum('a,b,c->abc', ua, va, wa)
        flat_terms.append(tensor.flatten())
        
    # Greedy loop
    max_iter = 100
    for _ in range(max_iter):
        if time.time() > deadline:
            break
            
        # Calculate correlations
        # Residual is 1D. flat_terms is list of 1D.
        # corr = [np.dot(Residual, ft) / np.linalg.norm(ft) for ft in flat_terms]
        # To speed up, we can use matrix multiplication if we stack them.
        # But 164 terms x 20736 size is small.
        
        best_idx = -1
        best_corr = 0
        for idx, ft in enumerate(flat_terms):
            norm_sq = np.dot(ft, ft)
            if norm_sq == 0:
                continue
            corr = np.dot(Residual, ft) / norm_sq
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_idx = idx
                
        if best_idx == -1 or abs(best_corr) < 1e-12:
            break
            
        # Update Residual
        Residual -= best_corr * flat_terms[best_idx]
        Selected_Coeffs.append(best_corr)
        Selected_Terms.append(all_terms[best_idx])
        
        # Check if Residual is zero
        if np.linalg.norm(Residual) < 1e-9:
            break
            
    # 5. Construct result from Selected_Terms
    if Selected_Terms:
        rank = len(Selected_Terms)
        U = [[0] * rank for _ in range(n*m)]
        V = [[0] * rank for _ in range(m*p)]
        W = [[0] * rank for _ in range(p*n)]
        
        for r, (u, v, w) in enumerate(Selected_Terms):
            # Coefficient is applied to the term?
            # The decomposition is sum coeff_r * (u_r v_r w_r).
            # We can absorb the coefficient into one of the vectors.
            # Since we need integer/half-integer entries, we should check if coeff is "nice".
            # For now, just store the term. The coefficient is 1 in the sum if we scale the term.
            # Wait, the greedy algorithm found coefficients.
            # We must verify if the resulting U, V, W have valid entries.
            
            # Scale u by best_corr?
            # If best_corr is 0.5, then u entries become 0.5, -0.5, etc.
            # This is allowed.
            
            # We just add the term as is, but we need to handle the coefficient.
            # Actually, the term in the dictionary is fixed.
            # The decomposition is sum c_r * T_r.
            # We can define U[:, r] = u_r, V[:, r] = v_r, W[:, r] = w_r.
            # Then the tensor is sum U[:, r] V[:, r] W[:, r].
            # This equals T if and only if the coefficients are 1.
            # But our greedy algorithm used coefficients.
            # So we must scale one vector by the coefficient.
            
            c = Selected_Coeffs[r]
            # Scale u by c
            U_col = [u[i] * c for i in range(n*m)]
            V_col = [v[i] for i in range(m*p)]
            W_col = [w[i] for i in range(p*n)]
            
            # Check if entries are close to half-integers
            valid = True
            for val in U_col + V_col + W_col:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    valid = False
                    break
            
            if valid:
                for i in range(n*m): U[i][r] = U_col[i]
                for i in range(m*p): V[i][r] = V_col[i]
                for i in range(p*n): W[i][r] = W_col[i]
            else:
                # If not valid, we can't use this term directly in this way.
                # This greedy approach is flawed for exact integer constraints.
                # Fallback to base terms if greedy fails validity.
                return _build_rank114(n, m, p)
                
        # Verify the greedy result
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}

    # Fallback to base 114
    return _build_rank114(n, m, p)
