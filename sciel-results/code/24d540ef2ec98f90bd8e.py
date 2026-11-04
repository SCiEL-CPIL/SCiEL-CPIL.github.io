"""Program for matrix multiplication 2x3x7 tensor decomposition search."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix
    by a 3 by 7 matrix.
    
    Returns:
        dict: {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}
    """
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 7
    nm, mp, pn = n * m, m * p, p * n  # 6, 21, 14
    
    # Start with schoolbook decomposition (rank 42)
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()
    
    def verify_decomposition(U, V, W, tol=1e-10):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        rank = U.shape[1]
        
        # Check 1: For all (i,j,k), T[i*m+j, j*p+k, k*n+i] should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val - 1.0) > tol:
                        return False
        
        # Check 2: For random invalid (a,b,c), T should be 0
        for _ in range(2000):
            a = np.random.randint(0, nm)
            b = np.random.randint(0, mp)
            c = np.random.randint(0, pn)
            # Check if this is a valid multiplication term
            i_a = a // m
            j_a = a % m
            j_b = b // p
            k_b = b % p
            k_c = c // n
            i_c = c % n
            if not (j_a == j_b and i_a == i_c and k_b == k_c):
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val) > tol:
                    return False
        
        return True
    
    def berlekamp_massey(sequence):
        """
        Berlekamp-Massey algorithm to find the shortest linear recurrence.
        Returns the recurrence coefficients and the minimal rank estimate.
        """
        if len(sequence) < 2:
            return [], 0
        
        # Initialize
        n_seq = len(sequence)
        # C(x) = 1 + c1*x + c2*x^2 + ...
        C = [1.0] + [0.0] * (n_seq - 1)
        B = [1.0] + [0.0] * (n_seq - 1)
        L = 0  # current length
        m = 1  # number of steps since last update
        b = 1.0  # previous discrepancy
        
        for N in range(n_seq):
            # Compute discrepancy
            d = sequence[N]
            for i in range(1, L + 1):
                if N - i >= 0:
                    d += C[i] * sequence[N - i]
            
            if abs(d) < 1e-12:
                m += 1
            else:
                if 2 * L <= N:
                    # Update L
                    T = C[:]
                    factor = d / b
                    # C(x) = C(x) - factor * x^m * B(x)
                    for i in range(n_seq - m + 1):
                        C[i + m] -= factor * B[i]
                    L = N + 1 - L
                    B = T
                    b = d
                    m = 1
                else:
                    # C(x) = C(x) - factor * x^m * B(x)
                    factor = d / b
                    for i in range(n_seq - m + 1):
                        C[i + m] -= factor * B[i]
                    m += 1
        
        # The recurrence is: s[n] = -sum_{i=1}^{L} C[i] * s[n-i]
        # Minimal rank is L (for Hankel determinants)
        return C[:L + 1], L
    
    # Strategy 1: Use Berlekamp-Massey on Hankel determinants of tensor flattening
    # to empirically determine the minimal rank
    
    # Flatten the tensor T into a vector
    # T[a,b,c] = 1 if (a,b,c) corresponds to (i,j,k) with a=i*m+j, b=j*p+k, c=k*n+i
    # else 0
    total_size = nm * mp * pn
    T_flat = np.zeros(total_size)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                idx = a * mp * pn + b * pn + c
                T_flat[idx] = 1.0
    
    # Create different flattening modes for Hankel analysis
    # Mode 1: (a, b*p*n + c) - treat a as one index, (b,c) as another
    T_mat1 = T_flat.reshape(nm, mp * pn)  # (6, 294)
    
    # Mode 2: (b, a*p*n + c)
    T_mat2 = T_flat.reshape(mp, nm * pn)  # (21, 84)
    
    # Mode 3: (c, a*m*p + b)
    T_mat3 = T_flat.reshape(pn, nm * mp)  # (14, 126)
    
    # For each mode, compute the sequence of Hankel determinants
    # D_k = det of k x k leading principal submatrix
    # The minimal rank is where the determinants first become zero
    
    def hankel_sequence(mat, max_k):
        """Compute sequence of Hankel determinants for a matrix."""
        rows, cols = mat.shape
        seq = []
        for k in range(1, min(rows, cols, max_k) + 1):
            # Take k x k submatrix
            sub = mat[:k, :k]
            try:
                det = np.linalg.det(sub)
                seq.append(det)
            except np.linalg.LinAlgError:
                seq.append(0.0)
        return seq
    
    # Compute Hankel sequences for each mode
    max_k = min(nm, mp * pn)
    seq1 = hankel_sequence(T_mat1, max_k)
    seq2 = hankel_sequence(T_mat2, min(mp, nm * pn))
    seq3 = hankel_sequence(T_mat3, min(pn, nm * mp))
    
    # Apply Berlekamp-Massey to estimate minimal rank
    # The rank is the smallest k such that all (k+1)x(k+1) Hankel determinants are zero
    # This corresponds to the point where the sequence becomes zero
    
    def estimate_rank_from_sequence(seq):
        """Estimate rank from Hankel determinant sequence."""
        if not seq:
            return 0
        # Find the first zero
        for i, val in enumerate(seq):
            if abs(val) < 1e-10:
                return i  # rank is i (0-indexed, so rank = i)
        return len(seq)
    
    rank_est1 = estimate_rank_from_sequence(seq1)
    rank_est2 = estimate_rank_from_sequence(seq2)
    rank_est3 = estimate_rank_from_sequence(seq3)
    
    # The true rank is the maximum of these estimates
    bm_rank_estimate = max(rank_est1, rank_est2, rank_est3)
    
    # For the matrix multiplication tensor, we expect rank >= n*m*p = 42
    # but the BM approach might give a lower bound
    
    # Strategy 2: Structured rank reduction using the small dimensions
    # Since n=2 is small, we can try to exploit this
    
    # Key insight: For each output position (k, i), we need:
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # This is a dot product of length m=3
    
    # We can try to share computation between different (k,i) pairs
    
    # Let's try a structured approach: group by i (output row)
    # For i=0: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] for k=0..6
    # For i=1: C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k] for k=0..6
    
    # Total terms: 2 * 3 * 7 = 42
    
    # Can we reduce? Let's try to find if there's a linear dependency
    
    # Approach: Try to express one term as a combination of others
    # This is equivalent to finding if the 42 rank-1 tensors are linearly dependent
    
    # Let's try random projections with verification for ranks below 42
    
    for target_rank in range(min(bm_rank_estimate, 41), 19, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations with different seeds
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            seed = target_rank * 10000 + attempt * 100 + 42
            np.random.seed(seed)
            
            # Random projection approach
            # Project the 42-dimensional space to target_rank dimensions
            U_proj = np.random.randn(target_rank, rank_sb)
            V_proj = np.random.randn(target_rank, rank_sb)
            W_proj = np.random.randn(target_rank, rank_sb)
            
            U_new = U_sb @ U_proj.T  # (nm, target_rank)
            V_new = V_sb @ V_proj.T  # (mp, target_rank)  
            W_new = W_sb @ W_proj.T  # (pn, target_rank)
            
            if verify_decomposition(U_new, V_new, W_new):
                best_rank = target_rank
                best_U = U_new.tolist()
                best_V = V_new.tolist()
                best_W = W_new.tolist()
                break
        
        if best_rank < 42:
            break
    
    # If no improvement found, try a more targeted structured approach
    if best_rank == 42:
        # Try to find rank 41 by looking for a single linear dependency
        # among the 42 schoolbook terms
        
        # The 42 terms are indexed by (i,j,k)
        # We need to find coefficients c_{i,j,k} not all zero such that
        # sum_{i,j,k} c_{i,j,k} * e_i ⊗ e_j ⊗ e_k = 0 in the tensor space
        
        # But each e_i ⊗ e_j ⊗ e_k is a basis vector, so they're linearly independent
        # This means rank 42 is minimal for the schoolbook structure
        
        # However, we can try different decompositions that don't correspond
        # to the schoolbook basis
        
        # Let's try a parameterized family of decompositions
        # For each (i,k), we can write:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
        
        # We can introduce auxiliary variables to share computation
        # For example, let X_j = A[0,j] + A[1,j] for j=0,1,2
        # Then we can compute some combinations using X_j
        
        # This is getting complex; let's try a direct search
        
        # Try to find a rank 41 decomposition by solving a system
        # Assume we can remove one term, say (i0,j0,k0)
        # Then we need to adjust the remaining 41 terms to compensate
        
        # This is a nonlinear problem, so let's try random perturbations
        
        for i0 in range(n):
            for j0 in range(m):
                for k0 in range(p):
                    if time.time() > deadline - 20:
                        break
                    
                    # Try to remove term (i0, j0, k0)
                    # and adjust others to compensate
                    
                    # The term (i0,j0,k0) contributes to T[i0*m+j0, j0*p+k0, k0*n+i0]
                    # We need to distribute this contribution among other terms
                    
                    # This is equivalent to finding a rank-1 tensor that equals
                    # the removed term, which is impossible unless we change the structure
                    
                    # Let's try a different approach: use the fact that
                    # for fixed (i,k), we have 3 terms (j=0,1,2)
                    # Can we combine two of them?
                    
                    # For (i0, k0), the terms are (i0,0,k0), (i0,1,k0), (i0,2,k0)
                    # We can't simply combine them because they have different
                    # dependencies on A and B
                    
                    pass
    
    # Final strategy: Try to find a lower rank using gradient-based optimization
    # with a smooth surrogate for the exactness constraint
    
    if best_rank == 42 and time.time() < deadline - 30:
        # Use a smooth penalty for exactness
        # Objective: minimize rank + penalty * sum of squared violations
        
        # Start from a random initialization
        for target_rank in [41, 40]:
            if time.time() > deadline - 10:
                break
            
            np.random.seed(12345)
            
            # Random initialization
            U_opt = np.random.randn(nm, target_rank) * 0.1
            V_opt = np.random.randn(mp, target_rank) * 0.1
            W_opt = np.random.randn(pn, target_rank) * 0.1
            
            # Gradient descent with exactness penalty
            lr = 0.01
            penalty_weight = 100.0
            
            for step in range(500):
                if time.time() > deadline - 10:
                    break
                
                # Compute gradient of penalty
                # Penalty = sum_{valid (a,b,c)} (sum_r U[a,r]*V[b,r]*W[c,r] - 1)^2
                #        + sum_{invalid (a,b,c)} (sum_r U[a,r]*V[b,r]*W[c,r])^2
                
                U_grad = np.zeros_like(U_opt)
                V_grad = np.zeros_like(V_opt)
                W_grad = np.zeros_like(W_opt)
                
                # For valid terms
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            val = np.sum(U_opt[a, :] * V_opt[b, :] * W_opt[c, :])
                            err = val - 1.0
                            # d(err^2)/dU[a,r] = 2*err*V_opt[b,r]*W_opt[c,r]
                            U_grad[a, :] += 2 * penalty_weight * err * V_opt[b, :] * W_opt[c, :]
                            V_grad[b, :] += 2 * penalty_weight * err * U_opt[a, :] * W_opt[c, :]
                            W_grad[c, :] += 2 * penalty_weight * err * U_opt[a, :] * V_opt[b, :]
                
                # For invalid terms (sample a few)
                for _ in range(100):
                    a = np.random.randint(0, nm)
                    b = np.random.randint(0, mp)
                    c = np.random.randint(0, pn)
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if not (j_a == j_b and i_a == i_c and k_b == k_c):
                        val = np.sum(U_opt[a, :] * V_opt[b, :] * W_opt[c, :])
                        U_grad[a, :] += 2 * penalty_weight * val * V_opt[b, :] * W_opt[c, :]
                        V_grad[b, :] += 2 * penalty_weight * val * U_opt[a, :] * W_opt[c, :]
                        W_grad[c, :] += 2 * penalty_weight * val * U_opt[a, :] * V_opt[b, :]
                
                # Update
                U_opt -= lr * U_grad
                V_opt -= lr * V_grad
                W_opt -= lr * W_grad
                
                # Check if we've found a valid decomposition
                if step % 50 == 0 and verify_decomposition(U_opt, V_opt, W_opt):
                    best_rank = target_rank
                    best_U = U_opt.tolist()
                    best_V = V_opt.tolist()
                    best_W = W_opt.tolist()
                    break
            
            if best_rank < 42:
                break
    
    # Return the best result found
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
