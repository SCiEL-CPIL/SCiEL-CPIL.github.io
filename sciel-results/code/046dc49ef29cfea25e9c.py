import numpy as np
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
        
    # Try to find rank 55
    target_rank = 55
    if target_rank >= rank_schoolbook:
        return {"U": U_school, "V": V_school, "W": W_school}
        
    start_time = time.time()
    deadline = start_time + 110  # Leave 40s margin
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Pre-compute indices for the linear system
    # We solve for W given U, V.
    # A * W_flat = b
    # A is (n*m*m*p*p*n, pn*target_rank)
    # This is too large for dense numpy if we do it naively?
    # 3600 x 1100 is fine.
    
    A = np.zeros((n * m * m * p * p * n, pn * target_rank))
    b = np.zeros(n * m * m * p * p * n)
    
    eq_idx = 0
    for I in range(n):
        for K in range(p):
            for i in range(n):
                for j in range(m):
                    for j_prime in range(m):
                        for k_prime in range(p):
                            if i == I and j == j_prime and k_prime == K:
                                b[eq_idx] = 1.0
                            # A[eq_idx, (K*n + I)*r + c] = U[i*m+j, c] * V[j_prime*p+k_prime, c]
                            # We fill A in the loop below
                            eq_idx += 1
                            
    # We need to fill A based on U, V.
    # This is slow if we do it inside the loop.
    # But U, V change.
    
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        U_rand = np.random.randn(nm, target_rank)
        V_rand = np.random.randn(mp, target_rank)
        
        # Fill A
        A[:] = 0
        eq_idx = 0
        for I in range(n):
            for K in range(p):
                for i in range(n):
                    for j in range(m):
                        for j_prime in range(m):
                            for k_prime in range(p):
                                # Coeff for W[(K,I), c] is U[i*m+j, c] * V[j_prime*p+k_prime, c]
                                # We need to add this to A[eq_idx, (K*n+I)*r + c]
                                # This is a rank-1 update? No, just filling.
                                # Vectorized:
                                # A[eq_idx, (K*n+I)*r : (K*n+I)*r + r] = U_rand[i*m+j, :] * V_rand[j_prime*p+k_prime, :]
                                A[eq_idx, (K * n + I) * target_rank : (K * n + I) * target_rank + target_rank] = \
                                    U_rand[i * m + j, :] * V_rand[j_prime * p + k_prime, :]
                                eq_idx += 1
                                
        try:
            W_flat, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        except:
            continue
            
        if residuals is None or len(residuals) == 0 or residuals[0] > 1e-9:
            continue
            
        W_candidate = W_flat.reshape(pn, target_rank)
        
        # Check rationality (optional, but good for verification)
        # For now, if residual is small, assume it's valid.
        # The problem allows half-integers, but float solution might be irrational.
        # However, if a rational solution exists, lstsq might find a close one.
        # We should verify the tensor identity exactly with the found U, V, W.
        
        # Verify
        U_np = U_rand
        V_np = V_rand
        W_np = W_candidate
        
        # Compute tensor T = sum_c U[:,c] outer V[:,c] outer W[:,c]
        # T[i*m+j, j'*p+k', K*n+I] should be delta(i,I)delta(j,j')delta(k',K)
        
        # We can check a few random entries or all 3600.
        # 3600 is small.
        valid = True
        for I in range(n):
            for K in range(p):
                for i in range(n):
                    for j in range(m):
                        for j_prime in range(m):
                            for k_prime in range(p):
                                # Sum over c
                                s_val = 0.0
                                for c in range(target_rank):
                                    s_val += U_np[i*m+j, c] * V_np[j_prime*p+k_prime, c] * W_np[K*n+I, c]
                                target = 1.0 if (i==I and j==j_prime and k_prime==K) else 0.0
                                if abs(s_val - target) > 1e-5:
                                    valid = False
                                    break
                        if not valid: break
                    if not valid: break
                if not valid: break
            if not valid: break
        if not valid: break
        
        if valid:
            return {
                "U": U_np.tolist(),
                "V": V_np.tolist(),
                "W": W_np.tolist()
            }
            
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
