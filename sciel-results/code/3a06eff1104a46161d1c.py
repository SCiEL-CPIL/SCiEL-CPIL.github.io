import numpy as np
import time

DIMENSIONS = [2, 2, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=int)
    V_school = np.zeros((m * p, rank_school), dtype=int)
    W_school = np.zeros((p * n, rank_school), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    # Try to find lower rank
    for R in range(14, rank_school):
        found = False
        for _ in range(1000): # Limit iterations to avoid timeout
            U = np.random.randint(-1, 2, (n * m, R))
            V = np.random.randint(-1, 2, (m * p, R))
            W = np.random.randint(-1, 2, (p * n, R))
            
            # Check validity
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Target is 1 for matching indices
                        # U index: i*m+j
                        # V index: j*p+k
                        # W index: k*n+i
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        s = np.dot(U[u_idx], V[v_idx]) # Wait, this is not the right contraction
                        # The contraction is sum_r U[i,j,r] V[j,k,r] W[i,k,r]
                        # We need to compute this for all i,j,k
                        pass
            
            # Efficient check
            # Build the tensor from U, V, W
            # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # We need T[i,j,k] == 1 for all i,j,k
            # And T[i,j,k'] == 0 for mismatched?
            # Actually, the tensor T is fixed.
            # The decomposition defines a tensor T'.
            # We need T' == T.
            # T has 1s at (i,j,k) corresponding to the multiplication terms.
            # But wait, the tensor space is A* x B* x C.
            # The basis is e_ij x e_jk x e_ik.
            # So T'_{ij, jk, ik} must be 1.
            # And T'_{ij, j'k', i'k'} must be 0 if (j,k) != (j',k') or (i,k) != (i',k').
            # Actually, the indices are linked.
            # For a fixed r, the term is u_r x v_r x w_r.
            # The sum is T'.
            # We check if T' matches T.
            
            match = True
            # Check 1s
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(R):
                            val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        if val != 1:
                            match = False
                            break
                    if not match: break
                if not match: break
            if not match: continue
            
            # Check 0s (mismatches)
            # We only need to check a few mismatches to be sure?
            # If the 1s are correct, are the 0s automatically correct?
            # No.
            # But for small R, it's likely.
            # Let's check all possible (i,j) and (j',k') and (i',k')
            # This is n*m * m*p * p*n = 4 * 10 * 10 = 400 checks.
            # Each check is O(R). 400 * 19 = 7600 ops. Fast.
            
            for i in range(n):
                for j in range(m):
                    for jp in range(m * p): # j', k'
                        j_prime = jp // p
                        k_prime = jp % p
                        for ip in range(p * n): # i', k'
                            i_prime = ip // n
                            k_prime_2 = ip % n
                            # Target is 1 if j==j_prime and i==i_prime and k==k_prime
                            # Wait, the indices in the tensor are (i,j), (j,k), (i,k).
                            # So the "position" in the tensor is determined by i,j,k.
                            # The value T_{i,j,k} is 1.
                            # Any other combination of indices (i,j), (j',k'), (i',k') that is NOT (i,j,k)
                            # must have value 0.
                            # But the tensor is sparse.
                            # T_{alpha, beta, gamma} is 1 only if alpha=(i,j), beta=(j,k), gamma=(i,k) for some i,j,k.
                            # So if we pick alpha=(i,j), beta=(j',k'), gamma=(i',k')
                            # and it is not the case that j==j' and i==i' and k==k', then T should be 0.
                            
                            target = 1 if (j == j_prime and i == i_prime and k == k_prime) else 0
                            # Wait, if j!=j_prime, then beta is not (j,k).
                            # So the entry T_{(i,j), (j',k'), (i',k')} is 0.
                            
                            val = 0
                            for r in range(R):
                                val += U[i*m+j, r] * V[j_prime*p+k_prime, r] * W[i_prime*n+k_prime_2, r]
                            if val != target:
                                match = False
                                break
                        if not match: break
                    if not match: break
                if not match: break
            
            if match:
                found = True
                best_U, best_V, best_W = U, V, W
                best_rank = R
                break
        
        if found:
            break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
