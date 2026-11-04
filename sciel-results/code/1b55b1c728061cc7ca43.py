import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook solution (Rank 30)
    rank_school = nm * mp // p # No, rank is n*m*p
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    # Try to find a lower rank solution (Rank 26)
    target_rank = 26
    deadline = time.time() + 110 # Safety margin
    
    def residual(params):
        # Unpack params into U, V, W
        size_u = nm * target_rank
        size_v = mp * target_rank
        size_w = pn * target_rank
        
        U_flat = params[:size_u]
        V_flat = params[size_u:size_u+size_v]
        W_flat = params[size_u+size_v:]
        
        U = U_flat.reshape(nm, target_rank)
        V = V_flat.reshape(mp, target_rank)
        W = W_flat.reshape(pn, target_rank)
        
        # Calculate residual for all i, j, k, j', k', i''
        # We only need to check the 30 non-zero entries and a sample of zero entries
        # Actually, we must check ALL entries to ensure exactness later, 
        # but for optimization, we can focus on the non-zero ones and a subset of zeros
        # However, the zero entries are crucial to prevent "leakage".
        
        # Efficient calculation:
        # C_ik = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A_ij) * (sum_j' V[j'*p+k, r] B_j'k)
        # This is not a linear system in U,V,W directly in the residual sense of least_squares
        # because the equations are:
        # sum_r U[i*m+j, r] V[j'*p+k', r] W[k''*n+i'', r] = delta
        
        # This is a system of polynomial equations (cubic).
        # least_squares minimizes sum (f(x) - y)^2.
        # Here f(x) is the tensor entry, y is 0 or 1.
        
        res = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Non-zero entry: U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] should sum to 1
                    # Wait, the indices for the tensor entry (i,j), (j,k), (i,k)
                    # are U[i*m+j], V[j*p+k], W[k*n+i]
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    res.append(val - 1)
                    
        # Add a subset of zero entries to constrain the solution
        # Number of zero entries is nm*mp*pn - n*m*p
        # We can't add all 870 to the residual vector if we want speed, but 870 is small.
        # Let's add all zero entries.
        for i in range(n):
            for j in range(m):
                for j_prime in range(m):
                    for k in range(p):
                        for k_prime in range(p):
                            for i_prime in range(n):
                                # Check if this is a non-zero entry
                                if i == i_prime and j == j_prime and k == k_prime:
                                    continue
                                # Entry is (i,j), (j_prime, k_prime), (i_prime, k)
                                # Wait, the third index is (i', k'').
                                # The tensor is sum_{i,j,k} e_{ij} otimes e_{jk} otimes e_{ik}
                                # So the entry for indices (I, J, K) where I=(i,j), J=(j',k'), K=(i'',k'')
                                # is 1 if i=i'', j=j', k=k''
                                # My loop above for non-zero used i,j,k.
                                # Here I am iterating over all possible I, J, K.
                                
                                # Index mapping:
                                # I = i*m + j
                                # J = j_prime*p + k_prime
                                # K = k_prime*n + i_prime  <-- WAIT.
                                # The output index is (i'', k'').
                                # In the non-zero case, i''=i, k''=k.
                                # So K index is k_prime*n + i_prime.
                                # If it's a non-zero entry, k_prime must be k and i_prime must be i.
                                # So the condition for non-zero is:
                                # i == i_prime AND j == j_prime AND k == k_prime.
                                # And the K index used in the tensor definition is (i, k).
                                # So W index is k*n + i.
                                
                                # In the zero case:
                                # We pick arbitrary i, j, j', k', i'', k''.
                                # The tensor entry is 0 unless i=i'', j=j', k=k''.
                                # Note: The k in the condition comes from the B matrix index (j,k).
                                # The k'' in the C matrix index (i'', k'').
                                # So if i=i'', j=j', and k=k'', then it is 1.
                                
                                # Let's re-iterate carefully.
                                # Tensor T has indices (a, b, c) where a in [0, nm), b in [0, mp), c in [0, pn).
                                # a = i*m + j
                                # b = j'*p + k'
                                # c = k''*n + i''
                                # T[a,b,c] = 1 if i=i'' and j=j' and k=k''.
                                # Note: k is the second part of b. k'' is the second part of c.
                                # So condition: i (from a) == i'' (from c) AND j (from a) == j' (from b) AND k (from b) == k'' (from c).
                                
                                val = np.sum(U[i*m+j, :] * V[j_prime*p+k_prime, :] * W[k_prime*n+i_prime, :])
                                res.append(val) # Target is 0
                                
        return np.array(res)
        
    # This residual function is too slow if called many times in least_squares
    # because it loops over ~900 entries and does numpy ops.
    # 900 * (26 mults) = 23400 ops per residual.
    # least_squares might call this 100-1000 times.
    # 23M ops is fine.
    
    best_result = None
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    for attempt in range(5):
        if time.time() > deadline:
            break
            
        # Initialize with schoolbook + noise
        # We need to map schoolbook (rank 30) to rank 26?
        # No, just random init.
        np.random.seed(42 + attempt)
        U0 = np.random.randn(nm, target_rank)
        V0 = np.random.randn(mp, target_rank)
        W0 = np.random.randn(pn, target_rank)
        
        # Scale to help convergence
        U0 /= np.sqrt(target_rank)
        V0 /= np.sqrt(target_rank)
        W0 /= np.sqrt(target_rank)
        
        params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            sol = least_squares(residual, params0, method='lm', max_nfev=1000)
            if sol.success:
                # Check if we can round to half-integers
                U_sol = sol.x[:nm*target_rank].reshape(nm, target_rank)
                V_sol = sol.x[nm*target_rank:nm*target_rank+mp*target_rank].reshape(mp, target_rank)
                W_sol = sol.x[nm*target_rank+mp*target_rank:].reshape(pn, target_rank)
                
                # Round to nearest 0.5
                U_round = np.round(U_sol * 2) / 2
                V_round = np.round(V_sol * 2) / 2
                W_round = np.round(W_sol * 2) / 2
                
                # Verify exactness
                if _verify_decomposition(U_round, V_round, W_round, n, m, p):
                    return {
                        "U": U_round.tolist(),
                        "V": V_round.tolist(),
                        "W": W_round.tolist()
                    }
        except Exception:
            pass
            
    # Fallback to schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }

def _verify_decomposition(U, V, W, n, m, p):
    nm = n * m
    mp = m * p
    pn = p * n
    R = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for j_prime in range(m):
                for k in range(p):
                    for k_prime in range(p):
                        for i_prime in range(n):
                            # Calculate sum
                            s = np.sum(U[i*m+j, :] * V[j_prime*p+k, :] * W[k_prime*n+i_prime, :])
                            # Check target
                            if i == i_prime and j == j_prime and k == k_prime:
                                if abs(s - 1) > 1e-9:
                                    return False
                            else:
                                if abs(s) > 1e-9:
                                    return False
    return True
