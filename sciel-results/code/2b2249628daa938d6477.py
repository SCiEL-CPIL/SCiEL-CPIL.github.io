import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0  # 10s safety margin
    
    # 1. Construct Schoolbook Decomposition
    R0 = n * m * p
    U0 = np.zeros((n * m, R0), dtype=np.float64)
    V0 = np.zeros((m * p, R0), dtype=np.float64)
    W0 = np.zeros((p * n, R0), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1
                
    def to_list(mat):
        lst = []
        for row in mat:
            lst_row = []
            for val in row:
                val_rounded = round(val * 2) / 2
                if abs(val_rounded - round(val_rounded)) < 1e-9:
                    lst_row.append(int(val_rounded))
                else:
                    lst_row.append(val_rounded)
            lst.append(lst_row)
        return lst

    # Fallback: return schoolbook if we run out of time or fail to improve
    def get_schoolbook():
        return {"U": to_list(U0), "V": to_list(V0), "W": to_list(W0)}

    # Strategy: Greedy Column Merging
    # We attempt to reduce the rank by merging pairs of columns.
    # For a set of active columns S, we check if we can find a new column r*
    # (with small integer/half-integer entries) such that the tensor contribution
    # of r* can "absorb" the contributions of two existing columns r1, r2.
    # 
    # Specifically, if we have a current decomposition with columns C = {r1, ..., rR},
    # and we want to replace r1 and r2 with a new column r*, we need:
    # sum_{r in C\{r1,r2}} T_r + T_{r*} = T_{r1} + T_{r2} + sum_{r in C\{r1,r2}} T_r
    # => T_{r*} = T_{r1} + T_{r2}
    #
    # This means the tensor defined by r* must be the sum of the tensors defined by r1 and r2.
    # Let T_r be the tensor defined by column r. T_r[i,j,j',k,k',i'] = U[i,j,r] V[j',k,r] W[k',i',r].
    # We need to find U*, V*, W* such that:
    # U*[i,j] V*[j',k] W*[k',i'] = U[i,j,r1] V[j',k,r1] W[k',i',r1] + U[i,j,r2] V[j',k,r2] W[k',i',r2]
    #
    # This is a system of non-linear equations. However, we can try to solve it by fixing
    # a small set of "anchor" points in the tensor space and solving for the remaining coefficients.
    #
    # Alternatively, we can use a randomized approach:
    # 1. Pick two random columns r1, r2.
    # 2. Compute the target tensor T_target = T_{r1} + T_{r2}.
    # 3. Try to find a rank-1 tensor T_{r*} that approximates T_target.
    # 4. If the approximation is exact (or close enough to round to half-integers), we have a merge.
    #
    # A simpler heuristic:
    # Check if T_{r1} + T_{r2} is itself a rank-1 tensor.
    # If yes, we can merge them into a single column.
    #
    # How to check if a tensor is rank-1?
    # A tensor T[i,j,j',k,k',i'] is rank-1 if it can be written as a[i,j] b[j',k] c[k',i'].
    # This is equivalent to the matrix unfolding having rank 1.
    # For example, unfold T into a matrix of size (n*m) x (m*p*p*n) and check if rank is 1.
    # Or simpler: pick a few non-zero entries and check consistency.
    #
    # Let's implement a check:
    # Given T_target (as a 6D array), check if it is rank-1.
    # We can do this by checking if all 2x2 minors of any matrix unfolding are zero.
    # But constructing the full 6D tensor is expensive (14112 elements).
    # However, we only need to check for specific pairs (r1, r2) where the sum might be rank-1.
    #
    # In the schoolbook decomposition, each T_r is a "unit" tensor e_i e_j e_k (in the appropriate indices).
    # T_{r1} + T_{r2} is the sum of two such unit tensors.
    # When is the sum of two unit tensors rank-1?
    # T_{r1} = e_{i1} e_{j1} e_{k1} (where indices are mapped to the 3 modes)
    # T_{r2} = e_{i2} e_{j2} e_{k2}
    # The sum is rank-1 only if they are "compatible" in a way that allows factoring.
    # Generally, e_a + e_b is not rank-1 in any single mode unless a=b.
    # If i1 != i2, then the i-mode has two non-zero entries, so the tensor cannot be factored as a[i] * (rest).
    # Thus, T_{r1} + T_{r2} is rank-1 ONLY IF i1=i2, j1=j2, k1=k2, which means r1=r2.
    # So simple merging of two schoolbook terms is impossible.
    #
    # However, we can merge more than two, or adjust other terms.
    # This is the "Strassen-like" idea: replace a set of terms with a smaller set, adjusting the coefficients.
    #
    # Given the difficulty of finding such a set in 120s, we will try a different structural change:
    # Use a known low-rank decomposition for a sub-problem and extend it.
    # For example, 3x4x7 can be seen as 3x4x4 + 3x4x3.
    # If we have a low-rank decomposition for 3x4x4 and 3x4x3, we can combine them.
    # But we don't have those.
    #
    # Alternative: Use the "Border Rank" concept? No, we need exact rank.
    #
    # Let's try a random search for a valid decomposition of rank R < 84.
    # We will generate random U, V, W with entries in {-1, 0, 1} and check if they decompose T.
    # This is a "lottery ticket" approach. The probability is low, but we can try many times.
    #
    # To make it more likely, we can fix U and V to be "sparse" and solve for W.
    # If we fix U and V, the condition is:
    # sum_r U[i,j,r] V[j',k,r] W[k',i',r] = delta(i,i')delta(j,j')delta(k,k')
    # For fixed U, V, this is a linear system in W.
    # Let M be the matrix where M[(i,j,j',k), (k',i')] = sum_r U[i,j,r] V[j',k,r] W[k',i',r]? No.
    # Let's rearrange:
    # For each (i, j, j', k), we have a vector of values on the LHS for all (k', i').
    # Actually, for fixed i, j, j', k, the equation is:
    # sum_r U[i,j,r] V[j',k,r] W[k',i',r] = delta(i,i')delta(j,j')delta(k,k')
    # This must hold for all (k', i').
    # Let A_{r, (k',i')} = W[k',i',r].
    # Let b_{(i,j,j',k), (k',i')} = delta(i,i')delta(j,j')delta(k,k').
    # Let C_{(i,j,j',k), r} = U[i,j,r] V[j',k,r].
    # Then sum_r C_{(i,j,j',k), r} W_{(k',i'), r} = b_{(i,j,j',k), (k',i')}.
    # This is a system of linear equations in W.
    # The number of unknowns is p*n*R = 21*R.
    # The number of equations is n*m*m*p * p*n = 3*4*4*7 * 21 = 14112.
    # This is an overdetermined system.
    # We can try to solve it using least squares, and check if the residual is zero.
    #
    # Algorithm:
    # 1. For R in [70, 75, 80]:
    #    2. Generate random U, V with entries in {-1, 0, 1}.
    #    3. Construct the linear system C W = b.
    #    4. Solve for W using np.linalg.lstsq.
    #    5. Check if the residual is small (e.g., < 1e-9).
    #    6. If yes, round W to half-integers and check exactness.
    #
    # This is feasible.

    def check_decomposition(U, V, W):
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        
        # Compute tensor
        # einsum: 'ijr,jkr,kir->ijjkk i'
        try:
            tensor = np.einsum('ijr,jkr,kir->ijjkk i', U_r, V_r, W_r)
        except:
            return False
            
        # Check against target
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                if abs(tensor[i, j, jp, k, kp, ip] - expected) > 1e-9:
                                    return False
        return True

    def try_random_decomposition(R, num_trials=100):
        for _ in range(num_trials):
            if time.time() - start_time > deadline:
                return None
            
            # Generate random U, V with entries in {-1, 0, 1}
            U_rand = np.random.choice([-1, 0, 1], size=(n * m, R))
            V_rand = np.random.choice([-1, 0, 1], size=(m * p, R))
            
            # Construct linear system C W = b
            # C shape: (n*m*m*p, R)
            # b shape: (n*m*m*p, p*n)
            
            C = np.zeros((n * m * m * p, R))
            b = np.zeros((n * m * m * p, p * n))
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for jp in range(m):
                        for k in range(p):
                            # C[idx, r] = U[i, j, r] * V[jp, k, r]
                            C[idx, :] = U_rand[i * m + j, :] * V_rand[jp * p + k, :]
                            
                            # b[idx, (kp, ip)] = delta(i, ip) delta(j, jp) delta(k, kp)
                            for kp in range(p):
                                for ip in range(n):
                                    if i == ip and j == jp and k == kp:
                                        b[idx, kp * n + ip] = 1.0
                            idx += 1
            
            # Solve C W = b
            # W shape: (R, p*n) -> transpose to (p*n, R)
            W_sol, residuals, rank, sv = np.linalg.lstsq(C, b, rcond=None)
            
            # Check residual
            if len(residuals) > 0 and residuals[0] > 1e-8:
                continue
            
            # W_sol is (R, p*n). We need W (p*n, R).
            W_candidate = W_sol.T
            
            # Round to half-integers
            W_rounded = np.round(W_candidate * 2) / 2
            
            # Check exactness with rounded W
            if check_decomposition(U_rand, V_rand, W_rounded):
                return U_rand, V_rand, W_rounded
        
        return None

    # Try to find a lower rank decomposition
    for R in range(70, 84):
        if time.time() - start_time > deadline:
            break
        
        result = try_random_decomposition(R, num_trials=50)
        if result is not None:
            U, V, W = result
            return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

    # If no improvement found, return schoolbook
    return get_schoolbook()
