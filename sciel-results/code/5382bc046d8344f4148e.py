import numpy as np
import time
import random

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Build the tensor T as a list of slices T_c (c is output index)
    # T_c is a matrix of size (n*m) x (m*p)
    # T_c[a, b] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Precompute vectorized slices of T
    # vec(T_c) is a vector of length nm * mp
    # We need to check if vec(T_c) is in the column space of S
    # S has columns vec(u_l v_l^T)
    
    slices = []
    for c in range(pn):
        # c corresponds to k*n + i
        k = c // n
        i = c % n
        T_c = np.zeros((nm, mp))
        for j in range(m):
            a = i * m + j
            b = j * p + k
            T_c[a, b] = 1
        slices.append(T_c.flatten())
    
    slices_np = np.array(slices) # shape (pn, nm*mp)
    
    # Try to find a decomposition with rank < rank_schoolbook
    # Start from rank_schoolbook - 1 down to some lower bound
    # Lower bound: nm + mp + pn - m^2 = 6 + 21 + 14 - 9 = 32
    # Let's try a few ranks
    target_ranks = [rank_schoolbook - 1, rank_schoolbook - 2]
    
    start_time = time.time()
    time_limit = 100.0 # Leave 50s for safety and fallback
    
    best_rank = rank_schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    for r in target_ranks:
        if time.time() - start_time > time_limit:
            break
            
        # Random search for rank r
        # We use random integer matrices U and V
        # To avoid degenerate cases, we use small integers
        max_attempts = 500
        attempt = 0
        while attempt < max_attempts and (time.time() - start_time) < time_limit:
            # Generate random U and V
            # Using small integers to keep numbers manageable
            U_rand = np.random.randint(-10, 11, size=(nm, r))
            V_rand = np.random.randint(-10, 11, size=(mp, r))
            
            # Construct S: columns are vec(u_l v_l^T)
            # u_l is U_rand[:, l], v_l is V_rand[:, l]
            # vec(u v^T) = kron(u, v) ? No, vec(u v^T) = kron(v, u) if using column-major?
            # Numpy flatten is row-major by default.
            # (u v^T)_{a,b} = u_a v_b.
            # Flatten index idx = a * mp + b.
            # We can compute S directly.
            # S[:, l] = vec(U_rand[:, l] * V_rand[:, l].T)
            
            S = np.zeros((nm * mp, r))
            for l in range(r):
                outer = np.outer(U_rand[:, l], V_rand[:, l])
                S[:, l] = outer.flatten()
            
            # Check if all slices are in Col(S)
            # We can check this by projecting slices onto Col(S) and checking residual
            # Use QR decomposition of S
            try:
                Q, R = np.linalg.qr(S)
                # Q has orthonormal columns spanning Col(S)
                # Check if slices_np @ Q == slices_np @ Q @ Q.T (projection)
                # Or simply check if norm(slices_np - slices_np @ Q @ Q.T) is small
                proj = slices_np @ Q @ Q.T
                residuals = slices_np - proj
                if np.max(np.abs(residuals)) < 1e-8:
                    # Found a valid U, V
                    # Now solve for W
                    # slices_np = S @ W.T ?
                    # vec(T_c) = sum_l W_{c,l} S[:, l]
                    # So slices_np (pn x nm*mp) = W.T (pn x r) @ S (r x nm*mp) ?
                    # No, vec(T_c) = S @ w_c where w_c is row c of W.T?
                    # Let W be (pn, r).
                    # (S @ W.T)_{c, idx} = sum_l S_{idx, l} W.T_{l, c} = sum_l S_{idx, l} W_{c, l}
                    # We want vec(T_c)[idx] = sum_l W_{c, l} S_{idx, l}
                    # So slices_np = W @ S.T ?
                    # (W @ S.T)_{c, idx} = sum_l W_{c, l} S.T_{l, idx} = sum_l W_{c, l} S_{idx, l}
                    # Yes.
                    # So W = slices_np @ S.T @ (S @ S.T)^-1 ?
                    # Since S has orthonormal columns Q, W = slices_np @ Q.T
                    W_sol = slices_np @ Q.T
                    # Verify
                    # Check if W_sol has integer/half-integer entries?
                    # The problem allows integer/half-integer.
                    # Random integers might produce rational W.
                    # We need to check if W_sol is close to integer/half-integer.
                    # If not, we might need to scale or adjust.
                    # However, if U, V are integers, W is rational.
                    # We can return the rational numbers as floats if they are close to halves/integers.
                    
                    # Check if entries are close to k/2
                    W_check = W_sol * 2
                    if np.allclose(W_check, np.round(W_check), atol=1e-6):
                        # Valid solution found
                        best_rank = r
                        best_U = U_rand
                        best_V = V_rand
                        best_W = W_sol
                        break
            except np.linalg.LinAlgError:
                continue
            
            attempt += 1
            
    if best_U is not None:
        # Convert to lists
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        # Fallback to schoolbook
        U = [[0] * rank_schoolbook for _ in range(nm)]
        V = [[0] * rank_schoolbook for _ in range(mp)]
        W = [[0] * rank_schoolbook for _ in range(pn)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
