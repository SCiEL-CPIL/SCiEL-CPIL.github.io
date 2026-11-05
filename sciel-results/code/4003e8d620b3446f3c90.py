import numpy as np
import time
from fractions import Fraction

def construct():
    n, m, p = 3, 3, 4
    N, M, P = n * m, m * p, p * n  # 9, 12, 12
    
    # Define the target tensor T[a, b, c] = 1 if (a,b,c) corresponds to (i,j,k) where
    # a = i*m + j, b = j*p + k, c = k*n + i
    T = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0

    # Precompute matricizations for ALS
    T_c = T.reshape(N * M, P)
    T_a = T.reshape(N, M * P).T  # (M*P, N)
    T_b = T.transpose(1, 0, 2).reshape(M, N * P).T # (N*P, M)

    def check_solution(U, V, W, tol=1e-9):
        # Check if U, V, W decompose T
        # T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        # For speed, we check the flattened forms
        # But exactness is required. 
        # We assume U, V, W are already rounded to half-integers.
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        if not np.allclose(T_calc, T, atol=tol):
            return False, U, V, W
        
        # Try to find rational representation with small denominators
        # We know entries are half-integers (denom 2). 
        # But maybe we can find integers? Or mixed?
        # The task says "integer/half-integer". 
        # So we just need to verify they are close to x.0 or x.5.
        
        def is_valid_entries(mat):
            # mat is float array
            # check if 2*mat is close to integer
            two_mat = 2.0 * mat
            int_part = np.round(two_mat)
            return np.allclose(two_mat, int_part, atol=1e-8)
        
        if not is_valid_entries(U):
            # Try rounding to nearest integer? No, task says half-integer.
            # If it's not close to half-integer, it's invalid for the "exact" requirement 
            # unless we allow the float representation itself if it's precise enough?
            # "integer/half-integer entries" implies the *values* must be of that form.
            # So we must round and re-verify.
            pass
            
        # Round to nearest half-integer
        U_r = np.round(U * 2) / 2.0
        V_r = np.round(V * 2) / 2.0
        W_r = np.round(W * 2) / 2.0
        
        # Re-verify with rounded values
        T_calc_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if not np.allclose(T_calc_r, T, atol=1e-8):
            return False, U, V, W
            
        return True, U_r, V_r, W_r

    def try_als(R, max_iter=1000, tol=1e-10):
        # Random initialization
        # Use structured initialization? 
        # Just random is standard.
        np.random.seed(R) # Fixed seed for reproducibility in this run? No, let's vary
        U = np.random.randn(N, R)
        V = np.random.randn(M, R)
        W = np.random.randn(P, R)
        
        # Normalize to avoid large numbers
        for r in range(R):
            norm = np.linalg.norm(U[:, r])
            if norm > 1e-10:
                U[:, r] /= norm
                V[:, r] *= norm
            norm = np.linalg.norm(W[:, r])
            if norm > 1e-10:
                W[:, r] /= norm
                V[:, r] *= norm
            # Re-normalize V
            norm = np.linalg.norm(V[:, r])
            if norm > 1e-10:
                V[:, r] /= norm
                U[:, r] *= norm
                W[:, r] *= norm

        for it in range(max_iter):
            # Update W
            E_W = np.einsum('ar,br->ab r', U, V).reshape(N * M, R)
            try:
                W_T, *_ = np.linalg.lstsq(E_W, T_c.T, rcond=None)
                W = W_T.T
            except:
                return False, None, None, None

            # Update U
            E_U = np.einsum('br,cr->bc r', V, W).reshape(M * P, R)
            try:
                U_T, *_ = np.linalg.lstsq(E_U.T, T_a, rcond=None) # T_a is (M*P, N)
                U = U_T.T
            except:
                return False, None, None, None

            # Update V
            E_V = np.einsum('ar,cr->ac r', U, W).reshape(N * P, R)
            try:
                V_T, *_ = np.linalg.lstsq(E_V.T, T_b, rcond=None) # T_b is (N*P, M)
                V = V_T.T
            except:
                return False, None, None, None

            # Check error
            # Efficient error check:
            # T_calc_c = E_W @ W.T
            # T_calc_a = E_U.T @ U.T ... wait, dimensions.
            # Let's just compute full error every 10 iterations to save time
            if it % 10 == 0 or it == max_iter - 1:
                T_calc = np.einsum('ar,br,cr->abc', U, V, W)
                error = np.linalg.norm(T_calc - T)
                if error < tol:
                    return True, U, V, W
        return False, U, V, W

    def try_search_rank(R, attempts=20):
        start_time = time.time()
        for attempt in range(attempts):
            if time.time() - start_time > 5.0: # 5s per rank
                break
            # Different initialization for each attempt
            np.random.seed(42 + R * 100 + attempt)
            
            # Sometimes random is enough, sometimes we need to be smarter.
            # Let's stick to random for simplicity but increase attempts.
            success, U, V, W = try_als(R, max_iter=200, tol=1e-9)
            if success:
                valid, U_r, V_r, W_r = check_solution(U, V, W)
                if valid:
                    return True, U_r, V_r, W_r
        return False, None, None, None

    best_rank = N * M * P
    best_U = None
    best_V = None
    best_W = None

    # Try ranks from 23 upwards.
    # 3x3x3 is 23. 3x3x4 is likely 24 or 25.
    # Let's try 23, 24, 25, 26, 27, 28, 29, 30.
    
    # We will prioritize finding a valid solution.
    # If we find rank 24, that's great.
    # If we find rank 30, that's better than 36.
    
    # To ensure we return *something* good, we start with a known valid high rank (schoolbook)
    # and try to improve.
    
    # Schoolbook rank 36
    U_sb = np.zeros((N, N*M*P))
    V_sb = np.zeros((M, N*M*P))
    W_sb = np.zeros((P, N*M*P))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, idx] = 1.0
                V_sb[j*p+k, idx] = 1.0
                W_sb[k*n+i, idx] = 1.0
                idx += 1
    
    best_rank = N*M*P
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    # Now try to find lower ranks
    # We use a deadline of 100 seconds to leave buffer for the process to finish.
    total_start = time.time()
    deadline = 110.0
    
    # Try ranks 23 to 35
    for R in range(23, 36):
        if time.time() - total_start > deadline:
            break
        
        # Adjust attempts based on time remaining
        time_left = deadline - (time.time() - total_start)
        attempts = max(1, int(time_left / 2.0)) # Roughly 2s per attempt?
        
        # For lower ranks, we might need more iterations but fewer attempts?
        # Or more attempts?
        # Let's try 5 attempts for R=23,24, 10 for 25, etc.
        # Actually, let's just do a fixed number of attempts that fits in ~10s.
        
        # Strategy: 
        # R=23: 5 attempts, 500 iters
        # R=24: 10 attempts, 500 iters
        # R=25: 5 attempts, 300 iters
        # ...
        
        if R <= 24:
            att = 10
            iters = 500
        elif R <= 27:
            att = 20
            iters = 200
        else:
            att = 30
            iters = 100

        found = False
        for a in range(att):
            if time.time() - total_start > deadline:
                break
            np.random.seed(42 + R * 100 + a)
            # Custom ALS with early termination
            U = np.random.randn(N, R)
            V = np.random.randn(M, R)
            W = np.random.randn(P, R)
            
            # Normalize
            for r in range(R):
                nU = np.linalg.norm(U[:, r])
                nV = np.linalg.norm(V[:, r])
                nW = np.linalg.norm(W[:, r])
                if nU > 1e-10 and nV > 1e-10 and nW > 1e-10:
                    U[:, r] /= nU
                    V[:, r] /= nV
                    W[:, r] /= nW
                    # Rescale V to keep product same?
                    # T = U V W. If we divide U by nU, multiply V by nU.
                    # Let's do: U/=nU, V*=nU. Then V*=nV, W/=nV.
                    # Simpler:
                    # U = U / nU
                    # V = V * nU
                    # V = V / nV
                    # W = W * nV
                    # W = W / nW
                    # U = U * nW
                    pass # Skip complex normalization, standard ALS handles scaling via lstsq mostly.
                    
            success = False
            for it in range(iters):
                # Update W
                E_W = np.einsum('ar,br->ab r', U, V).reshape(N * M, R)
                try:
                    W_T, *_ = np.linalg.lstsq(E_W, T_c.T, rcond=None)
                    W = W_T.T
                except:
                    break

                # Update U
                E_U = np.einsum('br,cr->bc r', V, W).reshape(M * P, R)
                try:
                    U_T, *_ = np.linalg.lstsq(E_U.T, T_a, rcond=None)
                    U = U_T.T
                except:
                    break

                # Update V
                E_V = np.einsum('ar,cr->ac r', U, W).reshape(N * P, R)
                try:
                    V_T, *_ = np.linalg.lstsq(E_V.T, T_b, rcond=None)
                    V = V_T.T
                except:
                    break

                # Check convergence
                if it % 50 == 0:
                    T_calc = np.einsum('ar,br,cr->abc', U, V, W)
                    error = np.linalg.norm(T_calc - T)
                    if error < 1e-8:
                        success = True
                        break
            
            if success:
                valid, U_r, V_r, W_r = check_solution(U, V, W)
                if valid:
                    best_rank = R
                    best_U, best_V, best_W = U_r, V_r, W_r
                    found = True
                    break # Found valid for this rank, move to next rank
        
        # If we found a valid solution for this rank, we could break if we want minimal rank.
        # But maybe a lower rank was missed? No, we iterate 23, 24, 25...
        # So if we found 23, we can return immediately?
        # Yes, 23 is the theoretical lower bound for 3x3x3, so 3x3x4 >= 23.
        # If we find 23, it's optimal.
        if found and R == 23:
            break

    # Return the best found
    # Convert to lists of floats (or ints if integer)
    # The task says "integer/half-integer entries".
    # We rounded to half-integers in check_solution.
    
    def to_list(mat):
        # Check if it's effectively integer
        mat_int = np.round(mat)
        if np.allclose(mat, mat_int, atol=1e-6):
            return mat_int.astype(int).tolist()
        else:
            # Keep as float, representing half-integers
            return mat.tolist()

    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
