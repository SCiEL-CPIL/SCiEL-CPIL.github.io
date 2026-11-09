"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]
DEADLINE = 120  # seconds, safety margin from 150s limit

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm = n * m  # 20
    mp = m * p  # 25
    pn = p * n  # 20
    
    # Start with schoolbook decomposition (rank = 100)
    schoolbook_rank = nm * mp
    U_school = np.zeros((nm, schoolbook_rank), dtype=int)
    V_school = np.zeros((mp, schoolbook_rank), dtype=int)
    W_school = np.zeros((pn, schoolbook_rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Use rank-revealing decomposition to find a lower-rank approximation
    # Build the tensor matrix: reshape U, V, W into a matrix of size (nm*mp, pn)
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # Actually, let's build a matrix M of size (nm*mp, pn) where
    # M[a, c] = sum over b of T[a,b,c] * V[b,:] ... 
    # Better approach: use the fact that the tensor can be represented as a matrix
    # of size (nm*mp, pn) via the "flattening" trick.
    
    # Construct the tensor as a 3D array
    T = np.zeros((nm, mp, pn), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Try different rank values and use SVD to find best low-rank approximation
    # The tensor can be seen as a matrix of size (nm*mp, pn) by flattening first two dims
    T_mat = T.reshape(nm * mp, pn)
    
    # Get singular values
    try:
        U_svd, s_vals, Vt_svd = np.linalg.svd(T_mat, full_matrices=False)
    except:
        # Fall back to schoolbook
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Find the smallest rank where the approximation is exact (or very close)
    # Since we need exact integer decomposition, we'll try to find a low-rank
    # integer solution
    
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = schoolbook_rank
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = 25, but realistically higher
    # For 4x5x5, known lower bound is around 48-50, but let's try from 40
    min_rank_to_try = 40
    max_rank_to_try = min(schoolbook_rank - 1, 70)  # Don't try too close to schoolbook
    
    for target_rank in range(min_rank_to_try, max_rank_to_try + 1):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Use SVD to get initial low-rank approximation
        U_approx = U_svd[:, :target_rank] * s_vals[:target_rank]
        Vt_approx = Vt_svd[:target_rank, :]
        
        # Now we need to convert this back to U, V, W format
        # This is non-trivial. Let's use a different approach.
        
        # Alternative: use iterative optimization with integer constraints
        # Initialize U, V, W with random integers
        np.random.seed(42 + target_rank)
        U_rand = np.random.randint(-3, 4, size=(nm, target_rank))
        V_rand = np.random.randint(-3, 4, size=(mp, target_rank))
        W_rand = np.random.randint(-3, 4, size=(pn, target_rank))
        
        # Iteratively refine to minimize error
        success = False
        for iteration in range(200):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Compute error for all (i,j,k)
            max_err = 0
            worst_i, worst_j, worst_k = 0, 0, 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_rand[i*m+j], V_rand[j*p+k] * W_rand[k*n+i])
                        err = abs(val - 1)
                        if err > max_err:
                            max_err = err
                            worst_i, worst_j, worst_k = i, j, k
            
            if max_err < 1e-10:
                success = True
                break
            
            # Adjust one entry to reduce error
            i, j, k = worst_i, worst_j, worst_k
            r = np.random.randint(0, target_rank)
            
            # Try adjusting U
            denom = V_rand[j*p+k, r] * W_rand[k*n+i, r]
            if denom != 0:
                current_val = np.dot(U_rand[i*m+j], V_rand[j*p+k] * W_rand[k*n+i])
                needed = 1 - current_val
                delta = needed / denom
                U_rand[i*m+j, r] += int(round(delta))
            
            # Occasionally adjust V or W
            if np.random.random() < 0.3:
                r2 = np.random.randint(0, target_rank)
                current_val = np.dot(U_rand[i*m+j], V_rand[j*p+k] * W_rand[k*n+i])
                needed = 1 - current_val
                denom2 = U_rand[i*m+j, r2] * W_rand[k*n+i, r2]
                if denom2 != 0:
                    delta2 = needed / denom2
                    V_rand[j*p+k, r2] += int(round(delta2))
            elif np.random.random() < 0.3:
                r3 = np.random.randint(0, target_rank)
                current_val = np.dot(U_rand[i*m+j], V_rand[j*p+k] * W_rand[k*n+i])
                needed = 1 - current_val
                denom3 = U_rand[i*m+j, r3] * V_rand[j*p+k, r3]
                if denom3 != 0:
                    delta3 = needed / denom3
                    W_rand[k*n+i, r3] += int(round(delta3))
        
        if success:
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_rand[i*m+j], V_rand[j*p+k] * W_rand[k*n+i])
                        if abs(val - 1) > 1e-10:
                            valid = False
                            break
                if not valid:
                    break
            if valid:
                best_rank = target_rank
                best_result = {
                    "U": U_rand.tolist(),
                    "V": V_rand.tolist(),
                    "W": W_rand.tolist()
                }
                break
    
    return best_result
# EVOLVE-BLOCK-END
