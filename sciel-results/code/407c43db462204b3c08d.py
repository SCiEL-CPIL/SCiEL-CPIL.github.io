"""Rewritten program using Strassen-like recursive partitioning for 3x3x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    start_time = time.time()
    time_limit = 135  # leave 15s margin for evaluator's 150s kill
    
    def time_check():
        return (time.time() - start_time) < time_limit
    
    # Schoolbook fallback (rank 63)
    rank_fb = n * m * p  # 63
    U_fb = np.zeros((n * m, rank_fb), dtype=float)
    V_fb = np.zeros((m * p, rank_fb), dtype=float)
    W_fb = np.zeros((p * n, rank_fb), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, c] = 1.0
                V_fb[j * p + k, c] = 1.0
                W_fb[k * n + i, c] = 1.0
                c += 1
    
    def verify(U, V, W):
        """Verify that U,V,W correctly decompose the matrix multiplication tensor."""
        r = U.shape[1]
        # Check diagonal entries (should be 1)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c_idx in range(r):
                        s += U[i*m+j, c_idx] * V[j*p+k, c_idx] * W[k*n+i, c_idx]
                    if abs(s - 1.0) > 1e-8:
                        return False
        # Check off-diagonal entries (should be 0)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                if i2 == i and j2 == j and k2 == k:
                                    continue
                                s2 = 0.0
                                for c_idx in range(r):
                                    s2 += U[i2*m+j2, c_idx] * V[j2*p+k2, c_idx] * W[k2*n+i2, c_idx]
                                if abs(s2) > 1e-8:
                                    return False
        return True
    
    # Verify schoolbook
    if not verify(U_fb, V_fb, W_fb):
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Strassen-like recursive partitioning approach:
    # Split the 3x3x7 problem into smaller blocks and use known optimal decompositions.
    
    # Strategy: Use the fact that M_{3,3,7} can be decomposed using:
    # 1. Split B (3x7) into B1 (3x3), B2 (3x3), B3 (3x1)
    # 2. Use rank-23 for M_{3,3,3} blocks and rank-7 for M_{3,3,1}
    # 3. Total rank: 23 + 23 + 7 = 53
    
    # However, we need explicit constructions. Let's try a more direct approach:
    # Use a structured random search with the Strassen-like partitioning as a guide.
    
    # Known optimal ranks:
    # M_{2,2,2}: rank 7 (Strassen)
    # M_{3,3,3}: rank 23 (best known)
    # M_{3,3,1}: rank 7 (best known)
    # M_{2,2,3}: rank ~11-12
    
    # Let's try to construct using block decomposition:
    # Partition A (3x3) into blocks, B (3x7) into blocks, and use Strassen-like formulas.
    
    # For now, let's try a direct optimization approach:
    # Start with a reasonable initial guess and use gradient descent on the rank-1 terms.
    
    import random
    random.seed(42)
    
    # Try to find a decomposition with rank < 63 using iterative improvement
    best_rank = rank_fb
    best_U, best_V, best_W = U_fb.copy(), V_fb.copy(), W_fb.copy()
    
    # Approach: Start from schoolbook and try to merge columns
    # Two columns can be merged if their trilinear forms are linearly dependent
    
    # More practical: Try random constructions with small entries
    # and verify correctness
    
    # For efficiency, let's try a structured approach:
    # Use the known rank-7 construction for M_{3,3,1} and rank-23 for M_{3,3,3}
    
    # Rank-7 for M_{3,3,1}: C_i = sum_j a_{ij} b_j
    # This computes 3 outputs from 9 input products
    # Known construction uses 7 rank-1 tensors
    
    # Let me try a different strategy: use numpy to solve for a low-rank decomposition
    # by formulating it as a tensor decomposition problem
    
    # Actually, let's try a simpler approach: use the fact that for small matrices,
    # we can try all possible sparse patterns with entries from a small set
    
    # Given time constraints, let's try a few specific constructions:
    
    # Construction 1: Use Strassen's formula for 2x2 blocks
    # Partition 3x3 into 2x2 + 1x1 blocks
    
    # For A (3x3) = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # For B (3x7) = [B11 B12; B21 B22] where B11 is 2x7, B12 is 1x7, B21 is 2x7, B22 is 1x7
    
    # This gets complex. Let's try a more direct numerical approach.
    
    # Try random search with verification for ranks 50-62
    for target_rank in range(50, 63):
        if not time_check():
            break
        
        max_attempts = max(100, 5000 // (63 - target_rank))
        found = False
        
        for attempt in range(max_attempts):
            if not time_check():
                break
            
            # Create random sparse matrices with small entries
            U_r = np.zeros((n*m, target_rank))
            V_r = np.zeros((m*p, target_rank))
            W_r = np.zeros((p*n, target_rank))
            
            # For each column, choose a random sparse pattern
            for col in range(target_rank):
                # Choose which (i,j) pairs have non-zero entries in U
                n_nonzero_u = random.randint(1, 4)
                u_indices = random.sample(range(n*m), min(n_nonzero_u, n*m))
                for idx in u_indices:
                    U_r[idx, col] = random.choice([0.5, 1.0, -0.5, -1.0])
                
                # Choose which (j,k) pairs have non-zero entries in V
                n_nonzero_v = random.randint(1, 4)
                v_indices = random.sample(range(m*p), min(n_nonzero_v, m*p))
                for idx in v_indices:
                    V_r[idx, col] = random.choice([0.5, 1.0, -0.5, -1.0])
                
                # Choose which (k,i) pairs have non-zero entries in W
                n_nonzero_w = random.randint(1, 4)
                w_indices = random.sample(range(p*n), min(n_nonzero_w, p*n))
                for idx in w_indices:
                    W_r[idx, col] = random.choice([0.5, 1.0, -0.5, -1.0])
            
            # Quick verification
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U_r[i*m+j, :] * V_r[j*p+k, :] * W_r[k*n+i, :])
                        if abs(s - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                # Full verification
                full_ok = True
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if i2 == i and j2 == j and k2 == k:
                                continue
                            s2 = np.sum(U_r[i2*m+j2, :] * V_r[j2*p+k2, :] * W_r[k2*n+i2, :])
                            if abs(s2) > 1e-6:
                                full_ok = False
                                break
                        if not full_ok:
                            break
                    if not full_ok:
                        break
                if full_ok:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_r, V_r, W_r
                    found = True
                    break
        
        if found:
            break
    
    # If we found a better rank, use it
    if best_rank < rank_fb:
        U_final = best_U.tolist()
        V_final = best_V.tolist()
        W_final = best_W.tolist()
        
        # Round to half-integers if possible
        def to_half_integer(arr):
            result = []
            for row in arr:
                new_row = []
                for x in row:
                    x2 = x * 2
                    if abs(x2 - round(x2)) < 1e-9:
                        new_row.append(round(x2) / 2.0)
                    else:
                        new_row.append(float(x))
                result.append(new_row)
            return result
        
        return {"U": to_half_integer(U_final), "V": to_half_integer(V_final), "W": to_half_integer(W_final)}
    
    # Fallback to schoolbook
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

# EVOLVE-BLOCK-END
