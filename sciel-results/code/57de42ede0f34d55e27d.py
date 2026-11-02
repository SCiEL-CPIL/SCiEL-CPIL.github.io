import time
import numpy as np
import random

DIMENSIONS = [2, 3, 8]

def construct():
    """
    Constructs a rank-minimal matrix multiplication decomposition for 2x3 * 3x8.
    Returns a dictionary with matrices U, V, W of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n, m, p = DIMENSIONS
    full_rank = n * m * p  # 48
    
    # Initialize best result as schoolbook
    best_rank = full_rank
    U_best = [[0] * full_rank for _ in range(n * m)]
    V_best = [[0] * full_rank for _ in range(m * p)]
    W_best = [[0] * full_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_best[i * m + j][col] = 1
                V_best[j * p + k][col] = 1
                W_best[k * n + i][col] = 1
                col += 1
                
    best_result = {"U": U_best, "V": V_best, "W": W_best}
    
    if time.time() - start_time > deadline - 5:
        return best_result
    
    # Verify a decomposition is valid
    def verify(U, V, W):
        """Check that sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] == 1 for all i,j,k."""
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for rr in range(r):
                        total += U[i * m + j][rr] * V[j * p + k][rr] * W[k * n + i][rr]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    # Check if all entries are integer or half-integer
    def is_nice(U, V, W):
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        return True
    
    # Strategy: Try to find rank-47 decomposition by exploiting linear dependencies
    # in the tensor. The matrix multiplication tensor for 2x3x8 has 48 non-zero entries.
    # If there's a linear dependency among the 48 rank-1 tensors, we can reduce rank to 47.
    
    # For a general 2x3x8, the rank is known to be 48 (no reduction possible via simple means),
    # but let's try a structured search for rank < 48.
    
    # Approach: Use the fact that C[i,k] = sum_j A[i,j] B[j,k]
    # We can try to express this with fewer than 48 multiplications by
    # combining terms. For example, for fixed i, sum_j A[i,j] B[j,k] is a dot product.
    # If we can precompute combinations of A entries, we might reduce the count.
    
    # Let's try a specific structured approach:
    # For n=2, m=3, p=8, try rank 47 by finding if one schoolbook term is redundant.
    
    # Try random search with verification, but use a smarter initialization
    def try_rank_reduction(target_rank):
        if time.time() - start_time > deadline - 5:
            return None
        
        # Try to construct a decomposition of the given rank
        # Use a greedy approach: start with schoolbook and try to merge terms
        
        # For rank 47, we need to find a linear combination of 47 rank-1 tensors
        # that equals the full tensor.
        
        # Alternative: Use numerical linear algebra to find a low-rank decomposition
        # Build the tensor and try to find its tensor rank
        
        # Construct the full tensor
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        
        # Reshape to matrix for SVD-based approximation
        # But tensor rank is NP-hard, so we need a smarter approach
        
        # Try a specific structured family: assume W has a linear structure in k
        # For each (i,j), the contribution to output k is A[i,j]*B[j,k]
        # If we assume B[j,k] follows a linear recurrence, we can reduce rank
        
        # For p=8, assume B[j,k] = c1[j]*x1^k + c2[j]*x2^k + ... 
        # This is getting complex. Let's try a direct search for small rank reductions.
        
        # Try rank 47: modify the schoolbook by combining two terms
        # Term (i,j,k1) and (i,j,k2) can be combined if B[j,k1] and B[j,k2] are linearly related
        
        # For a concrete attempt: try to find if the tensor has rank 47
        # by checking if the 48x48 matrix of coefficients is singular
        
        # The tensor can be viewed as a 48-dimensional space of rank-1 tensors.
        # If they are linearly dependent, rank < 48.
        
        # Construct the matrix where each column is a flattened rank-1 tensor
        cols = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec = np.zeros((n * m) * (m * p) * (p * n))
                    vec[(i * m + j) * (m * p) * (p * n) + (j * p + k) * (p * n) + (k * n + i)] = 1.0
                    cols.append(vec)
        
        M = np.array(cols).T  # shape (48, 48*48*48) - too large
        
        # This approach is infeasible. Let's use a different strategy.
        
        # Use the verification function with a structured search
        # Try small modifications to the schoolbook decomposition
        
        # For rank 47, try to find a linear dependency
        # Assume we can write the tensor as a sum of 47 rank-1 tensors
        # by finding coefficients
        
        # Given the complexity, let's try a randomized approach with verification
        max_trials = 200
        for trial in range(max_trials):
            if time.time() - start_time > deadline - 5:
                return None
            
            # Generate a candidate by perturbing the schoolbook
            U_cand = [row[:] for row in U_best]
            V_cand = [row[:] for row in V_best]
            W_cand = [row[:] for row in W_best]
            
            # Try to merge two columns by finding a linear combination
            # Pick two random columns and try to combine them
            c1 = random.randint(0, full_rank - 1)
            c2 = random.randint(0, full_rank - 1)
            if c1 == c2:
                continue
            
            # Try to express column c2 as a linear combination of column c1 and others
            # This is complex. Instead, try a direct numerical approach.
            
            # Use least squares to find a rank-47 approximation
            # Fix U and V, solve for W
            U_fix = np.array(U_best[:, :target_rank])  # Use first target_rank columns
            V_fix = np.array(V_best[:, :target_rank])
            
            # Build the linear system for W
            # T[u,v,w] = sum_a U[u,a] V[v,a] W[w,a]
            # For fixed U, V, this is linear in W
            
            B_mat = np.zeros((n * m * m * p, target_rank))
            for a in range(target_rank):
                M_a = np.outer(U_fix[:, a], V_fix[:, a])
                B_mat[:, a] = M_a.flatten()
            
            T_flat = T.reshape(n * m * m * p, p * n)
            
            try:
                W_trans, residuals, rank_b, s = np.linalg.lstsq(B_mat, T_flat, rcond=None)
                residual = np.linalg.norm(B_mat @ W_trans - T_flat)
                
                if residual < 1e-8:
                    W_sol = W_trans.T
                    # Check if entries are nice
                    if is_nice(U_fix.tolist(), V_fix.tolist(), W_sol.tolist()):
                        return {"U": U_fix.tolist(), "V": V_fix.tolist(), "W": W_sol.tolist()}
            except np.linalg.LinAlgError:
                pass
        
        return None
    
    # Try rank 47
    res_47 = try_rank_reduction(47)
    if res_47 is not None and verify(res_47["U"], res_47["V"], res_47["W"]):
        return res_47
    
    # Try rank 46 if time permits
    if time.time() - start_time < deadline - 15:
        res_46 = try_rank_reduction(46)
        if res_46 is not None and verify(res_46["U"], res_46["V"], res_46["W"]):
            return res_46
    
    # Try a structured Karatsuba-like approach for 2x3x8
    # Split the 8 columns into two groups of 4
    # For each group, use a 2x3x4 decomposition
    # If we can find a rank-r decomposition for 2x3x4, total rank might be 2r or less
    
    # For 2x3x4, schoolbook rank is 24. Try to find a lower rank.
    # Known: rank(2x2x2) = 7 (Strassen), rank(2x3x3) is known to be <= 20
    
    # For 2x3x4, let's try to construct a rank-20 decomposition
    # and then combine two such decompositions for 2x3x8
    
    if time.time() - start_time < deadline - 20:
        # Try to find a 2x3x4 decomposition of rank < 24
        # This is a smaller problem, so we can search more exhaustively
        
        def find_2x3x4_decomposition():
            n2, m2, p2 = 2, 3, 4
            rank2 = n2 * m2 * p2  # 24
            
            # Schoolbook
            U2 = [[0] * rank2 for _ in range(n2 * m2)]
            V2 = [[0] * rank2 for _ in range(m2 * p2)]
            W2 = [[0] * rank2 for _ in range(p2 * n2)]
            col = 0
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        U2[i * m2 + j][col] = 1
                        V2[j * p2 + k][col] = 1
                        W2[k * n2 + i][col] = 1
                        col += 1
            
            # Try to find a lower rank
            # For 2x3x4, the minimum rank is at least max(6, 12, 8) = 12
            # Let's try ranks from 12 to 23
            
            for target_r in range(12, rank2):
                if time.time() - start_time > deadline - 20:
                    break
                
                # Try random search for rank target_r
                for trial in range(100):
                    if time.time() - start_time > deadline - 20:
                        break
                    
                    U_rand = np.zeros((n2 * m2, target_r))
                    V_rand = np.zeros((m2 * p2, target_r))
                    
                    for a in range(target_r):
                        for idx in np.random.choice(n2 * m2, size=max(1, n2 * m2 // 3), replace=False):
                            U_rand[idx, a] = np.random.choice([-1, 1])
                        for idx in np.random.choice(m2 * p2, size=max(1, m2 * p2 // 3), replace=False):
                            V_rand[idx, a] = np.random.choice([-1, 1])
                    
                    # Solve for W
                    B_mat = np.zeros((n2 * m2 * m2 * p2, target_r))
                    for a in range(target_r):
                        M_a = np.outer(U_rand[:, a], V_rand[:, a])
                        B_mat[:, a] = M_a.flatten()
                    
                    T2 = np.zeros((n2 * m2, m2 * p2, p2 * n2))
                    for i in range(n2):
                        for j in range(m2):
                            for k in range(p2):
                                T2[i * m2 + j, j * p2 + k, k * n2 + i] = 1.0
                    
                    T2_flat = T2.reshape(n2 * m2 * m2 * p2, p2 * n2)
                    
                    try:
                        W_trans, residuals, rank_b, s = np.linalg.lstsq(B_mat, T2_flat, rcond=None)
                        residual = np.linalg.norm(B_mat @ W_trans - T2_flat)
                        
                        if residual < 1e-8:
                            W_sol = W_trans.T
                            # Check if nice
                            nice = True
                            for mat in [U_rand.tolist(), V_rand.tolist(), W_sol.tolist()]:
                                for row in mat:
                                    for val in row:
                                        if abs(val * 2 - round(val * 2)) > 1e-9:
                                            nice = False
                                            break
                                    if not nice:
                                        break
                                if not nice:
                                    break
                            
                            if nice:
                                # Verify
                                U_l = U_rand.tolist()
                                V_l = V_rand.tolist()
                                W_l = W_sol.tolist()
                                valid = True
                                for i in range(n2):
                                    for j in range(m2):
                                        for k in range(p2):
                                            total = 0.0
                                            for rr in range(target_r):
                                                total += U_l[i * m2 + j][rr] * V_l[j * p2 + k][rr] * W_l[k * n2 + i][rr]
                                            if abs(total - 1.0) > 1e-9:
                                                valid = False
                                                break
                                        if not valid:
                                            break
                                    if not valid:
                                        break
                                if valid:
                                    return U_l, V_l, W_l, target_r
                    except np.linalg.LinAlgError:
                        pass
            
            return U2, V2, W2, rank2
        
        U2, V2, W2, r2 = find_2x3x4_decomposition()
        
        # If we found a lower-rank 2x3x4, use it for 2x3x8
        # by duplicating for the two halves of the 8 columns
        if r2 < 24:
            # Create 2x3x8 decomposition by combining two 2x3x4 decompositions
            U8 = [[0] * (2 * r2) for _ in range(n * m)]
            V8 = [[0] * (2 * r2) for _ in range(m * p)]
            W8 = [[0] * (2 * r2) for _ in range(p * n)]
            
            # First half (k=0..3)
            for i in range(n):
                for j in range(m):
                    for r in range(r2):
                        U8[i * m + j][r] = U2[i * m2 + j][r]
            for j in range(m):
                for k in range(4):
                    for r in range(r2):
                        V8[j * p + k][r] = V2[j * m2 + k][r]
            for k in range(4):
                for i in range(n):
                    for r in range(r2):
                        W8[k * n + i][r] = W2[k * n2 + i][r]
            
            # Second half (k=4..7)
            for i in range(n):
                for j in range(m):
                    for r in range(r2):
                        U8[i * m + j][r2 + r] = U2[i * m2 + j][r]
            for j in range(m):
                for k in range(4, 8):
                    for r in range(r2):
                        V8[j * p + k][r2 + r] = V2[j * m2 + (k - 4)][r]
            for k in range(4, 8):
                for i in range(n):
                    for r in range(r2):
                        W8[k * n + i][r2 + r] = W2[(k - 4) * n2 + i][r]
            
            # Verify
            if verify(U8, V8, W8):
                return {"U": U8, "V": V8, "W": W8}
    
    return best_result
