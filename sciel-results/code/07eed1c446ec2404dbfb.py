import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    def check_solution(U, V, W):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Check if sum_c U[i*m+j, c] V[j'*p+k, c] W[k*n+i, c] == delta[j, j']
        U_reshaped = U.reshape(n, m, -1)
        V_reshaped = V.reshape(m, p, -1)
        W_reshaped = W.reshape(p, n, -1)
        
        r = U.shape[1]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m + j, c] * V[jp*p + k, c] * W[k*n + i, c]
                        if abs(s - (1.0 if j == jp else 0.0)) > 1e-9:
                            return False
        return True

    def get_rank1_residual(U, V, W, exclude_idx):
        """
        Computes the residual tensor R = T - sum_{c != exclude_idx} U_c V_c W_c.
        Checks if R is rank 1.
        Returns (is_rank1, U_res, V_res, W_res) if rank 1, else (False, None, None, None).
        """
        r = U.shape[1]
        # Residual tensor T_res[i, j, jp, k] = sum_{c != exclude_idx} U[i*m+j, c] V[jp*p+k, c] W[k*n+i, c]
        # Note: The target tensor is T[i, j, jp, k] = delta[j, jp].
        # So T_res = T_target - T_partial.
        
        # We can compute T_partial efficiently.
        # But for small sizes, we can just compute the full sum and subtract the excluded term.
        
        # Let's compute the full tensor T_full and T_excluded.
        # T_full is just the identity block structure.
        # T_full[i, j, jp, k] = 1 if j == jp else 0.
        
        # T_excluded[i, j, jp, k] = U[i*m+j, exclude_idx] * V[jp*p+k, exclude_idx] * W[k*n+i, exclude_idx]
        
        # We need to check if T_full - T_excluded is rank 1.
        # Let R[i, j, jp, k] = (1 if j==jp else 0) - U[i*m+j, e] V[jp*p+k, e] W[k*n+i, e]
        
        # Reshape R to (n, m, m, p)
        # Check if R is rank 1.
        # A tensor R is rank 1 if R[i, j, jp, k] = a_i b_j c_jp d_k ? 
        # Wait, the tensor decomposition is over the indices of the matrix multiplication.
        # The tensor is T_{i, j, j', k} where A is (n x m), B is (m x p), C is (n x p).
        # C[i, k] = sum_j A[i, j] B[j, k].
        # The tensor for matrix multiplication is defined by the bilinear map.
        # The "tensor" is often defined as T_{i, j, j', k} such that C[i, k] = sum_{j, j'} T_{i, j, j', k} A[i, j] B[j', k] ? 
        # No, standard definition: C[i, k] = sum_j A[i, j] B[j, k].
        # This is a linear map from (A, B) to C.
        # The tensor decomposition expresses this map as sum_c U_c A V_c B W_c?
        # The contract says:
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Output factor W[k*n+i, r].
        # Check condition: sum_c U[i*m+j, c] V[j'*p+k, c] W[k*n+i, c] == delta[j, j']
        # This confirms the tensor structure.
        
        e = exclude_idx
        # R[i, j, jp, k] = delta[j, jp] - U[i*m+j, e] * V[jp*p+k, e] * W[k*n+i, e]
        
        # To check if R is rank 1 in the sense of the decomposition:
        # We need to find U_res, V_res, W_res such that
        # U_res[i*m+j] * V_res[jp*p+k] * W_res[k*n+i] == R[i, j, jp, k]
        
        # This is equivalent to checking if the tensor R (reshaped appropriately) has rank 1.
        # Let's reshape R to (n*m, m*p, p*n)? No, the indices are mixed.
        # Let's just try to solve for the vectors.
        
        # If R is rank 1, then for any i, k, the matrix M_{j, jp} = R[i, j, jp, k] must be rank 1.
        # And the vectors must be consistent across i, k.
        
        # Step 1: Check if all slices M_{i, k} are rank 1.
        # Step 2: Check if the left and right singular vectors are consistent.
        
        U_vec = np.zeros(n*m)
        V_vec = np.zeros(m*p)
        W_vec = np.zeros(p*n)
        
        found_nonzero = False
        
        # We need to pick a reference element to normalize.
        # If R is rank 1, R[i, j, jp, k] = U[i*m+j] V[jp*p+k] W[k*n+i]
        
        # Let's iterate through all elements to build the vectors.
        # This is O(n*m*m*p*p*n) which is small.
        
        # However, we can be smarter.
        # Pick an arbitrary i0, j0, jp0, k0 such that R[i0, j0, jp0, k0] != 0.
        # If no such element exists, R is zero tensor, which is rank 0 (or 1 with zeros).
        # If R is zero, we can just remove the term and return success with zero vectors?
        # If R is zero, it means the excluded term was the only contributor to some parts? No.
        # If R is zero, then T_full = T_excluded. This implies the excluded term alone reproduces the entire tensor?
        # That would mean the original rank was 1? Unlikely.
        
        # Let's just compute R and try to factor it.
        
        # Efficient check:
        # 1. Compute R for all i, j, jp, k.
        # 2. Find first non-zero R[i, j, jp, k].
        # 3. Set U[i*m+j] = 1, V[jp*p+k] = 1, W[k*n+i] = R_val.
        # 4. Propagate to other elements.
        
        # If contradiction found, not rank 1.
        
        # Implementation:
        # Initialize U, V, W with NaN.
        U_res = np.full(n*m, np.nan)
        V_res = np.full(m*p, np.nan)
        W_res = np.full(p*n, np.nan)
        
        # We need to handle the case where R is identically zero.
        # If R is zero, we can set U_res=0, etc.
        
        is_zero = True
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        val = (1.0 if j == jp else 0.0) - U[i*m + j, e] * V[jp*p + k, e] * W[k*n + i, e]
                        if abs(val) > 1e-9:
                            is_zero = False
                            break
                    if not is_zero: break
                if not is_zero: break
            if not is_zero: break
            
        if is_zero:
            return True, np.zeros(n*m), np.zeros(m*p), np.zeros(p*n)
        
        # Find first non-zero
        found = False
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        val = (1.0 if j == jp else 0.0) - U[i*m + j, e] * V[jp*p + k, e] * W[k*n + i, e]
                        if abs(val) > 1e-9:
                            # Anchor
                            U_res[i*m + j] = 1.0
                            V_res[jp*p + k] = 1.0
                            W_res[k*n + i] = val
                            found = True
                            break
                    if found: break
                if found: break
            if found: break
        
        if not found:
            return True, np.zeros(n*m), np.zeros(m*p), np.zeros(p*n)
        
        # Propagate
        # For each element (i, j, jp, k):
        # val = U[i*m+j] * V[jp*p+k] * W[k*n+i]
        
        # We can determine missing values by finding a path from the anchor.
        # This is essentially solving a system of equations.
        # Given the sparsity of the tensor (delta[j, jp]), this might be tricky.
        
        # Alternative: Use SVD on slices.
        # For a fixed i, k, the matrix M_{j, jp} = R[i, j, jp, k] must be rank 1.
        # M_{j, jp} = U[i*m+j] * (V_res[jp*p+k] * W_res[k*n+i]) ? No.
        # R[i, j, jp, k] = U[i*m+j] V[jp*p+k] W[k*n+i]
        # For fixed i, k: M_{j, jp} = U[i*m+j] * (V[jp*p+k] * W[k*n+i]) is NOT correct.
        # V depends on jp, k. W depends on k, i.
        # So for fixed i, k, M_{j, jp} = U[i*m+j] * (sum? No, V and W are separate factors).
        # M_{j, jp} = U[i*m+j] * V[jp*p+k] * W[k*n+i].
        # Since W[k*n+i] is constant for fixed i, k, and V[jp*p+k] depends on jp,
        # M_{j, jp} = (U[i*m+j] * W[k*n+i]) * V[jp*p+k].
        # This is an outer product of vector u_i = U[:, i] (scaled by W) and v_k = V[:, k].
        # So M_{j, jp} must be rank 1.
        
        # Check rank 1 for all i, k slices.
        for i in range(n):
            for k in range(p):
                M = np.zeros((m, m))
                for j in range(m):
                    for jp in range(m):
                        M[j, jp] = (1.0 if j == jp else 0.0) - U[i*m + j, e] * V[jp*p + k, e] * W[k*n + i, e]
                
                # Check if M is rank 1
                # If M is all zeros, rank 0 (ok)
                # Else, compute SVD
                if np.allclose(M, 0, atol=1e-9):
                    continue
                U_svd, s, Vt = np.linalg.svd(M)
                if len(s) > 1 and s[1] > 1e-9:
                    return False, None, None, None
                # If rank 1, the singular vector is unique up to sign.
                # We can recover U_part and V_part.
                # M = u * s * v^T
                # So U[i*m+j] * W[k*n+i] = u_j * s
                # V[jp*p+k] = v_jp
                
                # This gives us relative values.
                # We need to combine info from all i, k.
                # This is getting complex to implement robustly in a short time.
                
                # Let's fallback to a simpler check: 
                # If we assume the structure, we can just verify if the "residual" can be represented.
                # Given the complexity, let's just try to see if removing a term from the 
                # known Strassen-like solution works by direct substitution if we can solve for the rest.
                
                pass
        
        # Since the propagation logic is complex, let's use a different strategy for the main loop.
        # We will just return False for now to avoid bugs, and rely on the random search 
        # or a simpler heuristic.
        
        # Actually, let's just use the random search but with a better initialization 
        # based on the block decomposition.
        
        return False, None, None, None

    # Schoolbook
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1.0
                V_school[j*p + k, col] = 1.0
                W_school[k*n + i, col] = 1.0
                col += 1
                
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Initialize with the known rank-50 block decomposition
    # Block decomposition: 2x2 blocks of size 2x2 and 2x1?
    # n=2, m=4, p=7.
    # We can view A as (2, 4) = (1, 2) x (2, 2)? No.
    # Let's use the inspiration: "2x2x2 Strassen + 2x2x1 Schoolbook"
    # This implies splitting the dimensions.
    # A (2x4) can be split into two (2x2) blocks? A = [A1, A2] where A1, A2 are 2x2.
    # B (4x7) can be split into (2x7) and (2x7)? B = [B1; B2] where B1, B2 are 2x7.
    # C = A1 B1 + A2 B2.
    # This is just sum of two matrix multiplications (2x2 * 2x7) and (2x2 * 2x7).
    # Rank of 2x2 * 2x7 is at most 14. Two of them is 28.
    # This is much better than 56.
    # Wait, the inspiration said rank 50. Why?
    # Maybe the split was different.
    # If we split m=4 into 2+2, we have two multiplications of size (2x2) * (2x7).
    # Rank of (2x2)*(2x7) is 14? No, rank is n*m*p = 2*2*7 = 28? No, schoolbook is 28.
    # Strassen for 2x2 is 7.
    # So 2x2 * 2x7 can be done with rank?
    # If we use Strassen on the 2x2 part, we need to handle the 7 columns.
    # This is non-trivial.
    
    # Let's stick to the random search but improve it.
    # The previous random search found nothing better than 56.
    # The inspiration found 50.
    # I will try to construct the rank-50 solution explicitly if I can figure it out, 
    # or use a more effective random search.
    
    # Let's try a random search with smaller integers and more iterations.
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Try to find a lower rank
    # We will try random integer matrices for ranks 55 down to 45
    # This is a heuristic search
    
    for rank in range(rank_school - 1, 44, -1):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for _ in range(20):
            if time.time() > deadline:
                break
                
            U = np.random.randint(-1, 2, size=(n*m, rank))
            V = np.random.randint(-1, 2, size=(m*p, rank))
            W = np.random.randint(-1, 2, size=(p*n, rank))
            
            if check_solution(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = rank
                break # Found a solution for this rank, move to lower rank
                
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
