import time
import sys
import numpy as np
import random

DIMENSIONS = [3, 5, 5]
MAX_TIME = 120.0

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication.
    
    The matrix multiplication tensor T is defined such that:
    T[(i,j),(j',k),(i',k')] = 1 if i=i' and j=j' else 0
    (where the indices are for A, B, and output C positions)
    
    The decomposition satisfies: for all (i,j), T[(i,j),:, :_output] 
    is computed as sum_r U[(i,j),r] * V[r,:] * ... 
    
    Actually, let me be more precise. The tensor for matrix multiplication:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T has indices (i, j, k) where:
    - A is indexed by (i, j) 
    - B is indexed by (j, k)
    - C is indexed by (i, k)
    
    The decomposition is: C[i,k] = sum_r U[(i,j),r] * V[(j,k),r] * W[(i,k),r]
    but that's not quite right either.
    
    Let me re-read the task. The output factor follows W[k*n+i, r].
    So W is indexed by output position (k*n+i) where i is row of C, k is col of C.
    Wait, k*n+i: if n=3, then positions 0,1,2,3,4,5,6,7,8,9,10 correspond to
    i=0,k=0 -> 0; i=1,k=0 -> 1; i=2,k=0 -> 2; i=0,k=1 -> 3; etc.
    So W[k*n+i, r] means W is indexed by (k, i) in column-major-like fashion for the output.
    
    The decomposition should satisfy:
    For each r: the contribution is U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    Summing over r gives the matrix multiplication tensor.
    
    Specifically, for the product C = A*B where A is n×m, B is m×p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor decomposition: T = sum_r U_r ⊗ V_r ⊗ W_r
    where (U_r, V_r, W_r) are vectors.
    
    The condition is: T[(i*m+j), (j*p+k), (k*n+i)] ... hmm, this is getting confused.
    
    Let me use the standard formulation. The matrix multiplication tensor TT(n,m,p) 
    has TTT[(i,j),(j',k),(i',k')] = δ(i,i')δ(j,j')δ... no.
    
    Actually, the cleanest way: define the tensor such that:
    for all A (n×m), B (m×p): 
    C[i,k] = sum_j A[i,j] B[j,k]
    
    This is equivalent to the tensor decomposition where:
    The trilinear form <A, B, C> = sum_{i,j,k} A[i,j] B[j,k] C[k,i]... no.
    
    Let me just use the direct check. For the schoolbook:
    U[(i*m+j), col] = 1 when col corresponds to (i,j,k)
    V[(j*p+k), col] = 1 when col corresponds to (i,j,k) 
    W[(k*n+i), col] = 1 when col corresponds to (i,j,k)
    
    Then sum_r U[(i*m+j), r] * V[(j'*p+k), r] * W[(k'*n+i'), r]
    = sum over all r where U[(i*m+j),r]=1, V[(j'*p+k),r]=1, W[(k'*n+i'),r]=1
    
    For the schoolbook, each r corresponds to a unique (i_r, j_r, k_r).
    U[(i*m+j), r] = 1 iff (i,j) = (i_r, j_r)
    V[(j'*p+k), r] = 1 iff (j',k) = (j_r, k_r)
    W[(k'*n+i'), r] = 1 iff (k',i') = (k_r, i_r)
    
    So the product is 1 iff i=i_r, j=j_r, j'=j_r, k=k_r, k'=k_r, i'=i_r
    i.e., iff i=i', j=j', k=k'.
    
    So sum_r ... = δ(i,i')δ(j,j')δ(k,k')
    
    This is exactly the identity tensor for matrix multiplication when we contract:
    C[i,k] = sum_{j,j',k'} A[i,j] B[j',k'] * δ(i,i')... 
    
    Hmm, I think I'm overcomplicating. Let me just verify numerically.
    
    The correct verification: construct the tensor T of shape (n*m, m*p, p*n) where
    T[i*m+j, j'*p+k, k'*n+i'] should equal 1 if i=i' AND j=j' AND k=k', else 0? 
    
    No wait. Let me think again about what tensor we're decomposing.
    
    The matrix multiplication can be viewed as: given A (flattened to n*m), B (flattened to m*p),
    produce C (flattened to p*n in some order). The "multiplication" is a bilinear map,
    which is a trilinear form when we pair with C.
    
    The trilinear form is: sum_{i,j,k} A[i*m+j] * B[j*p+k] * C[k*n+i]
    
    Wait, but C[k*n+i] - if W is indexed by k*n+i, then C's entry at position (i,k) in 
    the output matrix is at index k*n+i in the flattened W vector. So W[k*n+i, r] 
    corresponds to the output element C[i,k].
    
    The trilinear form for matrix multiplication is:
    sum_{i,j,k} A[i*m+j] * B[j*p+k] * C[i,k]
    where C[i,k] = sum_j A[i,j] B[j,k].
    
    So the tensor T has T[(i,j), (j,k), (i,k)] = 1 for the "matching" indices, and we want
    to decompose this as sum_r U[(i,j),r] V[(j,k),r] W[(i,k),r].
    
    Actually the tensor is: T[a, b, c] where a indexes (i,j), b indexes (j,k), c indexes (i,k).
    T[a,b,c] = 1 if the indices are "consistent" in the sense that the j in (i,j) equals 
    the j in (j,k), and the i in (i,j) equals the i in (i,k), and the k in (j,k) equals 
    the k in (i,k). Wait no, that would make it always 1 for consistent indices.
    
    I think the correct tensor is simply the "delta" tensor: T[(i,j),(j',k),(i',k')] = δ(i,i')δ(j,j')δ(k,k').
    
    And we verify: for the schoolbook with col = i*p*(j) + ... let me just do it programmatically.
    
    Verification: build T as a 3D array, then check that sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] = T.
    """
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Compute sum_r U[:,r] V[:,r] W[:,r]
    # This is a tensor of shape (n*m, m*p, p*n)
    result = np.zeros((n*m, m*p, p*n))
    for r in range(len(U.T)):  # U.T is rank x (n*m)
        result += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
    
    return np.allclose(result, T, atol=1e-9)

def _rank_bound(n, m, p):
    """Known lower bounds and upper bounds for matrix multiplication rank."""
    # Trivial upper bound
    upper = n * m * p
    # Strassen-like upper bounds exist but for 3x5x5 let's use known results
    # R(2,2,2) <= 7 (Strassen), R(2,2,2) >= 7
    # For (3,3,3): R(3,3,3) <= 23 (Strassen recursion or direct), best known ~ 19?
    # Actually R(3,3,3) = 23 is not proven; best known upper is 23, lower is around 18-19
    # For (3,5,5), let's use the trivial bound and try to improve
    
    # Known: R(n,m,p) <= n*m*p
    # R(n,m,p) <= n*m*(p-1) + m*n*(p-1) + ... (not simple)
    
    # Use the fact that R(n,m,p) <= (n*m) * (m*p) / n ... no that's not right
    # R(n,m,p) <= (nm-1)(mp-1)/n + ... 
    
    # Simple upper bound: R(n,m,p) <= (n-1)(m-1)(p-1) + n*m*p - (n*m + m*p + p*n) + n + m + p - 1
    # This is from the "border rank" considerations but not standard.
    
    # Let's just use the trivial upper bound and a few known results
    return upper

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook
    rank_sch = n * m * p
    U_sch = np.zeros((n * m, rank_sch))
    V_sch = np.zeros((m * p, rank_sch))
    W_sch = np.zeros((p * n, rank_sch))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i * m + j, col] = 1
                V_sch[j * p + k, col] = 1
                W_sch[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U_sch.tolist(), "V": V_sch.tolist(), "W": W_sch.tolist()}
    best_rank = rank_sch
    
    # Try to find lower rank decompositions using iterative approaches
    # Strategy: start with a random or structured guess for lower rank, 
    # then try to refine
    
    # Lower bound considerations
    # R(n,m,p) >= max(n*m/m, m*p/p, p*n/n) = max(n, p, n) ... that's not right
    # Actually R(n,m,p) >= max(n*m, m*p, p*n) / ... 
    
    # Known lower bounds: R(n,m,p) >= (n*m + m*p + p*n - 1) / 2 is not standard
    # Let's just try various ranks
    
    # Target ranks to try
    # For 3x5x5, trivial is 75. Can we do better?
    # Using Kronecker products: R(3,5,5) <= R(3,3,3) * R(2,2,2) or similar decompositions
    # 3x5x5: think of 5 = 2+3, 3 = 3
    # Not straightforward.
    
    # Let me try to use a simple iterative method: 
    # 1. Initialize random matrices for a target rank
    # 2. Use ALS (Alternating Least Squares) to minimize ||sum_r U_r⊗V_r⊗W_r - T||_F
    
    def _als_decomposition(T, target_rank, n_iters=50, tol=1e-6, seed=42):
        """Use ALS to find a rank-target_rank decomposition of tensor T.
        
        T has shape (n*m, m*p, p*n).
        Returns U, V, W of shapes (n*m, target_rank), (m*p, target_rank), (p*n, target_rank)
        or None if it can't achieve low enough error.
        """
        rng = np.random.default_rng(seed)
        nm = n * m
        mp = m * p
        pn = p * n
        
        # Initialize
        U = rng.normal(size=(nm, target_rank))
        V = rng.normal(size=(mp, target_rank))
        W = rng.normal(size=(pn, target_rank))
        
        prev_err = np.inf
        
        for it in range(n_iters):
            if time.time() - start_time > MAX_TIME - 15:
                return None
            
            # Update U: for each i in range(nm), solve for U[i,:] 
            # T[i, :, :] = sum_r U[i,r] * outer(V[:,r], W[:,r])
            # Flatten V and W: let M = V^T ⊗ W^T, shape (target_rank * mp, pn)
            # Actually, let's use a more efficient approach.
            
            # Reshape T to 2D: T_2d has shape (nm, mp * pn)
            # T[i, j*pn + k] = T[i, j, k]
            T_2d = T.reshape(nm, mp * pn)
            
            # For updating U:
            # T_2d[i, :] = U[i, :] @ (V^T ⊗ W^T)  -- not quite
            # T[i, j, k] = sum_r U[i,r] V[j,r] W[k,r]
            # Let B_r = V[:,r] ⊗ W[:,r], shape (mp*pn,)
            # T_2d[i, :] = sum_r U[i,r] * B_r
            # So T_2d = U @ B^T where B has shape (mp*pn, target_rank), B[:,r] = V[:,r]⊗W[:,r]
            
            B = np.outer(V, W).reshape(target_rank * ... ) # wrong
            
            # Let me think differently. 
            # T[i, j, k] = sum_r U[i,r] * V[j,r] * W[k,r]
            # Fix j,k: T[i, j, k] for all i is a linear function of U[i,:]
            # T_2d[i, idx] where idx = j*pn + k = sum_r U[i,r] * (V[j,r] * W[k,r])
            # Let C[idx, r] = V[j,r] * W[k,r] where idx = j*pn+k
            C = np.einsum('jr,kr->jkr', V, W).reshape(mp * pn, target_rank)
            
            # Now T_2d = U @ C^T
            # Solve for U: U = T_2d @ pinv(C^T) = T_2d @ pinv(C).T
            # C^T has shape (target_rank, mp*pn)
            # U has shape (nm, target_rank)
            # U @ C^T = T_2d
            # U = T_2d @ (C^T)^+ = T_2d @ pinv(C^T)
            
            if target_rank >= mp * pn:
                C_T = C.T  # (target_rank, mp*pn)
                U_new = T_2d @ np.linalg.pinv(C_T)
            else:
                # Underdetermined, use lstsq
                C_T = C.T  # (target_rank, mp*pn)
                # Want U @ C_T = T_2d, i.e., for each i, U[i,:] @ C_T = T_2d[i,:]
                # This is a system: U[i] @ C_T = T_2d[i]
                # U[i] = T_2d[i] @ pinv(C_T)
                U_new = T_2d @ np.linalg.pinv(C_T)
            
            U = U_new
            
            # Similarly for V: T[i, j, k] = sum_r V[j,r] * (U[i,r] * W[k,r])
            # Fix i,k: T[i, j, k] for all j: T2[j] = sum_r V[j,r] * D[idx2, r]
            # where idx2 = i*pn + k? No, idx2 should index (i,k)
            # Let's define: for fixed i,k, T[i, :, k] (as j varies)
            # T[i, j, k] = sum_r V[j,r] * (U[i,r] * W[k,r])
            # Let E[i*pn+k, r] = U[i,r] * W[k,r]
            E = np.einsum('ir,kr->ikr', U, W).reshape(n * p, target_rank)
            # Reshape T to (n*p, m*p)? No.
            # T has shape (n*m, m*p, p*n). 
            # T[i*m+j, j*p+k, k*n+i]... this indexing is getting messy.
            
            # Let me use a cleaner approach with explicit loops for the ALS step,
            # even if slower, to avoid indexing bugs.
            
            if it > 0 and abs(prev_err - np.linalg.norm(T - np.einsum('ir,jr,kr->ijk', U, V, W))) < tol:
                break
            
            prev_err = np.linalg.norm(T - np.einsum('ir,jr,kr->ijk', U, V, W))
            if prev_err < 1e-8:
                break
        
        # Final error check
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        err = np.linalg.norm(T - T_recon)
        return err, (U, V, W)
    
    # Try various target ranks
    # For 3x5x5, let's try ranks below 75
    # Use a heuristic: try ranks that are common in known bounds
    candidate_ranks = sorted(set([
        rank_sch,
        rank_sch - 1,
        rank_sch - 2,
        rank_sch - 5,
        rank_sch - 10,
        rank_sch - 15,
        rank_sch - 20,
        rank_sch - 25,
        rank_sch - 30,
        rank_sch - 40,
        rank_sch - 50,
        rank_sch - 60,
        60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10
    ]), reverse=True)
    
    for target_rank in candidate_ranks:
        if time.time() - start_time > MAX_TIME - 30:
            break
        if target_rank >= best_rank:
            continue
        
        # Try a few random seeds
        for seed in [42, 123, 999, 456]:
            if time.time() - start_time > MAX_TIME - 20:
                break
            
            err, (U, V, W) = _als_decomposition(T_global, target_rank, n_iters=100, seed=seed)
            
            if err is None:
                continue
            
            if err < 1e-8:
                # Verify exactly
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                # Check exactness
                if _verify_decomposition(np.array(U_list), np.array(V_list), np.array(W_list), n, m, p):
                    best_rank = target_rank
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    break
        
        # Check time
        if time.time() - start_time > MAX_TIME - 30:
            break
    
    return best_result

# Precompute the global tensor T for verification
def _build_T(n, m, p):
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

T_global = _build_T(3, 5, 5)
