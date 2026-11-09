"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """Verify exact decomposition. Returns True if valid."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Check shapes
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    # Build the multiplication tensor from decomposition
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This should equal 1 if indices match in the identity map, 0 otherwise.
    # For matrix multiplication: C_ik = sum_j A_ij * B_jk
    # Tensor form: T[i, j, k] = 1 if we are computing the term A_ij * B_jk
    # Wait, the standard decomposition for matrix multiplication is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has indices (i, j, k) for the operation A_ij * B_jk -> C_ik.
    # Actually, the tensor being decomposed is the identity tensor for the linear map?
    # No, the "matrix multiplication tensor" is defined such that:
    # sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = C[i,k] is not quite right.
    # Standard definition: The tensor $M_{n,m,p}$ has entries $T_{ijk} = 1$ if $i=j$ in some sense?
    # Let's look at the seed code structure:
    # U[i*m + j, c] corresponds to A_ij
    # V[j*p + k, c] corresponds to B_jk
    # W[k*n + i, c] corresponds to C_ik
    # The decomposition reconstructs the tensor of the multiplication operation.
    # The tensor T has indices (i, j, k) where the product is A_ij * B_jk.
    # Wait, the standard matrix multiplication tensor is usually defined on the space
    # of matrices A (n x m), B (m x p), and C (n x p).
    # The linear map is L(A, B) = AB.
    # The associated tensor T is in A^* \otimes B^* \otimes C.
    # T(A, B) = AB.
    # In coordinates: T_{(i,j), (j,k), (i,k)}? No.
    # Let's check the seed code's implicit tensor:
    # For each term c:
    #   u_c is a vector in R^{n*m}
    #   v_c is a vector in R^{m*p}
    #   w_c is a vector in R^{p*n} (indexed by k*n+i)
    #   The term contributes to the bilinear form?
    #   Actually, a tensor decomposition of rank r gives:
    #   T = sum_c u_c \otimes v_c \otimes w_c
    #   where T is in R^{n*m} \otimes R^{m*p} \otimes R^{p*n}?
    #   This doesn't look like standard matrix multiplication tensor which is in
    #   R^{n*m} \otimes R^{m*p} \otimes R^{n*p}.
    #   The W matrix in the seed has shape (p*n, rank).
    #   The index is k*n + i.
    #   Standard C matrix is n x p. Flattened is i*p + k or k*n + i.
    #   If W index is k*n + i, then W is flattened column-major? Or just a specific ordering.
    #   Let's verify the seed code's correctness by checking the tensor equation.
    #   The evaluator says: "exactly decompose the matrix multiplication tensor"
    #   and "output factor follows the official trace convention W[k*n+i,r]".
    #   This implies the third mode of the tensor is indexed by (k, i) with stride n for i?
    #   No, index = k*n + i. So for fixed k, i varies from 0 to n-1.
    #   This corresponds to C^T? Or C stored column-wise?
    #   If C is n x p, C[i, k]. Index k*n + i means k is the major index.
    #   This is column-major ordering of C (if C is n rows, p cols).
    #   
    #   The tensor T is defined such that:
    #   sum_{i,j,k} T_{(i,j), (j,k), (k,i)} A_{ij} B_{jk} = C_{ki}?
    #   No, the decomposition is:
    #   T = sum_c U_c \otimes V_c \otimes W_c
    #   This T must be the "identity" tensor for the multiplication map?
    #   Actually, the multiplication map is a bilinear map.
    #   It can be viewed as a tensor T in A^* \otimes B^* \otimes C.
    #   T(A, B) = C = AB.
    #   In coordinates: C_{ik} = sum_{j} A_{ij} B_{jk}.
    #   So the tensor T has components T_{(i,j), (j,k), (i,k)}?
    #   Wait, the indices in the tensor contraction are:
    #   (i, j) for A, (j, k) for B, (i, k) for C.
    #   So T_{\alpha \beta \gamma} where \alpha=(i,j), \beta=(j,k), \gamma=(i,k).
    #   T_{\alpha \beta \gamma} = 1 if the indices match the multiplication pattern, 0 otherwise.
    #   Specifically, T_{(i,j), (j',k), (i',k')} = 1 if i=i', j=j', k=k'.
    #   Note that the middle index j in A and j' in B must match for the term to exist in the sum?
    #   No, the tensor is fixed. It represents the *structure* of the operation.
    #   The operation is: for each i, k: C_ik = sum_j A_ij B_jk.
    #   The tensor T is such that C = T(A, B).
    #   C_ik = sum_{j} A_ij B_jk.
    #   So the non-zero entries of T are T_{(i,j), (j,k), (i,k)} = 1.
    #   Here the indices are:
    #   First mode: (i, j) -> index i*m + j
    #   Second mode: (j, k) -> index j*p + k
    #   Third mode: (i, k) -> index k*n + i (as per problem statement W[k*n+i])
    #   
    #   So we need to check:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if (i,j,k) is a valid triple?
    #   Wait, the tensor T_{\alpha \beta \gamma} is 1 if \alpha=(i,j), \beta=(j,k), \gamma=(i,k) are consistent?
    #   Actually, for *any* i, j, k, the term A_ij B_jk contributes to C_ik.
    #   So yes, for every i, j, k, the entry T_{(i,j), (j,k), (i,k)} should be 1?
    #   No, that would mean C_ik = sum_{i,j,k} A_ij B_jk which is wrong.
    #   The tensor T defines the linear map.
    #   T is an element of (R^{n*m})^* \otimes (R^{m*p})^* \otimes (R^{n*p}).
    #   T(A, B) = AB.
    #   The components of T are T_{\alpha \beta \gamma} such that:
    #   (T(A, B))_\gamma = sum_{\alpha, \beta} T_{\alpha \beta \gamma} A_\alpha B_\beta.
    #   Here \gamma corresponds to (i, k) with index k*n + i.
    #   \alpha corresponds to (i, j) with index i*m + j.
    #   \beta corresponds to (j, k) with index j*p + k.
    #   
    #   We want (AB)_{ik} = sum_j A_{ij} B_{jk}.
    #   So for a fixed output \gamma = (i, k), the sum is over \alpha=(i', j) and \beta=(j, k').
    #   T_{\alpha \beta \gamma} must be 1 if i'=i, k'=k, and the middle indices j match?
    #   No, the summation is over j.
    #   (AB)_{ik} = sum_j A_{ij} B_{jk}.
    #   This implies that T_{(i,j), (j,k), (i,k)} = 1.
    #   And T is 0 otherwise?
    #   Let's check:
    #   Sum_{\alpha, \beta} T_{\alpha \beta \gamma} A_\alpha B_\beta
    #   = Sum_{i', j, j', k'} T_{(i',j), (j',k'), (i,k)} A_{i'j} B_{j'k'}
    #   We want this to be A_{ik} B_{??} No, we want sum_j A_{ij} B_{jk}.
    #   So we need T_{(i',j), (j',k'), (i,k)} to be 1 if i'=i, k'=k, and j=j'.
    #   So T_{(i,j), (j,k), (i,k)} = 1.
    #   
    #   Therefore, the condition for validity is:
    #   For all i, j, k:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    #   
    #   Wait, is that right?
    #   If I pick a specific \gamma = (i, k), then:
    #   (T(A,B))_{ik} = sum_{j} sum_{j'} T_{(i,j), (j',k), (i,k)} A_{ij} B_{j'k}
    #   We want this to be sum_j A_{ij} B_{jk}.
    #   So T_{(i,j), (j',k), (i,k)} must be 1 if j=j' and 0 otherwise.
    #   So T_{(i,j), (j,k), (i,k)} = 1.
    #   
    #   So yes, the check is:
    #   For all i, j, k:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    
    # Efficient check using einsum or matmul
    # We can't just matmul U and V because the indices are coupled in W.
    # But we can check all (i, j, k) tuples.
    # n=3, m=5, p=6. Total 90 tuples.
    # Rank <= 90.
    # 90 * 90 * 90 operations? No.
    # We can do this:
    # For each c, we have a rank-1 term.
    # We want sum_c u_c[i*m+j] v_c[j*p+k] w_c[k*n+i] = 1.
    
    # Let's just iterate.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _random_init(rank, n, m, p, seed=None):
    """Initialize random matrices."""
    if seed is not None:
        np.random.seed(seed)
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    return U, V, W

def _gradient_step(U, V, W, lr, n, m, p):
    """Compute gradient and update for minimizing ||T - T_decomp||^2."""
    # T_decomp[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # Loss = sum_{i,j,k} (T_decomp[i,j,k] - 1)^2
    # Grad_U[i*m+j, c] = 2 * sum_{k} (T_decomp[i,j,k] - 1) * V[j*p+k, c] * W[k*n+i, c]
    # This is expensive. O(n*m*p*rank).
    # n=3, m=5, p=6, rank~90. 90*90 = 8100 ops per gradient?
    # 3*5*6*90 = 8100.
    # This is fast enough.
    
    # Precompute errors?
    # Error[i, j, k] = T_decomp[i,j,k] - 1
    # We can compute T_decomp efficiently?
    # T_decomp = einsum('ic,jc,kc->ijk', U_reshaped, V_reshaped, W_reshaped)?
    # U is (n*m, rank). Reshape to (n, m, rank).
    # V is (m*p, rank). Reshape to (m, p, rank).
    # W is (p*n, rank). Reshape to (p, n, rank).
    
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # T_decomp[i, j, k] = sum_c U_reshaped[i,j,c] * V_reshaped[j,k,c] * W_reshaped[k,i,c]
    # Note indices: U[i,j,c], V[j,k,c], W[k,i,c].
    # einsum: 'ijc,jkc,kic->ijk'
    T_decomp = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    # Error
    E = T_decomp - 1.0
    
    # Gradients
    # dL/dU[i,j,c] = 2 * sum_k E[i,j,k] * V[j,k,c] * W[k,i,c]
    # dL/dV[j,k,c] = 2 * sum_i E[i,j,k] * U[i,j,c] * W[k,i,c]
    # dL/dW[k,i,c] = 2 * sum_j E[i,j,k] * U[i,j,c] * V[j,k,c]
    
    # Compute dU
    # sum_k E[i,j,k] V[j,k,c] W[k,i,c]
    # Einsum: 'ijk,jkc,kic->ijc'
    dU = 2 * np.einsum('ijk,jkc,kic->ijc', E, V_reshaped, W_reshaped)
    
    # Compute dV
    # sum_i E[i,j,k] U[i,j,c] W[k,i,c]
    # Einsum: 'ijk,ijc,kic->jkc'
    dV = 2 * np.einsum('ijk,ijc,kic->jkc', E, U_reshaped, W_reshaped)
    
    # Compute dW
    # sum_j E[i,j,k] U[i,j,c] V[j,k,c]
    # Einsum: 'ijk,ijc,jkc->kic'
    dW = 2 * np.einsum('ijk,ijc,jkc->kic', E, U_reshaped, V_reshaped)
    
    U -= lr * dU
    V -= lr * dV
    W -= lr * dW
    
    return U, V, W, np.sum(E**2)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Start with schoolbook
    rank_sb = n * m * p
    U = np.zeros((n*m, rank_sb))
    V = np.zeros((m*p, rank_sb))
    W = np.zeros((p*n, rank_sb))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i*m*p + j*p + k
                U[i*m+j, c] = 1
                V[j*p+k, c] = 1
                W[k*n+i, c] = 1
    
    best_U, best_V, best_W = U, V, W
    best_loss = 0.0
    
    # Try to reduce rank via optimization
    # Start with a rank slightly below schoolbook? Or start with schoolbook and zero out columns?
    # Strategy: Start with rank R. Initialize from schoolbook by taking first R columns?
    # Or random init?
    # Let's try random init for rank R, optimize, see if loss goes to 0.
    
    # Possible ranks to try: from n*m*p-1 down to some limit.
    # But random init might get stuck.
    # Let's try a "peeling" or "zeroing" approach.
    # Start with schoolbook, try to set some columns to 0 and re-optimize the rest?
    # Or start with rank R = n*m*p - 1 and random init?
    
    # Given the time constraint, let's try a few ranks with random init and gradient descent.
    # If we find a solution with loss < 1e-6, we can round to integers/halves?
    # The problem asks for integer/half-integer entries.
    # Gradient descent gives floats.
    # We need to round.
    # If the true rank is achievable, the floats should be close to integers/halves.
    
    # Let's try ranks: 89, 88, ... down to maybe 70.
    # But 90 is the schoolbook rank.
    # Is 89 possible?
    # For 3x3x3, min rank is 23 (Strassen 7 for 2x2x2, 3x3x3 is 23). 27 is schoolbook.
    # So reduction is possible.
    
    # Let's try to optimize.
    # Use Adam optimizer? Simple gradient descent with decay.
    
    ranks_to_try = list(range(n*m*p - 1, 50, -1))
    
    for R in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        # Random init
        U, V, W = _random_init(R, n, m, p, seed=R)
        
        # Optimize
        lr = 0.01
        for step in range(200):
            if time.time() > deadline - 5:
                break
            U, V, W, loss = _gradient_step(U, V, W, lr, n, m, p)
            if loss < 1e-8:
                # Check if we can round
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if _verify(U_r, V_r, W_r):
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_loss = 0
                    # Found a valid lower rank!
                    # We can stop early if we want the minimum, but we need to check if lower is possible.
                    # Given time, return this.
                    return {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
            # Decay LR
            lr *= 0.95
        
        # If loss is low but not < 1e-8, maybe check rounding?
        if loss < 1e-4:
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            if _verify(U_r, V_r, W_r):
                best_U, best_V, best_W = U_r, V_r, W_r
                best_loss = 0
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
    
    # Return best found (schoolbook if no improvement)
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

# EVOLVE-BLOCK-END
