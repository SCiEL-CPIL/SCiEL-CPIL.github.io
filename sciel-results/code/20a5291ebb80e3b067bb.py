import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 2x3 matrix by a 3x6 matrix.

    Strategy:
    1. Start with the known schoolbook rank-36 decomposition.
    2. Try to reduce rank using a random sparse search and local refinement.
    3. If time permits, try more aggressive optimizations.

    The output factor follows the official trace convention W[k*n+i, r].
    """
    deadline = time.time() + 145  # Leave 5s safety margin

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 36

    # --- Build the target tensor condition ---
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta_{j,j'}
    # Actually, the matrix multiplication tensor is:
    # T[i, j, k, i', j', k'] = delta_{j,j'} * delta_{k,k'}
    # But in the "trace" convention for bilinear algorithms:
    # C_{i,k} = sum_{j} A_{i,j} B_{j,k}
    # Bilinear form: sum_r U[i*m+j, r] * V[j'*p+k', r] * W[k*n+i, r] 
    # Wait, the standard convention is:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # We decompose this as sum_r (sum_j U_{ij}^r A_{ij}) * (sum_k V_{jk}^r B_{jk}) * W_{ik}^r
    # Actually, the standard tensor decomposition for matrix mult:
    # T_{i,j,k} = delta_{j,j'} ... no.
    
    # Let's use the standard definition:
    # The matrix multiplication tensor T has entries T_{i,j,k} = 1 if and only if ...
    # No, the bilinear algorithm satisfies:
    # sum_{r} u_{i,j}^{(r)} v_{j',k}^{(r)} w_{k,i}^{(r)} = delta_{j,j'}
    # where u_{i,j}^{(r)} is the coefficient of A_{i,j}, v_{j',k}^{(r)} is the coefficient of B_{j',k},
    # and w_{k,i}^{(r)} is the coefficient of C_{k,i} (or C_{i,k}).

    # Let's be precise. The task says:
    # U shape: (n*m, rank) -> U[i*m + j, r]
    # V shape: (m*p, rank) -> V[j*p + k, r]
    # W shape: (p*n, rank) -> W[k*n + i, r]

    # The condition is:
    # sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = delta_{j, j'} * delta_{k, k}
    # Wait, that's not right either. Let's look at the schoolbook code:
    # U[i*m+j][col] = 1, V[j*p+k][col] = 1, W[k*n+i][col] = 1
    # For a fixed (i,j,k), the column contributes 1 to U[i*m+j], V[j*p+k], W[k*n+i].
    # The product U[i*m+j] * V[j'*p+k'] * W[k'*n+i'] is 1 only if j=j' and k=k' and i=i' (wait, no).
    
    # Let's re-derive.
    # The bilinear algorithm computes C = A * B.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The algorithm is: sum_r (sum_j u_{i,j}^{(r)} A_{i,j}) * (sum_k v_{j,k}^{(r)} B_{j,k}) * w_{i,k}^{(r)}
    # This must equal C_{i,k} for all i,k.
    # So, sum_r sum_j sum_k u_{i,j}^{(r)} v_{j,k}^{(r)} w_{i,k}^{(r)} A_{i,j} B_{j,k} = sum_j A_{i,j} B_{j,k}
    # This implies: sum_r u_{i,j}^{(r)} v_{j',k}^{(r)} w_{i,k}^{(r)} = delta_{j,j'}
    # Wait, the indices in the product are u_{i,j}, v_{j',k}, w_{i,k}.
    # The sum over r must give delta_{j,j'}. The i and k are free indices on both sides?
    # No, the equation is: for all i, j, j', k:
    # sum_r u_{i,j}^{(r)} v_{j',k}^{(r)} w_{i,k}^{(r)} = delta_{j,j'}
    # This must hold for all i and k.
    
    # Let's check the schoolbook:
    # For a fixed (i0, j0, k0), we have one term with u_{i0,j0}=1, v_{j0,k0}=1, w_{i0,k0}=1.
    # Sum over r: only the r corresponding to (i0,j0,k0) contributes.
    # LHS = u_{i,j}^{(r)} v_{j',k}^{(r)} w_{i,k}^{(r)}
    # If i=i0, j=j0, j'=j0, k=k0, then LHS = 1*1*1 = 1. RHS = delta_{j0,j0} = 1. Correct.
    # If i=i0, j=j0, j'=j1 (j1!=j0), k=k0, then LHS = 0. RHS = delta_{j0,j1} = 0. Correct.
    # If i=i0, j=j0, j'=j0, k=k1 (k1!=k0), then LHS = 0. RHS = delta_{j0,j0} = 1. WRONG!
    
    # Ah, the schoolbook code in the prompt has:
    # U[i*m+j][col] = 1, V[j*p+k][col] = 1, W[k*n+i][col] = 1
    # This corresponds to:
    # u_{i,j} = 1, v_{j,k} = 1, w_{k,i} = 1
    # Note: W is indexed by [k*n+i]. So w_{k,i} is the element at row k*n+i.
    # The product is u_{i,j} * v_{j',k'} * w_{k',i}
    # Wait, the standard formula is sum_r u_{i,j}^{(r)} v_{j',k}^{(r)} w_{i,k}^{(r)} = delta_{j,j'}
    # But here w is w_{k,i}. So the formula is:
    # sum_r u_{i,j}^{(r)} v_{j',k}^{(r)} w_{k,i}^{(r)} = delta_{j,j'} * delta_{k,k} ... no.
    
    # Let's look at the dimensions.
    # U: (n*m, rank). Index i*m+j.
    # V: (m*p, rank). Index j*p+k.
    # W: (p*n, rank). Index k*n+i.
    
    # The condition must be:
    # sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = delta_{j,j'} * delta_{k,k} ... no.
    # It should be:
    # sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = delta_{j,j'}
    # But this doesn't involve k and k' properly.
    
    # Actually, the standard tensor T for matrix multiplication M_{n,m,p} has dimensions (n*m) x (m*p) x (p*n).
    # T_{(i,j), (j',k), (k',i')} = 1 if j=j' and k=k' and i=i'.
    # Wait, no.
    # The bilinear form is <A, B> -> C.
    # T is a 3-tensor. T_{(i,j), (j',k), (k',i')} = 1 if the term A_{i,j} B_{j',k} contributes to C_{k',i'}.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So A_{i,j} B_{j',k} contributes to C_{i,k} only if j=j' and the output index is (i,k).
    # So T_{(i,j), (j',k), (k',i')} = 1 if j=j' and i'=i and k'=k.
    # In the decomposition:
    # T_{a,b,c} = sum_r U_{a,r} V_{b,r} W_{c,r}
    # where a = i*m+j, b = j'*p+k, c = k'*n+i'.
    # So we need:
    # sum_r U[i*m+j, r] V[j'*p+k, r] W[k'*n+i', r] = 1 if j=j', i'=i, k'=k; else 0.
    
    # Let's check schoolbook with this:
    # For a specific (i0, j0, k0), the column has U[i0*m+j0]=1, V[j0*p+k0]=1, W[k0*n+i0]=1.
    # Product U[i*m+j] V[j'*p+k] W[k'*n+i'] is 1 only if (i,j)=(i0,j0), (j',k)=(j0,k0), (k',i')=(k0,i0).
    # This implies j=j0, k=k0, i'=i0, and i=i0.
    # So the product is 1 if j=j0, k=k0, i'=i0, i=i0.
    # The condition for the tensor is 1 if j=j', i'=i, k'=k.
    # Here j'=j0, k'=k0, i'=i0.
    # So we need j=j0 (which is j=j'), i'=i0 (which is i'=i), k'=k0 (which is k'=k).
    # This matches perfectly!
    
    # So the condition is:
    # For all i, j, j', k, k', i':
    # sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = 1 if (j==j' and i==i' and k==k') else 0.

    # Let's define the target tensor T of shape (n*m, m*p, p*n).
    # T[a, b, c] = 1 if conditions met, else 0.
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for j_prime in range(m):
                for k in range(p):
                    for k_prime in range(p):
                        for i_prime in range(n):
                            a = i*m + j
                            b = j_prime*p + k
                            c = k_prime*n + i_prime
                            if j == j_prime and i == i_prime and k == k_prime:
                                T[a, b, c] = 1.0

    # --- Helper to verify a decomposition ---
    def verify_decomposition(U, V, W, tol=1e-8):
        """Check if U, V, W exactly decompose T."""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        rank = U.shape[1]
        # Compute the tensor: T_recon[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # This is a mode-3 contraction.
        # We can do: T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_recon, T, atol=tol)

    # --- Helper to convert to list of lists ---
    def to_list(matrix):
        return [[float(x) for x in row] for row in matrix]

    # --- Start with schoolbook ---
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school

    # --- Try to find a lower rank decomposition ---
    # We'll try ranks from 32 up to 35.
    # For each rank, we'll do a random search.
    
    # Strategy: 
    # 1. Random initialization of U, V, W with small values.
    # 2. Use a simple iterative method or just random search?
    # Random search for rank 32 is extremely unlikely to hit the exact solution.
    # We need a smarter approach.
    
    # Alternative: Use the fact that we can try to "merge" terms in the schoolbook decomposition.
    # Or use a known lower-rank algorithm if available.
    # For 2x3x6, is there a known rank < 36?
    # 2x2x2 has rank 7 (Strassen).
    # 2x2x3 has rank 11?
    # 2x3x3 has rank 14?
    # 2x3x6 could be related.
    
    # Let's try a random restart local search.
    # We'll define a loss function: ||T_recon - T||_F^2
    # And try to minimize it.
    
    def loss(U, V, W):
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((T_recon - T)**2)

    # We'll try to find a solution for rank R.
    # We can use a simple gradient descent if we can compute gradients.
    # Grad_U = einsum('br,cr->ar', V, W) * 2 * (T_recon - T)
    # This is expensive but doable for small sizes.
    
    # Let's try a few random restarts for rank 32 and 33.
    # If we find one, we return it.
    # If not, we return schoolbook.
    
    for target_rank in [32, 33, 34, 35]:
        if time.time() > deadline - 10:
            break
            
        found = False
        num_restarts = 5
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            # Simple gradient descent
            # We'll do a few steps.
            lr = 0.01
            for step in range(100):
                if time.time() > deadline - 5:
                    break
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                diff = T_recon - T
                if np.allclose(diff, 0, atol=1e-8):
                    found = True
                    break
                # Gradients
                grad_U = 2 * np.einsum('bc,br->cr', diff, V * W) # Wait, grad_U[a,r] = 2 * sum_{b,c} diff[a,b,c] V[b,r] W[c,r]
                grad_U = 2 * np.einsum('abc,br,cr->ar', diff, V, W)
                grad_V = 2 * np.einsum('abc,ar,cr->br', diff, U, W)
                grad_W = 2 * np.einsum('abc,ar,br->cr', diff, U, V)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Normalize to prevent explosion?
                # U = U / (np.linalg.norm(U) + 1e-10) * 10
                # This might break the structure.
                
            if np.allclose(np.einsum('ar,br,cr->abc', U, V, W), T, atol=1e-8):
                found = True
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
        if found:
            break

    # --- Final check and return ---
    # Ensure we return the best valid decomposition.
    if not verify_decomposition(best_U, best_V, best_W):
        # Fallback to schoolbook if something went wrong
        best_rank = rank_school
        best_U, best_V, best_W = U_school, V_school, W_school

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
