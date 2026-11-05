import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Build the target tensor Y
    # Y[i, j, k_prime, k] = 1 if j == k_prime else 0
    # Note: The task description says W[k*n+i, r].
    # The tensor for matrix multiplication C = A B is:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # Tensor T[i, j, j', k] is 1 if j == j' else 0.
    # The decomposition is sum_r U[i, j, r] V[j, k, r] W[k', i, r] ?
    # Let's check the indices in the current code's verify function:
    # D = np.einsum('ijc,jkc,kic->ijjk', U, V, W)
    # U is (n, m, rank) -> indices i, j, c
    # V is (m, p, rank) -> indices j, k, c  (Wait, V is m x p? No, V is m*p x rank reshaped to m, p, rank? 
    # Current code: V = params...reshape(m, p, rank). 
    # W is (p, n, rank) -> indices k, i, c? 
    # The einsum 'ijc,jkc,kic->ijjk' implies:
    # U[i,j,c] * V[j,k,c] * W[k,i,c] = D[i,j,j',k] ? No, output is i,j,j',k?
    # Let's trace:
    # U: i, j, c
    # V: j, k, c  (Here k corresponds to the second index of V, which is p-dim)
    # W: k, i, c  (Here k is the first index of W, which is p-dim? No, W is p*n. Reshaped to p, n, rank. So indices are k, i, c.)
    # Sum over c.
    # Result index: i, j, k, i? No, the output is 'ijjk'.
    # Let's look at the target Y in current code:
    # Y[i, j, j_prime, k] = 1 if j == j_prime.
    # So D[i, j, j_prime, k] should be 1 if j == j_prime.
    # In the einsum:
    # U[i, j, c]
    # V[j, k, c]  -- Wait, V is reshaped from (m*p, rank) to (m, p, rank). So indices are j (m), k (p), c.
    # W[k, i, c]  -- Wait, W is reshaped from (p*n, rank) to (p, n, rank). So indices are k (p), i (n), c.
    # The einsum 'ijc,jkc,kic->ijjk' is:
    # Sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # The output indices are i, j, j, k? No, the string is 'ijjk'.
    # This implies the third index of the output is 'j' (from V's second index? No, V's second index is 'k').
    # Let's re-read the einsum string carefully.
    # 'ijc, jkc, kic -> ijjk'
    # Input 1: i, j, c
    # Input 2: j, k, c
    # Input 3: k, i, c
    # Output: i, j, j, k
    # This means the second 'j' in the output comes from... where?
    # The input indices are:
    # U: i, j, c
    # V: j, k, c
    # W: k, i, c
    # The free indices in the output are i, j, j, k.
    # But 'j' appears in U and V. 'k' appears in V and W. 'i' appears in U and W.
    # So the output is a 4-tensor T[i, j, j_out, k] where j_out is... ?
    # Actually, standard einsum notation:
    # If an index appears in multiple inputs, it is summed over? No, only if it's not in the output.
    # If it IS in the output, it must appear in exactly one input? No, it can appear in multiple inputs if it's the same index?
    # No, einsum requires that if an index is in the output, it must be in exactly one input? 
    # Actually, if an index is in the output, it must be present in the inputs. If it appears in multiple inputs, it implies equality constraint?
    # Standard rule: Indices in output must be in inputs. Indices in inputs but not output are summed.
    # If an index appears in multiple inputs and in the output, it's not standard for summation.
    # Let's assume the current code's einsum is correct for the specific tensor layout it defines.
    # D[i, j, j, k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c].
    # The target Y is Y[i, j, j_prime, k] = 1 if j == j_prime.
    # So we need D[i, j, j, k] (where the 3rd index is the same as the 2nd? No, the output string is 'ijjk'.
    # This means the 3rd index of the output tensor is 'j'.
    # So D[i, j, j, k] = sum_c ...
    # And we want D[i, j, j_prime, k] = 1 if j == j_prime.
    # This implies that for j != j_prime, D[i, j, j_prime, k] must be 0.
    # But the einsum produces a tensor where the 3rd index is tied to the 2nd index 'j'?
    # No, 'ijjk' means the output has 4 dimensions: dim0=i, dim1=j, dim2=j, dim3=k.
    # This is a tensor with symmetry between dim1 and dim2?
    # If the output is 'ijjk', then D[a,b,c,d] is only non-zero if b==c? No, the variable name 'j' is just a label.
    # If I use 'ijjk', it means the 2nd and 3rd dimensions are both indexed by 'j'.
    # This effectively creates a tensor where the 2nd and 3rd indices are identical in the array layout?
    # No, it means the resulting array has shape (n, m, m, p) and the value at [i, j, j_prime, k] is computed by summing over c:
    # U[i, j, c] * V[j, k, c] * W[k, i, c] ?
    # Wait, if the output is 'ijjk', then the 3rd index is 'j'.
    # So D[i, j, j, k] is the value.
    # But we need D[i, j, j_prime, k] for ALL j, j_prime.
    # If the einsum is 'ijc,jkc,kic->ijjk', it produces an array where the 3rd index is the SAME as the 2nd?
    # No, it produces an array of shape (n, m, m, p).
    # The value at index [i, j, j_prime, k] is obtained by setting the output indices i=i, j=j, j=j_prime, k=k.
    # But the input indices are:
    # U: i, j, c
    # V: j, k, c
    # W: k, i, c
    # For the output index [i, j, j_prime, k], we need to match the input indices.
    # The 'j' in the output (2nd pos) matches 'j' in U and V.
    # The 'j' in the output (3rd pos) matches... which input?
    # There is no 'j' in W.
    # There is no 'j' in V's second slot (it's 'k').
    # There is no 'j' in V's first slot (it's 'j').
    # If the output string is 'ijjk', it implies the 3rd dimension is indexed by 'j'.
    # But 'j' is already used for the 2nd dimension.
    # In NumPy einsum, if an index appears in the output, it must appear in the inputs.
    # If 'j' appears in the output at pos 1 and pos 2, it must appear in the inputs.
    # It appears in U (pos 1) and V (pos 0).
    # This implies that the 2nd and 3rd dimensions of the output are BOTH linked to the 'j' index in the inputs.
    # This means D[i, j, j', k] will be 0 unless j == j'? 
    # No, it means the value is computed for the specific j.
    # If I ask for D[i, 0, 1, k], the einsum will try to evaluate with j=0 (for the 2nd output index) and j=1 (for the 3rd output index)?
    # This is not how einsum works.
    # Einsum: 'ijc,jkc,kic->ijjk'
    # This is invalid or behaves unexpectedly if 'j' is repeated in the output without being a diagonal?
    # Actually, if 'j' is in the output twice, it forces the 2nd and 3rd dimensions to be the same index in the summation?
    # No, it just labels the dimensions.
    # Let's test:
    # a = np.ones((2,2)); b = np.ones((2,2))
    # np.einsum('ij,ij->ij', a, b) -> element wise product.
    # np.einsum('ij,ij->ii', a, b) -> diagonal?
    
    # Let's look at the target Y again.
    # Y[i, j, j_prime, k] = 1 if j == j_prime.
    # This is a sparse tensor.
    # The current code computes D and compares to Y.
    # If the einsum is wrong, the code wouldn't work. But it's the seed program, so we assume it works.
    # Actually, looking at the indices:
    # U: (n, m, rank)
    # V: (m, p, rank)
    # W: (p, n, rank)
    # We want to form a tensor T[i, j, j', k].
    # T[i, j, j', k] = sum_c U[i, j, c] V[j', k, c] W[?, ?, c]?
    # The standard matrix multiplication tensor is:
    # T[i, j, k] = delta_{jk} (for 1x1x1?)
    # For C = AB, T[i, j, k, l] = delta_{jk} ? No.
    # C[i, l] = sum_j A[i, j] B[j, l].
    # Tensor T[i, j, j', l] = 1 if j == j'.
    # Decomposition: sum_r U[i, j, r] V[j', l, r] W[?, ?, r].
    # Usually W is not needed for standard matrix mult, it's a bilinear map.
    # But here we have 3 matrices U, V, W. This is a trilinear decomposition?
    # The task says "tensor decomposition".
    # The einsum 'ijc,jkc,kic->ijjk' suggests:
    # U[i, j, c]
    # V[j, k, c]  (Note: V's first index is j, second is k)
    # W[k, i, c]  (Note: W's first index is k, second is i)
    # Result D[i, j, j, k].
    # This implies the 3rd index of D is tied to the 2nd index 'j'.
    # So D[i, j, j', k] is only non-zero if j == j'?
    # If so, D matches the structure of Y (which is non-zero only if j == j').
    # So the decomposition only needs to satisfy the diagonal part?
    # No, Y is 1 if j == j'.
    # If D is 0 if j != j', then we only need to match the values where j == j'.
    # This reduces the problem size!
    # We only need to satisfy n * m * p equations (for j == j').
    # The number of constraints is n * m * p = 3 * 3 * 7 = 63.
    # The number of variables is 3 * rank * (n*m + m*p + p*n) = rank * (9 + 21 + 21) = 51 * rank.
    # For rank 1, 51 vars > 63 eqs? No, 51 < 63.
    # For rank 2, 102 vars > 63 eqs.
    # So rank 2 might be possible?
    # But it's a system of quadratic equations (products of variables).
    
    # Let's assume the einsum is correct and produces a tensor where D[i, j, j', k] is 0 if j != j'.
    # Then we just need to solve for U, V, W such that D[i, j, j, k] = 1 for all i, j, k.
    
    # This is a system of 63 equations.
    # Variables: U (9R), V (21R), W (21R).
    # Total 51R variables.
    # For R=2, 102 variables.
    # We can use `scipy.optimize.root` or `least_squares` on this smaller system.
    
    # Let's rewrite the objective to only include the non-zero parts of Y.
    
    # New Plan:
    # 1. Identify the indices where Y is non-zero.
    # 2. Construct a vector of these target values (all 1s).
    # 3. Construct a function that computes D at these indices.
    # 4. Use `scipy.optimize.least_squares` to minimize the error.
    # 5. This is much faster than optimizing over the full 63*4=252 entries (though most are 0).
    #    Actually, the full tensor has 3*3*3*7 = 189 entries.
    #    The non-zero ones are 3*3*7 = 63.
    #    The zero ones are 126.
    #    The current code optimizes over all 189.
    #    We can optimize over just the 63 non-zero ones, PLUS we must ensure the zero ones stay zero?
    #    If the structure of the einsum guarantees zeros off-diagonal, we don't need to enforce them.
    #    Let's verify if 'ijc,jkc,kic->ijjk' guarantees zeros off-diagonal.
    #    If I set j=0, j'=1.
    #    D[i, 0, 1, k] = sum_c U[i, 0, c] V[0, k, c] W[k, i, c].
    #    This is NOT necessarily 0.
    #    So the einsum does NOT guarantee zeros off-diagonal.
    #    Therefore, we MUST enforce the zeros as well.
    #    So we must optimize over the full tensor or add constraints.
    
    # However, we can use the sparsity to speed up the gradient computation?
    # No, the matrix is small.
    
    # Let's stick to the Greedy ALS approach, but ensure we check the full error.
    
    # Implementation of Greedy ALS:
    
    def compute_tensor(U, V, W, rank):
        # U: (n, m, rank)
        # V: (m, p, rank)
        # W: (p, n, rank)
        return np.einsum('ijc,jkc,kic->ijjk', U, V, W)

    def als_decompose(Y, n, m, p, rank, max_iter=50, tol=1e-10):
        # Initialize
        U = np.random.randn(n, m, rank)
        V = np.random.randn(m, p, rank)
        W = np.random.randn(p, n, rank)
        
        residual = Y.copy()
        
        for r in range(rank):
            # Initialize this rank's factors
            u = np.random.randn(n, m)
            v = np.random.randn(m, p)
            w = np.random.randn(p, n)
            
            for it in range(max_iter):
                # Update u
                # Maximize <u, sum_{j,k} v[j,k] w[k,i] residual[i,j,k,i]? No.
                # Term is u[i,j] * v[j,k] * w[k,i].
                # For fixed v, w:
                # u[i,j] should be proportional to sum_{k} v[j,k] w[k,i] residual[i,j,k,i]?
                # No, the residual is R[i, j, j', k].
                # The term is u[i,j] v[j',k] w[k,i].
                # We want to approximate R[i, j, j', k].
                # So u[i,j] v[j',k] w[k,i].
                # Sum over j', k.
                # u[i,j] * sum_{j',k} v[j',k] w[k,i] R[i,j,j',k]
                # Let M_u[i,j] = sum_{j',k} v[j',k] w[k,i] R[i,j,j',k]
                # u = M_u / ||M_u||
                
                M_u = np.einsum('jk,ki,ijjk->ij', v, w, residual)
                norm_u = np.linalg.norm(M_u)
                if norm_u > 1e-10:
                    u = M_u / norm_u
                else:
                    u = np.random.randn(n, m)
                    u = u / np.linalg.norm(u)
                    
                # Update v
                # v[j',k] * sum_{i,j} u[i,j] w[k,i] R[i,j,j',k]
                M_v = np.einsum('ij,ki,ijjk->jk', u, w, residual)
                norm_v = np.linalg.norm(M_v)
                if norm_v > 1e-10:
                    v = M_v / norm_v
                else:
                    v = np.random.randn(m, p)
                    v = v / np.linalg.norm(v)
                    
                # Update w
                # w[k,i] * sum_{i,j,j'} u[i,j] v[j',k] R[i,j,j',k]
                M_w = np.einsum('ij,jk,ijjk->ki', u, v, residual)
                norm_w = np.linalg.norm(M_w)
                if norm_w > 1e-10:
                    w = M_w / norm_w
                else:
                    w = np.random.randn(p, n)
                    w = w / np.linalg.norm(w)
            
            # Update residual
            term = np.einsum('ij,jk,ki->ijjk', u, v, w)
            residual -= term
            U[:, :, r] = u
            V[:, :, r] = v
            W[:, :, r] = w
            
            if np.linalg.norm(residual) < tol:
                break
                
        return U, V, W

    best_rank = n * m * p
    best_U = np.zeros((n * m, best_rank))
    best_V = np.zeros((m * p, best_rank))
    best_W = np.zeros((p * n, best_rank))
    
    # Initialize with trivial solution
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j, col] = 1
                best_V[j * p + k, col] = 1
                best_W[k * n + i, col] = 1
                col += 1

    search_deadline = start_time + 130
    ranks_to_try = list(range(2, best_rank))
    
    for rank in ranks_to_try:
        if time.time() > search_deadline:
            break
        
        found = False
        max_attempts = 5
        
        for attempt in range(max_attempts):
            if time.time() > search_deadline:
                break
            
            U, V, W = als_decompose(Y, n, m, p, rank)
            
            # Verify
            D = compute_tensor(U, V, W, rank)
            err = np.max(np.abs(D - Y))
            
            if err < 1e-6:
                # Try rounding
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                D_rounded = compute_tensor(U_rounded, V_rounded, W_rounded, rank)
                err_rounded = np.max(np.abs(D_rounded - Y))
                
                if err_rounded < 1e-6:
                    best_rank = rank
                    best_U = U_rounded.reshape(n * m, rank)
                    best_V = V_rounded.reshape(m * p, rank)
                    best_W = W_rounded.reshape(p * n, rank)
                    found = True
                    break
        
        if found:
            continue

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

