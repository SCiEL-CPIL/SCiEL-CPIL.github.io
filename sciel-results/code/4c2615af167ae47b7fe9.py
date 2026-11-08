"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145
    n, m, p = DIMENSIONS  # 2, 3, 10
    
    # Strategy: decompose 2x3x10 as 2 blocks of 2x3x5
    # Find a low-rank decomposition for 2x3x5, then use 2 copies
    # 
    # For 2x3x5, let's try to find rank r decomposition
    # A is 2x3, B is 3x5, C is 2x5
    # 
    # We'll use a structured approach:
    # Let A = [[a00,a01,a02],[a10,a11,a12]]
    # Let B = [[b00,b01,b02,b03,b04],[b10,b11,b12,b13,b14],[b20,b21,b22,b23,b24]]
    # C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # Rank 1 tensors: for each (i,j,k), we have A[i,j]*B[j,k] contributing to C[i,k]
    # Trivial rank = 2*3*5 = 30
    #
    # Let's try to use Strassen-like ideas to reduce rank
    # 
    # Alternative: use the known result that M_{2,3,5} might have rank ~24-26
    # 
    # Let me try a direct construction. I'll use the following approach:
    # Group the 5 columns of B into patterns that allow sharing.
    
    # Actually, let me try a more systematic approach.
    # I'll construct a rank-r decomposition for 2x3x5 with r as small as possible.
    
    # For now, let's use a block decomposition with 2x3x2 (rank 11) and 2x3x3 (rank 17?)
    # 2x3x10 = 2x3x2 + 2x3x2 + 2x3x2 + 2x3x2 + 2x3x2 = 5*11 = 55 (current)
    # Or: 2x3x10 = 2x3x3 + 2x3x3 + 2x3x4
    # If M_{2,3,3} has rank 17 and M_{2,3,4} has rank 22, total = 17+17+22 = 56 (worse)
    
    # Let me try to find a better decomposition for 2x3x5
    # 
    # I'll use a heuristic search to find a rank decomposition for 2x3x5
    # with integer/half-integer entries.
    
    # The tensor T[i,j,k] = delta_{j,j} for the multiplication
    # For 2x3x5: T is a 2x3x5 tensor where T[i,j,k] = 1 if the j-th column of B 
    # contributes, i.e., T[i,j,k] = 1 for all i,k when we're at position j
    # Wait, no. The multiplication tensor is:
    # T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B multiply to give (i,k) of C
    # So T[i,j,k] = 1 for all valid i,j,k (it's the "identity" in the j dimension)
    
    # Actually, the matrix multiplication tensor M_{n,m,p} is defined as:
    # M = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p e_i^A e_j^B e_k^C * delta
    # No, that's not right either.
    
    # Let me be precise. The multiplication tensor T in the space A* ⊗ B* ⊗ C is:
    # T(a,b) = a*b (matrix product)
    # In coordinates: T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B 
    # contributes to (i,k) entry of C. Since C[i,k] = sum_j A[i,j] B[j,k],
    # we have T[i,j,k] = 1 for all i,j,k (each term A[i,j]B[j,k] appears exactly once).
    
    # So T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,1,2,3,4}
    # This is a 2x3x5 all-ones tensor? No!
    
    # Wait, I'm confusing myself. Let me re-read the problem.
    # "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor for M_{n,m,p} is:
    # T = sum_{i,j,k} x_{i,j}^A x_{j,k}^B y_{i,k}^C where the coefficient is 1
    # But that's not a tensor decomposition, that's the bilinear map.
    
    # The tensor itself, as an element of (A*) ⊗ (B*) ⊗ C, is:
    # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i^A ⊗ e_j^B ⊗ e_k^C * δ_{j,j}
    # Hmm, this is getting confusing.
    
    # Let me just use the standard definition: the multiplication tensor M_{n,m,p} 
    # satisfies that for any A (n×m) and B (m×p), the product C = AB satisfies
    # C[i,k] = <T, A ⊗ B ⊗ e_i ⊗ e_k>... no.
    
    # Standard: M_{n,m,p} is the tensor such that 
    # (M_{n,m,p})(A,B) = AB
    # In terms of tensor rank, we want to write M as a sum of rank-1 tensors.
    # A rank-1 tensor in A* ⊗ B* ⊗ C is u ⊗ v ⊗ w, which corresponds to
    # C[i,k] += u[i] * v[k] * w[i,k]... no.
    
    # Let me use the interface: U (n*m × rank), V (m*p × rank), W (p*n × rank)
    # The decomposition says: for each rank index r, we have
    # U[:,r] is a vector in R^{n*m}, V[:,r] is a vector in R^{m*p}, W[:,r] is a vector in R^{p*n}
    # And the condition is that for all A (n×m) and B (m×p):
    # sum_r (U[:,r] · vec(A)) * (V[:,r] · vec(B)) = W[:,r] · vec(C)
    # where vec(C) uses the convention W[k*n+i, r] for C[i,k]
    
    # So: sum_r U_r(i,j) * A[i,j] * V_r(j,k) * B[j,k] = W_r(k*n+i) * C[i,k]
    # But C[i,k] = sum_j A[i,j] B[j,k]
    # So we need: sum_r U_r(i,j) V_r(j,k) W_r(k*n+i) = 1 for all i,j,k
    # Wait, that's not quite right either. Let me re-derive.
    
    # The bilinear form is: f(A,B) = vec(C)^T * something
    # Actually, the condition for the tensor decomposition is:
    # For all A, B: C = AB iff for all i,k: C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T satisfies: T[i,j,k] = 1 (the coefficient of A[i,j] B[j,k] in C[i,k])
    # 
    # A rank-1 decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r ∈ R^{n*m}, v_r ∈ R^{m*p}, w_r ∈ R^{p*n}
    # The condition is: T[i,j,k] = sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
    # for all i,j,k.
    # 
    # Since T[i,j,k] = 1 for all i,j,k, we need:
    # sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1 for all i,j,k
    
    # This is the key equation. Let me verify with the current code.
    # In the current code, for the 2x3x2 block:
    # U is 6x11, V is 6x11, W is 4x11
    # The condition is sum_r U[i*3+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1 for all i,j,k
    
    # OK so I need to find U, V, W such that this holds.
    
    # Let me try to find a low-rank solution for 2x3x5 first.
    # 
    # I'll use a numerical optimization approach.
    
    n2, m2, p2 = 2, 3, 5  # for the sub-problem
    # Try ranks from 20 upward
    for rank in range(20, 26):
        if time.time() > deadline - 10:
            break
        # Use random initialization and gradient descent
        np.random.seed(42)
        U = np.random.randn(n2*m2, rank)
        V = np.random.randn(m2*p2, rank)
        W = np.random.randn(p2*n2, rank)
        
        # We want: for all i,j,k: sum_r U[i*m2+j, r] * V[j*p2+k, r] * W[k*n2+i, r] = 1
        # This is a constraint satisfaction problem. Let's minimize the sum of squares of (expr - 1)^2
        
        def loss_fn(params):
            U_flat, V_flat, W_flat = params
            U = U_flat.reshape(n2*m2, rank)
            V = V_flat.reshape(m2*p2, rank)
            W = W_flat.reshape(p2*n2, rank)
            total_loss = 0.0
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        s = 0.0
                        for r in range(rank):
                            s += U[i*m2+j, r] * V[j*p2+k, r] * W[k*n2+i, r]
                        total_loss += (s - 1.0)**2
            return total_loss
        
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Simple gradient descent
        lr = 0.01
        for step in range(500):
            if time.time() > deadline - 5:
                break
            # Compute gradient numerically or analytically
            # Analytical gradient:
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        s = 0.0
                        for r in range(rank):
                            s += U[i*m2+j, r] * V[j*p2+k, r] * W[k*n2+i, r]
                        err = 2.0 * (s - 1.0)
                        for r in range(rank):
                            grad_U[i*m2+j, r] += err * V[j*p2+k, r] * W[k*n2+i, r]
                            grad_V[j*p2+k, r] += err * U[i*m2+j, r] * W[k*n2+i, r]
                            grad_W[k*n2+i, r] += err * U[i*m2+j, r] * V[j*p2+k, r]
            
            params -= lr * np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            U = params[:n2*m2*rank].reshape(n2*m2, rank)
            V = params[n2*m2*rank:n2*m2*rank+m2*p2*rank].reshape(m2*p2, rank)
            W = params[n2*m2*rank+m2*p2*rank:].reshape(p2*n2, rank)
            
            if step % 50 == 0:
                loss = loss_fn(params)
                if loss < 1e-6:
                    break
        
        loss = loss_fn(params)
        if loss < 1e-4:
            # Found a solution! Now try to round to integers/half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify
            max_err = 0
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        s = 0.0
                        for r in range(rank):
                            s += U_rounded[i*m2+j, r] * V_rounded[j*p2+k, r] * W_rounded[k*n2+i, r]
                        max_err = max(max_err, abs(s - 1.0))
            
            if max_err < 1e-6:
                # Great! Use this for the full 2x3x10
                # We have 2 blocks of 5
                total_rank = 2 * rank
                U_full = np.zeros((n*m, total_rank))
                V_full = np.zeros((m*p, total_rank))
                W_full = np.zeros((p*n, total_rank))
                
                for block in range(2):
                    offset = block * rank
                    for i in range(n2):
                        for j in range(m2):
                            U_full[i*m+j, offset:offset+rank] = U_rounded[i*m2+j, :]
                    for j in range(m2):
                        for k in range(p2):
                            k_global = block * p2 + k
                            V_full[j*p+k_global, offset:offset+rank] = V_rounded[j*p2+k, :]
                    for k in range(p2):
                        for i in range(n2):
                            k_global = block * p2 + k
                            W_full[k_global*n+i, offset:offset+rank] = W_rounded[k*n2+i, :]
                
                # Convert to lists
                U_list = U_full.tolist()
                V_list = V_full.tolist()
                W_list = W_full.tolist()
                
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: use the original 55-rank solution
    num_blocks = p // 2
    block_size = 2
    rank_per_block = 11
    total_rank = num_blocks * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    strassen_terms = [
        {'U': [(0,0,1), (1,1,1)], 'V': [(0,0,1), (1,1,1)], 'W': [(0,0,1), (1,1,1)]},
        {'U': [(1,0,1), (1,1,1)], 'V': [(0,0,1)], 'W': [(0,1,1), (1,1,-1)]},
        {'U': [(0,0,1)], 'V': [(0,1,1), (1,1,-1)], 'W': [(1,0,1), (1,1,1)]},
        {'U': [(1,1,1)], 'V': [(1,0,1), (0,0,-1)], 'W': [(0,0,1), (0,1,1)]},
        {'U': [(0,0,1), (0,1,1)], 'V': [(1,1,1)], 'W': [(0,0,-1), (1,0,1)]},
        {'U': [(1,0,1), (0,0,-1)], 'V': [(0,0,1), (0,1,1)], 'W': [(1,1,1)]},
        {'U': [(0,1,1), (1,1,-1)], 'V': [(1,0,1), (1,1,1)], 'W': [(0,0,1)]}
    ]
    
    outer_terms = [
        {'U': [(0,2,1)], 'V': [(2,0,1)], 'W': [(0,0,1)]},
        {'U': [(0,2,1)], 'V': [(2,1,1)], 'W': [(1,0,1)]},
        {'U': [(1,2,1)], 'V': [(2,0,1)], 'W': [(0,1,1)]},
        {'U': [(1,2,1)], 'V': [(2,1,1)], 'W': [(1,1,1)]}
    ]
    
    all_terms = strassen_terms + outer_terms
    
    col_offset = 0
    for b in range(num_blocks):
        for term in all_terms:
            col = col_offset
            for i, j, val in term['U']:
                U[i * m + j][col] += val
            for j, k_local, val in term['V']:
                k_global = b * block_size + k_local
                V[j * p + k_global][col] += val
            for k_local, i, val in term['W']:
                k_global = b * block_size + k_local
                W[k_global * n + i][col] += val
            col_offset += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
