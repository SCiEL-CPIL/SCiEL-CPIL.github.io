import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    # Target Tensor T is defined by T[i, j, j', k', k'', i''] = 1 if j==j', k==k'', i==i''
    # We represent T implicitly.
    # We want to minimize || sum_c U_c o V_c o W_c - T ||^2
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    best_U, best_V, best_W, best_rank = schoolbook()

    # Try to find a lower rank solution
    # Lower bound is max(n*m, m*p, p*n) = 30
    # Try ranks from 35 to 59
    
    for rank in range(35, 60):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Simple Gradient Descent / ALS
            # We compute the residual R = T - Approx
            # To save time, we only update if we can compute the gradient efficiently.
            # Given the small size, we can compute the full tensor approximation.
            
            def compute_approx(U, V, W):
                # Approx_{i,j,j',k',k'',i''} = sum_c U[i,j,c] V[j',k',c] W[k'',i'',c]
                # This is a 6D tensor.
                # We can reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
                # Approx = np.einsum('ijc,jkc,kic->ijk', U.reshape(n,m,rank), V.reshape(m,p,rank), W.reshape(p,n,rank))
                # Wait, indices:
                # U: i, j, c
                # V: j', k', c -> must match j with j'?
                # The tensor T has indices (i, j, j', k', k'', i'') with constraints j=j', k'=k'', i''=i.
                # So we sum over c.
                # Term is U[i,j,c] * V[j,k,c] * W[k,i,c].
                # This results in a tensor A[i,k] = sum_{j,c} U[i,j,c] V[j,k,c] W[k,i,c]?
                # No, the decomposition is of the tensor T.
                # T[i,j,j',k',k'',i''] = delta...
                # The approximation is A[i,j,j',k',k'',i''] = sum_c U[i,j,c] V[j',k',c] W[k'',i'',c].
                # We want A to match T.
                # T is non-zero only when j=j', k'=k'', i''=i.
                # So we only care about the diagonal entries in the j,j', k',k'', i'',i dimensions?
                # No, T is zero elsewhere. So we must ensure A is zero elsewhere too.
                
                # Efficient check:
                # For all i, k:
                # A[i, j, j, k, k, i] should be 1.
                # A[i, j, j', k', k'', i''] should be 0 if any index mismatch.
                
                # Let's compute the "matched" part:
                # M[i,k] = sum_{j,c} U[i,j,c] V[j,k,c] W[k,i,c]
                # This should be 1 for all i,k.
                
                # Let's compute the "mismatched" parts.
                # This is hard to do efficiently without constructing the full 6D tensor.
                # However, n=2, m=5, p=6.
                # Total size 2*5*5*6*6*2 = 3600 entries.
                # rank ~ 40.
                # Computing full tensor is fast.
                
                U_r = U.reshape(n, m, rank)
                V_r = V.reshape(m, p, rank)
                W_r = W.reshape(p, n, rank)
                
                # Approx_{i, j, j', k, k', i'} = sum_c U[i,j,c] V[j',k,c] W[k',i',c]
                # Note: V index is j', k. W index is k', i'.
                # We want this to be 1 if j=j', k=k', i'=i. Else 0.
                
                # We can compute the full 6D tensor using einsum
                # 'ijc,jkc,kic->ijjkki' ? No, indices are distinct in the output.
                # Output indices: i, j, j', k, k', i'
                # U: i, j, c
                # V: j', k, c
                # W: k', i', c
                # Sum over c.
                approx = np.einsum('ijc,jkc,kic->ijjkki', U_r, V_r, W_r)
                
                # Construct Target T (6D)
                # T[i, j, j', k, k', i'] = 1 if j==j', k==k', i==i' else 0
                # We can create this by broadcasting.
                # T = (j==j') & (k==k') & (i==i')
                # Let's create index arrays
                ii = np.arange(n)
                jj = np.arange(m)
                kk = np.arange(p)
                
                # T shape (n, m, m, p, p, n)
                # T[i, j, j', k, k', i']
                # We can build it:
                # t1 = np.zeros((n, m, m, p, p, n))
                # for i in range(n):
                #   for j in range(m):
                #     for k in range(p):
                #       t1[i, j, j, k, k, i] = 1
                # This is slow in Python loop, but n,m,p are small.
                # Vectorized:
                # We can use broadcasting to create the mask.
                # i_idx = np.arange(n)[:, None, None, None, None, None]
                # j_idx = np.arange(m)[None, :, None, None, None, None]
                # jp_idx = np.arange(m)[None, None, :, None, None, None]
                # k_idx = np.arange(p)[None, None, None, :, None, None]
                # kp_idx = np.arange(p)[None, None, None, None, :, None]
                # ip_idx = np.arange(n)[None, None, None, None, None, :]
                
                # mask = (j_idx == jp_idx) & (k_idx == kp_idx) & (i_idx == ip_idx)
                # T = mask.astype(float)
                
                # Compute Error
                error = np.sum((approx - best_T)**2) if 'best_T' in locals() else 0
                
                return error, approx

            # Pre-compute T
            T = np.zeros((n, m, m, p, p, n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, j, k, k, i] = 1.0

            # Optimize
            lr = 0.01
            for step in range(100):
                if time.time() > deadline:
                    break
                    
                U_r = U.reshape(n, m, rank)
                V_r = V.reshape(m, p, rank)
                W_r = W.reshape(p, n, rank)
                
                approx = np.einsum('ijc,jkc,kic->ijjkki', U_r, V_r, W_r)
                residual = approx - T
                
                # Gradient for U:
                # dE/dU[i,j,c] = 2 * sum_{j',k,k',i'} residual[i,j,j',k,k',i'] * V[j',k,c] * W[k',i',c]
                # We can compute this with einsum.
                # grad_U[i,j,c] = 2 * np.einsum('ijjkki,jkc,kic->ijc', residual, V_r, W_r)
                # Note: V_r is (m, p, rank). V[j',k,c].
                # W_r is (p, n, rank). W[k',i',c].
                
                grad_U = 2.0 * np.einsum('ijjkki,jkc,kic->ijc', residual, V_r, W_r)
                grad_V = 2.0 * np.einsum('ijjkki,ijc,kic->jkc', residual, U_r, W_r)
                grad_W = 2.0 * np.einsum('ijjkki,ijc,jkc->kic', residual, U_r, V_r)
                
                U -= lr * grad_U.reshape(n*m, rank)
                V -= lr * grad_V.reshape(m*p, rank)
                W -= lr * grad_W.reshape(p*n, rank)
                
                if step % 10 == 0:
                    err = np.sum(residual**2)
                    if err < 1e-6:
                        break

            # Check if solution is close to integer/half-integer
            # Round to nearest 0.5
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify exactness
            U_r = U_rounded.reshape(n, m, rank)
            V_r = V_rounded.reshape(m, p, rank)
            W_r = W_rounded.reshape(p, n, rank)
            approx = np.einsum('ijc,jkc,kic->ijjkki', U_r, V_r, W_r)
            
            if np.allclose(approx, T, atol=1e-9):
                best_U = U_rounded
                best_V = V_rounded
                best_W = W_rounded
                best_rank = rank
                break # Found a solution for this rank, try lower? No, we are iterating up.
                # Actually we iterate rank from 35 up. So first found is minimal.
                return {
                    "U": best_U.tolist(),
                    "V": best_V.tolist(),
                    "W": best_W.tolist()
                }

    # If no better solution found, return schoolbook
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
