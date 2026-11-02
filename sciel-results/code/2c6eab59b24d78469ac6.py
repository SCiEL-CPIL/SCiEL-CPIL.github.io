import numpy as np
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin

    n, m, p = 3, 5, 7
    trivial_rank = n * m * p  # 105

    # Build the trivial decomposition first as a safe fallback
    U_triv = np.zeros((n * m, trivial_rank))
    V_triv = np.zeros((m * p, trivial_rank))
    W_triv = np.zeros((p * n, trivial_rank))
    
    # The tensor T_{i,j,k} = 1 if j matches, else 0.
    # Actually, the multiplication tensor is defined such that 
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The bilinear form is sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k} * T_{i,j,k}?
    # No, the tensor entries are 1 at (i,j,k) if the term A_{i,j}B_{j,k} contributes to C_{i,k}.
    # So T_{i,j,k} is 1 if the index k in the third mode corresponds to the output index?
    # Let's define the tensor T with dimensions (n, m, p) ??
    # Standard convention: T has indices (i, j, k) where i is row of A, j is col of A / row of B, k is col of B.
    # But the output is C_{i,k}.
    # The decomposition is C_{i,k} = sum_r U_{i,r} V_{k,r} ? No.
    # The interface says:
    # U: (n*m, rank) -> indexed by (i, j) ?
    # V: (m*p, rank) -> indexed by (j, k) ?
    # W: (p*n, rank) -> indexed by (k, i) ?
    # And T_{i,j,k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
    
    # Let's verify the trivial case.
    # If U_{i*m+j, r} = 1, V_{j*p+k, r} = 1, W_{k*n+i, r} = 1 for a specific (i,j,k)
    # Then sum_r ... = 1 for that specific (i,j,k) tuple.
    # This matches the tensor that is 1 at (i,j,k) and 0 elsewhere.
    # This tensor represents the multiplication?
    # Yes, because C_{i,k} = sum_j A_{i,j} B_{j,k} * 1.
    
    # So the target tensor T is the "identity" tensor in the middle dimension?
    # T_{i,j,k} = 1 if the "flow" goes from i to j to k?
    # No, the multiplication tensor is non-zero at (i, j, k) for ALL i, j, k?
    # Wait.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The bilinear expansion is sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k}.
    # The coefficient of A_{i,j} B_{j,k} in the expression for C_{i,k} is 1.
    # So the tensor T has entries T_{i,j,k} = 1 for all i,j,k?
    # No, that would be the rank 1 tensor $1 \otimes 1 \otimes 1$?
    # No, the indices are coupled.
    # The term is $A_{i,j} B_{j,k} C_{i,k}$.
    # This term appears in the sum for the output element $C_{i,k}$.
    # So for a fixed output $(i,k)$, the input terms are $(i,j)$ and $(j,k)$.
    # The tensor is a 3-way array.
    # $T_{i,j,k} = 1$ if the indices are consistent?
    # Actually, the tensor is defined by $T_{i,j,k} = \delta_{j, \text{something}}$?
    # No.
    # Let's look at the trivial decomposition code provided in the seed:
    # U[i * m + j][col] = 1
    # V[j * p + k][col] = 1
    # W[k * n + i][col] = 1
    # This sets up a rank-1 tensor that is 1 at (i, j, k) and 0 elsewhere.
    # Summing over all (i,j,k) gives a tensor that is 1 everywhere?
    # If T_{i,j,k} = 1 for all i,j,k, then the rank is 1?
    # No, the sum is over r.
    # If we have one term for each (i,j,k), then the tensor is the sum of basis tensors.
    # The basis tensor $E_{i,j,k}$ is 1 at (i,j,k) and 0 elsewhere.
    # The multiplication tensor is $\sum_{i,j,k} E_{i,j,k}$?
    # No.
    # The multiplication tensor $M$ satisfies $M(A,B) = A B$.
    # $M_{i,j,k}$ is the coefficient of $A_{i,j} B_{j,k}$ in $C_{i,k}$.
    # For a fixed $(i,k)$, $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
    # So the coefficient of $A_{i,j} B_{j,k}$ is 1.
    # This coefficient is associated with the output index $(i,k)$.
    # So the tensor is defined on indices $(i, j, k)$?
    # Usually, the tensor is $T_{i, j, k}$ where $i$ is row of A, $j$ is col of A, $k$ is col of B.
    # But $k$ is also the col of C.
    # The "third mode" of the tensor usually corresponds to the output index.
    # So $T$ has dimensions $(n \times m) \times (m \times p) \times (n \times p)$?
    # No, the interface defines U, V, W shapes as $(n*m, r)$, $(m*p, r)$, $(p*n, r)$.
    # This implies the tensor lives in the space of triple products of these vectors.
    # The target tensor $T$ must be such that $\sum_r U_{:,r} V_{:,r} W_{:,r}^T = T$.
    # Here $T$ is a tensor of size $(n*m) \times (m*p) \times (p*n)$.
    # Entry $T_{(i,j), (j',k), (k',i')}$ is 1 if $i=i', j=j', k=k'$?
    # Let's check the trivial code:
    # Col r corresponds to a specific $(i,j,k)$.
    # $U_{i*m+j, r} = 1$.
    # $V_{j*p+k, r} = 1$.
    # $W_{k*n+i, r} = 1$.
    # So the term is $e_{(i,j)} \otimes e_{(j,k)} \otimes e_{(k,i)}$.
    # Summing over all $(i,j,k)$, we get a tensor that is 1 at $( (i,j), (j,k), (k,i) )$.
    # Is this the multiplication tensor?
    # Yes. The multiplication map is linear.
    # $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
    # This can be written as $\sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k} \cdot \mathbb{1}$.
    # The tensor $T$ has $T_{(i,j), (j,k), (k,i)} = 1$ and 0 otherwise.
    
    # So we need to decompose this specific sparse tensor.
    
    # Target Tensor T_flat
    # Size: (n*m) * (m*p) * (p*n) = 15 * 35 * 21 = 11025.
    # Non-zeros: 105.
    
    # We can try to find a lower rank decomposition of this sparse tensor.
    
    # Heuristic: Randomized SVD / ALS
    # Since the tensor is sparse, we can use the non-zero entries to guide the search.
    
    best_rank = trivial_rank
    best_U = U_triv
    best_V = V_triv
    best_W = W_triv
    
    # Try ranks from 104 down to 50
    # This is a heuristic search.
    
    for rank in range(trivial_rank - 1, 49, -1):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for attempt in range(10):
            if time.time() > deadline:
                break
                
            # Random init
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # ALS Iterations
            # Minimize || T - sum U_r V_r W_r^T ||
            # This is equivalent to finding low rank approximation.
            
            # Since T is sparse, we can project the error onto the support of T?
            # Or just run standard ALS on the full tensor (padded with zeros).
            # Full tensor size 11025 is small.
            
            T_full = np.zeros((n * m, m * p, p * n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx_u = i * m + j
                        idx_v = j * p + k
                        idx_w = k * n + i
                        T_full[idx_u, idx_v, idx_w] = 1.0
            
            # ALS
            for it in range(50):
                # Update U
                # T_{:, :, k} = U * (V * W_k)^T ?
                # T = sum_r U_r V_r W_r^T
                # Fix V, W. Solve for U.
                # T_{a, b, c} = sum_r U_{a,r} V_{b,r} W_{c,r}
                # For fixed a: T_{a, :, :} = sum_r U_{a,r} (V_{:,r} \otimes W_{:,r})^T ?
                # No, V_{b,r} W_{c,r}.
                # Let M = V \otimes W (size (m*p)*(p*n) x rank).
                # T_{a, b, c} = sum_r U_{a,r} M_{(b,c), r}
                # So T_{a, :, :} (flattened) = U_{a, :} M^T
                # U_{a, :} = T_{a, flat} M^+
                
                M_vw = np.kron(V, W) # (m*p*p*n) x rank ? No.
                # V is (m*p) x rank. W is (p*n) x rank.
                # We need M_{b,c,r} = V_{b,r} W_{c,r}.
                # This is the element-wise product of V and W, flattened?
                # M = (V * W) ? No, dimensions don't match for simple element-wise.
                # M_{(b,c), r} = V_{b,r} W_{c,r}.
                # This is the Khatri-Rao product of V and W?
                # No, Kronecker?
                # If we flatten b and c, the matrix M has rows indexed by (b,c).
                # M = V[:, r] \otimes W[:, r] stacked?
                # Actually, M = np.kron(V.T, W.T).T ?
                # V.T is rank x (m*p). W.T is rank x (p*n).
                # np.kron(V.T, W.T) is rank*rank x (m*p*p*n).
                # We want rank x (m*p*p*n).
                # M_{r, (b,c)} = V_{b,r} W_{c,r}.
                # This is exactly the Kronecker product of V.T and W.T, but transposed?
                # Let's construct M explicitly for speed (small sizes).
                # Or use broadcasting.
                
                # T_flat_a = T_full[a, :, :].flatten() (size (m*p)*(p*n))
                # M_r = V[:, r] * W[:, r] (broadcasted to (m*p, p*n))
                # M = stack(M_r) -> (rank, (m*p)*(p*n))
                
                M = np.zeros((rank, (m * p) * (p * n)))
                for r in range(rank):
                    M[r, :] = (V[:, r] * W[:, r]).flatten()
                
                # Solve U = T @ M.T @ pinv(M @ M.T)
                # U (n*m x rank)
                # T (n*m x (m*p)*(p*n))
                
                T_flat = T_full.reshape(n * m, -1)
                
                # Normal equations: U M M^T = T M^T
                # U = T M^T (M M^T)^{-1}
                
                MMt = M @ M.T
                T_Mt = T_flat @ M.T
                try:
                    U = np.linalg.solve(MMt, T_Mt.T).T
                except:
                    U = np.linalg.lstsq(MMt, T_Mt.T, rcond=None)[0].T
                
                # Update V
                M_wu = np.zeros((rank, (p * n) * (n * m)))
                for r in range(rank):
                    M_wu[r, :] = (W[:, r] * U[:, r]).flatten()
                
                T_flat_v = T_full.transpose(1, 2, 0).reshape(m * p, -1) # (m*p, p*n*n*m)
                
                MMt = M_wu @ M_wu.T
                T_Mt = T_flat_v @ M_wu.T
                try:
                    V = np.linalg.solve(MMt, T_Mt.T).T
                except:
                    V = np.linalg.lstsq(MMt, T_Mt.T, rcond=None)[0].T
                    
                # Update W
                M_uv = np.zeros((rank, (n * m) * (m * p)))
                for r in range(rank):
                    M_uv[r, :] = (U[:, r] * V[:, r]).flatten()
                
                T_flat_w = T_full.transpose(2, 0, 1).reshape(p * n, -1) # (p*n, n*m*m*p)
                
                MMt = M_uv @ M_uv.T
                T_Mt = T_flat_w @ M_uv.T
                try:
                    W = np.linalg.solve(MMt, T_Mt.T).T
                except:
                    W = np.linalg.lstsq(MMt, T_Mt.T, rcond=None)[0].T
            
            # Check residual
            # Reconstruct T_rec
            # T_rec[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # Use einsum
            T_rec = np.einsum('ar,br,cr->abc', U, V, W)
            
            err = np.linalg.norm(T_rec - T_full)
            
            if err < 1e-9:
                # Snap to half-integers
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                # Verify exactness with integers
                T_rec_int = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
                if np.allclose(T_rec_int, T_full, atol=1e-9):
                    best_rank = rank
                    best_U = U_int
                    best_V = V_int
                    best_W = W_int
                    # Convert to lists of Fractions or floats?
                    # The interface accepts integer/half-integer entries.
                    # We can return lists of floats or Fractions.
                    # Let's return lists of floats for simplicity, or Fractions for exactness.
                    # The seed used lists of ints.
                    
                    # Break out of loops
                    return {
                        "U": best_U.tolist(),
                        "V": best_V.tolist(),
                        "W": best_W.tolist()
                    }
            
    # If no better rank found, return trivial
    return {
        "U": U_triv.tolist(),
        "V": V_triv.tolist(),
        "W": W_triv.tolist()
    }
