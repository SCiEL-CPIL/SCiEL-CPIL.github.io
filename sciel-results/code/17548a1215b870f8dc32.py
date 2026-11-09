import numpy as np
import time
import sys

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Target ranks to try: start low, go up. Schoolbook is 32.
    # Strassen for 2x2 is 7. 2x4x4 can be seen as 2x2x2 blocks.
    # Rank 28 is a strong candidate (4 * 7).
    # Rank 24, 20, 16 are possible but harder.
    candidate_ranks = [16, 20, 24, 28, 32]
    
    best_rank = None
    best_factors = None
    
    # Precompute the multiplication tensor T[i, j, k] = delta_{i,k}
    # Actually, the tensor is defined such that C[i,k] = sum_j A[i,j] B[j,k]
    # The bilinear form is sum_{i,j,k} C[i,k] A[i,j] B[j,k]
    # The tensor T has indices (i, j, k) where T[i,j,k] = 1 if k==i? No.
    # T_{i,j,k} is the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
    # So T_{i,j,k} = 1 if the output index matches the input index?
    # Let's stick to the matrix form.
    # U (n*m, r), V (m*p, r), W (p*n, r)
    # C = (U * A) * (V * B) * W^T ? No.
    # Standard decomposition:
    # C_{i,k} = sum_r (sum_j U_{i,j,r} A_{i,j}) (sum_l V_{j,l,r} B_{j,l}) W_{k,i,r} ?
    # Let's use the explicit definition from the task:
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # C[i,k] = sum_j sum_l A[i,j] B[j,l] * (coefficient)
    # The product is sum_c (sum_j U[i*m+j, c] A[i,j]) * (sum_l V[j*p+l, c] B[j,l]) * W[k*n+i, c]
    # Wait, the indices in V are j*p+k. So V depends on j.
    # This implies the sum over j is inside the V term?
    # No, the standard tensor decomposition for matrix multiplication C = AB is:
    # C_{i,k} = sum_{j=1}^m A_{i,j} B_{j,k}
    # This is a sum over j.
    # A rank-1 term in the tensor decomposition corresponds to a product of 3 vectors.
    # T_{i,j,k} = 1 if we are computing C_{i,k} from A_{i,j} and B_{j,k}?
    # Actually, the multiplication tensor M has entries M_{i,j,k} = 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # In the standard basis, M_{i,j,k} is non-zero only if the indices align.
    # Specifically, for fixed i, j, k, the term A_{i,j} B_{j,k} contributes to C_{i,k} with weight 1.
    # So the tensor T has T[i, j, k] = 1? No, that's not a tensor in the right space.
    
    # Let's use the matrix factorization view which is equivalent.
    # We want to minimize rank r such that there exist U, V, W
    # U: (n*m) x r
    # V: (m*p) x r
    # W: (p*n) x r
    # such that for all A, B:
    # C = A B
    # C_{i,k} = sum_{j} A_{i,j} B_{j,k}
    
    # The decomposition says:
    # C_{i,k} = sum_{c=1}^r (sum_{j} U[i*m+j, c] A[i,j]) * (sum_{j} V[j*p+k, c] B[j,k]) * W[k*n+i, c]
    # Wait, V index is j*p+k. So the second factor is sum_j V[j*p+k, c] B[j,k].
    # This matches the structure of matrix multiplication where we sum over j.
    
    # Let's define the error function.
    # We can sample random A, B and check if the reconstructed C matches.
    # Or better, use the algebraic identity.
    
    # For a fixed r, we want to find U, V, W such that:
    # sum_c U_{i*m+j, c} V_{j'*p+k, c} W_{k'*n+i', c} = delta_{i,i'} delta_{j,j'} delta_{k,k'} ?
    # No, the tensor T_{i,j,k} for multiplication is:
    # T_{i,j,k} = 1 if we are looking at the term A_{i,j} B_{j,k} -> C_{i,k}.
    # The indices of the tensor are (i, j, k) for inputs A, B and output C?
    # Usually, the multiplication tensor is T_{i,j,k} where i indexes A, j indexes B, k indexes C?
    # No, T_{i,j,k} is the coefficient of A_i B_j in C_k.
    # Here A has indices (i, j), B has (j, k), C has (i, k).
    # So the "tensor" is T[(i,j), (j',k), (i',k')] = 1 if i=i', j=j', k=k'.
    # Wait, the sum is over j. So A_{i,j} and B_{j,k} produce C_{i,k}.
    # So the tensor entry for indices (i,j) in A, (j',k) in B, (i',k') in C is 1 if i=i', j=j', k=k'.
    # Note that j and j' must be the same for the term to exist in the product.
    
    # So we want:
    # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 if (i,j,k) is a valid term?
    # Actually, the decomposition must hold for all i, j, k.
    # For a specific i, j, k, the term A[i,j] B[j,k] contributes to C[i,k].
    # So we require:
    # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1
    # And for any other combination of indices that doesn't form a valid multiplication term, it should be 0?
    # No, the linearity means we just need the identity to hold for the basis elements.
    # The basis elements of A are E_{i,j}, of B are E_{j,k}, of C are E_{i,k}.
    # The product E_{i,j} E_{j,k} = E_{i,k} (if the middle indices match).
    # So we need:
    # For all i, j, k:
    #   sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1
    # And for "mismatched" indices?
    # The tensor is zero if the indices don't align.
    # But in the matrix multiplication, A_{i,j} only multiplies B_{j,k}.
    # So the term A_{i,j} B_{j',k} is 0 in the product if j != j'.
    # Does our decomposition respect this?
    # The expression sum_c U[i*m+j, c] V[j'*p+k, c] W[k*n+i, c]
    # If we set j' != j, does it vanish?
    # Not necessarily automatically. We need to enforce it.
    # However, usually the tensor rank is defined for the specific tensor T.
    # T_{(i,j), (j',k), (i',k')} = 1 if i=i', j=j', k=k', else 0.
    # So we need:
    # sum_c U[i*m+j, c] V[j'*p+k, c] W[k'*n+i', c] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    
    # This is a system of equations.
    # Number of equations: n*m * m*p * p*n = n^2 m^2 p^2.
    # For 2,4,4: 4 * 16 * 16 = 1024 equations.
    # Number of variables: r * (n*m + m*p + p*n) = r * (8 + 16 + 8) = 32r.
    # For r=28, 32*28 = 896 variables.
    # Underdetermined? 896 vars vs 1024 eqs.
    # Wait, the equations are not all independent.
    # Also, the equations are quadratic/cubic in variables?
    # No, U, V, W are variables. The equation is linear in U for fixed V,W, etc.
    # It's a multilinear system.
    
    # Strategy: Gradient Descent on the sum of squared errors.
    # Loss = sum_{i,j,j',k,k',i'} (sum_c U[i*m+j, c] V[j'*p+k, c] W[k'*n+i', c] - delta_{i,i'} delta_{j,j'} delta_{k,k'})^2
    
    # This is computationally heavy if done naively.
    # But we can use random sampling of the indices to approximate the loss.
    
    def solve_for_rank(r):
        # Initialize U, V, W randomly
        np.random.seed(42)
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        # We will optimize U, V, W
        # Use a simple gradient descent with momentum or Adam
        # Since we have time limits, we'll do a few steps.
        
        # Precompute the target tensor indices?
        # We'll sample random tuples (i, j, j', k, k', i')
        
        lr = 0.01
        momentum = 0.9
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        
        steps = 1000
        batch_size = 100
        
        for step in range(steps):
            if time.time() > deadline:
                return None
            
            # Sample random indices
            i = np.random.randint(0, n, batch_size)
            j = np.random.randint(0, m, batch_size)
            jp = np.random.randint(0, m, batch_size)
            k = np.random.randint(0, p, batch_size)
            kp = np.random.randint(0, p, batch_size)
            ip = np.random.randint(0, n, batch_size)
            
            # Target: delta_{i,ip} delta_{j,jp} delta_{k,kp}
            target = (i == ip).astype(float) * (j == jp).astype(float) * (k == kp).astype(float)
            
            # Compute predicted values
            # pred[b] = sum_c U[i[b]*m+j[b], c] V[jp[b]*p+k[b], c] W[kp[b]*n+ip[b], c]
            
            # Indices for U
            idx_U = i * m + j
            # Indices for V
            idx_V = jp * p + k
            # Indices for W
            idx_W = kp * n + ip
            
            # Get rows
            U_rows = U[idx_U, :]  # (batch, r)
            V_rows = V[idx_V, :]  # (batch, r)
            W_rows = W[idx_W, :]  # (batch, r)
            
            # pred = sum_c U_rows * V_rows * W_rows
            pred = np.sum(U_rows * V_rows * W_rows, axis=1)
            
            # Error
            err = pred - target
            loss = np.mean(err**2)
            
            # If loss is very small, we found a solution
            if loss < 1e-6:
                return U, V, W, loss
            
            # Gradients
            # dLoss/dU_rows = 2 * err * V_rows * W_rows
            # dLoss/dV_rows = 2 * err * U_rows * W_rows
            # dLoss/dW_rows = 2 * err * U_rows * V_rows
            
            grad_U_rows = 2 * err[:, np.newaxis] * V_rows * W_rows
            grad_V_rows = 2 * err[:, np.newaxis] * U_rows * W_rows
            grad_W_rows = 2 * err[:, np.newaxis] * U_rows * V_rows
            
            # Accumulate gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            np.add.at(dU, (idx_U, np.arange(r)[np.newaxis, :]), grad_U_rows)
            np.add.at(dV, (idx_V, np.arange(r)[np.newaxis, :]), grad_V_rows)
            np.add.at(dW, (idx_W, np.arange(r)[np.newaxis, :]), grad_W_rows)
            
            # Normalize by batch size
            dU /= batch_size
            dV /= batch_size
            dW /= batch_size
            
            # Update
            vU = momentum * vU - lr * dU
            vV = momentum * vV - lr * dV
            vW = momentum * vW - lr * dW
            
            U += vU
            V += vV
            W += vW
            
        # Final check
        # Verify on a larger sample
        i = np.random.randint(0, n, 1000)
        j = np.random.randint(0, m, 1000)
        jp = np.random.randint(0, m, 1000)
        k = np.random.randint(0, p, 1000)
        kp = np.random.randint(0, p, 1000)
        ip = np.random.randint(0, n, 1000)
        
        target = (i == ip).astype(float) * (j == jp).astype(float) * (k == kp).astype(float)
        
        idx_U = i * m + j
        idx_V = jp * p + k
        idx_W = kp * n + ip
        
        U_rows = U[idx_U, :]
        V_rows = V[idx_V, :]
        W_rows = W[idx_W, :]
        
        pred = np.sum(U_rows * V_rows * W_rows, axis=1)
        final_loss = np.mean((pred - target)**2)
        
        if final_loss < 1e-5:
            return U, V, W, final_loss
        else:
            return None

    for r in candidate_ranks:
        if time.time() > deadline:
            break
            
        result = solve_for_rank(r)
        if result:
            U, V, W, loss = result
            best_rank = r
            best_factors = (U, V, W)
            break # Found a solution, stop searching for higher ranks? 
            # Actually, we want MINIMAL rank. So if we find one at r, 
            # we should try to see if lower r works? 
            # We are iterating from low to high. So the first success is the minimal found.
            
    if best_factors is None:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    else:
        U, V, W = best_factors
        # Convert to list of lists as required
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
