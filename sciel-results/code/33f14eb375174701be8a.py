import time
import numpy as np
import random
import sys

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Target Tensor M
    # M[i, j, k] = 1 if this term is part of the multiplication sum
    # Actually, the tensor M is defined such that C = sum_r U_r V_r W_r
    # The condition is that the bilinear form matches.
    # We can represent the tensor M as a 3D array M[i, j, k].
    # M[i, j, k] = 1 for all i, j, k? No.
    # The multiplication tensor is M[i, j, k] = 1 if the indices are valid?
    # Let's check the schoolbook:
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1.
    # This implies M[i, j, k] = 1 for all i, j, k?
    # If M[i, j, k] = 1, then the sum is sum_{i,j,k} A_{ij} B_{jk} C_{ki}? No.
    # The output is C_{ik}.
    # The tensor M has entries M_{i, j, k}.
    # The contraction is sum_{i,j,k} A_{ij} B_{jk} M_{i, j, k} C_{ki}? No.
    # The standard form is C_{ik} = sum_j A_{ij} B_{jk}.
    # This can be viewed as sum_{i,j,k} (delta_{ij} delta_{jk}) A_{ij} B_{jk} C_{ki}? No.
    
    # Let's use the explicit check:
    # For random A, B, compute C = A @ B.
    # Compute C' = sum_r (U_r * A) * (V_r * B) * W_r? No.
    # C'_{ik} = sum_r (sum_j U_{ij}^{(r)} A_{ij}) (sum_k' V_{jk}^{(r)} B_{jk}) W_{ki}^{(r)}?
    # The indices in U, V, W are flattened.
    # U is (n*m, R). V is (m*p, R). W is (p*n, R).
    # Term r:
    # u_r is a vector of length n*m.
    # v_r is a vector of length m*p.
    # w_r is a vector of length p*n.
    # The contribution to C is:
    # C_{ik} += (sum_{j} U_{i*m+j, r} A_{ij}) * (sum_{j'} V_{j'*p+k, r} B_{j'k}) * W_{k*n+i, r}
    # Wait, the indices in V are j*p+k.
    # So v_r[j*p+k] multiplies B[j,k].
    # The term is (u_r^T A_flat) (v_r^T B_flat) w_r[k*n+i].
    # This is not quite right. The standard factorization is:
    # C = sum_r u_r v_r^T w_r^T? No.
    # C_{ik} = sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)}? No.
    
    # Let's look at the seed code again.
    # U[i*m+j][column] = 1
    # V[j*p+k][column] = 1
    # W[k*n+i][column] = 1
    # This means for a specific column r (corresponding to i, j, k):
    # U_r has 1 at index i*m+j.
    # V_r has 1 at index j*p+k.
    # W_r has 1 at index k*n+i.
    # The product term is A_{ij} * B_{jk} * 1.
    # And it contributes to C_{ik} via W_{k*n+i}.
    # So C_{ik} = sum_r (A_{ij} if U_{i*m+j,r}=1) * (B_{jk} if V_{j*p+k,r}=1) * W_{k*n+i,r}.
    # Since U and V are 0/1 in schoolbook, this works.
    
    # General form:
    # C_{ik} = sum_r (sum_j U_{i*m+j, r} A_{ij}) * (sum_j' V_{j'*p+k, r} B_{j'k}) * W_{k*n+i, r}
    # This is NOT the standard tensor decomposition form.
    # Standard: C_{ik} = sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)}?
    # No, the standard is C_{ik} = sum_r (u_r \otimes v_r \otimes w_r)_{i,k}?
    # The tensor M is M_{i,j,k}.
    # C_{ik} = sum_j M_{i,j,k} A_{ij} B_{jk}.
    # Here M_{i,j,k} = 1.
    # So we want sum_r u_r[i] v_r[j] w_r[k] = 1 for all i,j,k? No.
    # We want sum_r u_r[i] v_r[j] w_r[k] A_{ij} B_{jk} = sum_j A_{ij} B_{jk}.
    # This implies that the tensor sum_r u_r \otimes v_r \otimes w_r must equal the tensor M where M_{i,j,k} = 1.
    # Wait, if M_{i,j,k} = 1, then the rank is the rank of the all-ones tensor?
    # No, the indices i, j, k are coupled.
    # The tensor is M_{i, j, k} = \delta_{ij} \delta_{jk}? No.
    # The multiplication is C_{ik} = sum_j A_{ij} B_{jk}.
    # This is a bilinear map.
    # The tensor is M = sum_{i,j,k} e_i \otimes e_j \otimes e_k?
    # If we contract M with A and B:
    # <M, A \otimes B \otimes C> = sum_{i,j,k} M_{i,j,k} A_{ij} B_{jk} C_{ki}.
    # We want this to be sum_{i,k} C_{ik} (sum_j A_{ij} B_{jk})? No.
    # We want the map (A, B) -> C.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # This can be written as C = M(A, B).
    # The tensor M has components M_{i, j, k} such that C_{ik} = sum_j M_{i, j, k} A_{ij} B_{jk}.
    # Here M_{i, j, k} = 1.
    # So we are looking for the rank of the tensor M_{i, j, k} = 1 for all i, j, k?
    # No, that would be the rank of the all-ones tensor, which is 1.
    # That's clearly wrong.
    
    # The error is in the index mapping.
    # The tensor for matrix multiplication is NOT the all-ones tensor.
    # The tensor is M_{i, j, k} = 1 if the indices are consistent?
    # Let's check the seed code again.
    # U[i*m+j] = 1. V[j*p+k] = 1. W[k*n+i] = 1.
    # This creates a term that picks out A_{ij} and B_{jk} and puts it in C_{ik}.
    # So the "target" tensor is the sum of these specific rank-1 tensors.
    # There are n*m*p such tensors.
    # They are linearly independent?
    # Yes, because each tensor has a unique support pattern in the (i, j, k) space?
    # No, the support is a single entry in U, V, W.
    # The tensors are u_{i,j} \otimes v_{j,k} \otimes w_{k,i}.
    # These are n*m*p distinct rank-1 tensors.
    # Are they linearly independent?
    # Yes, because each corresponds to a unique (i, j, k) triplet.
    # So the rank is exactly n*m*p.
    
    # To reduce the rank, we need to find a linear combination of fewer rank-1 tensors that equals the sum of these n*m*p tensors.
    # This is equivalent to finding a low-rank decomposition of the tensor M.
    
    # We will implement a search for such a decomposition.
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    
    # Initialize schoolbook
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1

    # Target Tensor M (flattened or 3D)
    # M[i, j, k] = 1 for all i, j, k? No.
    # The tensor is T = sum_{i,j,k} u_{i,j} \otimes v_{j,k} \otimes w_{k,i}.
    # We want to approximate T with R terms.
    # We can define the error as the Frobenius norm of the difference between T and the approximation.
    # T is a 3D array of shape (n*m, m*p, p*n)? No.
    # The tensors u, v, w are vectors of lengths n*m, m*p, p*n.
    # The tensor M is a 3D array of shape (n*m, m*p, p*n).
    # M[a, b, c] = 1 if there exists i, j, k such that a=i*m+j, b=j*p+k, c=k*n+i?
    # No, M is the sum of rank-1 tensors.
    # M = sum_{i,j,k} e_{i*m+j} \otimes e_{j*p+k} \otimes e_{k*n+i}.
    # So M[a, b, c] = 1 if (a, b, c) corresponds to a valid (i, j, k).
    # Specifically, a=i*m+j, b=j*p+k, c=k*n+i.
    # Note that j is determined by a and i? No, i is not in a.
    # a = i*m + j. So i = a // m, j = a % m.
    # b = j*p + k. So j = b // p, k = b % p.
    # c = k*n + i. So k = c // n, i = c % n.
    # For M[a, b, c] to be 1, we need consistency:
    # i from a == i from c
    # j from a == j from b
    # k from b == k from c
    # If consistent, M[a, b, c] = 1. Else 0.
    
    M = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                M[a, b, c] = 1

    # Search for lower rank
    # We try ranks from 80 down to 70.
    # For each rank, we try a few random initializations and local search.
    
    def check_rank(U, V, W):
        # Compute approximation
        # App[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # This is a tensor product sum.
        # We can compute this efficiently?
        # App = sum_r outer(U[:, r], outer(V[:, r], W[:, r]))
        # This is O(R * n*m * m*p * p*n) which is large.
        # n*m=15, m*p=30, p*n=18. Size 15*30*18 = 8100.
        # R=80. 80 * 8100 = 648,000 operations. Very fast.
        App = np.zeros_like(M)
        for r in range(len(U[0])):
            u = np.array(U[:, r])
            v = np.array(V[:, r])
            w = np.array(W[:, r])
            App += np.einsum('i,j,k->ijk', u, v, w)
        
        # Error
        err = np.linalg.norm(App - M)
        return err

    # Local search
    def local_search(U, V, W, max_iters=1000):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        current_err = check_rank(U, V, W)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random rank component
            r = random.randint(0, len(U[0]) - 1)
            
            # Perturb U, V, W slightly
            # Or use gradient descent step?
            # Gradient of err^2 w.r.t U[a, r] is 2 * sum_{b,c} (App[a,b,c] - M[a,b,c]) V[b,r] W[c,r]
            
            # Compute gradients
            App = np.zeros_like(M)
            for rr in range(len(U[0])):
                u = U[:, rr]
                v = V[:, rr]
                w = W[:, rr]
                App += np.einsum('i,j,k->ijk', u, v, w)
            
            diff = App - M
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for rr in range(len(U[0])):
                v = V[:, rr]
                w = W[:, rr]
                u = U[:, rr]
                
                # grad_U[a, rr] = sum_{b,c} diff[a,b,c] V[b,rr] W[c,rr]
                # This is a tensor contraction.
                grad_U[:, rr] = np.einsum('abc,b,c->a', diff, v, w)
                
                grad_V[:, rr] = np.einsum('abc,a,c->b', diff, u, w)
                
                grad_W[:, rr] = np.einsum('abc,a,b->c', diff, u, v)
            
            # Update
            lr = 0.01
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            new_err = check_rank(U, V, W)
            if new_err < current_err:
                current_err = new_err
            else:
                # Revert
                U += lr * grad_U
                V += lr * grad_V
                W += lr * grad_W
                # Try smaller step or break
                break
        
        return U, V, W, current_err

    # Try to find lower rank
    for R in range(85, 60, -1):
        if time.time() > deadline:
            break
        
        found = False
        for trial in range(20):
            if time.time() > deadline:
                break
            
            # Random init
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # Scale to match magnitude
            # M has entries 0/1.
            # Initialize with small values?
            U *= 0.1
            V *= 0.1
            W *= 0.1
            
            U, V, W, err = local_search(U, V, W, max_iters=500)
            
            if err < 1e-5:
                # Convert to list of lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                # Check if we can simplify to integers/half-integers?
                # The task allows integer/half-integer entries.
                # We can round if close.
                
                best_rank = R
                best_U = U_list
                best_V = V_list
                best_W = W_list
                found = True
                break
        
        if found:
            break

    if best_rank == n * m * p:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": best_U, "V": best_V, "W": best_W}

