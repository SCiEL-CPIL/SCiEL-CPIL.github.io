"""Matrix multiplication 2x3x8 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # We'll try to find a low-rank decomposition.
    # Schoolbook rank is n*m*p = 48.
    # Let's try to use a known structure or search for a lower rank.
    # For 2x3x8, the tensor is T[i,j,k] = e_{i,k} * e_{j} (where e_{i,k} is (i,k) entry of A, e_j is j-th entry of B).
    # Actually, the tensor for matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p.
    # T_{i,j,k} is the coefficient of a_{i,j} * b_{j,k} in c_{i,k}.
    # Wait, the standard tensor is T_{i,k,j} = delta_{i,k} ? No.
    # The tensor is defined such that c_{i,k} = sum_{j} a_{i,j} b_{j,k}.
    # So the tensor T has indices (i, k, j) and T_{i,k,j} = 1 if the term a_{i,j} b_{j,k} contributes to c_{i,k}.
    # But a_{i,j} and b_{j,k} are multiplied. So T_{i,k,j} = 1 for all i,k,j? No.
    # The tensor is T_{i,k}^{(j)}? No, it's a 3-tensor.
    # T_{i,k,j} = 1 if the j-th column of A and j-th row of B are used.
    # Specifically, c_{i,k} = sum_j a_{i,j} b_{j,k}.
    # So the tensor T has components T_{i,k,j} which is 1 if we are multiplying a_{i,j} and b_{j,k}.
    # But a_{i,j} is fixed for a given i,j. b_{j,k} is fixed for a given j,k.
    # The tensor is T_{i,j,k} = 1? No.
    # The tensor T is defined by T_{i,k,j} = 1.
    # Let's check the interface: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank).
    # U rows indexed by (i,j) -> i*m+j.
    # V rows indexed by (j,k) -> j*p+k.
    # W rows indexed by (k,i) -> k*n+i. (Note: trace convention W[k*n+i, r])
    # The decomposition is: sum_r U[(i,j),r] * V[(j,k),r] * W[(k,i),r] = delta_{i,k} ? No.
    # The product U * V * W should reconstruct the tensor.
    # The tensor T_{i,j,k} is 1 if the (i,j) entry of A and (j,k) entry of B are multiplied to form (i,k) entry of C.
    # Wait, the (i,j) entry of A is a_{i,j}. The (j,k) entry of B is b_{j,k}.
    # Their product a_{i,j} b_{j,k} contributes to c_{i,k}.
    # So the tensor T has non-zero entries at (i, j, k) ? No, the indices of the tensor correspond to the free indices of the three factors.
    # Factor 1: A_{i,j} -> index (i,j)
    # Factor 2: B_{j,k} -> index (j,k)
    # Factor 3: C_{i,k} -> index (i,k)
    # The tensor T_{(i,j), (j,k), (i,k)} is 1.
    # In the interface:
    # U row index: i*m + j
    # V row index: j*p + k
    # W row index: k*n + i
    # So T_{i*m+j, j*p+k, k*n+i} = 1.
    # We need U[(i,j), r] * V[(j,k), r] * W[(k,i), r] summed over r to be 1 if the indices match, 0 otherwise.
    # Specifically, sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 if (i,j,k) is a valid triple (which it always is), but wait.
    # The tensor is 1 for ALL i,j,k?
    # No, the tensor is 1 for the specific combination that forms the product.
    # Actually, for matrix multiplication, the tensor T is defined as:
    # T_{i,j,k} = 1 if we are computing c_{i,k} using a_{i,j} and b_{j,k}.
    # But a_{i,j} and b_{j,k} are the only terms that involve both i and k via j.
    # So T_{i*m+j, j*p+k, k*n+i} = 1.
    # And it is 0 for any other combination of indices?
    # No, the indices (i,j), (j,k), (k,i) are linked.
    # The tensor is non-zero only when the indices are consistent.
    # i.e., T_{u, v, w} is non-zero only if u = i*m+j, v = j*p+k, w = k*n+i for some i,j,k.
    # And for such a triple, T_{u,v,w} = 1.
    # So the tensor is a permutation-like tensor?
    # Let's check the rank.
    # The schoolbook decomposition uses rank n*m*p.
    # For each (i,j,k), we have a rank-1 term.
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1.
    # This is valid.
    # Can we do better?
    # For 2x3x8, n=2, m=3, p=8.
    # Rank is at most 48.
    # Let's try to find a decomposition of rank < 48.
    # We can use the fact that the tensor is symmetric in some way?
    # Or use known results for small matrix multiplication.
    # For 2x2x2, rank is 7 (Strassen).
    # For 2x3x3, rank is ?
    # For 2x3x8, maybe we can use a block structure.
    # Let's try to use a random search or a structured search for a low-rank decomposition.
    # Given the time limit (120s), we can try a few random restarts with gradient descent or ALS.
    # But the entries must be integer/half-integer.
    # This suggests that the optimal decomposition might have simple entries.
    # Let's try to find a decomposition with rank 44 (as hinted in the knowledge).
    # Or maybe rank 40, 36, etc.
    # Let's implement a search that tries to minimize the rank by solving a feasibility problem.
    # However, solving a feasibility problem for tensor rank is NP-hard.
    # Let's try a heuristic: start with a random initialization of U, V, W with rank R, and use ALS to minimize the error.
    # If the error is close to 0, we have a valid decomposition of rank R.
    # We can try R = 44, 43, ... until we find one.
    # To ensure integer/half-integer entries, we can round the solution to the nearest half-integer and check if it still works.
    # Or, we can restrict the search to half-integers from the start.
    # Let's try a simple approach:
    # 1. Generate the full tensor T of shape (n*m, m*p, p*n).
    # 2. For R in [44, 45, ..., 48]:
    #    a. Initialize U, V, W randomly with entries in {-1, 0, 1} or {-0.5, 0, 0.5, 1}.
    #    b. Use ALS to minimize ||T - U*V*W||_F.
    #    c. If the error is < 1e-6, round U, V, W to the nearest half-integer.
    #    d. Check if the rounded U, V, W still give error < 1e-6.
    #    e. If yes, return the decomposition.
    # 3. If no decomposition found, return the schoolbook one.
    
    # Let's implement this.
    
    # First, generate the tensor T.
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1.0
    
    # Schoolbook decomposition as fallback.
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=np.float64)
        V = np.zeros((m*p, rank), dtype=np.float64)
        W = np.zeros((p*n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = schoolbook()
    
    # Try to find a lower rank decomposition.
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds margin
    
    # We'll try ranks from 44 down to 40.
    for R in range(44, 39, -1):
        if time.time() > deadline:
            break
        # Try a few random initializations.
        for attempt in range(10):
            if time.time() > deadline:
                break
            # Initialize U, V, W with random half-integers.
            # Entries in {-1, -0.5, 0, 0.5, 1}.
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, R))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, R))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, R))
            
            # ALS iterations.
            for it in range(50):
                if time.time() > deadline:
                    break
                # Update U.
                # For each row u of U, minimize sum_v,w T[u,v,w] V[v,r] W[w,r] - U[u,r]
                # This is a least squares problem.
                # Let's vectorize.
                # For each r, U[:, r] = argmin_u sum_v,w T[u,v,w] V[v,r] W[w,r] - U[u,r]
                # This is equivalent to: U[:, r] = (T * (V[r] * W[r])) / (V[r] * W[r])
                # But T is a 3-tensor, so we need to contract it with V[r] and W[r].
                # Let's do it with einsum.
                # T[u, v, w] V[v, r] W[w, r] -> sum_v,w T[u,v,w] V[v,r] W[w,r]
                # This is a vector of length n*m.
                # Let's compute it for all r at once.
                # V is (m*p, R), W is (p*n, R).
                # We want to compute for each r: sum_v,w T[u,v,w] V[v,r] W[w,r]
                # This is T @ (V[r] * W[r]) for each r.
                # Let's use einsum: 'uvw,vr,wr->ur'
                # But T is large, so this might be slow.
                # Let's try it.
                try:
                    # Compute the numerator for U.
                    num_U = np.einsum('uvw,vr,wr->ur', T, V, W)
                    # Compute the denominator: sum_v,w (V[v,r] W[w,r])^2
                    den_U = np.einsum('vr,wr->r', V**2, W**2)
                    # Avoid division by zero.
                    den_U[den_U == 0] = 1
                    U = num_U / den_U[None, :]
                    # Round to half-integers.
                    U = np.round(U * 2) / 2
                    
                    # Update V.
                    num_V = np.einsum('uvw,ur,wr->vr', T, U, W)
                    den_V = np.einsum('ur,wr->r', U**2, W**2)
                    den_V[den_V == 0] = 1
                    V = num_V / den_V[None, :]
                    V = np.round(V * 2) / 2
                    
                    # Update W.
                    num_W = np.einsum('uvw,ur,vr->wr', T, U, V)
                    den_W = np.einsum('ur,vr->r', U**2, V**2)
                    den_W[den_W == 0] = 1
                    W = num_W / den_W[None, :]
                    W = np.round(W * 2) / 2
                except Exception:
                    break
                
                # Check error.
                # Compute T - U*V*W.
                # U*V*W is a 3-tensor: sum_r U[u,r] V[v,r] W[w,r]
                # Use einsum: 'ur,vr,wr->uvw'
                try:
                    approx = np.einsum('ur,vr,wr->uvw', U, V, W)
                    error = np.linalg.norm(T - approx)
                    if error < 1e-6:
                        # Found a valid decomposition.
                        # Convert to lists.
                        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                except Exception:
                    break
    
    # If no lower rank found, return schoolbook.
    return {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}
