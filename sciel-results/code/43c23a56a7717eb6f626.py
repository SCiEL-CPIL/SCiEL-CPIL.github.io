"""Matrix multiplication 4,5,5 tensor decomposition search."""
import numpy as np
import time
import itertools
import random

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    
    # Schoolbook baseline
    rank_schoolbook = n * m * p  # 100
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    def to_dict(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = to_dict(U_sb, V_sb, W_sb)
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions using structured search
    # The tensor T[i,j,k] = delta_{i,j,k} for matrix multiplication
    # We look for rank-r decompositions: T = sum_t u_t ⊗ v_t ⊗ w_t
    
    # Strategy: Use the known lower bounds and try random/structured initializations
    # with iterative refinement (like ALS - Alternating Least Squares)
    
    # Lower bound for 4x5x5: at least max(n*m, m*p, p*n) = max(20,25,20) = 25
    # But actual lower bound is higher. Known: R(4,5,5) >= 26 (by various bounds)
    # Let's try to find decompositions with rank < 100
    
    def tensor_entry(i, j, k):
        """The matrix multiplication tensor entry."""
        return 1.0  # T[i,j,k] = 1 for all i,j,k in this convention? 
        # Wait, let me reconsider. The tensor for matrix multiplication is:
        # T[i,j,k] such that C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor is T[i,j,k] = 1 if we think of it as A⊗B -> C
        # Actually the standard form: T_{i,j,k} where contraction over j gives C_{i,k}
        # So T[i,j,k] = 1 for all valid i,j,k? No...
        # The tensor is defined such that <T, A⊗B> = trace(C^T) where C = A*B
        # In index form: sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,k} C[i,k]
        # C[i,k] = sum_j A[i,j] B[j,k]
        # So sum_{i,j,k} A[i,j] B[j,k] = sum_{i,k} C[i,k]
        # This means T[i,j,k] = 1 for all i,j,k? That can't be right for rank.
        # 
        # Let me re-read the problem. The decomposition is:
        # U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] summed over t should give 1
        # when this represents the multiplication tensor.
        # 
        # Actually, the matrix multiplication tensor has entries T[i,j,k] = 1 
        # only in the sense that the structure is:
        # C_{i,k} = sum_j A_{i,j} B_{j,k}
        # The tensor form: T_{i,j,k} where the "fiber" along j connects A and B
        # Standard: T_{i,j,k} = 1 if we identify it correctly.
        # 
        # For the rank decomposition: we need
        # sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = T[i,j,k]
        # where T[i,j,k] is the matrix multiplication tensor.
        # 
        # The matrix multiplication tensor is: T[i,j,k] = 1 for all i,j,k? 
        # No. Let's think again. The tensor is a 3-tensor of size n x m x p.
        # Its entries are T[i,j,k] = 1 if the (i,k)-entry of C gets contribution 
        # from A[i,j]*B[j,k]. Since C[i,k] = sum_j A[i,j] B[j,k], every 
        # (i,j,k) triple contributes. So T[i,j,k] = 1 for all i,j,k.
        # 
        # Wait, that would make the tensor all-ones, which has rank 1. That's wrong.
        # 
        # I think the issue is the indexing convention. Let me re-read:
        # "The output factor follows the official trace convention W[k*n+i,r]."
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # 
        # The tensor equation is:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
        # 
        # For matrix multiplication, T[i,j,k] should be such that this gives the 
        # correct structure. In the schoolbook case, each (i,j,k) has its own rank-1 term,
        # so T[i,j,k] = 1 for all i,j,k and the decomposition is trivial with rank n*m*p.
        # 
        # But that means the tensor IS all-ones? Let me verify with a small case.
        # For 2x2x2: rank should be at most 8 (schoolbook), but actually R(2,2,2) = 7.
        # If T is all-ones, rank would be 1. So T is NOT all-ones.
        # 
        # I think I'm confusing myself. Let me look at the schoolbook construction again:
        # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
        # So for each (i,j,k), we have a rank-1 term that is 1 only at that specific
        # (i,j,k) position. This means T[i,j,k] = 1 for all i,j,k, and the schoolbook
        # decomposition just writes it as a sum of n*m*p rank-1 tensors each supported 
        # on a single entry. But that's not minimal.
        # 
        # Actually no. If T[i,j,k] = 1 for all i,j,k, then the all-ones tensor has
        # rank 1: U[i*m+j,0]=1, V[j*p+k,0]=1, W[k*n+i,0]=1 gives sum = 1 for all (i,j,k).
        # So the tensor can't be all-ones.
        # 
        # I think the correct interpretation is that the tensor represents the bilinear map
        # (A,B) -> C where C = A*B, and the tensor entries encode which (i,j) of A and 
        # (j,k) of B contribute to (i,k) of C. The key is that j must be the same in A and B.
        # 
        # So T[i,j,k] = 1 if and only if the j-index of A[i,j] matches the j-index of B[j,k].
        # But that's always true by construction. 
        # 
        # I think the issue is that in the tensor formulation, the three indices (i, j, k)
        # are NOT independent. The tensor is defined on the space of (i,j) pairs for A,
        # (j,k) pairs for B, and (i,k) pairs for C. The "tensor" is really a 3-tensor where
        # the middle index j is shared.
        # 
        # Let me just trust the schoolbook construction and the problem statement.
        # The schoolbook gives rank n*m*p with each (i,j,k) having a separate term.
        # A better decomposition would have rank < n*m*p.
        
        # I'll implement a numerical search using ALS (Alternating Least Squares)
        # to find low-rank decompositions.
        return 1.0
    
    # Let me reconsider the problem more carefully.
    # The matrix multiplication tensor for n x m times m x p:
    # T is a tensor of size (n*m) x (m*p) x (p*n)
    # T[(i,j), (j',k), (k',i')] = 1 if i=i' and j=j' and k=k'
    # i.e., T is the identity-like tensor that picks out matching indices.
    # 
    # So T[A, B, C] = sum_{i,j,k} A[i,j] B[j,k] C[k,i] (with appropriate indices)
    # And we want T to be decomposed as sum_r U_r ⊗ V_r ⊗ W_r where
    # U_r is a vector of size n*m, V_r of size m*p, W_r of size p*n.
    # 
    # The schoolbook decomposition: for each (i,j,k), U has 1 at position (i,j),
    # V has 1 at position (j,k), W has 1 at position (k,i).
    # This gives T[(i,j),(j,k),(k,i)] = 1 and 0 elsewhere. Correct.
    # 
    # Now we want to find rank-r decompositions. Let's use ALS.
    
    def als_decompose(n, m, p, rank, max_iter=200, tol=1e-10, seed=0):
        """Find rank-r decomposition of the matrix multiplication tensor using ALS."""
        rng = np.random.RandomState(seed)
        
        # Initialize randomly
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        
        # Normalize columns to avoid scale issues
        for r in range(rank):
            norm_u = np.linalg.norm(U[:, r])
            norm_v = np.linalg.norm(V[:, r])
            norm_w = np.linalg.norm(W[:, r])
            if norm_u > 1e-10 and norm_v > 1e-10 and norm_w > 1e-10:
                scale = (norm_u * norm_v * norm_w) ** (1/3)
                U[:, r] /= norm_u * scale
                V[:, r] /= norm_v * scale
                W[:, r] /= norm_w * scale
        
        prev_loss = float('inf')
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each (i,j), solve least squares
            for i in range(n):
                for j in range(m):
                    if time.time() > deadline - 5:
                        return None
                    # T[(i,j), :, :] should equal U[i*m+j, :] ⊗ (something)
                    # Actually, T[(i,j), (j',k), (k',i')] = 1 iff i=i' and j=j' and k=k'
                    # So for fixed (i,j), T[(i,j), (j',k), (k',i')] is nonzero only when
                    # j'=j and i'=i, and then it's 1 if k=k'.
                    # So T[(i,j), (j,k), (k,i)] = 1 for all k.
                    # 
                    # In the decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                    # and 0 otherwise.
                    # 
                    # For fixed (i,j), define:
                    # A_r = V[j*p+k, r] for all k -> this is a vector of length p
                    # B_r = W[k*n+i, r] for all k -> this is a vector of length p
                    # We need: sum_r U[i*m+j, r] * (A_r ⊙ B_r) = ones(p)
                    # where ⊙ is element-wise product.
                    # 
                    # So: U[i*m+j, :] @ M = ones(p) where M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    # This is underdetermined if rank > p, so we solve least squares.
                    
                    M = np.zeros((p, rank))
                    for k in range(p):
                        M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    
                    target = np.ones(p)
                    
                    # Solve M^T M u = M^T target
                    try:
                        A_mat = M.T @ M
                        b_vec = M.T @ target
                        # Add small regularization for stability
                        A_mat += 1e-10 * np.eye(rank)
                        u_new = np.linalg.solve(A_mat, b_vec)
                    except np.linalg.LinAlgError:
                        u_new = np.linalg.lstsq(M, target, rcond=None)[0]
                    
                    U[i * m + j, :] = u_new
            
            # Update V: for each (j,k)
            for j in range(m):
                for k in range(p):
                    if time.time() > deadline - 5:
                        return None
                    # T[(i,j), (j,k), (k',i')] = 1 iff i=i' and j=j and k=k'
                    # So for fixed (j,k), T[(i,j), (j,k), (k,i)] = 1 for all i
                    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
                    # V[j*p+k, r] * sum_r U[i*m+j, r] * W[k*n+i, r] = 1
                    # Let C_r = U[i*m+j, r] * W[k*n+i, r] for each r
                    # Then V[j*p+k, :] @ (C matrix) = ones(n)
                    # where C[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    
                    C = np.zeros((n, rank))
                    for i in range(n):
                        C[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    
                    target = np.ones(n)
                    
                    try:
                        A_mat = C.T @ C
                        b_vec = C.T @ target
                        A_mat += 1e-10 * np.eye(rank)
                        v_new = np.linalg.solve(A_mat, b_vec)
                    except np.linalg.LinAlgError:
                        v_new = np.linalg.lstsq(C, target, rcond=None)[0]
                    
                    V[j * p + k, :] = v_new
            
            # Update W: for each (k,i)
            for k in range(p):
                for i in range(n):
                    if time.time() > deadline - 5:
                        return None
                    # T[(i,j), (j,k), (k,i)] = 1 iff i=i' and j=j' and k=k'
                    # For fixed (k,i), T[(i,j), (j,k), (k,i)] = 1 for all j
                    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
                    # W[k*n+i, r] * sum_r U[i*m+j, r] * V[j*p+k, r] = 1
                    # Let D_j = U[i*m+j, r] * V[j*p+k, r]
                    # W[k*n+i, :] @ (D matrix) = ones(m)
                    # where D[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    
                    D = np.zeros((m, rank))
                    for j in range(m):
                        D[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    
                    target = np.ones(m)
                    
                    try:
                        A_mat = D.T @ D
                        b_vec = D.T @ target
                        A_mat += 1e-10 * np.eye(rank)
                        w_new = np.linalg.solve(A_mat, b_vec)
                    except np.linalg.LinAlgError:
                        w_new = np.linalg.lstsq(D, target, rcond=None)[0]
                    
                    W[k * n + i, :] = w_new
            
            # Compute loss
            loss = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                        loss += (val - 1.0) ** 2
            # Also need to check off-diagonal entries are zero
            # For entries where indices don't match, the value should be 0
            # But in our parameterization, the tensor is only defined on the "diagonal"
            # (i,j,k) triples. The off-diagonal entries are automatically 0 because
            # the tensor T[i,j,k] = 0 when the indices don't form a valid triple.
            # Actually, I need to reconsider. The tensor is a 3-tensor with dimensions
            # (n*m) x (m*p) x (p*n). The entry T[a, b, c] where a=(i,j), b=(j',k'), c=(k'',i'')
            # is 1 iff i=i'' and j=j' and k=k''.
            # 
            # In the decomposition: sum_r U[a,r] V[b,r] W[c,r]
            # This should equal T[a,b,c].
            # 
            # For the "diagonal" entries (where i=i'', j=j', k=k''), we need it to be 1.
            # For "off-diagonal" entries, we need it to be 0.
            # 
            # My ALS only enforces the diagonal constraints. I need to also enforce
            # that off-diagonal entries are 0. But there are many off-diagonal entries.
            # 
            # Let me add a penalty for off-diagonal entries.
            
            # Check a sample of off-diagonal entries
            off_loss = 0.0
            for _ in range(50):  # Sample 50 random off-diagonal entries
                i = rng.randint(n)
                j = rng.randint(m)
                k = rng.randint(p)
                i2 = rng.randint(n)
                j2 = rng.randint(m)
                k2 = rng.randint(p)
                # This is an off-diagonal if not (i==i2 and j==j2 and k==k2)
                if not (i == i2 and j == j2 and k == k2):
                    a = i * m + j
                    b = j2 * p + k2
                    c = k * n + i2
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    off_loss += val ** 2
            
            total_loss = loss + 100.0 * off_loss / 50.0
            
            if abs(prev_loss - total_loss) < tol:
                break
            prev_loss = total_loss
        
        # Verify the solution
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    max_err = max(max_err, abs(val - 1.0))
        
        # Check some off-diagonal entries
        for _ in range(200):
            i = rng.randint(n)
            j = rng.randint(m)
            k = rng.randint(p)
            i2 = rng.randint(n)
            j2 = rng.randint(m)
            k2 = rng.randint(p)
            if not (i == i2 and j == j2 and k == k2):
                a = i * m + j
                b = j2 * p + k2
                c = k * n + i2
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                max_err = max(max_err, abs(val))
        
        if max_err < 1e-6:
            return U, V, W, max_err
        else:
            return None
    
    # Try different ranks starting from a reasonable lower bound
    # Known lower bound for R(4,5,5) is at least 26
    # Let's try ranks from 26 upward
    
    for rank in range(26, min(100, best_rank)):
        if time.time() > deadline - 10:
            break
        
        found = False
        for seed in range(20):
            if time.time() > deadline - 10:
                break
            if found:
                break
            
            result = als_decompose(n, m, p, rank, max_iter=100, tol=1e-8, seed=seed)
            if result is not None:
                U, V, W, err = result
                if err < 1e-5:
                    # Try to round to integers or half-integers
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    # Check if rounded version still works
                    max_err_rounded = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U_rounded[i * m + j, :], V_rounded[j * p + k, :] * W_rounded[k * n + i, :])
                                max_err_rounded = max(max_err_rounded, abs(val - 1.0))
                    
                    for _ in range(100):
                        i = random.randint(n)
                        j = random.randint(m)
                        k = random.randint(p)
                        i2 = random.randint(n)
                        j2 = random.randint(m)
                        k2 = random.randint(p)
                        if not (i == i2 and j == j2 and k == k2):
                            a = i * m + j
                            b = j2 * p + k2
                            c = k * n + i2
                            val = np.dot(U_rounded[a, :], V_rounded[b, :] * W_rounded[c, :])
                            max_err_rounded = max(max_err_rounded, abs(val))
                    
                    if max_err_rounded < 1e-5:
                        best_result = to_dict(U_rounded, V_rounded, W_rounded)
                        best_rank = rank
                        found = True
                    else:
                        # Use unrounded if it's accurate enough
                        if max_err < 1e-6:
                            best_result = to_dict(U, V, W)
                            best_rank = rank
                            found = True
        
        if found:
            # Try to find even lower rank
            continue
    
    # If no better decomposition found, return schoolbook
    if best_rank == rank_schoolbook:
        return best_result
    else:
        return best_result

# EVOLVE-BLOCK-END
