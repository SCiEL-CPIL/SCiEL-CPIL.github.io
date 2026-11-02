import numpy as np
import time
import random

DIMENSIONS = [2, 3, 6]

def verify(U, V, W, n, m, p):
    """
    Verifies if U, V, W form a valid decomposition of the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    r = U.shape[1]
    # The tensor T_{a,b,c,d,i,k} = 1 if a=i, b=d, c=k else 0
    # Sum_l U_{a,b,l} V_{c,d,l} W_{k,i,l} = T_{a,b,c,d,i,k}
    
    # Flatten indices for easier checking
    # T is 1 at (a, b, c, d, i, k) iff a==i and b==d and c==k
    # There are n*m*p such non-zero entries in the "diagonal" sense of the multiplication tensor.
    # Actually, the tensor has dimension n*m * m*p * p*n.
    # The condition is: for each l, the rank-1 tensor is U_l (size nm) x V_l (size mp) x W_l (size pn).
    # We need the sum of these to equal the standard multiplication tensor.
    
    # Efficient check:
    # Construct the full tensor T (sparse)
    # T[a*m+b, c*p+d, k*n+i] = 1 if a==i and b==d and c==k else 0
    
    # Let's iterate over the non-zero entries of T.
    # Non-zero entries are at indices:
    # a in 0..n-1
    # b in 0..m-1
    # c in 0..m-1
    # d in 0..p-1
    # i in 0..n-1
    # k in 0..p-1
    # Constraint: a==i, b==d, c==k
    
    # So we iterate a, b, c, d (where d=b? No, d is col of B, b is col of A. 
    # A is n x m. B is m x p. C is n x p.
    # C_{i,k} = sum_b A_{i,b} B_{b,k}.
    # Tensor T_{i,b,k,c,d? No.
    # Standard tensor T_{i,b, k, j, l?}
    # Let's stick to the indices in the prompt:
    # U index: a*m+b (row a, col b of A)
    # V index: c*p+d (row c, col d of B)
    # W index: k*n+i (row k, col i of C? Prompt says W[k*n+i,r]. 
    # Usually C is n x p. So row i, col k. Index i*p+k?
    # Prompt says "output factor follows the official trace convention W[k*n+i,r]".
    # This implies W index is k*n + i. So k is the "row" index of the flattened W?
    # If W has shape (p*n, rank), and index is k*n+i, then k is 0..p-1 and i is 0..n-1.
    # This corresponds to C_{i,k} if we view C as n x p.
    # Let's check the target condition in the seed code:
    # target = 1.0 if (a == i and c == k and b == d) else 0.0
    # s += U[a * m + b, l] * V[c * p + d, l] * W[k * n + i, l]
    # This confirms:
    # U index: a*m+b
    # V index: c*p+d
    # W index: k*n+i
    # Target 1 if a==i, b==d, c==k.
    
    # There are n * m * m * p * n * p total combinations? No.
    # The non-zero targets are determined by (a, b, c, d, i, k) satisfying constraints.
    # a==i, b==d, c==k.
    # Free variables: a (n), b (m), c (m), d (p, but d=b so fixed), k (p, but k=c so fixed).
    # Wait, b is 0..m-1. d is 0..p-1. If b==d, then b must be < min(m,p).
    # Here m=3, p=6. So b can be 0,1,2. d must equal b. So d is 0,1,2.
    # c is 0..m-1 (0,1,2). k must equal c. So k is 0,1,2.
    # So non-zero entries exist only for specific combinations.
    
    # Let's just iterate all possible non-zero positions.
    # Positions (a, b, c, d, i, k) such that a==i, b==d, c==k.
    # a in range(n)
    # b in range(m)
    # c in range(m)
    # d = b
    # i = a
    # k = c
    
    # Check if d < p and k < p.
    # Since b < m and m=3 < p=6, d=b is always valid.
    # Since c < m and m=3 < p=6, k=c is always valid.
    
    # So we iterate:
    for a in range(n):
        for b in range(m):
            for c in range(m):
                # d = b, i = a, k = c
                # Check sum
                idx_u = a * m + b
                idx_v = c * p + b  # d=b
                idx_w = c * n + a  # k=c, i=a
                
                target_sum = 0.0
                for l in range(r):
                    target_sum += U[idx_u, l] * V[idx_v, l] * W[idx_w, l]
                
                if not np.isclose(target_sum, 1.0, atol=1e-9):
                    return False
    
    # We must also ensure that all OTHER entries are 0.
    # This is the hard part. The number of entries is (n*m)*(m*p)*(p*n) = 6*18*12 = 1296.
    # We can't iterate all 1296 for every candidate in a tight loop if we have many candidates.
    # But 1296 is small.
    # Let's iterate all entries to be safe.
    
    # Actually, the previous check only verified the 1s.
    # We must verify the 0s.
    # Total entries: 6 * 18 * 12 = 1296.
    # For each entry (u, v, w), sum_l U[u,l] V[v,l] W[w,l] must be 0 if not (a==i, b==d, c==k).
    
    # Let's map indices back to (a,b), (c,d), (k,i).
    # u = a*m + b
    # v = c*p + d
    # w = k*n + i
    
    for u in range(n*m):
        a = u // m
        b = u % m
        for v in range(m*p):
            c = v // p
            d = v % p
            for w in range(p*n):
                k = w // n
                i = w % n
                
                is_target = (a == i and b == d and c == k)
                
                s = 0.0
                for l in range(r):
                    s += U[u, l] * V[v, l] * W[w, l]
                
                if is_target:
                    if not np.isclose(s, 1.0, atol=1e-9):
                        return False
                else:
                    if not np.isclose(s, 0.0, atol=1e-9):
                        return False
                        
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Dimensions
    dim_u = n * m  # 6
    dim_v = m * p  # 18
    dim_w = p * n  # 12
    
    # We want to find a set of rank-1 tensors (u_l, v_l, w_l) with integer/half-integer entries
    # such that their sum is the multiplication tensor.
    
    # Strategy:
    # 1. Start with the schoolbook solution (rank 36).
    # 2. Try to reduce rank by finding linear dependencies or replacing terms.
    # 3. Alternatively, use a greedy search to build a minimal set.
    
    # Given the time constraint and the difficulty of proving optimality,
    # we will attempt to find a rank < 34 solution via a randomized local search
    # starting from a valid decomposition, trying to merge or remove terms.
    
    # However, a more robust "different" approach is to use the structure of p=6.
    # We can view the 3x6 matrix B as three 3x2 blocks: B1, B2, B3.
    # C = A * [B1 B2 B3] = [A*B1 A*B2 A*B3].
    # This is just 3 independent 2x3x2 multiplications.
    # If we have a rank r decomposition for 2x3x2, we can use rank 3*r for 2x3x6.
    # Schoolbook 2x3x2 is rank 6 (2*3*2/2? No, schoolbook is n*m*p = 12? 
    # Wait. 2x3x2 schoolbook:
    # A is 2x3, B is 3x2.
    # C_{i,k} = sum_b A_{i,b} B_{b,k}.
    # Terms: 2*3*2 = 12? No.
    # Standard algorithm: for i in 2, for b in 3, for k in 2: C[i,k] += A[i,b]*B[b,k].
    # This is 12 operations.
    # But tensor rank of M_{2,3,2} is known to be 9? Or 10?
    # Actually, rank(M_{2,3,2}) = 9 is a known result (Strassen-like).
    # If rank(M_{2,3,2}) = 9, then rank(M_{2,3,6}) <= 27.
    # Current best is 34.
    # If we can find a rank 9 decomposition for 2x3x2, we can use it 3 times to get rank 27.
    # This would be a huge improvement (34 -> 27).
    
    # Let's try to find a rank 9 decomposition for 2x3x2.
    # If we fail, we fall back to the known rank 34 solution (or try to improve it slightly).
    
    # Known rank 9 decomposition for 2x3x2?
    # Let's try to construct one or find one in literature/knowledge.
    # Actually, let's just try to search for it.
    
    # Search for rank 9 in 2x3x2
    # Dim U: 6, V: 6, W: 4.
    # Target tensor T.
    
    # We will use a simple randomized gradient descent or simulated annealing on the error.
    # Error = || Sum_l u_l v_l w_l - T ||^2
    
    def find_rank_decomposition(n, m, p, target_rank, time_limit):
        start = time.time()
        dim_u = n * m
        dim_v = m * p
        dim_w = p * n
        
        # Initialize with random vectors
        # Try multiple restarts
        best_error = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        for restart in range(50):
            if time.time() - start > time_limit:
                break
                
            U = np.random.randn(dim_u, target_rank)
            V = np.random.randn(dim_v, target_rank)
            W = np.random.randn(dim_w, target_rank)
            
            # Normalize
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            # Gradient descent
            # Compute error
            def compute_error(U, V, W):
                # Sum_l U_l V_l W_l
                # This is a tensor of size dim_u x dim_v x dim_w
                # We only need to compare with T.
                # T is sparse.
                # Let's compute the full tensor and subtract T.
                # Size 6x6x4 = 144 for 2x3x2. Small.
                T_comp = np.einsum('ul,vl,wl->uvw', U, V, W)
                
                # Construct T
                T = np.zeros((dim_u, dim_v, dim_w))
                for a in range(n):
                    for b in range(m):
                        for c in range(m):
                            for d in range(p):
                                for i in range(n):
                                    for k in range(p):
                                        if a == i and b == d and c == k:
                                            T[a*m+b, c*p+d, k*n+i] = 1.0
                                        
                return np.sum((T_comp - T)**2)
            
            # Simple gradient descent
            lr = 0.1
            for step in range(1000):
                if time.time() - start > time_limit:
                    break
                
                err = compute_error(U, V, W)
                if err < best_error:
                    best_error = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    
                if best_error < 1e-8:
                    # Snap to integers if close
                    # Check if we can snap
                    U_int = np.round(U)
                    V_int = np.round(V)
                    W_int = np.round(W)
                    if compute_error(U_int, V_int, W_int) < 1e-8:
                        return U_int, V_int, W_int
                    return U, V, W # Return float if integer snap fails but error is low? 
                                  # The task requires integer/half-integer.
                                  # So we MUST snap.
                                  # If we can't snap, it's invalid.
                                  # So we should continue searching or return None.
                                  # But if error is 0 with floats, it might not be 0 with ints.
                                  # So we return None if we can't snap.
                    return None
                
                # Compute gradients
                # dE/dU = 2 * ( (T - T_comp) contracted with V, W )
                # This is expensive to do explicitly for each step if we recompute T_comp.
                # But for 2x3x2, it's small.
                
                T_comp = np.einsum('ul,vl,wl->uvw', U, V, W)
                T = np.zeros((dim_u, dim_v, dim_w))
                for a in range(n):
                    for b in range(m):
                        for c in range(m):
                            for d in range(p):
                                for i in range(n):
                                    for k in range(p):
                                        if a == i and b == d and c == k:
                                            T[a*m+b, c*p+d, k*n+i] = 1.0
                                            
                diff = T - T_comp
                
                # grad_U[u,l] = sum_{v,w} diff[u,v,w] V[v,l] W[w,l]
                grad_U = np.einsum('uvw,vl,wl->ul', diff, V, W)
                grad_V = np.einsum('uvw,ul,wl->vl', diff, U, W)
                grad_W = np.einsum('uvw,ul,vl->wl', diff, U, V)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Renormalize to prevent drift
                U /= np.linalg.norm(U)
                V /= np.linalg.norm(V)
                W /= np.linalg.norm(W)
                
                # Try to snap to integers occasionally
                if step % 100 == 0:
                    U_s = np.round(U)
                    V_s = np.round(V)
                    W_s = np.round(W)
                    if compute_error(U_s, V_s, W_s) < 1e-8:
                        return U_s, V_s, W_s
                        
        return None
    
    # Try to find rank 9 for 2x3x2
    # If successful, we can build 2x3x6 with rank 27.
    # We have 140s. Let's give 60s to this search.
    
    U_232, V_232, W_232 = find_rank_decomposition(2, 3, 2, 9, 60.0)
    
    if U_232 is not None:
        # Construct 2x3x6 from three 2x3x2 blocks
        # A is 2x3. B is 3x6 = [B1, B2, B3] where B_j is 3x2.
        # C is 2x6 = [C1, C2, C3] where C_j = A B_j.
        
        # U for 2x3x6:
        # U has shape (6, 27).
        # For each block j (0,1,2), the U part is the same as U_232.
        # U_232 has shape (6, 9).
        # So U_236[:, 0:9] = U_232
        # U_236[:, 9:18] = U_232
        # U_236[:, 18:27] = U_232
        
        # V for 2x3x6:
        # V has shape (18, 27).
        # V index: c*p+d. c in 0..2, d in 0..5.
        # Block j corresponds to d in {2j, 2j+1}.
        # For block j, V_232 index is c*2+d_local where d_local in 0..1.
        # So V_236[c*6 + 2j + d_local, 9*j + l] = V_232[c*2 + d_local, l]
        
        # W for 2x3x6:
        # W has shape (12, 27).
        # W index: k*n+i. k in 0..5, i in 0..1.
        # Block j corresponds to k in {2j, 2j+1}.
        # For block j, W_232 index is k_local*n+i where k_local in 0..1.
        # So W_236[(2j+k_local)*2 + i, 9*j + l] = W_232[k_local*2 + i, l]
        
        rank_232 = U_232.shape[1]
        total_rank = 3 * rank_232
        
        U_236 = np.zeros((6, total_rank))
        V_236 = np.zeros((18, total_rank))
        W_236 = np.zeros((12, total_rank))
        
        for j in range(3):
            # U
            U_236[:, j*rank_232:(j+1)*rank_232] = U_232
            
            # V
            for c in range(3):
                for d_local in range(2):
                    v_idx_236 = c * 6 + (j * 2 + d_local)
                    v_idx_232 = c * 2 + d_local
                    V_236[v_idx_236, j*rank_232:(j+1)*rank_232] = V_232[v_idx_232]
                    
            # W
            for k_local in range(2):
                for i in range(2):
                    w_idx_236 = (j * 2 + k_local) * 2 + i
                    w_idx_232 = k_local * 2 + i
                    W_236[w_idx_236, j*rank_232:(j+1)*rank_232] = W_232[w_idx_232]
                    
        # Verify
        if verify(U_236, V_236, W_236, 2, 3, 6):
            return {
                "U": U_236.tolist(),
                "V": V_236.tolist(),
                "W": W_236.tolist()
            }
            
    # Fallback: Return the known rank 34 solution from the seed program
    # This ensures we don't score 0 if the search fails.
    
    U_comb = np.zeros((6, 34))
    V_comb = np.zeros((18, 34))
    W_comb = np.zeros((12, 34))
    
    U233_seed = np.zeros((6, 17))
    V233_seed = np.zeros((9, 17))
    W233_seed = np.zeros((6, 17))
    
    U233_seed[0, 0] = 1; U233_seed[4, 0] = 1
    V233_seed[0, 0] = 1; V233_seed[4, 0] = 1
    W233_seed[0, 0] = 1
    W233_seed[3, 0] = 1
    
    U233_seed[3, 1] = 1; U233_seed[4, 1] = 1
    V233_seed[0, 1] = 1
    W233_seed[1, 1] = 1
    W233_seed[3, 1] = -1
    
    U233_seed[0, 2] = 1
    V233_seed[1, 2] = 1; V233_seed[4, 2] = -1
    W233_seed[2, 2] = 1
    W233_seed[3, 2] = 1
    
    U233_seed[4, 3] = 1
    V233_seed[3, 3] = 1; V233_seed[0, 3] = -1
    W233_seed[0, 3] = 1
    W233_seed[1, 3] = 1
    
    U233_seed[0, 4] = 1; U233_seed[1, 4] = 1
    V233_seed[4, 4] = 1
    W233_seed[0, 4] = -1
    W233_seed[2, 4] = 1
    
    U233_seed[3, 5] = 1; U233_seed[0, 5] = -1
    V233_seed[0, 5] = 1; V233_seed[1, 5] = 1
    W233_seed[3, 5] = 1
    
    U233_seed[1, 6] = 1; U233_seed[4, 6] = -1
    V233_seed[3, 6] = 1; V233_seed[4, 6] = 1
    W233_seed[0, 6] = 1
    
    U233_seed[0, 7] = 1
    V233_seed[2, 7] = 1
    W233_seed[4, 7] = 1
    
    U233_seed[1, 8] = 1
    V233_seed[5, 8] = 1
    W233_seed[4, 8] = 1
    
    U233_seed[3, 9] = 1
    V233_seed[2, 9] = 1
    W233_seed[5, 9] = 1
    
    U233_seed[4, 10] = 1
    V233_seed[5, 10] = 1
    W233_seed[5, 10] = 1
    
    U233_seed[2, 11] = 1
    V233_seed[6, 11] = 1
    W233_seed[0, 11] = 1
    
    U233_seed[2, 12] = 1
    V233_seed[7, 12] = 1
    W233_seed[2, 12] = 1
    
    U233_seed[2, 13] = 1
    V233_seed[8, 13] = 1
    W233_seed[4, 13] = 1
    
    U233_seed[5, 14] = 1
    V233_seed[6, 14] = 1
    W233_seed[1, 14] = 1
    
    U233_seed[5, 15] = 1
    V233_seed[7, 15] = 1
    W233_seed[3, 15] = 1
    
    U233_seed[5, 16] = 1
    V233_seed[8, 16] = 1
    W233_seed[5, 16] = 1
    
    U_comb[:, :17] = U233_seed
    U_comb[:, 17:] = U233_seed
    
    for j in range(3):
        for k in range(3):
            V_comb[j * 6 + k, :17] = V233_seed[j * 3 + k]
            V_comb[j * 6 + 3 + k, 17:] = V233_seed[j * 3 + k]
            
    for k in range(3):
        for i in range(2):
            W_comb[k * 2 + i, :17] = W233_seed[k * 2 + i]
            W_comb[(k + 3) * 2 + i, 17:] = W233_seed[k * 2 + i]
            
    return {
        "U": U_comb.tolist(),
        "V": V_comb.tolist(),
        "W": W_comb.tolist()
    }

if __name__ == "__main__":
    construct()
