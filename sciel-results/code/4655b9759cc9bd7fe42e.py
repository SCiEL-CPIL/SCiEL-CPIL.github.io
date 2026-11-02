"""Matrix multiplication 2x3 x 3x6 rank-minimal decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 18
    pn = p * n  # 12
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 36
    
    # Build the tensor T of shape (nm, mp, pn)
    # T[i*m+j, j'*p+k, k'*n+i'] = 1 if i==i' and j==j' (and the j in V matches)
    # Actually: T[a, b, c] where a=i*m+j, b=j2*p+k, c=k2*n+i2
    # T[a,b,c] = 1 iff there exists (i,j,k) such that a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def verify(U, V, W):
        """Check if U, V, W form a valid decomposition."""
        r = U.shape[1]
        # For each (i,j,k), check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val - 1.0) > 1e-6:
                        return False
        # Check all other entries are 0
        # We need to check all (a,b,c) not in the "diagonal" set
        diagonal_set = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    diagonal_set.add((i*m+j, j*p+k, k*n+i))
        # Check a sample of off-diagonal entries
        # For efficiency, check all entries but skip diagonal
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if (a, b, c) in diagonal_set:
                        continue
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val) > 1e-6:
                        return False
        return True
    
    def verify_fast(U, V, W):
        """Faster verification using tensor contraction."""
        r = U.shape[1]
        # Compute the full tensor from decomposition
        # T_decomp[a,b,c] = sum_c U[a,c]*V[b,c]*W[c,c]
        # This is a rank-r tensor
        # T_decomp = sum_r U[:,r] outer V[:,r] outer W[:,r]
        # Check against T
        # Use einsum for speed
        T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
        if not np.allclose(T_decomp, T, atol=1e-6):
            return False
        return True
    
    # Try to find lower rank decompositions
    # Start from rank 30 down to 1
    best_rank = schoolbook_rank
    best_result = None
    
    # First, create the schoolbook solution
    U_school = np.zeros((nm, schoolbook_rank))
    V_school = np.zeros((mp, schoolbook_rank))
    W_school = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    # Try ranks from 30 to 35
    for rank in range(30, schoolbook_rank):
        if time.time() > deadline - 5:
            break
        
        np.random.seed(rank * 1000)
        
        # Try multiple random initializations
        found = False
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # Use alternating minimization / gradient descent
            # Objective: sum over all (a,b,c) of (T_decomp[a,b,c] - T[a,b,c])^2
            # But this is expensive. Instead, use the bilinear structure.
            
            # Alternate: fix V and W, solve for U (linear system)
            # For each a: sum_c U[a,c]*V[b,c]*W[c,c] should equal T[a,b,c] for all (b,c)
            # This is: U[a,:] * (V[b,:]*W[c,:]) = T[a,b,c] for all b,c
            # So U[a,:] is a vector such that its dot product with V[b,:]*W[c,:] = T[a,b,c]
            # This is a linear system: U[a,:] * M = T[a,:,:] where M[b,c] = V[b,:]*W[c,:]
            
            # Actually, let's use a simpler approach: gradient descent on the bilinear objective
            # f(U,V,W) = ||sum_r U[:,r] outer V[:,r] outer W[:,r] - T||^2
            
            lr = 0.01
            for step in range(500):
                if time.time() > deadline - 5:
                    break
                
                # Compute gradient
                # d/dU: for each a, sum over (b,c) where T[a,b,c]!=0: 2*(T_decomp[a,b,c]-T[a,b,c])*V[b,:]*W[c,:]
                # But computing T_decomp fully is expensive
                # Instead, compute gradients directly
                
                # grad_U[a,:] = 2 * sum_{b,c} (T_decomp[a,b,c] - T[a,b,c]) * V[b,:] * W[c,:]
                # = 2 * sum_c V[b,c]*W[c,c] * (T_decomp[a,b,c] - T[a,b,c])
                # This is still expensive. Let's just do a few steps with full computation.
                
                # For small dimensions, full computation is OK
                T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
                diff = T_decomp - T
                
                # Gradient with respect to U
                grad_U = np.einsum('abc,br,cr->ar', diff, V, W) * 2
                grad_V = np.einsum('abc,ar,cr->br', diff, U, W) * 2
                grad_W = np.einsum('abc,ar,br->cr', diff, U, V) * 2
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check convergence
                if np.max(np.abs(diff)) < 1e-8:
                    break
            
            # Check if we found a valid solution
            if verify_fast(U, V, W):
                found = True
                best_rank = rank
                # Convert to list format
                U_list = [[float(U[i,j]) for j in range(rank)] for i in range(nm)]
                V_list = [[float(V[i,j]) for j in range(rank)] for i in range(mp)]
                W_list = [[float(W[i,j]) for j in range(rank)] for i in range(pn)]
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                break
        
        if found:
            break
    
    if best_result is None:
        # Return schoolbook
        U_list = [[float(U_school[i,j]) for j in range(schoolbook_rank)] for i in range(nm)]
        V_list = [[float(V_school[i,j]) for j in range(schoolbook_rank)] for i in range(mp)]
        W_list = [[float(W_school[i,j]) for j in range(schoolbook_rank)] for i in range(pn)]
        best_result = {"U": U_list, "V": V_list, "W": W_list}
    
    return best_result
