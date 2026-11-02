import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_rank = rank_schoolbook
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    
    # Build the tensor structure
    # T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(k,i) for some i,j,k
    # a = i*m + j, b = j*p + k, c = k*n + i
    
    # For alternating minimization, we need to solve linear systems
    # Fix U and V, solve for W:
    # For each c, we need W[c,:] such that sum over (a,b) where T[a,b,c]=1: U[a,:] * V[b,:] * W[c,:] = 1
    # And for (a,b) where T[a,b,c]=0: U[a,:] * V[b,:] * W[c,:] = 0
    
    # Actually, this is more subtle. The tensor equation is:
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # Fix U and V, this is linear in W:
    # For each (a,b,c): sum_r (U[a,r] * V[b,r]) * W[c,r] = T[a,b,c]
    # Let M[c, r] = W[c,r], and let A[(a,b), c, r] = U[a,r] * V[b,r]
    # Then: sum_r A[(a,b), c, r] * M[c, r] = T[a,b,c]
    
    # This is a linear system: for each (a,b), we have equations for all c.
    # But it's coupled across c through the shared r dimension.
    
    # Let me think differently. For fixed U and V, define for each c:
    # vec(T[:, :, c]) = (U ⊙ V)_{:, c} * W[c, :]^T  where ⊙ is some product
    
    # Actually, let's just do it directly with the full linear system.
    
    # The full system: for all (a,b,c) in {0,...,nm-1} x {0,...,mp-1} x {0,...,pn-1}:
    # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    
    # Unknowns: W (pn x r) = pn*r unknowns
    # Equations: nm*mp*pn = 6*21*14 = 1764
    
    # For fixed U,V, this is linear in W. Let's set up the matrix.
    
    # Actually, this is getting complex. Let me try a simpler approach.
    
    # Try alternating minimization with a few iterations
    
    def verify_decomposition(U, V, W, tol=1e-10):
        """Check if U, V, W form a valid decomposition."""
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        
        # Check all nonzero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U[a] * V[b] * W[c])
                    if abs(val - 1.0) > tol:
                        return False
        
        # Check some zero entries (sample)
        np.random.seed(42)
        for _ in range(200):
            a = np.random.randint(0, n*m)
            b = np.random.randint(0, m*p)
            c = np.random.randint(0, p*n)
            # Check if this is a nonzero entry
            i1 = a // m
            j1 = a % m
            j2 = b // p
            k1 = b % p
            k2 = c // n
            i2 = c % n
            if j1 == j2 and i1 == i2:
                continue
            val = np.sum(U[a] * V[b] * W[c])
            if abs(val) > tol:
                return False
        
        return True
    
    # Try alternating minimization for various ranks
    for target_rank in range(35, rank_schoolbook):
        if time.time() > deadline - 10:
            break
            
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(target_rank * 100 + attempt)
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # Alternating minimization
            for iteration in range(50):
                if time.time() > deadline - 5:
                    break
                    
                # Fix U and V, solve for W
                # For each c, we need to solve: sum_r (U[a,r]*V[b,r]) * W[c,r] = T[a,b,c] for all (a,b)
                # This is: A_c @ W[c,:] = b_c where A_c is (nm*mp) x r, b_c is (nm*mp)
                
                # Actually, let's set up the full linear system for W
                # Unknowns: W flattened (pn * r)
                # Equation for each (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
                
                # Build the linear system
                num_eqs = (n*m) * (m*p) * (p*n)
                num_unk = (p*n) * target_rank
                
                # This is too large to build explicitly. Let's use a smarter approach.
                
                # For fixed U,V, the equation for W is:
                # For each c: sum over (a,b) where T[a,b,c]=1: (U[a,:]*V[b,:]) @ W[c,:] = 1
                # For each c: sum over (a,b) where T[a,b,c]=0: (U[a,:]*V[b,:]) @ W[c,:] = 0
                
                # For a given c, the nonzero (a,b) pairs are those where:
                # a = i*m+j, b = j'*p+k, c = k'*n+i' with j=j' and i=i' and k=k'... 
                # Wait, c = k*n + i, so k = c//n, i = c%n
                # Then a = i*m + j, b = j*p + k for any j in [0,m)
                # So for fixed c, the nonzero (a,b) are: a = (c%n)*m + j, b = j*p + (c//n) for j in [0,m)
                
                c_val = c // n
                i_val = c % n
                
                # For this c, the nonzero (a,b) pairs are:
                # a = i_val * m + j, b = j * p + c_val for j in range(m)
                
                # The equation is: sum_j (U[a_j, :] * V[b_j, :]) @ W[c, :] = 1
                # where a_j = i_val * m + j, b_j = j * p + c_val
                
                # And for zero entries: for all other (a,b), sum_r U[a,r]*V[b,r]*W[c,r] = 0
                
                # Let's build the linear system for W[c, :]
                # A @ W[c,:] = b where A has rows for each (a,b) pair
                # Row for (a,b): U[a,:] * V[b,:] (element-wise product, shape (r,))
                # b = 1 if (a,b,c) is nonzero, 0 otherwise
                
                # For efficiency, only use the nonzero constraints and a subset of zero constraints
                A_rows = []
                b_vals = []
                
                # Nonzero constraints for this c
                for j in range(m):
                    a = i_val * m + j
                    b = j * p + c_val
                    A_rows.append(U[a] * V[b])
                    b_vals.append(1.0)
                
                # Add some zero constraints (random sample)
                for _ in range(10):
                    a = np.random.randint(0, n*m)
                    b = np.random.randint(0, m*p)
                    # Check if this is a nonzero entry for this c
                    i1 = a // m
                    j1 = a % m
                    j2 = b // p
                    k1 = b % p
                    if i1 == i_val and j1 == j2 and k1 == c_val:
                        continue
                    A_rows.append(U[a] * V[b])
                    b_vals.append(0.0)
                
                A = np.array(A_rows)
                b = np.array(b_vals)
                
                # Solve least squares
                if A.shape[0] >= target_rank:
                    W_new, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W[c] = W_new
                else:
                    W[c] = np.linalg.pinv(A) @ b
                
                # Check convergence
                if iteration > 10:
                    # Verify
                    if verify_decomposition(U, V, W, tol=1e-8):
                        if target_rank < best_rank:
                            best_rank = target_rank
                            best_result = {
                                "U": U.tolist(),
                                "V": V.tolist(),
                                "W": W.tolist()
                            }
                        break
            
            if target_rank < best_rank:
                break
    
    if best_rank < rank_schoolbook:
        return best_result
    else:
        return best_result
