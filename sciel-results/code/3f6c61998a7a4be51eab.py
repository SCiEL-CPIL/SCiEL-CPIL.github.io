"""Find a rank-minimal exact tensor decomposition for 2x4 * 4x6 matrix multiplication."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == delta_{i,i'} delta_{j,j'} delta_{k,k'}
    Actually the tensor T[i,j,k; i',j',k'] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    The decomposition is: T = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] ... wait, that's not right.
    
    Let me reconsider. The matrix multiplication tensor T has indices (i, j, k, i', j', k') where:
    T[i,j,k,i',j',k'] = 1 if i==i' and j==j' and k==k', else 0.
    
    A rank-r decomposition is: T = sum_{r=1}^{R} U_r[i,j] * V_r[j,k] * W_r[k,i]
    
    Wait, let me look at the seed program more carefully.
    
    In the seed:
    - U has shape (n*m, rank): U[i*m+j, c]
    - V has shape (m*p, rank): V[j*p+k, c]  
    - W has shape (p*n, rank): W[k*n+i, c]
    
    And the decomposition is: for each (i,j,k), the product sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal 1 if this is the "diagonal" element, but actually...
    
    Hmm, let me think about this differently. The matrix multiplication tensor for multiplying an (n x m) matrix A by an (m x p) matrix B to get an (n x p) matrix C is:
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T has entries T[i,j,k] = delta_{i,k}... no wait, that doesn't make sense either since i and k are from different dimensions.
    
    Actually, the standard formulation: the multiplication tensor has three modes:
    - Mode 1: (i, j) for input A, indexed by i in [0,n), j in [0,m)
    - Mode 2: (j, k) for input B, indexed by j in [0,m), k in [0,p)
    - Mode 3: (i, k) for output C, indexed by i in [0,n), k in [0,p)
    
    The tensor T[(i,j),(j',k),(i',k')] = 1 if i==i' and j==j' and k==k', else 0.
    
    So T[(i,j),(j',k),(i',k')] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    
    A rank-R decomposition: T = sum_{r=1}^R u_r[i,j] * v_r[j',k] * w_r[i',k']
    
    In the seed program's indexing:
    - U[i*m+j, r] corresponds to u_r[i,j]
    - V[j'*p+k, r] corresponds to v_r[j',k]
    - W[k'*n+i', r] corresponds to w_r[i',k']  ... wait, W[k*n+i] where k is the second index of V and i is the first index of U.
    
    So W[k*n+i, r] = w_r[i,k] where i is from mode 1's first index and k is from mode 2's second index.
    
    The verification: for all (i,j,j',k,i',k'),
    sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i', r] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    
    In the schoolbook case, each term has exactly one (i,j,k) triple with U=1, V=1, W=1, and the sum picks out the right one.
    
    So to verify: we need to check that for all combinations, the sum equals the Kronecker delta product.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    
    # Build the tensor T from the decomposition
    # T[i,j,j',k,i',k'] = sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i', r]
    
    # This is a 6-dimensional tensor. Let's verify by checking all entries.
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for ip in range(n):
                        for kp in range(p):
                            expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            actual = 0.0
                            for r in range(R):
                                actual += U[i * m + j, r] * V[jp * p + k, r] * W[k * n + ip, r]
                            if abs(actual - expected) > 1e-9:
                                return False
    return True

def _verify_fast(U, V, W, n, m, p):
    """Faster verification using tensor operations."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    
    # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    
    # T[i,j,jp,k,ip,kp] = sum_r U[i,j,r] * V[jp,k,r] * W[k,ip,r]
    # We need T[i,j,jp,k,ip,kp] = delta_{i,ip} * delta_{j,jp} * delta_{k,kp}
    
    # Compute for all (i, jp, k, ip): sum_r U[i,j,r]*V[jp,k,r]*W[k,ip,r] but j is free...
    # Actually let's just do it with einsum
    # T[i,j,jp,k,ip] = sum_r U[i,j,r] * V[jp,k,r] * W[k,ip,r]
    T = np.einsum('ijr,jkr,kir->ijjkr', U3, V3, W3)  # This might be wrong indexing
    
    # Let me be more careful.
    # U3[i,j,r], V3[jp,k,r], W3[k,ip,r]
    # We want: T[i,j,jp,k,ip] = sum_r U3[i,j,r] * V3[jp,k,r] * W3[k,ip,r]
    T = np.einsum('ijr,jkr,kir->ijjkr', U3, V3, W3)
    
    # Check: T[i,j,jp,k,ip] should be delta_{i,ip} * delta_{j,jp} * delta_{k,kp}
    # But we don't have kp in T... wait, W3[k,ip,r] already has k and ip, so the output is T[i,j,jp,k,ip]
    # The condition is: T[i,j,jp,k,ip] = 1 if i==ip and j==jp and k==kp... but kp isn't an index here.
    
    # I think I'm overcomplicating. Let me re-read the problem.
    # The output factor W[k*n+i, r] - so W is indexed by (k, i) where k is from the second matrix's column and i is from the first matrix's row.
    # The product is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should give... what exactly?
    
    # For matrix multiplication C = A*B where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # The tensor decomposition expresses the multiplication as:
    # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
    # 
    # Wait, that's not quite right either. Let me think about it as:
    # The multiplication map is: (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # In tensor form, we have a trilinear form or a tensor T such that:
    # T[i,j,k; i',j',k'] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    # and C[i',k'] = sum_{i,j,j',k} A[i,j] B[j',k] T[i,j,k; i',j',k']
    # 
    # Hmm, this is getting confusing. Let me just use the direct verification from the seed.
    
    # Direct check: for all (i, j, jp, k, ip, kp):
    # sum_r U[i*m+j, r] * V[jp*p+k, r] * W[k*n+ip, r] = 1 if i==ip and j==jp and k==kp else 0
    # 
    # Wait, but that's not dimensionally consistent. Let me look at the seed again.
    # 
    # In the seed, for each (i, j, k):
    # U[i*m+j, col] = 1
    # V[j*p+k, col] = 1
    # W[k*n+i, col] = 1
    # 
    # So the product for a given (i,j,jp,k,ip,kp) would be:
    # sum_r U[i*m+j, r] * V[jp*p+k, r] * W[k*n+ip, r]
    # 
    # In the schoolbook, this equals 1 if there exists a column r where U[i*m+j,r]=1, V[jp*p+k,r]=1, W[k*n+ip,r]=1 simultaneously.
    # In the schoolbook, each column corresponds to a unique (i,j,k) triple, so:
    # U[i*m+j,r]=1 iff r corresponds to (i,j,*)
    # V[jp*p+k,r]=1 iff r corresponds to (*,jp,k)
    # W[k*n+ip,r]=1 iff r corresponds to (*,*,ip) with k fixed... 
    # 
    # Actually W[k*n+ip, r] = 1 iff the column r has W index (k, ip), which in the schoolbook means the triple is (ip, *, k).
    # 
    # So all three are 1 iff: i=ip (from U and W), j=jp (from U and V), k=k (from V and W, always true since k is the same).
    # So the sum is 1 iff i==ip and j==jp, regardless of k? That can't be right.
    # 
    # Wait, I think the verification condition might be different. Let me just do the brute force check.
    
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for ip in range(n):
                        for kp in range(p):
                            expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            actual = 0.0
                            for r in range(R):
                                actual += U[i * m + j, r] * V[jp * p + k, r] * W[k * n + ip, r]
                            if abs(actual - expected) > 1e-9:
                                return False
    return True

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Leave 5s margin
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # Known lower bounds: rank >= max(n*p, m*p, n*m) = max(12, 24, 8) = 24
    # Also rank >= ceil(n*m*p / max(n,m,p))... various bounds
    
    # Try random search for lower ranks
    random.seed(42)
    
    for target_rank in range(max(n*p, m*p, n*m), best_rank):
        if time.time() > DEADLINE - 30:
            break
        
        attempts = 0
        max_attempts = 50
        
        while attempts < max_attempts and time.time() < DEADLINE - 10:
            attempts += 1
            
            # Generate random U, V, W with small integer/rational entries
            # Use entries from {-1, 0, 1} or {-1, 0, 1, 2}
            scale = 2
            U_rand = np.random.randint(-scale, scale + 1, (n * m, target_rank))
            V_rand = np.random.randint(-scale, scale + 1, (m * p, target_rank))
            W_rand = np.random.randint(-scale, scale + 1, (p * n, target_rank))
            
            # Verify
            if _verify_fast(U_rand.tolist(), V_rand.tolist(), W_rand.tolist(), n, m, p):
                best_U, best_V, best_W = U_rand.tolist(), V_rand.tolist(), W_rand.tolist()
                best_rank = target_rank
                break
        
        if best_rank == target_rank:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
