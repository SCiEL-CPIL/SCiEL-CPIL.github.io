import numpy as np
import time
import random

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    n, m, p = DIMENSIONS
    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10
    
    # Build the target tensor T
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    def snap_to_half_integers(arr):
        return np.round(arr * 2) / 2
    
    def validate_full(U, V, W, atol=1e-6):
        U_snap = snap_to_half_integers(U)
        V_snap = snap_to_half_integers(V)
        W_snap = snap_to_half_integers(W)
        
        if not (np.allclose(U, U_snap, atol=atol) and 
                np.allclose(V, V_snap, atol=atol) and 
                np.allclose(W, W_snap, atol=atol)):
            return False, None, None, None
            
        prod = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap, optimize=True)
        if np.allclose(prod, T, atol=atol):
            return True, U_snap, V_snap, W_snap
        return False, None, None, None
    
    def schoolbook():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1.0
                    V[j * p + k][column] = 1.0
                    W[k * n + i][column] = 1.0
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Linear code approach:
    # For rank R, each rank position r defines an intermediate product:
    #   s_r = sum_{j,k} V[j*p+k, r] * A[j,k] * B[k, :]  (conceptually)
    # Actually, the tensor decomposition says:
    #   T[u,v,w] = sum_r U[u,r] * V[v,r] * W[w,r]
    # For matrix multiplication, U[u,r] = U[i*m+j, r] depends on i,j
    # V[v,r] = V[j*p+k, r] depends on j,k
    # W[w,r] = W[k*n+i, r] depends on k,i
    #
    # Key insight: The product U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # must equal 1 when (i,j,k) is a valid index, 0 otherwise.
    #
    # Since the tensor is sparse (only 50 non-zero entries out of 10*25*10=2500),
    # we can think of it as: for each output entry (i,k), we need
    #   sum_j U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all j
    # But this isn't quite right. Let me reconsider.
    #
    # Actually T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k.
    # So the constraint is: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # And for all other (u,v,w), the sum must be 0.
    #
    # The "other" constraints: T[u,v,w] = 0 unless there exists i,j,k with
    # u=i*m+j, v=j*p+k, w=k*n+i. This means the indices must be consistent:
    # j = u%m, j = v//p, k = v%p, k = w//n, i = u//m, i = w%n
    # So consistency: u%m == v//p, v%p == w//n, u//m == w%n
    #
    # Linear code approach: For each rank position r, define:
    #   a_r(i,j) = U[i*m+j, r]
    #   b_r(j,k) = V[j*p+k, r]  
    #   c_r(k,i) = W[k*n+i, r]
    #
    # The constraint for valid (i,j,k): sum_r a_r(i,j)*b_r(j,k)*c_r(k,i) = 1
    # The constraint for invalid (u,v,w): sum_r a_r(u//m,u%m)*b_r(v//p,v%p)*c_r(w%n,w//n) = 0
    #
    # For the invalid constraints, note that if we set a_r(i,j), b_r(j,k), c_r(k,i)
    # to be zero for "most" combinations, we can satisfy many constraints.
    #
    # Strategy: Try to find a rank-R decomposition where each rank position
    # "covers" a subset of the 50 valid (i,j,k) triples, and the invalid
    # constraints are automatically satisfied.
    #
    # A sufficient condition for invalid constraints: For each rank position r,
    # the set of (i,j,k) where a_r(i,j)*b_r(j,k)*c_r(k,i) != 0 must be such that
    # no invalid (u,v,w) is "activated".
    #
    # Simpler approach: Try rank R = 10, 12, 15, 20, 25, 30, 35, 40, 45, 50
    # For each R, try to find a structured decomposition.
    
    # Approach: For each rank position r, assign it to handle a subset of (i,j,k) triples.
    # If rank position r handles triple (i,j,k), we need:
    #   a_r(i,j) * b_r(j,k) * c_r(k,i) contributes to the sum for that triple.
    #
    # Let's try: For rank R, assign each of the 50 triples to one of R positions.
    # For position r handling triples S_r, set:
    #   a_r(i,j) = 1 if (i,j) appears in some triple in S_r, else 0
    #   b_r(j,k) = 1 if (j,k) appears in some triple in S_r, else 0
    #   c_r(k,i) = 1 if (k,i) appears in some triple in S_r, else 0
    #
    # Then for a valid triple (i,j,k) assigned to position r:
    #   a_r(i,j)*b_r(j,k)*c_r(k,i) = 1 (if all three are 1)
    # For other valid triples (i',j',k') assigned to position r':
    #   If r' != r, the contribution is a_r(i',j')*b_r(j',k')*c_r(k',i')
    #   This could be 1 if the indices happen to match, causing overcounting.
    #
    # To avoid overcounting, we need: for position r, the set of (i,j) pairs,
    # (j,k) pairs, and (k,i) pairs must be such that no "cross" product
    # activates an unintended triple.
    #
    # This is equivalent to: the bipartite graph defined by position r must
    # not contain any "cycle" that would create an unintended valid triple.
    #
    # Let's try a simpler structured approach:
    # For R = 10: Assign triples based on (i, j) pairs. There are 10 (i,j) pairs.
    #   Position r corresponds to (i,j) where r = i*5+j.
    #   a_r(i',j') = 1 if i'=i and j'=j, else 0
    #   b_r(j',k') = 1 for all j',k' (since j is fixed, but k varies)
    #   Wait, this doesn't work because b_r depends on j and k.
    #
    # Let me try: For R = 10, position r = i*5+j handles all k for that (i,j).
    #   a_r(i',j') = 1 if (i',j')=(i,j)
    #   b_r(j',k') = 1 if j'=j (for any k')
    #   c_r(k',i') = 1 if i'=i (for any k')
    # Then for triple (i,j,k): a_r(i,j)*b_r(j,k)*c_r(k,i) = 1*1*1 = 1. Good.
    # For triple (i',j',k') with (i',j') != (i,j): a_r(i',j') = 0. Good.
    # For invalid (u,v,w): u=i'*5+j', v=j'*5+k', w=k'*2+i'
    #   a_r(i',j') = 1 only if (i',j')=(i,j)
    #   If (i',j')=(i,j), then b_r(j',k') = b_r(j,k') = 1 (since j'=j)
    #   c_r(k',i') = c_r(k',i) = 1 (since i'=i)
    #   So the product is 1, but this is for a VALID triple (i,j,k'), not invalid.
    #   Wait, if (i',j')=(i,j), then the triple (i,j,k') is valid!
    #   So this actually works for all valid triples.
    #   For truly invalid (u,v,w), we need u%m != v//p or v%p != w//n or u//m != w%n.
    #   If u%m == v//p and v%p == w//n and u//m == w%n, then it's valid.
    #   So any (u,v,w) that passes the consistency check is valid.
    #   Therefore, this decomposition works!
    #
    # Let's verify: R=10, position r=i*5+j.
    #   U[i*5+j, i*5+j] = 1 (diagonal in a 10x10 block)
    #   V[j*5+k, i*5+j] = 1 if j matches, i.e., V[j*5+k, r] = 1 if j == r%5
    #   W[k*2+i, i*5+j] = 1 if i matches, i.e., W[k*2+i, r] = 1 if i == r//5
    #
    # Check T[i*5+j, j*5+k, k*2+i]:
    #   sum_r U[i*5+j,r]*V[j*5+k,r]*W[k*2+i,r]
    #   = U[i*5+j, i*5+j]*V[j*5+k, i*5+j]*W[k*2+i, i*5+j]
    #   = 1 * 1 * 1 = 1. Correct!
    #
    # Check an invalid entry, e.g., T[0, 1, 0] (u=0, v=1, w=0):
    #   u=0: i=0, j=0
    #   v=1: j=0, k=1
    #   w=0: k=0, i=0
    #   Inconsistency: j from u is 0, j from v is 0 (ok), k from v is 1, k from w is 0 (mismatch)
    #   So this is invalid.
    #   sum_r U[0,r]*V[1,r]*W[0,r]
    #   U[0,r] = 1 only if r=0
    #   V[1,r] = V[0*5+1, r] = 1 if j=0, so r%5==0, i.e., r=0 or r=5
    #   W[0,r] = W[0*2+0, r] = 1 if i=0, so r//5==0, i.e., r=0,1,2,3,4
    #   Intersection: r=0
    #   U[0,0]*V[1,0]*W[0,0] = 1*1*1 = 1. But T[0,1,0] should be 0!
    #
    # Wait, let me recheck. T[0,1,0]: u=0, v=1, w=0.
    #   i = u//m = 0//5 = 0, j = u%m = 0%5 = 0
    #   j' = v//p = 1//5 = 0, k' = v%p = 1%5 = 1
    #   k'' = w//n = 0//2 = 0, i'' = w%n = 0%2 = 0
    #   Consistency: j==j' (0==0 ok), k'==k'' (1==0 FAIL), i==i'' (0==0 ok)
    #   So this is indeed invalid, and T[0,1,0] should be 0.
    #   But our decomposition gives 1. So R=10 with this structure doesn't work!
    #
    # The issue: V[1,0] = 1 because j=0 (1//5=0), and W[0,0]=1 because i=0.
    # But k is inconsistent. We need to also check k consistency.
    #
    # So we need: a_r(i,j)*b_r(j,k)*c_r(k,i) = 1 only when i,j,k are consistent.
    # The problem is that b_r(j,k) depends on both j and k, and c_r(k,i) depends on k and i.
    #
    # For the R=10 structure: b_r(j,k) = 1 if j == r%5 (independent of k)
    #                         c_r(k,i) = 1 if i == r//5 (independent of k)
    # So the product is 1 whenever i=r//5 and j=r%5, regardless of k.
    # This means for a fixed (i,j), ALL k values give product 1.
    # But T[i*5+j, j*5+k, k*2+i] = 1 for all k, so this is actually correct!
    #
    # Wait, I think I made an error. Let me recheck T[0,1,0].
    # u=0, v=1, w=0. Is this a valid tensor index?
    # The tensor T has indices (u,v,w) where u in [0,10), v in [0,25), w in [0,10).
    # T[u,v,w] = 1 if there exist i,j,k such that u=i*5+j, v=j*5+k, w=k*2+i.
    # For u=0: i=0, j=0
    # For v=1: j=0, k=1
    # For w=0: k=0, i=0
    # k from v is 1, k from w is 0. Inconsistent. So T[0,1,0] = 0.
    #
    # In our decomposition: sum_r U[0,r]*V[1,r]*W[0,r]
    # U[0,r] = 1 if r=0
    # V[1,r]: v=1, so j=1//5=0, k=1%5=1. V[1,r] = 1 if j==r%5, i.e., 0==r%5. r=0,5.
    # W[0,r]: w=0, so k=0//2=0, i=0%2=0. W[0,r] = 1 if i==r//5, i.e., 0==r//5. r=0,1,2,3,4.
    # Intersection: r=0.
    # Product: U[0,0]*V[1,0]*W[0,0] = 1*1*1 = 1.
    # But T[0,1,0] = 0. So the decomposition is WRONG.
    #
    # The issue: V[1,0] should be 0 because k=1 doesn't match the k implied by w=0 (which is k=0).
    # But in our structure, V[j*5+k, r] = 1 if j==r%5, regardless of k.
    # This is the problem.
    #
    # To fix: We need V[j*5+k, r] to also depend on k. But then we need more rank positions.
    #
    # Alternative: For R=25, position r = j*5+k.
    #   U[i*5+j, r] = 1 if j == r//5
    #   V[j*5+k, r] = 1 if j*5+k == r
    #   W[k*2+i, r] = 1 if k == r%5
    # Check T[i*5+j, j*5+k, k*2+i]:
    #   sum_r U[i*5+j,r]*V[j*5+k,r]*W[k*2+i,r]
    #   Only r=j*5+k can have V[j*5+k,r]=1.
    #   U[i*5+j, j*5+k] = 1 if j==(j*5+k)//5 = j. Yes.
    #   W[k*2+i, j*5+k] = 1 if k==(j*5+k)%5 = k. Yes.
    #   So product = 1. Correct!
    # Check T[0,1,0]: u=0,v=1,w=0.
    #   U[0,r] = 1 if j=0==r//5. r=0,1,2,3,4.
    #   V[1,r] = 1 if r=1 (since j*5+k=1 means j=0,k=1).
    #   W[0,r] = 1 if k=0==r%5. r=0,5,10,15,20.
    #   Intersection: r must be in {0,1,2,3,4} ∩ {1} ∩ {0,5,10,15,20} = empty.
    #   So sum = 0. Correct!
    #
    # So R=25 works with this structure. But can we do better?
    #
    # For R=10, we need to share rank positions between different k values.
    # The issue is that V[j*5+k, r] must distinguish k values.
    #
    # Let's try R=15. Assign positions based on (i, j) with some k sharing.
    # This gets complex. Let's just try R=25 first, then try to optimize.
    
    # R=25 structured solution
    R = 25
    U_arr = np.zeros((U_dim, R))
    V_arr = np.zeros((V_dim, R))
    W_arr = np.zeros((W_dim, R))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = j * p + k
                U_arr[i * m + j, r] = 1.0
                V_arr[j * p + k, r] = 1.0
                W_arr[k * n + i, r] = 1.0
    
    # Validate
    prod = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr, optimize=True)
    if np.allclose(prod, T, atol=1e-6):
        U_snap = snap_to_half_integers(U_arr)
        V_snap = snap_to_half_integers(V_arr)
        W_snap = snap_to_half_integers(W_arr)
        prod2 = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap, optimize=True)
        if np.allclose(prod2, T, atol=1e-6):
            U_list = [[float(x) for x in row] for row in U_snap]
            V_list = [[float(x) for x in row] for row in V_snap]
            W_list = [[float(x) for x in row] for row in W_snap]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Try to find lower rank via optimization
    # Start from R=25 and try to reduce by merging rank positions
    
    for R_target in [20, 18, 15, 12, 10]:
        if time.time() > deadline - 20:
            break
        
        # Initialize by taking the first R_target columns of the R=25 solution
        # and try to optimize
        U_init = np.zeros((U_dim, R_target))
        V_init = np.zeros((V_dim, R_target))
        W_init = np.zeros((W_dim, R_target))
        
        # Use a subset of the 25 rank positions
        for r in range(R_target):
            j = r // p
            k = r % p
            for i in range(n):
                U_init[i * m + j, r] = 1.0
                W_init[k * n + i, r] = 1.0
            V_init[j * p + k, r] = 1.0
        
        # Optimize
        lr = 0.01
        momentum = 0.9
        vU = np.zeros_like(U_init)
        vV = np.zeros_like(V_init)
        vW = np.zeros_like(W_init)
        
        max_iter = 300
        patience = 80
        no_improve = 0
        prev_loss = float('inf')
        
        for it in range(max_iter):
            if time.time() > deadline - 15:
                break
            
            prod = np.einsum('ar,br,cr->abc', U_init, V_init, W_init, optimize=True)
            diff = prod - T
            
            gradU = np.einsum('abc,br,cr->ar', diff, V_init, W_init, optimize=True)
            gradV = np.einsum('abc,ar,cr->br', diff, U_init, W_init, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U_init, V_init, optimize=True)
            
            vU = momentum * vU + lr * gradU
            vV = momentum * vV + lr * gradV
            vW = momentum * vW + lr * gradW
            
            U_init -= vU
            V_init -= vV
            W_init -= vW
            
            loss = np.sum(diff * diff)
            
            if loss < prev_loss - 1e-10:
                prev_loss = loss
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve > patience:
                break
            
            if loss < 1e-10:
                break
        
        # Try to snap
        U_snap = snap_to_half_integers(U_init)
        V_snap = snap_to_half_integers(V_init)
        W_snap = snap_to_half_integers(W_init)
        
        if (np.allclose(U_init, U_snap, atol=1e-4) and 
            np.allclose(V_init, V_snap, atol=1e-4) and 
            np.allclose(W_init, W_snap, atol=1e-4)):
            prod = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap, optimize=True)
            if np.allclose(prod, T, atol=1e-4):
                U_list = [[float(x) for x in row] for row in U_snap]
                V_list = [[float(x) for x in row] for row in V_snap]
                W_list = [[float(x) for x in row] for row in W_snap]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    return schoolbook()

# EVOLVE-BLOCK-END
