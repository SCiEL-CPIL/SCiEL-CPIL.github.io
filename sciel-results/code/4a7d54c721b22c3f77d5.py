import numpy as np
import time
from itertools import product as iproduct

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 28
    pn = p * n  # 14

    start_time = time.time()
    deadline = start_time + 120.0

    def schoolbook():
        rank = n * m * p  # 56
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, idx] = 1.0
                    V[j*p + k, idx] = 1.0
                    W[k*n + i, idx] = 1.0
                    idx += 1
        return U, V, W

    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p

    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m + j, c] * V[jp*p + k, c] * W[k*n + i, c]
                        expected = 1.0 if j == jp else 0.0
                        if abs(s - expected) > tol:
                            return False
        return True

    def time_left():
        return deadline - time.time()

    # Strategy: Use the structure that W[k*n+i, c] must be nonzero only
    # for specific (i,k) pairs. Since the output C[i,k] depends on all j,
    # we can try to find a decomposition where each rank-1 term contributes
    # to a specific subset of outputs.
    
    # Alternative approach: use the fact that we can combine the 4 terms
    # in each dot product using precomputed sums.
    
    # Let's try: for each output (i,k), C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    # We can write this as:
    # C[i,k] = (A[i,0]+A[i,1])*(B[0,k]+B[1,k]) - A[i,0]*B[1,k] - A[i,1]*B[0,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    # This doesn't obviously help.
    
    # Better: Use the identity: ab + cd = (a+c)(b+d) - (a-d)(b-d) - 2(ad+cb) ... not helpful.
    
    # Let's try a direct approach: assume entries are in {-1, 0, 1, 2, -2}
    # and use a smarter search.
    
    # Key observation: The tensor is separable in the i,k indices given j=j'.
    # Specifically, T[i,j,k,j',k'] = delta[j,j'] * delta[k,k'] for the (i,k) output.
    # Wait, no. Let me re-examine.
    
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the "tensor" we're decomposing has: for each (i,k), the bilinear form is sum_j A[i,j]B[j,k]
    # The rank-1 decomposition means: C[i,k] = sum_c u_{i,j,c} v_{j,k,c} w_{i,k,c}
    # where we sum over j for each c... no wait.
    
    # Actually the standard tensor rank decomposition for matrix mult (n×m, m×p)→(n×p):
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # = sum_c (sum_j A_{i,j} u_{j,c}) (sum_k B_{j,k} v_{k,c}) ... no.
    
    # The correct form: C_{i,k} = sum_c U_{i,c} V_{k,c} W_c where
    # U_{i,c} is a linear form in A, V_{k,c} is a linear form in B.
    
    # But the interface uses: U[i*m+j, c] which suggests U is indexed by (i,j) pairs.
    # And the constraint is: sum_c U[i*m+j, c] V[j'*p+k, c] W[k*n+i, c] = delta[j,j']
    
    # This means: for each (i,k), sum_j (sum_c U[i*m+j,c] W[k*n+i,c]) (sum_c V[j*p+k,c] W[k*n+i,c]) ... no.
    
    # Let me re-derive. We want C[i,k] = sum_j A[i,j] B[j,k].
    # If we write C[i,k] = sum_c X_c(i) Y_c(k) where X_c is linear in A and Y_c is linear in B,
    # then X_c(i) = sum_j A[i,j] alpha_{c,i,j} and Y_c(k) = sum_j B[j,k] beta_{c,j,k}.
    # Then C[i,k] = sum_c (sum_j A[i,j] alpha_{c,i,j})(sum_j' B[j',k] beta_{c,j',k})
    # = sum_j sum_j' A[i,j] B[j',k] (sum_c alpha_{c,i,j} beta_{c,j',k})
    # We need sum_c alpha_{c,i,j} beta_{c,j',k} = delta[j,j'] for all i,k.
    
    # In the interface: U[i*m+j, c] = alpha_{c,i,j} * something? Let's see.
    # The constraint is: sum_c U[i*m+j, c] V[j'*p+k, c] W[k*n+i, c] = delta[j,j']
    
    # If we set W[k*n+i, c] = 1 for all i,k,c, then we'd need sum_c U[i*m+j,c] V[j'*p+k,c] = delta[j,j']
    # which is a rank-r decomposition of the "identity" tensor in j,j' for each (i,k).
    
    # More generally, W allows us to weight different (i,k) pairs differently.
    
    # Let's try: assume W entries are in {0, 1} (or small integers), and search.
    
    # Even better: notice that the constraint must hold for ALL i,k simultaneously.
    # So for a given c, the term U[i*m+j,c] * V[j'*p+k,c] * W[k*n+i,c] must contribute
    # to all (i,k) in a way that sums to delta[j,j'].
    
    # Let's define: for each c, let a_c(i,j) = U[i*m+j,c], b_c(j,k) = V[j*p+k,c], w_c(i,k) = W[k*n+i,c].
    # Then we need: sum_c a_c(i,j) * b_c(j',k) * w_c(i,k) = delta[j,j'] for all i,k,j,j'.
    
    # This is a 4-index tensor equation. The left side is a 4-tensor T[i,j,k,j'] = sum_c a_c(i,j)b_c(j',k)w_c(i,k).
    # We need T[i,j,k,j'] = delta[j,j'] (independent of i,k).
    
    # So T[i,j,k,j'] = delta[j,j'] for all i,k. This means for each fixed (i,k), the matrix
    # M_{j,j'} = sum_c a_c(i,j)b_c(j',k)w_c(i,k) must equal the identity matrix.
    
    # So for each (i,k): sum_c w_c(i,k) a_c(i,j) b_c(j',k) = I_{j,j'}
    
    # This is like: for each (i,k), the product of the "i-row" of A-matrix and "k-column" of B-matrix
    # weighted by w_c(i,k) must give identity.
    
    # Let me define for each c: A_c is an n×m matrix (A_c[i,j] = a_c(i,j)), B_c is m×p matrix (B_c[j,k] = b_c(j,k)), W_c is p×n matrix (W_c[k,i] = w_c(i,k)).
    # Then for each (i,k): sum_c W_c[k,i] * (A_c * B_c)[i,k] ... no wait.
    
    # (A_c * B_c)[i,k] = sum_j A_c[i,j] B_c[j,k]
    # But we have sum_c W_c[k,i] * sum_j A_c[i,j] B_c[j,k] = sum_j sum_c W_c[k,i] A_c[i,j] B_c[j,k]
    
    # But we need sum_c W_c[k,i] A_c[i,j] B_c[j',k] = delta[j,j']
    
    # These are different! In the first, j is summed over; in the second, j and j' are separate.
    
    # So the correct interpretation: for each (i,k), we need the matrix (in j,j') defined by
    # (M_c[i,k])_{j,j'} = W_c[k,i] * A_c[i,j] * B_c[j',k]
    # to sum to identity over c.
    
    # So M_c[i,k] = W_c[k,i] * outer(A_c[i,:], B_c[:,k]) where A_c[i,:] is row i of A_c, B_c[:,k] is column k of B_c.
    
    # And sum_c M_c[i,k] = I_m for all i,k.
    
    # This is a strong constraint. Let's see if we can satisfy it.
    
    # For simplicity, let's assume W_c[k,i] is either 0 or 1, and A_c[i,j], B_c[j,k] are in {-1,0,1}.
    
    # Let's try a systematic search with small rank.
    
    def search_rank(rank, max_iters=500):
        if time_left() < 5.0:
            return None
        
        # Use random restarts with local improvement
        for trial in range(max_iters):
            if time_left() < 5.0:
                return None
            
            # Random initialization with small integers
            U = np.random.randint(-2, 3, size=(nm, rank))
            V = np.random.randint(-2, 3, size=(mp, rank))
            W = np.random.randint(-2, 3, size=(pn, rank))
            
            if verify(U, V, W):
                return U, V, W
            
            # Try flipping random entries
            for _ in range(10):
                if time_left() < 5.0:
                    break
                # Flip a random entry in U
                ui = np.random.randint(nm)
                uc = np.random.randint(rank)
                old = U[ui, uc]
                U[ui, uc] = np.random.randint(-2, 3)
                if verify(U, V, W):
                    return U, V, W
                U[ui, uc] = old
                
                # Flip a random entry in V
                vi = np.random.randint(mp)
                vc = np.random.randint(rank)
                old = V[vi, vc]
                V[vi, vc] = np.random.randint(-2, 3)
                if verify(U, V, W):
                    return U, V, W
                V[vi, vc] = old
                
                # Flip a random entry in W
                wi = np.random.randint(pn)
                wc = np.random.randint(rank)
                old = W[wi, wc]
                W[wi, wc] = np.random.randint(-2, 3)
                if verify(U, V, W):
                    return U, V, W
                W[wi, wc] = old
    
        return None

    # Try ranks from 55 down
    for rank in range(55, 39, -1):
        if time_left() < 10.0:
            break
        result = search_rank(rank, max_iters=200)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = rank
            break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
