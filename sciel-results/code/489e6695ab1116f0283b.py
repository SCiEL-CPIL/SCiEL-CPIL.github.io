"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    nm, mp, pn = n*m, m*p, p*n
    
    # Check all valid (i,j,k) tuples: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    
    # Check that entries outside the support are zero
    # The support of the multiplication tensor consists of triples (a,b,c) where
    # a = i*m+j (i in [0,n), j in [0,m)), b = j*p+k (j in [0,m), k in [0,p)),
    # c = k*n+i (k in [0,p), i in [0,n)) with consistent i,j,k.
    # For a given (a,b,c), it's in the support if there exist i,j,k such that
    # a = i*m+j, b = j*p+k, c = k*n+i.
    # We need to verify that for (a,b,c) NOT in the support, the sum is 0.
    
    # Build set of valid (a,b,c) triples
    valid_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid_set.add((i*m+j, j*p+k, k*n+i))
    
    # Check all (a,b,c) not in valid_set have sum = 0
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                if (a,b,c) in valid_set:
                    continue
                s = 0.0
                for cc in range(rank):
                    s += U[a, cc] * V[b, cc] * W[c, cc]
                if abs(s) > 1e-9:
                    return False
    
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook rank-n*m*p decomposition."""
    nm, mp, pn = n*m, m*p, p*n
    rank = n*m*p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 x 3x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 145  # 145 second deadline with 5s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Start with schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # The theoretical lower bound for rank is max(n*m, m*p, p*n) = max(6, 18, 12) = 18
    # For matrix multiplication M_{n,m,p}, the rank is at least max(nm, mp, pn).
    # Also, for n=2, m=3, p=6, we can try to find better decompositions.
    
    # Strategy: Use alternating least squares / tensor decomposition approach
    # The multiplication tensor T has T[a,b,c] = 1 if (a,b,c) is in the support, 0 otherwise.
    # We want to find rank-r factors U, V, W such that T = sum_c U[:,c] V[:,c] W[:,c] (outer product)
    
    # Build the full tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try ranks from lower bound up to schoolbook rank - 1
    # Use a simple approach: random initialization + alternating minimization
    # For each rank, try multiple random initializations
    
    lower_bound = max(nm, mp, pn)
    
    for rank in range(lower_bound, best_rank):
        if time.time() > deadline - 10:
            break
        
        found = False
        max_attempts = 5
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(42 + rank * 1000 + attempt * 7)
            
            # Initialize factors
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            # Alternating minimization
            converged = False
            for iteration in range(200):
                if time.time() > deadline - 10:
                    break
                
                # Fix V and W, solve for U
                # For each row a of U, we need: sum_c U[a,c] * (V[:,c] * W[:,c] flattened) = T[a,:,:].flatten()
                # This is a linear system: U[a,:] @ M = T[a,:,:].flatten()
                # where M[c, (b,c_idx)] = V[b,c] * W[c_idx,c] ... wait, let me think more carefully.
                
                # T[a,b,c] = sum_d U[a,d] * V[b,d] * W[c,d]
                # For fixed V, W: T[a,:,:] = sum_d U[a,d] * (V[:,d] outer W[:,d])
                # So T[a,:,:].flatten() = sum_d U[a,d] * (V[:,d] * W[:,d]).flatten()
                # Let M be mp*pn x rank matrix where M[(b,c),d] = V[b,d] * W[c,d]
                # Then T[a,:,:].flatten() = U[a,:] @ M
                # So U[a,:] = T[a,:,:].flatten() @ pinv(M)
                
                M = np.zeros((mp*pn, rank))
                for d in range(rank):
                    M[:, d] = (V[:, d] * W[:, d]).flatten()
                
                # Solve U[a,:] for each a
                # U[a,:] @ M = T[a,:,:].flatten()
                # This is U[a,:] = T[a,:,:].flatten() @ M^+ (pseudo-inverse)
                try:
                    M_pinv = np.linalg.pinv(M)
                    for a in range(nm):
                        U[a, :] = T[a, :, :].flatten() @ M_pinv
                except np.linalg.LinAlgError:
                    break
                
                # Fix U and W, solve for V
                # T[a,b,c] = sum_d U[a,d] * V[b,d] * W[c,d]
                # For fixed U, W: T[:,b,:] = sum_d V[b,d] * (U[:,d] outer W[:,d])
                # T[:,b,:].flatten() = V[b,:] @ N where N[(a,c),d] = U[a,d] * W[c,d]
                N = np.zeros((nm*pn, rank))
                for d in range(rank):
                    N[:, d] = (U[:, d] * W[:, d]).flatten()
                
                try:
                    N_pinv = np.linalg.pinv(N)
                    for b in range(mp):
                        V[b, :] = T[:, b, :].flatten() @ N_pinv
                except np.linalg.LinAlgError:
                    break
                
                # Fix U and V, solve for W
                # T[a,b,c] = sum_d U[a,d] * V[b,d] * W[c,d]
                # For fixed U, V: T[:,:,c] = sum_d W[c,d] * (U[:,d] outer V[:,d])
                # T[:,:,c].flatten() = W[c,:] @ P where P[(a,b),d] = U[a,d] * V[b,d]
                P = np.zeros((nm*mp, rank))
                for d in range(rank):
                    P[:, d] = (U[:, d] * V[:, d]).flatten()
                
                try:
                    P_pinv = np.linalg.pinv(P)
                    for c in range(pn):
                        W[c, :] = T[:, :, c].flatten() @ P_pinv
                except np.linalg.LinAlgError:
                    break
                
                # Check convergence
                # Compute residual
                residual = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for d in range(rank):
                                s += U[i*m+j, d] * V[j*p+k, d] * W[k*n+i, d]
                            residual += (s - 1.0)**2
                if residual < 1e-10:
                    converged = True
                    break
            
            if converged:
                if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    best_rank = rank
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    found = True
                    break
        
        if found:
            # Try to go even lower
            continue
    
    return best_result
# EVOLVE-BLOCK-END
