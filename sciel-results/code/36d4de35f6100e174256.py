import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 5]

def construct():
    """
    Constructs the matrix multiplication tensor decomposition for 2x3 by 3x5.
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices.
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    
    def verify(U, V, W):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        # Build the tensor from the decomposition
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # The (i,j,k) entry should be 1
                    val = 0.0
                    for t in range(r):
                        val += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Check off-diagonal entries
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    # Check if (a,b,c) is a valid (i,j,k) triple
                    i = a // m
                    j = a % m
                    jj = b // p
                    k = b % p
                    kk = c // n
                    ii = c % n
                    if i == ii and j == jj and k == kk:
                        continue  # This is a diagonal entry, already checked
                    val = 0.0
                    for t in range(r):
                        val += U[a, t] * V[b, t] * W[c, t]
                    if abs(val) > 1e-9:
                        return False
        return True
    
    def try_rank29():
        """Try to find a rank-29 decomposition using iterative methods."""
        if time.time() > deadline:
            return None
        
        r = 29
        num_outputs = p * n  # 10
        num_U_rows = n * m   # 6
        num_V_rows = m * p   # 15
        
        # Precompute valid (i,j,k) triples
        valid_triples = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    valid_triples.append((i, j, k))
        
        # Gauge: W[c, c] = 1 for c = 0..9, W[c, r] = 0 for r >= 10 when c < 10
        # W is fixed: W[c, r] = 1 if c == r and r < 10, else 0
        W = np.zeros((num_outputs, r))
        for c in range(num_outputs):
            W[c, c] = 1.0
        
        # We need to find U (6 x 29) and V (15 x 29)
        # Equations:
        # For each valid (i,j,k) with c = k*n+i:
        #   U[i*m+j, c] * V[j*p+k, c] + sum_{r=10}^{28} U[i*m+j, r] * V[j*p+k, r] * W[c, r] = 1
        #   But W[c, r] = 0 for r >= 10, so:
        #   U[i*m+j, c] * V[j*p+k, c] = 1
        #   => V[j*p+k, c] = 1 / U[i*m+j, c]
        #
        # For off-diagonal (a,b,c) where (a,b,c) is not a valid triple:
        #   sum_{r=0}^{28} U[a, r] * V[b, r] * W[c, r] = 0
        #   Since W[c, r] = 1 if r == c < 10, else 0:
        #   U[a, c] * V[b, c] = 0 for c < 10
        #   This means for each c < 10, U[a, c] * V[b, c] = 0 for all off-diagonal (a,b)
        
        # Wait, this is very restrictive! For each c in 0..9:
        #   U[a, c] * V[b, c] = 0 for all off-diagonal (a,b) pairs
        #   And U[i*m+j, c] * V[j*p+k, c] = 1 for the diagonal (a,b) = (i*m+j, j*p+k) where c = k*n+i
        
        # This means for each c, U[:, c] and V[:, c] must be such that their outer product
        # has 1 at the diagonal position and 0 everywhere else.
        # This is only possible if U[:, c] and V[:, c] are sparse in a compatible way.
        
        # Actually, let me reconsider. The gauge W[c,r] = 1 if c==r and r<10 means:
        # For output c (0..9), only term r=c contributes with weight 1.
        # So the equation for output c is just:
        #   U[a, c] * V[b, c] = 1 if (a,b,c) is diagonal
        #   U[a, c] * V[b, c] = 0 if (a,b,c) is off-diagonal
        
        # This means each "slice" c must independently satisfy the matrix multiplication pattern!
        # But that's impossible with rank 1 per slice - we'd need rank 30 total.
        
        # So the gauge W[c,r] = delta_{cr} for r<10 is too restrictive.
        # We need a different gauge or a different approach.
        
        # Alternative: use a more flexible gauge. Let W be arbitrary but fix
        # some entries to break the gauge symmetry.
        
        # Let's try: fix W[c, 0] = 1 for all c = 0..9 (all outputs use term 0 with weight 1)
        # Then for each (i,j,k) with c = k*n+i:
        #   U[i*m+j, 0] * V[j*p+k, 0] + sum_{r=1}^{28} U[i*m+j, r] * V[j*p+k, r] * W[c, r] = 1
        # For off-diagonal (a,b,c):
        #   U[a, 0] * V[b, 0] + sum_{r=1}^{28} U[a, r] * V[b, r] * W[c, r] = 0
        
        # This is still nonlinear. Let me try a different strategy:
        # Start from schoolbook (rank 30) and try to merge two terms.
        
        # Schoolbook terms: for each (i,j,k), term t = i*m*p + j*p + k
        # U[i*m+j, t] = 1, V[j*p+k, t] = 1, W[k*n+i, t] = 1
        
        # Can we merge terms (i1,j1,k1) and (i2,j2,k2) into one term?
        # We need to find u, v, w vectors such that:
        #   u[a] * v[b] * w[c] = 1 for (a,b,c) = (i1*m+j1, j1*p+k1, k1*n+i1)
        #   u[a] * v[b] * w[c] = 1 for (a,b,c) = (i2*m+j2, j2*p+k2, k2*n+i2)
        #   u[a] * v[b] * w[c] = 0 for all other (a,b,c)
        
        # This requires that the two triples have the same "pattern" in a certain sense.
        # Specifically, we need:
        #   u[i1*m+j1] * v[j1*p+k1] * w[k1*n+i1] = 1
        #   u[i2*m+j2] * v[j2*p+k2] * w[k2*n+i2] = 1
        #   And all other products = 0
        
        # This means u, v, w must be zero except at specific positions.
        # If i1*m+j1 != i2*m+j2, then u must be nonzero at both positions,
        # which means u[a] * v[b] * w[c] could be nonzero for other (b,c) combinations.
        
        # Let's enumerate all pairs of terms and check if they can be merged.
        # There are C(30, 2) = 435 pairs.
        
        merged_found = False
        for trial in range(200):  # Try up to 200 random restarts
            if time.time() > deadline:
                break
            
            # Randomly select two terms to merge
            idx1 = np.random.randint(0, 30)
            idx2 = np.random.randint(0, 30)
            if idx1 == idx2:
                continue
            
            i1, j1, k1 = idx1 // (m*p), (idx1 // p) % m, idx1 % p
            i2, j2, k2 = idx2 // (m*p), (idx2 // p) % m, idx2 % p
            
            # Check if these two terms can be merged
            # They share the same output c if k1*n+i1 == k2*n+i2
            c1 = k1 * n + i1
            c2 = k2 * n + i2
            
            if c1 != c2:
                continue  # Different outputs, harder to merge
            
            # Try to find U, V, W for the merged term
            # The merged term must contribute 1 to both (i1,j1,k1) and (i2,j2,k2)
            # and 0 to all other (i,j,k)
            
            # Let's set up the constraints:
            # u[i1*m+j1] * v[j1*p+k1] * w[c1] = 1
            # u[i2*m+j2] * v[j2*p+k2] * w[c1] = 1
            # For all other (i,j,k): u[i*m+j] * v[j*p+k] * w[k*n+i] = 0
            
            # From the first two equations:
            # u[i1*m+j1] * v[j1*p+k1] = u[i2*m+j2] * v[j2*p+k2] = 1/w[c1]
            
            # Let's set w[c1] = 1, so u[i1*m+j1] * v[j1*p+k1] = 1 and u[i2*m+j2] * v[j2*p+k2] = 1
            
            # For the off-diagonal constraints, we need:
            # u[i*m+j] * v[j*p+k] * w[k*n+i] = 0 for all (i,j,k) != (i1,j1,k1) and (i2,j2,k2)
            
            # This is complex. Let me try a specific construction:
            # Set u[i1*m+j1] = 1, u[i2*m+j2] = 1 (if different)
            # Set v[j1*p+k1] = 1, v[j2*p+k2] = 1 (if different)
            # Set w[c1] = 1
            # All other entries of u, v, w are 0
            
            # Check if this works:
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            
            u[i1*m+j1] = 1.0
            u[i2*m+j2] = 1.0
            v[j1*p+k1] = 1.0
            v[j2*p+k2] = 1.0
            w[c1] = 1.0
            
            # Verify
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        c = k*n+i
                        expected = 1.0 if (i==i1 and j==j1 and k==k1) or (i==i2 and j==j2 and k==k2) else 0.0
                        val = u[i*m+j] * v[j*p+k] * w[c]
                        if abs(val - expected) > 1e-9:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                # We can merge these two terms! Build the full rank-29 decomposition.
                # Start with schoolbook, remove terms idx1 and idx2, add the merged term.
                U = np.zeros((n*m, 29))
                V = np.zeros((m*p, 29))
                W = np.zeros((p*n, 29))
                
                # Add all schoolbook terms except idx1 and idx2
                col = 0
                for t in range(30):
                    if t == idx1 or t == idx2:
                        continue
                    i, j, k = t // (m*p), (t // p) % m, t % p
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
                
                # Add the merged term
                U[:, col] = u
                V[:, col] = v
                W[:, col] = w
                
                if verify(U.tolist(), V.tolist(), W.tolist()):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None
    
    result29 = try_rank29()
    if result29 is not None:
        best_result = result29
    
    return best_result
