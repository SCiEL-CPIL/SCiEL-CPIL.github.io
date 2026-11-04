import numpy as np
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    start_time = time.time()
    deadline = 130.0

    def to_lists(mat):
        return mat.tolist()

    # Build target tensor
    T_target = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m + j, j*p + k, k*n + i] = 1.0

    def verify(U, V, W, atol=1e-6):
        T = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T, T_target, atol=atol)

    def schoolbook():
        r = n * m * p
        U = np.zeros((nm, r)); V = np.zeros((mp, r)); W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W

    # Attempt to find a rank-16 decomposition using structured search
    # Based on the observation that rank(2x3x3) should be 16.
    # We'll use a hybrid approach:
    # 1. Try known explicit constructions
    # 2. Use iterative refinement with multiple random starts
    
    def try_explicit_constructions():
        """Try several explicit rank-16 constructions."""
        constructions = []
        
        # Construction 1: Based on splitting into 2x2 and 1x3 blocks
        # Use the fact that we can decompose as sum of rank-1 terms
        # where each term has a specific support pattern.
        
        # Try a pattern where we use 8 terms for "even" indices and 8 for "odd"
        for pattern_id in range(5):
            r = 16
            U = np.zeros((nm, r), dtype=np.float64)
            V = np.zeros((mp, r), dtype=np.float64)
            W = np.zeros((pn, r), dtype=np.float64)
            
            # Assign terms based on different patterns
            for t in range(r):
                if pattern_id == 0:
                    # Pattern: cyclic shifts
                    i = t % n
                    j = (t // n) % m
                    k = (t // (n*m)) % p
                    U[i*m + j, t] = 1.0
                    V[j*p + k, t] = 1.0
                    W[k*n + i, t] = 1.0
                    # Add a second term with shifted indices
                    i2 = (i + 1) % n
                    j2 = (j + 1) % m
                    k2 = (k + 1) % p
                    U[i2*m + j2, t] = 1.0
                    V[j2*p + k2, t] = 1.0
                    W[k2*n + i2, t] = 1.0
                elif pattern_id == 1:
                    # Pattern: pair-based
                    i = t % n
                    j = (t // n) % m
                    k = (t // (n*m)) % p
                    U[i*m + j, t] = 1.0
                    V[j*p + k, t] = 1.0
                    W[k*n + i, t] = 1.0
                    # Alternate between two k values
                    if t < 8:
                        k_alt = (k + 1) % p
                        V[j*p + k_alt, t] = 0.5
                        W[k_alt*n + i, t] = 0.5
                    else:
                        k_alt = (k + 2) % p
                        V[j*p + k_alt, t] = 0.5
                        W[k_alt*n + i, t] = 0.5
                elif pattern_id == 2:
                    # Pattern: use Strassen-like combinations
                    i = t % n
                    j = (t // n) % m
                    k = (t // (n*m)) % p
                    U[i*m + j, t] = 1.0
                    V[j*p + k, t] = 1.0
                    W[k*n + i, t] = 1.0
                    # Add cross terms
                    if j < m - 1:
                        U[i*m + j + 1, t] = 1.0
                        V[(j + 1)*p + k, t] = 1.0
                    if k < p - 1:
                        V[j*p + k + 1, t] = 1.0
                        W[(k + 1)*n + i, t] = 1.0
                elif pattern_id == 3:
                    # Pattern: use different weight distributions
                    i = t % n
                    j = (t // n) % m
                    k = (t // (n*m)) % p
                    U[i*m + j, t] = 1.0
                    V[j*p + k, t] = 1.0
                    W[k*n + i, t] = 1.0
                    # Spread weights across nearby indices
                    for di in range(-1, 2):
                        for dj in range(-1, 2):
                            ii = (i + di) % n
                            jj = (j + dj) % m
                            if (ii, jj) != (i, j):
                                U[ii*m + jj, t] = 0.5
                                V[jj*p + k, t] = 0.5
                elif pattern_id == 4:
                    # Pattern: use a grid-based approach
                    i = t % n
                    j = (t // n) % m
                    k = (t // (n*m)) % p
                    U[i*m + j, t] = 1.0
                    V[j*p + k, t] = 1.0
                    W[k*n + i, t] = 1.0
                    # Use a different second term based on t
                    if t % 2 == 0:
                        U[i*m + (j + 1) % m, t] = 1.0
                        V[(j + 1) % m * p + k, t] = 1.0
                    else:
                        U[i*m + (j + 2) % m, t] = 1.0
                        V[(j + 2) % m * p + k, t] = 1.0
            
            if verify(U, V, W):
                return U, V, W
        
        return None

    def polish_decomposition(U, V, W, max_iter=3000, lr=0.02):
        """Polish a decomposition using gradient descent."""
        r = U.shape[1]
        
        for it in range(max_iter):
            if time.time() - start_time > deadline - 10:
                break
            
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            diff = T_approx - T_target
            
            # Compute gradients
            grad_U = np.einsum('abc,bt,ct->at', diff, V, W)
            grad_V = np.einsum('abc,at,ct->bt', diff, U, W)
            grad_W = np.einsum('abc,at,bt->ct', diff, U, V)
            
            # Update with adaptive learning rate
            err = np.sum(diff**2)
            if err < 1e-12:
                break
            
            # Adaptive step size
            step = lr / (1 + it * 0.001)
            U -= step * grad_U
            V -= step * grad_V
            W -= step * grad_W
        
        return U, V, W

    def round_and_verify(U, V, W):
        """Round to nearest half-integer and verify."""
        # Try rounding to half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if verify(U_r, V_r, W_r):
            return U_r, V_r, W_r
        
        # Try rounding to integers
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if verify(U_i, V_i, W_i):
            return U_i, V_i, W_i
        
        return None

    # Try explicit constructions first
    result = try_explicit_constructions()
    if result is not None:
        U, V, W = result
        return {"U": to_lists(U), "V": to_lists(V), "W": to_lists(W)}

    # Try multiple random starts with polishing
    best_err = np.inf
    best_result = None
    
    for seed in range(30):
        if time.time() - start_time > deadline - 15:
            break
        
        np.random.seed(seed * 7919 + 13)
        
        # Structured random initialization
        U0 = np.zeros((nm, 16), dtype=np.float64)
        V0 = np.zeros((mp, 16), dtype=np.float64)
        W0 = np.zeros((pn, 16), dtype=np.float64)
        
        for t in range(16):
            # Each term should have a meaningful support
            # U: support on (i,j) pairs
            # V: support on (j,k) pairs
            # W: support on (k,i) pairs
            
            # Choose a base (i,j,k) triple
            i0 = np.random.randint(0, n)
            j0 = np.random.randint(0, m)
            k0 = np.random.randint(0, p)
            
            # Primary term
            U0[i0*m + j0, t] = np.random.choice([-1.0, 1.0])
            V0[j0*p + k0, t] = np.random.choice([-1.0, 1.0])
            W0[k0*n + i0, t] = np.random.choice([-1.0, 1.0])
            
            # Secondary terms to create structure
            if np.random.rand() < 0.7:
                i1 = (i0 + np.random.randint(0, 2)) % n
                j1 = (j0 + np.random.randint(0, 2)) % m
                k1 = (k0 + np.random.randint(0, 2)) % p
                U0[i1*m + j1, t] = np.random.choice([-1.0, 0.0, 1.0])
                V0[j1*p + k1, t] = np.random.choice([-1.0, 0.0, 1.0])
                W0[k1*n + i1, t] = np.random.choice([-1.0, 0.0, 1.0])
            
            # Tertiary terms for more structure
            if np.random.rand() < 0.4:
                i2 = (i0 + np.random.randint(0, 3)) % n
                j2 = (j0 + np.random.randint(0, 3)) % m
                k2 = (k0 + np.random.randint(0, 3)) % p
                U0[i2*m + j2, t] = np.random.choice([-0.5, 0.0, 0.5])
                V0[j2*p + k2, t] = np.random.choice([-0.5, 0.0, 0.5])
                W0[k2*n + i2, t] = np.random.choice([-0.5, 0.0, 0.5])
        
        # Polish
        U1, V1, W1 = polish_decomposition(U0.copy(), V0.copy(), W0.copy(), max_iter=2000, lr=0.02)
        
        # Check error
        T_approx = np.einsum('ar,br,cr->abc', U1, V1, W1)
        err = np.sum((T_approx - T_target)**2)
        
        if err < best_err:
            best_err = err
            best_result = (U1, V1, W1)
        
        # Try to round and verify
        result = round_and_verify(U1, V1, W1)
        if result is not None:
            U, V, W = result
            return {"U": to_lists(U), "V": to_lists(V), "W": to_lists(W)}
    
    # If we found a good result but couldn't round it exactly, try harder
    if best_result is not None:
        U1, V1, W1 = best_result
        # More aggressive polishing
        U1, V1, W1 = polish_decomposition(U1.copy(), V1.copy(), W1.copy(), max_iter=5000, lr=0.01)
        result = round_and_verify(U1, V1, W1)
        if result is not None:
            U, V, W = result
            return {"U": to_lists(U), "V": to_lists(V), "W": to_lists(W)}
    
    # Fallback: schoolbook rank 18
    U, V, W = schoolbook()
    return {"U": to_lists(U), "V": to_lists(V), "W": to_lists(W)}
# EVOLVE-BLOCK-END
