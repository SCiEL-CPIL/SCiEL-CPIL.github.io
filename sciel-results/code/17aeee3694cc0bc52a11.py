"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    def verify(U, V, W):
        """Verify exact decomposition. Returns True if valid."""
        n, m, p = DIMENSIONS
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        if V.shape[1] != r or W.shape[1] != r:
            return False
        # For each (i,j,k), check sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for t in range(r):
                        total += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Strategy: Model as linear code problem by fixing U to structured basis
    # and solving for V and W. Try various structured U patterns.
    
    def try_structured_U_W(r, U_pattern, W_pattern):
        """
        Try a specific structured U and W pattern, solve for V.
        U_pattern: callable or array-like for U
        W_pattern: callable or array-like for W
        Returns (U, V, W) if valid, else None
        """
        U = np.zeros((n * m, r))
        W = np.zeros((p * n, r))
        
        if callable(U_pattern):
            U_pattern(U, r)
        else:
            U = np.array(U_pattern, dtype=float)
            if U.shape != (n * m, r):
                return None
        
        if callable(W_pattern):
            W_pattern(W, r)
        else:
            W = np.array(W_pattern, dtype=float)
            if W.shape != (p * n, r):
                return None
        
        # Build linear system for V
        # For each (i,j,k): sum_t U[i*m+j,t] * W[k*n+i,t] * V[j*p+k,t] = 1
        # Unknowns: V[j*p+k, t] for j in [0,m), k in [0,p), t in [0,r)
        
        A = np.zeros((n * m * p, m * p * r))
        b = np.ones(n * m * p)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for t in range(r):
                        coeff = U[i * m + j, t] * W[k * n + i, t]
                        if coeff != 0:
                            var_idx = (j * p + k) * r + t
                            A[eq_idx, var_idx] += coeff
                    eq_idx += 1
        
        # Solve
        try:
            V_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
            V = V_flat.reshape(m * p, r)
            
            if verify(U, V, W):
                return U, V, W
        except:
            pass
        
        return None
    
    # Define various U patterns
    def U_mod_pattern(U, r):
        """U[i*m+j, t] = 1 if i*m+j == t % (n*m) else 0"""
        for t in range(r):
            idx = t % (n * m)
            U[idx, t] = 1.0
    
    def W_mod_pattern(W, r):
        """W[k*n+i, t] = 1 if k*n+i == t % (p*n) else 0"""
        for t in range(r):
            idx = t % (p * n)
            W[idx, t] = 1.0
    
    def U_kronecker_pattern(U, r):
        """Kronecker-like pattern: U[i*m+j, t] based on binary representation"""
        for t in range(r):
            # Map t to (a, b) where a in [0, n), b in [0, m)
            a = t % n
            b = (t // n) % m
            U[a * m + b, t] = 1.0
    
    def W_kronecker_pattern(W, r):
        """Kronecker-like pattern: W[k*n+i, t] based on binary representation"""
        for t in range(r):
            k = t % p
            i = (t // p) % n
            W[k * n + i, t] = 1.0
    
    def U_random_pattern(U, r, seed=42):
        """Random sparse pattern"""
        rng = np.random.RandomState(seed)
        for t in range(r):
            # Each column has ~2 non-zero entries
            num_nonzero = min(2, n * m)
            indices = rng.choice(n * m, size=num_nonzero, replace=False)
            for idx in indices:
                U[idx, t] = 1.0
    
    def W_random_pattern(W, r, seed=123):
        """Random sparse pattern"""
        rng = np.random.RandomState(seed)
        for t in range(r):
            num_nonzero = min(2, p * n)
            indices = rng.choice(p * n, size=num_nonzero, replace=False)
            for idx in indices:
                W[idx, t] = 1.0
    
    # Try different ranks and patterns
    patterns = [
        (U_mod_pattern, W_mod_pattern),
        (U_kronecker_pattern, W_kronecker_pattern),
        (U_random_pattern, W_random_pattern),
    ]
    
    for r in [80, 72, 64, 56, 48, 40]:
        for U_pat, W_pat in patterns:
            result = try_structured_U_W(r, U_pat, W_pat)
            if result is not None:
                U, V, W = result
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
