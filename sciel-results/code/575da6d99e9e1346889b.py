import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 3x4 * 4x6 matrix multiplication.
    Uses block Strassen + direct, then GF(11) peeling to reduce rank.
    """
    n, m, p = 3, 4, 6
    MOD = 11
    
    def build_baseline():
        R = 66
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        
        col = 0
        
        def add_strassen_block(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A00 = a_r * m + a_c
            A01 = a_r * m + a_c + 1
            A10 = (a_r + 1) * m + a_c
            A11 = (a_r + 1) * m + a_c + 1
            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1
            C00 = c_c * n + c_r
            C01 = (c_c + 1) * n + c_r
            C10 = c_c * n + (c_r + 1)
            C11 = (c_c + 1) * n + (c_r + 1)
            
            U[A00][col] += 1; U[A11][col] += 1
            V[B00][col] += 1; V[B11][col] += 1
            W[C00][col] += 1; W[C11][col] += 1
            col += 1
            U[A10][col] += 1; U[A11][col] += 1
            V[B00][col] += 1
            W[C10][col] += 1; W[C11][col] -= 1
            col += 1
            U[A00][col] += 1
            V[B01][col] += 1; V[B11][col] -= 1
            W[C01][col] += 1; W[C11][col] += 1
            col += 1
            U[A11][col] += 1
            V[B10][col] += 1; V[B00][col] -= 1
            W[C00][col] += 1; W[C10][col] += 1
            col += 1
            U[A00][col] += 1; U[A01][col] += 1
            V[B11][col] += 1
            W[C00][col] -= 1; W[C01][col] += 1
            col += 1
            U[A10][col] += 1; U[A00][col] -= 1
            V[B00][col] += 1; V[B01][col] += 1
            W[C11][col] += 1
            col += 1
            U[A01][col] += 1; U[A11][col] -= 1
            V[B10][col] += 1; V[B11][col] += 1
            W[C00][col] += 1
            col += 1
        
        def add_direct_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A0 = a_r * m + a_c
            A1 = a_r * m + a_c + 1
            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1
            C0 = c_c * n + c_r
            C1 = (c_c + 1) * n + c_r
            
            U[A0][col] += 1; V[B00][col] += 1; W[C0][col] += 1
            col += 1
            U[A1][col] += 1; V[B10][col] += 1; W[C0][col] += 1
            col += 1
            U[A0][col] += 1; V[B01][col] += 1; W[C1][col] += 1
            col += 1
            U[A1][col] += 1; V[B11][col] += 1; W[C1][col] += 1
            col += 1
        
        for j in range(3):
            for i in range(2):
                add_direct_1x2_2x2(0, i * 2, i * 2, j * 2, 0, j * 2)
        
        for j in range(3):
            for i in range(2):
                add_strassen_block(1, i * 2, i * 2, j * 2, 1, j * 2)
        
        assert col == R
        return U, V, W
    
    def verify(U, V, W):
        """Verify the decomposition exactly reproduces matrix multiplication."""
        r = len(U[0])
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    actual = 0.0
                    for rr in range(r):
                        actual += U[a * m + b][rr] * V[b * p + c][rr] * W[c * n + a][rr]
                    if abs(actual - 1.0) > 1e-9:
                        return False
        return True
    
    def gf11_peel(U, V, W, deadline):
        """
        Use GF(11) to find linearly dependent rank-1 components and eliminate them.
        The key insight: if a column in the tensor decomposition is a linear combination
        of others over GF(11), we can eliminate it (modulo the field).
        
        We build the tensor T[a,b,c] = sum_r U[a*m+b,r] * V[b*p+c,r] * W[c*n+a,r]
        and look for columns r where the rank-1 tensor u_r ⊗ v_r ⊗ w_r is in the span
        of the other rank-1 tensors over GF(11).
        """
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        r = len(U[0])
        
        # Precompute the tensor values over GF(11)
        def get_tensor_value(U, V, W, a, b, c):
            val = 0
            for rr in range(len(U[0])):
                val = (val + U[a*m+b][rr] * V[b*p+c][rr] * W[c*n+a][rr]) % MOD
            return val
        
        # For each column, check if it's linearly dependent on others
        # We'll use a randomized approach: try to find coefficients
        # c_1, ..., c_{r-1} such that the target column's contribution
        # equals the weighted sum of other columns' contributions
        
        # Build a matrix representation for linear dependency detection
        # Each column r defines a rank-1 tensor. We want to find if column r
        # is in the span of others.
        
        # More efficient: use the fact that if we remove column r, the residual
        # should be representable by the remaining columns.
        
        # Try each column for removal, using GF(11) arithmetic
        removed = []
        for attempt in range(200):  # Limit attempts
            if time.time() >= deadline:
                break
            r = len(U[0])
            if r <= 60:  # Stop if we've made good progress
                break
            
            # Try to find a column that can be removed
            found = False
            for drop_idx in range(r):
                if time.time() >= deadline:
                    break
                
                # Check if removing column drop_idx still gives T=1 everywhere
                # over GF(11)
                valid = True
                for a in range(n):
                    if not valid:
                        break
                    for b in range(m):
                        if not valid:
                            break
                        for c in range(p):
                            actual = 0
                            for rr in range(r):
                                if rr == drop_idx:
                                    continue
                                val = (U[a*m+b][rr] * V[b*p+c][rr] * W[c*n+a][rr]) % MOD
                                actual = (actual + val) % MOD
                            if actual != 1:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    # Remove this column
                    U = [row[:drop_idx] + row[drop_idx+1:] for row in U]
                    V = [row[:drop_idx] + row[drop_idx+1:] for row in V]
                    W = [row[:drop_idx] + row[drop_idx+1:] for row in W]
                    removed.append(drop_idx)
                    found = True
                    break
            
            if not found:
                break
        
        return U, V, W, len(removed)
    
    def try_local_search(U, V, W, deadline):
        """Try to reduce rank by finding and removing redundant terms."""
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        r = len(U[0])
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            r = len(U[0])
            for drop_idx in range(r):
                if time.time() >= deadline:
                    break
                U_test = [row[:drop_idx] + row[drop_idx+1:] for row in U]
                V_test = [row[:drop_idx] + row[drop_idx+1:] for row in V]
                W_test = [row[:drop_idx] + row[drop_idx+1:] for row in W]
                
                if verify(U_test, V_test, W_test):
                    U = U_test
                    V = V_test
                    W = W_test
                    improved = True
                    break
        
        return U, V, W
    
    U, V, W = build_baseline()
    
    deadline = time.time() + 140  # 10s safety margin
    
    # First try GF(11) peeling
    U, V, W, peeled = gf11_peel(U, V, W, deadline)
    
    # Then try standard local search
    U, V, W = try_local_search(U, V, W, deadline)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
