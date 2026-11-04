import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    max_time = 140  # Safety margin
    
    n, m, p = 2, 3, 4
    target_rank = 20  # Known lower bound/improvement over 24
    
    # Target tensor T[i, j, k] = 1
    T = np.ones((n, m, p))
    
    def verify_decomposition(U, V, W, R):
        """Verify that U, V, W with rank R exactly decompose the tensor."""
        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R)
        A_approx = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return np.allclose(A_approx, T, atol=1e-9)
    
    def solve_for_one_matrix(fixed1, fixed2, target, shape_fixed1, shape_fixed2, shape_solve, R):
        """
        Solves for the third matrix given two fixed matrices.
        The equation is sum_c M1[a, c] * M2[b, c] * M3[c, d] = Target[a, b, d]
        This is linear in M3 if M1 and M2 are fixed.
        However, the indices are mixed.
        
        Let's construct the linear system A x = b.
        Variables are entries of M3.
        M3 has shape (dim3, R). Let's flatten it to x.
        
        For each index (i, j, k) in T:
        T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        
        If solving for W:
        T[i, j, k] = sum_r (U[i, j, r] * V[j, k, r]) * W[k, i, r]
        
        This is a linear equation for W[k, i, r].
        For a fixed (i, j, k), the coefficient of W[k, i, r] is U[i, j, r] * V[j, k, r].
        
        We have 24 equations (for each i, j, k).
        We have R * n * p variables? No, W is p x n x R. Size p*n*R.
        Wait, W[k, i, r]. k in 0..p-1, i in 0..n-1, r in 0..R-1.
        So variables are W[k, i, r].
        
        The system is:
        For each (i, j, k):
        sum_{r=0}^{R-1} (U[i, j, r] * V[j, k, r]) * W[k, i, r] = T[i, j, k]
        
        This is underdetermined if R is small?
        Number of variables: p * n * R = 4 * 2 * R = 8R.
        Number of equations: n * m * p = 2 * 3 * 4 = 24.
        
        If R=20, variables = 160. Equations = 24.
        We can use least squares to find a solution.
        """
        pass # Placeholder, logic below
    
    def try_rank(R, max_restarts=20):
        """
        Try to find a rank R decomposition using alternating least squares.
        """
        
        def unpack_and_solve_W(U, V):
            """Solve for W given U and V."""
            # Variables: W[k, i, r]
            # Eq: sum_r U[i, j, r] V[j, k, r] W[k, i, r] = T[i, j, k]
            
            A = np.zeros((n * m * p, R * p * n))
            b = T.flatten()
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Coefficients for W[k, i, r]
                        for r in range(R):
                            coeff = U[i, j, r] * V[j, k, r]
                            # W index: (k, i, r)
                            # Flatten index: k * (n * R) + i * R + r
                            w_idx = k * (n * R) + i * R + r
                            A[idx, w_idx] = coeff
                        idx += 1
                        
            # Solve A W_flat = b
            # Use least squares
            try:
                W_flat, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                W = W_flat.reshape(p, n, R).transpose(2, 0, 1) # W[r, k, i] -> W[k*n+i, r]?
                # Wait, W shape in output is (p*n, R).
                # My W_flat is indexed by (k, i, r).
                # Output W should be (p*n, R).
                # W_out[k*n + i, r] = W_flat[k, i, r]
                W_out = W_flat.reshape(p, n, R).transpose(1, 0, 2) # i, k, r? No.
                # Let's be careful.
                # W_flat index: k * (n*R) + i * R + r.
                # We want W_out[row, col] where row = k*n + i, col = r.
                # So W_out[k*n + i, r] = W_flat[k, i, r].
                # Reshape W_flat to (p, n, R).
                W_3d = W_flat.reshape(p, n, R)
                # Transpose to (n, p, R)? No.
                # We need (p*n, R).
                # W_out = W_3d.transpose(1, 0, 2).reshape(n*p, R)? No.
                # Row index is k*n + i.
                # So we need to arrange such that index (k, i) maps to row k*n + i.
                # W_3d is (p, n, R).
                # We want (p*n, R).
                # W_out = W_3d.transpose(1, 0, 2) gives (n, p, R).
                # Flatten first two dims: n*p.
                # But we want k*n + i.
                # So we need (k, i) order.
                # W_3d is indexed [k, i, r].
                # If we just reshape (p, n, R) to (p*n, R), the order is k, then i.
                # So W_out = W_3d.reshape(p*n, R).
                return W_3d.reshape(p*n, R)
            except:
                return None

        def unpack_and_solve_U(V, W):
            """Solve for U given V and W."""
            # Eq: sum_r U[i, j, r] V[j, k, r] W[k, i, r] = T[i, j, k]
            # Variables: U[i, j, r]
            A = np.zeros((n * m * p, R * n * m))
            b = T.flatten()
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(R):
                            coeff = V[j, k, r] * W[k, i, r]
                            u_idx = i * (m * R) + j * R + r
                            A[idx, u_idx] = coeff
                        idx += 1
                        
            try:
                U_flat, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                U_3d = U_flat.reshape(n, m, R)
                return U_3d.reshape(n*m, R)
            except:
                return None

        def unpack_and_solve_V(U, W):
            """Solve for V given U and W."""
            # Eq: sum_r U[i, j, r] V[j, k, r] W[k, i, r] = T[i, j, k]
            # Variables: V[j, k, r]
            A = np.zeros((n * m * p, R * m * p))
            b = T.flatten()
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(R):
                            coeff = U[i, j, r] * W[k, i, r]
                            v_idx = j * (p * R) + k * R + r
                            A[idx, v_idx] = coeff
                        idx += 1
                        
            try:
                V_flat, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                V_3d = V_flat.reshape(m, p, R)
                return V_3d.reshape(m*p, R)
            except:
                return None

        for attempt in range(max_restarts):
            if time.time() - start_time > max_time:
                return None
            
            # Initialize with sparse random values
            U = np.random.uniform(-1, 1, (n*m, R))
            V = np.random.uniform(-1, 1, (m*p, R))
            W = np.random.uniform(-1, 1, (p*n, R))
            
            # Sparsify
            U[np.abs(U) < 0.5] = 0
            V[np.abs(V) < 0.5] = 0
            W[np.abs(W) < 0.5] = 0
            
            # Alternating minimization
            for step in range(50):
                W = unpack_and_solve_W(U, V)
                if W is None: continue
                U = unpack_and_solve_U(V, W)
                if U is None: continue
                V = unpack_and_solve_V(U, W)
                if V is None: continue
                
                # Check residual
                U3 = U.reshape(n, m, R)
                V3 = V.reshape(m, p, R)
                W3 = W.reshape(p, n, R)
                A_approx = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
                res = np.linalg.norm(T - A_approx)
                
                if res < 1e-6:
                    # Try to snap to integers
                    U_snap = np.round(U)
                    V_snap = np.round(V)
                    W_snap = np.round(W)
                    
                    if verify_decomposition(U_snap, V_snap, W_snap, R):
                        return {"U": U_snap, "V": V_snap, "W": W_snap}
                    
                    # Try half-integers
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    if verify_decomposition(U_snap, V_snap, W_snap, R):
                        return {"U": U_snap, "V": V_snap, "W": W_snap}
                
                if time.time() - start_time > max_time:
                    return None

        return None

    # Try ranks from 20 up to 23
    for R in range(20, 24):
        dec = try_rank(R)
        if dec:
            return {
                "U": dec["U"].tolist(),
                "V": dec["V"].tolist(),
                "W": dec["W"].tolist()
            }
            
    # Fallback to schoolbook rank 24
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
