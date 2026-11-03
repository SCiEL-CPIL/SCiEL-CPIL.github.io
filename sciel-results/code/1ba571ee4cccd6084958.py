import numpy as np
from sympy import symbols, solve, Poly, groebner, Rational

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Try ranks from lower bound upward
    lower_bound = max(n*m, m*p, p*n)
    
    for r in range(lower_bound, n*m*p + 1):
        # Create symbols for U, V, W
        # U: n*m x r, V: m*p x r, W: p*n x r
        U_syms = {}
        V_syms = {}
        W_syms = {}
        for i in range(n):
            for j in range(m):
                for t in range(r):
                    U_syms[(i*m+j, t)] = symbols(f"U_{i}_{j}_{t}")
        for j in range(m):
            for k in range(p):
                for t in range(r):
                    V_syms[(j*p+k, t)] = symbols(f"V_{j}_{k}_{t}")
        for k in range(p):
            for i in range(n):
                for t in range(r):
                    W_syms[(k*n+i, t)] = symbols(f"W_{k}_{i}_{t}")
        
        # Build the 36 equations
        equations = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq = 0
                    for t in range(r):
                        eq += U_syms[(i*m+j, t)] * V_syms[(j*p+k, t)] * W_syms[(k*n+i, t)]
                    equations.append(eq - 1)
        
        # This system is too large for direct Gröbner basis
        # Instead, let's try a different approach: fix some variables and solve
        # Or use a randomized approach with verification
        
        # Practical approach: use random initialization and ALS with half-integer rounding
        found = False
        for attempt in range(50):
            # Initialize with small rational numbers
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, r)).astype(float)
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, r)).astype(float)
            
            if np.all(V == 0): V[0,0] = 1
            if np.all(W == 0): W[0,0] = 1
            
            U = np.zeros((n*m, r))
            valid = True
            
            for step in range(20):
                # Solve for U
                for i in range(n):
                    for j in range(m):
                        A_ij = np.zeros((p, r))
                        for k in range(p):
                            for t in range(r):
                                A_ij[k, t] = V[j*p+k, t] * W[k*n+i, t]
                        b = np.ones(p)
                        try:
                            u, res, rank, s = np.linalg.lstsq(A_ij, b, rcond=None)
                            if np.allclose(A_ij @ u, b, atol=1e-9):
                                U[i*m+j, :] = u
                            else:
                                valid = False
                                break
                        except:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
                
                # Solve for V
                for j in range(m):
                    for k in range(p):
                        B_jk = np.zeros((n, r))
                        for i in range(n):
                            for t in range(r):
                                B_jk[i, t] = U[i*m+j, t] * W[k*n+i, t]
                        b = np.ones(n)
                        try:
                            v, res, rank, s = np.linalg.lstsq(B_jk, b, rcond=None)
                            if np.allclose(B_jk @ v, b, atol=1e-9):
                                V[j*p+k, :] = v
                            else:
                                valid = False
                                break
                        except:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
                
                # Solve for W
                for i in range(n):
                    for k in range(p):
                        C_ik = np.zeros((m, r))
                        for j in range(m):
                            for t in range(r):
                                C_ik[j, t] = U[i*m+j, t] * V[j*p+k, t]
                        b = np.ones(m)
                        try:
                            w, res, rank, s = np.linalg.lstsq(C_ik, b, rcond=None)
                            if np.allclose(C_ik @ w, b, atol=1e-9):
                                W[k*n+i, :] = w
                            else:
                                valid = False
                                break
                        except:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            
            if valid:
                # Round to half-integers
                U_r = np.round(2*U)/2
                V_r = np.round(2*V)/2
                W_r = np.round(2*W)/2
                
                # Verify
                ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for t in range(r):
                                val += U_r[i*m+j, t] * V_r[j*p+k, t] * W_r[k*n+i, t]
                            if abs(val - 1.0) > 1e-6:
                                ok = False
                                break
                        if not ok: break
                    if not ok: break
                
                if ok:
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    # Fallback: schoolbook
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
