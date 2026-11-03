import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build target tensor
    targets = np.zeros(nm * mp * pn)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                idx = (I * mp + J) * pn + K
                targets[idx] = 1.0

    def residual(params):
        R = len(params) // (nm + mp + pn)
        if R == 0:
            return np.zeros(len(targets))
        U = params[:nm*R].reshape(nm, R)
        V = params[nm*R:nm*R+mp*R].reshape(mp, R)
        W = params[nm*R+mp*R:].reshape(pn, R)
        T = np.einsum('ir,jr,kr->IJK', U, V, W)
        return T.flatten() - targets

    # Try rank 17
    R = 17
    num_vars = R * (nm + mp + pn)
    
    # Structured Initialization for Rank 17
    # We use Strassen for the 2x2 core (indices 0,1 in n; 0,1 in m; 0,1 in p)
    # And simple monomials for the rest.
    
    U_init = np.zeros((nm, R))
    V_init = np.zeros((mp, R))
    W_init = np.zeros((pn, R))
    
    # Strassen products for A[0:2, 0:2] * B[0:2, 0:2]
    # A indices: 0:(0,0), 1:(0,1), 3:(1,0), 4:(1,1)
    # B indices: 0:(0,0), 1:(0,1), 3:(1,0), 4:(1,1)
    # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> Wait, K = k*n + i.
    # n=2, p=3.
    # K for i=0, k=0 is 0.
    # K for i=0, k=1 is 1.
    # K for i=0, k=2 is 2.
    # K for i=1, k=0 is 3.
    # K for i=1, k=1 is 4.
    # K for i=1, k=2 is 5.
    
    # Strassen M1: (A00+A11)(B00+B11)
    U_init[0, 0] = 1; U_init[4, 0] = 1
    V_init[0, 0] = 1; V_init[4, 0] = 1
    W_init[0, 0] = 1; W_init[1, 0] = 1; W_init[3, 0] = 1; W_init[4, 0] = 1 # Approximate, will be refined
    
    # This manual initialization is error-prone. 
    # Better to use random init but with more restarts and a tighter deadline guard.
    # Or use the fact that rank 17 is close to 18.
    # Start with schoolbook (18) and try to merge two columns?
    
    # Let's try random restarts with rank 17, but initialize with a "near schoolbook" structure.
    # Schoolbook has 18 columns. We can try to set one column to zero and let the solver fix it?
    # Or just random.
    
    for restart in range(50):
        if time.time() > deadline - 5:
            break
            
        np.random.seed(restart)
        x0 = np.random.randn(num_vars) * 0.1
        
        # Bias initialization towards schoolbook structure
        # Schoolbook: U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
        # We have 17 columns. We can take 17 schoolbook columns and perturb them.
        # Which one to drop? Randomly drop one.
        col_to_drop = np.random.randint(18)
        
        U_init = np.zeros((nm, R))
        V_init = np.zeros((mp, R))
        W_init = np.zeros((pn, R))
        
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i*m*j*p + j*p*k + k == col_to_drop: # Just a dummy check, easier to list
                        pass
                    # Simpler: generate all 18, pick 17
                    pass
        
        # Generate 18 schoolbook vectors
        U_school = np.zeros((nm, 18))
        V_school = np.zeros((mp, 18))
        W_school = np.zeros((pn, 18))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m+j, col] = 1
                    V_school[j*p+k, col] = 1
                    W_school[k*n+i, col] = 1
                    col += 1
        
        # Pick 17 columns
        keep_cols = list(range(18))
        keep_cols.pop(np.random.randint(18))
        
        U_init = U_school[:, keep_cols]
        V_init = V_school[:, keep_cols]
        W_init = W_school[:, keep_cols]
        
        # Add small noise to escape local minima
        U_init += np.random.randn(nm, R) * 0.01
        V_init += np.random.randn(mp, R) * 0.01
        W_init += np.random.randn(pn, R) * 0.01
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            sol = least_squares(residual, x0, method='lm', max_nfev=200)
        except Exception:
            continue
            
        if sol.cost < 1e-10:
            U = sol.x[:nm*R].reshape(nm, R)
            V = sol.x[nm*R:nm*R+mp*R].reshape(mp, R)
            W = sol.x[nm*R+mp*R:].reshape(pn, R)
            
            # Verify and round
            U_round = np.round(U * 2) / 2
            V_round = np.round(V * 2) / 2
            W_round = np.round(W * 2) / 2
            
            T_round = np.einsum('ir,jr,kr->IJK', U_round, V_round, W_round)
            if np.allclose(T_round, targets.reshape(nm, mp, pn), atol=1e-9):
                return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
            
            # Try scaling
            for _ in range(5):
                c = np.random.uniform(0.5, 2.0)
                U_s = U * c
                V_s = V / c
                W_s = W
                U_r = np.round(U_s * 2) / 2
                V_r = np.round(V_s * 2) / 2
                W_r = np.round(W_s * 2) / 2
                T_r = np.einsum('ir,jr,kr->IJK', U_r, V_r, W_r)
                if np.allclose(T_r, targets.reshape(nm, mp, pn), atol=1e-9):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
