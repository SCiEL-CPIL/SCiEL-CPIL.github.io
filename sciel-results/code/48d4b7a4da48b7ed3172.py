import numpy as np
from scipy.optimize import minimize
from fractions import Fraction

def construct():
    n, m, p = 2, 3, 5
    na, nb, nc = n*m, m*p, p*n  # 6, 15, 10
    total_eq = na * nb * nc  # 900
    
    # Build the tensor
    tensor = np.zeros((na, nb, nc))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook decomposition (rank 30)
    rank_school = n*m*p
    U_school = np.zeros((na, rank_school))
    V_school = np.zeros((nb, rank_school))
    W_school = np.zeros((nc, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    # Try to find lower rank decompositions
    for r in range(29, 20, -1):
        # Try multiple random initializations
        found = False
        for trial in range(5):
            # Initialize with schoolbook terms + random perturbation
            if trial == 0:
                U0 = U_school[:, :r].copy()
                V0 = V_school[:, :r].copy()
                W0 = W_school[:, :r].copy()
            else:
                U0 = U_school[:, :r].copy() + 0.1 * np.random.randn(na, r)
                V0 = V_school[:, :r].copy() + 0.1 * np.random.randn(nb, r)
                W0 = W_school[:, :r].copy() + 0.1 * np.random.randn(nc, r)
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            def objective(x):
                Ux = x[:na*r].reshape(na, r)
                Vx = x[na*r:(na+nb)*r].reshape(nb, r)
                Wx = x[(na+nb)*r:].reshape(nc, r)
                T_hat = np.einsum('ad,bd,cd->abc', Ux, Vx, Wx)
                return np.sum((T_hat - tensor)**2)
            
            try:
                result = minimize(objective, x0, method='L-BFGS-B',
                                options={'maxiter': 500, 'gtol': 1e-14})
                if result.fun < 1e-10:
                    Ux = result.x[:na*r].reshape(na, r)
                    Vx = result.x[na*r:(na+nb)*r].reshape(nb, r)
                    Wx = result.x[(na+nb)*r:].reshape(nc, r)
                    
                    # Try to convert to rationals
                    U_rat = [[Fraction(x).limit_denominator(1000) for x in row] for row in Ux]
                    V_rat = [[Fraction(x).limit_denominator(1000) for x in row] for row in Vx]
                    W_rat = [[Fraction(x).limit_denominator(1000) for x in row] for row in Wx]
                    
                    # Verify exactly
                    T_check = np.zeros((na, nb, nc))
                    for d in range(r):
                        for a in range(na):
                            for b in range(nb):
                                for c in range(nc):
                                    T_check[a,b,c] += float(U_rat[a][d]) * float(V_rat[b][d]) * float(W_rat[c][d])
                    
                    if np.allclose(T_check, tensor, atol=1e-12):
                        return {"U": U_rat, "V": V_rat, "W": W_rat}
                    found = True
            except:
                pass
        
        if found:
            break
    
    # Fall back to schoolbook
    U_list = [[int(x) for x in row] for row in U_school]
    V_list = [[int(x) for x in row] for row in V_school]
    W_list = [[int(x) for x in row] for row in W_school]
    return {"U": U_list, "V": V_list, "W": W_list}
