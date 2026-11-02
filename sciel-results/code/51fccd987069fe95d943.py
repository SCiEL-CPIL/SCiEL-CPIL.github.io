import numpy as np
import time
from fractions import Fraction
import scipy.optimize as opt

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 2, 3, 9
    
    def find_rank_233(R, time_budget):
        # Variables: U (6, R), V (9, R), W (6, R)
        # Flatten to 1D vector x of size R * (6 + 9 + 6) = 21 R
        
        def loss(x):
            U = x[:6*R].reshape(6, R)
            V = x[6*R:15*R].reshape(9, R)
            W = x[15*R:].reshape(6, R)
            
            # Compute T[i,j,k]
            T = np.zeros((2,3,3))
            for i in range(2):
                for j in range(3):
                    for k in range(3):
                        T[i,j,k] = np.sum(U[i*3+j, :] * V[j*3+k, :] * W[k*2+i, :])
            
            E = T - 1.0
            return np.sum(E**2)
        
        def grad(x):
            U = x[:6*R].reshape(6, R)
            V = x[6*R:15*R].reshape(9, R)
            W = x[15*R:].reshape(6, R)
            
            T = np.zeros((2,3,3))
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(2):
                for j in range(3):
                    for k in range(3):
                        val = np.sum(U[i*3+j, :] * V[j*3+k, :] * W[k*2+i, :])
                        E = val - 1.0
                        dU[i*3+j, :] += E * V[j*3+k, :] * W[k*2+i, :]
                        dV[j*3+k, :] += E * U[i*3+j, :] * W[k*2+i, :]
                        dW[k*2+i, :] += E * U[i*3+j, :] * V[j*3+k, :]
                        
            # Flatten gradients
            gU = dU.flatten()
            gV = dV.flatten()
            gW = dW.flatten()
            return np.concatenate([gU, gV, gW])

        # Multiple random restarts
        best_loss = float('inf')
        best_x = None
        
        num_restarts = 50 # Adjust based on time
        
        for r in range(num_restarts):
            if time.time() > start_time + time_budget:
                break
            
            x0 = np.random.randn(21 * R) * 0.5
            
            try:
                res = opt.minimize(loss, x0, jac=grad, method='L-BFGS-B', 
                                   options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6})
                if res.fun < best_loss:
                    best_loss = res.fun
                    best_x = res.x
            except Exception:
                pass
        
        if best_x is not None and best_loss < 1e-5:
            U = best_x[:6*R].reshape(6, R)
            V = best_x[6*R:15*R].reshape(9, R)
            W = best_x[15*R:].reshape(6, R)
            return U, V, W
            
        return None

    U_block, V_block, W_block = None, None, None
    
    # Try to find rank 12 for 2x3x3
    if time.time() < deadline - 10:
        res = find_rank_233(12, 20)
        if res is not None:
            U_block, V_block, W_block = res
            
    if U_block is None and time.time() < deadline - 10:
        res = find_rank_233(14, 20)
        if res is not None:
            U_block, V_block, W_block = res

    if U_block is None and time.time() < deadline - 10:
        res = find_rank_233(16, 20)
        if res is not None:
            U_block, V_block, W_block = res

    if U_block is None:
        R_block = 18
        U_block = np.zeros((6, R_block))
        V_block = np.zeros((9, R_block))
        W_block = np.zeros((6, R_block))
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    U_block[i*3+j, col] = 1
                    V_block[j*3+k, col] = 1
                    W_block[k*2+i, col] = 1
                    col += 1

    R_total = 3 * R_block if R_block != 18 else 54
    # Wait, if R_block is 12, total is 36.
    # If R_block is 14, total is 42.
    # If R_block is 16, total is 48.
    # If R_block is 18, total is 54.
    
    U = np.zeros((6, R_total))
    V = np.zeros((27, R_total))
    W = np.zeros((18, R_total))
    
    for b in range(3):
        for c in range(R_block):
            c_full = b * R_block + c
            U[:, c_full] = U_block[:, c]
            for j in range(3):
                for l in range(3):
                    V[j*9 + 3*b + l, c_full] = V_block[j*3 + l, c]
            for l in range(3):
                for i in range(2):
                    W[(3*b + l)*2 + i, c_full] = W_block[l*2 + i, c]
                    
    # Verify
    valid = True
    for i in range(2):
        for j in range(3):
            for k in range(9):
                val = np.sum(U[i*3+j, :] * V[j*9+k, :] * W[k*2+i, :])
                if abs(val - 1.0) > 1e-5:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        rank = 54
        U = np.zeros((6, rank))
        V = np.zeros((27, rank))
        W = np.zeros((18, rank))
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(9):
                    U[i*3+j, col] = 1
                    V[j*9+k, col] = 1
                    W[k*2+i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
