import numpy as np
import time
from scipy.optimize import minimize

def construct():
    n, m, p = 2, 3, 10
    start_time = time.time()
    deadline = 145
    
    # 1. Solve 2x2x5 block
    T_block = np.zeros((4, 10, 10), dtype=float)
    for i in range(2):
        for j in range(2):
            for k in range(5):
                T_block[i*2 + j, j*5 + k, k*2 + i] = 1.0
                
    R_block = 16
    def loss_block(params):
        U = params[0:4*R_block].reshape(4, R_block)
        V = params[4*R_block:4*R_block+10*R_block].reshape(10, R_block)
        W = params[4*R_block+10*R_block:4*R_block+20*R_block].reshape(10, R_block)
        T_comp = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((T_comp - T_block)**2)
        
    U_block = None
    V_block = None
    W_block = None
    
    for attempt in range(20):
        if time.time() > deadline - 20:
            break
        np.random.seed(attempt)
        U0 = np.random.randn(4, R_block) * 0.5
        V0 = np.random.randn(10, R_block) * 0.5
        W0 = np.random.randn(10, R_block) * 0.5
        params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        res = minimize(loss_block, params0, method='BFGS', options={'maxiter': 1000, 'gtol': 1e-8})
        if res.fun < 1e-6:
            U_block = res.x[0:4*R_block].reshape(4, R_block)
            V_block = res.x[4*R_block:4*R_block+10*R_block].reshape(10, R_block)
            W_block = res.x[4*R_block+10*R_block:4*R_block+20*R_block].reshape(10, R_block)
            
            # Verify
            T_check = np.einsum('ar,br,cr->abc', U_block, V_block, W_block)
            if np.allclose(T_check, T_block, atol=1e-5):
                break
                
    if U_block is None:
        # Fallback to Strassen 2x2x10 (Rank 35)
        # ... (Implement Strassen fallback)
        pass
    else:
        # Construct 2x2x10
        R_22 = 32
        U_22 = np.zeros((4, R_22))
        V_22 = np.zeros((20, R_22))
        W_22 = np.zeros((20, R_22))
        
        # Block 1
        U_22[:, 0:16] = U_block
        for j in range(2):
            for k in range(5):
                V_22[j*10 + k, 0:16] = V_block[j*5 + k, 0:16]
                W_22[k*2 + i, 0:16] = W_block[k*2 + i, 0:16] # Wait, i loop missing
        # Correct W mapping
        for k in range(5):
            for i in range(2):
                W_22[k*2 + i, 0:16] = W_block[k*2 + i, 0:16]
                
        # Block 2
        U_22[:, 16:32] = U_block
        for j in range(2):
            for k in range(5):
                V_22[j*10 + (k+5), 16:32] = V_block[j*5 + k, 0:16]
        for k in range(5):
            for i in range(2):
                W_22[(k+5)*2 + i, 16:32] = W_block[k*2 + i, 0:16]
                
        # 2. Add 2x1x10 part (Rank 20)
        R_total = 52
        U_total = np.zeros((6, R_total))
        V_total = np.zeros((30, R_total))
        W_total = np.zeros((20, R_total))
        
        # Copy 2x2x10 part
        # U indices: i*2+j for i in 0,1; j in 0,1. (0..3)
        U_total[0:4, 0:32] = U_22
        # V indices: j*10+k for j in 0,1; k in 0..9. (0..19)
        V_total[0:20, 0:32] = V_22
        # W indices: k*2+i for k in 0..9; i in 0,1. (0..19)
        W_total[0:20, 0:32] = W_22
        
        # 2x1x10 part
        # A is 2x1 (row 2 of A in original 2x3). Index i*3+2.
        # B is 1x10 (col 2 of B in original 3x10). Index 2*10+k = 20+k.
        # C is 10x2. Index k*2+i.
        # Rank 20.
        for r in range(20):
            i = r // 10
            k = r % 10
            U_total[i*3 + 2, 32 + r] = 1.0
            V_total[20 + k, 32 + r] = 1.0
            W_total[k*2 + i, 32 + r] = 1.0
            
        # Return
        return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
