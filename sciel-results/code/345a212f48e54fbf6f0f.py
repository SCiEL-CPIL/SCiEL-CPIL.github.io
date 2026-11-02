"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Build the matrix multiplication tensor T with shape (n*m, m*p, p*n)
    # T[u, v, w] = 1 if u=im+j, v=jp+k, w=kn+i for some i,j,k
    nm = n * m
    mp = m * p
    pn = p * n
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    def verify(U, V, W):
        """Verify exact decomposition with integer/half-integer entries."""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        
        # Check entries are close to half-integers
        U2 = U * 2
        V2 = V * 2
        W2 = W * 2
        if not (np.allclose(U2, np.round(U2), atol=1e-6) and
                np.allclose(V2, np.round(V2), atol=1e-6) and
                np.allclose(W2, np.round(W2), atol=1e-6)):
            return False
        
        # Round to half-integers
        U_r = np.round(U2) / 2.0
        V_r = np.round(V2) / 2.0
        W_r = np.round(W2) / 2.0
        
        # Compute tensor product: T_comp[u,v,w] = sum_r U_r[u,r] * V_r[v,r] * W_r[w,r]
        # Use einsum for efficiency
        T_comp = np.einsum('ur,vr,wr->uvw', U_r, V_r, W_r)
        
        return np.allclose(T_comp, T, atol=1e-9)
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    # Try ranks from low to high
    best_rank = n * m * p
    best_result = None
    
    U0, V0, W0 = schoolbook()
    
    for rank in range(28, best_rank + 1):
        if time.time() > deadline:
            break
        
        # Try multiple random restarts
        found = False
        max_restarts = 50 if rank <= 35 else 20
        max_iter = 5000 if rank <= 35 else 2000
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization with small values
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            # Gradient descent on loss = sum (T_comp - T)^2
            lr = 0.01
            prev_loss = np.inf
            
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Compute T_comp and loss
                T_comp = np.einsum('ur,vr,wr->uvw', U, V, W)
                loss = np.sum((T_comp - T) ** 2)
                
                if loss < 1e-12:
                    break
                if loss > prev_loss * 1.1:
                    break
                prev_loss = loss
                
                # Gradients
                # dL/dU[u,r] = 2 * sum_{v,w} (T_comp[u,v,w] - T[u,v,w]) * V[v,r] * W[w,r]
                # dL/dV[v,r] = 2 * sum_{u,w} (T_comp[u,v,w] - T[u,v,w]) * U[u,r] * W[w,r]
                # dL/dW[w,r] = 2 * sum_{u,v} (T_comp[u,v,w] - T[u,v,w]) * U[u,r] * V[v,r]
                
                diff = T_comp - T  # shape (nm, mp, pn)
                
                dU = 2.0 * np.einsum('uvw,vr,wr->ur', diff, V, W)
                dV = 2.0 * np.einsum('uvw,ur,wr->vr', diff, U, W)
                dW = 2.0 * np.einsum('uvw,ur,vr->wr', diff, U, V)
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            # Check if we converged
            T_comp = np.einsum('ur,vr,wr->uvw', U, V, W)
            loss = np.sum((T_comp - T) ** 2)
            
            if loss < 1e-8:
                # Try rounding
                U2 = np.round(U * 2) / 2.0
                V2 = np.round(V * 2) / 2.0
                W2 = np.round(W * 2) / 2.0
                
                if verify(U2, V2, W2):
                    found = True
                    break
        
        if found:
            best_rank = rank
            best_result = (U2, V2, W2)
            break
    
    if best_result is not None:
        U, V, W = best_result
        # Convert to lists for return
        U_list = [[U[i, r] for r in range(len(U[0]))] for i in range(len(U))]
        V_list = [[V[i, r] for r in range(len(V[0]))] for i in range(len(V))]
        W_list = [[W[i, r] for r in range(len(W[0]))] for i in range(len(W))]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        U, V, W = schoolbook()
        U_list = [[U[i, r] for r in range(len(U[0]))] for i in range(len(U))]
        V_list = [[V[i, r] for r in range(len(V[0]))] for i in range(len(V))]
        W_list = [[W[i, r] for r in range(len(W[0]))] for i in range(len(W))]
        return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
