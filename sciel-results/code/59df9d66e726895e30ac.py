import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor T[a,b,c] where:
    # a = i*m+j, b = j*p+k, c = k*n+i
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook decomposition (rank 84)
    def schoolbook():
        r = n * m * p
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = schoolbook()
    best_rank = U_sb.shape[1]
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    def verify(U, V, W, tol=1e-6):
        """Check if decomposition is exact."""
        S = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(S, T, atol=tol)
    
    def try_optimize(r, U_init, V_init, W_init, max_iter=500, lr=0.01):
        """Try to optimize decomposition to rank r with Adam optimizer."""
        U = U_init.copy().astype(float)
        V = V_init.copy().astype(float)
        W = W_init.copy().astype(float)
        
        # Adam hyperparameters
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        
        mU = np.zeros_like(U)
        vU = np.zeros_like(U)
        mV = np.zeros_like(V)
        vV = np.zeros_like(V)
        mW = np.zeros_like(W)
        vW = np.zeros_like(W)
        
        t = 0
        best_loss = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            t += 1
            S = np.einsum('ar,br,cr->abc', U, V, W)
            err = S - T
            loss = np.sum(err**2)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Gradients
            dU = 2 * np.einsum('abc,br,cr->ar', err, V, W)
            dV = 2 * np.einsum('abc,ar,cr->br', err, U, W)
            dW = 2 * np.einsum('abc,ar,br->cr', err, U, V)
            
            # Adam update for U
            mU = beta1 * mU + (1 - beta1) * dU
            vU = beta2 * vU + (1 - beta2) * (dU ** 2)
            mU_hat = mU / (1 - beta1 ** t)
            vU_hat = vU / (1 - beta2 ** t)
            U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
            
            # Adam update for V
            mV = beta1 * mV + (1 - beta1) * dV
            vV = beta2 * vV + (1 - beta2) * (dV ** 2)
            mV_hat = mV / (1 - beta1 ** t)
            vV_hat = vV / (1 - beta2 ** t)
            V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
            
            # Adam update for W
            mW = beta1 * mW + (1 - beta1) * dW
            vW = beta2 * vW + (1 - beta2) * (dW ** 2)
            mW_hat = mW / (1 - beta1 ** t)
            vW_hat = vW / (1 - beta2 ** t)
            W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
            
            if loss < 1e-10:
                break
        
        # Polish phase: high precision gradient descent
        U, V, W = best_U, best_V, best_W
        for _ in range(50):
            if time.time() > deadline:
                break
            S = np.einsum('ar,br,cr->abc', U, V, W)
            err = S - T
            loss = np.sum(err**2)
            if loss < 1e-12:
                break
            dU = 2 * np.einsum('abc,br,cr->ar', err, V, W)
            dV = 2 * np.einsum('abc,ar,cr->br', err, U, W)
            dW = 2 * np.einsum('abc,ar,br->cr', err, U, V)
            
            # Simple gradient step with small lr
            U -= 1e-4 * dU
            V -= 1e-4 * dV
            W -= 1e-4 * dW
        
        if verify(best_U, best_V, best_W):
            return best_U, best_V, best_W, True
        return None, None, None, False
    
    def structured_init(target_rank, seed=42):
        """Create structured initialization by grouping schoolbook terms."""
        np.random.seed(seed)
        
        all_cols = np.arange(n * m * p)
        
        # Group indices into clusters
        # For matrix multiplication, terms (i,j,k) are related if they share i, j, or k
        groups = []
        for i in range(n):
            for j in range(m):
                group_i_j = [i*m*p + j*p + k for k in range(p)]
                groups.append(group_i_j)
        
        # Select target_rank columns from groups
        selected = set()
        for group in groups:
            if len(selected) >= target_rank:
                break
            # Add up to 3 terms from each group
            np.random.shuffle(group)
            for col in group[:3]:
                if col not in selected and len(selected) < target_rank:
                    selected.add(col)
        
        # Fill remaining with random
        remaining = [c for c in all_cols if c not in selected]
        np.random.shuffle(remaining)
        for col in remaining:
            if len(selected) >= target_rank:
                break
            selected.add(col)
        
        cols = np.array(sorted(selected))
        return cols
    
    # Try various target ranks from high to low
    for target_rank in range(best_rank - 1, 27, -1):
        if time.time() > deadline:
            break
        
        for attempt in range(6):
            if time.time() > deadline:
                break
            
            if attempt == 0:
                cols = np.arange(target_rank)
            elif attempt == 1:
                cols = structured_init(target_rank, seed=target_rank)
            elif attempt == 2:
                np.random.seed(target_rank + 100)
                cols = np.random.choice(best_rank, target_rank, replace=False)
            elif attempt == 3:
                np.random.seed(target_rank * 13 + 200)
                cols = np.random.choice(best_rank, target_rank, replace=False)
            elif attempt == 4:
                np.random.seed(target_rank * 7 + 50)
                cols = np.random.choice(best_rank, target_rank, replace=False)
            else:
                np.random.seed(target_rank * 11 + 77)
                cols = np.random.choice(best_rank, target_rank, replace=False)
            
            U_init = U_sb[:, cols].copy()
            V_init = V_sb[:, cols].copy()
            W_init = W_sb[:, cols].copy()
            
            Ur, Vr, Wr, success = try_optimize(target_rank, U_init, V_init, W_init, max_iter=300, lr=0.01)
            if success:
                best_rank = target_rank
                best_U, best_V, best_W = Ur, Vr, Wr
                break
    
    # Try random initialization for lower ranks
    for target_rank in range(max(28, best_rank - 15), min(best_rank, 50)):
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        
        for attempt in range(5):
            if time.time() > deadline:
                break
            np.random.seed(target_rank * 7 + attempt * 31)
            U_init = np.random.randn(nm, target_rank) * 0.5
            V_init = np.random.randn(mp, target_rank) * 0.5
            W_init = np.random.randn(pn, target_rank) * 0.5
            
            Ur, Vr, Wr, success = try_optimize(target_rank, U_init, V_init, W_init, max_iter=400, lr=0.005)
            if success:
                best_rank = target_rank
                best_U, best_V, best_W = Ur, Vr, Wr
                break
    
    # Final verification
    if not verify(best_U, best_V, best_W):
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = U_sb.shape[1]
    
    def to_clean_list(M):
        L = []
        for row in M:
            row_list = []
            for val in row:
                rounded = round(val * 2) / 2
                if abs(val - rounded) < 1e-6:
                    row_list.append(rounded)
                else:
                    row_list.append(float(val))
            L.append(row_list)
        return L
    
    return {
        "U": to_clean_list(best_U),
        "V": to_clean_list(best_V),
        "W": to_clean_list(best_W)
    }
