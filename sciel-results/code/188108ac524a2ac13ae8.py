"""Matrix multiplication 4x4 @ 4x5: rank-minimal tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 5]
N, M, P = 4, 4, 5
NM = N * M
MP = M * P
PN = P * N


def _schoolbook():
    n, m, p = N, M, P
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W):
    rank = len(U[0])
    for i in range(N):
        for j in range(M):
            for k in range(P):
                total = 0
                for r in range(rank):
                    total += U[i * M + j][r] * V[j * P + k][r] * W[k * N + i][r]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True


def _rank_of(decomp):
    return len(decomp["U"][0])


def _try_merge_columns(U, V, W, a, b):
    """Try to merge columns a and b if they are proportional in a way that preserves the product."""
    rank = len(U[0])
    if a >= rank or b >= rank or a == b:
        return None
    
    # Check if U[:,a] = s * U[:,b] for some s
    s = None
    for r in range(NM):
        if U[r][a] != 0 and U[r][b] != 0:
            s = U[r][a] / U[r][b]
            break
        elif U[r][a] != 0 or U[r][b] != 0:
            return None
    
    if s is None:
        # Check if both are zero
        if all(U[r][a] == 0 for r in range(NM)) and all(U[r][b] == 0 for r in range(NM)):
            s = 1.0
        else:
            return None
    
    # Verify U proportionality
    for r in range(NM):
        if abs(U[r][a] - s * U[r][b]) > 1e-9:
            return None
    
    # Similarly for V and W
    sv = None
    for r in range(MP):
        if V[r][a] != 0 and V[r][b] != 0:
            sv = V[r][a] / V[r][b]
            break
        elif V[r][a] != 0 or V[r][b] != 0:
            return None
    
    if sv is None:
        if all(V[r][a] == 0 for r in range(MP)) and all(V[r][b] == 0 for r in range(MP)):
            sv = 1.0
        else:
            return None
    
    for r in range(MP):
        if abs(V[r][a] - sv * V[r][b]) > 1e-9:
            return None
    
    sw = None
    for r in range(PN):
        if W[r][a] != 0 and W[r][b] != 0:
            sw = W[r][a] / W[r][b]
            break
        elif W[r][a] != 0 or W[r][b] != 0:
            return None
    
    if sw is None:
        if all(W[r][a] == 0 for r in range(PN)) and all(W[r][b] == 0 for r in range(PN)):
            sw = 1.0
        else:
            return None
    
    for r in range(PN):
        if abs(W[r][a] - sw * W[r][b]) > 1e-9:
            return None
    
    # Product scale: s * sv * sw should be 1 for the terms to be identical
    if abs(s * sv * sw - 1.0) > 1e-9:
        return None
    
    # They are identical in product, so we can drop one
    new_U = [row[:b] + row[b+1:] for row in U]
    new_V = [row[:b] + row[b+1:] for row in V]
    new_W = [row[:b] + row[b+1:] for row in W]
    
    return {"U": new_U, "V": new_V, "W": new_W}


def _build_target_tensor():
    """Build the target tensor T[a,b,c] = 1 if (a,b,c) corresponds to valid (i,j,k)."""
    target = np.zeros((NM, MP, PN))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                a = i * M + j
                b = j * P + k
                c = k * N + i
                target[a, b, c] = 1
    return target


def _adam_optimizer(U, V, W, target, deadline, max_steps=2000, lr=0.01):
    """Run Adam optimizer to minimize the residual between the tensor estimate and target."""
    # Adam state
    mU, vU = np.zeros_like(U), np.zeros_like(U)
    mV, vV = np.zeros_like(V), np.zeros_like(V)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    
    for step in range(max_steps):
        if time.time() > deadline - 5:
            break
        
        # Compute current tensor estimate
        est = np.einsum('ar,br,cr->abc', U, V, W)
        residual = est - target
        
        # Check convergence
        if np.max(np.abs(residual)) < 1e-10:
            break
        
        # Gradients
        dU = np.einsum('abc,br,cr->ar', residual, V, W)
        dV = np.einsum('abc,ar,cr->br', residual, U, W)
        dW = np.einsum('abc,ar,br->cr', residual, U, V)
        
        t += 1
        # Update Adam state
        mU = beta1 * mU + (1 - beta1) * dU
        vU = beta2 * vU + (1 - beta2) * dU**2
        mU_hat = mU / (1 - beta1**t)
        vU_hat = vU / (1 - beta2**t)
        
        mV = beta1 * mV + (1 - beta1) * dV
        vV = beta2 * vV + (1 - beta2) * dV**2
        mV_hat = mV / (1 - beta1**t)
        vV_hat = vV / (1 - beta2**t)
        
        mW = beta1 * mW + (1 - beta1) * dW
        vW = beta2 * vW + (1 - beta2) * dW**2
        mW_hat = mW / (1 - beta1**t)
        vW_hat = vW / (1 - beta2**t)
        
        # Update
        U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
        V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
        W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
    
    return U, V, W


def _polish_pass(U, V, W, target, deadline, max_iters=100):
    """High-precision polish pass using Newton-like refinement."""
    for _ in range(max_iters):
        if time.time() > deadline - 3:
            break
        
        est = np.einsum('ar,br,cr->abc', U, V, W)
        residual = est - target
        
        if np.max(np.abs(residual)) < 1e-12:
            break
        
        # Compute gradients
        dU = np.einsum('abc,br,cr->ar', residual, V, W)
        dV = np.einsum('abc,ar,cr->br', residual, U, W)
        dW = np.einsum('abc,ar,br->cr', residual, U, V)
        
        # Use line search to find good step
        lr = 0.1
        for _ls in range(20):
            new_U = U - lr * dU
            new_V = V - lr * dV
            new_W = W - lr * dW
            new_est = np.einsum('ar,br,cr->abc', new_U, new_V, new_W)
            new_res = np.max(np.abs(new_est - target))
            if new_res < np.max(np.abs(residual)):
                U, V, W = new_U, new_V, new_W
                break
            lr *= 0.5
    
    return U, V, W


def _try_random_search(rank, target, deadline, max_trials=3):
    """Try random initialization and Adam optimizer to find a valid decomposition."""
    for trial in range(max_trials):
        if time.time() > deadline - 10:
            break
        
        # Random initialization with multiple scales
        scale = np.random.choice([0.1, 0.5, 1.0, 2.0])
        U = np.random.randn(NM, rank) * scale
        V = np.random.randn(MP, rank) * scale
        W = np.random.randn(PN, rank) * scale
        
        # Run Adam optimizer
        U, V, W = _adam_optimizer(U, V, W, target, deadline=deadline, max_steps=1000, lr=0.01)
        
        # Check if we found a valid decomposition
        est = np.einsum('ar,br,cr->abc', U, V, W)
        if np.max(np.abs(est - target)) < 1e-6:
            # Polish pass for higher precision
            U, V, W = _polish_pass(U, V, W, target, deadline, max_iters=50)
            est = np.einsum('ar,br,cr->abc', U, V, W)
            if np.max(np.abs(est - target)) < 1e-8:
                # Found a valid decomposition!
                # Convert to lists
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    return None


def _try_schmidt_rank_reduction(target, deadline, start_rank=80):
    """Try to find lower-rank decomposition by starting from schoolbook and reducing."""
    # Start from schoolbook decomposition
    schoolbook = _schoolbook()
    U = np.array(schoolbook["U"], dtype=float)
    V = np.array(schoolbook["V"], dtype=float)
    W = np.array(schoolbook["W"], dtype=float)
    current_rank = start_rank
    
    # Try to reduce rank by removing terms one at a time and re-optimizing
    for target_rank in range(start_rank - 1, 60, -1):
        if time.time() > deadline - 30:
            break
        
        # Try multiple times to find a valid decomposition at this rank
        for attempt in range(5):
            if time.time() > deadline - 25:
                break
            
            # Initialize near the schoolbook solution
            U_init = U[:, :target_rank].copy()
            V_init = V[:, :target_rank].copy()
            W_init = W[:, :target_rank].copy()
            
            # Add small random perturbation
            U_init += np.random.randn(*U_init.shape) * 0.01
            V_init += np.random.randn(*V_init.shape) * 0.01
            W_init += np.random.randn(*W_init.shape) * 0.01
            
            # Optimize
            U_opt, V_opt, W_opt = _adam_optimizer(U_init, V_init, W_init, target, deadline=deadline, max_steps=500, lr=0.005)
            
            # Check convergence
            est = np.einsum('ar,br,cr->abc', U_opt, V_opt, W_opt)
            if np.max(np.abs(est - target)) < 1e-5:
                # Polish
                U_opt, V_opt, W_opt = _polish_pass(U_opt, V_opt, W_opt, target, deadline, max_iters=30)
                est = np.einsum('ar,br,cr->abc', U_opt, V_opt, W_opt)
                if np.max(np.abs(est - target)) < 1e-8:
                    return {
                        "U": U_opt.tolist(),
                        "V": V_opt.tolist(),
                        "W": W_opt.tolist()
                    }
    
    return None


def construct():
    deadline = time.time() + 110
    best = _schoolbook()
    best_rank = _rank_of(best)
    
    # Try to reduce rank by merging proportional columns
    U, V, W = best["U"], best["V"], best["W"]
    rank = _rank_of(best)
    
    changed = True
    while changed and time.time() < deadline - 40:
        changed = False
        for a in range(rank):
            if time.time() > deadline - 35:
                break
            for b in range(a + 1, rank):
                if time.time() > deadline - 35:
                    break
                merged = _try_merge_columns(U, V, W, a, b)
                if merged is not None:
                    if _verify(merged["U"], merged["V"], merged["W"]):
                        U, V, W = merged["U"], merged["V"], merged["W"]
                        rank = _rank_of({"U": U, "V": V, "W": W})
                        changed = True
                        best = {"U": U, "V": V, "W": W}
                        best_rank = rank
                        break
            if changed:
                break
    
    # Build target tensor
    target = _build_target_tensor()
    
    # Try Schmidt-style rank reduction
    if time.time() < deadline - 30:
        result = _try_schmidt_rank_reduction(target, deadline, start_rank=best_rank)
        if result is not None:
            if _verify(result["U"], result["V"], result["W"]):
                best = result
                best_rank = _rank_of(result)
    
    # Try random search for lower ranks
    if time.time() < deadline - 20:
        # Try various ranks below the current best
        ranks_to_try = list(range(max(60, best_rank - 5), max(45, best_rank - 20), -1))
        
        for r in ranks_to_try:
            if time.time() > deadline - 15:
                break
            if r >= best_rank:
                continue
            
            result = _try_random_search(r, target, deadline, max_trials=2)
            if result is not None:
                if _verify(result["U"], result["V"], result["W"]):
                    best = result
                    best_rank = r
                    break
    
    return best
