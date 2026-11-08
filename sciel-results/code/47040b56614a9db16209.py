"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[(a, b, c)] = 1
    
    deadline = time.time() + 115  # safety margin before 120s limit
    
    # Try ranks from lower bound up
    # Lower bound: max(n*m, m*p, p*n) = max(8, 20, 10) = 20
    for rank in range(20, n*m*p + 1):
        if time.time() > deadline:
            break
        result = try_rank(n, m, p, nm, mp, pn, rank, target, deadline)
        if result is not None:
            return result
    
    # Schoolbook fallback: rank = n*m*p = 40
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def try_rank(n, m, p, nm, mp, pn, rank, target, deadline):
    """Try to find a rank-r exact decomposition using structured search."""
    
    # Strategy: Use the Kronecker product structure
    # m = 4 = 2*2, so j = j1*2 + j2
    # This gives a block structure
    
    # For each rank, try random initializations and use alternating linear solves
    # The key: for fixed V and W, the system in U is linear
    # For fixed U and W, the system in V is linear
    # For fixed U and V, the system in W is linear
    
    # Use structured random search with alternating linearization
    np.random.seed(rank * 1000 + 42)
    
    for trial in range(50):
        if time.time() > deadline:
            return None
        
        # Random initialization with sparse structure
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        # Initialize with random ±1 or 0 patterns
        for c in range(rank):
            # Random support for U
            for a in range(nm):
                if np.random.random() < 0.3:
                    U[a, c] = np.random.choice([-1, 1])
            for b in range(mp):
                if np.random.random() < 0.2:
                    V[b, c] = np.random.choice([-1, 1])
            for c_idx in range(pn):
                if np.random.random() < 0.4:
                    W[c_idx, c] = np.random.choice([-1, 1])
        
        # Alternating linearization
        for it in range(100):
            if time.time() > deadline:
                return None
            
            # Update U: for each column c, solve linear system
            # The residual R[a,b,c] = sum_{c'} U[a,c']*V[b,c']*W[c,c'] - T[a,b,c]
            # Gradient w.r.t. U[a,c] = sum_{b,c'} V[b,c']*W[c,c']*(R)
            # Linear update: U_new = U - lr * grad
            
            # Compute residuals efficiently
            # R[a,b,c] = sum_{c'} U[a,c']*V[b,c']*W[c,c']
            # For fixed c: R_c[a,b] = sum_{c'} U[a,c']*V[b,c']*W[c,c']
            # This is (U * diag(W[c]) * V^T) but that's not quite right
            
            # Let's compute: M_c[a,b] = sum_{c'} U[a,c'] * V[b,c'] * W[c,c']
            # = W[c,c'] summed... no.
            # M_c[a,b] = sum_{c'} (U[a,c'] * W[c,c']) * V[b,c']
            # Let U_W[a,c'] = U[a,c'] * W[c,c'] for fixed c
            # Then M_c[a,b] = sum_{c'} U_W[a,c'] * V[b,c'] = (U_W @ V.T)[a,b]
            
            # Build U_W for each c
            U_W = U * W.T  # U_W[a,c'] = U[a,c'] * W[c,c']... no
            # W.T[c',c] = W[c,c'], so (U * W.T)[a,c'] = U[a,c'] * W.T[c',c] = U[a,c'] * W[c,c']
            # But we want to sum over c' with weight W[c,c']
            # So M_c[a,b] = sum_{c'} U[a,c'] * W[c,c'] * V[b,c']
            # = (U * W[c,:].reshape(1,-1) @ V.T)[a,b]
            
            # Compute all M_c at once:
            # M is a 3D array: M[c,a,b] = sum_{c'} U[a,c'] * W[c,c'] * V[b,c']
            # M = np.einsum('ac,cc_,bc_->cab', U, W, V) but that's not right either
            # M[c,a,b] = sum_{c'} U[a,c'] * W[c,c'] * V[b,c']
            # = np.einsum('ac,cc,cb->cab', U, W, V)
            
            M = np.einsum('ac,cc,cb->cab', U, W, V)  # M[c,a,b]
            
            # Target tensor
            T = np.zeros((pn, nm, mp))
            for (a, b, c) in target:
                T[c, a, b] = 1
            
            # Residual
            R = M - T
            
            # Gradient w.r.t. U[a,c'] = sum_{b,c} R[c,a,b] * V[b,c'] * W[c,c']
            # grad_U = np.einsum('cab,bc_,cc->ac', R, V, W)
            grad_U = np.einsum('cab,bc,cc->ac', R, V, W)
            grad_V = np.einsum('cab,ac,cc->bc', R, U, W)
            grad_W = np.einsum('cab,ac,bc->cc_', R, U, V)
            
            # Learning rate
            lr = 0.1 / (1 + it * 0.01)
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check convergence
            if np.max(np.abs(R)) < 1e-6:
                # Try to round to rationals
                result = round_and_verify(U, V, W, nm, mp, pn, target)
                if result is not None:
                    return result
        
        # After alternating, try to polish
        result = polish_and_verify(U, V, W, nm, mp, pn, target, deadline)
        if result is not None:
            return result
    
    return None

def round_and_verify(U, V, W, nm, mp, pn, target, max_denom=20):
    """Round to rationals with small denominators and verify."""
    def round_matrix(M, max_denom):
        M_r = np.zeros_like(M)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                val = M[i, j]
                if abs(val) < 1e-10:
                    M_r[i, j] = 0
                    continue
                best_num, best_den = 0, 1
                for den in range(1, max_denom + 1):
                    num = round(val * den)
                    if abs(val - num / den) < 1e-8:
                        best_num, best_den = num, den
                        break
                M_r[i, j] = best_num / best_den
        return M_r
    
    U_r = round_matrix(U, max_denom)
    V_r = round_matrix(V, max_denom)
    W_r = round_matrix(W, max_denom)
    
    # Verify
    if verify_exact(U_r, V_r, W_r, nm, mp, pn, target):
        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    return None

def verify_exact(U, V, W, nm, mp, pn, target, tol=1e-9):
    """Verify exact decomposition."""
    rank = U.shape[1]
    # Check target entries
    for (a, b, c) in target:
        s = 0.0
        for c_ in range(rank):
            s += U[a, c_] * V[b, c_] * W[c, c_]
        if abs(s - 1.0) > tol:
            return False
    # Check non-target entries (sample)
    np.random.seed(42)
    for _ in range(200):
        a = np.random.randint(nm)
        b = np.random.randint(mp)
        c = np.random.randint(pn)
        if (a, b, c) not in target:
            s = 0.0
            for c_ in range(rank):
                s += U[a, c_] * V[b, c_] * W[c, c_]
            if abs(s) > tol:
                return False
    return True

def polish_and_verify(U, V, W, nm, mp, pn, target, deadline):
    """Final polish pass with tighter tolerance."""
    for it in range(200):
        if time.time() > deadline:
            return None
        
        M = np.einsum('ac,cc,cb->cab', U, W, V)
        T = np.zeros((pn, nm, mp))
        for (a, b, c) in target:
            T[c, a, b] = 1
        R = M - T
        
        grad_U = np.einsum('cab,bc,cc->ac', R, V, W)
        grad_V = np.einsum('cab,ac,cc->bc', R, U, W)
        grad_W = np.einsum('cab,ac,bc->cc_', R, U, V)
        
        lr = 0.01
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        if np.max(np.abs(R)) < 1e-8:
            result = round_and_verify(U, V, W, nm, mp, pn, target)
            if result is not None:
                return result
    return None
# EVOLVE-BLOCK-END

