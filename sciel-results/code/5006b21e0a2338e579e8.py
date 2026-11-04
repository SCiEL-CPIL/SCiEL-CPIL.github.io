"""Matrix multiplication 4x4x5 rank-minimal decomposition."""
import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the multiplication tensor T[i,j,k] for (i,j) in U, (j,k) in V, (k,i) in W
    # T[i*m+j, j*p+k, k*n+i] = 1
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    # and sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] == 0 for j != j'
    # Actually the tensor is: T[a,b,c] where a=i*m+j, b=j'*p+k, c=k'*n+i
    # T[a,b,c] = 1 if j==j' and k==k' and i==i (consistent), else 0
    # More precisely: T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # And T[a,b,c] = 0 otherwise
    
    # Let's check all (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False, val, (a, b, c)
    
    # Check that non-diagonal entries are zero
    # For any (a, b, c) that doesn't correspond to a valid (i,j,k) triple
    # a = i*m+j, b = j'*p+k, c = k'*n+i'
    # Valid if j == j' and k == k' and i == i'
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                for j2 in range(m):
                    for k2 in range(p):
                        b = j2 * p + k2
                        for i2 in range(n):
                            c = k2 * n + i2
                            # This is valid only if i==i2, j==j2, k==k2
                            if i == i2 and j == j2 and k == k2:
                                continue
                            val = np.sum(U[a, :] * V[b, :] * W[c, :])
                            if abs(val) > tol:
                                return False, val, (a, b, c)
    return True, 0, None


def _nmtf_step(U, V, W, n, m, p, rank, lr=0.1, max_iter=500, tol=1e-10, deadline=None):
    """NMTF optimization step using gradient descent."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    def loss():
        total = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U[a] * V[b] * W[c])
                    total += (val - 1.0) ** 2
        # Non-diagonal terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    for j2 in range(m):
                        for k2 in range(p):
                            b = j2 * p + k2
                            for i2 in range(n):
                                c = k2 * n + i2
                                if i == i2 and j == j2 and k == k2:
                                    continue
                                val = np.sum(U[a] * V[b] * W[c])
                                total += val ** 2
        return total
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
        L = loss()
        if L < tol:
            break
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U[a] * V[b] * W[c])
                    err = val - 1.0
                    dU[a] += 2 * err * V[b] * W[c]
                    dV[b] += 2 * err * U[a] * W[c]
                    dW[c] += 2 * err * U[a] * V[b]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    for j2 in range(m):
                        for k2 in range(p):
                            b = j2 * p + k2
                            for i2 in range(n):
                                c = k2 * n + i2
                                if i == i2 and j == j2 and k == k2:
                                    continue
                                val = np.sum(U[a] * V[b] * W[c])
                                dU[a] += 2 * val * V[b] * W[c]
                                dV[b] += 2 * val * U[a] * W[c]
                                dW[c] += 2 * val * U[a] * V[b]
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Snap to nearest half-integer
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
    
    return U, V, W, loss()


def _try_rank_63_construction(n, m, p):
    """
    Try to construct a rank-63 decomposition using known algebraic methods.
    For 4x4x4, the optimal rank is 49. For 4x4x5, it's 63.
    
    Strategy: Use the structure of the 4x4x4 rank-49 decomposition and extend it.
    """
    # For 4x4x4, rank 49 is known. We can try to build 4x4x5 from 4x4x4.
    # 4x4x5 = 4x4x4 + 4x4x1, so rank <= 49 + 4 = 53? No, that's not how it works.
    # Actually, the multiplication tensor for 4x4x5 has rank 63.
    
    # Let's try a direct NMTF approach with rank 63
    rank = 63
    deadline = time.time() + 100  # Leave margin
    
    best_U, best_V, best_W = None, None, None
    best_loss = float('inf')
    
    # Initialize with structured random (half-integers)
    np.random.seed(42)
    
    for attempt in range(5):
        if time.time() > deadline:
            break
        
        # Initialize with sparse half-integer values
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Each column of U, V, W gets a few non-zero entries
        for r in range(rank):
            # U column: pick 1-2 positions
            nz_u = np.random.choice(n * m, size=np.random.randint(1, 3), replace=False)
            for idx in nz_u:
                U[idx, r] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
            
            # V column: pick 1-2 positions  
            nz_v = np.random.choice(m * p, size=np.random.randint(1, 3), replace=False)
            for idx in nz_v:
                V[idx, r] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
            
            # W column: pick 1-2 positions
            nz_w = np.random.choice(p * n, size=np.random.randint(1, 3), replace=False)
            for idx in nz_w:
                W[idx, r] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
        
        U, V, W, loss = _nmtf_step(U, V, W, n, m, p, rank, lr=0.5, max_iter=200, deadline=deadline)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < 1e-9:
            break
    
    if best_loss < 1e-9:
        # Convert to lists of half-integers
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return U_list, V_list, W_list
    
    return None


def _try_rank_66_construction(n, m, p):
    """Try rank 66 as a fallback."""
    rank = 66
    deadline = time.time() + 80
    
    np.random.seed(123)
    
    for attempt in range(3):
        if time.time() > deadline:
            break
        
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for r in range(rank):
            nz_u = np.random.choice(n * m, size=np.random.randint(1, 3), replace=False)
            for idx in nz_u:
                U[idx, r] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
            
            nz_v = np.random.choice(m * p, size=np.random.randint(1, 3), replace=False)
            for idx in nz_v:
                V[idx, r] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
            
            nz_w = np.random.choice(p * n, size=np.random.randint(1, 3), replace=False)
            for idx in nz_w:
                W[idx, r] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
        
        U, V, W, loss = _nmtf_step(U, V, W, n, m, p, rank, lr=0.5, max_iter=200, deadline=deadline)
        
        if loss < 1e-9:
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return U_list, V_list, W_list
    
    return None


def _schoolbook(n, m, p):
    """Fallback schoolbook decomposition."""
    rank = n * m * p
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
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Try rank 63 first
    result = _try_rank_63_construction(n, m, p)
    if result is not None:
        U, V, W = result
        # Verify
        valid, _, _ = _verify_decomposition(U, V, W, n, m, p)
        if valid:
            return {"U": U, "V": V, "W": W}
    
    if time.time() > deadline:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    # Try rank 66
    result = _try_rank_66_construction(n, m, p)
    if result is not None:
        U, V, W = result
        valid, _, _ = _verify_decomposition(U, V, W, n, m, p)
        if valid:
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
