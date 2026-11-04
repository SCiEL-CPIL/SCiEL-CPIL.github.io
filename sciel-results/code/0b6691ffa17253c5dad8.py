"""
Matrix multiplication 2x4x5 - rank-minimal tensor decomposition.

The tensor M has entries M[i,j,k] = 1 for all i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4}.
We seek U, V, W with shapes (8, R), (20, R), (10, R) such that:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all i, j, k

Strategy:
1. Try explicit Strassen-based block construction (target rank ~32-38).
2. If that fails, use a random-search / gradient-descent approach to find a valid decomposition.
3. Fall back to schoolbook (rank 40) if nothing else works.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    _start = time.time()
    _deadline = _start + 140.0

    n, m, p = DIMENSIONS  # 2, 4, 5

    def verify(U, V, W, tol=1e-6):
        """Check that the decomposition exactly reproduces the all-ones tensor."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        rank = U_arr.shape[1]

        # Reshape to 3D
        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)

        # Compute tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        max_err = np.max(np.abs(T - 1.0))
        return max_err < tol

    def schoolbook():
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
        return {"U": U, "V": V, "W": W}

    def strassen_2x2x2():
        """Rank-7 Strassen decomposition for 2x2x2.
        U: (4, 7), V: (4, 7), W: (4, 7)
        Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))

        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1

        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1

        # M3 = A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1

        # M4 = A11(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1

        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1

        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1

        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = 1

        return U, V, W

    def try_strassen_blocks():
        """Build a block-Strassen decomposition.
        
        Split j={0,1,2,3} into j1={0,1} and j2={2,3}.
        Split k={0,1,2,3,4} into k1={0,1}, k2={2,3}, k3={4}.
        
        Blocks:
        - (j1, k1): 2x2x2 -> rank 7 (Strassen)
        - (j1, k2): 2x2x2 -> rank 7 (Strassen)
        - (j1, k3): 2x2x1 -> rank 2
        - (j2, k1): 2x2x2 -> rank 7 (Strassen)
        - (j2, k2): 2x2x2 -> rank 7 (Strassen)
        - (j2, k3): 2x2x1 -> rank 2
        
        Total: 7+7+2+7+7+2 = 32
        """
        S_U, S_V, S_W = strassen_2x2x2()
        
        rank = 32
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        
        # Block (j1, k1): j in {0,1}, k in {0,1}
        for r in range(7):
            for i in range(n):
                for j in range(2):
                    U[i * m + j, col] = S_U[i * 2 + j, r]
            for j in range(2):
                for k in range(2):
                    V[j * p + k, col] = S_V[j * 2 + k, r]
            for k in range(2):
                for i in range(n):
                    W[k * n + i, col] = S_W[k * 2 + i, r]
            col += 1
        
        # Block (j1, k2): j in {0,1}, k in {2,3}
        for r in range(7):
            for i in range(n):
                for j in range(2):
                    U[i * m + j, col] = S_U[i * 2 + j, r]
            for j in range(2):
                for k in range(2):
                    V[j * p + (k + 2), col] = S_V[j * 2 + k, r]
            for k in range(2):
                for i in range(n):
                    W[(k + 2) * n + i, col] = S_W[k * 2 + i, r]
            col += 1
        
        # Block (j1, k3): j in {0,1}, k in {4} -> rank 2
        for j in range(2):
            for i in range(n):
                U[i * m + j, col] = 1
            V[j * p + 4, col] = 1
            for i in range(n):
                W[4 * n + i, col] = 1
            col += 1
        
        # Block (j2, k1): j in {2,3}, k in {0,1}
        for r in range(7):
            for i in range(n):
                for j in range(2):
                    U[i * m + (j + 2), col] = S_U[i * 2 + j, r]
            for j in range(2):
                for k in range(2):
                    V[(j + 2) * p + k, col] = S_V[j * 2 + k, r]
            for k in range(2):
                for i in range(n):
                    W[k * n + i, col] = S_W[k * 2 + i, r]
            col += 1
        
        # Block (j2, k2): j in {2,3}, k in {2,3}
        for r in range(7):
            for i in range(n):
                for j in range(2):
                    U[i * m + (j + 2), col] = S_U[i * 2 + j, r]
            for j in range(2):
                for k in range(2):
                    V[(j + 2) * p + (k + 2), col] = S_V[j * 2 + k, r]
            for k in range(2):
                for i in range(n):
                    W[(k + 2) * n + i, col] = S_W[k * 2 + i, r]
            col += 1
        
        # Block (j2, k3): j in {2,3}, k in {4} -> rank 2
        for j in range(2):
            for i in range(n):
                U[i * m + (j + 2), col] = 1
            V[(j + 2) * p + 4, col] = 1
            for i in range(n):
                W[4 * n + i, col] = 1
            col += 1
        
        if col != rank:
            raise ValueError(f"Column mismatch: {col} != {rank}")
        
        if verify(U.tolist(), V.tolist(), W.tolist()):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None

    def random_search(rank, n_iter=5000, seed=42):
        """Random search for a valid decomposition of given rank."""
        rng = np.random.RandomState(seed)
        
        best_err = float('inf')
        best = None
        
        for it in range(n_iter):
            if time.time() > _deadline - 10:
                break
            
            # Random initialization with small values
            U = rng.randn(n * m, rank) * 0.5
            V = rng.randn(m * p, rank) * 0.5
            W = rng.randn(p * n, rank) * 0.5
            
            U_3d = U.reshape(n, m, rank)
            V_3d = V.reshape(m, p, rank)
            W_3d = W.reshape(p, n, rank)
            
            T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
            err = np.max(np.abs(T - 1.0))
            
            if err < best_err:
                best_err = err
                best = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-6:
                break
            
            # Gradient-like update
            # dT/dU[i,j,r] = V[j,k,r] * W[k,i,r] summed over k
            # dT/dV[j,k,r] = U[i,j,r] * W[k,i,r] summed over i
            # dT/dW[k,i,r] = U[i,j,r] * V[j,k,r] summed over j
            
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
                        # dT[i,j,k]/dU[i,j,r] = V[j,k,r]*W[k,i,r]
                        # dT[i,j,k]/dV[j,k,r] = U[i,j,r]*W[k,i,r]
                        # dT[i,j,k]/dW[k,i,r] = U[i,j,r]*V[j,k,r]
                        for r in range(rank):
                            vu = V[j * p + k, r] * W[k * n + i, r]
                            vv = U[i * m + j, r] * W[k * n + i, r]
                            vw = U[i * m + j, r] * V[j * p + k, r]
                            grad_U[i * m + j, r] += vu
                            grad_V[j * p + k, r] += vv
                            grad_W[k * n + i, r] += vw
            
            # SGD step
            lr = 0.01
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Recheck
            U_3d = U.reshape(n, m, rank)
            V_3d = V.reshape(m, p, rank)
            W_3d = W.reshape(p, n, rank)
            T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
            err = np.max(np.abs(T - 1.0))
            
            if err < best_err:
                best_err = err
                best = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-6:
                break
        
        if best is not None and best_err < 1e-6:
            U, V, W = best
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None

    # Try explicit Strassen block construction first
    result = try_strassen_blocks()
    if result is not None:
        return result
    
    # Try rank 32 with random search
    if time.time() < _deadline - 20:
        result = random_search(32, n_iter=3000, seed=42)
        if result is not None:
            return result
    
    # Try rank 34
    if time.time() < _deadline - 20:
        result = random_search(34, n_iter=3000, seed=123)
        if result is not None:
            return result
    
    # Try rank 36
    if time.time() < _deadline - 20:
        result = random_search(36, n_iter=2000, seed=456)
        if result is not None:
            return result
    
    # Try rank 38
    if time.time() < _deadline - 20:
        result = random_search(38, n_iter=2000, seed=789)
        if result is not None:
            return result
    
    # Try rank 40 (schoolbook)
    if time.time() < _deadline - 20:
        result = random_search(40, n_iter=1000, seed=101)
        if result is not None:
            return result
    
    # Fall back to schoolbook
    return schoolbook()
