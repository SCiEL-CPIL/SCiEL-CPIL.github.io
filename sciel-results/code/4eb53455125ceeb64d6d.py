import time
import sys

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np

class TensorDecomposition:
    def __init__(self):
        self.cache = {}
        self.deadline = time.time() + 110  # Safety margin

    def _schoolbook(self, n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank

    def get_rank(self, n, m, p):
        if (n, m, p) in self.cache:
            return self.cache[(n, m, p)]
        
        # Base case: Schoolbook if no split is beneficial or small dims
        best_rank = n * m * p
        
        # Try splitting n
        for i in range(1, n):
            if time.time() > self.deadline:
                break
            r1 = self.get_rank(i, m, p)
            r2 = self.get_rank(n-i, m, p)
            best_rank = min(best_rank, r1 + r2)
            
        # Try splitting m
        for j in range(1, m):
            if time.time() > self.deadline:
                break
            r1 = self.get_rank(n, j, p)
            r2 = self.get_rank(n, m-j, p)
            best_rank = min(best_rank, r1 + r2)
            
        # Try splitting p
        for k in range(1, p):
            if time.time() > self.deadline:
                break
            r1 = self.get_rank(n, m, k)
            r2 = self.get_rank(n, m, p-k)
            best_rank = min(best_rank, r1 + r2)
            
        self.cache[(n, m, p)] = best_rank
        return best_rank

    def _try_rank_r_decomposition(self, n, m, p, r, max_iter=200, seed=42):
        """
        Try to find a rank-r decomposition using random initialization and 
        alternating least squares / gradient descent on the tensor entries.
        Returns (U, V, W, rank) if successful, else None.
        """
        if r < max(n*m, m*p, n*p):
            return None
            
        np.random.seed(seed)
        
        # Initialize U, V, W with random values
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        # Normalize to avoid scaling issues
        U = U / np.sqrt(np.sum(U**2, axis=0, keepdims=True) + 1e-12)
        V = V / np.sqrt(np.sum(V**2, axis=0, keepdims=True) + 1e-12)
        W = W / np.sqrt(np.sum(W**2, axis=0, keepdims=True) + 1e-12)
        
        # Target tensor T[i,j,k] = 1
        # We want sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        # Reshape for vectorization
        U_flat = U  # (n*m, r)
        V_flat = V  # (m*p, r)
        W_flat = W  # (p*n, r)
        
        # Precompute index mappings
        # For each (i, j, k), we need U[i*m+j, :], V[j*p+k, :], W[k*n+i, :]
        
        best_loss = float('inf')
        
        for it in range(max_iter):
            if time.time() > self.deadline:
                break
                
            # Compute current tensor
            # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            T = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] = np.sum(U_flat[i*m+j, :] * V_flat[j*p+k, :] * W_flat[k*n+i, :])
            
            # Loss: sum (T[i,j,k] - 1)^2
            loss = np.sum((T - 1.0)**2)
            
            if loss < 1e-10:
                # Success! Convert to lists
                U_list = U_flat.tolist()
                V_list = V_flat.tolist()
                W_list = W_flat.tolist()
                return U_list, V_list, W_list, r
            
            if loss < best_loss:
                best_loss = loss
                best_U = U_flat.copy()
                best_V = V_flat.copy()
                best_W = W_flat.copy()
            
            # Gradient descent step
            # dL/dU[i*m+j, c] = 2 * (T[i,j,k] - 1) * V[j*p+k, c] * W[k*n+i, c]
            # This is complex to vectorize fully, so we do it iteratively for small n,m,p
            
            lr = 0.01
            dU = np.zeros_like(U_flat)
            dV = np.zeros_like(V_flat)
            dW = np.zeros_like(W_flat)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        err = T[i, j, k] - 1.0
                        u_idx = i*m + j
                        v_idx = j*p + k
                        w_idx = k*n + i
                        for c in range(r):
                            dU[u_idx, c] += 2 * err * V_flat[v_idx, c] * W_flat[w_idx, c]
                            dV[v_idx, c] += 2 * err * U_flat[u_idx, c] * W_flat[w_idx, c]
                            dW[w_idx, c] += 2 * err * U_flat[u_idx, c] * V_flat[v_idx, c]
            
            U_flat -= lr * dU
            V_flat -= lr * dV
            W_flat -= lr * dW
            
            # Renormalize to prevent blowup
            U_flat = U_flat / (np.sqrt(np.sum(U_flat**2, axis=0, keepdims=True)) + 1e-12)
            V_flat = V_flat / (np.sqrt(np.sum(V_flat**2, axis=0, keepdims=True)) + 1e-12)
            W_flat = W_flat / (np.sqrt(np.sum(W_flat**2, axis=0, keepdims=True)) + 1e-12)
        
        # Check if best solution is good enough
        if best_loss < 1e-8:
            U_list = best_U.tolist()
            V_list = best_V.tolist()
            W_list = best_W.tolist()
            return U_list, V_list, W_list, r
            
        return None

    def construct(self, n, m, p):
        # Returns U, V, W for n x m x p
        if time.time() > self.deadline:
             # Fallback to schoolbook if deadline exceeded
             return self._schoolbook(n, m, p)
        
        # First, try to find a direct low-rank decomposition using optimization
        # Start with the theoretical lower bound
        min_possible = max(n*m, m*p, n*p)
        schoolbook_rank = n * m * p
        
        # Try ranks from lower to higher
        # For 3x5x6, min_possible = 30, schoolbook = 90
        # We'll try a limited range to stay within time budget
        for r in range(min_possible, min(schoolbook_rank + 1, min_possible + 30)):
            if time.time() > self.deadline:
                break
                
            # Try a few random seeds
            for seed in [42, 123, 777, 999, 2024]:
                if time.time() > self.deadline:
                    break
                result = self._try_rank_r_decomposition(n, m, p, r, max_iter=50, seed=seed)
                if result is not None:
                    U, V, W, rank = result
                    self.cache[(n, m, p)] = rank
                    return U, V, W, rank
        
        # If optimization fails, fall back to recursive splitting
        best_rank = n * m * p
        best_split = None
        
        # Try splits
        for i in range(1, n):
            if time.time() > self.deadline:
                break
            r1 = self.get_rank(i, m, p)
            r2 = self.get_rank(n-i, m, p)
            if r1 + r2 < best_rank:
                best_rank = r1 + r2
                best_split = ('n', i)
                
        for j in range(1, m):
            if time.time() > self.deadline:
                break
            r1 = self.get_rank(n, j, p)
            r2 = self.get_rank(n, m-j, p)
            if r1 + r2 < best_rank:
                best_rank = r1 + r2
                best_split = ('m', j)
                
        for k in range(1, p):
            if time.time() > self.deadline:
                break
            r1 = self.get_rank(n, m, k)
            r2 = self.get_rank(n, m, p-k)
            if r1 + r2 < best_rank:
                best_rank = r1 + r2
                best_split = ('p', k)

        if best_split is None:
            return self._schoolbook(n, m, p)
        else:
            dim, idx = best_split
            if dim == 'n':
                n1, n2 = idx, n - idx
                U1, V1, W1, r1 = self.construct(n1, m, p)
                U2, V2, W2, r2 = self.construct(n2, m, p)
                
                U = [[0] * (r1 + r2) for _ in range(n * m)]
                V = [[0] * (r1 + r2) for _ in range(m * p)]
                W = [[0] * (r1 + r2) for _ in range(p * n)]
                
                for i in range(n1):
                    for j in range(m):
                        for c in range(r1):
                            U[i*m+j][c] = U1[i*m+j][c]
                for i in range(n2):
                    for j in range(m):
                        for c in range(r2):
                            U[(n1+i)*m+j][r1+c] = U2[i*m+j][c]
                            
                for j in range(m):
                    for k in range(p):
                        for c in range(r1):
                            V[j*p+k][c] = V1[j*p+k][c]
                        for c in range(r2):
                            V[j*p+k][r1+c] = V2[j*p+k][c]
                            
                for k in range(p):
                    for i in range(n1):
                        for c in range(r1):
                            W[k*n+i][c] = W1[k*n1+i][c]
                    for i in range(n2):
                        for c in range(r2):
                            W[k*n + (n1+i)][r1+c] = W2[k*n2+i][c]
                            
                return U, V, W, r1 + r2
                
            elif dim == 'm':
                m1, m2 = idx, m - idx
                U1, V1, W1, r1 = self.construct(n, m1, p)
                U2, V2, W2, r2 = self.construct(n, m2, p)
                
                U = [[0] * (r1 + r2) for _ in range(n * m)]
                V = [[0] * (r1 + r2) for _ in range(m * p)]
                W = [[0] * (r1 + r2) for _ in range(p * n)]
                
                for i in range(n):
                    for j in range(m1):
                        for c in range(r1):
                            U[i*m+j][c] = U1[i*m1+j][c]
                    for j in range(m2):
                        for c in range(r2):
                            U[i*m + (m1+j)][r1+c] = U2[i*m2+j][c]
                            
                for j in range(m1):
                    for k in range(p):
                        for c in range(r1):
                            V[j*p+k][c] = V1[j*p+k][c]
                for j in range(m2):
                    for k in range(p):
                        for c in range(r2):
                            V[(m1+j)*p+k][r1+c] = V2[j*p+k][c]
                            
                for k in range(p):
                    for i in range(n):
                        for c in range(r1):
                            W[k*n+i][c] = W1[k*n+i][c]
                        for c in range(r2):
                            W[k*n+i][r1+c] = W2[k*n+i][c]
                            
                return U, V, W, r1 + r2
            else: # dim == 'p'
                p1, p2 = idx, p - idx
                U1, V1, W1, r1 = self.construct(n, m, p1)
                U2, V2, W2, r2 = self.construct(n, m, p2)
                
                U = [[0] * (r1 + r2) for _ in range(n * m)]
                V = [[0] * (r1 + r2) for _ in range(m * p)]
                W = [[0] * (r1 + r2) for _ in range(p * n)]
                
                for i in range(n):
                    for j in range(m):
                        for c in range(r1):
                            U[i*m+j][c] = U1[i*m+j][c]
                        for c in range(r2):
                            U[i*m+j][r1+c] = U2[i*m+j][c]
                            
                for j in range(m):
                    for k in range(p1):
                        for c in range(r1):
                            V[j*p+k][c] = V1[j*p1+k][c]
                    for k in range(p2):
                        for c in range(r2):
                            V[j*p + (p1+k)][r1+c] = V2[j*p2+k][c]
                            
                for k in range(p1):
                    for i in range(n):
                        for c in range(r1):
                            W[k*n+i][c] = W1[k*n+i][c]
                for k in range(p2):
                    for i in range(n):
                        for c in range(r2):
                            W[(p1+k)*n+i][r1+c] = W2[k*n+i][c]
                            
                return U, V, W, r1 + r2

def construct():
    n, m, p = DIMENSIONS
    solver = TensorDecomposition()
    U, V, W, rank = solver.construct(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
