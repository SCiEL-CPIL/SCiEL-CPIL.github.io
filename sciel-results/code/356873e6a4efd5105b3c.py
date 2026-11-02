"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 4, 5
    # Output is n x p = 3 x 5
    # U: (n*m) x rank = 12 x rank
    # V: (m*p) x rank = 20 x rank  
    # W: (p*n) x rank = 15 x rank
    # Convention: W[k*n + i, r]
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Verify a decomposition: check that for all i,j,k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if this is the "diagonal" 
    # Actually, the tensor decomposition condition is:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T[i,j,k] = 1 if (i,j,k) corresponds to a valid multiplication term
    # For matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor has T[i,j,k] = 1 for all valid (i,j,k) where i in [0,n), j in [0,m), k in [0,p)
    # Wait, that's not right. The bilinear complexity counts multiplications.
    # The tensor for matrix multiplication (n,m,p) has entries:
    # T_{i,j,k} where i in [n], j in [m], k in [p], and T_{i,j,k} = 1
    # because C[i,k] = sum_j A[i,j] B[j,k], so each (i,j,k) triple contributes one product.
    # The rank is the minimum r such that T_{i,j,k} = sum_{r=1}^r u_{i,j}^r v_{j,k}^r w_{k,i}^r
    # Wait, the W indexing: W[k*n+i, r] means the W factor has index (k,i) for output C[i,k]
    
    # So the condition is: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for "invalid" combinations, it should be 0, but since we only have valid (i,j,k) triples,
    # all entries of the tensor are 1.
    
    # Actually, the tensor T has shape (n*m, m*p, p*n) but it's not fully dense in a simple way.
    # T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T[a,b,c] = 1 if j = j' and i = i' and k = k', else 0
    # Wait no. Let me re-examine.
    
    # The matrix multiplication tensor: we want to compute C = AB where A is n x m, B is m x p.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The bilinear form: <C, X> = sum_{i,k} C[i,k] X[i,k] = sum_{i,j,k} A[i,j] B[j,k] X[i,k]
    # So the tensor T has T[i,j,k] = X[i,k] when contracted with A and B... 
    # No, T is the trilinear form: T(A,B,X) = sum_{i,j,k} A[i,j] B[j,k] X[i,k]
    # The "tensor" in the decomposition sense: we write T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r is a vector in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
    # and T[a,b,c] = sum_r u_r[a] v_r[b] w_r[c]
    # where a indexes (i,j), b indexes (j',k), c indexes (k',i')
    # T[a,b,c] = 1 if the indices are consistent: j = j', i = i', k = k'
    # Otherwise 0.
    
    # So for a = i*m+j, b = j'*p+k, c = k'*n+i':
    # T[a,b,c] = 1 if j == j' and i == i' and k == k', else 0
    
    def get_tensor_entry(a, b, c):
        i, j = divmod(a, m)
        jp, k = divmod(b, p)
        kn, ii = divmod(c, n)
        # j is the column of A, jp is the row of B
        # i is row of A, ii is row of C
        # k is col of B, kn is col of C
        if j == jp and i == ii and k == kn:
            return 1
        return 0
    
    def verify(U, V, W, rank):
        """Check if the decomposition is exact."""
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    expected = get_tensor_entry(a, b, c)
                    actual = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    def round_to_half_int(mat):
        """Round matrix entries to nearest half-integer."""
        return [[round(x * 2) / 2 for x in row] for row in mat]
    
    # Try to find a decomposition of given rank using random search
    def try_rank(rank, max_iters=5000):
        """Try to find a rank-r decomposition."""
        for attempt in range(max_iters):
            if time.time() > deadline:
                return None
            # Random initialization
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            # Try a few gradient steps (simple)
            # The objective: minimize sum_{a,b,c} (sum_r U[a,r]V[b,r]W[c,r] - T[a,b,c])^2
            # This is expensive to compute fully, so use random sampling
            for step in range(50):
                if time.time() > deadline:
                    return None
                # Sample random (a,b,c)
                for _ in range(20):
                    a = random.randint(0, n*m-1)
                    b = random.randint(0, m*p-1)
                    c = random.randint(0, p*n-1)
                    expected = get_tensor_entry(a, b, c)
                    actual = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                    err = actual - expected
                    # Gradient descent step (simplified)
                    lr = 0.01
                    for r in range(rank):
                        # d/dU[a,r] = err * V[b,r] * W[c,r]
                        U[a, r] -= lr * err * V[b, r] * W[c, r]
                        V[b, r] -= lr * err * U[a, r] * W[c, r]
                        W[c, r] -= lr * err * U[a, r] * V[b, r]
            
            # Check if close to integer/half-integer
            U_r = round_to_half_int(U.tolist())
            V_r = round_to_half_int(V.tolist())
            W_r = round_to_half_int(W.tolist())
            if verify(U_r, V_r, W_r, rank):
                return {"U": U_r, "V": V_r, "W": W_r}
        
        return None
    
    # Start from schoolbook and try to reduce rank
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try ranks from a reasonable lower bound upward
    # Lower bound: max(n*p, n*m, m*p) = max(15, 12, 20) = 20
    # But realistically, for 3x4x5, the rank is likely between 20 and 60
    # Try a few candidate ranks
    for rank in range(20, best_rank):
        if time.time() > deadline:
            break
        result = try_rank(rank, max_iters=200)
        if result is not None:
            best_result = result
            best_rank = rank
            break
    
    return best_result
# EVOLVE-BLOCK-END

