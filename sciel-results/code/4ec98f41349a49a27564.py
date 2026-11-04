"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]
import numpy as np
import time
import itertools
import random

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly decomposes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    U_arr = U.T  # (rank, n*m)
    V_arr = V.T  # (rank, m*p)
    W_arr = W.T  # (rank, p*n)
    
    for i in range(n):
        for k in range(p):
            for a in range(n):
                for b in range(m):
                    for c in range(m):
                        for d in range(p):
                            expected = 1.0 if (a == i and b == c and d == k) else 0.0
                            s = np.sum(W_arr[:, k*n+i] * U_arr[:, a*m+b] * V_arr[:, c*p+d])
                            if abs(s - expected) > tol:
                                return False
    return True

def _make_schoolbook(n, m, p):
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

def _try_rank_reduction(n, m, p, target_rank, time_budget=30.0):
    """Try to find a decomposition of the given target_rank.
    
    Uses a structured approach: start from schoolbook and try to merge terms
    by exploiting linear dependencies.
    """
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def still_going():
        return elapsed() < time_budget
    
    # For 2x5x5, the schoolbook rank is 50.
    # Let's try to find a rank-49 or lower decomposition.
    
    # Key insight: The tensor T[i,j,k,l] = delta_{i,i} * delta_{j,k} * delta_{l,l}
    # Actually, let me be more precise.
    # A is n x m = 2 x 5, B is m x p = 5 x 5, C is n x p = 2 x 5.
    # C[i,l] = sum_j A[i,j] * B[j,l]
    # The bilinear map is: (A, B) -> C
    # In tensor form: T[a, b, c] where a indexes A (n*m=10), b indexes B (m*p=25), c indexes C (n*p=10)
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+l, c = i*p+l for some i,j,l; else 0.
    # Wait, c should index C which is n x p, so c = i*p + l.
    # But the task says W has shape (p*n, rank), so W indexes (l, i) with l in [0,p), i in [0,n).
    # So W index = l*n + i = l*2 + i.
    
    # Let me re-read the task: "The output factor follows the official trace convention W[k*n+i,r]"
    # So W index = k*n + i where k is the column of C (and B), i is the row of C (and A).
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor entry is T[a, b, c] where:
    # a = i*m + j (index into A, row i, col j)
    # b = j*p + k (index into B, row j, col k)
    # c = k*n + i (index into C, row i, col k) -- wait, C is n x p, so C[i,k]
    # But W index = k*n + i, which would be C[k, i] if we think of W as indexing a p x n matrix.
    # Hmm, let me just trust the verify function.
    
    # The verify function checks:
    # sum_r W[k*n+i] * U[a*m+b] * V[c*p+d] = 1 if a==i and b==c and d==k
    # So U index = a*m + b where a in [0,n), b in [0,m) -- this is A[a,b]
    # V index = c*p + d where c in [0,m), d in [0,p) -- this is B[c,d]
    # W index = k*n + i where k in [0,p), i in [0,n) -- this is C[i,k] but indexed as k*n+i
    
    # So the tensor is: T[U_idx, V_idx, W_idx] = 1 if U_idx = a*m+b, V_idx = b*p+d, W_idx = d*n+a
    # for some a in [0,n), b in [0,m), d in [0,p).
    
    # For n=2, m=5, p=5:
    # U has 10 entries, V has 25 entries, W has 10 entries.
    # The tensor has 10*25*10 = 2500 entries, of which 2*5*5 = 50 are 1.
    
    # Schoolbook: 50 rank-1 terms, each with a single 1 in U, V, W.
    
    # To reduce rank, we need to find linear combinations of these terms that cancel out.
    
    # Let's try a specific approach: for each pair of terms that share a V index,
    # see if we can merge them.
    
    # Terms (a, b, d) and (a', b, d') share V index b*p+d if b==b' and d==d'.
    # Wait, V index = b*p + d. So terms (a, b, d) and (a', b, d) share V index.
    # U indices: a*m+b and a'*m+b -- different if a != a'.
    # W indices: d*n+a and d*n+a' -- different if a != a'.
    
    # For n=2, for fixed b and d, we have terms (0, b, d) and (1, b, d).
    # U indices: b and 5+b
    # V index: b*5+d
    # W indices: d*2+0 = 2d and d*2+1 = 2d+1
    
    # Can we replace these 2 terms with 1 term?
    # We need a vector u, v, w such that:
    # u[0*m+b] * v[b*p+d] * w[d*n+0] = 1 (for term (0,b,d))
    # u[1*m+b] * v[b*p+d] * w[d*n+1] = 1 (for term (1,b,d))
    # And all other entries should be 0.
    
    # If we set v[b*p+d] = 1, then we need:
    # u[b] * w[2d] = 1
    # u[5+b] * w[2d+1] = 1
    # And u[other] = 0, v[other] = 0, w[other] = 0.
    
    # This is possible! Set u[b] = 1, w[2d] = 1, u[5+b] = 1, w[2d+1] = 1.
    # But wait, this single term would also contribute to:
    # u[b] * v[b*p+d] * w[2d+1] = 1 * 1 * 1 = 1 (wrong, should be 0)
    # u[5+b] * v[b*p+d] * w[2d] = 1 * 1 * 1 = 1 (wrong, should be 0)
    
    # So a single term can't replace two terms that share V but have different U and W.
    
    # What if we allow the term to have support on multiple V indices?
    # Let's say the term has v[b*p+d] = 1 and v[b*p+d'] = 1 for some d'.
    # Then it would also contribute to terms involving d'.
    
    # This is getting complicated. Let's try a different approach.
    
    # Use the fact that the tensor is sparse and try to find a low-rank approximation
    # that is exact.
    
    # Actually, let's try a random search: initialize U, V, W randomly and try to
    # optimize them to match the tensor.
    
    if target_rank >= 50:
        return None
    
    # Try random initialization and gradient descent
    np.random.seed(42)
    
    U = np.random.randn(n*m, target_rank) * 0.1
    V = np.random.randn(m*p, target_rank) * 0.1
    W = np.random.randn(p*n, target_rank) * 0.1
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for a in range(n):
        for b in range(m):
            for d in range(p):
                T[a*m+b, b*p+d, d*n+a] = 1.0
    
    # Flatten for easier computation
    T_flat = T.reshape(-1)
    
    # Compute current tensor
    def compute_tensor(U, V, W):
        # sum_r U[:,r] * V[:,r] * W[:,r]
        # This is a 3-way tensor
        result = np.zeros((n*m, m*p, p*n))
        for r in range(target_rank):
            result += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
        return result
    
    # Loss function
    def loss(U, V, W):
        T_current = compute_tensor(U, V, W)
        return np.sum((T_current - T)**2)
    
    # Gradient descent
    lr = 0.01
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for step in range(10000):
        if not still_going():
            break
        
        # Compute loss
        T_current = compute_tensor(U, V, W)
        current_loss = np.sum((T_current - T)**2)
        
        if current_loss < best_loss:
            best_loss = current_loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < 1e-10:
            break
        
        # Compute gradients
        # dL/dU[i,r] = 2 * sum_{j,k} (T_current[i,j,k] - T[i,j,k]) * V[j,r] * W[k,r]
        # dL/dV[j,r] = 2 * sum_{i,k} (T_current[i,j,k] - T[i,j,k]) * U[i,r] * W[k,r]
        # dL/dW[k,r] = 2 * sum_{i,j} (T_current[i,j,k] - T[i,j,k]) * U[i,r] * V[j,r]
        
        diff = T_current - T
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for r in range(target_rank):
            # grad_U[:,r] = 2 * diff @ (V[:,r] * W[:,r])
            # Actually, diff is (n*m, m*p, p*n), and we want to contract over the last two dims
            # grad_U[i,r] = 2 * sum_{j,k} diff[i,j,k] * V[j,r] * W[k,r]
            # = 2 * (diff * np.einsum('j,k->jk', V[:,r], W[:,r])).sum(axis=(1,2))
            
            vw = np.einsum('j,k->jk', V[:,r], W[:,r])
            grad_U[:,r] = 2 * np.sum(diff * vw, axis=(1,2))
            
            uv = np.einsum('i,k->ik', U[:,r], W[:,r])
            grad_V[:,r] = 2 * np.sum(diff * uv, axis=(0,2))
            
            uv2 = np.einsum('i,j->ij', U[:,r], V[:,r])
            grad_W[:,r] = 2 * np.sum(diff * uv2, axis=(0,1))
        
        # Update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        lr *= 0.9999
    
    if best_loss < 1e-10:
        return best_U, best_V, best_W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Start with schoolbook
    best = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    
    def elapsed():
        return time.time() - start_time
    
    def still_going():
        return elapsed() < deadline
    
    # Try to reduce rank
    for rank in range(best_rank - 1, max(best_rank - 5, 40), -1):
        if not still_going():
            break
        
        remaining = deadline - elapsed()
        if remaining < 10:
            break
        
        budget = min(remaining * 0.8, 20.0)
        
        result = _try_rank_reduction(n, m, p, rank, time_budget=budget)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p, tol=1e-8):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                break
    
    return best
# EVOLVE-BLOCK-END
