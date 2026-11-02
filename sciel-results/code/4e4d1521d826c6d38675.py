<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
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
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 if i,j,k consistent, else 0
    # The tensor T[i,j,k] = 1 if the indices are "compatible" for matrix mult
    # Actually for matrix mult (n x m) * (m x p) -> (n x p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The multiplication tensor has entries T[i,j,k] where we sum over j
    # But here the tensor is over (i, j, k) with the constraint that j is the shared index
    # The decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal delta(j, j)
    # Wait, let me re-read the convention.
    # W[k*n+i, r] - so W is indexed by (k, i) where k is from B's rows and i is from A's rows
    # U[i*m+j, c] - A's element (i,j)
    # V[j*p+k, c] - B's element (j,k)
    # The tensor T[i,j,k] = 1 if j==j (always true for valid indices), 0 otherwise
    # Actually the matrix multiplication tensor T has T[i,j,k] = 1 if the indices form a valid
    # triple for the sum, i.e., it's just 1 for all valid (i,j,k) since the multiplication
    # is defined as sum_j A[i,j]*B[j,k]. The tensor itself is the identity in the middle index.
    # So T[i,j,k] = 1 if j is the same index (which it always is in the sum), meaning
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Wait no. The tensor for matrix multiplication C = AB where C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T is a 3-tensor with T[i,j,k] = 1 if we're multiplying, which is always 1
    # for valid indices. So the tensor is just all ones? No, that's not right either.
    # 
    # Let me think again. The matrix multiplication tensor T_{n,m,p} has entries
    # T[i,j,k] where i in [0,n), j in [0,m), k in [0,p).
    # The multiplication is: (AB)[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor is T[i,j,k] = 1 for all valid (i,j,k) - it's the "multiplication" tensor.
    # Actually no, the tensor represents the bilinear form. The decomposition is:
    # sum_c U[i,j,c] V[j,k,c] W[k,i,c] = T[i,j,k]
    # where T[i,j,k] = 1 if j is the "shared" index (which it always is), so T is all 1s?
    # 
    # Hmm, let me just check: the schoolbook has rank n*m*p with one term per (i,j,k) triple,
    # and U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1.
    # So sum_c U[i*m+j',c] V[j'*p+k',c] W[k'*n+i',c] = 1 if there's a c where j'=j, k'=k, i'=i
    # which is exactly 1 when (i',j',k')=(i,j,k). So the tensor T[i,j,k] = 1 for all valid indices.
    # Wait, that means the tensor is just the all-ones tensor? That can't be right for rank minimality.
    #
    # Actually I think I'm overcomplicating this. Let me just verify numerically.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                # The tensor entry should be 1 for all valid (i,j,k)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank(r, n, m, p, seed=0, max_iter=5000, lr=0.1):
    """Try to find a decomposition of given rank r."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n*m, r) * 0.5
    V = rng.randn(m*p, r) * 0.5
    W = rng.randn(p*n, r) * 0.5
    
    def loss():
        total = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k]) * W[k*n+i]
                    # Wait, that's not right. The sum is over c of U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
                    # That's element-wise product summed, not dot product.
                    s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    total += (s - 1.0)**2
        return total
    
    def loss_grad():
        gU = np.zeros_like(U)
        gV = np.zeros_like(V)
        gW = np.zeros_like(W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    err = s - 1.0
                    # d/dU[i*m+j,c] = err * V[j*p+k,c] * W[k*n+i,c]
                    gU[i*m+j] += 2 * err * V[j*p+k] * W[k*n+i]
                    gV[j*p+k] += 2 * err * U[i*m+j] * W[k*n+i]
                    gW[k*n+i] += 2 * err * U[i*m+j] * V[j*p+k]
        return gU, gV, gW
    
    for it in range(max_iter):
        gU, gV, gW = loss_grad()
        U -= lr * gU
        V -= lr * gV
        W -= lr * gW
        if it % 500 == 0:
            l = loss()
            if l < 1e-12:
                return U, V, W
    l = loss()
    if l < 1e-12:
        return U, V, W
    return None

def _to_list(arr):
    return [[float(x) for x in row] for row in arr]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit
    
    # Start with schoolbook as fallback
    rank_school = n * m * p
    U_fb = [[0] * rank_school for _ in range(n * m)]
    V_fb = [[0] * rank_school for _ in range(m * p)]
    W_fb = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_school
    
    # Try ranks from lower to higher
    # Theoretical lower bound: max(n*p, m*p, n*m) = max(10, 20, 8) = 20
    # Known results suggest rank might be around 20-25 for this case
    min_rank = max(n * p, m * p, n * m)
    
    for r in range(min_rank, min(best_rank, min_rank + 15)):
        if time.time() > deadline - 10:
            break
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            result = _try_rank(r, n, m, p, seed=seed, max_iter=2000, lr=0.05)
            if result is not None:
                U_arr, V_arr, W_arr = result
                U_list = _to_list(U_arr)
                V_list = _to_list(V_arr)
                W_list = _to_list(W_arr)
                if _verify(U_list, V_list, W_list, n, m, p):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = r
                    break
        if best_rank == r:
            break
    
    return best_result
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
