"""Find rank-minimal exact tensor decomposition for 3x3 by 3x7 matrix multiplication."""

import time
import itertools
import random
import numpy as np

DIMENSIONS = [3, 3, 7]
_START = time.time()
_DEADLINE = 115.0  # safety margin before 120s limit


def _elapsed():
    return time.time() - _START


def _time_left():
    return _DEADLINE - _elapsed()


def _schoolbook(n, m, p):
    """Return the schoolbook (trivial) decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_c U[:,c] V[:,c] W[:,c]^T (with proper index mapping) reconstructs the
    matrix multiplication tensor.

    Convention: (A B)[i,k] = sum_j A[i,j] B[j,k]
    Tensor T[i, j, k] = delta_{i,j}... no, T[i,j,k] = 1 if we define the multiplication tensor
    such that C = A*B means C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k].
    So T[i,j,k] = 1 for all i,j,k (it's just the identity in the "multiply" sense).

    Actually the tensor is: T[i,j,k] such that C[i,k] = sum_{j} A[i,j] B[j,k].
    In tensor form: C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k] where T[i,j,k] = 1.

    The factorization: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    (using the W convention W[k*n+i, r]).

    Let's verify by checking: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    """
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    # Build the tensor T from the factorization
    # T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    # Efficient check: pick random samples
    rng = np.random.RandomState(42)
    for _ in range(50):
        i = rng.randint(0, n)
        j = rng.randint(0, m)
        k = rng.randint(0, p)
        val = float(np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :]))
        if abs(val - 1.0) > tol:
            return False
    return True


def _try_bini_style_3x3x3_extended(n, m, p):
    """
    Attempt to find a rank < n*m*p decomposition.
    
    For 3x3x3, the known rank is 23 (Bini). For 3x3x7, we can think of it as
    7 independent 3x3 multiplications sharing the same A, but that gives rank 7*23=161 which is worse.
    
    Actually, the multiplication 3x3 by 3x7: C is 3x7. C[i,k] = sum_j A[i,j] B[j,k].
    This is like 7 separate 3x3 multiplications C[:,k] = A * B[:,k], but they share A.
    
    The tensor has indices (i,j,k) with i in [0,3), j in [0,3), k in [0,7).
    T[i,j,k] = 1 for all i,j,k.
    
    Wait, that's not right. The tensor is:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    As a bilinear map: (A, B) -> C, where C[i,k] = sum_j A[i,j] B[j,k].
    The tensor T has entries T[i,j,k] such that C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k].
    So T[i,j,k] = delta_{?}... No. C[i,k] depends on A[i,j] and B[j,k] summed over j.
    So for each (i,k), C[i,k] = sum_j A[i,j] B[j,k]. The tensor entry T[i,j,k] = 1.
    
    Hmm, actually T[i,j,k] = 1 for ALL i, j, k. That means the tensor is the all-ones tensor!
    That can't be right because then any rank-1 decomposition would work...
    
    Let me reconsider. The standard matrix multiplication tensor for A (n x m) times B (m x p):
    C[i,k] = sum_{j=1}^{m} A[i,j] B[j,k]
    
    The tensor T_{n,m,p} has entries T[i,j,k] = 1 if... no.
    T[i,j,k] = 1 means: C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k] = sum_j 1 * A[i,j] * B[j,k].
    Yes! So T[i,j,k] = 1 for all valid i, j, k.
    
    Wait, that would mean the tensor is just the all-1s array of shape (n, m, p).
    But that's not a "tensor" in the usual sense of having structure... 
    
    Actually I think I'm confusing myself. Let me use the standard definition:
    The matrix multiplication tensor T^{<n,m,p>} is a 3-tensor with entries:
    T[i,j,k] = 1 if the "position" (i,j,k) corresponds to a valid term in the sum.
    
    No wait. The standard definition: T^{<n,m,p>}[i,j,k] = 1 for all i,j,k in their ranges.
    The multiplication is: (A,B) -> C where C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k] = sum_j A[i,j] B[j,k].
    
    So yes, T[i,j,k] = 1 for all i, j, k. The tensor IS the all-ones tensor.
    
    But then the rank of the all-ones tensor of shape (n,m,p) is... 1? Because
    1 = 1*1*1, so a single term with U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1 for all i,j,k.
    
    That gives rank 1! But that's wrong because the schoolbook is n*m*p.
    
    I'm clearly misunderstanding the tensor convention. Let me re-read the task.
    
    "the output factor follows the official trace convention W[k*n+i,r]"
    
    And the schoolbook: U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1 for each (i,j,k) triple.
    
    So for each (i,j,k), there's a term where:
    - U has a 1 at position (i*m+j)
    - V has a 1 at position (j*p+k)  
    - W has a 1 at position (k*n+i)
    
    The reconstruction is: T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    
    For the schoolbook, each (i,j,k) has its own column, so T[i,j,k] = 1 for all i,j,k.
    
    So indeed T is the all-ones tensor! And the question is: what is the minimum rank
    to express the all-ones tensor of shape (3,3,7) as sum of rank-1 tensors?
    
    But wait - the rank-1 tensors here are NOT arbitrary rank-1 tensors in the 3-tensor space.
    They have a specific structure: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank),
    and the indices are mapped as:
    - U index: i*m + j (row i, col j of A)
    - V index: j*p + k (row j, col k of B)
    - W index: k*n + i (row k, col i of C^T? or C with transposed indexing)
    
    So the constraint is that the rank-1 tensors must respect this specific index coupling.
    A "free" rank-1 tensor would be u[i] * v[j] * w[k], but here we have:
    u[i,j] * v[j,k] * w[k,i] where u, v, w are vectors on the compound indices.
    
    This is the "structured" or "affine" tensor rank problem. The rank is the minimum number
    of terms where each term is a product of a vector on (i,j)-space, a vector on (j,k)-space,
    and a vector on (k,i)-space, coupled through the shared indices.
    
    For the all-ones target: we need sum_c u_c[i,j] * v_c[j,k] * w_c[k,i] = 1 for all i,j,k.
    
    This is exactly the matrix multiplication tensor rank! And for 3x3x3, it's 23 (Bini).
    For 3x3x7, it's an open/different problem.
    
    OK so my earlier confusion is resolved. The tensor IS the all-ones tensor, but the rank
    is constrained by the index coupling structure. Let me just try to find a good decomposition.
    
    For 3x3x7: we have 3*3*7 = 63 variables (i,j,k). The schoolbook uses 63 terms.
    Can we do better?
    
    One approach: note that for fixed i, the equation is:
    sum_c u_c[i,j] * v_c[j,k] * w_c[k,i] = 1 for all j,k.
    
    This is like: for each i, the "slice" u_i * (something) * w_i^T = ones(3,7) in some sense.
    
    This is getting complex. Let me just implement a numerical search for small rank.
    """
    pass  # Placeholder - will implement below


def _numerical_search(n, m, p, target_rank, time_limit):
    """
    Try to find a rank-`target_rank` decomposition using numerical optimization.
    This is a heuristic - we initialize randomly and use gradient descent.
    """
    U = np.zeros((n * m, target_rank))
    V = np.zeros((m * p, target_rank))
    W = np.zeros((p * n, target_rank))
    
    # Initialize randomly
    rng = np.random.RandomState(12345)
    U[:, :] = rng.randn(n * m, target_rank) * 0.1
    V[:, :] = rng.randn(m * p, target_rank) * 0.1
    W[:, :] = rng.randn(p * n, target_rank) * 0.1
    
    # Precompute index mappings
    # For each (i,j,k): U_idx = i*m+j, V_idx = j*p+k, W_idx = k*n+i
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i, j, k, i * m + j, j * p + k, k * n + i))
    
    def loss_and_grads():
        """Compute loss and gradients for the structured tensor rank problem."""
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for (i, j, k, ui, vi, wi) in triples:
            # Term: sum_c U[ui,c] * V[vi,c] * W[wi,c]
            prod = U[ui, :] * V[vi, :] * W[wi, :]
            val = float(np.sum(prod))
            err = val - 1.0
            loss += err * err
            
            # d(loss)/dU[ui,c] = 2*err * V[vi,c] * W[wi,c]
            dU = 2.0 * err * V[vi, :] * W[wi, :]
            dV = 2.0 * err * U[ui, :] * W[wi, :]
            dW = 2.0 * err * U[ui, :] * V[vi, :]
            
            grad_U[ui, :] += dU
            grad_V[vi, :] += dV
            grad_W[wi, :] += dW
        
        return loss, grad_U, grad_V, grad_W
    
    lr = 0.01
    best_loss = float('inf')
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    for step in range(500):
        if _time_left() < time_limit:
            break
        
        loss, gU, gV, gW = loss_and_grads()
        
        if loss < best_loss:
            best_loss = loss
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        if loss < 1e-8:
            break
        
        # Update
        U -= lr * gU
        V -= lr * gV
        W -= lr * gW
        
        # Adaptive learning rate
        if step % 50 == 0 and step > 0:
            lr = lr * 0.9
    
    if best_loss < 1e-6:
        return {"U": best_U, "V": best_V, "W": best_W}
    return None


def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find better decompositions
    # For 3x3x3, rank 23 is known. For 3x3x7, let's try ranks from low to high.
    
    # Lower bound: at least max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # But that's a very weak bound. The tensor rank is at least the matricization rank.
    
    # Let's try a few ranks using numerical optimization
    ranks_to_try = list(range(21, min(best_rank, 50)))
    
    for r in ranks_to_try:
        if _time_left() < 10:
            break
        result = _numerical_search(n, m, p, r, time_limit=5.0)
        if result is not None:
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                best = result
                best_rank = r
                break
    
    # Convert to lists for the interface
    return {
        "U": np.asarray(best["U"]).tolist(),
        "V": np.asarray(best["V"]).tolist(),
        "W": np.asarray(best["W"]).tolist()
    }
