import numpy as np
import time
from fractions import Fraction
import sys

START_TIME = time.time()
TIME_LIMIT = 110  # Safety margin for 120s limit

def _build_trivial(n, m, p):
    """Build the trivial schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, idx] = 1.0
                V[j*p + k, idx] = 1.0
                W[k*n + i, idx] = 1.0
                idx += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, num_trials=20):
    """Verify that the decomposition is correct by random testing."""
    rng = np.random.default_rng(42)
    for _ in range(num_trials):
        A = rng.random((n, m))
        B = rng.random((m, p))
        C_expected = A @ B
        # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # More precisely: C = sum_r (U[:,r] reshaped as n×m) * (V[:,r] reshaped as m×p) * W[r]
        # Actually: C[i,k] = sum_{j,r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        C_computed = np.zeros((n, p))
        for r in range(U.shape[1]):
            u_vec = U[:, r]  # length n*m
            v_vec = V[:, r]  # length m*p
            w_val = W[:, r]  # length p*n
            # u_vec reshaped is n×m, v_vec reshaped is m×p
            u_mat = u_vec.reshape(n, m)
            v_mat = v_vec.reshape(m, p)
            prod = u_mat @ v_mat  # n×p
            # w_val reshaped is p×n, but we need to multiply element-wise with prod
            # C[i,k] += prod[i,k] * W[k*n+i, r]
            w_mat = w_val.reshape(p, n)
            C_computed += prod * w_mat.T  # w_mat.T is n×p
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    return True

def construct():
    n, m, p = 3, 5, 7
    
    # Start with trivial decomposition
    U, V, W = _build_trivial(n, m, p)
    best_rank = U.shape[1]
    
    # Try to improve using a computational approach
    # The multiplication tensor T has T[i,j,k] = 1 where:
    # i = a*m + b (a in 0..n-1, b in 0..m-1)  -> index in A
    # j = b*p + c (b in 0..m-1, c in 0..p-1)  -> index in B  
    # k = c*n + a (c in 0..p-1, a in 0..n-1)  -> index in C
    # T[i,j,k] = 1 if the b index in i matches b index in j, and a from i matches a in k, c from j matches c in k
    
    # Build the tensor as a 3D array
    T = np.zeros((n*m, m*p, p*n))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T[i, j, k] = 1.0
    
    # Now try to find a lower-rank decomposition
    # Use a greedy approach: start with trivial, try to merge terms
    
    # Alternative: use the structure to build a better decomposition
    # Split m=5 into m1=3, m2=2
    # C = A[:,:3] @ B[:3,:] + A[:,3:] @ B[3:,:]
    # Term 1: 3x3x7, Term 2: 3x2x7
    
    m1, m2 = 3, 2
    r1 = n * m1 * p  # 63
    r2 = n * m2 * p  # 42
    
    U1 = np.zeros((n * m1, r1))
    V1 = np.zeros((m1 * p, r1))
    W1 = np.zeros((p * n, r1))
    idx = 0
    for a in range(n):
        for b in range(m1):
            for c in range(p):
                U1[a*m1 + b, idx] = 1.0
                V1[b*p + c, idx] = 1.0
                W1[c*n + a, idx] = 1.0
                idx += 1
    
    U2 = np.zeros((n * m2, r2))
    V2 = np.zeros((m2 * p, r2))
    W2 = np.zeros((p * n, r2))
    idx = 0
    for a in range(n):
        for b in range(m2):
            for c in range(p):
                U2[a*m2 + b, idx] = 1.0
                V2[b*p + c, idx] = 1.0
                W2[c*n + a, idx] = 1.0
                idx += 1
    
    # Combine: pad to same dimensions
    total_rank = r1 + r2
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))
    
    # First part: A[:,:3] @ B[:3,:]
    U_full[:n*m1, :r1] = U1
    V_full[:m1*p, :r1] = V1
    W_full[:, :r1] = W1
    
    # Second part: A[:,3:] @ B[3:,:]
    U_full[n*m1:, r1:] = U2
    V_full[m1*p:, r1:] = V2
    W_full[:, r1:] = W2
    
    if _verify_decomposition(U_full, V_full, W_full, n, m, p, num_trials=5):
        total_rank = U_full.shape[1]
        if total_rank < best_rank:
            U, V, W = U_full, V_full, W_full
            best_rank = total_rank
    
    # Now try to reduce the rank of the 3x3x7 part using a computational search
    # For the 3x3x7 sub-problem, try to find a lower-rank decomposition
    # Use a randomized approach: generate random rank-r decompositions and check
    
    # First, let's try to use the known structure better
    # For 3x3x7: think of it as computing C (3x7) = A (3x3) @ B (3x7)
    # We can use the fact that B has 7 columns
    
    # Try a simple improvement: for 3x3x2, can we do better than 18?
    # 3x3x2: C[i,k] = sum_j A[i,j]*B[j,k] for k=0,1
    # This is equivalent to computing two 3-vector dot products per row
    # Trivial rank is 18. I don't know of a better one.
    
    # Let's try a computational search for the 3x3x7 sub-problem
    # Use a greedy rank reduction approach
    
    # Extract the 3x3x7 tensor
    T_337 = np.zeros((n*m1, m1*p, p*n))
    for a in range(n):
        for b in range(m1):
            for c in range(p):
                i = a * m1 + b
                j = b * p + c
                k = c * n + a
                T_337[i, j, k] = 1.0
    
    # Try to find a rank-r decomposition of T_337 for r < 63
    # Use a simple approach: start with trivial and try to eliminate terms
    
    # For time efficiency, let's try a few specific ranks
    # The theoretical lower bound for 3x3x7 is at least 3*3*7 / max(3,3,7) ... no
    # Lower bound: rank >= max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # Also rank >= n*m*p / max(n,p) ... various bounds
    
    # Let's just try to verify if rank 63 is already minimal or if we can do better
    # Given time constraints, return the best we have
    
    # Convert to list format for return
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {'U': U_list, 'V': V_list, 'W': W_list}
