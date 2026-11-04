"""Structured low-rank search for 4x4x7 matrix multiplication.

Uses parametric Kronecker-structured terms to reduce search space.
Each term is defined by small parameter vectors, and we search over
a discrete/continuous parameter space for rank-57 solutions.
"""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]
DEADLINE = 115.0
START_TIME = time.time()

def _time_left():
    return DEADLINE - (time.time() - START_TIME)

def _tensor_from_factors(U, V, W):
    """Compute tensor from factors. U:(16,R), V:(28,R), W:(28,R) -> T:(4,4,7)"""
    R = U.shape[1]
    U3 = U.reshape(4, 4, R)
    V3 = V.reshape(4, 7, R)
    W3 = W.reshape(7, 4, R)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return T

def _target():
    return np.ones((4, 4, 7), dtype=np.float64)

def _verify_exact(U, V, W, tol=1e-10):
    T = _tensor_from_factors(U, V, W)
    return np.allclose(T, _target(), atol=tol)

def _schoolbook():
    n, m, p = 4, 4, 7
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
    return {"U": U, "V": V, "W": W}

# Parametric family: each term c is parameterized by vectors
# u_c in R^16, v_c in R^28, w_c in R^28
# We use a structured form: u_c = a_c ⊗ b_c (2x2 outer product flattened)
# where a_c in R^4, b_c in R^4 -> u_c in R^16
# Similarly for v and w with appropriate dimensions.

# For U (16 = 4*4): u_c[i,j] = a_c[i] * b_c[j]
# For V (28 = 4*7): v_c[j,k] = c_c[j] * d_c[k]  
# For W (28 = 7*4): w_c[k,i] = e_c[k] * f_c[i]

# This gives rank-1 terms in the "bilinear" sense but not in tensor rank.
# Actually this is exactly a rank-1 tensor: (a⊗b)⊗(c⊗d)⊗(e⊗f)
# T[i,j,k] += a[i]*b[j]*c[j]*d[k]*e[k]*f[i]
# = (a[i]*f[i]) * (b[j]*c[j]) * (d[k]*e[k])
# This is a product of three vectors, i.e., a rank-1 tensor!
# So each term is rank-1, and we're summing rank-1 tensors.

# So the problem reduces to: find rank-1 tensors (u,v,w) such that
# sum_c u_c ⊗ v_c ⊗ w_c = T
# where T[i,j,k] = 1.

# We parameterize: u_c = a_c ⊙ f_c (element-wise, length 4) -- no wait
# u_c is length 16 = 4*4, indexed by (i,j). 
# v_c is length 28 = 4*7, indexed by (j,k).
# w_c is length 28 = 7*4, indexed by (k,i).

# For rank-1 tensor: u_c[i,j] * v_c[j,k] * w_c[k,i] summed over c should give 1.
# If u_c[i,j] = p_c[i] * q_c[j], v_c[j,k] = r_c[j] * s_c[k], w_c[k,i] = t_c[k] * u_c[i]
# Then product = p_c[i]*q_c[j]*r_c[j]*s_c[k]*t_c[k]*u_c[i]
# = (p_c[i]*u_c[i]) * (q_c[j]*r_c[j]) * (s_c[k]*t_c[k])
# This is separable: A_c[i] * B_c[j] * C_c[k]
# So sum_c A_c[i] B_c[j] C_c[k] = 1 for all i,j,k.

# This means we need: sum_c A_c[i] B_c[j] C_c[k] = 1
# This is equivalent to: for each (i,j,k), the sum is 1.
# If we set A_c[i] = alpha_c[i], B_c[j] = beta_c[j], C_c[k] = gamma_c[k]
# Then sum_c alpha_c[i] beta_c[j] gamma_c[k] = 1 for all i,j,k.

# This is a system where the "tensor" is the all-ones tensor.
# The all-ones tensor of shape (4,4,7) has rank 1! (it's 1⊗1⊗1)
# So we can do it with rank 1? No wait, that's not right.

# Let me re-examine. The matrix multiplication tensor is NOT the all-ones tensor.
# T[i,j,k] = 1 means: (AB)[i,k] = sum_j A[i,j] B[j,k]
# The tensor T[i,j,k] = 1 for all i,j,k IS the all-ones tensor.
# And the all-ones tensor has rank 1: it equals (1,1,1,1) ⊗ (1,1,1,1) ⊗ (1,1,1,1,1,1,1)
# So rank 1 would suffice?? That can't be right.

# I think I'm confusing the tensor representation. Let me re-read the contract.
# "U, V, W factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
# The decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
# And T[i,j,k] should equal 1 for all i,j,k (since every (i,j,k) term appears in the product).
# 
# So yes, T is the all-ones tensor, which has rank 1.
# But that contradicts the known result that 4x4x7 requires rank 57.
#
# I think the issue is that the tensor for matrix multiplication is NOT simply all-ones.
# Let me reconsider: the multiplication tensor has entries T[i,j,k] = 1, but the 
# "tensor" in the context of tensor rank for matrix multiplication is defined differently.
#
# Actually, I think I've been correct: the matrix multiplication tensor of size (n,m,p) 
# has T[i,j,k] = 1 for all valid indices, and its tensor rank is what we're minimizing.
# For (4,4,7), the known minimum is 57.
#
# But if T is all-ones, its rank is 1. So there must be a different convention.
#
# Let me re-read: "U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]"
# The product sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] should give the (i,k)-th entry of AB
# as a multilinear form in A and B.
# 
# (AB)[i,k] = sum_j A[i,j] B[j,k]
# As a multilinear form: this is linear in A[i,j] and B[j,k] for each j.
# The tensor T[i,j,k] = 1 means the coefficient of A[i,j]B[j,k] is 1.
#
# So T is indeed the all-ones tensor of shape (4,4,7), and its tensor rank should be 1.
# But that's not the standard definition. I think the standard matrix multiplication tensor
# has a different structure.
#
# After reflection: I believe the convention here is that T[i,j,k]=1 and the tensor rank
# of the all-ones tensor (n,m,p) is indeed 1. So the "rank" being minimized here must 
# refer to something else, OR the tensor is not all-ones.
#
# Given the confusion, let me just use the schoolbook decomposition which is guaranteed correct.
# The seed program uses rank 112 = 4*4*7, which is the standard schoolbook.
# I'll attempt to find a lower-rank decomposition through search, but fall back to schoolbook.

def _random_rank1_terms(rank, seed=None):
    """Generate rank-1 terms with random small rational entries."""
    rng = random.Random(seed) if seed else random
    U = np.zeros((16, rank))
    V = np.zeros((28, rank))
    W = np.zeros((28, rank))
    alphabet = [0, 1, -1, 0.5, -0.5, 2, -2]
    for c in range(rank):
        for i in range(16):
            U[i,c] = rng.choice(alphabet) if rng.random() < 0.5 else 0
        for i in range(28):
            V[i,c] = rng.choice(alphabet) if rng.random() < 0.5 else 0
        for i in range(28):
            W[i,c] = rng.choice(alphabet) if rng.random() < 0.5 else 0
    return U, V, W

def _greedy_reduce(U, V, W, max_iters=200):
    """Try to reduce error by flipping entries."""
    target = _target()
    T = _tensor_from_factors(U, V, W)
    err = np.sum((T - target)**2)
    U = U.copy(); V = V.copy(); W = W.copy()
    alphabet = [0, 1, -1, 0.5, -0.5, 2, -2]
    
    for it in range(max_iters):
        if _time_left() < 5:
            break
        mat_choice = rng_choice = random.random()
        if rng_choice < 0.33:
            mat, dim = U, 16
        elif rng_choice < 0.66:
            mat, dim = V, 28
        else:
            mat, dim = W, 28
        r = random.randint(0, dim-1)
        c = random.randint(0, mat.shape[1]-1)
        old = mat[r,c]
        new = random.choice(alphabet)
        if new == old:
            continue
        mat[r,c] = new
        new_T = _tensor_from_factors(mat if mat is U else (V if mat is V else U),
                                     V if mat is not V else (U if mat is U else V),
                                     W if mat is not W else (U if mat is U else V))
        # Simpler: just recompute
        if mat is U:
            new_T = _tensor_from_factors(U, V, W)
        elif mat is V:
            new_T = _tensor_from_factors(U, V, W)
        else:
            new_T = _tensor_from_factors(U, V, W)
        new_err = np.sum((new_T - target)**2)
        if new_err < err:
            err = new_err
        else:
            mat[r,c] = old
    return U, V, W, err

# EVOLVE-BLOCK-START
def construct():
    global START_TIME
    START_TIME = time.time()
    
    # Try to find a low-rank decomposition
    target = _target()
    best = None
    best_rank = 112
    
    for rank in [57, 60, 65, 70, 80, 90, 100]:
        if _time_left() < 20:
            break
        for restart in range(10):
            if _time_left() < 10:
                break
            U, V, W = _random_rank1_terms(rank, seed=rank*1000+restart)
            U, V, W, err = _greedy_reduce(U, V, W, max_iters=100)
            if err < 1e-8:
                if _verify_exact(U, V, W):
                    best = (U, V, W, rank)
                    break
        if best is not None:
            break
    
    if best is not None:
        U, V, W, rank = best
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
