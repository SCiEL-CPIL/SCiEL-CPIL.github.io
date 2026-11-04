"""Bootstrap spawn: model-and-optimize for 4x5 * 5x6 matrix multiplication tensor rank."""
import numpy as np
import time
import random

DIMENSIONS = [4, 5, 6]
DEADLINE_SECONDS = 145

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _build_target_tensors(n, m, p):
    """Build target tensors for the matrix multiplication tensor.
    T_pos[a, b, g] = 1 if (a,b,g) is consistent, else 0.
    We store as compact arrays indexed by (a, b, g) flattened.
    """
    nm, mp, pn = n * m, m * p, p * n
    # For each a, which (b,g) pairs are consistent?
    # a = i*m+j, b = j'*p+k, g = k''*n+i'
    # consistent iff i==i', j==j', k==k''
    # So for fixed a=(i,j): b must have j'=j, so b = j*p+k for k in range(p)
    #                       g must have i'=i, so g = k''*n+i for k'' in range(p)
    #                       and k'' must equal k (from b)
    # So for fixed a: consistent (b,g) pairs are (j*p+k, k*n+i) for k in range(p)
    
    # Precompute for each a: list of (b, g) that are consistent
    consistent_by_a = []
    for a in range(nm):
        i = a // m
        j = a % m
        pairs = []
        for k in range(p):
            b = j * p + k
            g = k * n + i
            pairs.append((b, g))
        consistent_by_a.append(pairs)
    
    # For each b: list of (a, g) that are consistent
    consistent_by_b = []
    for b in range(mp):
        jp = b // p
        k = b % p
        pairs = []
        for i in range(n):
            a = i * m + jp
            g = k * n + i
            pairs.append((a, g))
        consistent_by_b.append(pairs)
    
    # For each g: list of (a, b) that are consistent
    consistent_by_g = []
    for g in range(pn):
        kn = g // n
        ip = g % n
        pairs = []
        for j in range(m):
            a = ip * m + j
            b = j * p + kn
            pairs.append((a, b))
        consistent_by_g.append(pairs)
    
    return consistent_by_a, consistent_by_b, consistent_by_g

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the tensor decomposition exactly reconstructs matrix multiplication."""
    rank = U.shape[1]
    # The tensor T[i, j, k, a, b] = delta_{i,a} * delta_{j,b} for A[i,j] * B[j,k]
    # Actually for matrix multiplication C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = A[i,j]*B[j,k] which decomposes as sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]?
    # Wait, let me re-read the task.
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # The decomposition is: (A ⊗ B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor form: T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And T_{i,j,k} should equal 1 if we're decomposing the "multiplication" operation
    # Actually the standard tensor for matrix mult: T(i,j,k) = 1 means the coefficient
    # The multiplication tensor has T_{i,j,k} = 1 for all i,j,k? No.
    # 
    # Let me think more carefully. Matrix multiplication C = AB where A is n x m, B is m x p.
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # The "matrix multiplication tensor" T has entries T[i, j, k] such that
    # C[i,k] = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k]
    # So T[i,j,k] = 1 if k matches... wait no.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all valid i,j,k? No, that's not right either.
    # 
    # Actually T[i,j,k] should be such that sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = C[i',k']
    # This means T[i,j,k] contributes to C[i,k] only when the indices match.
    # T[i,j,k] = 1 for all (i,j,k) in range(n) x range(m) x range(p)
    # And C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]... but that would give sum over all i,j,k.
    # 
    # Let me use the standard formulation. The matrix multiplication tensor of size (n,m,p)
    # is defined by: for each r, the rank-1 term is u_r ⊗ v_r ⊗ w_r where:
    # u_r is indexed by (i,j), v_r by (j,k), w_r by (k,i) [per the W[k*n+i] convention]
    # And the condition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta for the mult tensor.
    #
    # The matrix multiplication "tensor" T has T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    # Wait no. Let me look at the schoolbook construction:
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1 for each (i,j,k).
    # So sum over col: U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 if there's a term for exactly (i,j,k), 0 otherwise.
    # So the reconstructed tensor T'[i,j,k] = 1 for all (i,j,k). That's the full tensor of ones.
    # 
    # So the "matrix multiplication tensor" here is just the all-ones tensor of size n x m x p.
    # We need to find a rank-r decomposition of the all-ones tensor T[i,j,k] = 1.
    #
    # Wait, that doesn't make sense for "matrix multiplication". Let me re-read.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 4 by 5 matrix by a 5 by 6 matrix."
    #
    # I think the standard matrix multiplication tensor T_{n,m,p} is defined as:
    # T[i, j, k] = 1 for all i,j,k, and the multiplication is:
    # C[i,k] = sum_{j} A[i,j] B[j,k]
    # which can be written as C = T(A, B) where T is a bilinear (actually trilinear if we consider output) map.
    #
    # In the tensor network / algebraic complexity sense, the matrix multiplication tensor M_{n,m,p} has:
    # M_{n,m,p} = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i ⊗ e_j ⊗ e_k
    # where e_i is the i-th standard basis vector.
    # And a rank-r decomposition means: M = sum_{r=1}^r u_r ⊗ v_r ⊗ w_r
    # where u_r in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n} (with the flattening conventions).
    #
    # So yes, T[i,j,k] = 1 for all valid indices, and we need rank-r decomposition of the all-ones tensor.
    #
    # The rank of the all-ones tensor of size n x m x p... that's an interesting question.
    # For the 2x2x2 case, the rank is 7 (Strassen).
    # For general all-ones tensor of size n x m x p, the rank is at most n*m*p (schoolbook).
    # Can we do better?
    #
    # Actually wait - is the matrix multiplication tensor really the all-ones tensor? Let me reconsider.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # If we think of A as a vector in R^{n*m} (flattened) and B as a vector in R^{m*p} (flattened),
    # then C is in R^{n*p} (flattened). The multiplication is a bilinear map R^{n*m} x R^{m*p} -> R^{n*p}.
    # The tensor for this bilinear map is: T[i*m+j, j*p+k, k*n+i'] = delta_{i,i'} ... hmm this is getting confusing.
    #
    # Let me just use the schoolbook as ground truth. In the schoolbook:
    # For each (i,j,k): U[i*m+j, (i,j,k)] = 1, V[j*p+k, (i,j,k)] = 1, W[k*n+i, (i,j,k)] = 1
    # The product for fixed (i',j',k'): sum_r U[i'*m+j', r] V[j'*p+k', r] W[k'*n+i', r]
    # = 1 if (i',j',k') = (i,j,k) for some term, else 0.
    # So the reconstructed tensor has 1s everywhere and 0s nowhere - it's the all-ones tensor.
    #
    # So we need: minimize rank r such that there exist U (n*m x r), V (m*p x r), W (p*n x r)
    # with sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    #
    # This is the tensor rank of the all-ones tensor J_{n,m,p}.
    #
    # For n=m=p=2, rank(J_{2,2,2}) = 7 (Strassen's algorithm).
    # For n=m=p=3, the best known is... around 23? 
    # For 4x5x6, we need to figure out what's achievable.
    
    # Check: for fixed (i,j,k), is the value 1?
    total = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Start with schoolbook as baseline
    best_U, best_V, best_W = _build_schoolbook(n, m, p)
    best_rank = best_U.shape[1]

    # Precompute consistent triple structures
    consistent_by_a, consistent_by_b, consistent_by_g = _build_target_tensors(n, m, p)

    def alternating_minimization(r, max_iters=300, seed=42):
        """Alternating least squares for tensor decomposition."""
        if time.time() > deadline:
            return None

        rng = np.random.RandomState(seed)

        # Initialize
        U = rng.randn(nm, r) * 0.5
        V = rng.randn(mp, r) * 0.5
        W = rng.randn(pn, r) * 0.5

        # Precompute zero constraint samples for efficiency
        # We'll check a random subset of zero constraints
        zero_check_size = min(200, nm * mp * pn - n * m * p)
        zero_indices = rng.choice(nm * mp * pn, size=zero_check_size, replace=False)

        for it in range(max_iters):
            if time.time() > deadline:
                return None

            # Update U: for each a, solve least squares
            # For fixed V, W: sum_r U[a,r] * V[b,r] * W[g,r] = T[a,b,g]
            # For fixed a: define for each (b,g): c_{b,g,r} = V[b,r] * W[g,r]
            # Then: sum_r U[a,r] * c_{b,g,r} = T[a,b,g]
            # This is a linear system: M_a @ U[a,:] = t_a
            # where M_a has rows indexed by (b,g) and columns by r
            # M_a[b*pn+g, r] = V[b,r] * W[g,r]
            # t_a[b*pn+g] = 1 if (a,b,g) consistent else 0

            for a in range(nm):
                # Build M_a efficiently
                # M_a[b*pn+g, r] = V[b,r] * W[g,r]
                # We can build this as: for each b, V[b,:] outer with W[g,:] for each g
                M_mat = np.empty((mp * pn, r))
                for b in range(mp):
                    for g in range(pn):
                        M_mat[b * pn + g, :] = V[b, :] * W[g, :]

                # Target vector
                t_vec = np.zeros(mp * pn)
                for (b, g) in consistent_by_a[a]:
                    t_vec[b * pn + g] = 1.0

                try:
                    sol, _, _, _ = np.linalg.lstsq(M_mat, t_vec, rcond=None)
                    U[a, :] = sol
                except Exception:
                    pass

            # Update V: for each b, solve least squares
            for b in range(mp):
                M_mat = np.empty((nm * pn, r))
                for a in range(nm):
                    for g in range(pn):
                        M_mat[a * pn + g, :] = U[a, :] * W[g, :]

                t_vec = np.zeros(nm * pn)
                for (a, g) in consistent_by_b[b]:
                    t_vec[a * pn + g] = 1.0

                try:
                    sol, _, _, _ = np.linalg.lstsq(M_mat, t_vec, rcond=None)
                    V[b, :] = sol
                except Exception:
                    pass

            # Update W: for each g, solve least squares
            for g in range(pn):
                M_mat = np.empty((nm * mp, r))
                for a in range(nm):
                    for b in range(mp):
                        M_mat[a * mp + b, :] = U[a, :] * V[b, :]

                t_vec = np.zeros(nm * mp)
                for (a, b) in consistent_by_g[g]:
                    t_vec[a * mp + b] = 1.0

                try:
                    sol, _, _, _ = np.linalg.lstsq(M_mat, t_vec, rcond=None)
                    W[g, :] = sol
                except Exception:
                    pass

            # Check convergence every 5 iterations
            if it % 5 == 4:
                # Check positive constraints
                pos_loss = 0.0
                for a in range(nm):
                    for (b, g) in consistent_by_a[a]:
                        val = np.dot(U[a, :], V[b, :] * W[g, :])
                        pos_loss += (val - 1.0) ** 2

                # Check sampled zero constraints
                zero_loss = 0.0
                for idx in zero_indices:
                    a = idx // (mp * pn)
                    rem = idx % (mp * pn)
                    b = rem // pn
                    g = rem % pn
                    i = a // m
                    j = a % m
                    jp = b // p
                    k = b % p
                    kn = g // n
                    ip = g % n
                    if i != ip or j != jp or k != kn:
                        val = np.dot(U[a, :], V[b, :] * W[g, :])
                        zero_loss += val ** 2

                total_loss = pos_loss + zero_loss
                if total_loss < 1e-12:
                    # Full verification
                    full_loss = 0.0
                    for a in range(nm):
                        for b in range(mp):
                            for g in range(pn):
                                i = a // m
                                j = a % m
                                jp = b // p
                                k = b % p
                                kn = g // n
                                ip = g % n
                                if i == ip and j == jp and k == kn:
                                    val = np.dot(U[a, :], V[b, :] * W[g, :])
                                    full_loss += (val - 1.0) ** 2
                                else:
                                    val = np.dot(U[a, :], V[b, :] * W[g, :])
                                    full_loss += val ** 2
                    if full_loss < 1e-10:
                        return U, V, W

        return None

    # Try ranks from just below schoolbook down
    # Start with a few ranks close to 120 and work down
    candidate_ranks = list(range(min(best_rank - 1, 119), max(20, best_rank - 30), -1))
    # Also try some lower ranks
    candidate_ranks.extend([80, 70, 60, 50, 40, 35, 30, 25])
    # Remove duplicates and sort descending
    candidate_ranks = sorted(set(candidate_ranks), reverse=True)

    for r in candidate_ranks:
        if time.time() > deadline - 5:
            break
        if r >= best_rank:
            continue

        # Try multiple seeds for each rank
        for seed in [42, 123, 777, 2024]:
            if time.time() > deadline - 5:
                break
            result = alternating_minimization(r, max_iters=100, seed=seed)
            if result is not None:
                U, V, W = result
                # Verify
                if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
                    best_U, best_V, best_W = U, V, W
                    best_rank = r
                    break  # This rank works, try lower

    # Final check: if we found something better, return it
    if best_rank < n * m * p:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
