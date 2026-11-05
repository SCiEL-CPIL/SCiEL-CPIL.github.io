import numpy as np
import time
from itertools import combinations

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    def build_strassen(i1, i2, j1, j2, k1, k2):
        """Return list of 7 factor dicts for Strassen 2x2x2 block."""
        factors = []
        factors.append({'U': [(i1,j1,1),(i2,j2,1)], 'V': [(j1,k1,1),(j2,k2,1)], 'W': [(k1,i1,1),(k2,i2,1)]})
        factors.append({'U': [(i2,j1,1),(i2,j2,1)], 'V': [(j1,k1,1)], 'W': [(k1,i2,1),(k2,i2,-1)]})
        factors.append({'U': [(i1,j1,1)], 'V': [(j1,k2,1),(j2,k2,-1)], 'W': [(k2,i1,1),(k2,i2,1)]})
        factors.append({'U': [(i2,j2,1)], 'V': [(j2,k1,1),(j1,k1,-1)], 'W': [(k1,i1,1),(k1,i2,1)]})
        factors.append({'U': [(i1,j1,1),(i1,j2,1)], 'V': [(j2,k2,1)], 'W': [(k1,i1,-1),(k2,i1,1)]})
        factors.append({'U': [(i2,j1,1),(i1,j1,-1)], 'V': [(j1,k1,1),(j1,k2,1)], 'W': [(k2,i2,1)]})
        factors.append({'U': [(i1,j2,1),(i2,j2,-1)], 'V': [(j2,k1,1),(j2,k2,1)], 'W': [(k1,i1,1)]})
        return factors

    def build_outer(i_list, j, k_list):
        factors = []
        for i in i_list:
            for k in k_list:
                factors.append({'U': [(i,j,1)], 'V': [(j,k,1)], 'W': [(k,i,1)]})
        return factors

    def assemble(factors):
        rank = len(factors)
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for c, f in enumerate(factors):
            for (i, j, val) in f['U']:
                U[i*m+j, c] = val
            for (j, k, val) in f['V']:
                V[j*p+k, c] = val
            for (k, i, val) in f['W']:
                W[k*n+i, c] = val
        return U, V, W, rank

    def verify(U, V, W, rank):
        res = np.zeros((n, m, p))
        for c in range(rank):
            U_c = U[:, c].reshape(n, m)
            V_c = V[:, c].reshape(m, p)
            W_c = W[:, c].reshape(p, n)
            res += U_c[:, :, np.newaxis] * V_c[np.newaxis, :, :] * W_c.T[:, np.newaxis, :]
        return np.allclose(res, 1.0, atol=1e-9)

    def factor_tensor(U, V, W, c):
        """Return the 3D tensor for factor c."""
        U_c = U[:, c].reshape(n, m)
        V_c = V[:, c].reshape(m, p)
        W_c = W[:, c].reshape(p, n)
        return U_c[:, :, np.newaxis] * V_c[np.newaxis, :, :] * W_c.T[:, np.newaxis, :]

    def is_rank1_tensor(T, tol=1e-10):
        """Check if a 3D tensor is rank-1 (or zero)."""
        # Flatten and check if it's a rank-1 matrix
        # A rank-1 tensor T[i,j,k] = a[i]*b[j]*c[k]
        # Check: all 2x2 minors of any matrix unfolding are zero
        # Use SVD on matricization
        # T as (n*m) x p matrix
        T_mat = T.reshape(n*m, p)
        sv = np.linalg.svd(T_mat, compute_uv=False)
        if len(sv) == 0 or sv[0] < tol:
            return True
        if len(sv) > 1 and sv[1] > tol * sv[0]:
            return False
        # Also check another matricization to be safe
        T_mat2 = T.reshape(n, m*p)
        sv2 = np.linalg.svd(T_mat2, compute_uv=False)
        if len(sv2) == 0 or sv2[0] < tol:
            return True
        if len(sv2) > 1 and sv2[1] > tol * sv2[0]:
            return False
        return True

    def try_merge_factors(U, V, W, rank):
        """Try to merge pairs of factors whose sum is rank-1. Returns new (U,V,W,rank) or None."""
        if time.time() > deadline - 5:
            return None
        
        changed = True
        while changed:
            changed = False
            new_factors = list(range(rank))
            merged = False
            for a, b in combinations(range(rank), 2):
                if a >= rank or b >= rank:
                    continue
                if a not in new_factors or b not in new_factors:
                    continue
                # Check if factor a + factor b is rank-1
                T_a = factor_tensor(U, V, W, a)
                T_b = factor_tensor(U, V, W, b)
                T_sum = T_a + T_b
                if is_rank1_tensor(T_sum):
                    # Merge: replace a with the sum, remove b
                    U[:, a] = U[:, a] + U[:, b]
                    V[:, a] = V[:, a] + V[:, b]
                    W[:, a] = W[:, a] + W[:, b]
                    new_factors.remove(b)
                    rank -= 1
                    merged = True
                    changed = True
                    break
            if merged:
                # Rebuild arrays with new rank
                U = U[:, new_factors]
                V = V[:, new_factors]
                W = W[:, new_factors]
                rank = len(new_factors)
                # Continue to try more merges
            else:
                break
        return U, V, W, rank

    def try_remove_zero_factors(U, V, W, rank):
        """Remove factors that are all-zero."""
        new_factors = []
        for c in range(rank):
            if np.any(U[:, c] != 0) or np.any(V[:, c] != 0) or np.any(W[:, c] != 0):
                new_factors.append(c)
        if len(new_factors) < rank:
            U = U[:, new_factors]
            V = V[:, new_factors]
            W = W[:, new_factors]
            rank = len(new_factors)
        return U, V, W, rank

    def try_remove_proportional(U, V, W, rank):
        """If factor b is proportional to factor a, merge them."""
        if time.time() > deadline - 5:
            return U, V, W, rank
        changed = True
        while changed:
            changed = False
            new_factors = list(range(rank))
            for a, b in combinations(range(rank), 2):
                if a not in new_factors or b not in new_factors:
                    continue
                # Check if V_b is proportional to V_a
                Va = V[:, a]
                Vb = V[:, b]
                if np.allclose(Va, 0, atol=1e-12) or np.allclose(Vb, 0, atol=1e-12):
                    continue
                # Find ratio
                nonzero = np.abs(Va) > 1e-12
                if not np.any(nonzero):
                    continue
                ratio = Vb[nonzero] / Va[nonzero]
                if not np.allclose(ratio, ratio[0], atol=1e-10, rtol=1e-10):
                    continue
                r = ratio[0]
                # Check U and W are also proportional
                if not np.allclose(U[:, b], r * U[:, a], atol=1e-10, rtol=1e-10):
                    continue
                if not np.allclose(W[:, b], r * W[:, a], atol=1e-10, rtol=1e-10):
                    continue
                # Merge: add b to a, remove b
                U[:, a] += U[:, b]
                V[:, a] += V[:, b]
                W[:, a] += W[:, b]
                new_factors.remove(b)
                rank -= 1
                changed = True
                break
            if changed:
                U = U[:, new_factors]
                V = V[:, new_factors]
                W = W[:, new_factors]
                rank = len(new_factors)
        return U, V, W, rank

    # Primary construction: two Strassen blocks + outer products (rank 28)
    factors = []
    factors += build_strassen(0, 1, 0, 1, 0, 1)
    factors += build_outer([0, 1], 2, [0, 1])
    factors += build_strassen(0, 1, 0, 1, 2, 3)
    factors += build_outer([0, 1], 0, [4])
    factors += build_outer([0, 1], 1, [4])
    factors += build_outer([0, 1], 2, [2, 3, 4])

    U, V, W, rank = assemble(factors)

    # Refinement: try to reduce rank
    U, V, W, rank = try_remove_zero_factors(U, V, W, rank)
    U, V, W, rank = try_remove_proportional(U, V, W, rank)
    
    if time.time() < deadline - 10:
        merged = try_merge_factors(U, V, W, rank)
        if merged is not None:
            U, V, W, rank = merged
            U, V, W, rank = try_remove_zero_factors(U, V, W, rank)
            U, V, W, rank = try_remove_proportional(U, V, W, rank)

    if not verify(U, V, W, rank):
        # Fallback to schoolbook (rank 30)
        fb = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    fb.append({'U': [(i,j,1)], 'V': [(j,k,1)], 'W': [(k,i,1)]})
        U, V, W, rank = assemble(fb)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
