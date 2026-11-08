import numpy as np
import time
import itertools
import sys

DIMENSIONS = [2, 3, 4]
SEED_RANK = 22

def construct():
    n, m, p = DIMENSIONS
    nm = n * m      # 6
    mp = m * p      # 12
    pn = p * n      # 8

    # Start with the known rank-22 solution (Strassen 7+7 + standard 8)
    # We'll try to reduce rank by checking if any term can be removed or
    # combined via linear dependency on the support.

    # Build the base rank-22 as sparse arrays
    # Term structure: (U_vec, V_vec, W_vec)
    # U has nm=6 components (A row-major)
    # V has mp=12 components (B row-major: B[i,j] at i*p+j)
    # W has pn=8 components (C column-major per trace convention: W[k*n+i] for C[i,k])

    def idx_a(i, j): return i * m + j
    def idx_b(i, j): return i * p + j
    def idx_c(i, j): return j * n + i  # trace convention W[k*n+i,r]

    terms = []  # list of (U_list, V_list, W_list) where each is [(idx, val), ...]

    def add(a_terms, b_terms, c_terms, scale=1):
        u = [(idx_a(i, j), v) for i, j, v in a_terms]
        v = [(idx_b(i, j), val) for i, j, val in b_terms]
        w = [(idx_c(i, j), val) for i, j, val in c_terms]
        terms.append((u, v, w, scale))

    # Strassen for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # M1 = (A00+A11)(B00+B11) -> C00 + C11
    add([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)])
    # M2 = (A10+A11)B00 -> C10 - C11
    add([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)])
    # M3 = A00(B01-B11) -> C01 + C11
    add([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)])
    # M4 = A11(B10-B00) -> C00 + C10
    add([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)])
    # M5 = (A00+A01)B11 -> -C00 + C01
    add([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)])
    # M6 = (A10-A00)(B00+B01) -> C11
    add([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)])
    # M7 = (A01-A11)(B10+B11) -> C00
    add([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)])

    # Strassen for A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    # B columns 2,3 -> B[0,2]=idx 2, B[0,3]=idx 3, B[1,2]=idx 6, B[1,3]=idx 7
    add([(0,0,1),(1,1,1)], [(0,2,1),(1,3,1)], [(0,2,1),(1,3,1)])
    add([(1,0,1),(1,1,1)], [(0,2,1)], [(1,2,1),(1,3,-1)])
    add([(0,0,1)], [(0,3,1),(1,3,-1)], [(0,3,1),(1,3,1)])
    add([(1,1,1)], [(1,2,1),(0,2,-1)], [(0,2,1),(1,2,1)])
    add([(0,0,1),(0,1,1)], [(1,3,1)], [(0,2,-1),(0,3,1)])
    add([(1,0,1),(0,0,-1)], [(0,2,1),(0,3,1)], [(1,3,1)])
    add([(0,1,1),(1,1,-1)], [(1,2,1),(1,3,1)], [(0,2,1)])

    # Standard for A[0:2,2] * B[2,0:4] -> C[0:2,0:4]
    # A[0,2]=idx2, A[1,2]=idx5; B[2,0]=idx8, B[2,1]=idx9, B[2,2]=idx10, B[2,3]=idx11
    for a_i in [0, 1]:
        for b_j in [0, 1, 2, 3]:
            add([(a_i, 2, 1)], [(2, b_j, 1)], [(a_i, b_j, 1)])

    R0 = len(terms)  # 22

    # Try to reduce rank by checking linear dependencies.
    # A valid decomposition means: for all A, B, sum_r U[:,r] (A^T) * (B^T) V[:,r] = C in trace form.
    # Equivalently, the tensor T[A_i, B_j, C_k] = sum_r U[i,r] V[j,r] W[k,r] must equal the matmul tensor.
    #
    # Rank reduction idea: if two rank-1 terms can be combined into one (their sum is also rank-1),
    # or if a term is in the linear span of others on the combined U⊗V⊗W space, we can reduce.
    #
    # Practically: build the tensor for each rank-1 term and check if any linear combination of all R0
    # terms can be represented with fewer terms. This is hard. Instead, use a simpler heuristic:
    # check if any single term's tensor is a scalar multiple of another, or if we can merge two terms
    # by verifying their sum has rank 1.

    def term_tensor(u, v, w):
        """Return the 6x12x8 tensor for one term (sparse)."""
        T = np.zeros((nm, mp, pn), dtype=float)
        for (ai, uval), (bi, bval), (ci, cval) in itertools.product(u, v, w):
            T[ai, bi, ci] += uval * bval * cval
        return T

    def is_rank1(T):
        """Check if a tensor is rank-1 (up to numerical error)."""
        # Reshape to 2D and check rank
        # A tensor is rank-1 iff all its slices have rank 1 and consistent
        # Simple check: outer product of three vectors
        # Find support
        nz = np.nonzero(T)
        if len(nz[0]) == 0:
            return True, None
        # Get unique indices
        ais = set(nz[0])
        bis = set(nz[1])
        cis = set(nz[2])
        if len(ais) == 0 or len(bis) == 0 or len(cis) == 0:
            return True, None
        # For rank 1, T[a,b,c] = u[a]*v[b]*w[c]
        # Pick a reference point
        a0 = list(ais)[0]
        b0 = list(bis)[0]
        c0 = list(cis)[0]
        T000 = T[a0, b0, c0]
        if abs(T000) < 1e-12:
            return False, None
        # Recover vectors up to scaling
        u_vec = np.zeros(nm)
        v_vec = np.zeros(mp)
        w_vec = np.zeros(pn)
        for a in ais:
            for b in bis:
                for c in cis:
                    if abs(T[a, b, c]) > 1e-12:
                        u_vec[a] = T[a, b, c] / T000
                        v_vec[b] = T[a, b, c] / T000
                        w_vec[c] = T[a, b, c] / T000
                        break
                break
            break
        # Verify
        recon = np.zeros_like(T)
        for a in ais:
            for b in bis:
                for c in cis:
                    recon[a, b, c] = u_vec[a] * v_vec[b] * w_vec[c]
        if np.allclose(T, recon, atol=1e-10):
            return True, (u_vec, v_vec, w_vec)
        return False, None

    # Build tensors for each term
    term_tensors = []
    for u, v, w, s in terms:
        T = term_tensor(u, v, w)
        term_tensors.append(T)

    # Try to merge pairs of terms if their sum is rank-1
    # This is O(R^2) = ~231 checks, fast
    merged = False
    active = [True] * R0
    for i in range(R0):
        if not active[i]:
            continue
        for j in range(i+1, R0):
            if not active[j]:
                continue
            Sum = term_tensors[i] + term_tensors[j]
            ok, _ = is_rank1(Sum)
            if ok:
                # Can we represent the sum as a single term with valid integer/half-integer entries?
                # For now, just mark them for potential merging.
                # We need to find U, V, W vectors for the sum.
                # Extract from Sum tensor.
                nz = np.nonzero(Sum)
                if len(nz[0]) == 0:
                    active[i] = False
                    active[j] = False
                    merged = True
                    break
                a0, b0, c0 = nz[0][0], nz[1][0], nz[2][0]
                s0 = Sum[a0, b0, c0]
                if abs(s0) < 1e-12:
                    continue
                u_new = np.zeros(nm)
                v_new = np.zeros(mp)
                w_new = np.zeros(pn)
                # Recover vectors
                for a, b, c in zip(nz[0], nz[1], nz[2]):
                    val = Sum[a, b, c]
                    if abs(val) < 1e-12:
                        continue
                    u_new[a] = val / s0
                    v_new[b] = val / s0
                    w_new[c] = val / s0
                # Check for integer/half-integer entries
                def is_half_integer(arr):
                    arr = arr.copy()
                    arr *= 2
                    return np.allclose(arr, np.round(arr), atol=1e-10)
                if is_half_integer(u_new) and is_half_integer(v_new) and is_half_integer(w_new):
                    # Normalize to half-integers
                    u_new *= 2
                    u_new = np.round(u_new).astype(int) / 2
                    v_new *= 2
                    v_new = np.round(v_new).astype(int) / 2
                    w_new *= 2
                    w_new = np.round(w_new).astype(int) / 2
                    # Build the new term
                    u_list = [(int(i), float(u_new[i])) for i in range(nm) if abs(u_new[i]) > 1e-10]
                    v_list = [(int(i), float(v_new[i])) for i in range(mp) if abs(v_new[i]) > 1e-10]
                    w_list = [(int(i), float(w_new[i])) for i in range(pn) if abs(w_new[i]) > 1e-10]
                    # Replace term i with the new one, deactivate j
                    terms[i] = (u_list, v_list, w_list, 1)
                    term_tensors[i] = Sum
                    active[j] = False
                    merged = True
                    break
        if merged:
            break

    # Build final U, V, W
    active_idx = [i for i in range(R0) if active[i]]
    R = len(active_idx)

    # For the active terms, build U, V, W matrices
    U = np.zeros((nm, R))
    V = np.zeros((mp, R))
    W = np.zeros((pn, R))

    for col, idx in enumerate(active_idx):
        u_list, v_list, w_list, s = terms[idx]
        for (ai, val) in u_list:
            U[ai, col] = val * s
        for (bi, val) in v_list:
            V[bi, col] = val * s
        for (ci, val) in w_list:
            W[ci, col] = val * s

    # Verify the decomposition
    # T_recon = sum_r U[:,r] (A^T) * (B^T) V[:,r] ... no, let's verify via tensor.
    T_recon = np.zeros((nm, mp, pn))
    for r in range(R):
        T_recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])

    # The true matmul tensor: T[a,b,c] where A is nxm, B is mxp, C is nxp
    # T[A(i,k), B(k,j), C(i,j)] = 1 for all valid i,k,j
    # A index: i*m + k (row-major)
    # B index: k*p + j (row-major)
    # C index: j*n + i (trace convention)
    T_true = np.zeros((nm, mp, pn))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a_idx = i * m + k
                b_idx = k * p + j
                c_idx = j * n + i
                T_true[a_idx, b_idx, c_idx] = 1

    if not np.allclose(T_recon, T_true, atol=1e-10):
        # Fall back to the original rank-22
        U = np.zeros((nm, R0))
        V = np.zeros((mp, R0))
        W = np.zeros((pn, R0))
        for r in range(R0):
            u_list, v_list, w_list, s = terms[r]
            for (ai, val) in u_list:
                U[ai, r] = val * s
            for (bi, val) in v_list:
                V[bi, r] = val * s
            for (ci, val) in w_list:
                W[ci, r] = val * s
        R = R0

    # Convert to lists of lists with integer/half-integer entries
    def to_half_int_list(M):
        """Convert matrix to list of lists, rounding to half-integers if close."""
        L = []
        for row in M:
            r = []
            for val in row:
                # Check if it's close to a half-integer
                doubled = val * 2
                nearest = np.round(doubled)
                if abs(doubled - nearest) < 1e-8:
                    r.append(int(nearest) / 2.0)
                else:
                    r.append(val)
            L.append(r)
        return L

    U_list = to_half_int_list(U)
    V_list = to_half_int_list(V)
    W_list = to_half_int_list(W)

    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
    print("U shape:", (len(res['U']), len(res['U'][0])))
    print("V shape:", (len(res['V']), len(res['V'][0])))
    print("W shape:", (len(res['W']), len(res['W'][0])))
