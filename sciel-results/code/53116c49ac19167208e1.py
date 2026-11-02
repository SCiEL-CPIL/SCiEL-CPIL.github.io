import numpy as np
import time
import itertools
from scipy import linalg

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 * 2x6 matrix multiplication.
    Searches for rank-20 decomposition via elimination from Strassen-21,
    with fallback to rank-21 Strassen.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit

    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12

    # Build the target tensor T[a,b,c]
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Strassen steps for 2x2 multiplication
    strassen_steps = [
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
        {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
        {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
        {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
    ]

    # Build Strassen-21 decomposition
    def build_strassen_21():
        rank = 21
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        for g in range(3):
            k0, k1 = 2*g, 2*g+1
            v_idx = [0*6+k0, 0*6+k1, 1*6+k0, 1*6+k1]
            w_idx = [k0*2+0, k1*2+0, k0*2+1, k1*2+1]
            for s in range(7):
                r = 7*g + s
                step = strassen_steps[s]
                for li in range(4):
                    if step['u'][li] != 0:
                        U[li, r] = step['u'][li]
                    if step['v'][li] != 0:
                        V[v_idx[li], r] = step['v'][li]
                    if step['w'][li] != 0:
                        W[w_idx[li], r] = step['w'][li]
        return U, V, W

    U21, V21, W21 = build_strassen_21()

    # Verify Strassen-21
    T_check = np.einsum('ar,br,cr->abc', U21, V21, W21)
    if not np.allclose(T_check, T, atol=1e-10):
        # Fallback to schoolbook
        rank_s = 24
        Us = np.zeros((nm, rank_s))
        Vs = np.zeros((mp, rank_s))
        Ws = np.zeros((pn, rank_s))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    Us[a, col] = 1
                    Vs[b, col] = 1
                    Ws[c, col] = 1
                    col += 1
        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

    # Try to find rank-20 by removing one product from Strassen-21
    # and checking if the residual is rank-1 (absorbable)
    def try_remove_product(removal_idx):
        """
        Try removing product at index removal_idx from the 21-product Strassen.
        Check if T - sum_{r != removal_idx} U[:,r] V[:,r] W[:,r] is rank-1.
        If so, we have a rank-20 decomposition.
        """
        # Compute residual
        residual = T.copy()
        for r in range(21):
            if r == removal_idx:
                continue
            residual -= np.einsum('a,b,c->abc', U21[:, r], V21[:, r], W21[:, r])

        # Check if residual is rank-1: all 2x2 minors of any matricization are zero
        # Matricize as (nm*mp) x pn
        R = residual.reshape(nm * mp, pn)
        # Rank-1 means all 2x2 minors vanish
        # Use SVD to check
        try:
            svals = np.linalg.svd(R, compute_uv=False)
        except:
            return None
        if len(svals) >= 2 and svals[1] > 1e-8:
            return None
        if len(svals) == 1:
            if svals[0] < 1e-12:
                # Residual is zero - we have rank 20 (one product was redundant)
                # But that means the 21st product contributed nothing
                pass
            # Residual is rank-1
            # Extract the rank-1 vectors
            s, U_mat, Vt = np.linalg.svd(R, compute_uv=True)
            if s[0] < 1e-12:
                # Zero residual
                pass
            u_vec = U_mat[:, 0] * s[0]
            w_vec = Vt[0, :]
            # Reshape u_vec to (nm, mp) for the U and V parts
            # Actually we need to decompose u_vec into U_part and V_part
            # u_vec = U_part * V_part (outer product)
            # u_vec is (nm*mp), we need U_part (nm) and V_part (mp) such that
            # u_vec = U_part ⊗ V_part
            # This is only possible if u_vec is itself rank-1 as a matrix
            U_mat_res = u_vec.reshape(nm, mp)
            # Check if U_mat_res is rank-1
            svals2 = np.linalg.svd(U_mat_res, compute_uv=False)
            if len(svals2) >= 2 and svals2[1] > 1e-8:
                return None
            if svals2[0] < 1e-12:
                # Zero - the removed product was actually needed
                return None
            # Extract
            s2, U2_mat, V2t = np.linalg.svd(U_mat_res, compute_uv=True)
            u_part = U2_mat[:, 0] * s2[0]
            v_part = V2t[0, :]
            w_part = w_vec

            # Build rank-20 decomposition
            U20 = np.zeros((nm, 20))
            V20 = np.zeros((mp, 20))
            W20 = np.zeros((pn, 20))
            col = 0
            for r in range(21):
                if r == removal_idx:
                    continue
                U20[:, col] = U21[:, r]
                V20[:, col] = V21[:, r]
                W20[:, col] = W21[:, r]
                col += 1
            # Add the residual rank-1 product
            U20[:, col] = u_part
            V20[:, col] = v_part
            W20[:, col] = w_part

            # Verify
            T20 = np.einsum('ar,br,cr->abc', U20, V20, W20)
            if np.allclose(T20, T, atol=1e-8):
                return (U20, V20, W20)
            return None
        elif len(svals) == 0:
            return None
        return None

    # Try removing each of the 21 products
    found_rank20 = None
    for idx in range(21):
        if time.time() > deadline - 30:
            break
        result = try_remove_product(idx)
        if result is not None:
            found_rank20 = result
            break

    if found_rank20 is not None:
        U20, V20, W20 = found_rank20
        return {"U": U20.tolist(), "V": V20.tolist(), "W": W20.tolist()}

    # If no rank-20 found via simple removal, try a more general search:
    # Randomized search for rank-20 by fixing random U,V and solving for W
    # This is a different structural approach
    if time.time() < deadline - 20:
        # Try randomized rank-20 search
        rng = np.random.RandomState(42)
        best_rank20 = None

        for attempt in range(50):
            if time.time() > deadline - 20:
                break
            # Start from Strassen-21, randomly perturb and try to eliminate
            # Alternative: use the fact that rank-20 means the 20 products span T
            # Try: pick 20 random directions in the space, solve for coefficients
            # Actually, let's try a different approach:
            # Fix U and V for 20 products, solve for W
            # Use the 21 Strassen products, randomly drop one, and try to
            # adjust the remaining 20 to span T

            # Simpler: try all 21 removals with small perturbations
            # of the remaining products to see if rank-20 is achievable
            pass

    # Return Strassen-21 as the best we have
    return {"U": U21.tolist(), "V": V21.tolist(), "W": W21.tolist()}
