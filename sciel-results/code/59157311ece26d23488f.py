"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # safety margin

    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    best_rank = rank_sb
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}

    # Build the multiplication tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U, V, W):
        """Check if decomposition exactly matches the tensor."""
        T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
        return np.allclose(T_recon, T, atol=1e-9)

    def try_rank_am(r, seed=42, max_iter=300):
        """Alternating minimization for rank r."""
        if time.time() > deadline - 5:
            return None
        np.random.seed(seed)
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1

        for it in range(max_iter):
            if time.time() > deadline - 5:
                return None

            # Update U
            TW = V[:, None, :] * W[None, :, :]
            h = np.einsum('abc,bcd->ad', T, TW)
            G = np.einsum('bcd,bce->de', TW, TW)
            try:
                U = np.linalg.solve(G, h.T).T
            except np.linalg.LinAlgError:
                pass

            # Update V
            UW = U[:, None, :] * W[None, :, :]
            h_v = np.einsum('abc,acd->bd', T, UW)
            G_v = np.einsum('acd,ace->de', UW, UW)
            try:
                V = np.linalg.solve(G_v, h_v.T).T
            except np.linalg.LinAlgError:
                pass

            # Update W
            UV = U[:, None, :] * V[None, :, :]
            h_w = np.einsum('abc,abd->cd', T, UV)
            G_w = np.einsum('abd,abe->de', UV, UV)
            try:
                W = np.linalg.solve(G_w, h_w.T).T
            except np.linalg.LinAlgError:
                pass

        # Round to half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if verify(U_r, V_r, W_r):
            return (r, U_r, V_r, W_r)

        # Try integers
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if verify(U_i, V_i, W_i):
            return (r, U_i, V_i, W_i)

        return None

    def structured_strassen_ansatz(extra_rank=4):
        """
        Embed Strassen's rank-13 decomposition for the 3x3x3 core,
        then add extra_rank terms for the 2 extra output columns.
        Solve for the extra coefficients via least squares.
        """
        # Strassen's 7 terms for 3x3x3 (standard basis)
        # Each term: (U_matrix, V_matrix, W_matrix) as 3x3
        # Flattened to 9-vectors with row-major order
        # U acts on A (3x3), V on B (3x3), W on C (3x3)
        # C = A * B, so T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # where a=i*3+j, b=j*3+k, c=k*3+i for the 3x3 case

        # Strassen's 7 products for 2x2x2, extended to 3x3x3
        # Actually, for 3x3x3, the known minimal rank is 19 (Strassen) or lower?
        # Let me use a known rank-19 decomposition or try to construct one.
        
        # For simplicity, let's use the schoolbook for the 3x3 core (rank 9)
        # and add extra terms for the 2 extra columns.
        # Total rank would be 9 + extra_rank.
        
        # But we can do better: use a partial Strassen-like decomposition.
        
        # Let's try: rank 9 for 3x3x3 core (schoolbook) + extra_rank for extras
        # The 3x3x3 core uses indices k=0,1,2 (first 3 columns of B, first 3 rows of C)
        # The extra columns are k=3,4 (last 2 columns of B, last 2 rows of C)
        
        # Schoolbook for 3x3x3 core:
        U_core = np.zeros((n * m, 9))
        V_core = np.zeros((m * p, 9))
        W_core = np.zeros((p * n, 9))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(3):  # only first 3 columns
                    U_core[i * m + j, col] = 1
                    V_core[j * p + k, col] = 1
                    W_core[k * n + i, col] = 1
                    col += 1
        
        # Now we need to handle the extra columns k=3,4
        # For k=3,4: we need U[i*m+j, :], V[j*p+k, :], W[k*n+i, :]
        # such that the full tensor matches.
        
        # The tensor entries for k=3,4 are:
        # T[i*m+j, j*p+3, 3*n+i] = 1
        # T[i*m+j, j*p+4, 4*n+i] = 1
        # And all other entries involving k=3,4 should be 0.
        
        # Let's add extra_rank terms and solve for them.
        # The extra terms only need to contribute to the k=3,4 parts.
        
        # Set up the system: for each extra term r (r=0..extra_rank-1),
        # we have U_extra[a, r], V_extra[b, r], W_extra[c, r]
        # We want: sum_r U_extra[a,r]*V_extra[b,r]*W_extra[c,r] = T_extra[a,b,c]
        # where T_extra is the tensor with only k=3,4 entries nonzero.
        
        T_extra = np.zeros_like(T)
        for i in range(n):
            for j in range(m):
                for k in range(3, p):
                    T_extra[i * m + j, j * p + k, k * n + i] = 1.0
        
        # This is still a nonlinear problem. Let's use alternating minimization
        # on just the extra part, with the core fixed.
        
        if time.time() > deadline - 5:
            return None
        
        np.random.seed(123)
        U_extra = np.random.randn(n * m, extra_rank) * 0.1
        V_extra = np.random.randn(m * p, extra_rank) * 0.1
        W_extra = np.random.randn(p * n, extra_rank) * 0.1
        
        # But we want the extra terms to only affect k=3,4.
        # We can constrain V_extra and W_extra to be zero for k=0,1,2.
        # V_extra[j*p+k, r] = 0 for k=0,1,2
        # W_extra[k*n+i, r] = 0 for k=0,1,2
        
        for k in range(3):
            for j in range(m):
                V_extra[j * p + k, :] = 0
            for i in range(n):
                W_extra[k * n + i, :] = 0
        
        # Alternating minimization on T_extra
        for it in range(200):
            if time.time() > deadline - 5:
                return None
            
            # Update U_extra
            TW = V_extra[:, None, :] * W_extra[None, :, :]
            h = np.einsum('abc,bcd->ad', T_extra, TW)
            G = np.einsum('bcd,bce->de', TW, TW)
            try:
                U_extra = np.linalg.solve(G, h.T).T
            except np.linalg.LinAlgError:
                pass
            
            # Update V_extra (with constraint: zero for k=0,1,2)
            UW = U_extra[:, None, :] * W_extra[None, :, :]
            h_v = np.einsum('abc,acd->bd', T_extra, UW)
            G_v = np.einsum('acd,ace->de', UW, UW)
            try:
                V_extra = np.linalg.solve(G_v, h_v.T).T
            except np.linalg.LinAlgError:
                pass
            for k in range(3):
                for j in range(m):
                    V_extra[j * p + k, :] = 0
            
            # Update W_extra (with constraint: zero for k=0,1,2)
            UV = U_extra[:, None, :] * V_extra[None, :, :]
            h_w = np.einsum('abc,abd->cd', T_extra, UV)
            G_w = np.einsum('abd,abe->de', UV, UV)
            try:
                W_extra = np.linalg.solve(G_w, h_w.T).T
            except np.linalg.LinAlgError:
                pass
            for k in range(3):
                for i in range(n):
                    W_extra[k * n + i, :] = 0
        
        # Round to half-integers
        U_extra_r = np.round(U_extra * 2) / 2
        V_extra_r = np.round(V_extra * 2) / 2
        W_extra_r = np.round(W_extra * 2) / 2
        
        # Combine
        U_full = np.hstack([U_core, U_extra_r])
        V_full = np.hstack([V_core, V_extra_r])
        W_full = np.hstack([W_core, W_extra_r])
        
        if verify(U_full, V_full, W_full):
            total_rank = 9 + extra_rank
            return (total_rank, U_full, V_full, W_full)
        
        # Try integers
        U_extra_i = np.round(U_extra)
        V_extra_i = np.round(V_extra)
        W_extra_i = np.round(W_extra)
        U_full_i = np.hstack([U_core, U_extra_i])
        V_full_i = np.hstack([V_core, V_extra_i])
        W_full_i = np.hstack([W_core, W_extra_i])
        if verify(U_full_i, V_full_i, W_full_i):
            total_rank = 9 + extra_rank
            return (total_rank, U_full_i, V_full_i, W_full_i)
        
        return None

    # First, try the structured ansatz with various extra_rank values
    for extra_rank in [4, 5, 6, 7, 8]:
        if time.time() > deadline - 5:
            break
        total_rank = 9 + extra_rank
        if total_rank >= best_rank:
            break
        result = structured_strassen_ansatz(extra_rank)
        if result is not None:
            r_found, U_f, V_f, W_f = result
            if r_found < best_rank:
                best_rank = r_found
                U_list = [[float(x) for x in row] for row in U_f]
                V_list = [[float(x) for x in row] for row in V_f]
                W_list = [[float(x) for x in row] for row in W_f]
                best_result = {"U": U_list, "V": V_list, "W": W_list}
            break

    # Then, try full alternating minimization for lower ranks
    ranks_to_try = list(range(15, best_rank + 1))
    
    for r in ranks_to_try:
        if time.time() > deadline - 5:
            break
        if r >= best_rank:
            break
        
        for seed in [42, 0, 1, 2, 3, 7, 13, 99, 123, 456]:
            if time.time() > deadline - 5:
                break
            result = try_rank_am(r, seed)
            if result is not None:
                r_found, U_f, V_f, W_f = result
                if r_found < best_rank:
                    best_rank = r_found
                    U_list = [[float(x) for x in row] for row in U_f]
                    V_list = [[float(x) for x in row] for row in V_f]
                    W_list = [[float(x) for x in row] for row in W_f]
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                break

    return best_result
# EVOLVE-BLOCK-END
