import numpy as np
import time
import sys

def construct():
    n, m, p = 3, 4, 6
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # Build the multiplication tensor T
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def verify(U, V, W, atol=1e-9):
        T_comp = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_comp, T, atol=atol)

    def round_half(U, V, W):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        return U_r, V_r, W_r

    def round_int(U, V, W):
        U_r = np.round(U)
        V_r = np.round(V)
        W_r = np.round(W)
        return U_r, V_r, W_r

    def try_round_verify(U, V, W):
        Ui, Vi, Wi = round_int(U, V, W)
        if verify(Ui, Vi, Wi):
            return Ui, Vi, Wi
        Uh, Vh, Wh = round_half(U, V, W)
        if verify(Uh, Vh, Wh):
            return Uh, Vh, Wh
        return None

    # Schoolbook decomposition (baseline)
    rank_sch = n * m * p
    U_sch = np.zeros((n*m, rank_sch))
    V_sch = np.zeros((m*p, rank_sch))
    W_sch = np.zeros((p*n, rank_sch))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i*m*p + j*p + k
                U_sch[i*m+j, r] = 1.0
                V_sch[j*p+k, r] = 1.0
                W_sch[k*n+i, r] = 1.0

    best_U, best_V, best_W = U_sch, V_sch, W_sch
    best_rank = rank_sch

    # Matricizations for ALS
    T_A = T.reshape(n*m, m*p*p*n)
    T_B = T.transpose(1, 0, 2).reshape(m*p, n*m*p*n)
    T_C = T.transpose(2, 0, 1).reshape(p*n, n*m*m*p)

    def als_decompose(U, V, W, max_iter=50, tol=1e-10):
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            # Update U
            MW = np.einsum('br,cr->bcr', V, W).reshape(m*p*p*n, W.shape[1])
            G = MW.T @ MW
            try:
                G_inv = np.linalg.inv(G + tol * np.eye(W.shape[1]))
                U_new = T_A @ MW @ G_inv
            except:
                break
            # Update V
            UW = np.einsum('ar,cr->acr', U_new, W).reshape(n*m*p*n, W.shape[1])
            G = UW.T @ UW
            try:
                G_inv = np.linalg.inv(G + tol * np.eye(W.shape[1]))
                V_new = T_B @ UW @ G_inv
            except:
                break
            # Update W
            UV = np.einsum('ar,br->abr', U_new, V_new).reshape(n*m*m*p, W.shape[1])
            G = UV.T @ UV
            try:
                G_inv = np.linalg.inv(G + tol * np.eye(W.shape[1]))
                W_new = T_C @ UV @ G_inv
            except:
                break

            U, V, W = U_new, V_new, W_new
            if np.linalg.norm(U_new - U) < 1e-12:
                break
        return U, V, W

    def reduce_rank(U, V, W, max_reduction=10):
        """Try to find linearly dependent columns and reduce rank."""
        current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
        current_rank = U.shape[1]
        reduced = False

        for _ in range(max_reduction):
            if time.time() > deadline - 5:
                break
            # Check for nearly dependent columns in U
            # Use SVD to find rank-deficient columns
            try:
                U_svd = np.linalg.svd(current_U, full_matrices=False)
                V_svd = np.linalg.svd(current_V, full_matrices=False)
                W_svd = np.linalg.svd(current_W, full_matrices=False)

                # Find columns that can be expressed as linear combinations
                # Try removing each column and checking if we can still fit
                best_remove = -1
                best_error = np.inf

                for col in range(current_rank):
                    # Try removing this column
                    mask = [c for c in range(current_rank) if c != col]
                    U_sub = current_U[:, mask]
                    V_sub = current_V[:, mask]
                    W_sub = current_W[:, mask]

                    if len(mask) == 0:
                        continue

                    # Check if remaining columns can still span the tensor
                    T_comp = np.einsum('ar,br,cr->abc', U_sub, V_sub, W_sub)
                    error = np.linalg.norm(T_comp - T)

                    if error < best_error:
                        best_error = error
                        best_remove = col

                # If removing a column doesn't significantly increase error, do it
                if best_remove >= 0 and best_error < 1e-6:
                    mask = [c for c in range(current_rank) if c != best_remove]
                    current_U = current_U[:, mask]
                    current_V = current_V[:, mask]
                    current_W = current_W[:, mask]
                    current_rank = len(mask)
                    reduced = True
                else:
                    break
            except:
                break

        if reduced:
            return current_U, current_V, current_W, current_rank
        return None

    # Search for lower ranks using structured initialization
    # Strategy: Use block structure to create better initializations
    # Split the j-dimension into blocks and use Kronecker-like structure

    def structured_init(target_rank, seed):
        """Create structured initialization using block decomposition."""
        np.random.seed(seed)

        # For 3x4x6, we can try splitting j into two blocks of 2
        # Block structure: (i, j1, k) for j1 in {0,1} and (i, j2, k) for j2 in {2,3}
        # This gives 3*2*6 + 3*2*6 = 72 terms in schoolbook

        # Create a random mixing matrix to allow rank reduction
        if target_rank >= rank_sch:
            return U_sch[:, :target_rank].copy(), V_sch[:, :target_rank].copy(), W_sch[:, :target_rank].copy()

        # Use random Gaussian with structured scaling
        U_init = np.random.randn(n*m, target_rank)
        V_init = np.random.randn(m*p, target_rank)
        W_init = np.random.randn(p*n, target_rank)

        # Scale columns to have unit norm
        for r in range(target_rank):
            if np.linalg.norm(U_init[:, r]) > 1e-10:
                U_init[:, r] /= np.linalg.norm(U_init[:, r])
            if np.linalg.norm(V_init[:, r]) > 1e-10:
                V_init[:, r] /= np.linalg.norm(V_init[:, r])
            if np.linalg.norm(W_init[:, r]) > 1e-10:
                W_init[:, r] /= np.linalg.norm(W_init[:, r])

        return U_init, V_init, W_init

    # Try to find lower rank decompositions
    # Search from rank_sch-1 down to a reasonable minimum
    min_rank_to_try = max(50, rank_sch - 20)  # Try ranks from 71 down to 50

    for target_rank in range(rank_sch - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline - 15:
            break

        found = False
        # Multiple restarts with different seeds
        for restart in range(6):
            if time.time() > deadline - 15:
                break

            U_init, V_init, W_init = structured_init(target_rank, 1000 + restart * 7)

            # Run ALS
            U_sol, V_sol, W_sol = als_decompose(U_init, V_init, W_init, max_iter=60)

            # Verify continuous solution
            if verify(U_sol, V_sol, W_sol, atol=1e-8):
                # Try to round to integer/half-integer
                rounded = try_round_verify(U_sol, V_sol, W_sol)
                if rounded:
                    best_U, best_V, best_W = rounded
                    best_rank = target_rank
                    found = True
                    break

                # If continuous works but rounding fails, try polishing
                U_pol, V_pol, W_pol = als_decompose(U_sol, V_sol, W_sol, max_iter=80, tol=1e-12)
                rounded = try_round_verify(U_pol, V_pol, W_pol)
                if rounded:
                    best_U, best_V, best_W = rounded
                    best_rank = target_rank
                    found = True
                    break

                # Try rank reduction on the continuous solution
                reduced_result = reduce_rank(U_pol, V_pol, W_pol, max_reduction=5)
                if reduced_result:
                    rU, rV, rW, r_rank = reduced_result
                    if verify(rU, rV, rW, atol=1e-8):
                        rounded = try_round_verify(rU, rV, rW)
                        if rounded:
                            best_U, best_V, best_W = rounded
                            best_rank = r_rank
                            found = True
                            break

        if found:
            # Try to reduce further from this solution
            reduced_result = reduce_rank(best_U, best_V, best_W, max_reduction=5)
            if reduced_result:
                rU, rV, rW, r_rank = reduced_result
                if verify(rU, rV, rW, atol=1e-8):
                    rounded = try_round_verify(rU, rV, rW)
                    if rounded:
                        best_U, best_V, best_W = rounded
                        best_rank = r_rank
            break

    # Final: if we have a solution, try one more round of reduction
    if best_rank < rank_sch:
        reduced_result = reduce_rank(best_U, best_V, best_W, max_reduction=3)
        if reduced_result:
            rU, rV, rW, r_rank = reduced_result
            if verify(rU, rV, rW, atol=1e-8):
                rounded = try_round_verify(rU, rV, rW)
                if rounded:
                    best_U, best_V, best_W = rounded
                    best_rank = r_rank

    # Return as lists of integers or half-integers
    def to_list(mat):
        return mat.tolist()

    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
