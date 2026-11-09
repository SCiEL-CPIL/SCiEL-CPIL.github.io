import time
import numpy as np

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    def verify_decomp(U, V, W, tol=1e-10):
        rank = len(U[0])
        U_arr = np.array(U).reshape(3, 3, rank)
        V_arr = np.array(V).reshape(3, 3, rank)
        W_arr = np.array(W).reshape(3, 3, rank)
        
        for k in range(3):
            # W[k, i, :] is shape (3, rank) for i=0..2
            # V[j, k, :] is shape (3, rank) for j=0..2
            # U[i, j, :] is shape (3, 3, rank)
            
            # We want sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1 for all i,j
            
            # Construct M[i, j, r] = U[i, j, r] * W[k, i, r]
            # W_ki = W_arr[k, :, :] (shape 3, rank)
            # W_ki[i, r]
            # U_arr is (3, 3, rank) -> i, j, r
            # We need to multiply U[i, j, r] by W[k, i, r]
            # W_ki[np.newaxis, :, :] is (1, 3, rank) -> 1, i, r
            # Broadcasting: (3, 3, rank) * (1, 3, rank) -> (3, 3, rank)
            M = U_arr * W_arr[k, :, :][np.newaxis, :, :]
            
            # Now we need sum_r M[i, j, r] * V[j, k, r]
            # V_jk = V_arr[:, k, :] (shape 3, rank) -> j, r
            # V_jk[np.newaxis, :, :] is (1, 3, rank) -> 1, j, r
            # M * V_jk[np.newaxis, :, :] -> (3, 3, rank)
            prod = M * V_arr[:, k, :][np.newaxis, :, :]
            
            # Sum over r (axis 2)
            result_k = np.sum(prod, axis=2) # shape (3, 3) -> i, j
            
            if np.any(np.abs(result_k - 1.0) > tol):
                return False
        return True

    def try_round_and_verify(U, V, W):
        # Try to round to half-integers
        U_r = np.round(np.array(U) * 2) / 2
        V_r = np.round(np.array(V) * 2) / 2
        W_r = np.round(np.array(W) * 2) / 2
        
        # Check if rounding changed values significantly?
        # Just verify the rounded version
        if verify_decomp(U_r.tolist(), V_r.tolist(), W_r.tolist(), tol=1e-9):
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        return None

    def trivial_decomp():
        U = [[0] * 27 for _ in range(9)]
        V = [[0] * 27 for _ in range(9)]
        W = [[0] * 27 for _ in range(9)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i * 3 + j][col] = 1
                    V[j * 3 + k][col] = 1
                    W[k * 3 + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    target_rank = 23
    best_result = None

    # Vectorized ALS
    for restart in range(50):
        if time.time() > deadline - 15:
            break

        np.random.seed(1000 + restart)
        # Initialize with small random values
        U = np.random.randn(9, target_rank) * 0.5
        V = np.random.randn(9, target_rank) * 0.5
        W = np.random.randn(9, target_rank) * 0.5

        U_3d = U.reshape(3, 3, target_rank)
        V_3d = V.reshape(3, 3, target_rank)
        W_3d = W.reshape(3, 3, target_rank)

        for iteration in range(1000):
            if time.time() > deadline - 5:
                break

            # Update U
            for i in range(3):
                for j in range(3):
                    M_ij = V_3d[j, :, :] * W_3d[:, i, :]
                    try:
                        A = M_ij.T @ M_ij
                        b = M_ij.T @ np.ones(3)
                        A += 1e-8 * np.eye(target_rank)
                        U_3d[i, j, :] = np.linalg.solve(A, b)
                    except np.linalg.LinAlgError:
                        pass

            # Update V
            for j in range(3):
                for k in range(3):
                    M_jk = U_3d[:, j, :] * W_3d[k, :, :]
                    try:
                        A = M_jk.T @ M_jk
                        b = M_jk.T @ np.ones(3)
                        A += 1e-8 * np.eye(target_rank)
                        V_3d[j, k, :] = np.linalg.solve(A, b)
                    except np.linalg.LinAlgError:
                        pass

            # Update W
            for k in range(3):
                for i in range(3):
                    M_ki = U_3d[i, :, :] * V_3d[:, k, :]
                    try:
                        A = M_ki.T @ M_ki
                        b = M_ki.T @ np.ones(3)
                        A += 1e-8 * np.eye(target_rank)
                        W_3d[k, i, :] = np.linalg.solve(A, b)
                    except np.linalg.LinAlgError:
                        pass

            # Check error periodically
            if iteration % 100 == 0:
                max_err = 0.0
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            s = np.sum(U_3d[i, j, :] * V_3d[j, k, :] * W_3d[k, i, :])
                            err = abs(s - 1.0)
                            if err > max_err:
                                max_err = err
                if max_err < 1e-10:
                    break

        # Final verification and rounding attempt
        U_flat = U_3d.reshape(9, target_rank)
        V_flat = V_3d.reshape(9, target_rank)
        W_flat = W_3d.reshape(9, target_rank)
        
        # Try rounding to half-integers
        result = try_round_and_verify(U_flat, V_flat, W_flat)
        if result:
            best_result = result
            break

    if best_result is not None:
        return best_result

    return trivial_decomp()
