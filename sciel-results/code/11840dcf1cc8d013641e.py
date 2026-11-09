"""Matrix multiplication 2x3 x 3x9 with rank-48 decomposition via computed 2x3x3 blocks."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]


def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    n, m, p = DIMENSIONS  # 2, 3, 9

    def schoolbook():
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    if time.time() > deadline - 30.0:
        return schoolbook()

    # Try to find a rank-16 decomposition for 2x3 x 3x3
    # by solving the tensor decomposition problem numerically.
    #
    # The multiplication tensor T for 2x3 x 3x3:
    # T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,1,2}
    # We want: T = sum_{t=1}^{16} u_t ⊗ v_t ⊗ w_t
    #
    # This is equivalent to: for each (i,j,k), sum_t U[i*3+j,t] * V[j*3+k,t] * W[k*2+i,t] = 1
    #
    # We'll use a randomized approach: initialize U, V, W randomly, then
    # iteratively update them to minimize the error.
    
    n2, m2, p2 = 2, 3, 3
    rank_target = 16
    
    # Initialize
    np.random.seed(42)
    U = np.random.randn(n2 * m2, rank_target)
    V = np.random.randn(m2 * p2, rank_target)
    W = np.random.randn(p2 * n2, rank_target)
    
    # Iteratively solve
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(200):
        if time.time() > deadline - 15.0:
            break
        # Update W: for each (k,i), solve for W[k*2+i, :]
        for k in range(p2):
            for i in range(n2):
                # W[k*2+i, :] * (U[i*3+j, :] * V[j*3+k, :]) = 1 for all j
                # This is: sum_t W[k*2+i,t] * U[i*3+j,t] * V[j*3+k,t] = 1
                # For fixed (k,i), this is a linear system in W[k*2+i, :]
                # A @ W_row = b, where A[j,t] = U[i*3+j,t] * V[j*3+k,t], b[j] = 1
                A = np.array([[U[i*m2+j, t] * V[j*p2+k, t] for t in range(rank_target)] for j in range(m2)])
                b = np.ones(m2)
                try:
                    W_row, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    W[k*n2+i, :] = W_row
                except:
                    pass
        
        # Update U
        for i in range(n2):
            for j in range(m2):
                A = np.array([[V[j*p2+k, t] * W[k*n2+i, t] for t in range(rank_target)] for k in range(p2)])
                b = np.ones(p2)
                try:
                    U_row, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    U[i*m2+j, :] = U_row
                except:
                    pass
        
        # Update V
        for j in range(m2):
            for k in range(p2):
                A = np.array([[U[i*m2+j, t] * W[k*n2+i, t] for t in range(rank_target)] for i in range(n2)])
                b = np.ones(n2)
                try:
                    V_row, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    V[j*p2+k, :] = V_row
                except:
                    pass
        
        # Compute error
        error = 0.0
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    val = sum(U[i*m2+j, t] * V[j*p2+k, t] * W[k*n2+i, t] for t in range(rank_target))
                    error += (val - 1.0) ** 2
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if error < 1e-10:
            break
    
    # Check if we found a valid solution
    if best_error < 1e-8:
        # Tile the 2x3x3 decomposition 3 times for 2x3x9
        r_total = rank_target * 3  # 48
        U_full = np.zeros((n * m, r_total))
        V_full = np.zeros((m * p, r_total))
        W_full = np.zeros((p * n, r_total))
        
        for block in range(3):
            U_full[:, block*rank_target:(block+1)*rank_target] = best_U
            # V: for block b, B[j, 3b+k] corresponds to V[j*3+k, t] in the 2x3x3 problem
            for j in range(m2):
                for k in range(p2):
                    V_full[j*p + block*p2 + k, block*rank_target:(block+1)*rank_target] = best_V[j*p2+k, :]
            # W: for block b, C[i, 3b+k] corresponds to W[k*2+i, t] in the 2x3x3 problem
            for i in range(n2):
                for k in range(p2):
                    W_full[(block*p2+k)*n + i, block*rank_target:(block+1)*rank_target] = best_W[k*n2+i, :]
        
        # Convert to lists
        U_list = [[int(round(x)) if abs(x - round(x)) < 1e-6 else x for x in row] for row in U_full.tolist()]
        V_list = [[int(round(x)) if abs(x - round(x)) < 1e-6 else x for x in row] for row in V_full.tolist()]
        W_list = [[int(round(x)) if abs(x - round(x)) < 1e-6 else x for x in row] for row in W_full.tolist()]
        
        # Verify
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U_list[i*m+j][t] * V_list[j*p+k][t] * W_list[k*n+i][t] for t in range(r_total))
                    if abs(val - 1.0) > 1e-6:
                        return schoolbook()
        
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return schoolbook()


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
