"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Shapes: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # Tensor T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We want to find the minimal rank r such that this holds for all i,j,k.
    
    # The schoolbook rank is n*m*p = 2*3*9 = 54.
    # Lower bound: max(n*m, m*p, p*n) = max(6, 27, 18) = 27.
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """
        Verify if U, V, W form a valid decomposition.
        U, V, W are numpy arrays or lists.
        Returns True if valid, False otherwise.
        """
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = U_arr[i * m + j, :]
                    v = V_arr[j * p + k, :]
                    w = W_arr[k * n + i, :]
                    val = np.dot(u * v, w)
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def snap_to_half_int(arr):
        """Snap values to nearest half-integer."""
        return [[int(round(x * 2)) / 2.0 for x in row] for row in arr]

    def try_random_search(rank, attempts=100):
        """
        Try to find a decomposition of given rank using random initialization
        and alternating least squares.
        """
        if time.time() > deadline - 5:
            return None
        
        for _ in range(attempts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U_mat = np.random.randn(n * m, rank)
            V_mat = np.random.randn(m * p, rank)
            W_mat = np.random.randn(p * n, rank)
            
            for it in range(50):
                if time.time() > deadline - 5:
                    return None
                
                # Solve for U
                U_new = np.zeros((n * m, rank))
                for i in range(n):
                    for j in range(m):
                        A = np.zeros(rank)
                        for k in range(p):
                            A += V_mat[j * p + k, :] * W_mat[k * n + i, :]
                        norm_sq = np.sum(A * A)
                        if norm_sq > 1e-9:
                            U_new[i * m + j, :] = A / norm_sq
                U_mat = U_new
                
                # Solve for V
                V_new = np.zeros((m * p, rank))
                for j in range(m):
                    for k in range(p):
                        A = np.zeros(rank)
                        for i in range(n):
                            A += U_mat[i * m + j, :] * W_mat[k * n + i, :]
                        norm_sq = np.sum(A * A)
                        if norm_sq > 1e-9:
                            V_new[j * p + k, :] = A / norm_sq
                V_mat = V_new
                
                # Solve for W
                W_new = np.zeros((p * n, rank))
                for k in range(p):
                    for i in range(n):
                        A = np.zeros(rank)
                        for j in range(m):
                            A += U_mat[i * m + j, :] * V_mat[j * p + k, :]
                        norm_sq = np.sum(A * A)
                        if norm_sq > 1e-9:
                            W_new[k * n + i, :] = A / norm_sq
                W_mat = W_new
                
                # Check convergence
                loss = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u = U_mat[i * m + j, :]
                            v = V_mat[j * p + k, :]
                            w = W_mat[k * n + i, :]
                            val = np.dot(u * v, w)
                            loss += (val - 1.0) ** 2
                
                if loss < 1e-6:
                    U_snap = snap_to_half_int(U_mat.tolist())
                    V_snap = snap_to_half_int(V_mat.tolist())
                    W_snap = snap_to_half_int(W_mat.tolist())
                    
                    if verify(U_snap, V_snap, W_snap):
                        return {"U": U_snap, "V": V_snap, "W": W_snap}
        
        return None

    # Try ranks from lower bound upwards
    for rank in range(27, 55):
        if time.time() > deadline - 10:
            break
        
        attempts = max(5, 50 - (rank - 27) * 2)
        res = try_random_search(rank, attempts)
        if res is not None:
            return res
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
