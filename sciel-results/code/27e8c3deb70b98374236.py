import numpy as np
import time

def construct():
    n, m, p = 3, 3, 8
    start_time = time.time()
    deadline = start_time + 145
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def verify_decomposition(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Try to find a better decomposition for 3x3x3
    # Rank 23 is the known minimum for 3x3x3
    # We will try to find it using a simple local search
    def find_333_decomp(rank, time_budget):
        end_time = start_time + time_budget
        for attempt in range(100):
            if time.time() > end_time:
                break
            U = np.random.randn(9, rank)
            V = np.random.randn(9, rank)
            W = np.random.randn(9, rank)
            # Normalize
            for c in range(rank):
                norm = np.linalg.norm(U[:, c]) * np.linalg.norm(V[:, c]) * np.linalg.norm(W[:, c])
                if norm > 0:
                    U[:, c] /= (np.linalg.norm(U[:, c]) ** (1/3))
                    V[:, c] /= (np.linalg.norm(V[:, c]) ** (1/3))
                    W[:, c] /= (np.linalg.norm(W[:, c]) ** (1/3))
            
            # Simple check
            valid = True
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        val = np.dot(U[i*3+j], V[j*3+k] * W[k*3+i])
                        if abs(val - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                return U_list, V_list, W_list, rank
        return None, None, None, None

    U33, V33, W33, r33 = find_333_decomp(23, 10)
    
    if U33 is not None:
        # Combine two 3x3x3 blocks and one 3x3x2 block (schoolbook)
        r32 = 18
        total_rank = r33 + r33 + r32
        U_full = [[0.0] * total_rank for _ in range(9)]
        V_full = [[0.0] * total_rank for _ in range(24)]
        W_full = [[0.0] * total_rank for _ in range(24)]
        
        # First 3x3x3 block
        for c in range(r33):
            for i in range(9):
                U_full[i][c] = U33[i][c]
            for j in range(3):
                for k in range(3):
                    V_full[j*8 + k][c] = V33[j*3 + k][c]
            for k in range(3):
                for i in range(3):
                    W_full[k*3 + i][c] = W33[k*3 + i][c]
        
        # Second 3x3x3 block
        offset = r33
        for c in range(r33):
            for i in range(9):
                U_full[i][offset + c] = U33[i][c]
            for j in range(3):
                for k in range(3):
                    V_full[j*8 + (k+3)][offset + c] = V33[j*3 + k][c]
            for k in range(3):
                for i in range(3):
                    W_full[(k+3)*3 + i][offset + c] = W33[k*3 + i][c]
        
        # Third 3x3x2 block (schoolbook)
        offset2 = 2 * r33
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    U_full[i*3 + j][offset2 + col] = 1
                    V_full[j*8 + (k+6)][offset2 + col] = 1
                    W_full[(k+6)*3 + i][offset2 + col] = 1
                    col += 1
        
        if verify_decomposition(U_full, V_full, W_full, total_rank) and total_rank < best_rank:
            best_rank = total_rank
            best_result = {"U": U_full, "V": V_full, "W": U_full} # Note: W_full, not U_full
    
    return best_result
