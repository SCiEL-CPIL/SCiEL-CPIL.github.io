import numpy as np
import random
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Try to find a lower rank solution
    # Target rank 12
    target_rank = 12
    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin
    
    # Random search for integer solutions
    # We assume entries are in {-1, 0, 1}
    # This is a heuristic, might not find it, but worth a try.
    # If not found, return schoolbook.
    
    # A more robust approach: use the known result that rank is 12.
    # Can we construct it explicitly?
    # For 2x3x2, rank is 6.
    # Let's try to solve 2x3x2 first.
    
    def solve_small(n_s, m_s, p_s, r_s, timeout):
        # n_s=2, m_s=3, p_s=2
        # Equations: for all i,j,k, sum_c U[i*m_s+j,c] V[j*p_s+k,c] W[k*n_s+i,c] == 1
        # Random restarts
        start = time.time()
        while time.time() - start < timeout:
            U_rand = np.random.randint(-1, 2, size=(n_s * m_s, r_s))
            V_rand = np.random.randint(-1, 2, size=(m_s * p_s, r_s))
            W_rand = np.random.randint(-1, 2, size=(p_s * n_s, r_s))
            
            # Check if this solution works
            valid = True
            for i in range(n_s):
                for j in range(m_s):
                    for k in range(p_s):
                        s = 0
                        for c in range(r_s):
                            s += U_rand[i * m_s + j, c] * V_rand[j * p_s + k, c] * W_rand[k * n_s + i, c]
                        if s != 1:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            if valid:
                return U_rand.tolist(), V_rand.tolist(), W_rand.tolist()
        return None

    # Try to solve 2x3x2 with rank 6
    sol_232 = solve_small(2, 3, 2, 6, 5.0)
    
    if sol_232:
        U_232, V_232, W_232 = sol_232
        # Construct 2x3x4 solution by duplicating
        # The output indices k are 0,1 for the first part and 2,3 for the second.
        # We can map k=0,1 to the 2x3x2 solution and k=2,3 to another copy.
        # U is same for both (depends on i,j).
        # V depends on j,k. For k=0,1 use V_232. For k=2,3 use V_232 (shifted index?).
        # V index is j*p + k.
        # For 2x3x4, p=4.
        # V_232 has p=2. Index j*2 + k.
        # In 2x3x4, V index is j*4 + k.
        # For k=0,1: V_232[j*2+k] corresponds to V_234[j*4+k].
        # For k=2,3: V_232[j*2+(k-2)] corresponds to V_234[j*4+k].
        
        U_234 = [[0] * 12 for _ in range(6)]
        V_234 = [[0] * 12 for _ in range(12)]
        W_234 = [[0] * 12 for _ in range(8)]
        
        # Copy U_232 to first 6 columns of U_234
        for i in range(6):
            for c in range(6):
                U_234[i][c] = U_232[i][c]
                U_234[i][c+6] = U_232[i][c]
                
        # Copy V_232 to V_234
        for j in range(3):
            for k in range(2):
                # First part (k=0,1)
                for c in range(6):
                    V_234[j*4 + k][c] = V_232[j*2 + k][c]
                # Second part (k=2,3)
                for c in range(6):
                    V_234[j*4 + (k+2)][c+6] = V_232[j*2 + k][c]
                    
        # Copy W_232 to W_234
        # W index is k*n + i.
        # For 2x3x2, n=2. Index k*2 + i.
        # For 2x3x4, n=2. Index k*2 + i.
        # k=0,1 for first part. k=2,3 for second part.
        for k in range(2):
            for i in range(2):
                for c in range(6):
                    W_234[k*2 + i][c] = W_232[k*2 + i][c]
                    W_234[(k+2)*2 + i][c+6] = W_232[k*2 + i][c]
                    
        # Verify the constructed solution
        valid = True
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    s = 0
                    for c in range(12):
                        s += U_234[i*3+j][c] * V_234[j*4+k][c] * W_234[k*2+i][c]
                    if s != 1:
                        valid = False
                        break
                if not valid: break
            if not valid: break
            
        if valid:
            return {"U": U_234, "V": V_234, "W": W_234}
            
    # If search failed, return schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
