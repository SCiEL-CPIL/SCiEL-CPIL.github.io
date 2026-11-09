import numpy as np

DIMENSIONS = [3, 3, 3]

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

    # Laderman Rank 23
    # ... hardcoded matrices ...
    
    # Verification function
    def verify(U, V, W):
        R = len(U[0])
        # Check for all i, k, j, j'
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = 0
                        for r in range(R):
                            s += U[i * m + j][r] * V[jp * p + k][r] * W[k * n + i][r]
                        target = 1 if j == jp else 0
                        if abs(s - target) > 1e-9:
                            return False
        return True

    # Try Laderman
    # (Insert Laderman matrices here)
    
    return {"U": U_school, "V": V_school, "W": W_school}
