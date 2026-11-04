    import numpy as np

    def construct():
        n, m, p = 2, 4, 7
        rank = n * m * p
        
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    alpha = i * (m * p) + j * p + k
                    U[i * m + j, alpha] = 1
                    V[j * p + k, alpha] = 1
                    W[k * n + i, alpha] = 1
                    
        return {"U": U, "V": V, "W": W}
    