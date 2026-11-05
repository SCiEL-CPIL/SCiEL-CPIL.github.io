import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Try ranks from 36 down to 1
    for R in range(36, 0, -1):
        # Try a few random initializations
        for _ in range(1000):
            # Generate U and V over integers (or F2)
            # Using small integers to avoid overflow and keep it simple
            # F2 is {0, 1}
            U = np.random.randint(0, 2, size=(n, m, R))
            V = np.random.randint(0, 2, size=(m, p, R))
            
            # Compute m_ij vectors
            # m_ij[r] = sum_k U[i,k,r] * V[k,j,r]
            # We need m_ij != 0 for all i, j
            valid = True
            m_vecs = {}
            for i in range(n):
                for j in range(p):
                    m_vec = np.zeros(R, dtype=int)
                    for k in range(m):
                        m_vec += U[i, k, :] * V[k, j, :]
                    # Check if m_vec is non-zero
                    if np.all(m_vec == 0):
                        valid = False
                        break
                    m_vecs[(i, j)] = m_vec
                if not valid:
                    break
            
            if valid:
                # Construct W
                # W shape (p, n, R)
                # Index W[j, i, r]
                W = np.zeros((p, n, R), dtype=int)
                for i in range(n):
                    for j in range(p):
                        m_vec = m_vecs[(i, j)]
                        # Find a w_vec such that w_vec . m_vec == 1
                        # Since we are over integers, and m_vec has non-zero entries,
                        # we can pick a non-zero entry and set w to 1/m?
                        # But we want integer/half-integer entries.
                        # If we use F2, m_vec entries are 0 or 1.
                        # If any entry is 1, we can set w to 1 at that position.
                        # Let's stick to F2 for simplicity and validity.
                        # The task allows integer/half-integer. F2 is integer.
                        idx = np.nonzero(m_vec)[0]
                        if len(idx) > 0:
                            W[j, i, idx[0]] = 1
                        else:
                            # Should not happen if valid
                            pass
                
                # Convert to lists
                U_list = U.reshape(n*m, R).tolist()
                V_list = V.reshape(m*p, R).tolist()
                W_list = W.reshape(p*n, R).tolist()
                
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
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
