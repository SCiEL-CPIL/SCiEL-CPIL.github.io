import numpy as np
import time

def construct():
    n, m, p = 2, 5, 5
    deadline = time.time() + 140

    def verify(U, V, W):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Schoolbook fallback
    U_school = [[0]*50 for _ in range(10)]
    V_school = [[0]*50 for _ in range(25)]
    W_school = [[0]*50 for _ in range(10)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1

    best_solution = {"U": U_school, "V": V_school, "W": W_school}

    # Try to find lower rank
    for r in range(35, 50):
        if time.time() > deadline - 10:
            break

        found = False
        for t in range(30):
            if time.time() > deadline - 5:
                break

            np.random.seed(t * 13 + r)
            U = np.zeros((10, r))
            V = np.zeros((25, r))
            W = np.zeros((10, r))

            # Sparse init
            for c in range(r):
                for i in np.random.choice(10, size=2, replace=False):
                    U[i, c] = np.random.choice([-1, 1])
                for j in np.random.choice(25, size=2, replace=False):
                    V[j, c] = np.random.choice([-1, 1])
                for k in np.random.choice(10, size=2, replace=False):
                    W[k, c] = np.random.choice([-1, 1])

            for step in range(80):
                err = sum(abs(sum(U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] for c in range(r)) - 1.0)
                          for i in range(n) for j in range(m) for k in range(p))
                if err < 1e-9:
                    if verify(U, V, W):
                        found = True
                        best_solution = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        break
                    break

                var = np.random.randint(0, 3)
                if var == 0:
                    i, c = divmod(np.random.randint(0, 10*r), r)
                    U[i, c] = np.random.choice([-1, -0.5, 0, 0.5, 1])
                elif var == 1:
                    j, c = divmod(np.random.randint(0, 25*r), r)
                    V[j, c] = np.random.choice([-1, -0.5, 0, 0.5, 1])
                else:
                    k, c = divmod(np.random.randint(0, 10*r), r)
                    W[k, c] = np.random.choice([-1, -0.5, 0, 0.5, 1])

            if found:
                break

        if found:
            break

    return best_solution
