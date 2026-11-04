import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = 3, 3, 6
    dim_U = n * m  # 9
    dim_V = m * p  # 18
    dim_W = p * n  # 18

    T_target = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1.0

    def validate(U, V, W):
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        T_comp = np.einsum('ur,vr,wr->uvw', U_np, V_np, W_np)
        if not np.allclose(T_comp, T_target, atol=1e-9):
            return False
        # Check half-integers
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        return True

    # Trivial solution rank 54
    U0 = [[0]*54 for _ in range(9)]
    V0 = [[0]*54 for _ in range(18)]
    W0 = [[0]*54 for _ in range(18)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                U0[u][col] = 1
                V0[v][col] = 1
                W0[w][col] = 1
                col += 1
    best_solution = (U0, V0, W0)
    best_rank = 54

    # Try to find better rank with gradient descent
    for r in range(45, 39, -1):
        if time.time() > deadline:
            break
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            U = np.random.randn(dim_U, r)
            V = np.random.randn(dim_V, r)
            W = np.random.randn(dim_W, r)
            
            lr = 0.01
            for step in range(200):
                T_comp = np.einsum('ur,vr,wr->uvw', U, V, W)
                Error = T_comp - T_target
                loss = np.sum(Error**2)
                
                # Gradients
                G_W = np.einsum('uvw,ur,vr->wr', Error, U, V)
                G_U = np.einsum('uvw,vr,wr->ur', Error, V, W)
                G_V = np.einsum('uvw,ur,wr->vr', Error, U, W)
                
                U -= lr * G_U
                V -= lr * G_V
                W -= lr * G_W
                
                # Check convergence
                if loss < 1e-6:
                    break
            
            # Round to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            U_list = U_r.tolist()
            V_list = V_r.tolist()
            W_list = W_r.tolist()
            
            if validate(U_list, V_list, W_list):
                best_solution = (U_list, V_list, W_list)
                best_rank = r
                break
        
        if best_rank < 54:
            break

    U, V, W = best_solution
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
