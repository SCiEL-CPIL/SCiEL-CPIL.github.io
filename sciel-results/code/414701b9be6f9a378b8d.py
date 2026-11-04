import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Fallback: Rank 22 solution
    def get_rank_22():
        rank = 22
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        col = 0
        def add_strassen(col_offset):
            nonlocal col
            # Local indices 0..3 for A, B, C
            # m1
            U[0][col] += 1; U[3][col] += 1
            V[0][col] += 1; V[3][col] += 1
            W[0][col] += 1; W[3][col] += 1
            col += 1
            # m2
            U[2][col] += 1; U[3][col] += 1
            V[0][col] += 1
            W[2][col] += 1
            col += 1
            # m3
            U[0][col] += 1
            V[1][col] += 1; V[3][col] -= 1
            W[1][col] += 1
            col += 1
            # m4
            U[3][col] += 1
            V[2][col] += 1; V[0][col] -= 1
            W[0][col] += 1; W[2][col] += 1
            col += 1
            # m5
            U[0][col] += 1; U[1][col] += 1
            V[3][col] += 1
            W[1][col] += 1; W[0][col] -= 1
            col += 1
            # m6
            U[2][col] += 1; U[0][col] -= 1
            V[0][col] += 1; V[1][col] += 1
            W[3][col] += 1
            col += 1
            # m7
            U[1][col] += 1; U[3][col] -= 1
            V[2][col] += 1; V[3][col] += 1
            W[0][col] += 1
            col += 1

        # Part 1: i=0, j=0, k=0
        add_strassen(0)
        
        # Part 2: i=0, j=0, k=2
        # Shift V by 2 in k, W by 2 in k
        # U is same
        # We need to add to U, V, W with offsets
        # My add_strassen above assumes local indices 0..3 map to global 0..3.
        # For Part 2, V global indices are 2..5, W global indices are 2..5.
        
        # Let's just write a generic add_term
        def add_term(u_map, v_map, w_map):
            nonlocal col
            for idx, val in u_map.items():
                U[idx][col] += val
            for idx, val in v_map.items():
                V[idx][col] += val
            for idx, val in w_map.items():
                W[idx][col] += val
            col += 1

        col = 0
        # Part 1
        add_term({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1})
        add_term({2:1, 3:1}, {0:1}, {2:1})
        add_term({0:1}, {1:1, 3:-1}, {1:1})
        add_term({3:1}, {2:1, 0:-1}, {0:1, 2:1})
        add_term({0:1, 1:1}, {3:1}, {1:1, 0:-1})
        add_term({2:1, 0:-1}, {0:1, 1:1}, {3:1})
        add_term({1:1, 3:-1}, {2:1, 3:1}, {0:1})
        
        # Part 2: V indices +2, W indices +2
        add_term({0:1, 3:1}, {2:1, 5:1}, {2:1, 5:1})
        add_term({2:1, 3:1}, {2:1}, {4:1})
        add_term({0:1}, {3:1, 5:-1}, {3:1})
        add_term({3:1}, {4:1, 2:-1}, {2:1, 4:1})
        add_term({0:1, 1:1}, {5:1}, {3:1, 2:-1})
        add_term({2:1, 0:-1}, {2:1, 3:1}, {5:1})
        add_term({1:1, 3:-1}, {4:1, 5:1}, {2:1})
        
        # Part 3: Schoolbook
        for i in range(2):
            for k in range(4):
                U[i*m + 2][col] = 1
                V[2*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
                
        return {"U": U, "V": V, "W": W}

    # Try search for rank 19
    start_time = time.time()
    deadline = start_time + 100
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    for r in [19]:
        for attempt in range(20):
            if time.time() > deadline:
                break
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            for it in range(1000):
                R = T - np.einsum('ak,bk,ck->abc', U, V, W)
                grad_U = np.einsum('abc,bc,ac->ac', R, V, W)
                grad_V = np.einsum('abc,ac,bc->bc', R, U, W)
                grad_W = np.einsum('abc,ab,cb->cb', R, U, V)
                
                U -= 0.01 * grad_U
                V -= 0.01 * grad_V
                W -= 0.01 * grad_W
                
                if it % 50 == 0:
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    R_snap = T - np.einsum('ak,bk,ck->abc', U_snap, V_snap, W_snap)
                    if np.allclose(R_snap, 0, atol=1e-9):
                        return {
                            "U": U_snap.tolist(),
                            "V": V_snap.tolist(),
                            "W": W_snap.tolist()
                        }
                        
    return get_rank_22()
