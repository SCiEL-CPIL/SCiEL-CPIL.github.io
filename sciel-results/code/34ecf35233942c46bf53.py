import numpy as np
import time
import random

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, np_ = n*m, m*p, n*p
    
    # Active triples: (a,b,c) where T=1
    active_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                active_triples.append((a, b, c))
    active_set = set(active_triples)
    
    def schoolbook():
        r = nm * mp
        U = [[0.0]*r for _ in range(nm)]
        V = [[0.0]*r for _ in range(mp)]
        W = [[0.0]*r for _ in range(np_)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return U, V, W, r
    
    U_sb, V_sb, W_sb, r_sb = schoolbook()
    best = (U_sb, V_sb, W_sb, r_sb)
    
    def verify(U, V, W, r, tol=1e-9):
        for (a, b, c) in active_triples:
            s = sum(U[a][rc] * V[b][rc] * W[c][rc] for rc in range(r))
            if abs(s - 1.0) > tol:
                return False
        rng = random.Random(12345)
        for _ in range(300):
            a = rng.randint(0, nm-1)
            b = rng.randint(0, mp-1)
            c = rng.randint(0, np_-1)
            if (a, b, c) in active_set:
                continue
            s = sum(U[a][rc] * V[b][rc] * W[c][rc] for rc in range(r))
            if abs(s) > tol:
                return False
        return True
    
    val_set = [0.0, 0.5, 1.0, 1.5, 2.0, -0.5, -1.0, -1.5, -2.0]
    
    def random_search(r, num_trials=50):
        if time.time() > deadline - 15:
            return None
        
        rng = random.Random(777)
        for trial in range(num_trials):
            if time.time() > deadline - 10:
                return None
            
            U = [[rng.choice(val_set) for _ in range(r)] for _ in range(nm)]
            V = [[rng.choice(val_set) for _ in range(r)] for _ in range(mp)]
            W = [[rng.choice(val_set) for _ in range(r)] for _ in range(np_)]
            
            for it in range(100):
                if time.time() > deadline - 10:
                    break
                
                # Update W
                for c in range(np_):
                    k_val = c // n
                    i_val = c % n
                    rows = []
                    rhs = []
                    for j in range(m):
                        a_val = i_val*m + j
                        b_val = j*p + k_val
                        rows.append([U[a_val][rc] * V[b_val][rc] for rc in range(r)])
                        rhs.append(1.0)
                    if rows:
                        A_sys = np.array(rows)
                        b_sys = np.array(rhs)
                        sol, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        W[c] = [round(float(v)*2)/2 for v in sol]
                
                # Update V
                for b in range(mp):
                    j_val = b // p
                    k_val = b % p
                    rows = []
                    rhs = []
                    for i in range(n):
                        a_val = i*m + j_val
                        c_val = k_val*n + i
                        rows.append([U[a_val][rc] * W[c_val][rc] for rc in range(r)])
                        rhs.append(1.0)
                    if rows:
                        A_sys = np.array(rows)
                        b_sys = np.array(rhs)
                        sol, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        V[b] = [round(float(v)*2)/2 for v in sol]
                
                # Update U
                for a in range(nm):
                    i_val = a // m
                    j_val = a % m
                    rows = []
                    rhs = []
                    for k in range(p):
                        b_val = j_val*p + k
                        c_val = k*n + i_val
                        rows.append([V[b_val][rc] * W[c_val][rc] for rc in range(r)])
                        rhs.append(1.0)
                    if rows:
                        A_sys = np.array(rows)
                        b_sys = np.array(rhs)
                        sol, _, _, _ = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                        U[a] = [round(float(v)*2)/2 for v in sol]
            
            if verify(U, V, W, r):
                return U, V, W, r
        
        return None
    
    lower_bound = max(nm, mp, np_)
    for r in range(lower_bound, best[3]):
        if time.time() > deadline - 15:
            break
        result = random_search(r)
        if result is not None:
            best = result
            break
    
    U, V, W, r = best
    return {"U": U, "V": V, "W": W}
