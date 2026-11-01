import numpy as np
import time
import random

def get_decomp_222():
    terms = []
    u = np.array([1, 0, 0, 1]); v = np.array([1, 0, 0, 1]); w = np.array([1, 0, 0, 1]); terms.append((u, v, w))
    u = np.array([1, 0, 0, 1]); v = np.array([0, 0, 1, 0]); w = np.array([0, 0, 1, 0]); terms.append((u, v, w))
    u = np.array([1, 0, 0, 0]); v = np.array([0, 1, 0, -1]); w = np.array([0, 1, 0, 0]); terms.append((u, v, w))
    u = np.array([0, 0, 0, 1]); v = np.array([1, 0, -1, 0]); w = np.array([0, 0, 0, 1]); terms.append((u, v, w))
    u = np.array([0, 1, 0, 1]); v = np.array([0, 0, 0, 1]); w = np.array([0, 0, 0, 1]); terms.append((u, v, w))
    u = np.array([1, 0, -1, 0]); v = np.array([1, 1, 0, 0]); w = np.array([1, 0, 0, 0]); terms.append((u, v, w))
    u = np.array([0, 0, 1, -1]); v = np.array([0, 0, 1, 1]); w = np.array([0, 1, 0, 0]); terms.append((u, v, w))
    return terms

def get_decomp_221():
    terms = []
    u = np.array([1, 0, 0, 0]); v = np.array([1, 0]); w = np.array([1, 0]); terms.append((u, v, w))
    u = np.array([0, 1, 0, 0]); v = np.array([0, 1]); w = np.array([1, 0]); terms.append((u, v, w))
    u = np.array([0, 0, 1, 0]); v = np.array([1, 0]); w = np.array([0, 1]); terms.append((u, v, w))
    u = np.array([0, 0, 0, 1]); v = np.array([0, 1]); w = np.array([0, 1]); terms.append((u, v, w))
    return terms

def get_decomp_212():
    terms = []
    u = np.array([1, 0]); v = np.array([1, 0]); w = np.array([1, 0, 0, 0]); terms.append((u, v, w))
    u = np.array([1, 0]); v = np.array([0, 1]); w = np.array([0, 1, 0, 0]); terms.append((u, v, w))
    u = np.array([0, 1]); v = np.array([1, 0]); w = np.array([0, 0, 1, 0]); terms.append((u, v, w))
    u = np.array([0, 1]); v = np.array([0, 1]); w = np.array([0, 0, 0, 1]); terms.append((u, v, w))
    return terms

def get_decomp_211():
    terms = []
    u = np.array([1, 0]); v = np.array([1]); w = np.array([1, 0]); terms.append((u, v, w))
    u = np.array([0, 1]); v = np.array([1]); w = np.array([0, 1]); terms.append((u, v, w))
    return terms

def validate(U, V, W, n, m, p):
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = 2, 5, 5
    deadline = time.time() + 110
    
    # Try to find rank 19
    # Random search
    for r in range(19, 25):
        if time.time() > deadline:
            break
        for _ in range(1000):
            if time.time() > deadline:
                break
            U = np.random.choice([-1, 0, 1], size=(10, r))
            V = np.random.choice([-1, 0, 1], size=(25, r))
            W = np.random.choice([-1, 0, 1], size=(10, r))
            if validate(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    # Fallback to rank 46
    d222 = get_decomp_222()
    d221 = get_decomp_221()
    d212 = get_decomp_212()
    d211 = get_decomp_211()
    
    U = np.zeros((10, 46))
    V = np.zeros((25, 46))
    W = np.zeros((10, 46))
    
    col = 0
    def add_term(u, v, w, i_offset, j_offset, k_offset):
        nonlocal col
        u = u.reshape(2, u.shape[0]//2)
        v = v.reshape(u.shape[1], w.shape[0]//2)
        w = w.reshape(2, w.shape[0]//2)
        
        Ja = u.shape[1]
        Kb = w.shape[1]
        
        for i in range(2):
            for jj in range(Ja):
                U[i*5 + (j_offset + jj), col] = u[i, jj]
        for jj in range(Ja):
            for kk in range(Kb):
                V[(j_offset + jj)*5 + (k_offset + kk), col] = v[jj, kk]
        for i in range(2):
            for kk in range(Kb):
                W[(k_offset + kk)*2 + i, col] = w[i, kk]
        col += 1
        
    J_offsets = [0, 2, 4]
    K_offsets = [0, 2, 4]
    J_sizes = [2, 2, 1]
    K_sizes = [2, 2, 1]
    
    for a in range(3):
        for b in range(3):
            Ja = J_sizes[a]
            Kb = K_sizes[b]
            if Ja == 2 and Kb == 2: terms = d222
            elif Ja == 2 and Kb == 1: terms = d221
            elif Ja == 1 and Kb == 2: terms = d212
            else: terms = d211
            for (u, v, w) in terms:
                add_term(u, v, w, J_offsets[a], K_offsets[b])
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
