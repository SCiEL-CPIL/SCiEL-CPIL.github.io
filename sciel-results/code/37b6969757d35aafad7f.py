import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # --- Part 1: Kronecker Product of Strassen (4x4x4) ---
    # Strassen 2x2x2 decomposition
    # U, V, W are 4x7 for 2x2x2
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # r=0: m1 = (a+d)(e+h) -> C00, C11
    # A: a(0,0), d(1,1) -> U: [1,0,0,1]
    # B: e(0,0), h(1,1) -> V: [1,0,0,1]
    # C: C00(i=0,k=0), C11(i=1,k=1) -> W: [1,0,0,1] (idx k*2+i)
    U2[:, 0] = [1, 0, 0, 1]
    V2[:, 0] = [1, 0, 0, 1]
    W2[:, 0] = [1, 0, 0, 1]
    
    # r=1: m2 = (c+d)e -> C10
    # A: c(1,0), d(1,1) -> U: [0,0,1,1]
    # B: e(0,0) -> V: [1,0,0,0]
    # C: C10(i=1,k=0) -> W: [0,1,0,0]
    U2[:, 1] = [0, 0, 1, 1]
    V2[:, 1] = [1, 0, 0, 0]
    W2[:, 1] = [0, 1, 0, 0]
    
    # r=2: m3 = a(f-h) -> C01
    # A: a(0,0) -> U: [1,0,0,0]
    # B: f(0,1), -h(1,1) -> V: [0,1,0,-1]
    # C: C01(i=0,k=1) -> W: [0,0,1,0]
    U2[:, 2] = [1, 0, 0, 0]
    V2[:, 2] = [0, 1, 0, -1]
    W2[:, 2] = [0, 0, 1, 0]
    
    # r=3: m4 = d(g-e) -> C00, C10
    # A: d(1,1) -> U: [0,0,0,1]
    # B: g(1,0), -e(0,0) -> V: [-1,0,1,0]
    # C: C00(i=0,k=0), C10(i=1,k=0) -> W: [1,1,0,0]
    U2[:, 3] = [0, 0, 0, 1]
    V2[:, 3] = [-1, 0, 1, 0]
    W2[:, 3] = [1, 1, 0, 0]
    
    # r=4: m5 = (a+b)h -> C00, C01
    # A: a(0,0), b(0,1) -> U: [1,1,0,0]
    # B: h(1,1) -> V: [0,0,0,1]
    # C: C00(i=0,k=0), C01(i=0,k=1) -> W: [1,0,1,0]
    U2[:, 4] = [1, 1, 0, 0]
    V2[:, 4] = [0, 0, 0, 1]
    W2[:, 4] = [1, 0, 1, 0]
    
    # r=5: m6 = (c-a)(e+f) -> C11
    # A: c(1,0), -a(0,0) -> U: [-1,0,1,0]
    # B: e(0,0), f(0,1) -> V: [1,1,0,0]
    # C: C11(i=1,k=1) -> W: [0,0,0,1]
    U2[:, 5] = [-1, 0, 1, 0]
    V2[:, 5] = [1, 1, 0, 0]
    W2[:, 5] = [0, 0, 0, 1]
    
    # r=6: m7 = (b-d)(g+h) -> C00, C11
    # A: b(0,1), -d(1,1) -> U: [0,1,0,-1]
    # B: g(1,0), h(1,1) -> V: [0,0,1,1]
    # C: C00(i=0,k=0), C11(i=1,k=1) -> W: [1,0,0,1]
    U2[:, 6] = [0, 1, 0, -1]
    V2[:, 6] = [0, 0, 1, 1]
    W2[:, 6] = [1, 0, 0, 1]
    
    # Kronecker product for 4x4x4
    # Indices i = 2*i1 + i2, j = 2*j1 + j2, k = 2*k1 + k2
    # Rank 49
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            i = 2 * i1 + i2
                            j = 2 * j1 + j2
                            U4[i * 4 + j, r] = U2[i1 * 2 + j1, r1] * U2[i2 * 2 + j2, r2]
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(2):
                            j = 2 * j1 + j2
                            k = 2 * k1 + k2
                            V4[j * 4 + k, r] = V2[j1 * 2 + k1, r1] * V2[j2 * 2 + k2, r2]
            for k1 in range(2):
                for k2 in range(2):
                    for i1 in range(2):
                        for i2 in range(2):
                            k = 2 * k1 + k2
                            i = 2 * i1 + i2
                            W4[k * 4 + i, r] = W2[k1 * 2 + i1, r1] * W2[k2 * 2 + i2, r2]
    
    # --- Part 2: Schoolbook for 4x4x3 ---
    # Rank 48
    U3 = np.zeros((16, 48))
    V3 = np.zeros((16, 48))
    W3 = np.zeros((28, 48)) # p*n = 7*4 = 28? No, p=7, n=4. W is (p*n, rank).
    # Wait, W shape is (p*n, rank). p=7, n=4. So 28 rows.
    # But for the 4x4x3 part, k is in 0..2 (relative to the 3 columns).
    # The global k index is 4..6.
    # W index is k*n + i.
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k_rel in range(3):
                k = 4 + k_rel
                U3[i * 4 + j, col] = 1
                V3[j * 7 + k, col] = 1
                W3[k * 4 + i, col] = 1
                col += 1
                
    # Combine
    U_total = np.hstack([U4, U3])
    V_total = np.hstack([V4, V3])
    W_total = np.hstack([W4, W3])
    
    rank = U_total.shape[1]
    
    # Convert to lists of floats (integers)
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
