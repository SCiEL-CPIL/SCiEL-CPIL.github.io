import random
import time
import sys

def construct():
    # Deadline guard
    start_time = time.time()
    DEADLINE = 110.0

    n, m, p = 2, 4, 8

    # Strassen coefficients for 2x2 matrix multiplication (rank 7)
    # A = [[a0, a1], [a2, a3]], B = [[b0, b1], [b2, b3]]
    # m0 = (a0+a3)(b0+b3)
    # m1 = (a2+a3)b0
    # m2 = a0(b1-b2)
    # m3 = a3(b2+b3)
    # m4 = (a0+a1)b2
    # m5 = (a1-a2)(b0+b1)
    # m6 = (a3-a2)(b3-b2)
    
    # Linear forms for A (2x2)
    # Index: i*2 + j
    X_forms = [
        [1, 0, 0, 1],   # m0: a0 + a3
        [0, 0, 1, 1],   # m1: a2 + a3
        [1, 0, 0, 0],   # m2: a0
        [0, 0, 0, 1],   # m3: a3
        [1, 1, 0, 0],   # m4: a0 + a1
        [-1, 0, 1, 0],  # m5: a1 - a2  (Note: Standard Strassen m5 is (a1-a2) or (a2-a1)? 
                        # Let's check C11 = m0 - m1 + m2 + m5.
                        # C11 = a3b3.
                        # m0-m1+m2+m5 = (a0+a3)(b0+b3) - (a2+a3)b0 + a0(b1-b2) + (a1-a2)(b0+b1)
                        # This needs to equal a3b3? No, C11 is bottom-right.
                        # Let's stick to the verified forms from the seed.
                        # Seed used:
                        # X: [-1, 0, 1, 0] -> -a0 + a2? No, index 1 is a1, index 2 is a2.
                        # [0, -1, 1, 0] would be a2 - a1.
                        # The seed had [-1, 0, 1, 0] which is -a0 + a2?
                        # Wait, X_forms[5] = [-1, 0, 1, 0].
                        # Index 0: a0, 1: a1, 2: a2, 3: a3.
                        # So -a0 + a2.
                        # And Y_forms[5] = [1, 1, 0, 0] -> b0 + b1.
                        # Product: (-a0 + a2)(b0 + b1).
                        # Let's re-verify the seed's correctness assumption.
                        # The seed claimed it was verified. I will trust the seed's forms.
        [0, 1, 0, -1]   # m6: a1 - a3? Seed had [0, 1, 0, -1] -> a1 - a3.
    ]
    
    # Correcting X_forms based on standard Strassen to be safe, or sticking to seed if it passed.
    # Seed passed with score 1.1429.
    # Let's use the exact forms from the seed to ensure correctness.
    X_forms = [
        [1, 0, 0, 1],   # m0
        [0, 0, 1, 1],   # m1
        [1, 0, 0, 0],   # m2
        [0, 0, 0, 1],   # m3
        [1, 1, 0, 0],   # m4
        [-1, 0, 1, 0],  # m5
        [0, 1, 0, -1]   # m6
    ]
    
    Y_forms = [
        [1, 0, 0, 1],   # m0
        [1, 0, 0, 0],   # m1
        [0, 1, 0, -1],  # m2
        [-1, 0, 1, 0],  # m3
        [0, 0, 0, 1],   # m4
        [1, 1, 0, 0],   # m5
        [0, 0, 1, 1]    # m6
    ]
    
    Z_combos = [
        [1, 0, 0, 1, -1, 0, 1],   # z00
        [0, 0, 1, 0, 1, 0, 0],    # z01
        [0, 1, 0, 1, 0, 0, 0],    # z10
        [1, -1, 1, 0, 0, 1, 0]    # z11
    ]
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]   # (8, 56)
    V = [[0] * rank for _ in range(m * p)]   # (32, 56)
    W = [[0] * rank for _ in range(p * n)]   # (16, 56)
    
    r_idx = 0
    # 4 groups of columns in B (each 4x2)
    for g in range(4):
        # 2 blocks of rows in A (each 2x2)
        for b in range(2):
            # 7 Strassen terms
            for s in range(7):
                # Fill U: A[i, j] where i in [0,2), j = b*2 + jp
                for i in range(n):
                    for jp in range(2):
                        a_idx = i * m + b * 2 + jp
                        U[a_idx][r_idx] = X_forms[s][i * 2 + jp]
                
                # Fill V: B[j, k] where j = b*2 + jp, k = 2*g + kp
                for jp in range(2):
                    for kp in range(2):
                        b_idx = (b * 2 + jp) * p + (2 * g + kp)
                        V[b_idx][r_idx] = Y_forms[s][jp * 2 + kp]
                
                # Fill W: C[k, i] where k = 2*g + kp, i in [0,2)
                for i in range(n):
                    for kp in range(2):
                        k = 2 * g + kp
                        c_idx = k * n + i
                        W[c_idx][r_idx] = Z_combos[i * 2 + kp][s]
                
                r_idx += 1
                
    return {"U": U, "V": V, "W": W}

if __name__ == '__main__':
    res = construct()
    U = res['U']
    V = res['V']
    W = res['W']
    
    n, m, p = 2, 4, 8
    assert len(U) == n*m, "U shape error"
    assert len(U[0]) == len(V[0]), "Rank mismatch U V"
    assert len(V) == m*p, "V shape error"
    assert len(W) == p*n, "W shape error"
    assert len(W[0]) == len(U[0]), "Rank mismatch W U"
    
    import random
    random.seed(42)
    A = [[random.randint(-3, 3) for _ in range(m)] for _ in range(n)]
    B = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]
    
    C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
    
    C_prime = [[0] * p for _ in range(n)]
    for r in range(len(U[0])):
        u_val = sum(U[i*m + j][r] * A[i][j] for i in range(n) for j in range(m))
        v_val = sum(V[j*p + k][r] * B[j][k] for j in range(m) for k in range(p))
        prod = u_val * v_val
        for k in range(p):
            for i in range(n):
                C_prime[i][k] += W[k*n + i][r] * prod
    
    for i in range(n):
        for k in range(p):
            if C[i][k] != C_prime[i][k]:
                print(f"Mismatch at C[{i}][{k}]: {C[i][k]} vs {C_prime[i][k]}")
                sys.exit(1)
    
    print(f"Verified correct. Rank = {len(U[0])}")
