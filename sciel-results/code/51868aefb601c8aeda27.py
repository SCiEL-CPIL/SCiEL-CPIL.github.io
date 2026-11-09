import time
import random
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Target: Rank 50 (2 * Rank(3,4,4)=25)
    # We solve for 3x4x4, then duplicate.
    
    def solve_344():
        """Find rank-25 decomposition for 3x4x4."""
        n1, m1, p1 = 3, 4, 4
        T = np.zeros((n1 * m1, m1 * p1, p1 * n1), dtype=int)
        for i in range(n1):
            for j in range(m1):
                for k in range(p1):
                    a = i * m1 + j
                    b = j * p1 + k
                    c = k * n1 + i
                    T[a, b, c] = 1
        
        T_rem = T.copy().astype(float)
        U_cols = []
        V_cols = []
        W_cols = []
        
        target_rank = 25
        
        # Precompute valid (i,j,k) triples
        triples = [(i, j, k) for i in range(n1) for j in range(m1) for k in range(p1)]
        
        for s in range(target_rank):
            if time.time() > deadline - 30:
                return None
            
            if np.max(np.abs(T_rem)) < 1e-6:
                break
            
            best_score = -1
            best_u, best_v, best_w = None, None, None
            
            # Structured search: Pick subsets of I, J, K
            # Try small subsets first
            for trial in range(2000):
                if time.time() > deadline - 30:
                    break
                
                # Randomly select small sets of i, j, k
                # Size 1 or 2
                n_i = random.randint(1, 2)
                n_j = random.randint(1, 2)
                n_k = random.randint(1, 2)
                
                i_set = random.sample(range(n1), n_i)
                j_set = random.sample(range(m1), n_j)
                k_set = random.sample(range(p1), n_k)
                
                u = np.zeros(n1 * m1)
                v = np.zeros(m1 * p1)
                w = np.zeros(p1 * n1)
                
                # Assign random +/- 1 to supports
                for i in i_set:
                    for j in j_set:
                        u[i * m1 + j] = random.choice([-1, 1])
                for j in j_set:
                    for k in k_set:
                        v[j * p1 + k] = random.choice([-1, 1])
                for k in k_set:
                    for i in i_set:
                        w[k * n1 + i] = random.choice([-1, 1])
                
                # Calculate score
                score = 0
                # Only iterate over non-zeros in T_rem that are covered by u,v,w
                for i in i_set:
                    for j in j_set:
                        if u[i * m1 + j] == 0: continue
                        for k in k_set:
                            if v[j * p1 + k] == 0 or w[k * n1 + i] == 0: continue
                            a = i * m1 + j
                            b = j * p1 + k
                            c = k * n1 + i
                            score += u[a] * v[b] * w[c] * T_rem[a, b, c]
                
                if score > best_score:
                    best_score = score
                    best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
            
            if best_u is None or best_score <= 0:
                return None
            
            # Subtract
            for i in i_set:
                for j in j_set:
                    if best_u[i * m1 + j] == 0: continue
                    for k in k_set:
                        if best_v[j * p1 + k] == 0 or best_w[k * n1 + i] == 0: continue
                        a = i * m1 + j
                        b = j * p1 + k
                        c = k * n1 + i
                        T_rem[a, b, c] -= best_u[a] * best_v[b] * best_w[c]
            
            U_cols.append(best_u.tolist())
            V_cols.append(best_v.tolist())
            W_cols.append(best_w.tolist())
        
        if np.max(np.abs(T_rem)) < 1e-6:
            # Convert to U, V, W matrices
            rank = len(U_cols)
            U = [[U_cols[s][a] for s in range(rank)] for a in range(n1 * m1)]
            V = [[V_cols[s][b] for s in range(rank)] for b in range(m1 * p1)]
            W = [[W_cols[s][c] for s in range(rank)] for c in range(p1 * n1)]
            return U, V, W
        return None

    def build_348_from_344(U344, V344, W344):
        """Duplicate 3x4x4 decomposition for 3x4x8."""
        n1, m1, p1 = 3, 4, 4
        n2, m2, p2 = 3, 4, 8
        
        rank = len(U344[0])
        
        U = [[0] * (2 * rank) for _ in range(n2 * m2)]
        V = [[0] * (2 * rank) for _ in range(m2 * p2)]
        W = [[0] * (2 * rank) for _ in range(p2 * n2)]
        
        # First block: k=0..3
        for a in range(n1 * m1):
            for s in range(rank):
                U[a][s] = U344[a][s]
        
        for b in range(m1 * p1):
            # b in 0..15, k = b % 4, j = b // 4
            # In 3x4x8, b is j*8+k. For k=0..3, b = j*8+k.
            # In 3x4x4, b is j*4+k.
            # So for a given b in 3x4x4 (j,k), the corresponding b in 3x4x8 is j*8+k.
            j = b // p1
            k = b % p1
            b_new = j * p2 + k
            for s in range(rank):
                V[b_new][s] = V344[b][s]
        
        for c in range(p1 * n1):
            # c in 0..11, k = c // 3, i = c % 3
            # In 3x4x8, c is k*3+i. Same formula.
            for s in range(rank):
                W[c][s] = W344[c][s]
        
        # Second block: k=4..7
        # Shift k by 4
        for a in range(n1 * m1):
            for s in range(rank):
                U[a][rank + s] = U344[a][s]
        
        for b in range(m1 * p1):
            j = b // p1
            k = b % p1
            k_new = k + 4
            b_new = j * p2 + k_new
            for s in range(rank):
                V[b_new][rank + s] = V344[b][s]
        
        for c in range(p1 * n1):
            k = c // n1
            i = c % n1
            k_new = k + 4
            c_new = k_new * n1 + i
            for s in range(rank):
                W[c_new][rank + s] = W344[c][s]
        
        return U, V, W

    def validate(U, V, W):
        n, m, p = DIMENSIONS
        T = np.zeros((n * m, m * p, p * n), dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1
        
        rank = len(U[0])
        T_check = np.zeros((n * m, m * p, p * n), dtype=float)
        for s in range(rank):
            u = np.array(U, dtype=float)[:, s]
            v = np.array(V, dtype=float)[:, s]
            w = np.array(W, dtype=float)[:, s]
            for a in range(n * m):
                if u[a] == 0: continue
                for b in range(m * p):
                    if v[b] == 0: continue
                    T_check[a, b, :] += u[a] * v[b] * w
        return np.allclose(T_check, T, atol=1e-6)

    def schoolbook():
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    # Attempt to solve 3x4x4
    result_344 = solve_344()
    
    if result_344:
        U344, V344, W344 = result_344
        U, V, W = build_348_from_344(U344, V344, W344)
        if validate(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    # Fallback
    U, V, W = schoolbook()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
