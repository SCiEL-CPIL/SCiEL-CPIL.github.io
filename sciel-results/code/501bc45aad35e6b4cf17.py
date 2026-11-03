import numpy as np
import time
from collections import defaultdict

DIMENSIONS = [3, 5, 5]
_START_TIME = None
_DEADLINE = 145.0

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    n, m, p = DIMENSIONS
    
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE:
        # Emergency fallback: schoolbook rank 75
        U_fb = [[0.0] * (n*m*p) for _ in range(n*m)]
        V_fb = [[0.0] * (n*m*p) for _ in range(m*p)]
        W_fb = [[0.0] * (n*m*p) for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][col] = 1.0
                    V_fb[j*p+k][col] = 1.0
                    W_fb[k*n+i][col] = 1.0
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Strategy: Use Strassen blocks on 2x2x2 subcubes, then attempt to
    # find additional rank reductions by exploiting linear dependencies
    # in the remaining schoolbook terms.
    #
    # Key insight: After placing Strassen blocks, the remaining terms form
    # a structured tensor. We can try to find rank-1 tensors that are
    # linear combinations of the remaining schoolbook terms, effectively
    # replacing multiple schoolbook terms with fewer rank-1 terms.
    
    def build_strassen_decomposition(blocks):
        """Build U, V, W for given Strassen block positions."""
        total_strassen_rank = len(blocks) * 7
        total_terms = n * m * p
        strassen_terms_count = len(blocks) * 8
        schoolbook_terms_count = total_terms - strassen_terms_count
        total_rank = total_strassen_rank + schoolbook_terms_count
        
        U = [[0.0] * total_rank for _ in range(n * m)]
        V = [[0.0] * total_rank for _ in range(m * p)]
        W = [[0.0] * total_rank for _ in range(p * n)]
        
        col = 0
        
        def add_strassen_block(i0, j0, k0, col_offset):
            c = col_offset + 0
            U[i0*m + j0][c] += 1
            U[(i0+1)*m + (j0+1)][c] += 1
            V[j0*p + k0][c] += 1
            V[(j0+1)*p + (k0+1)][c] += 1
            W[k0*n + i0][c] += 1
            W[(k0+1)*n + (i0+1)][c] += 1
            
            c = col_offset + 1
            U[(i0+1)*m + j0][c] += 1
            U[(i0+1)*m + (j0+1)][c] += 1
            V[j0*p + k0][c] += 1
            W[k0*n + (i0+1)][c] += 1
            W[(k0+1)*n + (i0+1)][c] += -1
            
            c = col_offset + 2
            U[i0*m + j0][c] += 1
            V[j0*p + (k0+1)][c] += 1
            V[(j0+1)*p + (k0+1)][c] += -1
            W[(k0+1)*n + i0][c] += 1
            W[(k0+1)*n + (i0+1)][c] += 1
            
            c = col_offset + 3
            U[(i0+1)*m + (j0+1)][c] += 1
            V[(j0+1)*p + k0][c] += 1
            V[j0*p + k0][c] += -1
            W[k0*n + i0][c] += 1
            W[k0*n + (i0+1)][c] += 1
            
            c = col_offset + 4
            U[i0*m + j0][c] += 1
            U[i0*m + (j0+1)][c] += 1
            V[(j0+1)*p + (k0+1)][c] += 1
            W[k0*n + i0][c] += -1
            W[(k0+1)*n + i0][c] += 1
            
            c = col_offset + 5
            U[(i0+1)*m + j0][c] += 1
            U[i0*m + j0][c] += -1
            V[j0*p + k0][c] += 1
            V[j0*p + (k0+1)][c] += 1
            W[(k0+1)*n + (i0+1)][c] += 1
            
            c = col_offset + 6
            U[i0*m + (j0+1)][c] += 1
            U[(i0+1)*m + (j0+1)][c] += -1
            V[(j0+1)*p + k0][c] += 1
            V[(j0+1)*p + (k0+1)][c] += 1
            W[k0*n + i0][c] += 1

        for i0, j0, k0 in blocks:
            add_strassen_block(i0, j0, k0, col)
            col += 7
        
        covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
        for i0, j0, k0 in blocks:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered_3d[i0+di][j0+dj][k0+dk] = True
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not covered_3d[i][j][k]:
                        U[i*m+j][col] += 1
                        V[j*p+k][col] += 1
                        W[k*n+i][col] += 1
                        col += 1
        
        return U, V, W, total_rank
    
    def check_disjoint(blocks):
        for i in range(len(blocks)):
            for j in range(i+1, len(blocks)):
                i1, j1, k1 = blocks[i]
                i2, j2, k2 = blocks[j]
                if (abs(i1-i2) < 2 and abs(j1-j2) < 2 and abs(k1-k2) < 2):
                    return False
        return True
    
    # Try various block configurations
    best_result = None
    best_rank = 999
    
    candidate_configs = [
        [(0,0,0), (0,2,0), (0,0,2), (0,2,2)],
        [(1,0,0), (1,2,0), (1,0,2), (1,2,2)],
        [(0,0,0), (0,2,0), (1,0,2), (1,2,2)],
        [(0,0,0), (1,0,0), (0,2,2), (1,2,2)],
        [(0,0,0), (0,1,1), (1,2,2), (1,3,3)],
        [(0,0,0), (0,0,2), (1,2,0), (1,2,2)],
        [(0,1,0), (0,3,0), (1,1,2), (1,3,2)],
    ]
    
    valid_configs = []
    for cfg in candidate_configs:
        if check_disjoint(cfg):
            valid_configs.append(cfg)
    
    import random
    random.seed(123)
    for _ in range(80):
        blocks = []
        for attempt in range(100):
            i0 = random.choice([0, 1])
            j0 = random.randint(0, 3)
            k0 = random.randint(0, 3)
            new_block = (i0, j0, k0)
            if all(not (abs(i0-b[0]) < 2 and abs(j0-b[1]) < 2 and abs(k0-b[2]) < 2) for b in blocks):
                blocks.append(new_block)
                if len(blocks) == 4:
                    break
        if len(blocks) == 4 and check_disjoint(blocks):
            if blocks not in valid_configs:
                valid_configs.append(blocks)
    
    for cfg in valid_configs:
        elapsed = time.time() - _START_TIME
        if elapsed > _DEADLINE - 5:
            break
        
        U, V, W, rank = build_strassen_decomposition(cfg)
        if rank < best_rank:
            best_rank = rank
            best_result = (U, V, W, rank)
    
    if best_result is not None:
        U, V, W, rank = best_result
        elapsed = time.time() - _START_TIME
        if elapsed > _DEADLINE:
            U_fb = [[0.0] * (n*m*p) for _ in range(n*m)]
            V_fb = [[0.0] * (n*m*p) for _ in range(m*p)]
            W_fb = [[0.0] * (n*m*p) for _ in range(p*n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i*m+j][col] = 1.0
                        V_fb[j*p+k][col] = 1.0
                        W_fb[k*n+i][col] = 1.0
                        col += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}
        return {"U": U, "V": W, "W": W}
    
    # Fallback: schoolbook
    U_fb = [[0.0] * (n*m*p) for _ in range(n*m)]
    V_fb = [[0.0] * (n*m*p) for _ in range(m*p)]
    W_fb = [[0.0] * (n*m*p) for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j][col] = 1.0
                V_fb[j*p+k][col] = 1.0
                W_fb[k*n+i][col] = 1.0
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}

if __name__ == "__main__":
    _START_TIME = time.time()
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
