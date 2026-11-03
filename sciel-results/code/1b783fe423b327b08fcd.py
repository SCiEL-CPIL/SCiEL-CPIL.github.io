import numpy as np
import time
import itertools
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
    
    # Strategy: Place Strassen blocks on overlapping 2x2x2 subcubes,
    # then fill remaining positions with schoolbook, then try to
    # greedily merge pairs of schoolbook terms into one rank-1 term
    # if their tensor support allows (i.e., the sum of two rank-1 tensors
    # has rank 1, meaning they share the same "direction" in some sense).
    #
    # Actually, two rank-1 tensors u⊗v⊗w and u'⊗v'⊗w' sum to a rank-1 tensor
    # iff they are scalar multiples of each other (in the generic case).
    # So we can't simply merge two arbitrary schoolbook terms.
    #
    # Better approach: Use a different set of Strassen blocks that cover more positions.
    # Or use a combination of Strassen-like identities on larger blocks.
    #
    # Let's try: 5 Strassen blocks placed to cover maximum positions.
    # The 3x5x5 tensor has 75 positions. Each 2x2x2 block covers 8 positions.
    # With 4 blocks covering disjoint 2x2x2 regions: 32 covered, rank = 4*7 + 43 = 71.
    #
    # What if we use overlapping blocks? The Strassen decomposition of a 2x2x2 block
    # is specific to that block. If blocks overlap, the rank-1 terms interfere.
    #
    # Alternative: Try to find a decomposition with rank < 71 using a different structure.
    # Known results: R(3x5x5) is not well-known in literature.
    #
    # Let me try a different approach: use Strassen on a 2x2x2 block for the "core"
    # and then use a more clever decomposition for the remaining terms.
    #
    # Actually, let's try: what if we can cover 5 2x2x2 blocks?
    # That would be 40 positions covered with rank 35, leaving 35 schoolbook terms.
    # Total rank = 35 + 35 = 70.
    #
    # Can we place 5 disjoint 2x2x2 blocks in a 3x5x5 grid?
    # In dimension i (size 3): at most 1 block (positions 0-1 or 1-2)
    # In dimension j (size 5): at most 2 blocks (0-1, 2-3 or 1-2, 3-4)
    # In dimension k (size 5): at most 2 blocks
    # Max disjoint 2x2x2 blocks: 1 * 2 * 2 = 4. So we can't get 5 disjoint blocks.
    #
    # What about using a 2x2x2 Strassen block and a different kind of decomposition
    # for the remaining 43 terms that achieves rank < 43?
    #
    # The 43 remaining terms form a tensor with certain structure. Let me analyze
    # which positions are NOT covered by the 4 Strassen blocks.
    #
    # With blocks at (0,0,0), (0,2,0), (0,0,2), (0,2,2):
    # Covered i-indices: {0,1} for all blocks (since i0=0)
    # So i=2 is never covered.
    # 
    # Let me try blocks at different i positions.
    #
    # Block positions (i0, j0, k0) where i0 ∈ {0,1}, j0 ∈ {0,1,2,3}, k0 ∈ {0,1,2,3}
    # such that blocks are disjoint in 3D.
    
    def build_decomposition(blocks):
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
    
    # Try various block configurations
    best_result = None
    best_rank = 999
    
    # Generate candidate block configurations
    # i0 can be 0 or 1
    # j0 can be 0,1,2,3
    # k0 can be 0,1,2,3
    
    candidate_configs = []
    
    # Configuration 1: Current best (4 blocks, rank 71)
    config1 = [(0,0,0), (0,2,0), (0,0,2), (0,2,2)]
    candidate_configs.append(config1)
    
    # Configuration 2: Shift blocks to cover i=1,2
    config2 = [(1,0,0), (1,2,0), (1,0,2), (1,2,2)]
    candidate_configs.append(config2)
    
    # Configuration 3: Mixed i positions
    config3 = [(0,0,0), (0,2,0), (1,0,2), (1,2,2)]
    candidate_configs.append(config3)
    
    # Configuration 4: Spread across j and k more
    config4 = [(0,0,0), (0,1,1), (1,2,2), (1,3,3)]
    # Check if disjoint
    for b1 in config4:
        for b2 in config4:
            if b1 != b2:
                if not (abs(b1[0]-b2[0]) >= 2 or abs(b1[1]-b2[1]) >= 2 or abs(b1[2]-b2[2]) >= 2):
                    pass  # They overlap, skip this config
    
    # Only add disjoint configs
    def check_disjoint(blocks):
        for i in range(len(blocks)):
            for j in range(i+1, len(blocks)):
                i1, j1, k1 = blocks[i]
                i2, j2, k2 = blocks[j]
                # Check if 2x2x2 blocks overlap
                if (abs(i1-i2) < 2 and abs(j1-j2) < 2 and abs(k1-k2) < 2):
                    return False
        return True
    
    valid_configs = []
    for cfg in candidate_configs:
        if check_disjoint(cfg):
            valid_configs.append(cfg)
    
    # Also try some random configurations
    import random
    random.seed(42)
    for _ in range(50):
        # Try to place 4 disjoint blocks
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
    
    # Evaluate each config
    for cfg in valid_configs:
        elapsed = time.time() - _START_TIME
        if elapsed > _DEADLINE - 5:
            break
        
        U, V, W, rank = build_decomposition(cfg)
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
        return {"U": U, "V": V, "W": W}
    
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
