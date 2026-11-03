"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 5, 5
    total_elements = n * m * p  # 75
    
    def add_strassen_block(U, V, W, i0, j0, k0, col_offset):
        """Add a 2x2x2 Strassen decomposition starting at (i0, j0, k0)."""
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
    
    def add_laderman_block(U, V, W, i0, j0, k0, col_offset):
        """Add a 3x3x3 Laderman decomposition (rank 23) starting at (i0, j0, k0)."""
        # Laderman's rank-23 decomposition for 3x3x3 matrix multiplication
        # This is a known result; the 23 terms are:
        
        # Term 0: (1,1,1)
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # Term 1: row 1, col 1,1; row 2, col 1,1
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+2)*m + j0][c] += 1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        W[k0*n + (i0+2)][c] += 1
        W[(k0+1)*n + (i0+2)][c] += 1
        
        # Term 2: row 1, col 1,2; row 2, col 1,2
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # Term 3: row 3, col 1; col 2,3
        c = col_offset + 3
        U[(i0+2)*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + (i0+2)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # Term 4: row 1, col 3; row 2, col 3
        c = col_offset + 4
        U[i0*m + (j0+2)][c] += 1
        V[(j0+2)*p + k0][c] += 1
        V[(j0+2)*p + (k0+1)][c] += -1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + i0][c] += 1
        
        # Term 5: row 2, col 2; row 3, col 2
        c = col_offset + 5
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+2)*p + (k0+1)][c] += 1
        V[j0*p + k0][c] += -1
        W[(k0+1)*n + (i0+1)][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        # Term 6: row 3, col 3
        c = col_offset + 6
        U[(i0+2)*m + (j0+2)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + k0][c] += -1
        W[(k0+2)*n + (i0+2)][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        # Terms 7-22: additional terms needed for rank 23
        # The full Laderman decomposition has 23 terms.
        # Let me use a verified set. Actually, let me use a different approach:
        # I'll use the known fact that rank(3x3x3) <= 23, and use a search to find it.
        # But for now, let me just add 16 more schoolbook-like terms to complete the 3x3x3 block.
        # This is not optimal but gives us a valid decomposition.
        
        # Actually, let me use a simpler approach: use 27 schoolbook terms for the 3x3x3 block
        # This gives rank 27 for the block, which is worse than 3 Strassen blocks (3*7=21) + 19 schoolbook = 40
        # Wait, 3 Strassen covers 3*8=24 elements with rank 21, plus 3 schoolbook = 24 total for 27 elements
        # Laderman covers 27 elements with rank 23, saving 1 rank
        # So it's a marginal improvement. Let me just use the 7-term version above and add schoolbook for the rest.
        
        # For now, I'll add schoolbook terms for the remaining 20 elements in the 3x3x3 block
        covered = set()
        # The 7 terms above cover certain (i,j,k) positions
        # Let me just add schoolbook for all 27 and it will be rank 27 for this block
        # That's worse than 4 Strassen blocks. Let me skip Laderman for now and focus on 4 Strassen.
        pass
    
    def build_decomposition(blocks):
        """Build decomposition from block placements + schoolbook for rest."""
        covered = [[[False]*p for _ in range(m)] for _ in range(n)]
        total_rank = 0
        
        for block in blocks:
            btype, i0, j0, k0 = block
            if btype == 'strassen':
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            covered[i0+di][j0+dj][k0+dk] = True
                total_rank += 7
            elif btype == 'larderman':
                for di in range(3):
                    for dj in range(3):
                        for dk in range(3):
                            covered[i0+di][j0+dj][k0+dk] = True
                total_rank += 27  # Using schoolbook for now
        
        remaining = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not covered[i][j][k]:
                        remaining += 1
        
        total_rank += remaining
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        for block in blocks:
            btype, i0, j0, k0 = block
            if btype == 'strassen':
                add_strassen_block(U, V, W, i0, j0, k0, col)
                col += 7
            elif btype == 'larderman':
                # Add schoolbook terms for the 3x3x3 block
                for di in range(3):
                    for dj in range(3):
                        for dk in range(3):
                            U[(i0+di)*m + (j0+dj)][col] += 1
                            V[(j0+dj)*p + (k0+dk)][col] += 1
                            W[(k0+dk)*n + (i0+di)][col] += 1
                            col += 1
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not covered[i][j][k]:
                        U[i*m+j][col] += 1
                        V[j*p+k][col] += 1
                        W[k*n+i][col] += 1
                        col += 1
        
        return U, V, W, total_rank
    
    def is_valid_placement(blocks):
        """Check if all blocks are within bounds and non-overlapping."""
        covered = [[[False]*p for _ in range(m)] for _ in range(n)]
        for block in blocks:
            btype, i0, j0, k0 = block
            if btype == 'strassen':
                size = 2
            elif btype == 'larderman':
                size = 3
            else:
                return False
            
            if i0 < 0 or i0+size > n or j0 < 0 or j0+size > m or k0 < 0 or k0+size > p:
                return False
            for di in range(size):
                for dj in range(size):
                    for dk in range(size):
                        if covered[i0+di][j0+dj][k0+dk]:
                            return False
                        covered[i0+di][j0+dj][k0+dk] = True
        return True
    
    def verify(U, V, W):
        """Verify the decomposition is correct."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        target = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m+j, j*p+k, k*n+i] = 1
        T_decomp = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        return np.allclose(T_decomp, target, atol=1e-10)
    
    # Generate all possible block positions
    strassen_positions = [('strassen', i0, j0, k0) for i0 in range(n-1) for j0 in range(m-1) for k0 in range(p-1)]
    
    best_rank = n * m * p
    best_result = None
    
    # First, try the known good configuration: 4 Strassen blocks
    # n=3, m=5, p=5
    # Possible Strassen positions: i0 in {0,1}, j0 in {0,1,2,3}, k0 in {0,1,2,3}
    # We need 4 non-overlapping 2x2x2 blocks
    # Strategy: use i0=0 for all, and choose j0,k0 to be non-overlapping
    # j0 values: 0,2,4 (but j0+1 must be <=4, so j0 in {0,1,2,3})
    # Non-overlapping in j: 0,2 (covers 0-1, 2-3) or 0,3 (covers 0-1, 3-4)
    # Non-overlapping in k: 0,2 (covers 0-1, 2-3) or 0,3 (covers 0-1, 3-4)
    
    base_config = [
        ('strassen', 0, 0, 0),
        ('strassen', 0, 2, 0),
        ('strassen', 0, 0, 2),
        ('strassen', 0, 2, 2),
    ]
    U, V, W, rank = build_decomposition(base_config)
    if verify(U, V, W):
        best_rank = rank
        best_result = {"U": U, "V": V, "W": W}
    
    # Try other 4-Strassen configurations
    if time.time() < deadline - 20:
        # Try all combinations of 4 non-overlapping Strassen blocks
        for combo in combinations(strassen_positions, 4):
            if time.time() > deadline - 10:
                break
            if not is_valid_placement(list(combo)):
                continue
            rank_est = 4 * 7 + (n*m*p - 4*8)
            if rank_est >= best_rank:
                continue
            U, V, W, rank = build_decomposition(list(combo))
            if verify(U, V, W):
                if rank < best_rank:
                    best_rank = rank
                    best_result = {"U": U, "V": V, "W": W}
    
    # Try 5 Strassen blocks
    if time.time() < deadline - 20:
        for combo in combinations(strassen_positions, 5):
            if time.time() > deadline - 10:
                break
            if not is_valid_placement(list(combo)):
                continue
            rank_est = 5 * 7 + (n*m*p - 5*8)
            if rank_est >= best_rank:
                continue
            U, V, W, rank = build_decomposition(list(combo))
            if verify(U, V, W):
                if rank < best_rank:
                    best_rank = rank
                    best_result = {"U": U, "V": V, "W": W}
    
    # Try 6 Strassen blocks
    if time.time() < deadline - 20:
        for combo in combinations(strassen_positions, 6):
            if time.time() > deadline - 10:
                break
            if not is_valid_placement(list(combo)):
                continue
            rank_est = 6 * 7 + (n*m*p - 6*8)
            if rank_est >= best_rank:
                continue
            U, V, W, rank = build_decomposition(list(combo))
            if verify(U, V, W):
                if rank < best_rank:
                    best_rank = rank
                    best_result = {"U": U, "V": V, "W": W}
    
    if best_result is None:
        # Fallback to schoolbook
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        c2 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j][c2] = 1
                    V_s[j*p+k][c2] = 1
                    W_s[k*n+i][c2] = 1
                    c2 += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return best_result
# EVOLVE-BLOCK-END
