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
    
    def build_decomposition(strassen_blocks):
        """Build decomposition from Strassen block placements + schoolbook for rest."""
        covered = set()
        for i0, j0, k0 in strassen_blocks:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered.add((i0+di, j0+dj, k0+dk))
        
        remaining = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        remaining += 1
        
        total_rank = len(strassen_blocks) * 7 + remaining
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        for i0, j0, k0 in strassen_blocks:
            add_strassen_block(U, V, W, i0, j0, k0, col)
            col += 7
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        U[i*m+j][col] += 1
                        V[j*p+k][col] += 1
                        W[k*n+i][col] += 1
                        col += 1
        
        return U, V, W, total_rank
    

    def verify(U, V, W):
        """Verify the decomposition is correct - optimized version."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        # For each rank-1 term, check that it contributes only to valid (i,j,k) positions
        # and that all required positions are covered exactly once
        # More efficient: compute the full tensor and compare
        # But we can short-circuit: check that each column has at most one nonzero in each matrix
        # Actually, let's just do the einsum but with a smaller target
        target = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m+j, j*p+k, k*n+i] = 1
        T_decomp = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        return np.allclose(T_decomp, target, atol=1e-10)
    
    # Exhaustive search over all valid non-overlapping 2x2x2 block placements
    # Possible positions: i0 in {0,1}, j0 in {0,1,2,3}, k0 in {0,1,2,3}
    all_positions = [(i0, j0, k0) for i0 in range(n-1) for j0 in range(m-1) for k0 in range(p-1)]
    
    best_rank = n * m * p
    best_result = None
    
    # First, try the known good configuration
    base_config = [(0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 2)]
    U, V, W, rank = build_decomposition(base_config)
    if verify(U, V, W):
        best_rank = rank
        best_result = {"U": U, "V": V, "W": W}
    
    # Precompute a bitmask for each position to speed up overlap checking
    # Each position covers 8 cells. We can represent coverage as a bitmask of 75 bits.
    # But 75 bits doesn't fit in a Python int easily... actually it does, Python ints are arbitrary precision.
    
    # Better: use a set of covered cells for quick overlap check
    def position_cells(pos):
        i0, j0, k0 = pos
        cells = set()
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    cells.add((i0+di, j0+dj, k0+dk))
        return cells
    
    pos_cells = {pos: position_cells(pos) for pos in all_positions}
    
    def is_valid_fast(combo):
        """Fast overlap check using precomputed cell sets."""
        covered = set()
        for pos in combo:
            cells = pos_cells[pos]
            if covered & cells:
                return False
            covered |= cells
        return True
    
    # Try size-5 subsets first (best potential rank = 70)
    if time.time() < deadline - 20:
        for combo in combinations(all_positions, 5):
            if time.time() > deadline - 15:
                break
            if not is_valid_fast(combo):
                continue
            rank_est = 5 * 7 + (n*m*p - 5*8)
            if rank_est >= best_rank:
                continue
            U, V, W, rank = build_decomposition(list(combo))
            if verify(U, V, W):
                if rank < best_rank:
                    best_rank = rank
                    best_result = {"U": U, "V": V, "W": W}
                # We found a valid 5-block config; no need to search further for 5-block
                break
    
    # Try size-4 subsets (rank = 71)
    if time.time() < deadline - 15:
        for combo in combinations(all_positions, 4):
            if time.time() > deadline - 10:
                break
            if not is_valid_fast(combo):
                continue
            rank_est = 4 * 7 + (n*m*p - 4*8)
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
    
    # Rank-merging heuristic: try to find linear dependencies in the current decomposition
    # and exploit them to reduce rank
    if best_result is not None and time.time() < deadline - 15:
        U_np = np.array(best_result["U"], dtype=float)
        V_np = np.array(best_result["V"], dtype=float)
        W_np = np.array(best_result["W"], dtype=float)
        current_rank = U_np.shape[1]
        
        # Try to find pairs of rank-1 terms that can be merged
        # A pair (r1, r2) can be merged if there exist scalars a, b, c such that
        # U[:,r1] + a*U[:,r2] = U_merged, etc., and the tensor still decomposes
        # This is equivalent to finding if the 2D subspace spanned by the two terms
        # has a rank-1 structure in the tensor product
        
        # More practically: try random linear combinations of pairs
        # and check if the resulting matrix has lower "effective rank"
        
        # For now, try a simple approach: for each pair of columns, try to see if
        # replacing both with their sum and adjusting other terms works
        # This is heuristic and may not find improvements
        
        # Alternative: use SVD on the "interaction" matrices to find structure
        # But this is complex and time-consuming
        
        # For this iteration, just verify the result is valid and return it
        pass
    
    return best_result
# EVOLVE-BLOCK-END
