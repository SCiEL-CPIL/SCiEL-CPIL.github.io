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
        # strassen_blocks is a list of (i0, j0, k0)
        # Check for overlaps in 3D
        covered = [[[False]*p for _ in range(m)] for _ in range(n)]
        for i0, j0, k0 in strassen_blocks:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        if i0+di >= n or j0+dj >= m or k0+dk >= p:
                            return None
                        if covered[i0+di][j0+dj][k0+dk]:
                            return None
                        covered[i0+di][j0+dj][k0+dk] = True
        
        remaining = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not covered[i][j][k]:
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
                    if not covered[i][j][k]:
                        U[i*m+j][col] += 1
                        V[j*p+k][col] += 1
                        W[k*n+i][col] += 1
                        col += 1
        
        return U, V, W, total_rank

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

    # Generate all possible 2x2x2 block positions
    # i0 in {0,1}, j0 in {0,1,2,3}, k0 in {0,1,2,3}
    all_positions = [(i0, j0, k0) for i0 in range(n-1) for j0 in range(m-1) for k0 in range(p-1)]
    
    best_rank = n * m * p
    best_result = None
    
    # Try configurations with up to 6 blocks
    # We prioritize more blocks as they reduce rank
    # Max possible is 6 (if packing allows)
    
    # Pre-filter: find all valid sets of size 4, 5, 6
    # To save time, we just iterate combinations and check validity
    
    # Start with size 6
    for size in [6, 5, 4]:
        if time.time() > deadline - 20:
            break
            
        rank_est = size * 7 + (n*m*p - size*8)
        if rank_est >= best_rank:
            continue
            
        # Optimization: If we found a valid set of this size, we can stop
        # because smaller sizes will have higher rank
        found_valid = False
        
        # Iterate combinations
        # C(32, 6) is ~ 900k, might be slow.
        # We can prune: if two blocks overlap in j,k, they might still be valid if i is different?
        # No, if they overlap in j,k and i, they are invalid.
        # If they overlap in j,k but not i, they are valid.
        
        # Simple check is fast enough for 32 choose 6?
        # 900k * 6*8 checks = ~43M ops. Might be tight in 120s.
        # Let's limit to 5 if 6 is too slow, or optimize.
        
        # Actually, we can just try the known good 4 and then try to add more.
        # Or use a greedy approach to find a large set.
        
        # Let's try a greedy max independent set on the conflict graph of positions.
        # Conflict: two positions conflict if their 2x2x2 boxes overlap.
        
        # For now, stick to exhaustive search for size 4, 5, 6 but with time limit.
        
        for combo in combinations(all_positions, size):
            if time.time() > deadline - 10:
                break
            
            valid = True
            covered = [[[False]*p for _ in range(m)] for _ in range(n)]
            for i0, j0, k0 in combo:
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            if covered[i0+di][j0+dj][k0+dk]:
                                valid = False
                                break
                            covered[i0+di][j0+dj][k0+dk] = True
                        if not valid: break
                    if not valid: break
                if not valid: break
            
            if valid:
                U, V, W, rank = build_decomposition(list(combo))
                if U is not None and verify(U, V, W):
                    if rank < best_rank:
                        best_rank = rank
                        best_result = {"U": U, "V": V, "W": W}
                        found_valid = True
                        # Since we are iterating sizes descending, 
                        # and rank decreases with size, 
                        # we can break after finding one valid set of this size?
                        # No, there might be a valid set of size 6 with rank 69, 
                        # but we found one of size 6. All size 6 have same rank.
                        # So we can break the inner loop?
                        # Yes, rank depends only on size for this construction.
                        break
        
        if found_valid:
            break # Move to next size (smaller) only if we want to check? 
                   # No, we want the max size. So if we found a valid size 6, we are done with search for higher sizes.
                   # We can break the size loop.

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
