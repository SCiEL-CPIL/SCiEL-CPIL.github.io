"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    
    # Helper to add a Strassen 2x2x2 block
    def add_strassen(U, V, W, i0, j0, k0, col):
        # M1
        U[i0 * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M2
        U[(i0 + 1) * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] -= 1
        col += 1
        
        # M3
        U[i0 * m + j0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] -= 1
        W[(k0 + 1) * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M4
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0][col] += 1
        V[j0 * p + k0][col] -= 1
        W[k0 * n + i0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        col += 1
        
        # M5
        U[i0 * m + j0][col] += 1
        U[i0 * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0][col] += 1
        W[k0 * n + i0][col] -= 1
        col += 1
        
        # M6
        U[(i0 + 1) * m + j0][col] += 1
        U[i0 * m + j0][col] -= 1
        V[j0 * p + k0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M7
        U[i0 * m + j0 + 1][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] -= 1
        V[(j0 + 1) * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        col += 1
        
        return col

    # Helper to add schoolbook terms for a sub-block
    def add_schoolbook(U, V, W, i_range, j_range, k_range, col):
        for i in i_range:
            for j in j_range:
                for k in k_range:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return col

    # Greedy search for maximum number of disjoint Strassen blocks
    # Each block uses 2 consecutive i's, 2 consecutive j's, 2 consecutive k's
    # Two blocks are disjoint if they don't share any (i,j,k) tensor position
    
    def find_max_strassen_blocks():
        """Find maximum set of disjoint 2x2x2 Strassen blocks via greedy with backtracking."""
        # All possible block positions: (i0, j0, k0) where i0 in [0,1], j0 in [0,3], k0 in [0,5]
        all_blocks = []
        for i0 in range(n - 1):
            for j0 in range(m - 1):
                for k0 in range(p - 1):
                    all_blocks.append((i0, j0, k0))
        
        # For each block, determine which tensor positions it covers
        def block_positions(i0, j0, k0):
            positions = set()
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        positions.add((i0 + di, j0 + dj, k0 + dk))
            return positions
        
        # Precompute positions for each block
        block_pos = {b: block_positions(*b) for b in all_blocks}
        
        # Greedy with multiple orderings to find good packing
        best_packing = []
        
        # Try different orderings: sort by different criteria
        orderings = [
            sorted(all_blocks, key=lambda b: (b[0], b[1], b[2])),
            sorted(all_blocks, key=lambda b: (b[2], b[1], b[0])),
            sorted(all_blocks, key=lambda b: (b[1], b[0], b[2])),
            sorted(all_blocks, key=lambda b: (-b[0], -b[1], -b[2])),
            sorted(all_blocks, key=lambda b: (-b[2], -b[1], -b[0])),
        ]
        
        for ordering in orderings:
            used_positions = set()
            packing = []
            for b in ordering:
                if block_pos[b].isdisjoint(used_positions):
                    packing.append(b)
                    used_positions |= block_pos[b]
            if len(packing) > len(best_packing):
                best_packing = packing
        
        # Try a more sophisticated greedy: prioritize blocks that cover "scarce" positions
        # Positions near edges are more constrained
        def scarcity_score(b):
            i0, j0, k0 = b
            # Count how many other blocks share positions with this one
            shared = 0
            for b2 in all_blocks:
                if b != b2 and not block_pos[b].isdisjoint(block_pos[b2]):
                    shared += 1
            return -shared  # More shared = more scarce = higher priority
        
        ordering = sorted(all_blocks, key=scarcity_score)
        used_positions = set()
        packing = []
        for b in ordering:
            if block_pos[b].isdisjoint(used_positions):
                packing.append(b)
                used_positions |= block_pos[b]
        if len(packing) > len(best_packing):
            best_packing = packing
        
        return best_packing
    
    # Find the best Strassen block packing
    strassen_blocks = find_max_strassen_blocks()
    
    # Determine which positions are covered by Strassen blocks
    covered_positions = set()
    for (i0, j0, k0) in strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered_positions.add((i0 + di, j0 + dj, k0 + dk))
    
    # Uncovered positions need schoolbook terms
    uncovered_positions = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered_positions:
                    uncovered_positions.add((i, j, k))
    
    # Rank = 7 * (# Strassen blocks) + (# uncovered positions)
    rank = 7 * len(strassen_blocks) + len(uncovered_positions)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Add Strassen blocks
    for (i0, j0, k0) in strassen_blocks:
        col = add_strassen(U, V, W, i0, j0, k0, col)
    
    # Add schoolbook terms for uncovered positions
    for (i, j, k) in sorted(uncovered_positions):
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        col += 1
    
    assert col == rank, f"Expected {rank} terms, got {col}"
    
    # Convert to numpy for efficient verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Verify correctness with numpy
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                assert abs(total - 1) < 1e-10, f"Failed at ({i},{j},{k}): got {total}"
    
    # Convert back to lists
    U_out = U_np.tolist()
    V_out = V_np.tolist()
    W_out = W_np.tolist()
    
    # Round to nearest integer/half-integer
    def round_to_half(x):
        return round(x * 2) / 2
    
    U_out = [[round_to_half(x) for x in row] for row in U_out]
    V_out = [[round_to_half(x) for x in row] for row in V_out]
    W_out = [[round_to_half(x) for x in row] for row in W_out]
    
    # Final verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = sum(U_out[i*m+j][r] * V_out[j*p+k][r] * W_out[k*n+i][r] for r in range(rank))
                assert abs(total - 1) < 1e-10, f"Failed at ({i},{j},{k}): got {total}"
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
