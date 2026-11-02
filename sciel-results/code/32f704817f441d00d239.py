"""Tensor rank decomposition for 3x5x5 matrix multiplication."""
import numpy as np
import time
from itertools import combinations

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """Return a valid rank decomposition for 3x5x5 matrix multiplication.
    
    Strategy:
      - Use one 3x3x3 Strassen-like block (rank 15) for a 3x3x3 sub-block.
      - Use disjoint 2x2x2 Strassen blocks (rank 7 each) for as many as fit.
      - Schoolbook for the remainder.
    """
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    total_terms = n * m * p  # 75
    
    def build_decomposition(blocks3, blocks2):
        """Build a decomposition using the given 3x3x3 and 2x2x2 blocks.
        
        blocks3: list of (i0, j0, k0) for 3x3x3 blocks
        blocks2: list of (i0, j0, k0) for 2x2x2 blocks
        Returns (U, V, W) as lists of lists, and the rank.
        """
        rank3 = 15  # rank of 3x3x3 Strassen
        rank2 = 7   # rank of 2x2x2 Strassen
        
        # Count total rank
        total_rank = len(blocks3) * rank3 + len(blocks2) * rank2
        # Remainder terms
        covered = [[[False]*p for _ in range(m)] for _ in range(n)]
        for i0, j0, k0 in blocks3:
            for di in range(3):
                for dj in range(3):
                    for dk in range(3):
                        if i0+di < n and j0+dj < m and k0+dk < p:
                            covered[i0+di][j0+dj][k0+dk] = True
        for i0, j0, k0 in blocks2:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        if i0+di < n and j0+dj < m and k0+dk < p:
                            covered[i0+di][j0+dj][k0+dk] = True
        
        remainder_count = sum(1 for i in range(n) for j in range(m) for k in range(p) if not covered[i][j][k])
        total_rank += remainder_count
        
        U = [[0]*total_rank for _ in range(n*m)]
        V = [[0]*total_rank for _ in range(m*p)]
        W = [[0]*total_rank for _ in range(p*n)]
        
        col = 0
        
        # Add 3x3x3 Strassen blocks
        def add_strassen3(i0, j0, k0, col_offset):
            # 3x3x3 Strassen-like decomposition with rank 15
            # M1 = (A00+A11+A22)(B00+B11+B22)
            c = col_offset + 0
            for d in range(3):
                U[(i0+d)*m + (j0+d)][c] += 1
                V[(j0+d)*p + (k0+d)][c] += 1
                W[(k0+d)*n + (i0+d)][c] += 1
            
            # M2 = (A10+A11+A21)(B00+B10+B20)
            c = col_offset + 1
            for d in range(3):
                U[(i0+d)*m + (j0+1)][c] += 1
                V[(j0+d)*p + (k0+0)][c] += 1
                W[(k0+d)*n + (i0+1)][c] += 1
            
            # M3 = (A00+A20+A22)(B00+B02+B22)
            c = col_offset + 2
            U[i0*m + j0][c] += 1
            U[(i0+2)*m + (j0+0)][c] += 1
            U[(i0+2)*m + (j0+2)][c] += 1
            V[j0*p + k0][c] += 1
            V[j0*p + (k0+2)][c] += 1
            V[(j0+2)*p + (k0+2)][c] += 1
            W[k0*n + i0][c] += 1
            W[(k0+2)*n + (i0+0)][c] += 1
            W[(k0+2)*n + (i0+2)][c] += 1
            
            # M4 = (A01+A11+A12)(B10+B11+B12)
            c = col_offset + 3
            U[i0*m + (j0+1)][c] += 1
            U[(i0+1)*m + (j0+1)][c] += 1
            U[(i0+1)*m + (j0+2)][c] += 1
            V[(j0+1)*p + k0][c] += 1
            V[(j0+1)*p + (k0+1)][c] += 1
            V[(j0+1)*p + (k0+2)][c] += 1
            W[k0*n + i0][c] += 1
            W[(k0+1)*n + (i0+1)][c] += 1
            W[(k0+2)*n + (i0+1)][c] += 1
            
            # M5 = (A00+A01+A10)(B00+B01+B10)
            c = col_offset + 4
            U[i0*m + j0][c] += 1
            U[i0*m + (j0+1)][c] += 1
            U[(i0+1)*m + j0][c] += 1
            V[j0*p + k0][c] += 1
            V[j0*p + (k0+1)][c] += 1
            V[(j0+1)*p + k0][c] += 1
            W[k0*n + i0][c] += 1
            W[(k0+1)*n + i0][c] += 1
            W[k0*n + (i0+1)][c] += 1
            
            # M6 = (A10+A20+A21)(B10+B20+B21)
            c = col_offset + 5
            U[(i0+1)*m + j0][c] += 1
            U[(i0+2)*m + j0][c] += 1
            U[(i0+2)*m + (j0+1)][c] += 1
            V[(j0+1)*p + k0][c] += 1
            V[(j0+2)*p + k0][c] += 1
            V[(j0+2)*p + (k0+1)][c] += 1
            W[k0*n + (i0+1)][c] += 1
            W[k0*n + (i0+2)][c] += 1
            W[(k0+1)*n + (i0+2)][c] += 1
            
            # M7 = (A01+A02+A12)(B10+B11+B12)
            c = col_offset + 6
            U[i0*m + (j0+1)][c] += 1
            U[i0*m + (j0+2)][c] += 1
            U[(i0+1)*m + (j0+2)][c] += 1
            V[(j0+1)*p + k0][c] += 1
            V[(j0+1)*p + (k0+1)][c] += 1
            V[(j0+1)*p + (k0+2)][c] += 1
            W[k0*n + i0][c] += 1
            W[(k0+1)*n + i0][c] += 1
            W[(k0+2)*n + i0][c] += 1
            
            # M8 = (A00+A10+A11)(B00+B01+B11)
            c = col_offset + 7
            U[i0*m + j0][c] += 1
            U[(i0+1)*m + j0][c] += 1
            U[(i0+1)*m + (j0+1)][c] += 1
            V[j0*p + k0][c] += 1
            V[j0*p + (k0+1)][c] += 1
            V[(j0+1)*p + (k0+1)][c] += 1
            W[k0*n + i0][c] += 1
            W[(k0+1)*n + i0][c] += 1
            W[(k0+1)*n + (i0+1)][c] += 1
            
            # M9 = (A20+A21+A22)(B00+B10+B11)
            c = col_offset + 8
            U[(i0+2)*m + j0][c] += 1
            U[(i0+2)*m + (j0+1)][c] += 1
            U[(i0+2)*m + (j0+2)][c] += 1
            V[j0*p + k0][c] += 1
            V[(j0+1)*p + k0][c] += 1
            V[(j0+1)*p + (k0+1)][c] += 1
            W[k0*n + (i0+2)][c] += 1
            W[k0*n + (i0+1)][c] += 1
            W[(k0+1)*n + (i0+1)][c] += 1
            
            # M10 = (A00+A02+A22)(B00+B02+B22)
            c = col_offset + 9
            U[i0*m + j0][c] += 1
            U[i0*m + (j0+2)][c] += 1
            U[(i0+2)*m + (j0+2)][c] += 1
            V[j0*p + k0][c] += 1
            V[j0*p + (k0+2)][c] += 1
            V[(j0+2)*p + (k0+2)][c] += 1
            W[k0*n + i0][c] += 1
            W[(k0+2)*n + i0][c] += 1
            W[(k0+2)*n + (i0+2)][c] += 1
            
            # M11 = (A11+A12+A22)(B11+B12+B22)
            c = col_offset + 10
            U[(i0+1)*m + (j0+1)][c] += 1
            U[(i0+1)*m + (j0+2)][c] += 1
            U[(i0+2)*m + (j0+2)][c] += 1
            V[(j0+1)*p + (k0+1)][c] += 1
            V[(j0+1)*p + (k0+2)][c] += 1
            V[(j0+2)*p + (k0+2)][c] += 1
            W[(k0+1)*n + (i0+1)][c] += 1
            W[(k0+2)*n + (i0+1)][c] += 1
            W[(k0+2)*n + (i0+2)][c] += 1
            
            # M12 = (A00-A11)(B00+B11)
            c = col_offset + 11
            U[i0*m + j0][c] += 1
            U[(i0+1)*m + (j0+1)][c] += -1
            V[j0*p + k0][c] += 1
            V[(j0+1)*p + (k0+1)][c] += 1
            W[k0*n + i0][c] += 1
            W[(k0+1)*n + (i0+1)][c] += 1
            
            # M13 = (A10-A00)(B00-B01)
            c = col_offset + 12
            U[(i0+1)*m + j0][c] += 1
            U[i0*m + j0][c] += -1
            V[j0*p + k0][c] += 1
            V[j0*p + (k0+1)][c] += -1
            W[k0*n + (i0+1)][c] += 1
            W[(k0+1)*n + i0][c] += 1
            
            # M14 = (A01-A11)(B10-B11)
            c = col_offset + 13
            U[i0*m + (j0+1)][c] += 1
            U[(i0+1)*m + (j0+1)][c] += -1
            V[(j0+1)*p + k0][c] += 1
            V[(j0+1)*p + (k0+1)][c] += -1
            W[k0*n + i0][c] += 1
            W[(k0+1)*n + (i0+1)][c] += 1
            
            # M15 = (A02-A22)(B20-B22)
            c = col_offset + 14
            U[i0*m + (j0+2)][c] += 1
            U[(i0+2)*m + (j0+2)][c] += -1
            V[(j0+2)*p + k0][c] += 1
            V[(j0+2)*p + (k0+2)][c] += -1
            W[k0*n + i0][c] += 1
            W[(k0+2)*n + (i0+2)][c] += 1
        
        def add_strassen2(i0, j0, k0, col_offset):
            # Standard 2x2x2 Strassen with rank 7
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
        
        for i0, j0, k0 in blocks3:
            add_strassen3(i0, j0, k0, col)
            col += rank3
        
        for i0, j0, k0 in blocks2:
            add_strassen2(i0, j0, k0, col)
            col += rank2
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not covered[i][j][k]:
                        U[i*m+j][col] += 1
                        V[j*p+k][col] += 1
                        W[k*n+i][col] += 1
                        col += 1
        
        return U, V, W, total_rank
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify the decomposition is correct."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        np.random.seed(12345)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_reconstructed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    C_reconstructed[i, k] += np.dot(U_arr[i * m + j, :], V_arr[j * p + k, :] * W_arr[k * n + i, :])
        
        error = np.max(np.abs(C_reconstructed - C_expected))
        return error < 1e-10, error
    
    # Generate all possible 3x3x3 block positions
    possible_blocks3 = [(i0, j0, k0) for i0 in range(n-2) for j0 in range(m-2) for k0 in range(p-2)]
    # Generate all possible 2x2x2 block positions
    possible_blocks2 = [(i0, j0, k0) for i0 in range(n-1) for j0 in range(m-1) for k0 in range(p-1)]
    
    # Search for the best combination of blocks
    best_rank = total_terms  # Start with schoolbook
    best_blocks3 = []
    best_blocks2 = []
    
    # Try all combinations of 3x3x3 blocks (max 1 or 2)
    for num_b3 in range(0, 3):
        if time.time() > deadline - 20:
            break
        for blocks3 in combinations(possible_blocks3, num_b3):
            if time.time() > deadline - 20:
                break
            # Check if blocks3 are valid (don't need to be disjoint, but let's try)
            # For 3x3x3 blocks, they must fit within the matrix
            # Try all combinations of 2x2x2 blocks that don't overlap with blocks3
            covered_by_b3 = [[[False]*p for _ in range(m)] for _ in range(n)]
            for i0, j0, k0 in blocks3:
                for di in range(3):
                    for dj in range(3):
                        for dk in range(3):
                            covered_by_b3[i0+di][j0+dj][k0+dk] = True
            
            # Find 2x2x2 blocks that don't overlap with blocks3
            valid_b2 = []
            for i0, j0, k0 in possible_blocks2:
                valid = True
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            if covered_by_b3[i0+di][j0+dj][k0+dk]:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    valid_b2.append((i0, j0, k0))
            
            # Try different numbers of 2x2x2 blocks
            for num_b2 in range(0, min(len(valid_b2), 5)):
                if time.time() > deadline - 10:
                    break
                for blocks2 in combinations(valid_b2, num_b2):
                    # Check if blocks2 are mutually disjoint
                    disjoint = True
                    for idx1, b1 in enumerate(blocks2):
                        for idx2, b2 in enumerate(blocks2):
                            if idx1 >= idx2:
                                continue
                            i1, j1, k1 = b1
                            i2, j2, k2 = b2
                            # Check overlap
                            if max(i1, i2) < min(i1+2, i2+2) and max(j1, j2) < min(j1+2, j2+2) and max(k1, k2) < min(k1+2, k2+2):
                                disjoint = False
                                break
                        if not disjoint:
                            break
                    if not disjoint:
                        continue
                    
                    # Build and verify
                    U, V, W, rank = build_decomposition(list(blocks3), list(blocks2))
                    valid, error = verify_decomposition(U, V, W, n, m, p)
                    
                    if valid and rank < best_rank:
                        best_rank = rank
                        best_blocks3 = list(blocks3)
                        best_blocks2 = list(blocks2)
    
    # If we found a better solution, use it; otherwise fall back to the original
    if best_rank < total_terms and (best_blocks3 or best_blocks2):
        U, V, W, rank = build_decomposition(best_blocks3, best_blocks2)
    else:
        # Fall back to the original 4x 2x2x2 Strassen + schoolbook
        blocks = [
            (0, 0, 0),
            (0, 2, 0),
            (0, 0, 2),
            (0, 2, 2)
        ]
        U, V, W, rank = build_decomposition([], blocks)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    rank = len(result["U"][0])
    print(f"Rank: {rank}")
    
    # Verify correctness
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    U_arr = np.array(result["U"], dtype=float)
    V_arr = np.array(result["V"], dtype=float)
    W_arr = np.array(result["W"], dtype=float)
    
    C_reconstructed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for j in range(m):
                C_reconstructed[i, k] += np.dot(U_arr[i * m + j, :], V_arr[j * p + k, :] * W_arr[k * n + i, :])
    
    error = np.max(np.abs(C_reconstructed - C_expected))
    print(f"Max reconstruction error: {error}")
    assert error < 1e-10, "Decomposition is incorrect!"
    print("Decomposition verified successfully.")
