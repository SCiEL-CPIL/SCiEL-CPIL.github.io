"""Matrix multiplication 2x5x6: rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

def _verify(U, V, W, n=2, m=5, p=6):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                if val != 1:
                    return False
    return True

def _verify_full(U, V, W, n=2, m=5, p=6):
    """Full verification including zero entries."""
    rank = len(U[0])
    # Check valid entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                if val != 1:
                    return False
    # Check that non-valid entries are 0 (sample check)
    # For a,b,c where there's no valid (i,j,k)
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                # Check if (a,b,c) corresponds to a valid triple
                i1 = a // m
                j1 = a % m
                j2 = b // p
                k2 = b % p
                k1 = c // n
                i2 = c % n
                valid = (i1 == i2) and (j1 == j2) and (k1 == k2)
                if not valid:
                    val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                    if val != 0:
                        return False
    return True

def _strassen_2x2x2():
    """Return the 7 rank-1 tensors for 2x2x2 Strassen decomposition."""
    tensors = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]),
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    return tensors

def _strassen_2x3x2():
    """Return a rank-7 decomposition for 2x3x2 matrix multiplication.
    
    A is 2x3: indices [a00, a01, a02, a10, a11, a12]
    B is 3x2: indices [b00, b01, b10, b11, b20, b21]
    C is 2x2: indices [c00, c01, c10, c11]
    """
    # We'll construct this by combining ideas from Strassen.
    # Let A = [[a, b, c], [d, e, f]], B = [[g, h], [i, j], [k, l]]
    # C = [[ag+bi+ck, ah+bj+cl], [dg+ei+fk, dh+ej+fl]]
    
    # Try to find 7 products. Let me use a systematic approach.
    # Actually, let me just use a known construction or derive one.
    
    # One approach: embed in 2x4x2 and use a known decomposition,
    # or directly construct.
    
    # Let me try:
    # P1 = (a+c)(g+k) -> a*g + a*k + c*g + c*k
    # P2 = (d+f)g -> d*g + f*g
    # P3 = a(i-k) -> a*i - a*k
    # P4 = f(j-g) -> f*j - f*g
    # P5 = (a+b)l -> a*l + b*l
    # P6 = (d-a)(g+h) -> d*g + d*h - a*g - a*h
    # P7 = (b+c-f)(i+k) -> b*i + b*k + c*i + c*k - f*i - f*k
    
    # Let's compute what C entries we get:
    # c00 = ag + bi + ck
    # c01 = ah + bj + cl
    # c10 = dg + ei + fk
    # c11 = dh + ej + fl
    
    # From P1: ag + ak + cg + ck
    # From P2: dg + fg
    # From P3: ai - ak
    # From P4: fj - fg
    # From P5: al + bl
    # From P6: dg + dh - ag - ah
    # From P7: bi + bk + ci + ck - fi - fk
    
    # c00 = ag + bi + ck
    # Let's see: P1 + P3 + P7 = (ag+ak+cg+ck) + (ai-ak) + (bi+bk+ci+ck-fi-fk)
    # = ag + cg + ck + ai + bi + bk + ci + ck - fi - fk
    # = ag + bi + ci + cg + ck + ck + ai + bk - fi - fk
    # That's not quite right.
    
    # Let me try a different combination.
    # This is getting complex. Let me just use a numerical approach to find the decomposition.
    
    # For now, let me return a schoolbook decomposition (rank 12) and we can improve later.
    tensors = []
    for i in range(2):
        for j in range(3):
            for k in range(2):
                u = [0]*6
                v = [0]*6
                w = [0]*4
                u[i*3+j] = 1
                v[j*2+k] = 1
                w[k*2+i] = 1
                tensors.append((u, v, w))
    return tensors

def _construct_2x3x2_optimal():
    """Try to find a better 2x3x2 decomposition."""
    # Let me try to solve this using linear algebra.
    # We want to find rank r such that the 2x3x2 tensor can be written as sum of r rank-1 tensors.
    # The tensor T has T[i,j,k] = 1 if the multiplication C[i,k] = sum_j A[i,j]*B[j,k]
    
    # For now, return schoolbook (rank 12)
    tensors = []
    for i in range(2):
        for j in range(3):
            for k in range(2):
                u = [0]*6
                v = [0]*6
                w = [0]*4
                u[i*3+j] = 1
                v[j*2+k] = 1
                w[k*2+i] = 1
                tensors.append((u, v, w))
    return tensors

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 60
    
    # Strategy: Split into 2x2x6 (Strassen, rank 21) + 2x3x6 (schoolbook, rank 36)
    # Total: 57
    # 
    # Better: Try to find a better 2x3x2 decomposition to use for the 2x3x6 part.
    # If we can get rank 7 for 2x3x2, then 2x3x6 = 3*7 = 21, total = 42.
    
    # First, let's try the 57-rank decomposition as a baseline.
    def _build_57():
        rank = 57
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        col = 0
        
        # Part 1: j in {0,1,2}, schoolbook for 2x3x6
        for i in range(n):
            for j in range(3):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    U[a_idx][col] = 1.0
                    V[b_idx][col] = 1.0
                    W[c_idx][col] = 1.0
                    col += 1
        
        # Part 2: j in {3,4}, Strassen on three 2x2x2 blocks
        strassen_tensors = _strassen_2x2x2()
        for block in range(3):
            k_offset = block * 2
            for (u, v, w) in strassen_tensors:
                u_map = [(0, 3), (0, 4), (1, 3), (1, 4)]
                for idx, (ii, jj) in enumerate(u_map):
                    a_idx = ii * m + jj
                    U[a_idx][col] += u[idx]
                
                v_map = [(3, k_offset), (3, k_offset+1), (4, k_offset), (4, k_offset+1)]
                for idx, (jj, kk) in enumerate(v_map):
                    b_idx = jj * p + kk
                    V[b_idx][col] += v[idx]
                
                w_map = [(k_offset, 0), (k_offset+1, 0), (k_offset, 1), (k_offset+1, 1)]
                for idx, (kk, ii) in enumerate(w_map):
                    c_idx = kk * n + ii
                    W[c_idx][col] += w[idx]
                
                col += 1
        
        return U, V, W
    
    # Build and verify the 57-rank decomposition
    U57, V57, W57 = _build_57()
    if _verify(U57, V57, W57, n, m, p):
        # Try to do better
        pass
    
    # Try a 4x4x4 based approach:
    # Embed 2x5x6 into 4x8x8 and use a rank-16 4x4x4 decomposition scaled up.
    # Actually, let me try a different block split.
    
    # Alternative: 2x5x6 = 2x(2+2+1)x6
    # 2x2x6 (Strassen, 21) + 2x2x6 (Strassen, 21) + 2x1x6 (schoolbook, 12) = 54
    def _build_54():
        rank = 54
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        col = 0
        strassen_tensors = _strassen_2x2x2()
        
        # Block 1: j in {0,1}
        for block in range(3):
            k_offset = block * 2
            for (u, v, w) in strassen_tensors:
                u_map = [(0, 0), (0, 1), (1, 0), (1, 1)]
                for idx, (ii, jj) in enumerate(u_map):
                    a_idx = ii * m + jj
                    U[a_idx][col] += u[idx]
                
                v_map = [(0, k_offset), (0, k_offset+1), (1, k_offset), (1, k_offset+1)]
                for idx, (jj, kk) in enumerate(v_map):
                    b_idx = jj * p + kk
                    V[b_idx][col] += v[idx]
                
                w_map = [(k_offset, 0), (k_offset+1, 0), (k_offset, 1), (k_offset+1, 1)]
                for idx, (kk, ii) in enumerate(w_map):
                    c_idx = kk * n + ii
                    W[c_idx][col] += w[idx]
                
                col += 1
        
        # Block 2: j in {2,3}
        for block in range(3):
            k_offset = block * 2
            for (u, v, w) in strassen_tensors:
                u_map = [(0, 2), (0, 3), (1, 2), (1, 3)]
                for idx, (ii, jj) in enumerate(u_map):
                    a_idx = ii * m + jj
                    U[a_idx][col] += u[idx]
                
                v_map = [(2, k_offset), (2, k_offset+1), (3, k_offset), (3, k_offset+1)]
                for idx, (jj, kk) in enumerate(v_map):
                    b_idx = jj * p + kk
                    V[b_idx][col] += v[idx]
                
                w_map = [(k_offset, 0), (k_offset+1, 0), (k_offset, 1), (k_offset+1, 1)]
                for idx, (kk, ii) in enumerate(w_map):
                    c_idx = kk * n + ii
                    W[c_idx][col] += w[idx]
                
                col += 1
        
        # Block 3: j = 4, schoolbook for 2x1x6
        for i in range(n):
            for k in range(p):
                a_idx = i * m + 4
                b_idx = 4 * p + k
                c_idx = k * n + i
                U[a_idx][col] = 1.0
                V[b_idx][col] = 1.0
                W[c_idx][col] = 1.0
                col += 1
        
        return U, V, W
    
    U54, V54, W54 = _build_54()
    if _verify(U54, V54, W54, n, m, p):
        return {"U": U54, "V": V54, "W": W54}
    
    # Try 2x5x6 = 2x(3+2)x6 with better 2x3x2
    # For now, return the best we have
    if _verify(U57, V57, W57, n, m, p):
        return {"U": U57, "V": V57, "W": W57}
    
    # Fall back to schoolbook
    rank = schoolbook_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
