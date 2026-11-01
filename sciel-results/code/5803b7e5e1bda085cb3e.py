import numpy as np
import time

def construct():
    n, m, p = 2, 2, 8
    
    # Build the rank-28 Strassen decomposition as the baseline
    U, V, W = _strassen_28()
    
    # Verify the baseline works
    if _verify_decomposition(U, V, W, n, m, p):
        # Try to find a lower-rank decomposition
        best_rank = 28
        best_result = {"U": U, "V": V, "W": W}
        
        start_time = time.time()
        deadline = 120.0 - 10.0  # 110 seconds, leave 10s safety margin
        
        # Try a smarter approach: exploit the shared A structure
        # The key insight: in Strassen for 2x2*2x2, we compute 7 products
        # of linear combinations. For 2x2*2x8, we have 4 blocks of 2x2.
        # The A-side linear combinations are the SAME for all blocks.
        # So we can potentially share some computations.
        
        # Actually, let's think more carefully. The tensor is:
        # T[i,j,k,l] = 1 if output[i,l] depends on A[i,j]*B[j,l]
        # where i in {0,1}, j in {0,1}, k in {0,1}, l in {0..7}
        
        # Wait, let me re-read the problem. We're multiplying (2x2) * (2x8) -> (2x8)
        # So A is 2x2, B is 2x8, C is 2x8
        # C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, k in {0..7}
        
        # The tensor has non-zero entries at:
        # T[a,b,c] where a = i*m+j = 2*i+j (A index), 
        # b = j*p+k = 8*j+k (B index), 
        # c = k*n+i = 8*k+i (C index)
        # for i in {0,1}, j in {0,1}, k in {0..7}
        
        # Total non-zeros: 2*2*8 = 32
        
        # Let me try to find a rank-24 decomposition by exploiting symmetry
        # or by finding a better Strassen-like algorithm.
        
        # For now, let's just return the verified rank-28 solution
        return best_result
    else:
        # Fallback: schoolbook rank-32
        U, V, W = _schoolbook_32(n, m, p)
        return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    num_a = n * m
    num_b = m * p
    num_c = p * n
    rank = len(U[0]) if U else 0
    
    # Check all entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                
                # Should be 1
                val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check some zero entries
    import random
    random.seed(42)
    for _ in range(1000):
        a = random.randint(0, num_a - 1)
        b = random.randint(0, num_b - 1)
        c = random.randint(0, num_c - 1)
        
        # Check if this is a non-zero entry
        is_nonzero = False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i * m + j == a and j * p + k == b and k * n + i == c:
                        is_nonzero = True
        
        if not is_nonzero:
            val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
            if abs(val) > 1e-9:
                return False
    
    return True


def _strassen_28():
    n, m, p = 2, 2, 8
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    num_blocks = p // 2  # 4 blocks of 2x2
    
    for b in range(num_blocks):
        base = b * 7
        
        # Strassen for 2x2 * 2x2:
        # A = [[a00, a01], [a10, a11]]
        # B = [[b00, b01], [b10, b11]]
        # C = A * B
        #
        # M1 = (a00 + a11) * (b00 + b11)
        # M2 = (a10 + a11) * b00
        # M3 = a00 * (b01 - b11)
        # M4 = a11 * (b10 - b00)
        # M5 = (a00 + a01) * b11
        # M6 = (a10 - a00) * (b00 + b01)
        # M7 = (a01 - a11) * (b10 + b11)
        #
        # C00 = M1 - M2 + M4 + M5
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M3 + M6 - M7
        
        # A indices: a00=0, a01=1, a10=2, a11=3
        # B indices for block b: b00=8*0+2b=2b, b01=8*0+2b+1=2b+1, 
        #                       b10=8*1+2b=8+2b, b11=8*1+2b+1=9+2b
        # C indices: c00=8*2b+0=16b, c01=8*2b+1=16b+1,
        #           c10=8*(2b+1)+0=16b+8, c11=8*(2b+1)+1=16b+9
        
        # M1 = (a00 + a11) * (b00 + b11)
        U[0][base + 0] += 1   # a00
        U[3][base + 0] += 1   # a11
        V[2*b][base + 0] += 1       # b00 = B[0, 2b]
        V[8 + 2*b][base + 0] += 1   # b11 = B[1, 2b+1]
        W[16*b][base + 0] += 1       # c00
        W[16*b + 9][base + 0] += 1   # c11
        
        # M2 = (a10 + a11) * b00
        U[2][base + 1] += 1   # a10
        U[3][base + 1] += 1   # a11
        V[2*b][base + 1] += 1       # b00
        W[16*b][base + 1] -= 1       # c00
        W[16*b + 8][base + 1] += 1   # c10
        
        # M3 = a00 * (b01 - b11)
        U[0][base + 2] += 1   # a00
        V[2*b + 1][base + 2] += 1       # b01 = B[0, 2b+1]
        V[8 + 2*b + 1][base + 2] -= 1   # b11 = B[1, 2b+1]
        W[16*b + 1][base + 2] += 1       # c01
        W[16*b + 9][base + 2] -= 1       # c11
        
        # M4 = a11 * (b10 - b00)
        U[3][base + 3] += 1   # a11
        V[8 + 2*b][base + 3] += 1       # b10 = B[1, 2b]
        V[2*b][base + 3] -= 1           # b00
        W[16*b][base + 3] += 1           # c00
        W[16*b + 8][base + 3] += 1       # c10
        
        # M5 = (a00 + a01) * b11
        U[0][base + 4] += 1   # a00
        U[1][base + 4] += 1   # a01
        V[8 + 2*b + 1][base + 4] += 1   # b11
        W[16*b][base + 4] += 1           # c00
        W[16*b + 1][base + 4] += 1       # c01
        
        # M6 = (a10 - a00) * (b00 + b01)
        U[2][base + 5] += 1   # a10
        U[0][base + 5] -= 1   # a00
        V[2*b][base + 5] += 1           # b00
        V[2*b + 1][base + 5] += 1       # b01
        W[16*b + 9][base + 5] += 1       # c11
        
        # M7 = (a01 - a11) * (b10 + b11)
        U[1][base + 6] += 1   # a01
        U[3][base + 6] -= 1   # a11
        V[8 + 2*b][base + 6] += 1       # b10
        V[8 + 2*b + 1][base + 6] += 1   # b11
        W[16*b + 9][base + 6] -= 1       # c11
    
    return U, V, W


def _schoolbook_32(n, m, p):
    """Schoolbook multiplication: rank n*m*p = 32"""
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a][r] = 1
                V[b][r] = 1
                W[c][r] = 1
    
    return U, V, W
