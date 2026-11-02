"""Tensor decomposition for 2x2 x 2x6 matrix multiplication."""
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 2, 6]

def construct():
    """Return a rank-minimal exact tensor decomposition for 2x2 * 2x6."""
    start_time = time.time()
    deadline = start_time + 120.0  # 120 second deadline with safety margin
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=6
    
    # The matrix multiplication tensor T[i,j,k] = 1 if and only if the output
    # element (i,k) depends on input A[i,j] * B[j,k].
    # For 2x2 * 2x6: A is 2x2, B is 2x6, output C is 2x6.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has non-zero entries at (i,j,k) for all i in {0,1}, j in {0,1}, k in {0,...,5}
    
    # This is equivalent to 6 independent 2x2 matrix multiplications (one for each column k of B).
    # But we can share structure.
    
    # The known minimum rank for 2x2 * 2x2 is 7 (Strassen).
    # For 2x2 * 2x6, we can think of it as 6 separate 2x2 multiplications, but the structure
    # allows sharing.
    
    # Actually, let me think about this differently. The tensor for 2x2 * 2x6 has:
    # - U dimension: n*m = 4 (rows of A)
    # - V dimension: m*p = 12 (elements of B)  
    # - W dimension: p*n = 12 (elements of C)
    
    # The tensor T is a 4 x 12 x 12 tensor with T[a,b,c] = 1 when:
    # a = i*m + j = 2*i + j (for i in {0,1}, j in {0,1})
    # b = j*p + k = 6*j + k (for j in {0,1}, k in {0,...,5})
    # c = k*n + i = 2*k + i (for k in {0,...,5}, i in {0,1})
    # and the constraint is that j in the a-index and j in the b-index must match.
    
    # So T[a,b,c] = 1 if j_a = j_b where j_a = a % 2, j_b = b // 6, and c = 2*k + i_a where i_a = a // 2
    
    # Let's build the support:
    support = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j  # index in U (0-3)
                b = j * p + k  # index in V (0-11)
                c = k * n + i  # index in W (0-11)
                support.append((a, b, c))
    
    # We need to find the minimum rank r such that there exist matrices U (4xr), V (12xr), W (12xr)
    # with integer/half-integer entries such that for all (a,b,c) in support:
    # T[a,b,c] = sum_s U[a,s] * V[b,s] * W[c,s] = 1
    # And for all (a,b,c) NOT in support:
    # sum_s U[a,s] * V[b,s] * W[c,s] = 0
    
    # The schoolbook gives rank 24. Strassen for 2x2 gives rank 7.
    # For 2x2 * 2x6, we can potentially do better than 24.
    
    # One approach: use Strassen's algorithm on pairs of columns.
    # B is 2x6, so we can split it into three 2x2 blocks: B[:, 0:2], B[:, 2:4], B[:, 4:6]
    # Each 2x2 block multiplication has rank 7 (Strassen), so total would be 21.
    
    # Let's implement Strassen for 2x2 * 2x2 and then extend to 2x2 * 2x6.
    
    # Strassen's algorithm for 2x2 * 2x2:
    # Given A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A * B = [[ae+cg, af+dh], [ce+dg, cf+gh]]
    # Strassen computes 7 products:
    # m1 = (a+d)*(e+h)
    # m2 = (c+d)*e
    # m3 = a*(f-h)
    # m4 = d*(g-e)
    # m5 = (a+b)*h
    # m6 = (c-a)*(e+f)
    # m7 = (b-d)*(g+h)
    # C[0,0] = m1 + m4 - m5 + m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 + m6
    
    # For 2x2 * 2x6, split B into three 2x2 blocks.
    # We'll use 3 * 7 = 21 rank terms.
    
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each block b_idx in {0, 1, 2}, corresponding to columns 2*b_idx, 2*b_idx+1 of B:
    for b_idx in range(3):
        base = b_idx * 7
        k0 = 2 * b_idx
        k1 = 2 * b_idx + 1
        
        # Strassen indices for this block:
        # m1 = (a+d)*(e+h): e=B[0,k0], h=B[1,k1]
        # m2 = (c+d)*e: c=A[1,0], d=A[1,1], e=B[0,k0]
        # m3 = a*(f-h): a=A[0,0], f=B[0,k1], h=B[1,k1]
        # m4 = d*(g-e): d=A[1,1], g=B[1,k0], e=B[0,k0]
        # m5 = (a+b)*h: a=A[0,0], b=A[0,1], h=B[1,k1]
        # m6 = (c-a)*(e+f): c=A[1,0], a=A[0,0], e=B[0,k0], f=B[0,k1]
        # m7 = (b-d)*(g+h): b=A[0,1], d=A[1,1], g=B[1,k0], h=B[1,k1]
        
        # Each Strassen product m_s involves linear combinations of A and B entries.
        # For the tensor decomposition, each "product" corresponds to one column in U, V, W.
        
        # m1: A part = a + d = A[0,0] + A[1,1], B part = e + h = B[0,k0] + B[1,k1]
        # U for A part: A[0,0] is at index 0*2+0=0, A[1,1] is at index 1*2+1=3
        # V for B part: B[0,k0] is at index 0*6+k0=k0, B[1,k1] is at index 1*6+k1=6+k1
        s = base + 0
        U[0][s] += 1  # A[0,0]
        U[3][s] += 1  # A[1,1]
        V[k0][s] += 1  # B[0,k0]
        V[6 + k1][s] += 1  # B[1,k1]
        # W: C[0,0] = m1 + ..., C[1,1] = m1 - ...
        # C[0,0] is at index k0*2+0 = 2*k0, C[1,1] is at index k1*2+1 = 2*k1+1
        W[2 * k0][s] += 1  # C[0,0]
        W[2 * k1 + 1][s] += 1  # C[1,1]
        
        # m2: A part = c + d = A[1,0] + A[1,1], B part = e = B[0,k0]
        s = base + 1
        U[1*2+0][s] += 1  # A[1,0] at index 2
        U[1*2+1][s] += 1  # A[1,1] at index 3
        V[0*6+k0][s] += 1  # B[0,k0] at index k0
        # W: C[1,0] = m2 + ..., C[1,1] = m1 - m2 + ...
        # C[1,0] at index k0*2+1 = 2*k0+1, C[1,1] at index 2*k1+1
        W[2 * k0 + 1][s] += 1  # C[1,0]
        W[2 * k1 + 1][s] += -1  # C[1,1]
        
        # m3: A part = a = A[0,0], B part = f - h = B[0,k1] - B[1,k1]
        s = base + 2
        U[0*2+0][s] += 1  # A[0,0] at index 0
        V[0*6+k1][s] += 1  # B[0,k1] at index k1
        V[1*6+k1][s] += -1  # B[1,k1] at index 6+k1
        # W: C[0,1] = m3 + ..., C[1,1] = m1 - m2 + m3 + ...
        # C[0,1] at index k1*2+0 = 2*k1, C[1,1] at index 2*k1+1
        W[2 * k1][s] += 1  # C[0,1]
        W[2 * k1 + 1][s] += 1  # C[1,1]
        
        # m4: A part = d = A[1,1], B part = g - e = B[1,k0] - B[0,k0]
        s = base + 3
        U[1*2+1][s] += 1  # A[1,1] at index 3
        V[1*6+k0][s] += 1  # B[1,k0] at index 6+k0
        V[0*6+k0][s] += -1  # B[0,k0] at index k0
        # W: C[0,0] = m1 + m4 - ..., C[1,0] = m2 + m4
        # C[0,0] at index 2*k0, C[1,0] at index 2*k0+1
        W[2 * k0][s] += 1  # C[0,0]
        W[2 * k0 + 1][s] += 1  # C[1,0]
        
        # m5: A part = a + b = A[0,0] + A[0,1], B part = h = B[1,k1]
        s = base + 4
        U[0*2+0][s] += 1  # A[0,0] at index 0
        U[0*2+1][s] += 1  # A[0,1] at index 1
        V[1*6+k1][s] += 1  # B[1,k1] at index 6+k1
        # W: C[0,0] = m1 + m4 - m5 + ..., C[0,1] = m3 + m5
        # C[0,0] at index 2*k0, C[0,1] at index 2*k1
        W[2 * k0][s] += -1  # C[0,0]
        W[2 * k1][s] += 1  # C[0,1]
        
        # m6: A part = c - a = A[1,0] - A[0,0], B part = e + f = B[0,k0] + B[0,k1]
        s = base + 5
        U[1*2+0][s] += 1  # A[1,0] at index 2
        U[0*2+0][s] += -1  # A[0,0] at index 0
        V[0*6+k0][s] += 1  # B[0,k0] at index k0
        V[0*6+k1][s] += 1  # B[0,k1] at index k1
        # W: C[1,1] = m1 - m2 + m3 + m6
        # C[1,1] at index 2*k1+1
        W[2 * k1 + 1][s] += 1  # C[1,1]
        
        # m7: A part = b - d = A[0,1] - A[1,1], B part = g + h = B[1,k0] + B[1,k1]
        s = base + 6
        U[0*2+1][s] += 1  # A[0,1] at index 1
        U[1*2+1][s] += -1  # A[1,1] at index 3
        V[1*6+k0][s] += 1  # B[1,k0] at index 6+k0
        V[1*6+k1][s] += 1  # B[1,k1] at index 6+k1
        # W: C[0,0] = m1 + m4 - m5 + m7
        # C[0,0] at index 2*k0
        W[2 * k0][s] += 1  # C[0,0]
    
    # Verify the decomposition
    # Check that for all (a,b,c) in support, sum_s U[a,s]*V[b,s]*W[c,s] == 1
    # And for all (a,b,c) not in support, sum_s U[a,s]*V[b,s]*W[c,s] == 0
    
    support_set = set(support)
    valid = True
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                val = 0
                for s in range(rank):
                    val += U[a][s] * V[b][s] * W[c][s]
                if (a, b, c) in support_set:
                    if val != 1:
                        valid = False
                        break
                else:
                    if val != 0:
                        valid = False
                        break
        if not valid:
            break
    
    if valid:
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if verification fails
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    return {"U": U_school, "V": V_school, "W": W_school}
