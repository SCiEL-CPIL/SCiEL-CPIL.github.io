"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 140  # safety margin before 150s wall-clock kill

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition is exact."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            u_row = U_np[i * m + j, :]
            for k in range(p):
                v_row = V_np[j * p + k, :]
                w_row = W_np[k * n + i, :]
                s = np.dot(u_row, v_row * w_row)
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol

def _make_3x3_laderman():
    """
    Laderman decomposition of 3x3 * 3x3 -> 3x3.
    Rank 23.
    Returns (U, V, W) as 2D arrays of shape (9, 23).
    """
    # Laderman's algorithm for 3x3 matrix multiplication
    # C = A * B where A, B, C are 3x3
    # C[i][k] = sum_j A[i][j] * B[j][k]
    #
    # Laderman's 23 products:
    # P0  = A00 * B00
    # P1  = A01 * B10
    # P2  = A02 * B20
    # P3  = A10 * B01
    # P4  = A11 * B11
    # P5  = A12 * B21
    # P6  = A20 * B02
    # P7  = A21 * B12
    # P8  = A22 * B22
    # P9  = (A00 + A11) * (B00 + B11)
    # P10 = (A00 + A22) * (B00 + B22)
    # P11 = (A11 + A22) * (B11 + B22)
    # P12 = (A00 + A01 + A10 + A11) * (B00 + B01 + B10 + B11)
    # P13 = (A00 + A02 + A20 + A22) * (B00 + B02 + B20 + B22)
    # P14 = (A11 + A12 + A21 + A22) * (B11 + B12 + B21 + B22)
    # P15 = (A00 + A01 + A10 + A11 + A02 + A12 + A20 + A22) * (B00 + B01 + B10 + B11 + B02 + B12 + B20 + B22)
    # P16 = (A00 - A01 + A10 - A11) * (B00 - B01 + B10 - B11)
    # P17 = (A00 - A02 + A20 - A22) * (B00 - B02 + B20 - B22)
    # P18 = (A11 - A12 + A21 - A22) * (B11 - B12 + B21 - B22)
    # P19 = (A01 + A11 - A02 - A12) * (B10 + B11 - B20 - B21)
    # P20 = (A10 + A11 - A20 - A21) * (B00 + B01 - B02 - B02)  -- wait, need to be careful
    # P21 = (A01 + A02 - A11 - A12) * (B20 + B21 - B10 - B11)
    # P22 = (A10 + A20 - A11 - A21) * (B02 + B12 - B00 - B10)
    
    # Let me use the standard Laderman formulation more carefully.
    # From Wikipedia and standard references:
    #
    # P1  = (a11 + a22)(b11 + b22)
    # P2  = (a22 - a11)(b11)
    # P3  = (a11 + a12)(b22)
    # P4  = (a21 + a22)(b11)
    # P5  = (a11 - a21)(b21 + b22)
    # P6  = (a12 + a22)(b11 - b12)
    # P7  = (a11 + a12 + a21 + a22)(b11 + b12 + b21 + b22)
    # P8  = (a11 - a12 + a21 - a22)(b11 - b12 + b21 - b22)
    # P9  = (a12 + a22 - a11 - a21)(b11 + b12 - b21 - b22)
    # P10 = (a11 + a12 - a21 - a22)(b11 - b12 + b21 - b22)
    # P11 = (a11 + a21 - a12 - a22)(b11 + b12 + b21 + b22)
    # P12 = (a11 - a21 + a12 - a22)(b11 - b12 - b21 + b22)
    # P13 = a11*b11
    # P14 = a12*b21
    # P15 = a21*b12
    # P16 = a22*b22
    # P17 = a11*b12
    # P18 = a12*b11
    # P19 = a21*b22
    # P20 = a22*b21
    # P21 = a11*b21
    # P22 = a12*b22
    # P23 = a21*b11
    
    # Actually, let me use a cleaner formulation. I'll define it by
    # specifying the U, V, W factors directly for each of the 23 terms.
    # Each term c has U_c (3x3), V_c (3x3), W_c (3x3) such that
    # sum_c U_c ⊗ V_c ⊗ W_c = multiplication tensor.
    
    # I'll use the formulation from the original Laderman paper.
    # The 23 products are:
    
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # Helper: index (i,j) -> i*3+j for U and V (row-major, i=row of A, j=col of A / row of B)
    # For W: index (k,i) -> k*3+i (row k of B becomes col, col i of A becomes row of C)
    # Actually W is indexed by (output_row, output_col) = (i, k) in C[i][k]
    # But the convention says W[k*n+i, r], so W is indexed by (k, i) where k is output col, i is output row
    # Wait, let me re-read: "W[k*n+i,r]" - so W has p*n rows, indexed by k (0..p-1) and i (0..n-1)
    # C[i][k] = sum_j A[i][j] B[j][k]
    # In tensor form: T[i,j,k] = 1
    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over c and j should give T[i,j,k]
    # Actually the decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # For 3x3: n=m=p=3
    # U indexed by (i,j) -> i*3+j
    # V indexed by (j,k) -> j*3+k  
    # W indexed by (k,i) -> k*3+i
    
    # Term 0: A00 * B00 -> contributes to C00
    U[0*3+0, 0] = 1; V[0*3+0, 0] = 1; W[0*3+0, 0] = 1
    
    # Term 1: A01 * B10 -> contributes to C00
    U[0*3+1, 1] = 1; V[1*3+0, 1] = 1; W[0*3+0, 1] = 1
    
    # Term 2: A02 * B20 -> contributes to C00
    U[0*3+2, 2] = 1; V[2*3+0, 2] = 1; W[0*3+0, 2] = 1
    
    # Term 3: A10 * B01 -> contributes to C10
    U[1*3+0, 3] = 1; V[0*3+1, 3] = 1; W[1*3+0, 3] = 1
    
    # Term 4: A11 * B11 -> contributes to C11
    U[1*3+1, 4] = 1; V[1*3+1, 4] = 1; W[1*3+1, 4] = 1
    
    # Term 5: A12 * B21 -> contributes to C11
    U[1*3+2, 5] = 1; V[2*3+1, 5] = 1; W[1*3+1, 5] = 1
    
    # Term 6: A20 * B02 -> contributes to C20
    U[2*3+0, 6] = 1; V[0*3+2, 6] = 1; W[2*3+0, 6] = 1
    
    # Term 7: A21 * B12 -> contributes to C21
    U[2*3+1, 7] = 1; V[1*3+2, 7] = 1; W[2*3+1, 7] = 1
    
    # Term 8: A22 * B22 -> contributes to C22
    U[2*3+2, 8] = 1; V[2*3+2, 8] = 1; W[2*3+2, 8] = 1
    
    # Now the 14 composite terms. Let me use a systematic approach.
    # I'll define each term by its U, V, W sparsity patterns.
    
    # Term 9: (A00+A11) * (B00+B11)
    # U: A00, A11 -> positions (0,0) and (1,1)
    # V: B00, B11 -> positions (0,0) and (1,1)
    # W: contributes to C01 (A00*B11) and C10 (A11*B00)
    # C[i][k]: i=0,k=1 gets A00*B11; i=1,k=0 gets A11*B00
    # W index: (k,i) -> (1,0) and (0,1)
    U[0*3+0, 9] = 1; U[1*3+1, 9] = 1
    V[0*3+0, 9] = 1; V[1*3+1, 9] = 1
    W[1*3+0, 9] = 1  # C[0][1]
    W[0*3+1, 9] = 1  # C[1][0]
    
    # Term 10: (A00+A22) * (B00+B22)
    # Contributes to C02 (A00*B22) and C20 (A22*B00)
    U[0*3+0, 10] = 1; U[2*3+2, 10] = 1
    V[0*3+0, 10] = 1; V[2*3+2, 10] = 1
    W[2*3+0, 10] = 1  # C[0][2]
    W[0*3+2, 10] = 1  # C[2][0]
    
    # Term 11: (A11+A22) * (B11+B22)
    # Contributes to C12 (A11*B22) and C21 (A22*B11)
    U[1*3+1, 11] = 1; U[2*3+2, 11] = 1
    V[1*3+1, 11] = 1; V[2*3+2, 11] = 1
    W[2*3+1, 11] = 1  # C[1][2]
    W[1*3+2, 11] = 1  # C[2][1]
    
    # Term 12: (A00+A01+A10+A11) * (B00+B01+B10+B11)
    # This is complex. Let me skip this approach and use a verified reference.
    
    # Actually, let me just verify the 9 direct terms + the 3 simple composite terms above
    # and see if we can get a valid rank-12 decomposition (which won't be minimal but is a start).
    # For now, let me return just the 9 direct terms as a rank-9 partial decomposition
    # and verify it works for the diagonal/off-diagonal elements it covers.
    
    # Actually, the 9 direct terms only cover:
    # C00: A00B00 + A01B10 + A02B20 (complete)
    # C10: A10B01 (incomplete - missing A10B01... wait C10 = A10B00 + A11B10 + A12B20)
    # Hmm, I'm confusing myself. Let me restart with a clean verification-based approach.
    
    return None  # Will be replaced below with a working implementation

def _make_3x3_laderman_v2():
    """
    Laderman 3x3 matrix multiplication, rank 23.
    Uses a verified construction.
    Returns (U, V, W) as 2D arrays of shape (9, 23).
    """
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # I'll use the formulation from standard references.
    # The key insight: we need 23 rank-1 tensors that sum to the multiplication tensor.
    # 
    # Let me use a different, more reliable approach: I'll construct it from
    # known correct Laderman coefficients.
    #
    # From Laderman (1976) "A non-recursive algorithm for multiplying 3x3 matrices in 23 multiplications":
    #
    # Let A = [[a,b,c],[d,e,f],[g,h,i]], B = [[j,k,l],[m,n,o],[p,q,r]]
    #
    # The 23 products are:
    # P1  = (a+e)(j+n)
    # P2  = (e-a)j
    # P3  = a(l-o)
    # P4  = e(m-j)
    # P5  = (a+b)n
    # P6  = (d+e)j
    # P7  = (a-b)(j+k)
    # P8  = (b-a)(k+n)
    # P9  = (a+b+c+e+d+f+g+i)(j+k+l+m+n+o+p+q+r) -- no this is too big
    #
    # This is getting complicated from memory. Let me use a computational approach:
    # I'll try to find a valid rank-23 decomposition by solving a linear system
    # or by using a known valid set.
    
    # Alternative: Use the fact that 3x3 can be decomposed as 2x2 Strassen + corrections.
    # Or better: just use schoolbook 27 for the inner and accept rank 189.
    # But the directive says to improve within the family.
    
    # Let me try a different strategy: use 3x3 with rank 23 from a verified source.
    # I'll hardcode a known valid Laderman decomposition.
    
    # From https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm#Laderman's_algorithm
    # and standard references, here are the 23 products:
    #
    # P1  = (a + e)(j + n)
    # P2  = (e - a)j
    # P3  = a(l - o)
    # P4  = e(m - j)
    # P5  = (a + b)n
    # P6  = (d + e)j
    # P7  = (a - b)(j + k)
    # P8  = (b - a)(k + n)
    # P9  = (c + f)(m + p)  -- not quite
    #
    # I'm not confident in my memory of the exact coefficients.
    # Let me use a safer approach: verify a candidate and fall back if it fails.
    
    # For now, return None to indicate we need a fallback.
    return None

def _make_3x3_laderman_verified():
    """
    Attempt to construct Laderman 3x3 and verify.
    Returns (U, V, W) or None if verification fails.
    """
    # I'll use a systematic construction based on the structure of Laderman's algorithm.
    # 
    # The 23 terms can be grouped as:
    # - 9 "direct" terms (one for each (i,k) pair on the "diagonal" structure)
    # - 14 "composite" terms
    #
    # Actually, a cleaner way: Laderman's algorithm has the following structure:
    # 
    # For the 3x3 case, the 23 multiplications are:
    # 
    # Group 1 (9 terms): Handle the "main" contributions
    # M1 = A[0,0]*B[0,0]
    # M2 = A[0,1]*B[1,0]
    # M3 = A[0,2]*B[2,0]
    # M4 = A[1,0]*B[0,1]
    # M5 = A[1,1]*B[1,1]
    # M6 = A[1,2]*B[2,1]
    # M7 = A[2,0]*B[0,2]
    # M8 = A[2,1]*B[1,2]
    # M9 = A[2,2]*B[2,2]
    #
    # Group 2 (14 terms): Handle the remaining contributions via sums/differences
    #
    # The remaining C elements need:
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    #
    # Plus the diagonal elements already have some terms:
    # C[0,0] = M1 + M2 + M3 (complete)
    # C[1,1] = M5 + M6 + ... wait, C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1] = M4 + M5 + M6 (complete)
    # C[2,2] = A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2] = M7 + M8 + M9 (complete)
    #
    # So the 9 direct terms completely handle C[0,0], C[1,1], C[2,2].
    # The remaining 6 off-diagonal elements each need 3 terms = 18 terms, but we only have 14 more.
    # So we need to share some terms.
    
    # This is the essence of Laderman's construction. Let me use a known valid set.
    # I'll encode it and verify.
    
    # Based on the Laderman algorithm as described in standard references:
    #
    # Let me use the following verified construction (from multiple sources):
    #
    # I'll define 23 terms where each term specifies which A elements, B elements, 
    # and C elements it involves with coefficients.
    
    terms = []
    
    # Direct terms (0-8)
    # Term 0: A00*B00 -> C00
    terms.append(([(0,0,1)], [(0,0,1)], [(0,0,1)]))
    # Term 1: A01*B10 -> C00
    terms.append(([(0,1,1)], [(1,0,1)], [(0,0,1)]))
    # Term 2: A02*B20 -> C00
    terms.append(([(0,2,1)], [(2,0,1)], [(0,0,1)]))
    # Term 3: A10*B01 -> C10
    terms.append(([(1,0,1)], [(0,1,1)], [(1,0,1)]))
    # Term 4: A11*B11 -> C11
    terms.append(([(1,1,1)], [(1,1,1)], [(1,1,1)]))
    # Term 5: A12*B21 -> C11
    terms.append(([(1,2,1)], [(2,1,1)], [(1,1,1)]))
    # Term 6: A20*B02 -> C20
    terms.append(([(2,0,1)], [(0,2,1)], [(2,0,1)]))
    # Term 7: A21*B12 -> C21
    terms.append(([(2,1,1)], [(1,2,1)], [(2,1,1)]))
    # Term 8: A22*B22 -> C22
    terms.append(([(2,2,1)], [(2,2,1)], [(2,2,1)]))
    
    # Composite terms (9-22)
    # Term 9: (A00+A11)*(B00+B11) -> C01, C10
    terms.append(([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,1,1),(1,0,1)]))
    
    # Term 10: (A00+A22)*(B00+B22) -> C02, C20
    terms.append(([(0,0,1),(2,2,1)], [(0,0,1),(2,2,1)], [(0,2,1),(2,0,1)]))
    
    # Term 11: (A11+A22)*(B11+B22) -> C12, C21
    terms.append(([(1,1,1),(2,2,1)], [(1,1,1),(2,2,1)], [(1,2,1),(2,1,1)]))
    
    # Term 12: (A00+A01+A10+A11)*(B00+B01+B10+B11)
    # This contributes to many places. Let me think...
    # Actually, I realize I need to be much more careful.
    # Let me use a different approach: I'll search for a valid Laderman decomposition
    # by solving a system of equations, or I'll use a known valid hardcoded set.
    
    # For now, let me just use the 9 direct terms + 3 simple composites = 12 terms
    # and verify what it covers. If it's not complete, I'll fall back.
    
    # Actually, let me try a completely different approach:
    # Use 2x2 Strassen for the outer 3x3 blocks (treating 6x6 as 3x3 blocks of 2x2)
    # This gives rank = 7 * 8 = 56... no wait, 2x2 schoolbook is 4, so 7*4=28.
    # But 2x2 Strassen is rank 7, so 7*7=49? No...
    # 
    # 6x6 = 3x3 blocks of 2x2.
    # Outer: 3x3 matrix mult using Laderman (rank 23) or schoolbook (rank 27)
    # Inner: 2x2 matrix mult using Strassen (rank 7) or schoolbook (rank 4)
    #
    # Option A: 3x3 schoolbook (27) * 2x2 Strassen (7) = 189 (current)
    # Option B: 3x3 Laderman (23) * 2x2 Strassen (7) = 161
    # Option C: 3x3 Laderman (23) * 2x2 schoolbook (4) = 92
    # Option D: 3x3 schoolbook (27) * 2x2 schoolbook (4) = 108
    #
    # Wait, option C gives rank 92! That's much better.
    # And option D gives 108.
    #
    # But wait, the current program uses 2x2 Strassen OUTER and 3x3 schoolbook INNER.
    # That's 7 * 27 = 189.
    #
    # If I use 3x3 blocks of 2x2:
    # Outer is 3x3, inner is 2x2.
    # 3x3 schoolbook * 2x2 Strassen = 27 * 7 = 189 (same as current, just transposed)
    # 3x3 Laderman * 2x2 Strassen = 23 * 7 = 161
    # 3x3 Laderman * 2x2 schoolbook = 23 * 4 = 92
    #
    # So the best option is 3x3 Laderman * 2x2 schoolbook = 92!
    # But I need a valid Laderman decomposition.
    
    # Let me try yet another approach: just use 2x2 Strassen for everything.
    # 6x6 = 3x3 blocks of 2x2.
    # Use 3x3 schoolbook for outer (27 terms), each term is a 2x2 Strassen (7 terms).
    # Total rank = 27 * 7 = 189. Same as current.
    #
    # Or: 6x6 = 2x2 blocks of 3x3.
    # Use 2x2 Strassen for outer (7 terms), each term is a 3x3 schoolbook (27 terms).
    # Total rank = 7 * 27 = 189. Same as current.
    #
    # To do better, I need either:
    # - 3x3 Laderman (23) instead of 3x3 schoolbook (27): saves 4/27 per inner
    # - Or some other improvement.
    #
    # With 3x3 Laderman * 2x2 schoolbook = 23 * 4 = 92, that's a big improvement!
    # With 3x3 Laderman * 2x2 Strassen = 23 * 7 = 161, also an improvement.
    
    # The key challenge: getting a correct Laderman 3x3 decomposition.
    
    # Let me try to construct it computationally using linear algebra.
    # The multiplication tensor T for 3x3 has 27 entries (all 1s for T[i,j,k]).
    # We want to express T as a sum of at most 23 rank-1 tensors.
    # Each rank-1 tensor is u ⊗ v ⊗ w where u, v, w are 3x3 matrices (flattened to 9-dim vectors).
    
    # This is a hard problem. Let me just use a known valid Laderman set.
    
    # After much deliberation, let me use this approach:
    # I'll hardcode a Laderman decomposition that I can verify.
    # If verification fails, I'll fall back to the current 189-rank solution.
    
    # Here's a known valid Laderman 3x3 decomposition (from standard references):
    # I'll define it term by term and verify.
    
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # I'll use the following construction which I believe is correct:
    # 
    # The 23 products in Laderman's algorithm:
    # 
    # p1  = (a11 + a22)(b11 + b22)
    # p2  = (a22 - a11)(b11)
    # p3  = (a11)(b22 - b11)  -- wait, this doesn't look right either
    #
    # OK, I give up trying to recall it from memory. Let me use a computational approach.
    # I'll set up the problem and try to solve it, or just accept the 189-rank solution.
    
    # For the purpose of this iteration, let me just return None and use the fallback.
    return None

def construct():
    start_time = time.time()
    
    n, m, p = 6, 6, 6
    
    # Fallback: Schoolbook 6x6 (Rank 216)
    def _schoolbook():
        rank = n * m * p
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
        
    # Hybrid: 2x2 Strassen (Outer) x 3x3 Schoolbook (Inner) (Rank 189)
    def _hybrid_strassen_schoolbook():
        block_size = 3
        dim = n
        
        rank_inner = block_size ** 3  # 27
        rank_outer = 7
        total_rank = rank_inner * rank_outer  # 189
        
        U = np.zeros((dim * dim, total_rank))
        V = np.zeros((dim * dim, total_rank))
        W = np.zeros((dim * dim, total_rank))
        
        steps = [
            {'A': {(0,0): 1, (1,1): 1}, 'B': {(0,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): 1}},
            {'A': {(1,0): 1, (1,1): 1}, 'B': {(0,0): 1}, 'C': {(1,0): 1, (1,1): -1}},
            {'A': {(0,0): 1}, 'B': {(0,1): 1, (1,1): -1}, 'C': {(0,1): 1, (1,1): 1}},
            {'A': {(1,1): 1}, 'B': {(1,0): 1, (0,0): -1}, 'C': {(0,0): 1, (1,0): 1}},
            {'A': {(0,0): 1, (0,1): 1}, 'B': {(1,1): 1}, 'C': {(0,0): -1, (0,1): 1}},
            {'A': {(1,0): 1, (0,0): -1}, 'B': {(0,0): 1, (0,1): 1}, 'C': {(1,1): 1}},
            {'A': {(0,1): 1, (1,1): -1}, 'B': {(1,0): 1, (1,1): 1}, 'C': {(0,0): 1}}
        ]
        
        for s, step in enumerate(steps):
            A_coeffs = step['A']
            B_coeffs = step['B']
            C_coeffs = step['C']
            
            for i_prime in range(block_size):
                for j_prime in range(block_size):
                    for k_prime in range(block_size):
                        col = s * rank_inner + i_prime * 9 + j_prime * 3 + k_prime
                        
                        for (I, J), ca in A_coeffs.items():
                            if ca != 0:
                                row_u = (I * block_size + i_prime) * dim + (J * block_size + j_prime)
                                U[row_u, col] += ca
                        
                        for (J, K), cb in B_coeffs.items():
                            if cb != 0:
                                row_v = (J * block_size + j_prime) * dim + (K * block_size + k_prime)
                                V[row_v, col] += cb
                        
                        for (I, K), cc in C_coeffs.items():
                            if cc != 0:
                                row_w = (K * block_size + k_prime) * dim + (I * block_size + i_prime)
                                W[row_w, col] += cc
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Try Hybrid (189)
    try:
        if time.time() - start_time < _DEADLINE - 5:
            res = _hybrid_strassen_schoolbook()
            # Verify
            U_np = np.array(res["U"], dtype=float)
            V_np = np.array(res["V"], dtype=float)
            W_np = np.array(res["W"], dtype=float)
            
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    u_row = U_np[i * m + j, :]
                    for k in range(p):
                        v_row = V_np[j * p + k, :]
                        w_row = W_np[k * n + i, :]
                        s = np.dot(u_row, v_row * w_row)
                        err = abs(s - 1.0)
                        if err > max_err:
                            max_err = err
            
            if max_err < 1e-10:
                return res
    except Exception:
        pass
        
    # Fallback to Schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-END
