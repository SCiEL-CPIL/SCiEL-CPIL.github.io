"""Matrix multiplication 3,3,6: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    deadline = time.time() + 145.0
    
    # We use Laderman's rank-23 decomposition for 3x3 x 3x3.
    # Since p=6, we split the second matrix into two 3x3 blocks: B = [B1 | B2]
    # C = A * B = A * B1 + A * B2
    # We use Laderman for A*B1 and A*B2 independently, giving rank 23 + 23 = 46.
    
    # Laderman's 23 products for C = A * B where A, B are 3x3:
    # A = [[a, b, c], [d, e, f], [g, h, i]]
    # B = [[j, k, l], [m, n, o], [p, q, r]]
    
    # The 23 linear forms for A (rows of U contribution) and B (rows of V contribution)
    # and the reconstruction coefficients for C (W contribution)
    
    # Let's define the 23 multiplications M_t = (sum_a alpha_{t,a} * A_a) * (sum_b beta_{t,b} * B_b)
    # Then C_c = sum_t gamma_{t,c} * M_t
    
    # Indexing: A row-major: a=A[0,0], b=A[0,1], c=A[0,2], d=A[1,0], e=A[1,1], f=A[1,2], g=A[2,0], h=A[2,1], i=A[2,2]
    # B row-major: j=B[0,0], k=B[0,1], l=B[0,2], m=B[1,0], n=B[1,1], o=B[1,2], p=B[2,0], q=B[2,1], r=B[2,2]
    # C row-major: C[0,0]=M1+M4+M8+M10+M14-M9-M13-M17-M22
    #              C[0,1]=M5+M7+M11+M15-M6-M12-M16-M20
    #              C[0,2]=M2+M3+M13-M1-M4-M8-M10-M14
    #              C[1,0]=M9+M17+M22-M8-M10-M14
    #              C[1,1]=M6+M16+M20-M5-M7-M11-M15
    #              C[1,2]=M12+M20-M2-M3-M13-M17-M22
    #              C[2,0]=M10+M14-M9-M13
    #              C[2,1]=M7+M15-M6-M12
    #              C[2,2]=M1+M4-M8
    
    # Define alpha[t] as dict from A-index to coefficient
    # A-indices: 0=a,1=b,2=c,3=d,4=e,5=f,6=g,7=h,8=i
    
    def idx_A(i, j): return i * 3 + j
    def idx_B(i, j): return i * 3 + j
    def idx_C(i, j): return i * 3 + j
    
    # Laderman's decomposition (23 terms)
    # Each term t: alpha (9-vector for A), beta (9-vector for B), gamma (9-vector for C)
    
    terms = []
    
    # M1 = a * (j - m + p + q)
    alpha = [1,0,0, 0,0,0, 0,0,0]
    beta = [1,0,0, -1,0,0, 1,1,0]
    gamma = [1,0,0, 0,0,0, 0,0,1]  # C[0,0]+=M1, C[2,2]+=M1
    terms.append((alpha, beta, gamma))
    
    # M2 = (a+b) * m
    alpha = [1,1,0, 0,0,0, 0,0,0]
    beta = [0,0,0, 1,0,0, 0,0,0]
    gamma = [0,0,1, 0,0,0, 0,0,0]  # C[0,2]+=M2, C[1,2]-=M2
    terms.append((alpha, beta, gamma))
    
    # M3 = (a+c) * p
    alpha = [1,0,1, 0,0,0, 0,0,0]
    beta = [0,0,0, 0,0,0, 1,0,0]
    gamma = [0,0,1, 0,0,0, 0,0,0]  # C[0,2]+=M3, C[1,2]-=M3
    terms.append((alpha, beta, gamma))
    
    # M4 = i * (j + k - m - n)
    alpha = [0,0,0, 0,0,0, 0,0,1]
    beta = [1,1,0, -1,-1,0, 0,0,0]
    gamma = [1,0,0, 0,0,0, 0,0,1]  # C[0,0]+=M4, C[0,2]-=M4, C[2,2]+=M4
    terms.append((alpha, beta, gamma))
    
    # M5 = (g+i) * k
    alpha = [0,0,0, 0,0,0, 1,0,1]
    beta = [0,1,0, 0,0,0, 0,0,0]
    gamma = [0,1,0, 0,0,0, 0,0,0]  # C[0,1]+=M5, C[1,1]-=M5
    terms.append((alpha, beta, gamma))
    
    # M6 = (g+i) * n
    alpha = [0,0,0, 0,0,0, 1,0,1]
    beta = [0,0,0, 0,1,0, 0,0,0]
    gamma = [0,0,0, 0,1,0, 0,0,0]  # C[0,1]-=M6, C[1,1]+=M6
    terms.append((alpha, beta, gamma))
    
    # M7 = (c+i) * (m + n - p - q)
    alpha = [0,1,0, 0,0,0, 0,0,1]
    beta = [0,0,0, 1,1,0, -1,-1,0]
    gamma = [0,1,0, 0,0,0, 0,1,0]  # C[0,1]+=M7, C[1,1]-=M7, C[2,1]+=M7
    terms.append((alpha, beta, gamma))
    
    # M8 = (a+c+g+i-e) * (j+r)
    alpha = [1,0,1, 0,-1,0, 1,0,1]
    beta = [1,0,0, 0,0,0, 0,0,1]
    gamma = [1,0,0, 0,0,0, 0,0,-1]  # C[0,0]+=M8, C[1,0]-=M8, C[2,2]-=M8
    terms.append((alpha, beta, gamma))
    
    # M9 = (c+i-e) * (j+r)
    alpha = [0,0,1, 0,-1,0, 0,0,1]
    beta = [1,0,0, 0,0,0, 0,0,1]
    gamma = [0,0,0, 1,0,0, 0,0,0]  # C[0,0]-=M9, C[1,0]+=M9, C[2,0]-=M9
    terms.append((alpha, beta, gamma))
    
    # M10 = (c+a-e) * (p+r)
    alpha = [1,0,1, 0,-1,0, 0,0,0]
    beta = [0,0,0, 0,0,0, 1,0,1]
    gamma = [1,0,0, 0,0,0, 1,0,0]  # C[0,0]+=M10, C[1,0]-=M10, C[2,0]+=M10
    terms.append((alpha, beta, gamma))
    
    # M11 = (c+a) * (p+r-m-o)
    alpha = [1,0,1, 0,0,0, 0,0,0]
    beta = [0,0,0, -1,0,-1, 1,0,1]
    gamma = [0,1,0, 0,0,0, 0,0,0]  # C[0,1]+=M11, C[1,1]-=M11
    terms.append((alpha, beta, gamma))
    
    # M12 = (g+i) * (k+l-n-o)
    alpha = [0,0,0, 0,0,0, 1,0,1]
    beta = [0,1,1, 0,-1,-1, 0,0,0]
    gamma = [0,0,0, 0,0,1, 0,0,0]  # C[0,1]-=M12, C[1,2]+=M12, C[2,1]-=M12
    terms.append((alpha, beta, gamma))
    
    # M13 = (g+i-e) * (k+l)
    alpha = [0,0,0, 0,-1,0, 1,0,1]
    beta = [0,1,1, 0,0,0, 0,0,0]
    gamma = [0,0,1, 0,0,0, 0,0,0]  # C[0,0]-=M13, C[0,2]+=M13, C[1,2]-=M13, C[2,0]-=M13
    terms.append((alpha, beta, gamma))
    
    # M14 = (c+i-e) * (j+l)
    alpha = [0,0,1, 0,-1,0, 0,0,1]
    beta = [1,0,1, 0,0,0, 0,0,0]
    gamma = [1,0,0, 0,0,0, 1,0,0]  # C[0,0]+=M14, C[1,0]-=M14, C[2,0]+=M14
    terms.append((alpha, beta, gamma))
    
    # M15 = (c+i) * (j+l-m-o)
    alpha = [0,0,1, 0,0,0, 0,0,1]
    beta = [1,0,1, -1,0,-1, 0,0,0]
    gamma = [0,1,0, 0,0,0, 0,1,0]  # C[0,1]+=M15, C[1,1]-=M15, C[2,1]+=M15
    terms.append((alpha, beta, gamma))
    
    # M16 = b * (n-q)
    alpha = [0,1,0, 0,0,0, 0,0,0]
    beta = [0,0,0, 0,1,0, 0,-1,0]
    gamma = [0,0,0, 0,1,0, 0,0,0]  # C[0,1]-=M16, C[1,1]+=M16
    terms.append((alpha, beta, gamma))
    
    # M17 = d * (j-m)
    alpha = [0,0,0, 1,0,0, 0,0,0]
    beta = [1,0,0, -1,0,0, 0,0,0]
    gamma = [0,0,0, 1,0,0, 0,0,0]  # C[0,0]-=M17, C[1,0]+=M17, C[1,2]-=M17
    terms.append((alpha, beta, gamma))
    
    # M18 = e * (n-q)  # Wait, let me re-check. The original says M18 = e*(n-q)
    # Actually looking at the reconstruction, M18 doesn't appear. Let me re-examine.
    # From the reconstruction formulas given, I see M1-M23 but let me verify M18.
    # C[1,1] = M6+M16+M20-M5-M7-M11-M15. No M18.
    # Hmm, maybe M18 is not used? That seems odd for a rank-23 decomposition.
    # Let me re-check the standard Laderman algorithm.
    
    # Actually, I may have made errors in transcribing. Let me use a verified Laderman decomposition.
    # Standard Laderman 3x3x3 uses 23 multiplications. Let me use a known correct set.
    
    # Let me restart with a cleaner, verified Laderman implementation.
    
    # The Laderman algorithm for C = A*B (3x3):
    # M1 = (a+b-c-d-e-f+g+h-i)*(j+k-l-m+n-o+p+q-r)  # This is getting complex
    # 
    # Let me use a different approach: I'll hardcode the known Laderman coefficients
    # from a reliable source.
    
    # Actually, the simplest approach for this task: use the schoolbook for 3x3x6 directly
    # but look for structure. However, Laderman for 3x3 gives rank 23, so two copies give 46.
    # Schoolbook gives 54. So 46 < 54.
    
    # Let me verify my Laderman transcription by checking if it's consistent.
    # The issue is I need a CORRECT Laderman decomposition.
    
    # Let me use a well-known formulation. I'll define it carefully.
    
    # A = [[a11, a12, a13],
    #      [a21, a22, a23],
    #      [a31, a32, a33]]
    # B = [[b11, b12, b13],
    #      [b21, b22, b23],
    #      [b31, b32, b33]]
    
    # Laderman's 23 products:
    # P1 = a11 * (b11 - b21 + b31 + b32)
    # P2 = (a11 + a12) * b21
    # P3 = (a11 + a13) * b31
    # P4 = a33 * (b11 + b12 - b21 - b22)
    # P5 = (a31 + a33) * b12
    # P6 = (a31 + a33) * b22
    # P7 = (a13 + a33) * (b21 + b22 - b31 - b32)
    # P8 = (a11 + a13 + a31 + a33 - a22) * (b11 + b33)
    # P9 = (a13 + a33 - a22) * (b11 + b33)
    # P10 = (a11 + a13 - a22) * (b31 + b33)
    # P11 = (a11 + a13) * (b31 + b33 - b21 - b23)
    # P12 = (a31 + a33) * (b12 + b13 - b22 - b23)
    # P13 = (a31 + a33 - a22) * (b12 + b13)
    # P14 = (a13 + a33 - a22) * (b11 + b13)
    # P15 = (a13 + a33) * (b11 + b13 - b21 - b23)
    # P16 = a12 * (b22 - b32)
    # P17 = a21 * (b11 - b21)
    # P18 = a22 * (b22 - b32)
    # P19 = a23 * (b31 - b33)
    # P20 = a32 * (b12 - b13)
    # P21 = a32 * (b22 - b23)
    # P22 = a21 * (b22 + b23)
    # P23 = a23 * (b11 + b13)
    
    # C = A*B:
    # C11 = P1 + P4 + P8 + P10 + P14 - P9 - P13 - P17 - P22
    # C12 = P5 + P7 + P11 + P15 - P6 - P12 - P16 - P20
    # C13 = P2 + P3 + P13 - P1 - P4 - P8 - P10 - P14
    # C21 = P9 + P17 + P22 - P8 - P10 - P14
    # C22 = P6 + P16 + P20 - P5 - P7 - P11 - P15
    # C23 = P12 + P20 - P2 - P3 - P13 - P17 - P22
    # C31 = P10 + P14 - P9 - P13
    # C32 = P7 + P15 - P6 - P12
    # C33 = P1 + P4 - P8
    
    # Let me encode this properly.
    
    # I'll build the U, V, W for 3x3x3 first, then duplicate for the two 3x3 blocks.
    
    # For 3x3x3:
    # U has shape (9, rank), V has shape (9, rank), W has shape (9, rank)
    # U[i*3+j, t] = coefficient of A[i,j] in product t
    # V[i*3+j, t] = coefficient of B[i,j] in product t
    # W[i*3+j, t] = coefficient of product t in C[i,j]
    
    # Let's define the 23 terms with their alpha, beta, gamma vectors.
    
    terms23 = []
    
    # P1: a11 * (b11 - b21 + b31 + b32)
    alpha = [1,0,0, 0,0,0, 0,0,0]  # a11
    beta = [1,0,0, -1,0,0, 1,1,0]  # b11 - b21 + b31 + b32
    gamma = [1,0,0, 0,0,0, 0,0,1]  # C11 += P1, C33 += P1
    terms23.append((alpha, beta, gamma))
    
    # P2: (a11 + a12) * b21
    alpha = [1,1,0, 0,0,0, 0,0,0]
    beta = [0,0,0, 1,0,0, 0,0,0]
    gamma = [0,0,1, 0,0,0, 0,0,0]  # C13 += P2, C23 -= P2
    terms23.append((alpha, beta, gamma))
    
    # P3: (a11 + a13) * b31
    alpha = [1,0,1, 0,0,0, 0,0,0]
    beta = [0,0,0, 0,0,0, 1,0,0]
    gamma = [0,0,1, 0,0,0, 0,0,0]  # C13 += P3, C23 -= P3
    terms23.append((alpha, beta, gamma))
    
    # P4: a33 * (b11 + b12 - b21 - b22)
    alpha = [0,0,0, 0,0,0, 0,0,1]
    beta = [1,1,0, -1,-1,0, 0,0,0]
    gamma = [1,0,0, 0,0,0, 0,0,1]  # C11 += P4, C13 -= P4, C33 += P4
    terms23.append((alpha, beta, gamma))
    
    # P5: (a31 + a33) * b12
    alpha = [0,0,0, 0,0,0, 1,0,1]
    beta = [0,1,0, 0,0,0, 0,0,0]
    gamma = [0,1,0, 0,0,0, 0,0,0]  # C12 += P5, C22 -= P5
    terms23.append((alpha, beta, gamma))
    
    # P6: (a31 + a33) * b22
    alpha = [0,0,0, 0,0,0, 1,0,1]
    beta = [0,0,0, 0,1,0, 0,0,0]
    gamma = [0,0,0, 0,1,0, 0,0,0]  # C12 -= P6, C22 += P6
    terms23.append((alpha, beta, gamma))
    
    # P7: (a13 + a33) * (b21 + b22 - b31 - b32)
    alpha = [0,0,1, 0,0,0, 0,0,1]
    beta = [0,0,0, 1,1,0, -1,-1,0]
    gamma = [0,1,0, 0,0,0, 0,1,0]  # C12 += P7, C22 -= P7, C32 += P7
    terms23.append((alpha, beta, gamma))
    
    # P8: (a11 + a13 + a31 + a33 - a22) * (b11 + b33)
    alpha = [1,0,1, 0,-1,0, 1,0,1]
    beta = [1,0,0, 0,0,0, 0,0,1]
    gamma = [1,0,0, 0,0,0, 0,0,-1]  # C11 += P8, C21 -= P8, C33 -= P8
    terms23.append((alpha, beta, gamma))
    
    # P9: (a13 + a33 - a22) * (b11 + b33)
    alpha = [0,0,1, 0,-1,0, 0,0,1]
    beta = [1,0,0, 0,0,0, 0,0,1]
    gamma = [0,0,0, 1,0,0, 0,0,0]  # C11 -= P9, C21 += P9, C31 -= P9
    terms23.append((alpha, beta, gamma))
    
    # P10: (a11 + a13 - a22) * (b31 + b33)
    alpha = [1,0,1, 0,-1,0, 0,0,0]
    beta = [0,0,0, 0,0,0, 1,0,1]
    gamma = [1,0,0, 0,0,0, 1,0,0]  # C11 += P10, C21 -= P10, C31 += P10
    terms23.append((alpha, beta, gamma))
    
    # P11: (a11 + a13) * (b31 + b33 - b21 - b23)
    alpha = [1,0,1, 0,0,0, 0,0,0]
    beta = [0,0,0, -1,0,-1, 1,0,1]
    gamma = [0,1,0, 0,0,0, 0,0,0]  # C12 += P11, C22 -= P11
    terms23.append((alpha, beta, gamma))
    
    # P12: (a31 + a33) * (b12 + b13 - b22 - b23)
    alpha = [0,0,0, 0,0,0, 1,0,1]
    beta = [0,1,1, 0,-1,-1, 0,0,0]
    gamma = [0,0,0, 0,0,1, 0,0,0]  # C12 -= P12, C23 += P12, C32 -= P12
    terms23.append((alpha, beta, gamma))
    
    # P13: (a31 + a33 - a22) * (b12 + b13)
    alpha = [0,0,0, 0,-1,0, 1,0,1]
    beta = [0,1,1, 0,0,0, 0,0,0]
    gamma = [0,0,1, 0,0,0, 0,0,0]  # C11 -= P13, C13 += P13, C23 -= P13, C31 -= P13
    terms23.append((alpha, beta, gamma))
    
    # P14: (a13 + a33 - a22) * (b11 + b13)
    alpha = [0,0,1, 0,-1,0, 0,0,1]
    beta = [1,0,1, 0,0,0, 0,0,0]
    gamma = [1,0,0, 0,0,0, 1,0,0]  # C11 += P14, C21 -= P14, C31 += P14
    terms23.append((alpha, beta, gamma))
    
    # P15: (a13 + a33) * (b11 + b13 - b21 - b23)
    alpha = [0,0,1, 0,0,0, 0,0,1]
    beta = [1,0,1, -1,0,-1, 0,0,0]
    gamma = [0,1,0, 0,0,0, 0,1,0]  # C12 += P15, C22 -= P15, C32 += P15
    terms23.append((alpha, beta, gamma))
    
    # P16: a12 * (b22 - b32)
    alpha = [0,1,0, 0,0,0, 0,0,0]
    beta = [0,0,0, 0,1,0, 0,-1,0]
    gamma = [0,0,0, 0,1,0, 0,0,0]  # C12 -= P16, C22 += P16
    terms23.append((alpha, beta, gamma))
    
    # P17: a21 * (b11 - b21)
    alpha = [0,0,0, 1,0,0, 0,0,0]
    beta = [1,0,0, -1,0,0, 0,0,0]
    gamma = [0,0,0, 1,0,0, 0,0,0]  # C11 -= P17, C21 += P17, C23 -= P17
    terms23.append((alpha, beta, gamma))
    
    # P18: a22 * (b22 - b32)
    alpha = [0,0,0, 0,1,0, 0,0,0]
    beta = [0,0,0, 0,1,0, 0,-1,0]
    gamma = [0,0,0, 0,0,0, 0,0,0]  # P18 is not used in reconstruction? Let me check.
    # Looking at the reconstruction, P18 doesn't appear. This is suspicious.
    # Maybe my transcription is wrong. Let me just use this and verify numerically.
    terms23.append((alpha, beta, gamma))
    
    # P19: a23 * (b31 - b33)
    alpha = [0,0,0, 0,0,1, 0,0,0]
    beta = [0,0,0, 0,0,0, 1,0,-1]
    gamma = [0,0,0, 0,0,0, 0,0,0]  # P19 also not in reconstruction
    terms23.append((alpha, beta, gamma))
    
    # P20: a32 * (b12 - b13)
    alpha = [0,0,0, 0,0,0, 0,1,0]
    beta = [0,1,-1, 0,0,0, 0,0,0]
    gamma = [0,0,0, 0,1,0, 0,0,0]  # C12 -= P20, C22 += P20, C23 += P20
    terms23.append((alpha, beta, gamma))
    
    # P21: a32 * (b22 - b23)
    alpha = [0,0,0, 0,0,0, 0,1,0]
    beta = [0,0,0, 0,1,-1, 0,0,0]
    gamma = [0,0,0, 0,0,0, 0,0,0]  # P21 not in reconstruction
    terms23.append((alpha, beta, gamma))
    
    # P22: a21 * (b22 + b23)
    alpha = [0,0,0, 1,0,0, 0,0,0]
    beta = [0,0,0, 0,1,1, 0,0,0]
    gamma = [0,0,0, 0,0,0, 0,0,0]  # C11 -= P22, C21 += P22, C23 -= P22
    terms23.append((alpha, beta, gamma))
    
    # P23: a23 * (b11 + b13)
    alpha = [0,0,0, 0,0,1, 0,0,0]
    beta = [1,0,1, 0,0,0, 0,0,0]
    gamma = [0,0,0, 0,0,0, 0,0,0]  # P23 not in reconstruction
    terms23.append((alpha, beta, gamma))
    
    # This looks like my Laderman transcription has issues with P18-P23 not appearing.
    # This suggests my transcription is incorrect.
    
    # Given the time constraint and the complexity of verifying Laderman by hand,
    # let me take a safer approach: use the schoolbook decomposition but with a 
    # different structure that might allow for some optimization.
    
    # Actually, the safest bet is to just return the schoolbook rank-54 solution,
    # but that scores 1.0. To do better, I need a valid lower-rank decomposition.
    
    # Let me try a different strategy: use the fact that for 3x3x6, we can view it
    # as 3x3x3 + 3x3x3, and if I can get a valid Laderman, I get rank 46.
    
    # Let me verify my Laderman numerically before using it.
    
    def verify_laderman():
        """Verify the Laderman decomposition numerically."""
        np.random.seed(42)
        A = np.random.rand(3, 3)
        B = np.random.rand(3, 3)
        C_expected = A @ B
        
        # Build U, V, W for 3x3x3
        U3 = np.zeros((9, 23))
        V3 = np.zeros((9, 23))
        W3 = np.zeros((9, 23))
        
        for t, (alpha, beta, gamma) in enumerate(terms23):
            U3[:, t] = alpha
            V3[:, t] = beta
            W3[:, t] = gamma
        
        # C = U @ diag(1) @ V^T ... no, that's not right.
        # The tensor decomposition: T_{i,j,k} = sum_t U_{i,t} V_{j,t} W_{k,t}
        # Where T_{i,j,k} = 1 if C[i,j] depends on A[i',j'] * B[i'',j''] appropriately.
        # Actually, for matrix multiplication: (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
        # So the tensor has T_{(i,j),(j,k),(i,k)} = 1 for appropriate indices.
        # In our factorization: C_{i,k} = sum_t U_{i,t} * (sum_j V_{j,t} B_{j,k}) * W_{i,k,t}
        # Wait, the standard form is: C_{p,q} = sum_r U_{p,r} V_{q,r} W_{r,r} ... no.
        
        # The correct interpretation: 
        # C_{i,k} = sum_j sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        # where n=3, m=3, p=3 for the 3x3x3 case.
        
        C_computed = np.zeros((3, 3))
        for i in range(3):
            for k in range(3):
                s = 0.0
                for j in range(3):
                    for t in range(23):
                        s += U3[i*3+j, t] * V3[j*3+k, t] * W3[k*3+i, t]
                C_computed[i, k] = s
        
        return np.allclose(C_computed, C_expected, atol=1e-10)
    
    if verify_laderman():
        # Great, Laderman is correct. Use it for 3x3x6.
        rank = 46
        U = np.zeros((n*m, rank))  # (9, 46)
        V = np.zeros((m*p, rank))  # (18, 46)
        W = np.zeros((p*n, rank))  # (18, 46)
        
        # For the first 3x3 block (B1 = B[:, 0:3]):
        # U[i*3+j, t] = alpha_t[i,j]
        # V[j*3+k, t] = beta_t[j,k] for k in 0..2
        # W[k*3+i, t] = gamma_t[k,i] for k in 0..2
        
        for t, (alpha, beta, gamma) in enumerate(terms23):
            # First block
            for i in range(3):
                for j in range(3):
                    U[i*3+j, t] = alpha[i*3+j]
            for j in range(3):
                for k in range(3):
                    V[j*3+k, t] = beta[j*3+k]
            for k in range(3):
                for i in range(3):
                    W[k*3+i, t] = gamma[k*3+i]
            
            # Second block (t+23)
            for i in range(3):
                for j in range(3):
                    U[i*3+j, t+23] = alpha[i*3+j]
            for j in range(3):
                for k in range(3):
                    # B2 has columns 3..5, so B2[j,k] corresponds to B[j, k+3]
                    # In V, row index is j*p + k' where k' is the column within the block
                    # But V has shape (m*p, rank) = (18, 46)
                    # V[j*6 + k, t+23] should be beta[j*3+k] for k in 0..2
                    V[j*6 + (k+3), t+23] = beta[j*3+k]
            for k in range(3):
                for i in range(3):
                    # W has shape (p*n, rank) = (18, 46)
                    # W[k'*n + i, t+23] where k' is the column index in B
                    # For the second block, k' = k+3
                    W[(k+3)*3 + i, t+23] = gamma[k*3+i]
        
        # Convert to lists
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
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
# EVOLVE-BLOCK-END
