import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def _strassen_2x2(A, B):
    """
    Compute A * B for 2x2 matrices using Strassen's algorithm.
    Returns the 2x2 result and the 7 multiplication factors (U, V, W parts).
    Actually, we need to construct the global U, V, W matrices.
    So this function will just return the 7 products and how to combine them.
    """
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-g)
    # m4 = d(g-h)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # P11 = m1 + m4 - m5 + m7
    # P12 = m3 + m5
    # P21 = m2 + m4
    # P22 = m1 - m2 + m3 + m6
    
    return None # Placeholder, we will construct the global matrices directly

def construct():
    n, m, p = DIMENSIONS
    # We construct a rank-26 decomposition using block partitioning.
    # M = [[A, B], [C, D]] where A is 2x2, B is 2x1, C is 1x2, D is 1x1
    # N = [[E, F], [G, H]] where E is 2x2, F is 2x1, G is 1x2, H is 1x1
    
    # Indices for M (9 elements):
    # 0: A00, 1: A01, 2: A10, 3: A11
    # 4: B0, 5: B1
    # 6: C0, 7: C1
    # 8: D
    
    # Indices for N (9 elements):
    # 0: E00, 1: E01, 2: E10, 3: E11
    # 4: F0, 5: F1
    # 6: G0, 7: G1
    # 8: H
    
    # Indices for P (9 elements):
    # P00 = A00*E00 + A01*E10 + B0*G0
    # P01 = A00*E01 + A01*E11 + B0*G1
    # P02 = A00*F0 + A01*F1 + B0*H
    # P10 = A10*E00 + A11*E10 + B1*G0
    # P11 = A10*E01 + A11*E11 + B1*G1
    # P12 = A10*F0 + A11*F1 + B1*H
    # P20 = C0*E00 + C1*E10 + D*G0
    # P21 = C0*E01 + C1*E11 + D*G1
    # P22 = C0*F0 + C1*F1 + D*H
    
    # We use Strassen for A*E (7 mults)
    # Standard for B*G (4 mults)
    # Standard for A*F (4 mults)
    # Standard for C*E (4 mults)
    # Standard for B*H (2 mults)
    # Standard for D*G (2 mults)
    # Standard for C*F (2 mults)
    # Standard for D*H (1 mult)
    # Total: 7 + 4 + 4 + 4 + 2 + 2 + 2 + 1 = 26
    
    rank = 26
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # 1. Strassen for A * E
    # A = [[m0, m1], [m2, m3]], E = [[n0, n1], [n2, n3]]
    # m1_str = (m0+m3)(n0+n3)
    U[0, col] = 1; U[3, col] = 1
    V[0, col] = 1; V[3, col] = 1
    W[0, col] = 1; W[3, col] = 1; W[4, col] = 1; W[8, col] = 1 # P00, P11, P22? No.
    # P00 gets m1, P11 gets m1, P22 gets m1?
    # P00 = A00*E00 + ...
    # P11 = A10*E01 + ...
    # P22 = C0*F0 + ...
    # Strassen result P_AE:
    # P_AE00 = m1 + m4 - m5 + m7
    # P_AE01 = m3 + m5
    # P_AE10 = m2 + m4
    # P_AE11 = m1 - m2 + m3 + m6
    
    # Let's map the 7 Strassen mults to columns.
    # Mult 1: (A00+A11)(E00+E11)
    U[0, col] = 1; U[3, col] = 1
    V[0, col] = 1; V[3, col] = 1
    W[0, col] = 1; W[4, col] = 1; W[8, col] = 1 # P00, P11, P22?
    # P00 uses P_AE00. P11 uses P_AE11. P22 does NOT use A*E.
    # P22 = C*F + D*H.
    # So W[8] should be 0.
    W[0, col] = 1; W[4, col] = 1
    col += 1
    
    # Mult 2: (A10+A11)E00
    U[2, col] = 1; U[3, col] = 1
    V[0, col] = 1
    W[4, col] = 1; W[8, col] = 1 # P11, P22? No.
    # P_AE10 = m2 + m4. P_AE11 = m1 - m2 + ...
    # P10 uses P_AE10. P11 uses P_AE11.
    # P10 is index 3? No.
    # P indices:
    # 0: P00, 1: P01, 2: P02
    # 3: P10, 4: P11, 5: P12
    # 6: P20, 7: P21, 8: P22
    
    # P10 (index 3) gets P_AE10.
    # P11 (index 4) gets P_AE11.
    W[3, col] = 1; W[4, col] = 1
    col += 1
    
    # Mult 3: A00(E01-E10)
    U[0, col] = 1
    V[1, col] = 1; V[2, col] = -1
    W[1, col] = 1; W[4, col] = 1 # P01, P11
    col += 1
    
    # Mult 4: A11(E10-E11)
    U[3, col] = 1
    V[2, col] = 1; V[3, col] = -1
    W[3, col] = 1; W[4, col] = 1 # P10, P11
    col += 1
    
    # Mult 5: (A00+A01)E11
    U[0, col] = 1; U[1, col] = 1
    V[3, col] = 1
    W[0, col] = 1; W[1, col] = 1 # P00, P01
    col += 1
    
    # Mult 6: (A10-A00)(E00+E01)
    U[2, col] = 1; U[0, col] = -1
    V[0, col] = 1; V[1, col] = 1
    W[4, col] = 1 # P11
    col += 1
    
    # Mult 7: (A01-A11)(E10+E11)
    U[1, col] = 1; U[3, col] = -1
    V[2, col] = 1; V[3, col] = 1
    W[0, col] = 1; W[1, col] = 1 # P00, P01
    col += 1
    
    # 2. B * G (4 mults)
    # B = [m4, m5], G = [n6, n7]
    # P00 += B0*G0 (m4*n6)
    U[4, col] = 1; V[6, col] = 1; W[0, col] = 1; col += 1
    # P01 += B0*G1 (m4*n7)
    U[4, col] = 1; V[7, col] = 1; W[1, col] = 1; col += 1
    # P10 += B1*G0 (m5*n6)
    U[5, col] = 1; V[6, col] = 1; W[3, col] = 1; col += 1
    # P11 += B1*G1 (m5*n7)
    U[5, col] = 1; V[7, col] = 1; W[4, col] = 1; col += 1
    
    # 3. A * F (4 mults)
    # A = [[m0, m1], [m2, m3]], F = [n4, n5]
    # P02 += A00*F0 (m0*n4)
    U[0, col] = 1; V[4, col] = 1; W[2, col] = 1; col += 1
    # P02 += A01*F1 (m1*n5)
    U[1, col] = 1; V[5, col] = 1; W[2, col] = 1; col += 1
    # P12 += A10*F0 (m2*n4)
    U[2, col] = 1; V[4, col] = 1; W[5, col] = 1; col += 1
    # P12 += A11*F1 (m3*n5)
    U[3, col] = 1; V[5, col] = 1; W[5, col] = 1; col += 1
    
    # 4. C * E (4 mults)
    # C = [m6, m7], E = [[n0, n1], [n2, n3]]
    # P20 += C0*E00 (m6*n0)
    U[6, col] = 1; V[0, col] = 1; W[6, col] = 1; col += 1
    # P21 += C0*E01 (m6*n1)
    U[6, col] = 1; V[1, col] = 1; W[7, col] = 1; col += 1
    # P20 += C1*E10 (m7*n2)
    U[7, col] = 1; V[2, col] = 1; W[6, col] = 1; col += 1
    # P21 += C1*E11 (m7*n3)
    U[7, col] = 1; V[3, col] = 1; W[7, col] = 1; col += 1
    
    # 5. B * H (2 mults)
    # P02 += B0*H (m4*n8)
    U[4, col] = 1; V[8, col] = 1; W[2, col] = 1; col += 1
    # P12 += B1*H (m5*n8)
    U[5, col] = 1; V[8, col] = 1; W[5, col] = 1; col += 1
    
    # 6. D * G (2 mults)
    # P20 += D*G0 (m8*n6)
    U[8, col] = 1; V[6, col] = 1; W[6, col] = 1; col += 1
    # P21 += D*G1 (m8*n7)
    U[8, col] = 1; V[7, col] = 1; W[7, col] = 1; col += 1
    
    # 7. C * F (2 mults)
    # P22 += C0*F0 (m6*n4)
    U[6, col] = 1; V[4, col] = 1; W[8, col] = 1; col += 1
    # P22 += C1*F1 (m7*n5)
    U[7, col] = 1; V[5, col] = 1; W[8, col] = 1; col += 1
    
    # 8. D * H (1 mult)
    # P22 += D*H (m8*n8)
    U[8, col] = 1; V[8, col] = 1; W[8, col] = 1; col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
