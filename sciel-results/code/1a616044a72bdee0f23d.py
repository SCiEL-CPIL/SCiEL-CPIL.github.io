import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def _strassen_2x2():
    """Return the 7 rank-1 terms for 2x2x2 Strassen."""
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-g)
    # M4 = d(g-h)
    # M5 = (a+b)(e+f)
    # M6 = (c-d)(g+h)
    # M7 = (a-c)(d+e)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # 
    # U (2x2, for A), V (2x2, for B), W (2x2, for C)
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # U[i,k] is coefficient of A[i,k], V[k,j] is coefficient of B[k,j], W[i,j] is coefficient for C[i,j]
    
    terms = []
    # M1 = (a+d)(e+h)
    terms.append((np.array([[1,0],[1,0]]), np.array([[1,1],[0,1]]), np.array([[1,0],[0,1]])))  # Wait, this isn't right
    # Actually, M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # U for M1: [[1,0],[1,0]]? No, U[i,k] should give the linear form.
    # L1(A) = A[0,0] + A[1,1], so U1 = [[1,0],[0,1]]
    # R1(B) = B[0,0] + B[1,1], so V1 = [[1,0],[0,1]]
    # C11 += 1, C22 += 1, so W1 = [[1,0],[0,1]]
    terms.append((np.array([[1,0],[0,1]]), np.array([[1,0],[0,1]]), np.array([[1,0],[0,1]])))
    
    # M2 = (c+d)*e = (A[1,0]+A[1,1])*B[0,0]
    # U2 = [[0,0],[1,1]], V2 = [[1,0],[0,0]], W2: C21 += 1, so [[0,0],[1,0]]
    terms.append((np.array([[0,0],[1,1]]), np.array([[1,0],[0,0]]), np.array([[0,0],[1,0]])))
    
    # M3 = a*(f-g) = A[0,0]*(B[0,1]-B[1,1])
    # U3 = [[1,0],[0,0]], V3 = [[0,1],[0,-1]], W3: C12 += 1, so [[0,1],[0,0]]
    terms.append((np.array([[1,0],[0,0]]), np.array([[0,1],[0,-1]]), np.array([[0,1],[0,0]])))
    
    # M4 = d*(g-h) = A[1,1]*(B[1,0]-B[1,1])
    # U4 = [[0,0],[0,1]], V4 = [[0,0],[1,-1]], W4: C11 += 1, C22 += 1, so [[1,0],[0,1]]
    terms.append((np.array([[0,0],[0,1]]), np.array([[0,0],[1,-1]]), np.array([[1,0],[0,1]])))
    
    # M5 = (a+b)*(e+f) = (A[0,0]+A[0,1])*(B[0,0]+B[0,1])
    # U5 = [[1,1],[0,0]], V5 = [[1,1],[0,0]], W5: C11 -= 1, C12 += 1, so [[-1,1],[0,0]]
    terms.append((np.array([[1,1],[0,0]]), np.array([[1,1],[0,0]]), np.array([[-1,1],[0,0]])))
    
    # M6 = (c-d)*(g+h) = (A[1,0]-A[1,1])*(B[1,0]+B[1,1])
    # U6 = [[0,0],[1,-1]], V6 = [[0,0],[1,1]], W6: C22 += 1, so [[0,0],[0,1]]
    terms.append((np.array([[0,0],[1,-1]]), np.array([[0,0],[1,1]]), np.array([[0,0],[0,1]])))
    
    # M7 = (a-c)*(d+e) = (A[0,0]-A[1,0])*(A[1,1]+B[0,0])
    # Wait, this mixes A and B! That's not allowed in a bilinear algorithm.
    
    # I think I have the Strassen formulas wrong. Let me look them up.
    
    # Standard Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-g)
    # M4 = d(g-h)
    # M5 = (a+b)(e+f)
    # M6 = (c-d)(g+h)
    # M7 = (a-c)(d+e)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # M7 = (a-c)(d+e) = (A[0,0]-A[1,0])*(A[1,1]+B[0,0])
    # This is NOT bilinear! It involves A[1,1] in the second factor.
    
    # I think I'm misremembering Strassen. Let me reconsider.
    
    # Actually, Strassen's 7 products are:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-g)
    # M4 = d(g-h)
    # M5 = (a+b)(e+f)
    # M6 = (c-d)(g+h)
    # M7 = (a-c)(d+e)
    
    # And:
    # p = M1 - M5 + M7
    # q = M5 + M4
    # r = M2 + M4
    # s = M1 - M3 + M6
    
    # Let me verify: 
    # C11 = a*e + b*g
    # M1 - M5 + M7 = (a+d)(e+h) - (a+b)(e+f) + (a-c)(d+e)
    # = ae + ah + de + dh - ae - af - be - bf + ad + ae - cd - ce
    # = ah + de + dh - af - be - bf + ad + ae - cd - ce
    # This doesn't simplify to ae + bg. I'm making errors.
    
    # Let me just use the standard Strassen decomposition from a reliable source.
    
    pass
