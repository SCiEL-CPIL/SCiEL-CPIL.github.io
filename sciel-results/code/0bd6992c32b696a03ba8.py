import numpy as np

def strassen_2x2(A, B):
    """Compute 2x2 matrix product using Strassen's algorithm."""
    A11, A12 = A[0][0], A[0][1]
    A21, A22 = A[1][0], A[1][1]
    B11, B12 = B[0][0], B[0][1]
    B21, B22 = B[1][0], B[1][1]
    
    M1 = (A11 + A22) * (B11 + B22)
    M2 = (A21 + A22) * B11
    M3 = A11 * (B21 - B22)
    M4 = A22 * (B12 - B11)
    M5 = (A11 + A12) * B22
    M6 = (A21 - A11) * (B11 + B12)
    M7 = (A12 - A22) * (B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    return [[C11, C12], [C21, C22]]

# Test
A = [[1,2],[3,4]]
B = [[5,6],[7,8]]
C = strassen_2x2(A, B)
print(C)  # Should be [[19,22],[43,50]]
print(np.array(A) @ np.array(B))
