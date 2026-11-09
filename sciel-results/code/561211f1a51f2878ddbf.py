"""Matrix multiplication 6x6x6: Kronecker product of Strassen(2x2) and schoolbook(3x3)."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def _strassen_2x2():
    """Return (U, V, W) for 2x2 matrix multiplication using Strassen's algorithm.
    
    U shape: (4, 7)  -- (n*m, rank)
    V shape: (4, 7)  -- (m*p, rank)
    W shape: (4, 7)  -- (p*n, rank), with trace convention W[k*n+i, r]
    
    Strassen's 7 products:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B01)
    M7 = (A01-A11)*(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Index mapping: U[i*m + j, c] where m=2
    # A indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # B indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # C indices (trace convention W[k*n+i]): C[i,j] -> W[j*n+i]
    # C(0,0)->W[0], C(0,1)->W[1], C(1,0)->W[2], C(1,1)->W[3]
    
    # M1 = (A00+A11)*(B00+B11)
    # A00=U[0], A11=U[3]; B00=V[0], B11=V[3]
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C00 += M1, C11 += M1
    
    # M2 = (A10+A11)*B00
    # A10=U[2], A11=U[3]; B00=V[0]
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1  # C10 += M2, C11 -= M2
    
    # M3 = A00*(B01-B11)
    # A00=U[0]; B01=V[1], B11=V[3]
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1  # C01 += M3, C11 += M3
    
    # M4 = A11*(B10-B00)
    # A11=U[3]; B10=V[2], B00=V[0]
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1  # C00 += M4, C10 += M4
    
    # M5 = (A00+A01)*B11
    # A00=U[0], A01=U[1]; B11=V[3]
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1  # C00 -= M5, C01 += M5
    
    # M6 = (A10-A00)*(B00+B01)
    # A10=U[2], A00=U[0]; B00=V[0], B01=V[1]
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C11 += M6
    
    # M7 = (A01-A11)*(B10+B11)
    # A01=U[1], A11=U[3]; B10=V[2], B11=V[3]
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C00 += M7
    
    return U, V, W

def _schoolbook_3x3():
    """Return (U, V, W) for 3x3 matrix multiplication using schoolbook.
    
    U shape: (9, 27)
    V shape: (9, 27)
    W shape: (9, 27)
    """
    n = 3
    rank = 27
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    
    col = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                U[i * n + j, col] = 1
                V[j * n + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W

def construct():
    deadline = time.time() + 120
    
    # Get the two component decompositions
    U2, V2, W2 = _strassen_2x2()
    U3, V3, W3 = _schoolbook_3x3()
    
    # Kronecker product: rank = 7 * 27 = 189
    U = np.kron(U2, U3)  # shape (4*9, 7*27) = (36, 189)
    V = np.kron(V2, V3)  # shape (4*9, 7*27) = (36, 189)
    W = np.kron(W2, W3)  # shape (4*9, 7*27) = (36, 189)
    
    # The Kronecker product gives us the decomposition for the tensor product
    # of two matrix multiplication tensors. For 6x6x6 = (2x2x2) ⊗ (3x3x3),
    # the Kronecker product of the factor matrices gives the correct decomposition.
    
    # Convert to lists for the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    if time.time() > deadline - 5:
        # Safety margin: return what we have
        pass
    
    return {"U": U_list, "V": V_list, "W": W_list}
