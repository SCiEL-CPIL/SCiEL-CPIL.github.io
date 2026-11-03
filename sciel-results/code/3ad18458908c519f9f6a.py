import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = 3, 3, 5
    
    # Target tensor T
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0

    def validate(U, V, W):
        try:
            R = np.einsum('ic,jc,kc->ijk', U, V, W)
            if np.allclose(R, T, atol=1e-9):
                return True
        except Exception:
            pass
        return False

    # Try Rank 41: 23 (3x3x3) + 18 (3x3x2)
    # Part 2: 3x3x2 schoolbook (k=3,4) - 18 terms
    U_332 = np.zeros((9, 18))
    V_332 = np.zeros((15, 18))
    W_332 = np.zeros((15, 18))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in [3, 4]:
                U_332[i*3+j, col] = 1
                V_332[j*5+k, col] = 1
                W_332[k*3+i, col] = 1
                col += 1
                
    # Part 1: 3x3x3 Rank 23
    # Using a known valid 23-term algorithm
    # Coefficients for U, V, W of shape (9, 23)
    U_333 = np.zeros((9, 23))
    V_333 = np.zeros((9, 23)) # Note: V for 3x3x3 is 9x23, but in full problem it's 15x23 (zeros for k=3,4)
    W_333 = np.zeros((9, 23)) # W for 3x3x3 is 9x23, in full problem 15x23 (zeros for k=3,4)
    
    # I will use a generic 23-term algorithm structure.
    # If I cannot guarantee correctness, I will use schoolbook for this part (27 terms)
    # which gives total 27+18=45.
    # To improve, I must get the 23 terms right.
    
    # Let's try to use the 23-term algorithm from "The rank of the 3x3x3 matrix multiplication tensor"
    # I will hardcode the 23 terms.
    
    # Term 0-6: Strassen on 2x2 block (indices 0,1)
    # U, V, W for 2x2x2 Strassen
    # Indices in 3x3: 0,1 are the 2x2 block.
    
    # M1 = (A00+A11)(B00+B11)
    U_333[0,0]=1; V_333[0,0]=1; W_333[0,0]=1
    U_333[1,0]=1; V_333[1,0]=1; W_333[1,0]=1
    # M2 = (A10+A11)B00
    U_333[2,1]=1; V_333[0,1]=1; W_333[2,1]=1
    U_333[3,1]=1; V_333[0,1]=1; W_333[3,1]=1
    # M3 = A00(B10-B11)
    U_333[0,2]=1; V_333[3,2]=1; W_333[0,2]=1
    U_333[0,2]=1; V_333[4,2]=-1; W_333[1,2]=-1
    # M4 = A11(B01-B00)
    U_333[1,3]=1; V_333[1,3]=1; W_333[1,3]=1
    U_333[1,3]=1; V_333[0,3]=-1; W_333[0,3]=-1
    # M5 = (A00+A01)B11
    U_333[0,4]=1; V_333[4,4]=1; W_333[1,4]=1
    U_333[1,4]=1; V_333[4,4]=1; W_333[2,4]=1
    # M6 = (A10-A00)(B00+B10)
    U_333[2,5]=1; V_333[0,5]=1; W_333[2,5]=1
    U_333[2,5]=1; V_333[3,5]=1; W_333[3,5]=1
    U_333[0,5]=-1; V_333[0,5]=-1; W_333[0,5]=-1
    U_333[0,5]=-1; V_333[3,5]=-1; W_333[1,5]=-1
    # M7 = (A01-A11)(B01+B11)
    U_333[1,6]=1; V_333[1,6]=1; W_333[2,6]=1
    U_333[1,6]=1; V_333[4,6]=1; W_333[3,6]=1
    U_333[3,6]=-1; V_333[1,6]=-1; W_333[1,6]=-1
    U_333[3,6]=-1; V_333[4,6]=-1; W_333[2,6]=-1
    
    # Terms 7-22: 16 more terms for the 3rd row/col interactions.
    # This is where I am unsure. I will use schoolbook for the remaining 16 elements of the 3x3x3 tensor?
    # No, that would be 7+16=23.
    # The 16 elements are:
    # A[i,2]*B[2,k] -> C[i,k] for i,k in 0,1,2 (9 terms)
    # A[i,0:2]*B[0:2,2] -> C[i,2] for i in 0,1,2 (3 terms)
    # A[2,0:2]*B[0:2,k] -> C[2,k] for k in 0,1,2 (3 terms)
    # Wait, the tensor is defined by T[i,j,k]=1.
    # The 2x2x2 block covers i,j,k in 0,1. (8 terms)
    # The remaining 19 terms are those where at least one index is 2.
    # If I use 16 terms for these 19 elements, it's not a simple schoolbook.
    
    # Given the risk, I will return schoolbook (45) if I cannot complete the 23-term algorithm.
    # However, I can try to use a **Rank 24** algorithm if I can find one?
    # Or just return schoolbook.
    
    # Let's try to return schoolbook for the 3x3x3 part (27 terms) + 18 for 3x3x2 = 45.
    # This is no improvement.
    
    # I will implement the schoolbook for the whole thing to be safe.
    U_sb = np.zeros((9, 45))
    V_sb = np.zeros((15, 45))
    W_sb = np.zeros((15, 45))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(5):
                U_sb[i*3+j, col] = 1
                V_sb[j*5+k, col] = 1
                W_sb[k*3+i, col] = 1
                col += 1
                
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
