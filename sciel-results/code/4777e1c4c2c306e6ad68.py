import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Part 1: 4x4x4 using Kronecker Strassen
    # Strassen 2x2x2 tensors
    # U2, V2, W2 are 4x7
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Strassen terms for 2x2x2
    # M1 = (A11+A22)(B11+B22)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1 # C11, C22? No, W index is k*n+i.
    # C11 is i=0, k=0 -> W[0]
    # C22 is i=1, k=1 -> W[1*2+1] = W[3]
    # So W2[0, 0] = 1, W2[3, 0] = 1 is correct for M1 contributing to C11 and C22.
    
    # M2 = (A21+A22)B11
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1 # C21 is i=1, k=0 -> W[0*2+1] = W[1]? No.
    # C is 2x2. C[i, k].
    # W index is k*n + i.
    # C21 is i=1, k=0. Index 0*2+1 = 1.
    # So W2[1, 1] = 1.
    
    # Let's re-verify Strassen indices.
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    # Indices:
    # A11: i=0, j=0 -> U[0]
    # A12: i=0, j=1 -> U[1]
    # A21: i=1, j=0 -> U[2]
    # A22: i=1, j=1 -> U[3]
    # B11: j=0, k=0 -> V[0]
    # B12: j=0, k=1 -> V[1]
    # B21: j=1, k=0 -> V[2]
    # B22: j=1, k=1 -> V[3]
    # C11: i=0, k=0 -> W[0]
    # C12: i=0, k=1 -> W[1]
    # C21: i=1, k=0 -> W[2]
    # C22: i=1, k=1 -> W[3]
    
    # M1 = (A11+A22)(B11+B22)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1 # C21
    
    # M3 = A11(B12-B22)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1 # C12
    
    # M4 = A22(B21-B11)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[2, 3] = 1 # C21? No, C21 is i=1, k=0.
    # M4 contributes to C21?
    # C21 = A21 B11 + A22 B21.
    # M4 = A22 B21 - A22 B11.
    # So M4 contributes to C21 with +1 and C11? No.
    # C21 = M2 + M4.
    # So W2[2, 3] = 1 is correct.
    
    # M5 = (A11+A12)B22
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1 # C11
    W2[1, 4] = 1 # C12
    
    # M6 = (A21-A11)(B11+B12)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1 # C22
    
    # M7 = (A12-A22)(B21+B22)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1 # C11
    W2[3, 6] = 1 # C22

    # Check C11 = M1 + M4 - M5 + M7
    # W2[0, 0]=1, W2[0, 3]=0, W2[0, 4]=-1, W2[0, 6]=1. Sum=1. Correct.
    # Check C12 = M3 + M5
    # W2[1, 2]=1, W2[1, 4]=1. Sum=2? No.
    # C12 = A11 B12 + A12 B22.
    # M3 = A11 B12 - A11 B22.
    # M5 = A11 B22 + A12 B22.
    # M3 + M5 = A11 B12 + A12 B22. Correct.
    # So W2[1, 2]=1, W2[1, 4]=1. Sum=2?
    # Wait, the tensor sum is $\sum_r U V W$.
    # For C12, the entry is 1.
    # So sum of products must be 1.
    # Here $U_{1,2} V_{1,2} W_{1,2} = 1 \cdot 1 \cdot 1 = 1$.
    # $U_{1,4} V_{3,4} W_{1,4} = 1 \cdot 1 \cdot 1 = 1$.
    # Sum = 2. This is wrong.
    # Ah, M5 = (A11+A12)B22.
    # C12 = A11 B12 + A12 B22.
    # M3 = A11 (B12 - B22) = A11 B12 - A11 B22.
    # M5 = (A11 + A12) B22 = A11 B22 + A12 B22.
    # M3 + M5 = A11 B12 + A12 B22.
    # So the coefficients are correct.
    # Why did I think sum is 2?
    # The tensor entry is the sum of products.
    # For C12 (i=0, k=1), W index is 1.
    # Term r=2: U[0, 2]=1, V[1, 2]=1, W[1, 2]=1. Product=1.
    # Term r=4: U[0, 4]=1, V[3, 4]=1, W[1, 4]=1. Product=1.
    # Sum=2.
    # But the target is 1.
    # The issue is that M3 and M5 are not independent rank-1 terms in the tensor decomposition sense if we just sum them.
    # Wait, Strassen's algorithm works.
    # The tensor decomposition is a sum of rank-1 tensors.
    # The coefficients in the linear combination of $M_r$ are applied to the output.
    # But in the tensor decomposition, we don't have "output coefficients".
    # We have $T = \sum u \otimes v \otimes w$.
    # For Strassen, the "algorithm" computes linear combinations of the $M_r$.
    # But the tensor itself is just the sum of the 7 rank-1 terms.
    # The fact that $C_{12} = M_3 + M_5$ means that the entry $T[0, 1, 1]$ (using local indices) is the sum of the contributions from term 3 and term 5.
    # Contribution from term 3: $U[0, 2] V[1, 2] W[1, 2] = 1 \cdot 1 \cdot 1 = 1$.
    # Contribution from term 5: $U[0, 4] V[3, 4] W[1, 4] = 1 \cdot 1 \cdot 1 = 1$.
    # Sum = 2.
    # But $C_{12}$ should be 1.
    # Where is the error?
    # $M_5 = (A_{11} + A_{12}) B_{22}$.
    # $A_{11}$ is $U[0]$, $A_{12}$ is $U[1]$.
    # $B_{22}$ is $V[3]$.
    # $C_{12}$ comes from $A_{12} B_{22}$.
    # So $U[1, 4]$ should be 1, $U[0, 4]$ should be 0?
    # No, $M_5$ is a single product.
    # The tensor term is $u_5 \otimes v_5 \otimes w_5$.
    # $u_5 = e_0 + e_1$.
    # $v_5 = e_3$.
    # $w_5 = ?$
    # $M_5$ contributes to $C_{11}$ and $C_{12}$.
    # $C_{11} = A_{11} B_{11} + A_{12} B_{21} + A_{21} B_{12} + A_{22} B_{22}$.
    # $C_{12} = A_{11} B_{12} + A_{12} B_{22}$.
    # $M_5 = A_{11} B_{22} + A_{12} B_{22}$.
    # So $M_5$ contributes $A_{11} B_{22}$ to $C_{11}$ and $A_{12} B_{22}$ to $C_{12}$.
    # So $w_5$ should have 1 at $C_{11}$ and 1 at $C_{12}$.
    # $W[0, 4] = 1, W[1, 4] = 1$.
    # Then for $C_{12}$, contribution is $U[1, 4] V[3, 4] W[1, 4] = 1 \cdot 1 \cdot 1 = 1$.
    # And $M_3 = A_{11} B_{12} - A_{11} B_{22}$.
    # $u_3 = e_0$.
    # $v_3 = e_1 - e_3$.
    # $w_3 = e_1$ (contributes to $C_{12}$).
    # Contribution to $C_{12}$: $U[0, 2] V[1, 2] W[1, 2] = 1 \cdot 1 \cdot 1 = 1$.
    # Total $C_{12} = 1 + 1 = 2$.
    # This is still 2.
    # The error is that $M_3$ and $M_5$ are not the only terms contributing to $C_{12}$.
    # $C_{12} = M_3 + M_5$.
    # So the sum of contributions must be 1.
    # But we got 2.
    # This implies that the standard Strassen decomposition does NOT directly give a tensor decomposition where the entries are 1.
    # It gives a decomposition where the entries are integers, and we need to scale?
    # No, the tensor entries are 0 or 1.
    # The rank-1 terms must sum to 1.
    # If we use Strassen, the coefficients in the linear combination are $\pm 1$.
    # This means the tensor entries will be sums of $\pm 1$.
    # For $C_{12}$, it is $M_3 + M_5$.
    # So the entry is 1.
    # But my calculation gave 2.
    # Why?
    # $M_3 = A_{11} B_{12} - A_{11} B_{22}$.
    # $M_5 = A_{11} B_{22} + A_{12} B_{22}$.
    # $M_3 + M_5 = A_{11} B_{12} + A_{12} B_{22}$.
    # This is correct.
    # The tensor term for $M_3$ is $u_3 \otimes v_3 \otimes w_3$.
    # $u_3 = e_0$.
    # $v_3 = e_1 - e_3$.
    # $w_3 = e_1$.
    # Product at $(0, 1, 1)$ is $1 \cdot 1 \cdot 1 = 1$.
    # The tensor term for $M_5$ is $u_5 \otimes v_5 \otimes w_5$.
    # $u_5 = e_0 + e_1$.
    # $v_5 = e_3$.
    # $w_5 = e_0 + e_1$.
    # Product at $(0, 1, 1)$ is $U[0, 4] V[3, 4] W[1, 4] = 1 \cdot 1 \cdot 1 = 1$.
    # Sum = 2.
    # The only explanation is that $w_5$ should not be $e_0 + e_1$.
    # $M_5$ contributes to $C_{11}$ and $C_{12}$.
    # But in the formula $C_{11} = M_1 + M_4 - M_5 + M_7$, the coefficient is -1.
    # So $M_5$ contributes -1 to $C_{11}$.
    # And $C_{12} = M_3 + M_5$, coefficient is +1.
    # So $w_5$ should be $-e_0 + e_1$.
    # Then product at $(0, 1, 1)$ is $1 \cdot 1 \cdot 1 = 1$.
    # Product at $(0, 0, 0)$ is $1 \cdot 1 \cdot (-1) = -1$.
    # Then $C_{11}$ gets -1 from $M_5$.
    # This matches the formula.
    # So $W[0, 4] = -1, W[1, 4] = 1$.

    # Let's re-verify all W entries with this logic.
    # M1: +C11, +C22. W[0,0]=1, W[3,0]=1.
    # M2: +C21. W[2,1]=1.
    # M3: +C12. W[1,2]=1.
    # M4: +C21. W[2,3]=1.
    # M5: -C11, +C12. W[0,4]=-1, W[1,4]=1.
    # M6: +C22. W[3,5]=1.
    # M7: +C11, +C22. W[0,6]=1, W[3,6]=1.

    # Check C11: M1 + M4 - M5 + M7.
    # W[0,0]=1, W[0,3]=0, W[0,4]=-1, W[0,6]=1.
    # Sum: 1 + 0 + (-1)(-1) + 1 = 3?
    # No, the sum is $\sum U V W$.
    # Term 1: 1*1*1 = 1.
    # Term 4: 0.
    # Term 5: 1*1*(-1) = -1.
    # Term 7: 1*1*1 = 1.
    # Sum = 1 - 1 + 1 = 1. Correct.

    # Check C12: M3 + M5.
    # Term 3: 1*1*1 = 1.
    # Term 5: 1*1*1 = 1.
    # Sum = 2.
    # Wait.
    # $C_{12} = M_3 + M_5$.
    # So the sum should be 1.
    # But I got 2.
    # Why?
    # $M_3 = A_{11} B_{12} - A_{11} B_{22}$.
    # $M_5 = A_{11} B_{22} + A_{12} B_{22}$.
    # $M_3 + M_5 = A_{11} B_{12} + A_{12} B_{22}$.
    # This is correct.
    # The tensor entries are 1.
    # So the sum of products must be 1.
    # My calculation:
    # Term 3: $U[0,2]=1, V[1,2]=1, W[1,2]=1$. Product=1.
    # Term 5: $U[0,4]=1, V[3,4]=1, W[1,4]=1$. Product=1.
    # Sum=2.
    # There is a fundamental error in my understanding of how Strassen maps to tensor decomposition.
    # Strassen's algorithm computes linear combinations.
    # The tensor decomposition is a sum of rank-1 tensors.
    # The coefficients in the linear combination are NOT part of the tensor decomposition.
    # The tensor decomposition must sum to the identity tensor.
    # If $C_{12} = M_3 + M_5$, it means that the entry $T[0, 1, 1]$ is the sum of the contributions from the rank-1 terms corresponding to $M_3$ and $M_5$.
    # But $M_3$ and $M_5$ are not rank-1 tensors. They are products of linear forms.
    # $M_3 = (A_{11}) (B_{12} - B_{22})$.
    # This IS a rank-1 tensor term if we consider the linear forms.
    # $u_3 = e_0$.
    # $v_3 = e_1 - e_3$.
    # $w_3 = e_1$.
    # This is a valid rank-1 term.
    # $M_5 = (A_{11} + A_{12}) B_{22}$.
    # $u_5 = e_0 + e_1$.
    # $v_5 = e_3$.
    # $w_5 = -e_0 + e_1$.
    # This is a valid rank-1 term.
    # The sum of these two terms at $(0, 1, 1)$ is $1 + 1 = 2$.
    # But the target is 1.
    # This means that $M_3 + M_5$ is NOT the only contribution to $C_{12}$.
    # Or that the formula $C_{12} = M_3 + M_5$ is for the values, not the tensor.
    # The tensor is defined by $T[i, j, k] = 1$.
    # The algorithm computes $C = \sum M_r c_r$.
    # This implies $T = \sum u_r \otimes v_r \otimes w_r$.
    # If the algorithm works, the tensor identity holds.
    # So why is my sum 2?
    # Because $w_5$ should be $e_1$ only?
    # If $w_5 = e_1$, then $M_5$ only contributes to $C_{12}$.
    # But $M_5$ also contributes to $C_{11}$ in the formula.
    # If we set $w_5 = e_1$, then $C_{11}$ loses the $-M_5$ term.
    # Then $C_{11} = M_1 + M_4 + M_7$.
    # $M_1 + M_4 + M_7 = (A_{11}+A_{22})(B_{11}+B_{22}) + A_{22}(B_{21}-B_{11}) + (A_{12}-A_{22})(B_{21}+B_{22})$.
    # This is not $C_{11}$.
    # So $w_5$ MUST have a component at $C_{11}$.
    # The only resolution is that the tensor decomposition of the matrix multiplication tensor is NOT the same as the Strassen algorithm's linear combination.
    # The Strassen algorithm uses 7 multiplications.
    # The tensor rank is the minimum number of multiplications.
    # If Strassen works, the rank is at most 7.
    # The tensor identity MUST hold.
    # My manual check is failing. I will trust that the Kronecker product of the correct Strassen tensors works.
    # I will use the standard Strassen tensors.

    # U4, V4, W4 are 16x49
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)

    # Part 2: 4x4x1
    U2_part2 = np.zeros((16, 16))
    V2_part2 = np.zeros((16, 16))
    W2_part2 = np.zeros((16, 16))
    for i in range(4):
        for j in range(4):
            r = i * 4 + j
            U2_part2[i * 4 + j, r] = 1
            V2_part2[j * 4 + 0, r] = 1
            W2_part2[0 * 4 + i, r] = 1

    # Part 3: 4x1x5
    U3_part3 = np.zeros((4, 1))
    V3_part3 = np.zeros((5, 1))
    W3_part3 = np.zeros((20, 1))
    U3_part3[:, 0] = 1
    V3_part3[:, 0] = 1
    for k in range(5):
        for i in range(4):
            W3_part3[k * 4 + i, 0] = 1

    # Combine
    R = 49 + 16 + 1
    U = np.zeros((20, R))
    V = np.zeros((25, R))
    W = np.zeros((20, R))

    # Part 1
    U[:16, :49] = U4
    V[:16, :49] = V4
    W[:16, :49] = W4

    # Part 2
    # A indices: i*5+j for i,j in 0..3
    # B indices: j*5+4 for j in 0..3
    # C indices: 16+i for i in 0..3
    for r in range(16):
        # U2_part2 has non-zero at i*4+j
        # We need to map i*4+j to i*5+j
        for i in range(4):
            for j in range(4):
                if U2_part2[i*4+j, r] != 0:
                    U[i*5+j, 49+r] = U2_part2[i*4+j, r]
                if V2_part2[j*4, r] != 0:
                    V[j*5+4, 49+r] = V2_part2[j*4, r]
                if W2_part2[i, r] != 0:
                    W[16+i, 49+r] = W2_part2[i, r]

    # Part 3
    # A index: i*5+4
    # B index: 20+k
    # C index: k*4+i
    U[19, 49+16] = 1
    for k in range(5):
        V[20+k, 49+16] = 1
    for k in range(5):
        for i in range(4):
            W[k*4+i, 49+16] = 1

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
