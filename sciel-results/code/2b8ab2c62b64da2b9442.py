import time
import numpy as np

DIMENSIONS = [3, 4, 8]
DEADLINE = time.time() + 145.0

def construct():
    n, m, p = DIMENSIONS
    # We expect rank 64
    # 8 blocks of 2x2x2 (rank 7 each) = 56
    # 8 blocks of 1x2x2 (rank 1 each) = 8
    # Total 64
    
    rank = 64
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Strassen decomposition for 2x2x2
    # Indices in block: i in {0,1}, j in {0,1}, k in {0,1}
    # U indices: i*m + j -> i*2 + j (local)
    # V indices: j*p + k -> j*2 + k (local)
    # W indices: k*n + i -> k*1 + i (local)
    
    # Strassen factors
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # We need to express this as sum of rank 1 terms.
    # Each P is a product of a linear combo of A and a linear combo of B.
    # This corresponds to a rank 1 term in the tensor decomposition?
    # No, Strassen's algorithm is a bilinear algorithm.
    # The tensor decomposition rank is the number of multiplications.
    # Each multiplication corresponds to a rank 1 term in the tensor?
    # Yes, $T_{ijk} = \sum_r U_{ir} V_{jr} W_{kr}$.
    # If we define $U_{ir}$ and $V_{jr}$ such that $\sum_r U_{ir} V_{jr}$ computes the product of the linear combos?
    # Actually, the decomposition is:
    # $C_{ik} = \sum_j A_{ij} B_{jk} = \sum_r (\sum_i U_{ir} A_{ir}) (\sum_j V_{jr} B_{jr}) W_{kr}$? No.
    
    # The standard tensor decomposition for Strassen:
    # We define 7 vectors $u_r \in \mathbb{R}^4$ (for A), $v_r \in \mathbb{R}^4$ (for B), $w_r \in \mathbb{R}^4$ (for C).
    # $T_{ijk} = \sum_r u_{r, i, j} v_{r, j, k} w_{r, k, i}$?
    # Indices: $i$ (row A), $j$ (col A/row B), $k$ (col B).
    
    # Let's list the coefficients for Strassen.
    # A is 2x2. Indices (0,0), (0,1), (1,0), (1,1).
    # B is 2x2. Indices (0,0), (0,1), (1,0), (1,1).
    # C is 2x2. Indices (0,0), (0,1), (1,0), (1,1).
    
    # P1: A=(1,0,0,1), B=(1,0,0,1). C=(1,0,0,1)
    # P2: A=(0,0,1,1), B=(1,0,0,0). C=(0,0,1,0)
    # P3: A=(1,0,0,0), B=(0,1,0,-1). C=(0,1,0,0)
    # P4: A=(0,0,0,1), B=(0,0,1,-1). C=(0,0,1,0)
    # P5: A=(1,1,0,0), B=(0,0,0,1). C=(0,0,-1,0)
    # P6: A=(0,-1,1,0), B=(1,1,0,0). C=(0,0,0,1)
    # P7: A=(0,1,0,-1), B=(0,0,1,1). C=(1,0,0,1)
    
    # Check C11: P1(1) + P4(0) - P5(0) + P7(1) = 2? No.
    # C11 = P1 + P4 - P5 + P7
    # Coeff of C11 in P1: 1
    # Coeff of C11 in P4: 0
    # Coeff of C11 in P5: 0
    # Coeff of C11 in P7: 1
    # Sum: 2?
    # Wait, Strassen formula:
    # C11 = P1 + P4 - P5 + P7
    # P1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
    # P4 = A22(B21-B11) = A22B21 - A22B11
    # P5 = (A11+A12)B22 = A11B22 + A12B22
    # P7 = (A12-A22)(B21+B22) = A12B21 + A12B22 - A22B21 - A22B22
    
    # Sum for C11:
    # A11B11: 1 (from P1)
    # A11B22: 1 (P1) - 1 (P5) = 0
    # A22B11: 1 (P1) - 1 (P4) = 0
    # A22B22: 1 (P1) - 1 (P7) = 0
    # A22B21: -1 (P4) + 1 (P7) = 0
    # A12B21: 1 (P7)
    # A12B22: 1 (P7) - 1 (P5) = 0
    
    # Result: A11B11 + A12B21?
    # This is not C11. C11 = A11B11 + A12B21.
    # Yes! $C_{11} = A_{11}B_{11} + A_{12}B_{21}$.
    
    # So the coefficients are correct.
    
    # We will store these 7 patterns.
    # A_idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B_idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C_idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    strassen_A = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 0, 0, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [0, -1, 1, 0],# P6
        [0, 1, 0, -1] # P7
    ]
    strassen_B = [
        [1, 0, 0, 1], # P1
        [1, 0, 0, 0], # P2
        [0, 1, 0, -1],# P3
        [0, 0, 1, -1],# P4
        [0, 0, 0, 1], # P5
        [1, 1, 0, 0], # P6
        [0, 0, 1, 1]  # P7
    ]
    strassen_C = [
        [1, 0, 0, 0], # P1 -> C11
        [0, 0, 1, 0], # P2 -> C21
        [0, 1, 0, 0], # P3 -> C12
        [0, 0, 1, 0], # P4 -> C21
        [0, 0, -1, 0],# P5 -> -C21
        [0, 0, 0, 1], # P6 -> C22
        [1, 0, 0, 1]  # P7 -> C11 + C22
    ]
    # Wait, P7 contributes to C11 and C22.
    # C11 = P1 + P4 - P5 + P7. Coeff 1 for P7.
    # C22 = P1 - P2 + P3 + P6. Coeff 0 for P7?
    # Let's re-verify C22.
    # C22 = A11B12 + A12B22.
    # P1: A11B12 (from A11B11+A11B22+A22B11+A22B22? No, B12 is index 1. P1 uses B11, B22. So P1 has no B12).
    # P1 = (A11+A22)(B11+B22). No B12.
    # P2 = (A21+A22)B11. No B12.
    # P3 = A11(B12-B22). Has A11B12.
    # P6 = (A21-A11)(B11+B12). Has -A11B12.
    # Sum P3+P6 for A11B12: 1 - 1 = 0?
    # Something is wrong with my manual check or Strassen formula.
    
    # Standard Strassen:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # Let's check C22 = A11B12 + A12B22.
    # P1: No B12, No A12B22? P1 has A11B22, A22B22.
    # P2: No.
    # P3: A11B12 - A11B22.
    # P6: A21B11 + A21B12 - A11B11 - A11B12.
    
    # Sum for A11B12:
    # P3: 1
    # P6: -1
    # Total: 0.
    
    # Sum for A12B22:
    # P1: 0 (A12 not in P1)
    # P5: A11B22 + A12B22. Coeff 1.
    # P7: A12B21 + A12B22 - A22B21 - A22B22. Coeff 1.
    # C22 formula: P1 - P2 + P3 + P6.
    # P5 and P7 are NOT in C22 formula!
    
    # Ah, C22 = P1 - P2 + P3 + P6.
    # My previous check used P5, P7.
    
    # Let's re-evaluate A11B12 in C22:
    # P3: 1
    # P6: -1
    # Total 0.
    
    # This implies Strassen formula I recalled is wrong or I am misapplying it.
    # Correct Strassen:
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # Let's check C12 = A11B12 + A12B22.
    # P3: A11B12 - A11B22.
    # P5: A11B22 + A12B22.
    # Sum: A11B12 + A12B22. Correct.
    
    # Let's check C21 = A21B11 + A22B21.
    # P2: A21B11 + A22B11.
    # P4: A22B21 - A22B11.
    # Sum: A21B11 + A22B21. Correct.
    
    # Let's check C11 = A11B11 + A12B21.
    # P1: A11B11 + A11B22 + A22B11 + A22B22.
    # P4: A22B21 - A22B11.
    # P5: A11B22 + A12B22. (-P5)
    # P7: A12B21 + A12B22 - A22B21 - A22B22.
    
    # A11B11: 1 (P1)
    # A11B22: 1 (P1) - 1 (P5) = 0
    # A22B11: 1 (P1) - 1 (P4) = 0
    # A22B22: 1 (P1) - 1 (P7) = 0
    # A22B21: -1 (P4) + 1 (P7) = 0
    # A12B22: -1 (P5) + 1 (P7) = 0
    # A12B21: 1 (P7)
    
    # Result: A11B11 + A12B21. Correct.
    
    # Now C22 = A11B12 + A12B22.
    # P1: A11B11 + A11B22 + A22B11 + A22B22.
    # P2: A21B11 + A22B11. (-P2)
    # P3: A11B12 - A11B22.
    # P6: A21B11 + A21B12 - A11B11 - A11B12.
    
    # A11B11: 1 (P1) - 1 (P6) = 0
    # A11B22: 1 (P1) - 1 (P3) = 0
    # A22B11: 1 (P1) - 1 (P2) = 0
    # A22B22: 1 (P1)
    # A21B11: -1 (P2) + 1 (P6) = 0
    # A21B12: 1 (P6)
    # A11B12: 1 (P3) - 1 (P6) = 0
    
    # Result: A22B22 + A21B12.
    # But C22 should be A11B12 + A12B22.
    
    # There is a mismatch.
    # P1 has A22B22. C22 has A12B22.
    # It seems my Strassen coefficients for A and B might be swapped or indices are different.
    
    # Let's use a known correct set of coefficients.
    # Or just use the naive rank 8 for 2x2x2?
    # Rank 8 is safe.
    # If I use rank 8 for 2x2x2:
    # 8 blocks * 8 = 64.
    # 8 blocks * 1 = 8.
    # Total 72.
    
    # 72 is still better than 96.
    # But 64 is better.
    # I will implement rank 8 for simplicity and correctness first, then try to fix Strassen if time permits.
    # Actually, let's just use rank 8 for 2x2x2.
    
    # Rank 8 decomposition for 2x2x2 is just the 8 terms $A_{ij} B_{jk}$.
    
    # Let's switch to Rank 8 for 2x2x2 to ensure correctness.
    
    # Recalculate rank:
    # 8 blocks of 2x2x2 (rank 8 each) = 64
    # 8 blocks of 1x2x2 (rank 1 each) = 8
    # Total 72.
    
    # Is 72 better than 96? Yes.
    
    # Let's try to fix Strassen.
    # The issue is likely in my manual expansion.
    # I will use the standard Strassen matrices from a reliable source.
    
    # A vectors (4 components: 00, 01, 10, 11)
    # B vectors (4 components: 00, 01, 10, 11)
    # C vectors (4 components: 00, 01, 10, 11)
    
    # Correct Strassen coefficients:
    # P1: A=[1,0,0,1], B=[1,0,0,1], C=[1,0,0,0]
    # P2: A=[0,0,1,1], B=[1,0,0,0], C=[0,0,1,0]
    # P3: A=[1,0,0,0], B=[0,1,0,-1], C=[0,1,0,0]
    # P4: A=[0,0,0,1], B=[0,0,1,-1], C=[0,0,1,0]
    # P5: A=[1,1,0,0], B=[0,0,0,1], C=[0,0,-1,0]
    # P6: A=[0,-1,1,0], B=[1,1,0,0], C=[0,0,0,1]
    # P7: A=[0,1,0,-1], B=[0,0,1,1], C=[1,0,0,1]
    
    # This is the same set I used.
    # Why did C22 fail?
    # C22 = P1 - P2 + P3 + P6.
    # My expansion gave A22B22 + A21B12.
    # Expected A11B12 + A12B22.
    
    # Let's check P1 again.
    # P1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
    # This is correct.
    
    # Let's check P6.
    # P6 = (A21-A11)(B11+B12) = A21B11 + A21B12 - A11B11 - A11B12.
    # This is correct.
    
    # Let's check P3.
    # P3 = A11(B12-B22) = A11B12 - A11B22.
    # This is correct.
    
    # Let's check P2.
    # P2 = (A21+A22)B11 = A21B11 + A22B11.
    # This is correct.
    
    # Sum for C22:
    # A11B11: 1(P1) - 1(P6) = 0.
    # A11B22: 1(P1) - 1(P3) = 0.
    # A22B11: 1(P1) - 1(P2) = 0.
    # A22B22: 1(P1).
    # A21B11: -1(P2) + 1(P6) = 0.
    # A21B12: 1(P6).
    # A11B12: 1(P3) - 1(P6) = 0.
    
    # Result: A22B22 + A21B12.
    
    # It seems this set of coefficients does NOT produce C22 = A11B12 + A12B22.
    # It produces A22B22 + A21B12.
    
    # This implies the C vector for P1, P2, P3, P6 is wrong for C22.
    # Or the A/B vectors are wrong.
    
    # Actually, Strassen's algorithm is symmetric.
    # If I swap A and B, and swap indices appropriately, it should work.
    
    # I will stick to Rank 8 for 2x2x2 to be safe.
    
    # Total Rank 72.
    
    # Implementation of Rank 8 for 2x2x2:
    # For each (i,j,k) in 2x2x2, add 1 to U, V, W.
    
    # Implementation of Rank 1 for 1x2x2:
    # U[0,0]=1, U[0,1]=1
    # V[0,0]=1, V[1,0]=1, V[0,1]=1, V[1,1]=1
    # W[0]=1, W[1]=1
    
    # Let's code this.

    # Block 2x2x2
    def add_2x2x2(i_off, j_off, k_off):
        nonlocal col
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    U[(i_off+di)*m + (j_off+dj)][col] = 1
                    V[(j_off+dj)*p + (k_off+dk)][col] = 1
                    W[(k_off+dk)*n + (i_off+di)][col] = 1
                    col += 1

    # Block 1x2x2
    def add_1x2x2(i_off, j_off, k_off):
        nonlocal col
        U[i_off*m + j_off][col] = 1
        U[i_off*m + j_off + 1][col] = 1
        
        V[(j_off)*p + k_off][col] = 1
        V[(j_off+1)*p + k_off][col] = 1
        V[(j_off)*p + k_off + 1][col] = 1
        V[(j_off+1)*p + k_off + 1][col] = 1
        
        W[k_off*n + i_off][col] = 1
        W[(k_off+1)*n + i_off][col] = 1
        
        col += 1

    # Loop over p_offset (0, 4)
    for p_offset in [0, 4]:
        # 4 blocks of 2x2x2
        add_2x2x2(0, 0, p_offset)
        add_2x2x2(0, 2, p_offset)
        add_2x2x2(0, 0, p_offset+2)
        add_2x2x2(0, 2, p_offset+2)
        
        # 4 blocks of 1x2x2
        add_1x2x2(2, 0, p_offset)
        add_1x2x2(2, 2, p_offset)
        add_1x2x2(2, 0, p_offset+2)
        add_1x2x2(2, 2, p_offset+2)
        
    return {"U": U, "V": V, "W": W}

# Verification
U, V, W = construct()["U"], construct()["V"], construct()["W"]
n, m, p = 3, 4, 8
T = np.zeros((n, m, p))
for i in range(n):
    for j in range(m):
        for k in range(p):
            T[i,j,k] = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
print("Max error:", np.max(np.abs(T - 1)))
print("Rank:", U.shape[1])
