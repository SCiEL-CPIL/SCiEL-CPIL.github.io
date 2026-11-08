import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # We use a Strassen-based decomposition for M_{4,4,8} and restrict to M_{4,4,7}.
    # Rank of M_{4,4,8} using two M_{4,4,4} Strassen decompositions is 98.
    # We generate the factors for M_{4,4,4} Strassen and duplicate them for M_{4,4,8}.
    
    # Helper to generate 2x2 Strassen factors
    def get_2x2_strassen_factors():
        # Returns U, V, W for 2x2x2 Strassen
        # U: 2x2, V: 2x2, W: 2x2
        # Indices: U[i,j], V[j,k], W[k,i]
        U = []
        V = []
        W = []
        
        # M1 = (a+d)(e+h)
        U.append([1, 0, 0, 1]) # a=11, d=22
        V.append([1, 0, 0, 1]) # e=11, h=22
        W.append([1, 1, 1, 1]) # k,i: (1,1), (2,1), (1,2), (2,2)
        
        # M2 = (c+d)e
        U.append([0, 0, 1, 1]) # c=21, d=22
        V.append([1, 0, 0, 0]) # e=11
        W.append([1, 0, 0, 0]) # k=1, i=2 -> (1,2)
        
        # M3 = a(f-h)
        U.append([1, 0, 0, 0]) # a=11
        V.append([0, 1, 0, -1]) # f=12, h=22
        W.append([0, 1, 0, 0]) # k=2, i=1 -> (2,1)
        
        # M4 = d(g-e)
        U.append([0, 0, 0, 1]) # d=22
        V.append([0, 0, 1, -1]) # g=21, e=11
        W.append([0, 0, 1, 0]) # k=1, i=2 -> (1,2)
        
        # M5 = (a+b)h
        U.append([1, 1, 0, 0]) # a=11, b=12
        V.append([0, 0, 0, 1]) # h=22
        W.append([0, 1, 0, 0]) # k=2, i=1 -> (2,1)
        
        # M6 = (c-a)(e+f)
        U.append([1, 0, -1, 0]) # a=11, c=21
        V.append([1, 1, 0, 0]) # e=11, f=12
        W.append([1, 0, 0, 0]) # k=1, i=2 -> (1,2)
        
        # M7 = (b-d)(g+h)
        U.append([0, 1, 0, -1]) # b=12, d=22
        V.append([0, 0, 1, 1]) # g=21, h=22
        W.append([0, 1, 0, 0]) # k=2, i=1 -> (2,1)
        
        return U, V, W

    # Helper to generate 4x4 Strassen factors using 2x2 Strassen
    def get_4x4_strassen_factors():
        U_2x2, V_2x2, W_2x2 = get_2x2_strassen_factors()
        U = []
        V = []
        W = []
        
        # 7 multiplications of 2x2 matrices
        # M1 = (A11+A22)(B11+B22)
        # A' = A11+A22, B' = B11+B22
        # A' is 2x2, B' is 2x2
        # A' entries: A'11=A11, A'12=A12, A'21=A21, A'22=A22
        # Wait, A11, A12, A21, A22 are 2x2 blocks.
        # So A'11 = A11 (2x2), A'12 = A12 (2x2), etc.
        # This is not correct. A11 is a 2x2 block.
        # So A' is a 2x2 matrix of 2x2 blocks.
        # The Strassen decomposition applies to the 2x2 block matrix.
        # Each block multiplication is a 2x2 matrix multiplication.
        # So we need to apply the 2x2 Strassen decomposition to each block multiplication.
        
        # Let's define the 7 block multiplications
        # M1: (A11+A22) * (B11+B22)
        # M2: (A21+A22) * B11
        # M3: A11 * (B12-B22)
        # M4: A22 * (B21-B11)
        # M5: (A11+A12) * B22
        # M6: (A21-A11) * (B11+B12)
        # M7: (A12-A22) * (B21+B22)
        
        # For each block multiplication, we use the 2x2 Strassen factors.
        # The 2x2 Strassen factors are for 2x2 scalar matrices.
        # Here, the "scalars" are 2x2 matrices.
        # So we need to "expand" the 2x2 Strassen factors to 4x4.
        
        # For M1: A' = A11+A22, B' = B11+B22
        # A' is a 2x2 matrix of 2x2 blocks.
        # A'11 = A11, A'12 = A12, A'21 = A21, A'22 = A22
        # Wait, A11+A22 means block-wise addition.
        # So A'11 = A11 + A22? No.
        # A = [[A11, A12], [A21, A22]]
        # A11+A22 is not a block of A.
        # Strassen's algorithm defines new matrices:
        # M1 = (A11+A22)(B11+B22)
        # Here A11+A22 is a 2x2 matrix (sum of two 2x2 blocks).
        # So we need to compute the product of two 2x2 matrices: (A11+A22) and (B11+B22).
        # This is a 2x2 matrix multiplication.
        # We can use the 2x2 Strassen decomposition for this.
        # The "entries" of A11+A22 are the sums of the corresponding entries of A11 and A22.
        # So we need to generate the factors for this 2x2 multiplication.
        
        # Let's define a function to generate factors for a 2x2 multiplication of 2x2 blocks.
        def get_block_strassen_factors(A_coeffs, B_coeffs, W_coeffs):
            # A_coeffs: list of (i, j, coeff) for the 2x2 matrix A
            # B_coeffs: list of (j, k, coeff) for the 2x2 matrix B
            # W_coeffs: list of (k, i, coeff) for the 2x2 matrix W
            # This is too complex. Let's just generate the 4x4 factors directly.
            
            pass

        # Actually, it's easier to just use the known 4x4 Strassen factors.
        # I will hardcode the 49 factors for M_{4,4,4} Strassen.
        # Or I can generate them by simulating the algorithm.
        
        # Let's try to generate them.
        # The 4x4 Strassen algorithm uses 7 block multiplications.
        # Each block multiplication is a 2x2 matrix multiplication.
        # Each 2x2 matrix multiplication uses 7 scalar multiplications.
        # Total 49 scalar multiplications.
        
        # For each scalar multiplication, we have a rank-1 tensor.
        # The rank-1 tensor is U_{(i,j)} V_{(j,k)} W_{(k,i)}.
        # We need to find i,j,k for each scalar multiplication.
        
        # This is a standard exercise. I will implement it.
        
        return U, V, W

    # Since generating the factors is complex, I will use a simpler approach.
    # I will use the schoolbook decomposition for M_{4,4,4} (rank 64) and M_{4,4,4} (rank 64).
    # Total rank 128. This is worse than 112.
    # So I must use Strassen.
    
    # I will implement the 4x4 Strassen factors generation.
    
    return {"U": U, "V": V, "W": W}
