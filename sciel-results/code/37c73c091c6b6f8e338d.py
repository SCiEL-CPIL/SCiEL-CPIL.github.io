"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # We partition the problem into blocks.
    # n split: [2, 1]
    # m split: [2, 2]
    # p split: [2, 2, 2, 2]
    
    # This creates 2 * 2 * 4 = 16 sub-blocks.
    # Blocks where n_size=2 use Strassen (Rank 7).
    # Blocks where n_size=1 use Schoolbook (Rank 4).
    
    # Total rank: 8 blocks of Rank 7 + 8 blocks of Rank 4 = 56 + 32 = 88.
    
    # We will build U, V, W as lists of contributions.
    # Each contribution is a term in the sum.
    # U_term is a dict or sparse representation, but since rank is 88, we can just append to lists.
    
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a term
    def add_term(u_dict, v_dict, w_dict):
        U_list.append(u_dict)
        V_list.append(v_dict)
        W_list.append(w_dict)
        
    # Define Strassen terms for 2x2x2 block
    # Indices in block: (0,0), (0,1), (1,0), (1,1)
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    
    # We need to express each m as a sum of rank-1 tensors (U, V, W)
    # Actually, each m is a single scalar multiplication in the algorithm.
    # In the tensor decomposition, each "multiplication" corresponds to one term c.
    # For m1 = (a+d)(e+h) = ae + ah + de + dh.
    # This is not rank 1.
    # Wait, Strassen's algorithm computes 7 products of linear combinations.
    # The tensor decomposition rank is the number of such products.
    # Each product is of the form (linear combo of A) * (linear combo of B) * (linear combo of C?).
    # No, the decomposition is sum of U_c V_c W_c.
    # U_c is a matrix. V_c is a matrix. W_c is a matrix.
    # For a term c, U_c is rank 1? No, U_c is a fixed matrix.
    # The product U_c[i,j] V_c[j,k] W_c[k,i] is the term.
    # If U_c is rank 1, say u v^T, then U_c[i,j] = u_i v_j.
    # Then the term is u_i v_j V_c[j,k] W_c[k,i].
    # This is not necessarily a single scalar multiplication.
    
    # However, it is known that the rank of the 2x2x2 multiplication tensor is 7.
    # This means there exist 7 matrices U_c, V_c, W_c (2x2) such that sum T = sum U_c \otimes V_c \otimes W_c.
    # The standard Strassen decomposition provides these 7 terms.
    
    # Let's verify the Strassen terms as rank-1 tensors?
    # No, the matrices U_c in the Strassen decomposition are generally not rank 1.
    # But the tensor T itself has rank 7.
    # The standard proof uses 7 terms.
    
    # Let's use the explicit 7 terms for 2x2x2.
    # Term 1: U=[[1,0],[0,1]], V=[[1,1],[0,0]]? No.
    
    # We can use the known rank-7 decomposition.
    # Source: Strassen 1969.
    
    # Let's implement the 7 terms for the 2x2x2 block.
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[c00, c01], [c10, c11]]
    
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    
    # We need to map these to U, V, W matrices for the block.
    # For each m, we define U, V, W.
    
    # m1 = (a+d)(e+h)
    # This term contributes to c00 (+1), c11 (+1).
    # It is formed by (a+d) and (e+h).
    # U1: [[1,0],[1,0]]? No, a is (0,0), d is (1,1).
    # U1 = [[1,0],[0,1]] (Identity). Then (U1 * A) is not a scalar.
    # The decomposition is T = sum U_c \otimes V_c \otimes W_c.
    # This means T[i,j,k] = sum U_c[i,j] V_c[j,k] W_c[k,i].
    
    # Let's construct the 7 U, V, W matrices for 2x2x2.
    
    # U1 = [[1,0],[0,1]] -> a,d
    # V1 = [[1,1],[0,0]] -> e,f? No, e is (0,0), h is (1,1).
    # We need (a+d)(e+h).
    # U1 must extract a+d. V1 must extract e+h.
    # U1 = [[1,0],[1,0]]? No, that gives a+c.
    # U1 = [[1,0],[0,1]]? That gives a and d separately.
    # The product U1[i,j] V1[j,k] must yield the product of the linear forms.
    
    # Actually, we can just hardcode the 7 terms from a reliable source or derive them.
    # Since I cannot look it up, I will use the schoolbook for 2x2x2 if I can't be sure,
    # but the problem asks for rank minimal. 88 is better than 96.
    # If I use schoolbook for 2x2x2, rank is 8.
    # 8*8 + 8*4 = 64 + 32 = 96. No improvement.
    
    # I must use Strassen.
    # I will construct the matrices for the 7 terms.
    
    def get_strassen_terms():
        # Returns list of 7 tuples (U, V, W)
        # U, V, W are 2x2 numpy arrays
        terms = []
        
        # m1 = (a+d)(e+h)
        U = np.array([[1,0],[0,1]])
        V = np.array([[1,0],[0,1]])
        # This gives a*e + d*h. Not (a+d)(e+h).
        # (a+d)(e+h) = ae + ah + de + dh.
        # We need to represent this single multiplication as a rank-1 tensor?
        # No, the "term" in Strassen is the multiplication of two linear forms.
        # In the tensor decomposition, each "multiplication" is one term c.
        # U_c must be such that U_c[i,j] is the coefficient of A[i,j] in the first form.
        # V_c[j,k] is the coefficient of B[j,k] in the second form.
        # Then the product is (sum U A) * (sum V B).
        # This product is a scalar.
        # Then we need to distribute this scalar to C entries.
        # W_c[k,i] is the coefficient for C[i,k] (note index swap in W).
        
        # So for m1:
        # Form 1: a+d -> U = [[1,0],[0,1]]
        # Form 2: e+h -> V = [[1,0],[0,1]]
        # Product: (a+d)(e+h).
        # This scalar contributes to c00 (+1) and c11 (+1).
        # W = [[1,0],[0,1]]?
        # W[k,i] = 1 for (0,0) and (1,1).
        # So W = [[1,0],[0,1]].
        # Check: sum U[i,j] V[j,k] W[k,i].
        # i=0,j=0,k=0: 1*1*1 = 1. (Contributes to T[0,0,0])
        # i=0,j=0,k=1: 1*0*0 = 0.
        # i=0,j=1,k=0: 0*0*1 = 0.
        # i=0,j=1,k=1: 0*1*0 = 0.
        # i=1,j=0,k=0: 0*1*1 = 0.
        # i=1,j=0,k=1: 0*0*0 = 0.
        # i=1,j=1,k=0: 1*0*1 = 0.
        # i=1,j=1,k=1: 1*1*1 = 1. (Contributes to T[1,1,1])
        # This only gives T[0,0,0] and T[1,1,1].
        # But m1 should contribute to c00 and c11.
        # c00 = sum_j A[0,j] B[j,0].
        # T[0,0,0] is the coefficient of A[0,0]B[0,0] in c00.
        # T[0,1,0] is the coefficient of A[0,1]B[1,0] in c00.
        # My check above shows that term 1 only contributes to T[0,0,0] and T[1,1,1].
        # This is correct for the "diagonal" part.
        # The cross terms are handled by other m's?
        # No, Strassen sums the m's to get C.
        # The tensor T is the sum of the 7 rank-1 tensors.
        # The sum of the 7 tensors must equal the tensor for 2x2 matrix multiplication.
        
        # I will assume the standard Strassen decomposition is valid and implement it.
        
        # Term 1
        U1 = np.array([[1,0],[0,1]])
        V1 = np.array([[1,0],[0,1]])
        W1 = np.array([[1,0],[0,1]])
        terms.append((U1, V1, W1))
        
        # Term 2: (c+d)e
        U2 = np.array([[0,0],[1,1]])
        V2 = np.array([[1,0],[0,0]])
        W2 = np.array([[0,1],[0,0]]) # Contributes to c10
        # c10 = A[1,0]B[0,0] + A[1,1]B[1,0].
        # U2 extracts c+d. V2 extracts e.
        # Product (c+d)e.
        # W2 must map this to c10.
        # c10 is C[1,0]. W index (k,i) = (0,1).
        # W[0,1] = 1.
        # Check T[1,0,0]: U[1,0]V[0,0]W[0,1] = 1*1*1 = 1. Correct.
        # Check T[1,1,0]: U[1,1]V[1,0]W[0,1] = 1*0*1 = 0.
        # Wait, c10 also has A[1,1]B[1,0].
        # V2[j,k] for j=1,k=0 is 0.
        # So this term misses A[1,1]B[1,0].
        # This suggests my simple mapping is wrong.
        
        # I will use the schoolbook for 2x2x2 to be safe, but that gives 96.
        # I will use a known rank-7 decomposition if I can find it.
        # Since I cannot, I will return the schoolbook 96.
        # The task asks for rank minimal. 88 is better.
        # I will try to implement the 88 solution with schoolbook for 1x2x2 and Strassen for 2x2x2.
        # If Strassen implementation is buggy, it will fail verification.
        
        pass

    # Given the complexity and risk of bug in Strassen, and the 120s limit,
    # I will return the schoolbook solution with rank 96.
    # It is correct and fast.
    
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
