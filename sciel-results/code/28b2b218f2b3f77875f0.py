import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Schoolbook rank for n*m*p
    rank_schoolbook = n * m * p  # 54
    
    # Build schoolbook decomposition
    U = np.zeros((n * m, rank_schoolbook), dtype=object)
    V = np.zeros((m * p, rank_schoolbook), dtype=object)
    W = np.zeros((p * n, rank_schoolbook), dtype=object)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = Fraction(1)
                V[j * p + k, col] = Fraction(1)
                W[k * n + i, col] = Fraction(1)
                col += 1
    
    # Try to improve using rank-23 decomposition for 3x3*3x3
    # We can view 3x3 * 3x6 as two 3x3 * 3x3 multiplications
    # Use Winograd's rank-23 algorithm for each 3x3*3x3 block
    
    if time.time() < deadline:
        try:
            # Winograd's rank-23 decomposition for 3x3 * 3x3
            # The tensor T satisfies: T[i,j,k] = 1 if i==j or j==k or i==k, else 0
            # Actually, for matrix multiplication C = A*B where A is 3x3, B is 3x3:
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # So the tensor has T[i,j,k] = 1 for all valid (i,j,k) in the multiplication pattern
            
            # Winograd's 23-term decomposition:
            # We need U (9x23), V (9x23), W (9x23) such that
            # sum_t U[i,j,t] * V[j',k,t] * W[k',i',t] reconstructs the multiplication tensor
            
            # Let me define the rank-23 decomposition explicitly.
            # From literature, one such decomposition uses the following structure:
            
            # The 23 terms correspond to:
            # 9 diagonal terms (i=j=k)
            # 12 off-diagonal terms
            
            # Let's define the matrices using a known valid decomposition.
            # I'll construct it using the method from Winograd (1971).
            
            # For 3x3*3x3, the multiplication tensor can be decomposed as:
            # T = sum_{t=1}^{23} u_t ⊗ v_t ⊗ w_t
            
            # Here's one valid rank-23 decomposition (from known results):
            # The entries are integers and half-integers.
            
            # Let me define U, V, W for one 3x3*3x3 block
            # U is 9x23, V is 9x23, W is 9x23
            
            # Indexing: A is 3x3 with indices (i,j), B is 3x3 with indices (j,k), C is 3x3 with indices (i,k)
            # U[i*m+j, t] for i,j in 0..2, t in 0..22
            # V[j*p+k, t] for j,k in 0..2, t in 0..22  
            # W[k*n+i, t] for k,i in 0..2, t in 0..22
            
            # Winograd's decomposition:
            # Terms 0-8: diagonal terms where i=j=k
            # Terms 9-22: off-diagonal terms
            
            U_23 = np.zeros((9, 23), dtype=object)
            V_23 = np.zeros((9, 23), dtype=object)
            W_23 = np.zeros((9, 23), dtype=object)
            
            # Diagonal terms (9 terms): for each d in 0,1,2
            for d in range(3):
                t = d
                U_23[d*3+d, t] = Fraction(1)
                V_23[d*3+d, t] = Fraction(1)
                W_23[d*3+d, t] = Fraction(1)
            
            # Off-diagonal terms (14 terms)
            # We need to cover all (i,j,k) where not all are equal
            # There are 27 - 9 = 18 off-diagonal positions, but we can group them
            
            # Winograd's method uses pairs of off-diagonal terms
            # Let me use a known valid set of 14 additional terms
            
            # Actually, let me use a simpler approach: use the known fact that
            # rank(3x3*3x3) <= 23 and construct it step by step.
            
            # The key insight: we can use the following decomposition pattern.
            # For each pair (i,j) with i != j, we can create terms that handle
            # multiple k values simultaneously.
            
            # Let me define the 14 off-diagonal terms explicitly:
            # Term 9: handles (0,1,0), (0,1,1), (0,1,2) -> but this doesn't work directly
            
            # Let me use a different known decomposition.
            # From the literature, here's a valid rank-23 decomposition:
            
            # I'll use the following construction based on Winograd's algorithm:
            
            # Terms 9-11: for row 0 of A
            # Term 9: U[0,1]=1, V[1,0]=1, W[0,0]=1
            # Term 10: U[0,2]=1, V[2,0]=1, W[0,0]=1  
            # Term 11: U[1,2]=1, V[2,1]=1, W[1,0]=1
            
            # This is getting complex. Let me use a verified decomposition from literature.
            
            # Actually, the simplest approach: use the known result that 
            # M(3,3,3) <= 23 and construct it using the following pattern:
            
            # I'll use the decomposition from Bini et al. or similar.
            
            # For now, let me try a different strategy: use the fact that
            # we can decompose 3x3*3x6 as two 3x3*3x3 multiplications.
            
            # Let me define the rank-23 decomposition more carefully.
            
            # The multiplication tensor for 3x3*3x3 has 27 non-zero entries.
            # We need to find 23 rank-1 tensors whose sum equals this tensor.
            
            # One known decomposition uses the following structure:
            # 9 "diagonal" terms + 14 "off-diagonal" terms
            
            # Let me define them explicitly:
            
            # Diagonal terms (t=0,1,2):
            # t=0: U[0,0]=1, V[0,0]=1, W[0,0]=1  (i=0,j=0,k=0)
            # t=1: U[4,4]=1, V[4,4]=1, W[4,4]=1  (i=1,j=1,k=1)  
            # t=2: U[8,8]=1, V[8,8]=1, W[8,8]=1  (i=2,j=2,k=2)
            
            # Wait, the indexing is:
            # U index: i*m + j = i*3 + j
            # V index: j*p + k = j*3 + k
            # W index: k*n + i = k*3 + i
            
            # So for (i,j,k) = (0,0,0): U[0,0], V[0,0], W[0,0]
            # For (i,j,k) = (1,1,1): U[4,4], V[4,4], W[4,4]
            # For (i,j,k) = (2,2,2): U[8,8], V[8,8], W[8,8]
            
            # Now for off-diagonal terms, I need to be more careful.
            
            # Let me use a known valid rank-23 decomposition.
            # Here's one from the literature (I'll verify it works):
            
            # The 23 terms are defined as follows (using 0-indexed):
            
            # I'll construct it using the following approach:
            # Start with all 27 schoolbook terms, then combine 4 pairs to save 4 terms.
            
            # Pair 1: (0,1,0) and (1,0,0) can be combined
            # Pair 2: (0,1,1) and (1,0,1) can be combined
            # etc.
            
            # Actually, let me just use the schoolbook for now since I'm not sure
            # about the exact rank-23 decomposition coefficients.
            
            # Let me try to verify if rank 46 is achievable by checking if
            # two independent rank-23 decompositions work.
            
            # For now, return the schoolbook decomposition (rank 54).
            pass
        except Exception as e:
            pass
    
    # Convert to lists with integer or half-integer values
    def to_list(mat):
        result = []
        for row in mat:
            row_list = []
            for val in row:
                if isinstance(val, Fraction):
                    # Convert to float if it's a simple fraction
                    num = val.numerator
                    den = val.denominator
                    if den == 1:
                        row_list.append(num)
                    elif den == 2:
                        row_list.append(num / 2.0)
                    else:
                        row_list.append(float(val))
                else:
                    row_list.append(float(val))
            result.append(row_list)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
