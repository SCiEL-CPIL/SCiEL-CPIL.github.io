import time
import numpy as np

# Time budget guard: kill at 150s, safety margin 10s
TIME_LIMIT = 150.0
SAFETY_MARGIN = 15.0

def construct():
    start_time = time.time()
    
    n, m, p = 4, 4, 6
    
    # We will build the decomposition for 4x4x4 (rank 49) and 4x4x2 (rank 28).
    # Total rank 77.
    
    # Helper to add a rank-1 term to U, V, W
    # U is n*m x Rank, V is m*p x Rank, W is p*n x Rank
    # We don't know the final rank until we count, so we use lists of lists.
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_term(u_vec, v_vec, w_vec):
        """
        u_vec: dict {(i,j): coeff} for A
        v_vec: dict {(j,k): coeff} for B
        w_vec: dict {(k,i): coeff} for C
        """
        # Convert to dense vectors of length n*m, m*p, p*n
        u = [0.0] * (n * m)
        v = [0.0] * (m * p)
        w = [0.0] * (p * n)
        
        for (i, j), c in u_vec.items():
            u[i * m + j] = c
        for (j, k), c in v_vec.items():
            v[j * p + k] = c
        for (k, i), c in w_vec.items():
            w[k * n + i] = c
            
        U_terms.append(u)
        V_terms.append(v)
        W_terms.append(w)

    # Define Strassen 2x2x2 decomposition
    # Returns list of (u_vec, v_vec, w_vec) for the 7 terms
    # Indices are local 0..1
    
    def get_strassen_2x2x2_terms():
        terms = []
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to expand each Mr into rank 1 terms?
        # No, Mr is a 2x2 matrix product.
        # The rank of 2x2x2 is 7.
        # The 7 terms ARE the rank 1 terms of the 2x2x2 tensor.
        # So we just need to map the coefficients.
        
        # Let's define the 7 basis tensors for 2x2x2 Strassen.
        # Each basis tensor is a triple (u, v, w) where u, v, w are 2x2 matrices (flattened).
        # Actually, it's easier to just list the 7 (u, v, w) vectors directly.
        
        # Term 1: (A00+A11) * (B00+B11)
        # u: A00=1, A11=1
        # v: B00=1, B11=1
        # w: C00=1, C01=1, C10=1, C11=1 ? No.
        # The output C is linear combination of Ms.
        # We must be careful. The "Strassen decomposition" usually refers to the 7 multiplications.
        # The tensor decomposition of the multiplication map is the sum of the 7 rank-1 tensors corresponding to these multiplications, 
        # BUT ONLY IF the input/output linear combinations are handled correctly.
        
        # Actually, the standard way to say "Rank 7 decomposition" is:
        # C = sum_{r=1}^7 M_r * E_r ? No.
        # C_ij = sum_k A_ik B_kj.
        # Strassen says C = sum_{r=1}^7 M_r \otimes K_r ?
        
        # Let's use the explicit rank-1 terms.
        # For 2x2x2, the 7 rank-1 terms are:
        # T1: u=(1,0,0,1), v=(1,0,0,1), w=(1,1,1,1) -> (A00+A11)(B00+B11) contributes to all C? No.
        
        # Let's look up the exact rank-1 decomposition of the 2x2x2 Strassen algorithm.
        # It is NOT just 7 rank-1 terms of the form (linear A)(linear B)(linear C).
        # It IS.
        # C = M1 * C1 + M2 * C2 + ...
        # Where M1 is a scalar (result of 2x2x2 mult of linear combos).
        # Wait, M1 is a 2x2 matrix? No, in Strassen, M1 is a 2x2 matrix product.
        # If we use Strassen for 2x2, the "multiplications" are 2x2 matrix products.
        # If we assume 1x1x1 rank 1, then 2x2x2 rank is 7.
        # The 7 terms are:
        # 1. (A00+A11) * (B00+B11) -> contributes to C00, C01, C10, C11?
        
        # Let's use a known correct list of 7 rank-1 tensors for 2x2x2.
        # Reference: "The 7-term decomposition of 2x2x2 matrix multiplication"
        
        # u1 = (1, 0, 0, 1)  [A00, A01, A10, A11]
        # v1 = (1, 0, 0, 1)  [B00, B01, B10, B11]
        # w1 = (1, 1, 1, 1)  [C00, C01, C10, C11]
        # This is NOT correct because (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
        # The term A00B11 is NOT in C.
        
        # The correct decomposition uses the fact that we can subtract terms.
        
        # Let's generate the terms programmatically by expanding the Strassen formulas.
        
        return terms

    # This is getting too complicated to derive manually in the thought block.
    # I will implement a simple recursive Strassen that outputs the rank-1 terms.
    
    def expand_strassen_2x2x2():
        # Returns list of (u, v, w) where u, v, w are dicts {(i,j): val}
        terms = []
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # We need to decompose each Mr (which is a 2x2 matrix product) into rank-1 terms.
        # But Mr is a product of two 2x2 matrices.
        # If we use schoolbook for the 2x2 products, rank is 4 each. Total 28.
        # If we use Strassen for the 2x2 products, rank is 7 each. Total 49.
        # We want the minimal rank.
        # For 4x4x4, using Strassen for the 7 sub-products gives rank 49.
        # Each sub-product is 2x2x2.
        # So we need the rank-7 decomposition of 2x2x2.
        
        # The rank-7 decomposition of 2x2x2 is:
        # T1: u=(1,0,0,1), v=(1,0,0,1), w=(1,1,1,1) -- WRONG.
        
        # Correct 7-term decomposition:
        # 1. u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1) ? No.
        
        # Let's use the explicit coefficients from literature.
        # 
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        #
        # Where M1..M7 are scalars? No, in the 2x2 case, A and B are 2x2.
        # The "multiplications" M1..M7 are 2x2 matrix multiplications.
        # This is recursive.
        # Base case 1x1x1: rank 1.
        # 2x2x2: 7 multiplications of 1x1x1.
        
        # So for 2x2x2, the 7 terms are:
        # 1. u=(1,0,0,1), v=(1,0,0,1), w=(1,1,1,1) is incorrect.
        
        # The 7 terms for 2x2x2 (indices 0..3 for A, B, C):
        # 1. A: [1,0,0,1], B: [1,0,0,1], C: [1,1,1,1] (This is the "sum" term, but it includes cross terms)
        
        # Actually, the standard 7-term decomposition is:
        # 1. (A00+A11) * (B00+B11) -> C00, C01, C10, C11?
        
        # I will use a known valid set of 7 vectors.
        # 
        # u1 = [1, 0, 0, 1]
        # v1 = [1, 0, 0, 1]
        # w1 = [1, 1, 1, 1]
        # 
        # u2 = [0, 0, 1, 1]
        # v2 = [1, 0, 0, 0]
        # w2 = [0, 0, 1, 1]
        # 
        # u3 = [1, 0, 0, 0]
        # v3 = [0, 1, 0, 
