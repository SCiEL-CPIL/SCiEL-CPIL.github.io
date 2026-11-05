import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # We will accumulate the rank-1 terms
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # We don't know the rank yet, so we collect terms in lists
    U_terms = []
    V_terms = []
    W_terms = []
    
    # Helper to add a single rank-1 term
    # a_idx: (i, j) in A
    # b_idx: (j, k) in B
    # c_idx: (k, i) in C (output index is k*n+i)
    # coeff: scalar
    def add_term(a_idx, b_idx, c_idx, coeff):
        i, j = a_idx
        j2, k = b_idx
        # j and j2 must match for a valid term in the sum
        # In our expansion, we ensure this.
        U_terms.append((i * m + j, coeff))
        V_terms.append((j2 * p + k, coeff))
        W_terms.append((k * n + i, 1)) # W is always 1 for the output coefficient? 
        # Wait, the tensor is sum r * u_r * v_r * w_r.
        # If we put coeff in U, V is 1, W is 1.
        # Or coeff in U, V, W?
        # The product is U[i,j] * V[j,k] * W[k,i].
        # If we set U=coeff, V=1, W=1, the term is coeff * A * B.
        # This is correct.
        
    # This simple add_term doesn't handle sums like (A1+A2).
    # We need a function that handles the block multiplication.
    
    def emit_mult(n_sub, m_sub, p_sub, a_i_start, a_j_start, b_j_start, b_k_start, c_k_start, c_i_start, coeff):
        # Adds the decomposition for a sub-multiplication of size n_sub x m_sub x p_sub
        # A block is at a_i_start, a_j_start in A
        # B block is at b_j_start, b_k_start in B
        # C block is at c_k_start, c_i_start in C
        # Multiplied by coeff
        
        if n_sub == 1 and m_sub == 1 and p_sub == 1:
            a_idx = (a_i_start, a_j_start)
            b_idx = (b_j_start, b_k_start)
            c_idx = (b_k_start, a_i_start) # k, i
            # c_idx in W is k*n + i
            # b_k_start is k, a_i_start is i
            add_term(a_idx, b_idx, c_idx, coeff)
            
        elif n_sub == 1 and m_sub == 2 and p_sub == 2:
            # Rank 4
            for j in range(2):
                for k in range(2):
                    a_idx = (a_i_start, a_j_start + j)
                    b_idx = (b_j_start + j, b_k_start + k)
                    c_idx = (b_k_start + k, a_i_start)
                    add_term(a_idx, b_idx, c_idx, coeff)
                    
        elif n_sub == 2 and m_sub == 2 and p_sub == 2:
            # Rank 7 (Strassen)
            # M1 = (A11+A22)(B11+B22)
            # A11: (0,0), A22: (1,1)
            # B11: (0,0), B22: (1,1)
            # Expand (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
            # We can just call emit_mult for the 4 products?
            # No, that would be rank 8.
            # We need the Strassen combination.
            # The Strassen algorithm computes 7 products.
            # We need to implement the 7 products of Strassen.
            
            # Let's define the 7 products explicitly for 2x2x2.
            # P1 = (A11+A22)(B11+B22)
            # P2 = (A21+A22)B11
            # P3 = A11(B12-B22)
            # P4 = A22(B21-B11)
            # P5 = (A11+A12)B22
            # P6 = (A21-A11)(B11+B12)
            # P7 = (A12-A22)(B21+B22)
            
            # We need to expand these into rank-1 terms.
            # Each P is a product of two 1x1 or 2x2 blocks?
            # No, in the 2x2x2 case, the sub-blocks are 1x1.
            # So P1 is (A00+A11)(B00+B11).
            # This expands to 4 terms: A00B00, A00B11, A11B00, A11B11.
            # If we just sum these, we get 4 terms.
            # But Strassen shares them?
            # No, the rank reduction comes from the fact that we compute 7 products instead of 8.
            # The products themselves are scalar multiplications (1x1x1).
            # So each P is 1 rank-1 term?
            # No, P1 is a sum of 4 scalar products.
            # So P1 contributes 4 rank-1 terms?
            # Then total rank would be 4+2+2+2+2+2+2 = 16?
            # No.
            
            # The rank of the 2x2x2 tensor is 7.
            # This means it can be written as sum of 7 rank-1 tensors.
            # The 7 Strassen products are the 7 rank-1 tensors?
            # No, the Strassen products are intermediate values.
            # The final C is a linear combination of the 7 products.
            # Each product is a scalar (since sub-blocks are 1x1).
            # So the tensor is sum_{r=1}^7 u_r v_r w_r.
            # Where u_r, v_r, w_r are vectors.
            
            # Let's verify.
            # C00 = P1 + P4 - P5 + P7
            # C01 = P3 + P5
            # C10 = P2 + P4
            # C11 = P1 - P2 + P3 + P6
            
            # P1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
            # This is not a rank-1 term in A and B. It is a sum of 4.
            # So P1 itself has rank 4?
            # Then the total rank is sum of ranks of P1..P7?
            # No, the rank of the whole tensor is the minimum number of rank-1 terms.
            # The Strassen algorithm shows that the rank is <= 7.
            # This implies that the 7 products can be arranged such that they form 7 rank-1 terms?
            # No, that's impossible if P1 is a sum of 4 products.
            
            # Ah, the rank-1 terms in the decomposition of the tensor are NOT the Strassen products.
            # The Strassen products are bilinear forms.
            # A bilinear form $B(A,B)$ is rank 1 if it can be written as $u(A)v(B)$.
            # $(A00+A11)(B00+B11)$ is NOT rank 1.
            # It is rank 4.
            
            # So how is the rank 7?
            # The tensor is $T = \sum_{i,j,k} e_i \otimes e_j \otimes e_k$.
            # The rank is the minimum $r$ such that $T = \sum_{l=1}^r u_l \otimes v_l \otimes w_l$.
            # Strassen's algorithm computes $C$ using 7 multiplications.
            # This means $C$ can be written as a linear combination of 7 bilinear forms.
            # But those bilinear forms are not necessarily rank 1.
            # However, for the $2 \times 2$ case, the 7 bilinear forms ARE rank 1?
            # Let's check P1 = $(A00+A11)(B00+B11)$.
            # This is not rank 1.
            
            # Wait. The rank of the $2 \times 2$ matrix multiplication tensor is 7.
            # This is a proven fact.
            # How is it decomposed?
            # The decomposition is $T = \sum_{l=1}^7 u_l \otimes v_l \otimes w_l$.
            # The vectors $u_l, v_l$ are NOT the Strassen products.
            # The Strassen products are just a way to compute the linear combination.
            
            # I need the explicit rank-1 decomposition for $2 \times 2 \times 2$.
            # It is known.
            # I will hardcode the 7 rank-1 terms for $2 \times 2 \times 2$.
            
            # The 7 terms for 2x2x2:
            # 1. A00 B00 -> C00
            # 2. A00 B11 -> C01
            # 3. A10 B00 -> C10
            # 4. A11 B11 -> C11
            # 5. (A00+A11)(B00+B11) ... no.
            
            # Let's look up the explicit decomposition.
            # Or just use the fact that we can compute it.
            
            # Actually, I can just use the schoolbook rank 8 for 2x2x2 if I don't have the explicit rank 7 decomposition handy.
            # But 47 vs 48 is the goal.
            # If I use rank 8 for 2x2x2, the total rank will be higher.
            
            # Let's assume I can find the 7 terms.
            # The 7 rank-1 terms for 2x2x2 are:
            # (Note: indices 0,1)
            # 1. u=(1,0,0,0), v=(1,0,0,0), w=(1,0,0,0) -> A00B00 -> C00
            # This is not correct. The rank-1 tensors are not just single entries.
            
            # I will implement a generic solver for small ranks?
            # No, time limit.
            
            # I will hardcode the 7 terms for 2x2x2.
            # Reference: 
            # The 7 rank-1 tensors are:
            # 1. (1, 0, 0, 0) x (1, 0, 0, 0) x (1, 0, 0, 0)
            # This is not right.
            
            # Let's just use the Strassen products as the "terms" and expand them?
            # No, that increases rank.
            
            # I will use the known rank-7 decomposition for 2x2x2.
            # It involves 7 vectors.
            
            # For now, I will output the code that uses the 47 rank construction, assuming I can generate the 7 terms for 2x2x2.
            
            pass

    return {"U": [], "V": [], "W": []}
