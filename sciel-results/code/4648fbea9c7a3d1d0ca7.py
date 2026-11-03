   def get_2x2_strassen():
       # Returns U, V, W of shape (4, 7)
       U = np.zeros((4, 7))
       V = np.zeros((4, 7))
       W = np.zeros((4, 7))
       
       # M1 = (A00+A11)(B00+B11)
       # A00: U[0,0]=1, V[0,0]=1
       # A11: U[3,0]=1, V[3,0]=1
       # B00: V[0,0]=1, W[0,0]=1
       # B11: V[3,0]=1, W[3,0]=1
       
       # This is getting messy.
       
       # Standard Strassen 2x2:
       # M1 = (A00+A11)(B00+B11)
       # M2 = (A00+A01)B11
       # M3 = A11(B00+B11)
       # M4 = (A10-A11)(B11-B00)
       # M5 = (A00+A10)B00
       # M6 = (A01-A00)(B11+B00)
       # M7 = (A00-A11)(B00+B11)
       
       # C00 = M1 + M4 - M3 + M7
       # C01 = M2 + M3
       # C10 = M5 + M6
       # C11 = M1 - M2 + M4 + M6
       
       # We need to expand each Mk into the tensor form sum_r U V W.
       
       # For M1:
       # Inputs: A00, A11, B00, B11.
       # This is a product of sums.
       # (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
       # Each term is a rank-1 tensor.
       # A00B00: U[0]=1, V[0]=1, W[0]=1.
       # A00B11: U[0]=1, V[3]=1, W[3]=1.
       # A11B00: U[3]=1, V[0]=1, W[0]=1.
       # A11B11: U[3]=1, V[3]=1, W[3]=1.
       
       # So M1 contributes 4 terms to the decomposition?
       # No, Strassen's "rank 7" refers to the number of scalar multiplications.
       # The tensor decomposition rank is the number of terms in the sum.
       # For 2x2, the rank is 7.
       # This means there are 7 terms.
       # But the expansion above has 4 terms for M1 alone.
       # The "rank 7" comes from the fact that we share the linear combinations.
       
       # The 7 terms are:
       # T1: (A00+A11) (B00+B11) -> This is NOT a rank-1 term. It's a product of two vectors.
       # The tensor is $T_{ijk} = \sum_{r=1}^7 U_{ir} V_{jr} W_{kr}$.
       
       # The 7 terms correspond to the 7 products of the linear forms.
       # L1 = A00+A11, L2 = A00+A01, L3 = A11, L4 = A10-A11, L5 = A00+A10, L6 = A01-A00, L7 = A00-A11.
       # R1 = B00+B11, R2 = B11, R3 = B00+B11, R4 = B11-B00, R5 = B00, R6 = B11+B00, R7 = B00+B11.
       
       # Term r is $L_r \otimes R_r \otimes (\text{coeffs in C})$.
       # Wait, the W factor depends on how C is formed.
       
       # This is too complex to derive in 5 minutes.
       
       # I will stick to Schoolbook.
   