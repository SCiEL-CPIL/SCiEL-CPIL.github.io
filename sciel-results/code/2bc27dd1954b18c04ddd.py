import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = 4, 4, 7
    
    terms = []
    
    def add_term(i, j, k, u_val, v_val, w_val):
        # u index: i*m + j
        # v index: j*p + k
        # w index: k*n + i
        terms.append((i * m + j, j * p + k, k * n + i, u_val, v_val, w_val))

    def add_strassen_2x2(row_offset, col_offset, k_offset):
        # Adds the 7 terms for a 2x2x2 multiplication 
        # A block at rows [row_offset, row_offset+2), cols [col_offset, col_offset+2)
        # B block at rows [col_offset, col_offset+2), cols [k_offset, k_offset+2)
        # C block at rows [row_offset, row_offset+2), cols [k_offset, k_offset+2)
        
        # Indices within the 2x2 block:
        # 0,0 -> 0
        # 0,1 -> 1
        # 1,0 -> 2
        # 1,1 -> 3
        
        # Strassen terms:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # This adds 7 rank-1 terms? No, Strassen uses 7 multiplications.
        # Each multiplication is a rank-1 term in the tensor decomposition?
        # Yes, $A_{ij} B_{jk}$ is a rank-1 term.
        # So we add 7 terms.
        
        # M1: (A00+A11) * (B00+B11)
        # This corresponds to 4 scalar multiplications in schoolbook, but in Strassen it's one "multiplication" of sums.
        # Wait, the tensor decomposition is over the field.
        # The term is $\sum_{i,j,k} U_{ij} V_{jk} W_{ki}$.
        # A Strassen term like $M_1 = (A_{00}+A_{11})(B_{00}+B_{11})$ expands to $A_{00}B_{00} + A_{00}B_{11} + A_{11}B_{00} + A_{11}B_{11}$.
        # This is a sum of 4 rank-1 terms.
        # So if I use Strassen, I am using 7 * 4 = 28 terms?
        # No! Strassen reduces the number of multiplications, but the decomposition rank is the number of scalar multiplications.
        # In the tensor decomposition, each scalar multiplication $A_{ij} B_{jk}$ is a rank-1 tensor.
        # Strassen computes $C$ using 7 multiplications of linear combinations.
        # But the final result $C$ is a linear combination of these 7 products.
        # Does this give a rank-7 decomposition of the $2 \times 2 \times 2$ tensor?
        # Yes. The $2 \times 2 \times 2$ multiplication tensor has rank 7.
        # The 7 terms are the 7 Strassen products.
        # So for $4 \times 4 \times 4$, using Strassen, we use 7 blocks.
        # Each block is a $2 \times 2 \times 2$ multiplication.
        # Each $2 \times 2 \times 2$ multiplication has rank 7.
        # So total rank is $7 \times 7 = 49$.
        
        # So I need to add the 7 Strassen terms for each $2 \times 2 \times 2$ block.
        
        # Let's define the 7 terms for a local 2x2x2.
        # Local indices i,j,k in {0,1}.
        # Global i = row_offset + i
        # Global j = col_offset + j
        # Global k = k_offset + k
        
        # Term 1: (A00+A11)(B00+B11)
        # This is not a single rank-1 term in the original basis.
        # The rank-1 terms in the decomposition are of the form $u v^T w^T$.
        # The Strassen algorithm expresses the tensor as a sum of 7 rank-1 tensors.
        # What are these 7 rank-1 tensors?
        # They are not simply $A_{ij} B_{jk}$.
        # They are products of linear combinations.
        # Example: $M_1 = (A_{00}+A_{11})(B_{00}+B_{11})$.
        # This term contributes to $C_{00}, C_{01}, C_{10}, C_{11}$?
        # No, $M_1$ is a scalar.
        # $C_{00} = M_1 + M_4 - M_5 + M_7$.
        # $C_{01} = M_3 + M_5$.
        # $C_{10} = M_2 + M_4$.
        # $C_{11} = M_1 - M_2 + M_3 + M_6$.
        
        # So the decomposition is:
        # $T = \sum_{r=1}^7 u_r \otimes v_r \otimes w_r$
        # where $u_r, v_r, w_r$ are vectors.
        
        # For M1:
        # $u_1$ has 1 at (0,0) and 1 at (1,1).
        # $v_1$ has 1 at (0,0) and 1 at (1,1).
        # $w_1$ has 1 at (0,0), 1 at (1,1), -1 at (0,1), 1 at (1,0)?
        # Let's check $C_{00}$: $w_1[0,0] = 1$.
        # $C_{11}$: $w_1[1,1] = 1$.
        # $C_{01}$: $w_1[0,1] = 0$? No, $C_{01}$ doesn't use M1.
        # $C_{10}$: $w_1[1,0] = 0$? No, $C_{10}$ doesn't use M1.
        # So $w_1$ has 1 at (0,0) and 1 at (1,1).
        
        # So for each Strassen product $M_r$, we define:
        # $u_r$: linear combination of A entries.
        # $v_r$: linear combination of B entries.
        # $w_r$: linear combination of C entries (coefficients in the reconstruction formula).
        
        # I will implement this.
        
        pass

    # This is getting too complex to write correctly in a short edit.
    # I will stick to the schoolbook for the 4x4x3 part and use a known 4x4x4 rank 49 decomposition?
    # Actually, just using schoolbook for everything is rank 112.
    # Using Strassen for 4x4x4 (rank 49) and schoolbook for 4x4x3 (rank 48) gives 97.
    
    # Let's just output the schoolbook for now if I can't get Strassen right?
    # No, I must improve.
    
    # I will implement the 4x4x4 Strassen decomposition explicitly.
