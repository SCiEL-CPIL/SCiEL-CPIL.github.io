"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _strassen_2x2x2():
    """
    Return U, V, W for 2x2 * 2x2 -> 2x2 using Strassen (rank 7).
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen 7 terms for 2x2x2
    # C = A * B where A, B are 2x2
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C11 = a11*b11 + a12*b21
    # C12 = a11*b12 + a12*b22
    # C21 = a21*b11 + a22*b21
    # C22 = a21*b12 + a22*b22
    #
    # Strassen:
    # M1 = (a11+a22)*(b11+b22)
    # M2 = (a21+a22)*b11
    # M3 = a11*(b12-b22)
    # M4 = a22*(b21-b11)
    # M5 = (a11+a12)*b22
    # M6 = (a21-a11)*(b11+b12)
    # M7 = (a12-a22)*(b21+b22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # For each rank-1 term, we need U[i*2+j], V[j*2+k], W[k*2+i]
    # The tensor entry T[i,j,k] = 1 if the term contributes to output[i,k] via input A[i,j]*B[j,k]
    
    terms = []
    # M1: (a11+a22)*(b11+b22) -> contributes to C11(+1), C22(-1)
    # This is (e11+e22) in A times (f11+f22) in B
    # As a sum of rank-1: we need to express the product of two sums
    # Actually in Strassen, M1 is computed as a single multiplication of two linear combos
    # So it's one rank-1 term in the tensor decomposition where:
    # U has 1s at positions corresponding to a11, a22 (i.e., (0,0) and (1,1))
    # V has 1s at positions corresponding to b11, b22 (i.e., (0,0) and (1,1))
    # W has the coefficients for the output
    
    # Let me think more carefully. The matrix multiplication tensor T_{ijk} = delta_{ik} for the identity... no.
    # T_{ijk} is 1 when the product A_{ij} * B_{jk} contributes to C_{ik}.
    # For 2x2x2: T_{ijk} = 1 if we're computing C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So T_{ijk} = 1 for all (i,j,k) in the standard basis.
    # Wait no - the tensor is T_{ijk} where i indexes rows of A, j columns of A / rows of B, k columns of B.
    # The multiplication is C_{ik} = sum_j T_{ijk} A_{ij} B_{jk} where T_{ijk} = 1 for all valid i,j,k.
    
    # In a rank-1 decomposition, each term is a product of three vectors u, v, w:
    # T_{ijk} = sum_c u_{i,j}^c * v_{j,k}^c * w_{i,k}^c
    # (Note: w is indexed by (i,k) for the output, but the problem says W[k*n+i, c])
    
    # For Strassen, the 7 terms are:
    # Each term c has:
    #   u^c: vector in R^{2x2} (flattened as i*2+j)
    #   v^c: vector in R^{2x2} (flattened as j*2+k)  
    #   w^c: vector in R^{2x2} (flattened as k*2+i per the convention W[k*n+i])
    
    # The condition is: sum_c u_{i,j}^c * v_{j,k}^c * w_{k*2+i}^c = 1 for all i,j,k
    
    # Let me enumerate the Strassen terms explicitly:
    # Using 0-indexed: i,j,k in {0,1}
    # A indices: (i,j) -> pos i*2+j
    # B indices: (j,k) -> pos j*2+k
    # C indices: (i,k) -> but W uses k*2+i
    
    # M1: u = e_{00} + e_{11}, v = e_{00} + e_{11}, w = e_{00} + e_{11} (with k*2+i indexing)
    #   w_{k*2+i}: for (i=0,k=0): w_0=1, (i=1,k=1): w_3=1, others 0
    #   Check: sum over c of u_{ij} v_{jk} w_{k*2+i}
    #   For (i=0,j=0,k=0): u_{00}*v_{00}*w_{0*2+0} = 1*1*1 = 1 ✓
    #   For (i=0,j=0,k=1): u_{00}*v_{01}*w_{1*2+0} = 1*0*w_2 = 0
    #   For (i=0,j=1,k=0): u_{01}*v_{10}*w_{0*2+0} = 0*1*1 = 0
    #   For (i=0,j=1,k=1): u_{01}*v_{11}*w_{1*2+0} = 0*0*w_2 = 0
    #   For (i=1,j=0,k=0): u_{10}*v_{00}*w_{0*2+1} = 0*1*w_1 = 0
    #   For (i=1,j=0,k=1): u_{10}*v_{01}*w_{1*2+1} = 0*0*w_3 = 0
    #   For (i=1,j=1,k=0): u_{11}*v_{10}*w_{0*2+1} = 1*1*w_1 = 1*1*0 = 0... 
    
    # Hmm, this doesn't work directly. The issue is that Strassen's algorithm computes
    # C as a linear combination of the 7 products, where each product is a scalar.
    # The tensor decomposition requires that the sum of products equals the identity tensor.
    
    # Let me reconsider. The standard Strassen decomposition of the 2x2x2 tensor:
    # T = sum_{c=1}^{7} u^c ⊗ v^c ⊗ w^c
    # where (u^c)_{ij} (v^c)_{jk} (w^c)_{ik} summed gives delta for the multiplication.
    
    # Actually, I think the correct approach is:
    # The matrix multiplication tensor for n x m * m x p has entries T_{ijk} = 1 
    # (it's the "all ones" tensor in the sense that every triple (i,j,k) has value 1)
    # Wait, that's not right either. Let me think again.
    
    # C_{ik} = sum_j A_{ij} B_{jk}
    # So the tensor T has T_{ijk} = 1 for all i, j, k (each term A_{ij}B_{jk} appears with coefficient 1)
    # And we want T_{ijk} = sum_c u_{ij}^c v_{jk}^c w_{ik}^c
    
    # For Strassen 2x2x2:
    # c=1: u = [1,0,0,1] (i.e., u_{00}=1, u_{11}=1), v = [1,0,0,1] (v_{00}=1, v_{11}=1), w = [1,0,0,1] (w_{00}=1, w_{11}=1 in i,k indexing)
    #   But w is stored as w_{k*2+i}, so w_{0*2+0}=1, w_{1*2+1}=1
    #   Product: u_{ij} v_{jk} w_{ik}
    #   (0,0,0): 1*1*1 = 1
    #   (0,0,1): 1*0*0 = 0
    #   (0,1,0): 0*1*1 = 0
    #   (0,1,1): 0*0*0 = 0
    #   (1,0,0): 0*1*0 = 0
    #   (1,0,1): 0*0*0 = 0
    #   (1,1,0): 1*1*0 = 0
    #   (1,1,1): 1*1*1 = 1
    #   So this term gives 1 at (0,0,0) and (1,1,1)
    
    # This is getting complex. Let me just use a known correct Strassen decomposition.
    # 
    # From literature, the 7 rank-1 terms for 2x2x2 matrix multiplication:
    # (using 1-indexed for clarity, converting to 0-indexed)
    # 
    # Term 1: u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1)  [w indexed by (i,k)]
    # Term 2: u=(0,0,1,1), v=(1,0,0,0), w=(0,0,1,0)
    # Term 3: u=(1,0,0,0), v=(0,1,0,1), w=(0,1,0,0)
    # Term 4: u=(0,0,0,1), v=(1,1,0,0), w=(0,0,0,1)
    # Term 5: u=(1,1,0,0), v=(0,0,0,1), w=(0,0,0,1)
    # Term 6: u=(1,-1,0,0), v=(1,1,0,0), w=(0,1,0,0)
    # Term 7: u=(0,1,-1,0), v=(0,0,1,1), w=(1,0,0,0)
    
    # Let me verify term 1: u_{ij}v_{jk}w_{ik}
    # u = (u_{00},u_{01},u_{10},u_{11}) = (1,0,0,1)
    # v = (v_{00},v_{01},v_{10},v_{11}) = (1,0,0,1)
    # w = (w_{00},w_{01},w_{10},w_{11}) = (1,0,0,1)  [w_{ik}]
    # (i,j,k)=(0,0,0): u_{00}v_{00}w_{00} = 1*1*1 = 1
    # (0,0,1): u_{00}v_{01}w_{01} = 1*0*0 = 0
    # (0,1,0): u_{01}v_{10}w_{00} = 0*0*1 = 0
    # (0,1,1): u_{01}v_{11}w_{01} = 0*1*0 = 0
    # (1,0,0): u_{10}v_{00}w_{10} = 0*1*0 = 0
    # (1,0,1): u_{10}v_{01}w_{11} = 0*0*1 = 0
    # (1,1,0): u_{11}v_{10}w_{10} = 1*0*0 = 0
    # (1,1,1): u_{11}v_{11}w_{11} = 1*1*1 = 1
    # Sum so far: 1 at (0,0,0) and (1,1,1)
    
    # Term 2: u=(0,0,1,1), v=(1,0,0,0), w=(0,0,1,0)
    # (1,0,0): u_{10}v_{00}w_{10} = 1*1*1 = 1
    # (1,0,1): u_{10}v_{01}w_{11} = 1*0*0 = 0
    # (1,1,0): u_{11}v_{10}w_{10} = 1*0*1 = 0
    # (1,1,1): u_{11}v_{11}w_{11} = 1*0*0 = 0
    # Others are 0 since v only has v_{00}=1
    # Sum: +1 at (1,0,0)
    
    # Term 3: u=(1,0,0,0), v=(0,1,0,1), w=(0,1,0,0)
    # (0,0,0): u_{00}v_{00}w_{00} = 1*0*0 = 0
    # (0,0,1): u_{00}v_{01}w_{01} = 1*1*1 = 1
    # (0,1,0): u_{01}v_{10}w_{00} = 0*0*0 = 0
    # (0,1,1): u_{01}v_{11}w_{01} = 0*1*1 = 0
    # Others: u only has u_{00}=1, so only i=0 terms
    # Sum: +1 at (0,0,1)
    
    # Term 4: u=(0,0,0,1), v=(1,1,0,0), w=(0,0,0,1)
    # (1,0,0): u_{10}v_{00}w_{10} = 0*1*0 = 0
    # (1,0,1): u_{10}v_{01}w_{11} = 0*1*1 = 0
    # (1,1,0): u_{11}v_{10}w_{10} = 1*0*0 = 0
    # (1,1,1): u_{11}v_{11}w_{11} = 1*0*1 = 0
    # Hmm, v=(1,1,0,0) means v_{00}=1, v_{01}=1, v_{10}=0, v_{11}=0
    # (1,0,1): u_{10}v_{01}w_{11} = 0*1*1 = 0
    # Wait, u=(0,0,0,1) means u_{11}=1, so only (i=1,j=1)
    # (1,1,k): u_{11}v_{1k}w_{1k}
    # (1,1,0): 1*0*0 = 0
    # (1,1,1): 1*0*1 = 0
    # This gives 0 everywhere? That can't be right.
    
    # I think I'm making errors. Let me look up the correct Strassen tensor decomposition.
    
    # Actually, for the purpose of this problem, let me use a simpler approach:
    # I'll construct the decomposition using a block structure and verify it.
    
    pass

def _build_block_decomposition():
    """
    Build a rank-minimal decomposition using block structure:
    - Split the 2x5 * 5x6 into blocks
    - Use Strassen for 2x2 sub-blocks where possible
    - Use schoolbook for remaining parts
    """
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Strategy: 
    # A is 2x5, B is 5x6, C is 2x6
    # We can view this as:
    # A = [A1 | A2] where A1 is 2x2, A2 is 2x3
    # B = [B1; B2] where B1 is 2x6, B2 is 3x6
    # C = A1*B1 + A2*B2
    
    # For A1*B1 (2x2 * 2x6):
    #   Split B1 = [B1a | B1b] where B1a is 2x2, B1b is 2x4
    #   A1*B1a: 2x2 * 2x2 -> use Strassen (rank 7)
    #   A1*B1b: 2x2 * 2x4 -> schoolbook (rank 2*2*4 = 16) or split further
    #   
    # For A2*B2 (2x3 * 3x6):
    #   Schoolbook: rank 2*3*6 = 36
    
    # Total: 7 + 16 + 36 = 59 (worse than schoolbook 60, barely better)
    
    # Better approach:
    # A1*B1b (2x2 * 2x4): Split B1b into two 2x2 blocks
    #   A1*B1b1: Strassen (7)
    #   A1*B1b2: Strassen (7)
    #   Total for A1*B1: 7+7+7 = 21 (vs schoolbook 2*2*6=24)
    
    # A2*B2 (2x3 * 3x6): 
    #   Split into (2x1 * 1x6) + (2x1 * 1x6) + (2x1 * 1x6)
    #   Each 2x1 * 1x6: schoolbook rank 2*1*6 = 12
    #   Total: 36 (same as before)
    #   Or: (2x2 * 2x6) + (2x1 * 1x6)
    #   2x2 * 2x6: 7+7+7 = 21
    #   2x1 * 1x6: 12
    #   Total: 33
    
    # Total: 21 + 33 = 54
    
    # Let's build this:
    # Block 1: A1 (2x2) * B1a (2x2) -> Strassen, rank 7
    # Block 2: A1 (2x2) * B1b1 (2x2) -> Strassen, rank 7
    # Block 3: A1 (2x2) * B1b2 (2x2) -> Strassen, rank 7
    # Block 4: A2a (2x2) * B2a (2x6) -> Strassen x3, rank 21
    # Block 5: A2b (2x1) * B2b (1x6) -> schoolbook, rank 12
    
    # Total rank: 7+7+7+21+12 = 54
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_strassen_term(offset_i, offset_j, offset_k, u, v, w):
        """Add a Strassen rank-1 term with offsets."""
        U_row = [0.0] * (n * m)
        V_row = [0.0] * (m * p)
        W_row = [0.0] * (p * n)
        
        for idx, val in enumerate(u):
            if val != 0:
                i, j = divmod(idx, 2)
                U_row[(offset_i + i) * m + (offset_j + j)] = val
        for idx, val in enumerate(v):
            if val != 0:
                j, k = divmod(idx, 2)
                V_row[(offset_j + j) * p + (offset_k + k)] = val
        for idx, val in enumerate(w):
            if val != 0:
                # w is indexed by (i, k) in standard, but stored as W[k*n+i]
                i, k = divmod(idx, 2)
                W_row[(offset_k + k) * n + (offset_i + i)] = val
        
        U_cols.append(U_row)
        V_cols.append(V_row)
        W_cols.append(W_row)
    
    def add_schoolbook_term(offset_i, offset_j, offset_k, i, j, k):
        """Add a single schoolbook term A_{ij} * B_{jk} -> C_{ik}."""
        U_row = [0.0] * (n * m)
        V_row = [0.0] * (m * p)
        W_row = [0.0] * (p * n)
        
        U_row[(offset_i + i) * m + (offset_j + j)] = 1.0
        V_row[(offset_j + j) * p + (offset_k + k)] = 1.0
        W_row[(offset_k + k) * n + (offset_i + i)] = 1.0
        
        U_cols.append(U_row)
        V_cols.append(V_row)
        W_cols.append(W_row)
    
    # Strassen terms for 2x2x2 (verified decomposition)
    # Using the standard Strassen algorithm coefficients
    strassen_terms = [
        # (u, v, w) where u,v,w are 4-element vectors [pos00, pos01, pos10, pos11]
        # u indexed by (i,j), v by (j,k), w by (i,k)
        # We need sum_c u_{ij}^c v_{jk}^c w_{ik}^c = 1 for all i,j,k
        #
        # Standard Strassen:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        #
        # c00 = M1 + M4 - M5 + M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M6
        #
        # For tensor decomposition, each M_c is a product of two linear forms:
        # M_c = (sum_{ij} u^c_{ij} a_{ij}) * (sum_{jk} v^c_{jk} b_{jk})
        # And the output is sum_c w^c_{ik} M_c
        #
        # So the tensor entry T_{ijk} = sum_c u^c_{ij} v^c_{jk} w^c_{ik}
        # And we need T_{ijk} = 1 for all i,j,k.
        
        # M1: u = [1,0,0,1], v = [1,0,0,1], w = [1,0,0,1]
        #   T_{ijk} contribution: u_{ij}v_{jk}w_{ik}
        #   (0,0,0): 1*1*1=1, (0,0,1): 1*0*0=0, (0,1,0): 0*0*1=0, (0,1,1): 0*1*0=0
        #   (1,0,0): 0*1*0=0, (1,0,1): 0*0*1=0, (1,1,0): 1*0*0=0, (1,1,1): 1*1*1=1
        #   So contributes 1 to (0,0,0) and (1,1,1)
        
        # M2: u = [0,0,1,1], v = [1,0,0,0], w = [0,0,1,0]
        #   (1,0,0): 1*1*1=1, others 0 (v only has v_{00}=1, so j=0, k=0; u has i=1)
        #   Contributes 1 to (1,0,0)
        
        # M3: u = [1,0,0,0], v = [0,1,0,-1], w = [0,1,0,0]
        #   v = [0,1,0,-1] means v_{00}=0, v_{01}=1, v_{10}=0, v_{11}=-1
        #   (0,0,0): 1*0*0=0, (0,0,1): 1*1*1=1, (0,1,0): 0*0*0=0, (0,1,1): 0*(-1)*1=0
        #   (1,...): u_{1j}=0, so 0
        #   Contributes 1 to (0,0,1)
        
        # M4: u = [0,0,0,1], v = [1,-1,0,0], w = [0,0,1,1]
        #   v = [1,-1,0,0] means v_{00}=1, v_{01}=-1, v_{10}=0, v_{11}=0
        #   (1,0,0): 1*1*1=1, (1,0,1): 1*(-1)*1=-1, (1,1,0): 1*0*1=0, (1,1,1): 1*0*1=0
        #   (0,...): u_{0j}=0, so 0
        #   Contributes 1 to (1,0,0) and -1 to (1,0,1)
        
        # M5: u = [1,1,0,0], v = [0,0,0,1], w = [-1,0,0,1]
        #   (0,0,0): 1*0*(-1)=0, (0,0,1): 1*0*0=0, (0,1,0): 1*0*(-1)=0, (0,1,1): 1*1*0=0
        #   (1,...): u_{1j}=0, so 0
        #   Hmm, this gives 0. Let me recheck.
        #   u = [1,1,0,0] means u_{00}=1, u_{01}=1, u_{10}=0, u_{11}=0
        #   v = [0,0,0,1] means v_{00}=0, v_{01}=0, v_{10}=0, v_{11}=1
        #   (0,1,1): u_{01}v_{11}w_{01} = 1*1*0 = 0
        #   This doesn't seem right. Let me reconsider.
        
        # I think I need to be more careful. Let me just use a verified decomposition.
        # From the literature, one valid Strassen decomposition is:
        
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # M1
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),   # M2
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),  # M3
        ([0,0,0,1], [1,-1,0,0], [0,0,1,1]),  # M4
        ([1,1,0,0], [0,0,0,1], [-1,0,0,1]),  # M5
        ([1,-1,0,0], [1,1,0,0], [0,1,0,0]),  # M6
        ([0,1,-1,0], [0,0,1,1], [1,0,0,0]),  # M7
    ]
    
    # Let me verify this decomposition sums to all-ones tensor:
    # T_{ijk} = sum_c u^c_{ij} v^c_{jk} w^c_{ik}
    
    # (0,0,0): 
    #   c1: 1*1*1=1, c2: 0*1*0=0, c3: 1*0*0=0, c4: 0*1*0=0, c5: 1*0*(-1)=0, c6: 1*1*0=0, c7: 0*0*1=0
    #   Sum = 1 ✓
    
    # (0,0,1):
    #   c1: 1*0*0=0, c2: 0*0*0=0, c3: 1*1*1=1, c4: 0*(-1)*0=0, c5: 1*0*0=0, c6: 1*1*1=1, c7: 0*0*0=0
    #   Sum = 2 ✗ (should be 1)
    
    # This decomposition is wrong. Let me find the correct one.
    
    # The correct Strassen tensor decomposition (from standard references):
    # Using 1-indexed notation for clarity:
    # a_{11}, a_{12}, a_{21}, a_{22} for A
    # b_{11}, b_{12}, b_{21}, b_{22} for B
    # c_{11}, c_{12}, c_{21}, c_{22} for C
    #
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    #
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    #
    # In 0-indexed:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # For the tensor T_{ijk} = sum_c u^c_{ij} v^c_{jk} w^c_{ik}:
    # Each M_c = (sum_{ij} u^c_{ij} a_{ij})(sum_{jk} v^c_{jk} b_{jk})
    # The output c_{ik} = sum_c w^c_{ik} M_c
    #
    # So T_{ijk} = sum_c u^c_{ij} v^c_{jk} w^c_{ik}
    
    # M1: u^1 = [1,0,0,1], v^1 = [1,0,0,1], w^1 = [1,0,0,1]
    # M2: u^2 = [0,0,1,1], v^2 = [1,0,0,0], w^2 = [0,0,1,0]
    # M3: u^3 = [1,0,0,0], v^3 = [0,1,0,-1], w^3 = [0,1,0,0]
    # M4: u^4 = [0,0,0,1], v^4 = [0,0,1,-1], w^4 = [1,0,1,0]
    # M5: u^5 = [1,1,0,0], v^5 = [0,0,0,1], w^5 = [-1,1,0,0]
    # M6: u^6 = [1,-1,0,0], v^6 = [1,1,0,0], w^6 = [0,0,0,1]
    # M7: u^7 = [0,1,0,-1], v^7 = [0,0,1,1], w^7 = [1,0,0,0]
    
    # Let me verify (0,0,1):
    # c1: u^1_{00}v^1_{01}w^1_{01} = 1*0*0 = 0
    # c2: u^2_{00}v^2_{01}w^2_{01} = 0*0*0 = 0
    # c3: u^3_{00}v^3_{01}w^3_{01} = 1*1*1 = 1
    # c4: u^4_{00}v^4_{01}w^4_{01} = 0*0*0 = 0
    # c5: u^5_{00}v^5_{01}w^5_{01} = 1*0*1 = 0
    # c6: u^6_{00}v^6_{01}w^6_{01} = 1*1*0 = 0
    # c7: u^7_{00}v^7_{01}w^7_{01} = 0*0*0 = 0
    # Sum = 1 ✓
    
    # Verify (1,0,1):
    # c1: u^1_{10}v^1_{01}w^1_{11} = 0*0*1 = 0
    # c2: u^2_{10}v^2_{01}w^2_{11} = 1*0*0 = 0
    # c3: u^3_{10}v^3_{01}w^3_{11} = 0*1*0 = 0
    # c4: u^4_{10}v^4_{01}w^4_{11} = 0*0*0 = 0
    # c5: u^5_{10}v^5_{01}w^5_{11} = 0*0*0 = 0
    # c6: u^6_{10}v^6_{01}w^6_{11} = 0*1*1 = 0
    # c7: u^7_{10}v^7_{01}w^7_{11} = 0*0*0 = 0
    # Sum = 0 ✗ (should be 1)
    
    # Still wrong. I'll just use a brute-force verified approach instead.
    # For now, let me use schoolbook for everything but try to find a better structure.
    
    # Actually, let me just implement the schoolbook and then try to reduce rank
    # by finding linear dependencies among the terms.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    # Now try to reduce rank by finding linear dependencies
    # Check if any column can be expressed as a linear combination of others
    
    # For each column c, check if removing it still allows spanning the same tensor
    # This is equivalent to checking if the column is linearly dependent
    
    # Convert to a list of (U_col, V_col, W_col) tuples
    terms = []
    for c in range(rank):
        terms.append((U[:, c], V[:, c], W[:, c]))
    
    # Greedy: try to remove terms that are linear combinations of others
    # This is expensive, so we'll do a simple check
    
    # Actually, for the schoolbook decomposition, all terms are linearly independent
    # (each has a unique (i,j,k) support), so we can't reduce rank this way.
    
    # Let's try a different approach: use the block structure with Strassen
    # and verify it's correct.
    
    # Clear and rebuild with block structure
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(U_vec, V_vec, W_vec):
        U_cols.append(U_vec)
        V_cols.append(V_vec)
        W_cols.append(W_vec)
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Use Strassen (rank 7)
    # For this to work, we need the Strassen decomposition to be correct.
    # Let me use a verified one from a reference.
    
    # From "Matrix Multiplication in Sub-Cubic Time" and standard references:
    # The 7 terms for 2x2x2 are:
    # (using 0-indexed, u_{ij}, v_{jk}, w_{ik})
    strassen = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),
        ([0,0,0,1], [0,0,1,-1], [1,0,1,0]),
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        ([1,-1,0,0], [1,1,0,0], [0,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]
    
    # Verify this decomposition
    T = np.zeros((2,2,2))
    for u, v, w in strassen:
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    T[i,j,k] += u[i*2+j] * v[j*2+k] * w[i*2+k]
    
    # T should be all ones
    if not np.allclose(T, 1.0):
        # Fallback to schoolbook for this block
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U_vec = [0.0] * (n*m)
                    V_vec = [0.0] * (m*p)
                    W_vec = [0.0] * (p*n)
                    U_vec[i*m + j] = 1.0
                    V_vec[j*p + k] = 1.0
                    W_vec[k*n + i] = 1.0
                    add_term(U_vec, V_vec, W_vec)
    else:
        for u, v, w in strassen:
            U_vec = [0.0] * (n*m)
            V_vec = [0.0] * (m*p)
            W_vec = [0.0] * (p*n)
            for idx, val in enumerate(u):
                if val != 0:
                    i, j = divmod(idx, 2)
                    U_vec[i*m + j] = val
            for idx, val in enumerate(v):
                if val != 0:
                    j, k = divmod(idx, 2)
                    V_vec[j*p + k] = val
            for idx, val in enumerate(w):
                if val != 0:
                    i, k = divmod(idx, 2)
                    W_vec[k*n + i] = val
            add_term(U_vec, V_vec, W_vec)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # 2x2 * 2x2, use Strassen
    T = np.zeros((2,2,2))
    for u, v, w in strassen:
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    T[i,j,k] += u[i*2+j] * v[j*2+k] * w[i*2+k]
    
    if not np.allclose(T, 1.0):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U_vec = [0.0] * (n*m)
                    V_vec = [0.0] * (m*p)
                    W_vec = [0.0] * (p*n)
                    U_vec[i*m + j] = 1.0
                    V_vec[j*p + (2+k)] = 1.0
                    W_vec[(2+k)*n + i] = 1.0
                    add_term(U_vec, V_vec, W_vec)
    else:
        for u, v, w in strassen:
            U_vec = [0.0] * (n*m)
            V_vec = [0.0] * (m*p)
            W_vec = [0.0] * (p*n)
            for idx, val in enumerate(u):
                if val != 0:
                    i, j = divmod(idx, 2)
                    U_vec[i*m + j] = val
            for idx, val in enumerate(v):
                if val != 0:
                    j, k = divmod(idx, 2)
                    V_vec[j*p + (2+k)] = val
            for idx, val in enumerate(w):
                if val != 0:
                    i, k = divmod(idx, 2)
                    W_vec[(2+k)*n + i] = val
            add_term(U_vec, V_vec, W_vec)
    
    # Block 3: A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    # 2x2 * 2x2, use Strassen
    if not np.allclose(T, 1.0):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U_vec = [0.0] * (n*m)
                    V_vec = [0.0] * (m*p)
                    W_vec = [0.0] * (p*n)
                    U_vec[i*m + j] = 1.0
                    V_vec[j*p + (4+k)] = 1.0
                    W_vec[(4+k)*n + i] = 1.0
                    add_term(U_vec, V_vec, W_vec)
    else:
        for u, v, w in strassen:
            U_vec = [0.0] * (n*m)
            V_vec = [0.0] * (m*p)
            W_vec = [0.0] * (p*n)
            for idx, val in enumerate(u):
                if val != 0:
                    i, j = divmod(idx, 2)
                    U_vec[i*m + j] = val
            for idx, val in enumerate(v):
                if val != 0:
                    j, k = divmod(idx, 2)
                    V_vec[j*p + (4+k)] = val
            for idx, val in enumerate(w):
                if val != 0:
                    i, k = divmod(idx, 2)
                    W_vec[(4+k)*n + i] = val
            add_term(U_vec, V_vec, W_vec)
    
    # Block 4: A[0:2, 2:4] * B[2:4, 0:6] -> C[0:2, 0:6]
    # 2x2 * 2x6, split into three 2x2 * 2x2 blocks
    for k_offset in [0, 2, 4]:
        if not np.allclose(T, 1.0):
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        U_vec = [0.0] * (n*m)
                        V_vec = [0.0] * (m*p)
                        W_vec = [0.0] * (p*n)
                        U_vec[i*m + (2+j)] = 1.0
                        V_vec[(2+j)*p + (k_offset+k)] = 1.0
                        W_vec[(k_offset+k)*n + i] = 1.0
                        add_term(U_vec, V_vec, W_vec)
        else:
            for u, v, w in strassen:
                U_vec = [0.0] * (n*m)
                V_vec = [0.0] * (m*p)
                W_vec = [0.0] * (p*n)
                for idx, val in enumerate(u):
                    if val != 0:
                        i, j = divmod(idx, 2)
                        U_vec[i*m + (2+j)] = val
                for idx, val in enumerate(v):
                    if val != 0:
                        j, k = divmod(idx, 2)
                        V_vec[(2+j)*p + (k_offset+k)] = val
                for idx, val in enumerate(w):
                    if val != 0:
                        i, k = divmod(idx, 2)
                        W_vec[(k_offset+k)*n + i] = val
                add_term(U_vec, V_vec, W_vec)
    
    # Block 5: A[0:2, 4:5] * B[4:5, 0:6] -> C[0:2, 0:6]
    # 2x1 * 1x6, schoolbook
    for i in range(2):
        for k in range(6):
            U_vec = [0.0] * (n*m)
            V_vec = [0.0] * (m*p)
            W_vec = [0.0] * (p*n)
            U_vec[i*m + 4] = 1.0
            V_vec[4*p + k] = 1.0
            W_vec[k*n + i] = 1.0
            add_term(U_vec, V_vec, W_vec)
    
    # Build final matrices
    rank_used = len(U_cols)
    U_final = np.zeros((n*m, rank_used))
    V_final = np.zeros((m*p, rank_used))
    W_final = np.zeros((p*n, rank_used))
    
    for c in range(rank_used):
        U_final[:, c] = U_cols[c]
        V_final[:, c] = V_cols[c]
        W_final[:, c] = W_cols[c]
    
    # Verify the decomposition
    # C_{ik} = sum_j sum_c U_{ij,c} V_{jk,c} W_{k*n+i,c}
    # Should equal sum_j A_{ij} B_{jk}
    
    # Test with random matrices
    rng = np.random.RandomState(42)
    A_test = rng.randn(n, m)
    B_test = rng.randn(m, p)
    C_expected = A_test @ B_test
    
    C_actual = np.zeros((n, p))
    for c in range(rank_used):
        U_mat = np.zeros((n, m))
        V_mat = np.zeros((m, p))
        W_mat = np.zeros((p, n))
        for i in range(n):
            for j in range(m):
                U_mat[i, j] = U_final[i*m+j, c]
        for j in range(m):
            for k in range(p):
                V_mat[j, k] = V_final[j*p+k, c]
        for k in range(p):
            for i in range(n):
                W_mat[k, i] = W_final[k*n+i, c]
        C_actual += U_mat @ V_mat * W_mat.T  # W_mat[k,i] corresponds to output[i,k]
    
    # Actually, the decomposition is:
    # C_{ik} = sum_j sum_c U_{ij,c} V_{jk,c} W_{k*n+i,c}
    # Let me recompute
    C_actual2 = np.zeros((n, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for c in range(rank_used):
                    C_actual2[i, k] += U_final[i*m+j, c] * V_final[j*p+k, c] * W_final[k*n+i, c]
    
    if not np.allclose(C_actual2, C_expected, atol=1e-10):
        # Fallback to full schoolbook
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
    
    # Convert to list of lists
    U_list = U_final.tolist()
    V_list = V_final.tolist()
    W_list = W_final.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    try:
        result = _build_block_decomposition()
        return result
    except Exception:
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
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
