"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will build the U, V, W matrices for the decomposition.
    # The decomposition is based on Strassen's algorithm for 2x2x2 blocks.
    
    # Block indices for A (3x4)
    # A00: (0,0)-(1,1), A01: (0,2)-(1,3)
    # A10: (2,0)-(2,1), A11: (2,2)-(2,3)
    
    # Block indices for B (4x5)
    # B00: (0,0)-(1,1), B01: (0,2)-(1,3), B02: (0,4)-(1,4)
    # B10: (2,0)-(3,1), B11: (2,2)-(3,3), B12: (2,4)-(3,4)
    
    # We will create a list of tuples (u_indices, v_indices, w_indices, coeff)
    # for each rank-1 component.
    
    components = []
    
    # Helper to add a rank-1 component
    def add_component(u_idx, v_idx, w_idx, coeff):
        components.append((u_idx, v_idx, w_idx, coeff))
        
    # Strassen decomposition for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B11-B00)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B10
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We need to map the local 2x2 indices to global indices.
    # Local A: [[0,0],[0,1],[1,0],[1,1]]
    # Local B: [[0,0],[0,1],[1,0],[1,1]]
    # Local C: [[0,0],[0,1],[1,0],[1,1]]
    
    # For each block of C, we call Strassen.
    
    # C00 (2x2): A00*B00 + A01*B10
    # Let's compute A00*B00 and A01*B10 separately using Strassen.
    # This is not optimal, as Strassen is for one multiplication.
    # But we can't combine them easily because the addends are different.
    # So we use 2 Strassen decompositions for C00.
    
    # Strassen for A00*B00
    # A00 is at global (0:2, 0:2)
    # B00 is at global (0:2, 0:2)
    # C00 is at global (0:2, 0:2)
    
    def strassen_block(a_r1, a_c1, b_r1, b_c1, c_r1, c_c1):
        # a_r1, a_c1: top-left of A block
        # b_r1, b_c1: top-left of B block
        # c_r1, c_c1: top-left of C block
        
        # Local indices
        # A00: (0,0), A01: (0,1), A10: (1,0), A11: (1,1)
        # B00: (0,0), B01: (0,1), B10: (1,0), B11: (1,1)
        
        # M1 = (A00+A11)(B00+B11)
        # This requires adding A00 and A11.
        # In the tensor decomposition, we can't "add" matrices.
        # We must expand (A00+A11) into A00 + A11.
        # So M1 = A00*B00 + A00*B11 + A11*B00 + A11*B11.
        # This is 4 rank-1 terms? No, Strassen uses 7 multiplications of sums.
        # But in tensor decomposition, a multiplication of a sum is not a single rank-1 term.
        # The tensor decomposition is a sum of rank-1 tensors.
        # Strassen's algorithm reduces the number of scalar multiplications.
        # In tensor decomposition, each scalar multiplication corresponds to a rank-1 tensor.
        # So we need to express the 7 Strassen products as sums of rank-1 tensors.
        
        # M1 = (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
        # This is 4 rank-1 tensors? No, A00*B00 is a 2x2 matrix, which is a sum of 4 rank-1 tensors (schoolbook).
        # So Strassen doesn't directly give a tensor decomposition.
        # It gives a circuit.
        # To get a tensor decomposition, we need to expand the circuit into rank-1 tensors.
        # The rank of the tensor is the number of rank-1 tensors.
        # Strassen's circuit has 7 multiplications.
        # Each multiplication is of 2x2 matrices? No, of sums of entries.
        # In Strassen, the inputs are 2x2 matrices.
        # The 7 products are of 2x2 matrices? No, of linear combinations of entries.
        # Actually, Strassen's algorithm computes the product of two 2x2 matrices using 7 multiplications of linear combinations of the entries.
        # Each such multiplication is a scalar multiplication.
        # So each of the 7 terms is a rank-1 tensor?
        # No, a scalar multiplication $x \cdot y$ is a rank-1 tensor if $x$ and $y$ are linear forms in the input variables.
        # Yes, $x$ is a linear combination of $A_{ij}$ and $y$ is a linear combination of $B_{ij}$.
        # So each of the 7 terms in Strassen is a rank-1 tensor!
        # So the rank of the 2x2x2 tensor is 7.
        
        # So for each 2x2x2 block, we add 7 rank-1 tensors.
        
        # We need to define the linear forms for A, B, and C.
        
        # Let A be [[a00, a01], [a10, a11]]
        # Let B be [[b00, b01], [b10, b11]]
        
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b11-b00)
        # M4 = a11(b01-b10)
        # M5 = (a00+a01)b10
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        # C00 = M1 - M4 + M7
        # C01 = M3 + M4
        # C10 = M2 + M6
        # C11 = M1 - M3 + M5
        
        # We need to map a00, a01, etc. to global indices.
        
        a00_idx = (a_r1, a_c1)
        a01_idx = (a_r1, a_c1+1)
        a10_idx = (a_r1+1, a_c1)
        a11_idx = (a_r1+1, a_c1+1)
        
        b00_idx = (b_r1, b_c1)
        b01_idx = (b_r1, b_c1+1)
        b10_idx = (b_r1+1, b_c1)
        b11_idx = (b_r1+1, b_c1+1)
        
        c00_idx = (c_r1, c_c1)
        c01_idx = (c_r1, c_c1+1)
        c10_idx = (c_r1+1, c_c1)
        c11_idx = (c_r1+1, c_c1+1)
        
        # M1
        add_component([(a00_idx, 1), (a11_idx, 1)], [(b00_idx, 1), (b11_idx, 1)], [(c00_idx, 1), (c11_idx, 1)], 1)
        # Wait, the rank-1 tensor is $u \otimes v \otimes w$.
        # $u$ is a vector in the A-space, $v$ in B-space, $w$ in C-space.
        # The term is $u^T A v^T B w^T C$.
        # For M1: $u = e_{a00} + e_{a11}$, $v = e_{b00} + e_{b11}$, $w = e_{c00} + e_{c11}$.
        # This is one rank-1 tensor.
        
        # M2
        add_component([(a10_idx, 1), (a11_idx, 1)], [(b00_idx, 1)], [(c10_idx, 1)], 1)
        
        # M3
        add_component([(a00_idx, 1)], [(b11_idx, 1), (b00_idx, -1)], [(c01_idx, 1)], 1)
        
        # M4
        add_component([(a11_idx, 1)], [(b01_idx, 1), (b10_idx, -1)], [(c00_idx, -1), (c11_idx, 1)], 1)
        
        # M5
        add_component([(a00_idx, 1), (a01_idx, 1)], [(b10_idx, 1)], [(c11_idx, 1)], 1)
        
        # M6
        add_component([(a10_idx, 1), (a00_idx, -1)], [(b00_idx, 1), (b01_idx, 1)], [(c10_idx, 1)], 1)
        
        # M7
        add_component([(a01_idx, 1), (a11_idx, -1)], [(b10_idx, 1), (b11_idx, 1)], [(c00_idx, 1)], 1)
        
        # Now we need to combine these into the final C.
        # C00 = M1 - M4 + M7
        # C01 = M3 + M4
        # C10 = M2 + M6
        # C11 = M1 - M3 + M5
        
        # The components added above are for the intermediate M's.
        # We need to adjust the W coefficients to get the final C.
        # This is tricky because the W vector for each M is different.
        # We should have added the components directly to C.
        
        # Let's restart the Strassen block function to add directly to C.
        
    # This approach is getting complicated.
    # Let's just use the schoolbook method for the 2x2x2 blocks but with rank 7?
    # No, rank 7 is the minimum.
    
    # I will implement the 56-rank decomposition using schoolbook for the 2x2x2 blocks?
    # No, that would be 4 * 8 = 32 for the 2x2x2 blocks.
    # 32 + 8 + 16 + 4 = 60.
    # So using Strassen (rank 7) saves 4 ranks.
    # 28 + 8 + 16 + 4 = 56.
    
    # I will implement the Strassen decomposition for the 2x2x2 blocks.
    
    return {"U": [], "V": [], "W": []}
# EVOLVE-BLOCK-END
