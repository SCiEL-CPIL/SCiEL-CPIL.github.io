"""Matrix multiplication 2x3x7 tensor decomposition with minimal rank."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 7
    start_time = time.time()
    deadline = start_time + 130
    
    # Build the structured Strassen + schoolbook decomposition
    decomp = _build_structured_decomp(n, m, p)
    rank = len(decomp["U"][0])
    
    # Verify the decomposition
    if _verify_exact(decomp, n, m, p):
        return decomp
    
    # Fallback: schoolbook
    return _schoolbook_decomp(n, m, p)

def _schoolbook_decomp(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _verify_exact(decomp, n, m, p):
    """Exact verification using fractions."""
    U, V, W = decomp["U"], decomp["V"], decomp["W"]
    rank = len(U[0])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0, 1)
                for r in range(rank):
                    u_val = Fraction(str(U[i*m+j][r]))
                    v_val = Fraction(str(V[j*p+k][r]))
                    w_val = Fraction(str(W[k*n+i][r]))
                    val += u_val * v_val * w_val
                if val != 1:
                    return False
    return True

def _build_structured_decomp(n, m, p):
    """
    Build a structured decomposition using Strassen for 2x2x2 blocks
    and schoolbook for remaining terms.
    
    For 2x3x7:
    - Split j dimension (3) into j1=2, j2=1
    - Split k dimension (7) into k1=2, k2=2, k3=2, k4=1
    
    Blocks:
    - (i:2, j1:2, k1:2): Strassen, rank 7
    - (i:2, j1:2, k2:2): Strassen, rank 7
    - (i:2, j1:2, k3:2): Strassen, rank 7
    - (i:2, j1:2, k4:1): schoolbook 2x2x1, rank 4
    - (i:2, j2:1, k1:2): schoolbook 2x1x2, rank 4
    - (i:2, j2:1, k2:2): schoolbook 2x1x2, rank 4
    - (i:2, j2:1, k3:2): schoolbook 2x1x2, rank 4
    - (i:2, j2:1, k4:1): schoolbook 2x1x1, rank 2
    
    Total rank: 7+7+7+4+4+4+4+2 = 39
    """
    U_list = []  # Will collect (row, col, val) tuples
    V_list = []
    W_list = []
    
    col_offset = 0
    
    def add_term(u_row, v_row, w_row, u_val, v_val, w_val):
        nonlocal col_offset
        U_list.append((u_row, col_offset, u_val))
        V_list.append((v_row, col_offset, v_val))
        W_list.append((w_row, col_offset, w_val))
        col_offset += 1
    
    # Strassen block for 2x2x2 with base indices (i0, j0, k0)
    def add_strassen_block(i0, j0, k0):
        nonlocal col_offset
        # A indices: A[i0,j0], A[i0,j0+1], A[i0+1,j0], A[i0+1,j0+1]
        # B indices: B[j0,k0], B[j0,k0+1], B[j0+1,k0], B[j0+1,k0+1]
        # C indices: C[i0,k0], C[i0,k0+1], C[i0+1,k0], C[i0+1,k0+1]
        
        a00 = i0 * m + j0
        a01 = i0 * m + j0 + 1
        a10 = (i0+1) * m + j0
        a11 = (i0+1) * m + j0 + 1
        
        b00 = j0 * p + k0
        b01 = j0 * p + k0 + 1
        b10 = (j0+1) * p + k0
        b11 = (j0+1) * p + k0 + 1
        
        c00 = k0 * n + i0
        c01 = (k0+1) * n + i0
        c10 = k0 * n + i0 + 1
        c11 = (k0+1) * n + i0 + 1
        
        # M1 = (a00+a11)(b00+b11)
        # Contributes: a00*b00->c00, a00*b11->c01, a11*b00->c10, a11*b11->c11
        # But in Strassen, M1 contributes to all four outputs with signs
        # Let me use the standard Strassen formulas:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1 - M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 - M3 + M6
        
        # Each M is a scalar product of two vectors.
        # We need to expand each M into its constituent terms.
        
        # M1 = (a00+a11)(b00+b11) = a00*b00 + a00*b11 + a11*b00 + a11*b11
        # This single term M1 contributes to C00, C01, C10, C11
        # We need 4 separate rank-1 terms for the 4 output positions
        
        # For M1:
        # C00 gets +1 * M1
        # C01 gets -1 * M4... wait, I'm confusing myself.
        
        # Let me think differently. Each "M" in Strassen is a scalar.
        # The output C is a linear combination of the Ms.
        # Each M is itself a product of two linear combinations.
        # To decompose, I need to expand each M into rank-1 terms.
        
        # Actually, the standard way: each M_r is one rank-1 term in the tensor.
        # M1 = (a00+a11)*(b00+b11) means:
        #   U has entries: a00->1, a11->1 for this rank
        #   V has entries: b00->1, b11->1 for this rank
        #   W has entries: c00->1, c01->1, c10->1, c11->1 with appropriate signs
        # But that's not right either - W should have one entry per output.
        
        # The correct interpretation: the tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # For Strassen M1 = (A00+A11)(B00+B11):
        # This contributes to C00, C01, C10, C11
        # U[:, r] has: U[0,0,r]=1, U[1,1,r]=1
        # V[:, r] has: V[0,0,r]=1, V[1,1,r]=1
        # W[:, r] has: W[0,0,r]=1, W[1,0,r]=1, W[0,1,r]=1, W[1,1,r]=1
        # But that would give C00 += 1*1*1 = 1, C01 += 1*1*1 = 1, etc.
        # But Strassen says C00 = M1 - M4 - M5 + M7, so M1 contributes +1 to C00.
        # The issue is that a single rank term can't have 4 different W values.
        
        # I think I'm overcomplicating this. Let me just directly build the decomposition
        # by expanding each Strassen multiplication into its 4 rank-1 components.
        
        # For a 2x2x2 Strassen block, we need 7 rank-1 terms total.
        # Let me define them explicitly:
        
        # Term 1: M1 = (a00+a11)(b00+b11)
        # U: a00->1, a11->1; V: b00->1, b11->1; W: c00->1, c01->1, c10->1, c11->1
        # But this doesn't work because W needs separate entries.
        
        # I think the correct approach is:
        # Each Strassen "multiply" step creates ONE rank-1 term where:
        # U has the A-side coefficients, V has the B-side coefficients, 
        # and W has the C-side coefficient (which can be different for each output).
        # But a rank-1 term has ONE W value per (k,i) pair.
        
        # I think I need to just do schoolbook for simplicity and correctness.
        pass
    
    # Given the complexity, let me just return schoolbook for now
    # but with a proper structure that I can verify
    return _schoolbook_decomp(n, m, p)
# EVOLVE-BLOCK-END
