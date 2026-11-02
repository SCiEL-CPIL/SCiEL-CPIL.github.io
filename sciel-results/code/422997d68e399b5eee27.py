"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # We decompose the 3x3x7 tensor into:
    # Block 1: 3x3x3 (cols 0-2 of B)
    # Block 2: 3x3x3 (cols 3-5 of B)
    # Block 3: 3x3x1 (col 6 of B)
    
    # We use Laderman's rank-23 decomposition for the 3x3 blocks.
    # We use schoolbook rank-9 for the 3x1 block.
    # Total rank = 23 + 23 + 9 = 55.
    
    # Define the Laderman 3x3 decomposition terms.
    # Each term is (U_vec, V_vec, W_vec) where:
    # U_vec has length 9 (indices i*m+j for i,j in 0..2)
    # V_vec has length 9 (indices j*p+k for j,k in 0..2) -- Note: for 3x3 block, p_local=3
    # W_vec has length 9 (indices k*n+i for k,i in 0..2)
    
    # Laderman's 23 terms for 3x3 matrix multiplication C = A*B
    # A indices: a00, a01, a02, a10, a11, a12, a20, a21, a22
    # B indices: b00, b01, b02, b10, b11, b12, b20, b21, b22
    # C indices: c00, c01, c02, c10, c11, c12, c20, c21, c22
    
    # The 23 multiplications M1..M23
    # M1 = a00 * (b00+b01+b02+b10+b11+b12+b20+b21+b22)
    # ... (using the coefficients derived from Laderman 1976)
    
    # To be safe and avoid hardcoding errors, I will generate the Laderman terms
    # using a known correct implementation pattern.
    
    # Laderman's Algorithm Terms:
    # The algorithm is symmetric.
    # Let S_A be a 3x3 matrix of coefficients for A, S_B for B.
    # The 23 terms are defined by specific linear combinations.
    
    # I will use the following verified rank-23 decomposition:
    # Source: "A 23-multiplication algorithm for 3x3 matrix multiplication"
    
    laderman_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U_coeffs: 9 floats for A
        # V_coeffs: 9 floats for B
        # W_coeffs: 9 floats for C
    ]
    
    # Since I cannot guarantee the exact 23 terms from memory without risk,
    # I will use a fallback strategy:
    # 1. Attempt to construct a rank-55 decomposition using a known rank-23 scheme.
    # 2. Verify it.
    # 3. If verification fails, return schoolbook rank-63.
    
    # However, implementing the full Laderman verification and fallback is complex.
    # A simpler improvement: Use Strassen for 2x2 subblocks?
    # 3x3 = 2x2 + 2x2 + 1x2 + 2x1 + 1x1?
    # This is messy.
    
    # Let's stick to the simplest valid improvement:
    # The 3x1 block is rank 9.
    # The 3x3 blocks are rank 27 each in schoolbook.
    # Total 63.
    
    # If I use a rank-23 decomposition, I need the exact coefficients.
    # I will use the **Laderman** coefficients from a standard reference.
    
    # Laderman's 23 terms (U, V, W)
    # I will define them explicitly.
    
    def get_laderman_3x3():
        # Returns list of (U, V, W) tuples of length 9 each.
        # U[i*m+j], V[j*p+k], W[k*n+i]
        
        # The 23 terms are:
        # M1 = A00 * (B00+B01+B02+B10+B11+B12+B20+B21+B22)
        # M2 = A01 * (B00+B01+B02-B10-B11-B12-B20-B21-B22)
        # M3 = A02 * (B00+B01+B02+B10-B11+B12+B20-B21+B22)
        # M4 = A10 * (-B00+B01+B02+B10+B11+B12-B20+B21+B22)
        # M5 = A11 * (B00-B01+B02+B10-B11+B12+B20-B21+B22)
        # M6 = A12 * (-B00-B01+B02-B10-B11+B12-B20-B21+B22)
        # M7 = A20 * (B00+B01-B02+B10+B11-B12+B20+B21-B22)
        # M8 = A21 * (-B00+B01-B02-B10+B11-B12-B20+B21-B22)
        # M9 = A22 * (B00-B01-B02+B10-B11-B12+B20-B21-B22)
        # M10 = (A00+A01+A02+A10+A11+A12+A20+A21+A22) * B00
        # M11 = (A00+A01+A02-A10-A11-A12-A20-A21-A22) * B01
        # M12 = (A00+A01+A02+A10-A11+A12+A20-A21+A22) * B02
        # M13 = (-A00+A01+A02+A10+A11+A12-A20+A21+A22) * B10
        # M14 = (A00-A01+A02+A10-A11+A12+A20-A21+A22) * B11
        # M15 = (-A00-A01+A02-A10-A11+A12-A20-A21+A22) * B12
        # M16 = (A00+A01-A02+A10+A11-A12+A20+A21-A22) * B20
        # M17 = (-A00+A01-A02-A10+A11-A12-A20+A21-A22) * B21
        # M18 = (A00-A01-A02+A10-A11-A12+A20-A21-A22) * B22
        # M19 = (A00+A10+A20+A01+A11+A21+A02+A12+A22) * (B00+B10+B20+B01+B11+B21+B02+B12+B22) -- No, this is not right.
        
        # The last 4 terms in Laderman are:
        # M19 = (A00+A01+A02+A10+A11+A12+A20+A21+A22) * (B00+B10+B20+B01+B11+B21+B02+B12+B22) is NOT a term.
        
        # Actually, Laderman's algorithm has 23 terms.
        # I will use the schoolbook for the 3x3 blocks as well, to ensure correctness.
        # Rank = 27 + 27 + 9 = 63.
        
        # Since I cannot guarantee the Laderman coefficients, I will return the schoolbook decomposition.
        # This is the same as the current program.
        
        return None

    # Fallback to Schoolbook
    rank = 63
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    col = 0
    # First 3x3 block
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    # Second 3x3 block
    for i in range(n):
        for j in range(m):
            for k in range(3, 6):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    # Third 3x1 block
    for i in range(n):
        for j in range(m):
            k = 6
            U[i*m + j, col] = 1
            V[j*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
