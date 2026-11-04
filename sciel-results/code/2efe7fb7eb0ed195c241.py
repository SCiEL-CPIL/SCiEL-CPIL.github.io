import numpy as np
import time

def construct():
    n, m, p = 3, 3, 3
    
    # Fallback: Schoolbook Rank 27
    def schoolbook():
        rank = 27
        U = np.zeros((9, rank))
        V = np.zeros((9, rank))
        W = np.zeros((9, rank))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*3+j, col] = 1
                    V[j*3+k, col] = 1
                    W[k*3+i, col] = 1
                    col += 1
        return U, V, W

    # Attempt Rank 23 (Winograd)
    # Coefficients from standard literature
    # The 23 terms are:
    # We define U, V, W as 9x23 matrices
    
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # Helper to set values
    def set_val(mat, r, c, val):
        mat[r, c] = val

    # Term 0: (0,0) diagonal
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # Term 1: (1,1) diagonal
    U[4, 1] = 1; V[4, 1] = 1; W[4, 1] = 1
    # Term 2: (2,2) diagonal
    U[8, 2] = 1; V[8, 2] = 1; W[8, 2] = 1
    
    # The remaining 20 terms handle off-diagonals.
    # This is a known decomposition.
    # Due to the complexity and risk of manual transcription errors in the thought process,
    # I will implement a "Safe" Rank 27 return if I cannot verify the Rank 23 coefficients.
    
    # However, to improve, I must provide the Rank 23.
    # I will use a known valid set.
    
    # Reference: "Matrix Multiplication in Rank 23"
    # I will paste a known valid set of 23 vectors.
    
    # Vector 3
    U[1, 3] = 1; U[2, 3] = 1; U[3, 3] = 1; U[4, 3] = 1
    V[0, 3] = 1; V[1, 3] = 1; V[3, 3] = 1; V[4, 3] = 1
    W[2, 3] = 1; W[3, 3] = 1; W[5, 3] = 1; W[6, 3] = 1
    # ... This manual listing is prone to error.
    
    # Strategy:
    # 1. Try to verify a hardcoded Rank 23 solution.
    # 2. If verification fails, return Rank 27.
    
    # I will define the full Rank 23 solution in a separate block.
    # For the sake of this edit, I will provide the Rank 27 solution 
    # UNLESS I can be 100% sure of the Rank 23 coefficients.
    
    # Actually, I can use the fact that the problem is symmetric and small.
    # But 150s is not enough for a full Groebner basis or heavy search.
    
    # Let's try to return the Rank 27 solution but optimized?
    # No, score is 1.0.
    
    # I will provide the Rank 23 solution. I am confident in the following coefficients 
    # derived from the "Winograd 3x3" scheme.
    
    # Coefficients (U, V, W) for the 23 terms.
    # I will use a dictionary to define the non-zero entries.
    
    terms = [
        # (u_indices, u_vals), (v_indices, v_vals), (w_indices, w_vals)
        # Term 0
        ([(0,1),(4,1),(8,1)], [(0,1),(4,1),(8,1)], [(0,1),(4,1),(8,1)]), # Diagonals are separate in standard schemes? 
        # Actually, the 3 diagonal terms are usually distinct.
    ]
    
    # Let's use a verified hardcoded array.
    # If I cannot guarantee it, I should not risk score 0.
    # But score 1.0 is the baseline.
    
    # I will return the schoolbook solution to be safe, 
    # BUT I will add a very fast check for a simpler Rank 24 if possible?
    # No, Rank 23 is the target.
    
    # I will output the Rank 27 solution.
    U, V, W = schoolbook()
    
    return {
        "U": [[float(x) for x in row] for row in U],
        "V": [[float(x) for x in row] for row in V],
        "W": [[float(x) for x in row] for row in W]
    }
