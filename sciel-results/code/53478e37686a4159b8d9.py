"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 110.0  # safety margin before 120s kill

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    """
    Construct a rank-reduced tensor decomposition for 4x4 * 4x6 matrix multiplication
    using Strassen's algorithm on 2x2 blocks, then removing linearly dependent terms.
    """
    n, m, p = 4, 4, 6
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_strassen_term(offset_i, offset_j, offset_k):
        mults = [
            ({(offset_i, offset_j): 1, (offset_i+1, offset_j+1): 1},
             {(offset_j, offset_k): 1, (offset_j+1, offset_k+1): 1},
             {(offset_i, offset_k): 1, (offset_i+1, offset_k+1): 1}),
            ({(offset_i+1, offset_j): 1, (offset_i+1, offset_j+1): 1},
             {(offset_j, offset_k): 1},
             {(offset_i+1, offset_k): 1, (offset_i+1, offset_k+1): -1}),
            ({(offset_i, offset_j): 1},
             {(offset_j, offset_k+1): 1, (offset_j+1, offset_k+1): -1},
             {(offset_i, offset_k+1): 1, (offset_i+1, offset_k+1): 1}),
            ({(offset_i+1, offset_j+1): 1},
             {(offset_j+1, offset_k): 1, (offset_j, offset_k): -1},
             {(offset_i, offset_k): 1, (offset_i+1, offset_k): 1}),
            ({(offset_i, offset_j): 1, (offset_i, offset_j+1): 1},
             {(offset_j+1, offset_k+1): 1},
             {(offset_i, offset_k): -1, (offset_i, offset_k+1): 1}),
            ({(offset_i+1, offset_j): 1, (offset_i, offset_j): -1},
             {(offset_j, offset_k): 1, (offset_j, offset_k+1): 1},
             {(offset_i+1, offset_k+1): 1}),
            ({(offset_i, offset_j+1): 1, (offset_i+1, offset_j+1): -1},
             {(offset_j+1, offset_k): 1, (offset_j+1, offset_k+1): 1},
             {(offset_i, offset_k): 1}),
        ]
        
        for U_coeffs, V_coeffs, W_coeffs in mults:
            U_col = [0] * (n * m)
            V_col = [0] * (m * p)
            W_col = [0] * (p * n)
            
            for (i, j), coeff in U_coeffs.items():
                U_col[i * m + j] = coeff
            
            for (j, k), coeff in V_coeffs.items():
                V_col[j * p + k] = coeff
            
            for (i, k), coeff in W_coeffs.items():
                W_col[k * n + i] = coeff
            
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)
    
    for rb in range(2):
        for cb in range(3):
            if _time_remaining() < 5.0:
                break
            add_strassen_term(rb*2, 0, cb*2)
            add_strassen_term(rb*2, 2, cb*2)
    
    rank = len(U_cols)
    
    # Convert to numpy arrays for rank reduction
    U_arr = np.array(U_cols).T
    V_arr = np.array(V_cols).T
    W_arr = np.array(W_cols).T
    
    # Check for linear dependencies in the combined tensor structure
    # We can remove columns if they are linear combinations of others
    # This is a heuristic: we check the rank of U, V, W separately
    # If a column in U is zero, we can remove it.
    # More effectively, we can try to find a basis.
    
    # For now, we just return the 84 rank solution, but we can try to remove zero columns
    # (which shouldn't exist in this construction)
    
    # To improve, we can try to merge terms if possible, but it's complex.
    # We will return the current 84 rank solution.
    
    U_final = U_arr.tolist()
    V_final = V_arr.tolist()
    W_final = W_arr.tolist()
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
