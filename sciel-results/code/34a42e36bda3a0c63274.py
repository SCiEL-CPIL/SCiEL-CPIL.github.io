"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Strassen decomposition for 2x2 matrix multiplication (rank 7)
    # Indices for 2x2 matrices are 0..3 (row-major: 00, 01, 10, 11)
    # A indices: i*2 + j
    # B indices: j*2 + k
    # C indices: k*2 + i (per task convention W[k*n+i] with n=2)
    
    # u vectors (for A)
    u = [
        np.array([1, 0, 0, 1]), # M1: A00 + A11
        np.array([0, 0, 1, 1]), # M2: A10 + A11
        np.array([1, 0, 0, 0]), # M3: A00
        np.array([0, 0, 0, 1]), # M4: A11
        np.array([1, 1, 0, 0]), # M5: A00 + A01
        np.array([-1, 0, 1, 0]),# M6: A10 - A00
        np.array([0, 1, 0, -1]) # M7: A01 - A11
    ]
    
    # v vectors (for B)
    v = [
        np.array([1, 0, 0, 1]), # M1: B00 + B11
        np.array([1, 0, 0, 0]), # M2: B00
        np.array([0, 1, 0, -1]),# M3: B01 - B11
        np.array([-1, 0, 1, 0]),# M4: B10 - B00
        np.array([0, 0, 0, 1]), # M5: B11
        np.array([1, 1, 0, 0]), # M6: B00 + B01
        np.array([0, 0, 1, 1])  # M7: B10 + B11
    ]
    
    # w vectors (for C)
    # C indices: 0: C00, 1: C10, 2: C01, 3: C11
    # C00 = M1 + M4 - M5 + M7
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    
    w = [
        np.array([1, 0, 0, 1]), # M1: C00 + C11
        np.array([0, 1, 0, -1]),# M2: C10 - C11
        np.array([0, 0, 1, 1]), # M3: C01 + C11
        np.array([1, 1, 0, 0]), # M4: C00 + C10
        np.array([-1, 0, 1, 0]),# M5: -C00 + C01
        np.array([0, 0, 0, 1]), # M6: C11
        np.array([1, 0, 0, 0])  # M7: C00
    ]
    
    n, m, p = DIMENSIONS
    # The rank is 7 * 7 = 49
    rank = 49
    U = []
    V = []
    W = []
    
    for r in range(7):
        for s in range(7):
            U.append(np.kron(u[r], u[s]))
            V.append(np.kron(v[r], v[s]))
            W.append(np.kron(w[r], w[s]))
            
    # Convert to lists of lists as required by interface? 
    # The interface says "factor matrices of shapes...". Usually lists or numpy arrays are fine.
    # The seed used lists. I will return lists to be safe, or numpy arrays if accepted.
    # The seed returns {"U": U, "V": V, "W": W} where U is list of lists.
    
    U_list = [list(map(float, row)) for row in U]
    V_list = [list(map(float, row)) for row in V]
    W_list = [list(map(float, row)) for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
