import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Total rank
    # 3 blocks of k (size 2 each)
    # For each k-block:
    #   j in {0,1}: Strassen 2x2x2 (Rank 7)
    #   j in {2,3}: Strassen 2x2x2 (Rank 7)
    #   j in {4}:   Schoolbook 2x1x2 (Rank 4)
    # Total per k-block: 7 + 7 + 4 = 18?
    # Wait, my previous calculation was:
    # Block 1 (j in {0,1}): 3 * 7 = 21.
    # Block 2 (j in {2,3,4}): 3 * (7 + 4) = 33.
    # Total 54.
    
    rank = 54
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Helper to add a Strassen block for local 2x2x2
    # i_base: 0 (since n=2, we always use i=0,1)
    # j_base: start index for j
    # k_base: start index for k
    def add_strassen(j_base, k_base):
        nonlocal col
        # Indices in U: i*m + j
        # Indices in V: j*p + k
        # Indices in W: k*n + i
        
        # M1: (A00+A11)(B00+B11)
        U[0*m + (j_base+0), col] += 1
        U[1*m + (j_base+1), col] += 1
        V[(j_base+0)*p + (k_base+0), col] += 1
        V[(j_base+1)*p + (k_base+1), col] += 1
        W[(k_base+0)*n + 0, col] += 1 # C00
        W[(k_base+1)*n + 1, col] += 1 # C11
        col += 1
        
        # M2: (A10+A11)(B00+B11)
        U[1*m + (j_base+0), col] += 1
        U[1*m + (j_base+1), col] += 1
        V[(j_base+0)*p + (k_base+0), col] += 1
        V[(j_base+1)*p + (k_base+1), col] += 1
        W[(k_base+1)*n + 0, col] += 1 # C10
        W[(k_base+1)*n + 1, col] += -1 # C11
        col += 1
        
        # M3: (A00-A01)(B01-B11)
        U[0*m + (j_base+0), col] += 1
        U[0*m + (j_base+1), col] += -1
        V[(j_base+0)*p + (k_base+1), col] += 1
        V[(j_base+1)*p + (k_base+1), col] += -1
        W[(k_base+0)*n + 1, col] += 1 # C01
        W[(k_base+1)*n + 1, col] += 1 # C11
        col += 1
        
        # M4: (A10+A00)(B10-B11)
        U[1*m + (j_base+0), col] += 1
        U[0*m + (j_base+0), col] += 1
        V[(j_base+1)*p + (k_base+0), col] += 1
        V[(j_base+1)*p + (k_base+1), col] += -1
        W[(k_base+0)*n + 0, col] += 1 # C00
        W[(k_base+1)*n + 0, col] += 1 # C10
        col += 1
        
        # M5: (A01+A11)(B10+B00)
        U[0*m + (j_base+1), col] += 1
        U[1*m + (j_base+1), col] += 1
        V[(j_base+1)*p + (k_base+0), col] += 1
        V[(j_base+0)*p + (k_base+0), col] += 1
        W[(k_base+0)*n + 0, col] += -1 # C00
        W[(k_base+0)*n + 1, col] += 1 # C01
        col += 1
        
        # M6: (A00-A10)(B00+B01)
        U[0*m + (j_base+0), col] += 1
        U[1*m + (j_base+0), col] += -1
        V[(j_base+0)*p + (k_base+0), col] += 1
        V[(j_base+0)*p + (k_base+1), col] += 1
        W[(k_base+1)*n + 1, col] += 1 # C11
        col += 1
        
        # M7: (A01-A00)(B11+B01)
        U[0*m + (j_base+1), col] += 1
        U[0*m + (j_base+0), col] += -1
        V[(j_base+1)*p + (k_base+1), col] += 1
        V[(j_base+0)*p + (k_base+1), col] += 1
        W[(k_base+0)*n + 0, col] += 1 # C00
        col += 1

    # Helper to add a Schoolbook block for 2x1x2
    # j_idx: specific j
    # k_base: start index for k
    def add_schoolbook_212(j_idx, k_base):
        nonlocal col
        # C00 += A0j * Bj(k_base+0)
        U[0*m + j_idx, col] += 1
        V[j_idx*p + (k_base+0), col] += 1
        W[(k_base+0)*n + 0, col] += 1
        col += 1
        
        # C01 += A0j * Bj(k_base+1)
        U[0*m + j_idx, col] += 1
        V[j_idx*p + (k_base+1), col] += 1
        W[(k_base+0)*n + 1, col] += 1
        col += 1
        
        # C10 += A1j * Bj(k_base+0)
        U[1*m + j_idx, col] += 1
        V[j_idx*p + (k_base+0), col] += 1
        W[(k_base+1)*n + 0, col] += 1
        col += 1
        
        # C11 += A1j * Bj(k_base+1)
        U[1*m + j_idx, col] += 1
        V[j_idx*p + (k_base+1), col] += 1
        W[(k_base+1)*n + 1, col] += 1
        col += 1

    # Iterate over k blocks: 0-1, 2-3, 4-5
    for k_block in range(3):
        k_base = k_block * 2
        
        # Part 1: j in {0, 1} -> Strassen 2x2x2
        add_strassen(0, k_base)
        
        # Part 2: j in {2, 3} -> Strassen 2x2x2
        add_strassen(2, k_base)
        
        # Part 3: j in {4} -> Schoolbook 2x1x2
        add_schoolbook_212(4, k_base)
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
