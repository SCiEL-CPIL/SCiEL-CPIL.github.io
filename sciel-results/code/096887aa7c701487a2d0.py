import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Rank calculation:
    # Split m=3 into m1=2 and m2=1.
    # T_{2,3,8} = T_{2,2,8} + T_{2,1,8}
    # T_{2,2,8} consists of 4 blocks of size 2x2x2.
    # Each 2x2x2 block uses Strassen's algorithm with rank 7.
    # Total rank for T_{2,2,8} is 4 * 7 = 28.
    # T_{2,1,8} is a linear map from C^1 to C^{2x8} (effectively 2x1x8).
    # The rank of T_{2,1,8} is 2*1*8 = 16 (schoolbook is optimal here as it's just scaling).
    # Total rank = 28 + 16 = 44.
    
    rank = 44
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    # i_off: offset for row index i (0 or 1)
    # j_off: offset for col index j (0, 1, 2, 3...) but here we handle j=0,1 for the 2x2 block
    # k_off: offset for output col index k
    def add_strassen_2x2x2(i_off, j_off, k_off):
        nonlocal col
        # 1. (a+d)(e+h)
        U[i_off*m + j_off, col] += 1
        U[(i_off+1)*m + (j_off+1), col] += 1
        V[j_off*p + k_off, col] += 1
        V[(j_off+1)*p + (k_off+1), col] += 1
        W[k_off*n + i_off, col] += 1
        W[(k_off+1)*n + (i_off+1), col] += 1
        col += 1
        
        # 2. (c+d)e
        U[(i_off+1)*m + j_off, col] += 1
        U[(i_off+1)*m + (j_off+1), col] += 1
        V[j_off*p + k_off, col] += 1
        W[k_off*n + (i_off+1), col] += 1
        W[(k_off+1)*n + (i_off+1), col] += -1
        col += 1
        
        # 3. a(f-h)
        U[i_off*m + j_off, col] += 1
        V[j_off*p + (k_off+1), col] += 1
        V[(j_off+1)*p + (k_off+1), col] += -1
        W[(k_off+1)*n + i_off, col] += 1
        W[(k_off+1)*n + (i_off+1), col] += 1
        col += 1
        
        # 4. d(g-e)
        U[(i_off+1)*m + (j_off+1), col] += 1
        V[(j_off+1)*p + k_off, col] += 1
        V[j_off*p + k_off, col] += -1
        W[k_off*n + i_off, col] += 1
        W[k_off*n + (i_off+1), col] += 1
        col += 1
        
        # 5. (a+b)h
        U[i_off*m + j_off, col] += 1
        U[i_off*m + (j_off+1), col] += 1
        V[(j_off+1)*p + (k_off+1), col] += 1
        W[k_off*n + i_off, col] += -1
        W[(k_off+1)*n + i_off, col] += 1
        col += 1
        
        # 6. (c-a)(e+f)
        U[(i_off+1)*m + j_off, col] += 1
        U[i_off*m + j_off, col] += -1
        V[j_off*p + k_off, col] += 1
        V[j_off*p + (k_off+1), col] += 1
        W[(k_off+1)*n + (i_off+1), col] += 1
        col += 1
        
        # 7. (b-d)(g+h)
        U[i_off*m + (j_off+1), col] += 1
        U[(i_off+1)*m + (j_off+1), col] += -1
        V[(j_off+1)*p + k_off, col] += 1
        V[(j_off+1)*p + (k_off+1), col] += 1
        W[k_off*n + i_off, col] += 1
        col += 1

    # Helper to add schoolbook terms for 2x1x4 block
    # i_off: offset for row index i (0)
    # j_off: offset for col index j (2, the single column in the 2x1 block)
    # k_off: offset for output col index k
    def add_schoolbook_2x1x4(i_off, j_off, k_off):
        nonlocal col
        for i in range(2):
            for k in range(4):
                U[(i_off+i)*m + j_off, col] = 1
                V[j_off*p + (k_off+k), col] = 1
                W[(k_off+k)*n + (i_off+i), col] = 1
                col += 1

    # We have p=8 output columns.
    # We split the j dimension (m=3) into j in {0,1} for Strassen and j=2 for Schoolbook.
    # We process k in chunks of 2 for Strassen (since Strassen is 2x2x2).
    # k ranges from 0 to 7.
    # Chunks: (0,1), (2,3), (4,5), (6,7).
    
    for k_chunk in range(4):
        k_off = k_chunk * 2
        
        # Apply Strassen for the 2x2 part of the input (j=0,1)
        # i is always 0,1 so i_off=0
        add_strassen_2x2x2(0, 0, k_off)
        
        # Apply Schoolbook for the 1x2 part of the input (j=2)
        # Note: The "2x1" here refers to the 2 rows of A and 1 col of B.
        # The output is 2x2 (since k_chunk is 2 wide).
        # Wait, the helper add_schoolbook_2x1x4 handles 4 k's? 
        # No, let's look at the helper. It loops k in range(4).
        # But Strassen handles 2 k's.
        # So we need a helper that handles 4 k's for the schoolbook part?
        # Or we can just call a 2x1x2 schoolbook helper twice?
        # Let's stick to the 4-wide helper for efficiency in code, but we need to map it correctly.
        # Actually, the previous code used add_schoolbook_2x1x4(0, 2, 0) for k=0..3.
        # This means it covered the first half of the output matrix.
        # And add_schoolbook_2x1x4(0, 2, 4) for k=4..7.
        # So we only call the schoolbook helper twice, not 4 times.
        
    # Correction: The loop above calls Strassen 4 times.
    # We need to call Schoolbook 2 times (for k=0..3 and k=4..7).
    
    # Let's re-implement the loop structure to be clear.
    
    # Reset col
    col = 0
    U.fill(0)
    V.fill(0)
    W.fill(0)
    
    # k=0,1
    add_strassen_2x2x2(0, 0, 0)
    # k=2,3
    add_strassen_2x2x2(0, 0, 2)
    # k=4,5
    add_strassen_2x2x2(0, 0, 4)
    # k=6,7
    add_strassen_2x2x2(0, 0, 6)
    
    # Schoolbook for j=2
    # k=0..3
    add_schoolbook_2x1x4(0, 2, 0)
    # k=4..7
    add_schoolbook_2x1x4(0, 2, 4)
    
    if time.time() > deadline:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
