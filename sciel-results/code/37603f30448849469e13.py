"""Matrix multiplication 4,4,5 - rank-minimal exact tensor decomposition."""

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Returns {'U', 'V', 'W'} where:
      U has shape (n*m, rank) = (16, rank)
      V has shape (m*p, rank) = (20, rank)
      W has shape (p*n, rank) = (20, rank)
    
    The decomposition satisfies:
      C[i,k] = sum_r U[i*m+j, r] * A[i,j] * B[j,k] ... 
    
    Actually the tensor decomposition for matrix multiplication M_{n,m,p} is:
      (A * B)[i,k] = sum_j A[i,j] * B[j,k]
    
    In tensor form: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    where A is n x m, B is m x p, C is n x p.
    
    Wait, let me re-read the convention: W[k*n+i, r]
    So the output index for W is k*n+i, where i is the row of C and k is the column.
    A is 4x4 (n=4, m=4), B is 4x5 (m=4, p=5), C is 4x5 (n=4, p=5).
    
    U index: i*m + j = i*4 + j, for i in [0,4), j in [0,4) -> 16 entries
    V index: j*p + k = j*5 + k, for j in [0,4), k in [0,5) -> 20 entries
    W index: k*n + i = k*4 + i, for k in [0,5), i in [0,4) -> 20 entries
    """
    n, m, p = 4, 4, 5
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # We use nested Strassen for 4x4 * 4x4 (rank 49) + schoolbook for 4x4 * 4x1 (rank 16)
    # Total rank = 65
    
    rank = 65
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    def add_term(col, u_list, v_list, w_list):
        for idx, c in u_list:
            U[idx, col] += c
        for idx, c in v_list:
            V[idx, col] += c
        for idx, c in w_list:
            W[idx, col] += c
    
    col = 0
    
    # ===== Part 1: 4x4 * 4x4 using nested Strassen (rank 49) =====
    # Split into 2x2 blocks of 2x2 matrices
    
    # Strassen formulas for 2x2:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    # C00 = P1+P4-P5+P7, C01 = P3+P5, C10 = P2+P4, C11 = P1-P2+P3+P6
    
    # For the outer level, each "element" is a 2x2 block.
    # For the inner level, we compute each 2x2*2x2 product using Strassen.
    
    # Outer Strassen coefficients (for A blocks and B blocks):
    # Each coefficient is a 4-element vector [block00, block01, block10, block11]
    outer_A = [
        [1, 0, 0, 1],   # P1: A00+A11
        [0, 0, 1, 1],   # P2: A10+A11
        [1, 0, 0, 0],   # P3: A00
        [0, 0, 0, 1],   # P4: A11
        [1, 1, 0, 0],   # P5: A00+A01
        [-1, 0, 1, 0],  # P6: A10-A00
        [0, 1, 0, -1],  # P7: A01-A11
    ]
    outer_B = [
        [1, 0, 0, 1],   # P1: B00+B11
        [1, 0, 0, 0],   # P2: B00
        [0, 1, 0, -1],  # P3: B01-B11
        [-1, 0, 1, 0],  # P4: B10-B00
        [0, 0, 0, 1],   # P5: B11
        [1, 1, 0, 0],   # P6: B00+B01
        [0, 0, 1, 1],   # P7: B10+B11
    ]
    
    # Inner Strassen (same formulas, but now on scalar 2x2 blocks):
    inner_A = [
        [1, 0, 0, 1],   # S1: A00+A11
        [0, 0, 1, 1],   # S2: A10+A11
        [1, 0, 0, 0],   # S3: A00
        [0, 0, 0, 1],   # S4: A11
        [1, 1, 0, 0],   # S5: A00+A01
        [-1, 0, 1, 0],  # S6: A10-A00
        [0, 1, 0, -1],  # S7: A01-A11
    ]
    inner_B = [
        [1, 0, 0, 1],   # S1: B00+B11
        [1, 0, 0, 0],   # S2: B00
        [0, 1, 0, -1],  # S3: B01-B11
        [-1, 0, 1, 0],  # S4: B10-B00
        [0, 0, 0, 1],   # S5: B11
        [1, 1, 0, 0],   # S6: B00+B01
        [0, 0, 1, 1],   # S7: B10+B11
    ]
    
    # Output combination for inner Strassen:
    # C00 = S1+S4-S5+S7, C01 = S3+S5, C10 = S2+S4, C11 = S1-S2+S3+S6
    inner_out = [
        [1, 0, 0, 1],   # S1
        [0, 0, 1, -1],  # S2
        [0, 1, 0, 1],   # S3
        [1, 0, 1, 0],   # S4
        [-1, 1, 0, 0],  # S5
        [0, 0, 0, 1],   # S6
        [1, 0, 0, 0],   # S7
    ]
    
    # Output combination for outer Strassen:
    # C00 = P1+P4-P5+P7, C01 = P3+P5, C10 = P2+P4, C11 = P1-P2+P3+P6
    outer_out = [
        [1, 0, 0, 1],   # P1
        [0, 0, 1, -1],  # P2
        [0, 1, 0, 1],   # P3
        [1, 0, 1, 0],   # P4
        [-1, 1, 0, 0],  # P5
        [0, 0, 0, 1],   # P6
        [1, 0, 0, 0],   # P7
    ]
    
    def block_to_global(block_idx, local_r, local_c):
        """Convert block index and local position to global (i,j) for A or (j,k) for B."""
        br = block_idx // 2
        bc = block_idx % 2
        return (2*br + local_r, 2*bc + local_c)
    
    for s in range(7):  # outer Strassen term
        for t in range(7):  # inner Strassen term
            if time.time() > deadline:
                break
            
            # Build u_list: contributions to U from inner_A[t] combined with outer_A[s]
            u_list = []
            for ia_idx in range(4):  # inner A block index
                ia_coeff = inner_A[t][ia_idx]
                if ia_coeff == 0:
                    continue
                # For each local element in this 2x2 block
                for lr in range(2):
                    for lc in range(2):
                        # Get the outer block contribution for this local element
                        # The inner block ia_idx corresponds to a 2x2 sub-block
                        # We need to map (ia_idx, lr, lc) to global position
                        # inner_A coefficients apply to the 4 blocks of the 2x2 super-block
                        # Each block has 4 scalar elements
                        # ia_idx selects which 2x2 block, lr,lc select within that block
                        i_global, j_global = block_to_global(ia_idx, lr, lc)
                        
                        # Now multiply by the outer coefficient for the corresponding outer block
                        # The outer block that contains this element
                        outer_block = ia_idx  # same block index
                        outer_coeff = outer_A[s][outer_block]
                        total = ia_coeff * outer_coeff
                        if total != 0:
                            u_list.append((i_global * m + j_global, total))
            
            # Build v_list: contributions to V from inner_B[t] combined with outer_B[s]
            v_list = []
            for ib_idx in range(4):  # inner B block index
                ib_coeff = inner_B[t][ib_idx]
                if ib_coeff == 0:
                    continue
                for lr in range(2):
                    for lc in range(2):
                        j_global, k_global = block_to_global(ib_idx, lr, lc)
                        # k_global is in [0,4) since we're doing 4x4 part
                        outer_block = ib_idx
                        outer_coeff = outer_B[s][outer_block]
                        total = ib_coeff * outer_coeff
                        if total != 0:
                            v_list.append((j_global * p + k_global, total))
            
            # Build w_list: contributions to W from inner_out[t] combined with outer_out[s]
            w_list = []
            for io_idx in range(4):  # inner output block index
                io_coeff = inner_out[t][io_idx]
                if io_coeff == 0:
                    continue
                for lr in range(2):
                    for lc in range(2):
                        i_global, k_global = block_to_global(io_idx, lr, lc)
                        outer_block = io_idx
                        outer_coeff = outer_out[s][outer_block]
                        total = io_coeff * outer_coeff
                        if total != 0:
                            w_list.append((k_global * n + i_global, total))
            
            add_term(col, u_list, v_list, w_list)
            col += 1
    
    # ===== Part 2: 4x4 * 4x1 schoolbook (rank 16) =====
    # The 5th column of B (index k=4)
    for i in range(n):
        for j in range(m):
            if time.time() > deadline:
                break
            u_list = [(i * m + j, 1)]
            v_list = [(j * p + 4, 1)]
            w_list = [(4 * n + i, 1)]
            add_term(col, u_list, v_list, w_list)
            col += 1
    
    # Convert to lists of lists with integer entries
    U_out = U.tolist()
    V_out = V.tolist()
    W_out = W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
