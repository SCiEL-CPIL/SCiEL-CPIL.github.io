"""Matrix multiplication 4,4,5 - rank-minimal tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify tensor decomposition: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] == 1 for all i,j,k"""
    rank = U.shape[1]
    U_arr = U.reshape(n, m, rank)
    V_arr = V.reshape(m, p, rank)
    W_arr = W.reshape(p, n, rank)
    reconstructed = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    expected = np.ones((n, m, p))
    return np.allclose(expected, reconstructed, atol=tol)

def _build_strassen_4x4_4x5():
    """
    Build a structured decomposition for 4x4 * 4x5 using double Strassen.
    
    Strategy:
    - Split A (4x4) into four 2x2 blocks: A1,A2,A3,A4
    - Split B (4x5) into: B1,B2 (2x2 each for cols 0-3), B5_top,B5_bot (2x1 for col 4)
    - Use block-level Strassen for the 4x4 part (7 block products, each 2x2*2x2)
    - Each 2x2*2x2 product uses inner Strassen (rank 7)
    - Handle the 5th column separately
    
    Total rank: 7*7 + 16 = 65 (schoolbook for 5th column)
    But we can do better by integrating the 5th column into the Strassen structure.
    
    Actually, let me think more carefully.
    
    For 4x4 * 4x5:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    Split into blocks:
    A = [[A1, A2], [A3, A4]] where each Ai is 2x2
    B = [[B1, B2, B5t], [B3, B4, B5b]] where B1..B4 are 2x2, B5t/B5b are 2x1
    
    C = [[C11, C12, C15], [C21, C22, C25]]
    where C11,C12,C21,C22 are 2x2 and C15,C25 are 2x1.
    
    C11 = A1*B1 + A2*B3
    C12 = A1*B2 + A2*B4
    C21 = A3*B1 + A4*B3
    C22 = A3*B2 + A4*B4
    C15 = A1*B5t + A2*B5b
    C25 = A3*B5t + A4*B5b
    
    Using Strassen at the block level for the 4x4 part:
    M1 = (A1+A4)*(B1+B4)
    M2 = (A3+A4)*B1
    M3 = A1*(B2-B4)
    M4 = A4*(B3-B1)
    M5 = (A1+A2)*B4
    M6 = (A3-A1)*(B1+B2)
    M7 = (A2-A4)*(B3+B4)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Each Mi is a 2x2 * 2x2 product. Using inner Strassen, each has rank 7.
    Total for 4x4 part: 7 * 7 = 49.
    
    For the 5th column:
    C15 = A1*B5t + A2*B5b (2x2 * 2x1 + 2x2 * 2x1)
    C25 = A3*B5t + A4*B5b
    
    Schoolbook for these: 2*2*1 + 2*2*1 = 8 per row-block, 16 total.
    
    Total: 49 + 16 = 65.
    
    Let me build this explicitly.
    """
    n, m, p = 4, 4, 5
    rank = 65
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper: add an inner Strassen decomposition for a 2x2 * 2x2 product
    # a_indices: [(i,j), (i,j), (i,j), (i,j)] for A entries in order [00, 01, 10, 11]
    # b_indices: [(j,k), (j,k), (j,k), (j,k)] for B entries in order [00, 01, 10, 11]
    # c_indices: [(i,k), (i,k), (i,k), (i,k)] for C entries in order [00, 01, 10, 11]
    # signs: [s0, s1, s2, s3] coefficients for C00, C01, C10, C11
    # col_offset: starting column index
    
    def add_inner_strassen(a_idx, b_idx, c_idx, signs, col_offset):
        a00, a01, a10, a11 = a_idx
        b00, b01, b10, b11 = b_idx
        c00, c01, c10, c11 = c_idx
        s0, s1, s2, s3 = signs
        
        # M1 = (x00+x11)*(y00+y11)
        # Contributes: C00 += s0*M1, C11 += s3*M1
        c = col_offset
        U[a00, c] += 1; U[a11, c] += 1
        V[b00, c] += 1; V[b11, c] += 1
        W[c00, c] += s0; W[c11, c] += s3
        
        # M2 = (x10+x11)*y00
        # Contributes: C10 += s2*M2, C11 += -s3*M2
        c = col_offset + 1
        U[a10, c] += 1; U[a11, c] += 1
        V[b00, c] += 1
        W[c10, c] += s2; W[c11, c] -= s3
        
        # M3 = x00*(y01-y11)
        # Contributes: C01 += s1*M3, C11 += s3*M3
        c = col_offset + 2
        U[a00, c] += 1
        V[b01, c] += 1; V[b11, c] -= 1
        W[c01, c] += s1; W[c11, c] += s3
        
        # M4 = x11*(y10-y00)
        # Contributes: C00 += s0*M4, C10 += s2*M4
        c = col_offset + 3
        U[a11, c] += 1
        V[b10, c] += 1; V[b00, c] -= 1
        W[c00, c] += s0; W[c10, c] += s2
        
        # M5 = (x00+x01)*y11
        # Contributes: C00 += -s0*M5, C01 += s1*M5
        c = col_offset + 4
        U[a00, c] += 1; U[a01, c] += 1
        V[b11, c] += 1
        W[c00, c] -= s0; W[c01, c] += s1
        
        # M6 = (x10-x00)*(y00+y01)
        # Contributes: C11 += s3*M6
        c = col_offset + 5
        U[a10, c] += 1; U[a00, c] -= 1
        V[b00, c] += 1; V[b01, c] += 1
        W[c11, c] += s3
        
        # M7 = (x01-x11)*(y10+y11)
        # Contributes: C00 += s0*M7
        c = col_offset + 6
        U[a01, c] += 1; U[a11, c] -= 1
        V[b10, c] += 1; V[b11, c] += 1
        W[c00, c] += s0
        
        return col_offset + 7
    
    # Block indices
    # A1 = A[0:2, 0:2], A2 = A[0:2, 2:4], A3 = A[2:4, 0:2], A4 = A[2:4, 2:4]
    A1 = [(0,0), (0,2), (1,0), (1,2)]
    A2 = [(0,2), (0,3), (1,2), (1,3)]
    A3 = [(2,0), (2,2), (3,0), (3,2)]
    A4 = [(2,2), (2,3), (3,2), (3,3)]
    
    # B1 = B[0:2, 0:2], B2 = B[0:2, 2:4], B3 = B[2:4, 0:2], B4 = B[2:4, 2:4]
    B1 = [(0,0), (0,2), (1,0), (1,2)]
    B2 = [(0,2), (0,3), (1,2), (1,3)]
    B3 = [(2,0), (2,2), (3,0), (3,2)]
    B4 = [(2,2), (2,3), (3,2), (3,3)]
    
    # B5t = B[0:2, 4], B5b = B[2:4, 4]
    B5t = [(0,4), (1,4)]
    B5b = [(2,4), (3,4)]
    
    # C11 = C[0:2, 0:2], C12 = C[0:2, 2:4], C21 = C[2:4, 0:2], C22 = C[2:4, 2:4]
    C11 = [(0,0), (0,2), (1,0), (1,2)]
    C12 = [(0,2), (0,3), (1,2), (1,3)]
    C21 = [(2,0), (2,2), (3,0), (3,2)]
    C22 = [(2,2), (2,3), (3,2), (3,3)]
    
    # C15 = C[0:2, 4], C25 = C[2:4, 4]
    C15 = [(0,4), (1,4)]
    C25 = [(2,4), (3,4)]
    
    col = 0
    
    # Block-level Strassen terms
    # M1 = (A1+A4)*(B1+B4) -> contributes to C11 (+1) and C22 (+1)
    # The A block is A1+A4, B block is B1+B4
    # For inner Strassen, we need the combined indices
    
    # M1: A = A1+A4, B = B1+B4
    # a00 = A1[0]+A4[0] = A[0,0]+A[2,2], etc.
    # We handle this by adding contributions from both A1 and A4
    
    # Actually, the tensor decomposition allows linear combinations in U and V.
    # So for M1 = (A1+A4)*(B1+B4), each inner Strassen term will have
    # U entries from both A1 and A4, and V entries from both B1 and B4.
    
    # Let me define a helper that takes lists of (index, coefficient) pairs
    
    def add_combined_inner_strassen(a_entries, b_entries, c_idx, signs, col_offset):
        """
        a_entries: list of (index, coeff) for the 4 positions [00,01,10,11] of A block
        b_entries: list of (index, coeff) for the 4 positions [00,01,10,11] of B block
        c_idx: [c00, c01, c10, c11]
        signs: [s0, s1, s2, s3]
        """
        # Expand: each position may have multiple (index, coeff) pairs
        # For position p (00,01,10,11), collect all (index, coeff)
        
        # This is complex. Let me just enumerate directly for each block-level term.
        pass
    
    # Let me take a simpler approach: enumerate all 49 terms manually.
    
    # For M1 = (A1+A4)*(B1+B4):
    # Inner Strassen terms:
    # M1_1 = ((A1[00]+A4[00]) + (A1[11]+A4[11])) * ((B1[00]+B4[00]) + (B1[11]+B4[11]))
    #       = (A[0,0]+A[2,2]+A[1,2]+A[3,3]) * (B[0,0]+B[2,2]+B[1,2]+B[3,3])
    # Contributes to C11[00]=C[0,0] with +1, C22[11]=C[3,3] with +1
    
    # This is getting very verbose. Let me just use a programmatic approach.
    
    # Define the block-level Strassen structure
    # Each block-level term has:
    # - A combination: list of (block, sign) where block is one of A1,A2,A3,A4
    # - B combination: list of (block, sign) where block is one of B1,B2,B3,B4
    # - C contributions: list of (c_block, sign)
    
    block_terms = [
        # (A_blocks, B_blocks, C_contributions)
        # A_blocks: [(block_name, sign), ...]
        # B_blocks: [(block_name, sign), ...]
        # C_contributions: [(c_block_name, sign), ...]
        
        # M1 = (A1+A4)*(B1+B4)
        # C11 += M1, C22 += M1
        ([('A1', 1), ('A4', 1)], [('B1', 1), ('B4', 1)], [('C11', 1), ('C22', 1)]),
        
        # M2 = (A3+A4)*B1
        # C21 += M2, C22 += -M2
        ([('A3', 1), ('A4', 1)], [('B1', 1)], [('C21', 1), ('C22', -1)]),
        
        # M3 = A1*(B2-B4)
        # C12 += M3, C22 += M3
        ([('A1', 1)], [('B2', 1), ('B4', -1)], [('C12', 1), ('C22', 1)]),
        
        # M4 = A4*(B3-B1)
        # C11 += M4, C21 += M4
        ([('A4', 1)], [('B3', 1), ('B1', -1)], [('C11', 1), ('C21', 1)]),
        
        # M5 = (A1+A2)*B4
        # C11 += -M5, C12 += M5
        ([('A1', 1), ('A2', 1)], [('B4', 1)], [('C11', -1), ('C12', 1)]),
        
        # M6 = (A3-A1)*(B1+B2)
        # C22 += M6
        ([('A3', 1), ('A1', -1)], [('B1', 1), ('B2', 1)], [('C22', 1)]),
        
        # M7 = (A2-A4)*(B3+B4)
        # C11 += M7
        ([('A2', 1), ('A4', -1)], [('B3', 1), ('B4', 1)], [('C11', 1)]),
    ]
    
    # Block index mappings
    block_indices = {
        'A1': [(0,0), (0,2), (1,0), (1,2)],
        'A2': [(0,2), (0,3), (1,2), (1,3)],
        'A3': [(2,0), (2,2), (3,0), (3,2)],
        'A4': [(2,2), (2,3), (3,2), (3,3)],
        'B1': [(0,0), (0,2), (1,0), (1,2)],
        'B2': [(0,2), (0,3), (1,2), (1,3)],
        'B3': [(2,0), (2,2), (3,0), (3,2)],
        'B4': [(2,2), (2,3), (3,2), (3,3)],
        'C11': [(0,0), (0,2), (1,0), (1,2)],
        'C12': [(0,2), (0,3), (1,2), (1,3)],
        'C21': [(2,0), (2,2), (3,0), (3,2)],
        'C22': [(2,2), (2,3), (3,2), (3,3)],
    }
    
    # Inner Strassen structure
    # For a 2x2*2x2 product with entries:
    # A: [a00, a01, a10, a11], B: [b00, b01, b10, b11]
    # C: [c00, c01, c10, c11]
    # 
    # M1 = (a00+a11)*(b00+b11) -> C00+=1, C11+=1
    # M2 = (a10+a11)*b00 -> C10+=1, C11+=-1
    # M3 = a00*(b01-b11) -> C01+=1, C11+=1
    # M4 = a11*(b10-b00) -> C00+=1, C10+=1
    # M5 = (a00+a01)*b11 -> C00+=-1, C01+=1
    # M6 = (a10-a00)*(b00+b01) -> C11+=1
    # M7 = (a01-a11)*(b10+b11) -> C00+=1
    
    inner_strassen = [
        # (a_coeffs, b_coeffs, c_coeffs)
        # a_coeffs: [coeff for a00, a01, a10, a11]
        # b_coeffs: [coeff for b00, b01, b10, b11]
        # c_coeffs: [coeff for c00, c01, c10, c11]
        
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # Now build all 49 terms
    for a_blocks, b_blocks, c_contribs in block_terms:
        # Build the combined A entries for each of the 4 positions
        # Position 0: (0,0), Position 1: (0,1), Position 2: (1,0), Position 3: (1,1)
        
        a_entries = [[], [], [], []]  # Each is list of (index, coeff)
        for block_name, sign in a_blocks:
            idx = block_indices[block_name]
            for pos in range(4):
                a_entries[pos].append((idx[pos], sign))
        
        b_entries = [[], [], [], []]
        for block_name, sign in b_blocks:
            idx = block_indices[block_name]
            for pos in range(4):
                b_entries[pos].append((idx[pos], sign))
        
        # For each inner Strassen term
        for a_coeffs, b_coeffs, c_coeffs in inner_strassen:
            # Add this term
            c = col
            
            # U part: for each position, add contributions
            for pos in range(4):
                coeff = a_coeffs[pos]
                if coeff != 0:
                    for idx, block_sign in a_entries[pos]:
                        U[idx[0] * m + idx[1], c] += coeff * block_sign
            
            # V part
            for pos in range(4):
                coeff = b_coeffs[pos]
                if coeff != 0:
                    for idx, block_sign in b_entries[pos]:
                        V[idx[0] * p + idx[1], c] += coeff * block_sign
            
            # W part: apply C contributions
            for c_block_name, c_sign in c_contribs:
                c_idx = block_indices[c_block_name]
                for pos in range(4):
                    coeff = c_coeffs[pos]
                    if coeff != 0:
                        W[c_idx[pos][0] * n + c_idx[pos][1], c] += coeff * c_sign
            
            col += 1
    
    # Now add the 5th column terms (schoolbook)
    # C15 = A1*B5t + A2*B5b
    # C25 = A3*B5t + A4*B5b
    # 
    # C[0,4] = A[0,0]*B[0,4] + A[0,1]*B[1,4] + A[0,2]*B[2,4] + A[0,3]*B[3,4]
    # C[1,4] = A[1,0]*B[0,4] + A[1,1]*B[1,4] + A[1,2]*B[2,4] + A[1,3]*B[3,4]
    # C[2,4] = A[2,0]*B[0,4] + A[2,1]*B[1,4] + A[2,2]*B[2,4] + A[2,3]*B[3,4]
    # C[3,4] = A[3,0]*B[0,4] + A[3,1]*B[1,4] + A[3,2]*B[2,4] + A[3,3]*B[3,4]
    
    for i in range(4):
        for j in range(4):
            U[i * m + j, col] = 1
            V[j * p + 4, col] = 1
            W[4 * n + i, col] = 1
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        raise ValueError("Strassen 4x4*4x5 decomposition failed verification")
    
    return U.tolist(), V.tolist(), W.tolist(), rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Build the structured Strassen decomposition
    U_str, V_str, W_str, rank_str = _build_strassen_4x4_4x5()
    
    best_rank = rank_str
    best_result = {"U": U_str, "V": V_str, "W": W_str}
    
    # Try to find better decompositions using random search
    rng = random.Random(42)
    
    # Try various ranks below the Strassen rank
    for target_rank in range(best_rank - 1, 30, -1):
        if time.time() > deadline:
            break
        
        # Limit time per rank
        time_per_rank = min(5.0, max(0.5, (deadline - time.time()) / max(1, best_rank - target_rank)))
        attempts = int(time_per_rank / 0.01)  # Rough estimate
        attempts = min(attempts, 500)
        
        found = False
        for attempt in range(attempts):
            if time.time() > deadline:
                break
            
            # Random initialization with half-integer values
            choices = [-1, -0.5, 0, 0.5, 1]
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            for c in range(target_rank):
                U[:, c] = [rng.choice(choices) for _ in range(n * m)]
                V[:, c] = [rng.choice(choices) for _ in range(m * p)]
                W[:, c] = [rng.choice(choices) for _ in range(p * n)]
            
            if _verify(U, V, W, n, m, p):
                best_rank = target_rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                found = True
                break
        
        if found:
            # Continue trying lower ranks
            continue
    
    return best_result
# EVOLVE-BLOCK-END
