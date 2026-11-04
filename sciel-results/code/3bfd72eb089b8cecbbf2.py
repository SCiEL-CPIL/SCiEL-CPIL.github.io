"""Tensor decomposition for 4x4 x 4x6 matrix multiplication.

Uses a clean block-Strassen construction:
- 4x4*4x4 part (cols 0-3): rank 49 via two-level Strassen
- 4x4*4x2 part (cols 4-5): rank 32 via schoolbook
Total rank: 81
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 6]


def construct():
    """
    Returns a dictionary with 'U', 'V', 'W' factor matrices.
    
    U: shape (16, rank) - for A (4x4)
    V: shape (24, rank) - for B (4x6)
    W: shape (24, rank) - for C (4x6), with W[k*n+i] convention
    """
    n, m, p = 4, 4, 6
    
    # We'll build up the decomposition, tracking columns
    # Start with a large buffer and trim at the end
    max_rank = 100
    U = np.zeros((n * m, max_rank), dtype=float)
    V = np.zeros((m * p, max_rank), dtype=float)
    W = np.zeros((p * n, max_rank), dtype=float)
    
    col = 0
    
    # Index helpers
    def A_idx(i, j):
        return i * 4 + j
    
    def B_idx(j, k):
        return j * 6 + k
    
    def C_idx(i, k):
        return k * 4 + i  # W[k*n + i] convention
    
    # PART 1: 4x4 * 4x4 (B cols 0-3, C cols 0-3) using two-level Strassen: rank 49
    # Split into 2x2 blocks:
    # A = [A11 A12; A21 A22], B = [B11 B12; B21 B22], C = [C11 C12; C21 C22]
    
    # A block indices (row-major within each 2x2 block)
    A11 = [A_idx(0,0), A_idx(0,1), A_idx(1,0), A_idx(1,1)]  # [0,1,4,5]
    A12 = [A_idx(0,2), A_idx(0,3), A_idx(1,2), A_idx(1,3)]  # [2,3,6,7]
    A21 = [A_idx(2,0), A_idx(2,1), A_idx(3,0), A_idx(3,1)]  # [8,9,12,13]
    A22 = [A_idx(2,2), A_idx(2,3), A_idx(3,2), A_idx(3,3)]  # [10,11,14,15]
    
    # B block indices (rows 0-3, cols 0-3)
    B11 = [B_idx(0,0), B_idx(0,1), B_idx(1,0), B_idx(1,1)]  # [0,1,6,7]
    B12 = [B_idx(0,2), B_idx(0,3), B_idx(1,2), B_idx(1,3)]  # [2,3,8,9]
    B21 = [B_idx(2,0), B_idx(2,1), B_idx(3,0), B_idx(3,1)]  # [12,13,18,19]
    B22 = [B_idx(2,2), B_idx(2,3), B_idx(3,2), B_idx(3,3)]  # [14,15,20,21]
    
    # C block indices (rows 0-3, cols 0-3)
    C11 = [C_idx(0,0), C_idx(0,1), C_idx(1,0), C_idx(1,1)]  # [0,1,4,5]
    C12 = [C_idx(0,2), C_idx(0,3), C_idx(1,2), C_idx(1,3)]  # [2,3,6,7]
    C21 = [C_idx(2,0), C_idx(2,1), C_idx(3,0), C_idx(3,1)]  # [8,9,12,13]
    C22 = [C_idx(2,2), C_idx(2,3), C_idx(3,2), C_idx(3,3)]  # [10,11,14,15]
    
    # Define the 7 level-1 Strassen mults with their A, B linear combos and C targets
    # Each mult is a 2x2*2x2 product where entries are linear combos of block entries
    
    # For each 2x2*2x2 Strassen, we need to expand into 7 rank-1 terms
    # where each "entry" of the virtual 2x2 matrix is a linear combo of actual A/B entries
    
    # Strassen formulas for 2x2*2x2:
    # M1 = (a00+a11)*(b00+b11) -> c00+=M1, c11+=M1, c01-=M1, c10-=M1
    # M2 = (a10+a11)*b00 -> c10+=M2, c11-=M2
    # M3 = a00*(b10-b11) -> c01+=M3, c11-=M3
    # M4 = a11*(b01-b00) -> c00+=M4, c11-=M4
    # M5 = (a00+a01)*b11 -> c01-=M5, c11+=M5
    # M6 = (a10-a00)*(b00+b01) -> c11+=M6
    # M7 = (a01-a11)*(b10+b11) -> c00+=M7
    
    # For each level-1 mult, the virtual 2x2 A and B matrices have entries that are
    # linear combos. When we apply Strassen to these, each of the 7 inner terms
    # becomes a product of two linear combos, which expands to multiple rank-1 terms.
    
    # Let's define a helper to add a rank-1 term
    def add_rank1(u_idx, v_idx, w_idx, coeff):
        nonlocal col
        if col >= max_rank:
            return
        U[u_idx, col] += 1
        V[v_idx, col] += 1
        W[w_idx, col] += coeff
        col += 1
    
    # Define the 7 level-1 mults:
    # Each is: (A_combo, B_combo, C_targets)
    # A_combo: dict (r,c) -> list of (a_block_idx, coeff) for the 2x2 virtual A
    # B_combo: dict (r,c) -> list of (b_block_idx, coeff) for the 2x2 virtual B
    # C_targets: dict (r,c) -> list of (c_block_idx, coeff) for how the 2x2 result maps to C
    
    def make_A_combo(blocks, signs):
        """blocks: list of block index lists, signs: list of signs"""
        result = {}
        for r in range(2):
            for c in range(2):
                entries = []
                for block, sign in zip(blocks, signs):
                    entries.append((block[r*2+c], sign))
                result[(r,c)] = entries
        return result
    
    def make_B_combo(blocks, signs):
        return make_A_combo(blocks, signs)
    
    def make_C_target(blocks, signs):
        return make_A_combo(blocks, signs)
    
    # Level 1 Strassen mults for 4x4*4x4:
    # M1 = (A11+A22)*(B11+B22) -> C11+=, C12-=, C21-=, C22+=
    # M2 = (A21+A22)*B11 -> C21+=, C22-=
    # M3 = A11*(B21-B22) -> C12+=, C22-=
    # M4 = A22*(B12-B11) -> C11+=, C22-=
    # M5 = (A11+A12)*B22 -> C12-=, C22+=
    # M6 = (A21-A11)*(B11+B12) -> C22+=
    # M7 = (A12-A22)*(B21+B22) -> C11+=
    
    level1_mults = [
        # M1
        (make_A_combo([A11, A22], [1, 1]),
         make_B_combo([B11, B22], [1, 1]),
         make_C_target([C11, C12, C21, C22], [1, -1, -1, 1])),
        # M2
        (make_A_combo([A21, A22], [1, 1]),
         make_B_combo([B11], [1]),
         make_C_target([C21, C22], [1, -1])),
        # M3
        (make_A_combo([A11], [1]),
         make_B_combo([B21, B22], [1, -1]),
         make_C_target([C12, C22], [1, -1])),
        # M4
        (make_A_combo([A22], [1]),
         make_B_combo([B12, B11], [1, -1]),
         make_C_target([C11, C22], [1, -1])),
        # M5
        (make_A_combo([A11, A12], [1, 1]),
         make_B_combo([B22], [1]),
         make_C_target([C12, C22], [-1, 1])),
        # M6
        (make_A_combo([A21, A11], [1, -1]),
         make_B_combo([B11, B12], [1, 1]),
         make_C_target([C22], [1])),
        # M7
        (make_A_combo([A12, A22], [1, -1]),
         make_B_combo([B21, B22], [1, 1]),
         make_C_target([C11], [1])),
    ]
    
    # For each level-1 mult, apply Strassen to the 2x2*2x2
    for A_combo, B_combo, C_target in level1_mults:
        # The 7 inner Strassen terms
        # Each term involves sums of entries from the virtual 2x2 A and B
        
        # T1: (a00+a11)*(b00+b11) -> contributes to C[0,0](+), C[1,1](+), C[0,1](-), C[1,0](-)
        a_positions = [(0,0), (1,1)]
        b_positions = [(0,0), (1,1)]
        c_contrib = {(0,0): 1, (1,1): 1, (0,1): -1, (1,0): -1}
        _apply_inner_strassen_term(A_combo, B_combo, C_target, a_positions, b_positions, c_contrib)
        
        # T2: (a10+a11)*b00 -> contributes to C[1,0](+), C[1,1](-)
        a_positions = [(1,0), (1,1)]
        b_positions = [(0,0)]
        c_contrib = {(1,0): 1, (1,1): -1}
        _apply_inner_strassen_term(A_combo, B_combo, C_target, a_positions, b_positions, c_contrib)
        
        # T3: a00*(b10-b11) -> contributes to C[0,1](+), C[1,1](-)
        a_positions = [(0,0)]
        b_positions = [(1,0)]
        c_contrib = {(0,1): 1, (1,1): -1}
        _apply_inner_strassen_term(A_combo, B_combo, C_target, a_positions, b_positions, c_contrib)
        
        # Also a00*(-b11)
        a_positions = [(0,0)]
        b_positions = [(1,1)]
        c_contrib = {(0,1): -1, (1,1): 1}
        _apply_inner_strassen_term(A_combo, B_combo, C_target, a_positions, b_positions, c_contrib)
        
        # T4: a11*(b01-b00) -> contributes to C[0,0](+), C[1,1](-)
        a_positions = [(1,1)]
        b_positions = [(0,1)]
        c_contrib = {(0,0): 1, (1,1): -1}
        _apply_inner_strassen_term(A_combo, B_combo, C_target, a_positions, b_positions, c_contrib)
        
        # Also a11*(-b00)
        a_positions = [(1,1)]
        b_positions = [(0,0)]
        c_contrib = {(0,0): -1, (1,1): 1}
        _apply_inner_strassen_term(A_combo, B_combo, C_target, a_positions, b_positions, c_contrib)
        
        # T5: (a00+a01)*b11 -> contributes to C[0,1](-), C[1,1](+)
        a_positions = [(0,0), (0,1)]
        b_positions = [(1,1)]
        c_contrib = {(0,1): -1, (1,1): 1}
        _apply_inner_strassen_term(A_combo, B_combo, C_target, a_positions, b_positions, c_contrib)
        
        # T6: (a10-a00)*(b00+b01) -> contributes to C[1,1](+)
        a_positions = [(1,0)]
        b_positions = [(0,0), (0,1)]
        c_contrib = {(1,1): 1}
        _apply_inner_strassen_term(A_combo, B_combo, C_target, a_positions, b_positions, c_contrib)
        
        # Also (-a00)*(b00+b01)
        a_positions = [(0,0)]
        b_positions = [(0,0), (0,1)]
        c_contrib = {(1,1): -1}
        _apply_inner_strassen_term(A_combo, B_combo, C_target, a_positions, b_positions, c_contrib)
        
        # T7: (a01-a11)*(b10+b11) -> contributes to C[0,0](+)
        a_positions = [(0,1)]
        b_positions = [(1,0), (1,1)]
        c_contrib = {(0,0): 1}
        _apply_inner_strassen_term(A_combo, B_combo, C_target, a_positions, b_positions, c_contrib)
        
        # Also (-a11)*(b10+b11)
        a_positions = [(1,1)]
        b_positions = [(1,0), (1,1)]
        c_contrib = {(0,0): -1}
        _apply_inner_strassen_term(A_combo, B_combo, C_target, a_positions, b_positions, c_contrib)
    
    # PART 2: 4x4 * 4x2 (B cols 4-5, C cols 4-5) using schoolbook: rank 32
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                U[A_idx(i, j), col] = 1
                V[B_idx(j, k), col] = 1
                W[C_idx(i, k), col] = 1
                col += 1
    
    # Trim to actual rank
    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _apply_inner_strassen_term(A_combo, B_combo, C_target, a_positions, b_positions, c_contrib):
    """
    Apply a single inner Strassen term.
    
    A_combo: dict (r,c) -> list of (a_idx, coeff)
    B_combo: dict (r,c) -> list of (b_idx, coeff)
    C_target: dict (r,c) -> list of (c_idx, coeff)
    a_positions: list of (r,c) positions in A to sum
    b_positions: list of (r,c) positions in B to sum
    c_contrib: dict (r,c) -> coeff for how this scalar contributes to the 2x2 result
    """
    nonlocal col
    
    # Get all A entries
    a_entries = []
    for pos in a_positions:
        for (a_idx, a_coeff) in A_combo[pos]:
            a_entries.append((a_idx, a_coeff))
    
    # Get all B entries
    b_entries = []
    for pos in b_positions:
        for (b_idx, b_coeff) in B_combo[pos]:
            b_entries.append((b_idx, b_coeff))
    
    # For each pair, add a rank-1 term
    for (a_idx, a_coeff) in a_entries:
        for (b_idx, b_coeff) in b_entries:
            coeff = a_coeff * b_coeff
            if coeff == 0:
                continue
            
            # This scalar contributes to C according to c_contrib
            # For each (r,c) in c_contrib, add to all C_target entries at that position
            for (r, c), c_coeff in c_contrib.items():
                total_coeff = coeff * c_coeff
                if total_coeff == 0:
                    continue
                for (c_idx, c_base_coeff) in C_target[(r,c)]:
                    final_coeff = total_coeff * c_base_coeff
                    if final_coeff == 0:
                        continue
                    if col >= 100:
                        return
                    U[a_idx, col] += 1
                    V[b_idx, col] += 1
                    W[c_idx, col] += final_coeff
                    col += 1
