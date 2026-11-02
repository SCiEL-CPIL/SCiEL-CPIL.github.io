"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank is 56 based on recursive Strassen decomposition
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_222(base_u, base_v, base_w, offset):
        """
        Adds a 2x2x2 Strassen decomposition at the given offset.
        base_u, base_v, base_w are the starting indices in the global U, V, W arrays
        for the 2x2, 2x2, 2x2 blocks respectively.
        """
        # U indices: i*2 + j for i,j in 0..1
        # V indices: j*2 + k for j,k in 0..1
        # W indices: k*2 + i for k,i in 0..1
        
        # M1
        U[base_u + 0][offset] += 1
        U[base_u + 3][offset] += 1
        V[base_v + 0][offset] += 1
        V[base_v + 3][offset] += 1
        W[base_w + 0][offset] += 1
        W[base_w + 3][offset] += 1
        
        # M2
        U[base_u + 2][offset] += 1
        U[base_u + 3][offset] += 1
        V[base_v + 0][offset] += 1
        W[base_w + 1][offset] += 1
        
        # M3
        U[base_u + 0][offset] += 1
        V[base_v + 1][offset] += 1
        V[base_v + 3][offset] -= 1
        W[base_w + 2][offset] += 1
        W[base_w + 3][offset] += 1
        
        # M4
        U[base_u + 3][offset] += 1
        V[base_v + 2][offset] += 1
        V[base_v + 0][offset] -= 1
        W[base_w + 1][offset] += 1
        W[base_w + 0][offset] += 1
        
        # M5
        U[base_u + 0][offset] += 1
        U[base_u + 1][offset] += 1
        V[base_v + 3][offset] += 1
        W[base_w + 0][offset] -= 1
        W[base_w + 2][offset] += 1
        
        # M6
        U[base_u + 2][offset] += 1
        U[base_u + 0][offset] -= 1
        V[base_v + 0][offset] += 1
        V[base_v + 1][offset] += 1
        W[base_w + 3][offset] += 1
        
        # M7
        U[base_u + 1][offset] += 1
        U[base_u + 3][offset] -= 1
        V[base_v + 2][offset] += 1
        V[base_v + 3][offset] += 1
        W[base_w + 0][offset] += 1

    def add_strassen_224(base_u, base_v, base_w, offset):
        """
        Adds a 2x2x4 decomposition using two 2x2x2 Strassen blocks.
        V is split into two 2x2 blocks horizontally.
        """
        # First 2x2x2 block (columns 0-1 of V)
        add_strassen_222(base_u, base_v, base_w, offset)
        # Second 2x2x2 block (columns 2-3 of V)
        # V indices for second block: base_v + 2 (j=0,k=2) to base_v + 5 (j=1,k=3)
        # Wait, V index is j*p + k. Here p=2 for the sub-block.
        # In the 2x2x4 context, V has shape 2x4.
        # Index is j*4 + k.
        # For k=0,1: j*4 + k.
        # For k=2,3: j*4 + 2 + (k-2) = j*4 + k.
        # So base_v for the second block is just base_v + 2?
        # No, base_v points to the start of the 2x4 V block.
        # The first 2x2 V block uses indices base_v + 0, 1, 2, 3.
        # The second 2x2 V block uses indices base_v + 4, 5, 6, 7.
        # Let's check indices:
        # j=0, k=0 -> 0
        # j=0, k=1 -> 1
        # j=1, k=0 -> 4
        # j=1, k=1 -> 5
        # So for the second block (k=2,3), the indices are:
        # j=0, k=2 -> 2
        # j=0, k=3 -> 3
        # j=1, k=2 -> 6
        # j=1, k=3 -> 7
        
        # My add_strassen_222 assumes V indices are contiguous 0..3 for the 2x2 block.
        # Here they are 2,3,6,7.
        # So I need to pass a mapping or adjust.
        # Let's modify add_strassen_222 to take explicit index lists or offsets.
        pass

    # Refined approach: Generate the full 2x2x8 decomposition indices explicitly.
    
    # 2x2x2 Strassen coefficients
    # U indices: 0,1,2,3 (i*2+j)
    # V indices: 0,1,2,3 (j*2+k)
    # W indices: 0,1,2,3 (k*2+i)
    
    strassen_222 = [
        # (U_idx, coeff), (V_idx, coeff), (W_idx, coeff)
        ([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)]), # M1
        ([(2,1), (3,1)], [(0,1)], [(1,1)]),               # M2
        ([(0,1)], [(1,1), (3,-1)], [(2,1), (3,1)]),       # M3
        ([(3,1)], [(2,1), (0,-1)], [(1,1), (0,1)]),       # M4
        ([(0,1), (1,1)], [(3,1)], [(0,-1), (2,1)]),       # M5
        ([(2,1), (0,-1)], [(0,1), (1,1)], [(3,1)]),       # M6
        ([(1,1), (3,-1)], [(2,1), (3,1)], [(0,1)]),       # M7
    ]

    def apply_block(base_u, base_v, base_w, offset, u_map, v_map, w_map):
        """
        Applies the 222 Strassen block.
        u_map, v_map, w_map are dictionaries mapping local index (0..3) to global index.
        """
        for u_list, v_list, w_list in strassen_222:
            for idx, coeff in u_list:
                U[u_map[idx]][offset] += coeff
            for idx, coeff in v_list:
                V[v_map[idx]][offset] += coeff
            for idx, coeff in w_list:
                W[w_map[idx]][offset] += coeff

    # Build 2x2x8 decomposition
    # A: 2x2 (indices 0..3)
    # B: 2x8 (indices 0..15). Split into four 2x2 blocks.
    # C: 2x8 (indices 0..15). Split into four 2x2 blocks.
    
    # We need to map the local 2x2 indices to the global 2x8 indices.
    # Local 2x2 block t (0..3) of B corresponds to columns 2t, 2t+1.
    # Global V index for local (j, k) in block t is j*8 + (2t + k).
    # Local W index for local (k, i) in block t is (2t + k)*2 + i.
    
    # We compute 4 independent 2x2x2 multiplications?
    # No, Strassen mixes the columns.
    # But we are decomposing 2x2x8 into four 2x2x2 blocks of B and C.
    # Since A is the same for all, we can't just do 4 independent Strassen.
    # Wait, $A B = A [B_0, B_1, B_2, B_3] = [A B_0, A B_1, A B_2, A B_3]$.
    # Yes, we can compute each $A B_t$ independently using Strassen.
    # Each $A B_t$ is a 2x2x2 multiplication.
    # So we just apply Strassen 4 times, with different V and W mappings.
    
    # For 2x2x8:
    # A is fixed.
    # For t in 0..3:
    #   U map: identity (0->0, 1->1, 2->2, 3->3)
    #   V map: local (j,k) -> global j*8 + (2t+k)
    #   W map: local (k,i) -> global (2t+k)*2 + i
    
    u_map_id = {0:0, 1:1, 2:2, 3:3}
    
    # We need to build the 2x4x8 from two 2x2x8.
    # A1 (j=0,1) and A2 (j=2,3).
    
    # Term 1: A1 B1.
    # A1 indices in global U:
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=1, j=0 -> 4
    # i=1, j=1 -> 5
    # Local 2x2 indices for A1:
    # 0->0, 1->1, 2->4, 3->5
    
    u_map_1 = {0:0, 1:1, 2:4, 3:5}
    
    # B1 indices in global V:
    # j=0, k=0..7 -> 0..7
    # j=1, k=0..7 -> 8..15
    # Local 2x8 V indices:
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=1, k=0 -> 8
    # j=1, k=1 -> 9
    # ...
    # For 2x2x2 block t of B1:
    # Local (j,k) in 2x2 block t corresponds to global j*8 + (2t+k).
    
    # We will loop t=0..3 for the 2x2x8 part.
    
    current_offset = 0
    for t in range(4):
        # V map for block t
        v_map = {}
        for j in range(2):
            for k in range(2):
                local_idx = j * 2 + k
                global_idx = j * 8 + (2 * t + k)
                v_map[local_idx] = global_idx
        
        # W map for block t
        w_map = {}
        for k in range(2):
            for i in range(2):
                local_idx = k * 2 + i
                global_idx = (2 * t + k) * 2 + i
                w_map[local_idx] = global_idx
        
        apply_block(0, 0, 0, current_offset, u_map_1, v_map, w_map)
        current_offset += 7
        
    # Term 2: A2 B2.
    # A2 indices in global U:
    # i=0, j=2 -> 2
    # i=0, j=3 -> 3
    # i=1, j=2 -> 6
    # i=1, j=3 -> 7
    
    u_map_2 = {0:2, 1:3, 2:6, 3:7}
    
    # B2 indices in global V:
    # j=2, k=0..7 -> 16..23
    # j=3, k=0..7 -> 24..31
    
    for t in range(4):
        v_map = {}
        for j in range(2):
            for k in range(2):
                local_idx = j * 2 + k
                # Global j is 2+j
                global_idx = (2 + j) * 8 + (2 * t + k)
                v_map[local_idx] = global_idx
        
        w_map = {}
        for k in range(2):
            for i in range(2):
                local_idx = k * 2 + i
                global_idx = (2 * t + k) * 2 + i
                w_map[local_idx] = global_idx
        
        apply_block(0, 0, 0, current_offset, u_map_2, v_map, w_map)
        current_offset += 7

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
