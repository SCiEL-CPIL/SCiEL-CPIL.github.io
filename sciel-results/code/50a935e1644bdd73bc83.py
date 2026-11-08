import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _strassen_2x2_terms():
    """
    Return the 7 terms of Strassen's 2x2x2 decomposition.
    """
    terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (3, -1)])
    ]
    return terms

def _strassen_4x4x4():
    """
    Construct a rank-49 decomposition for 4x4x4 using Strassen.
    """
    n = m = p = 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    s2_terms = _strassen_2x2_terms()
    
    high_level_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (3, -1)])
    ]
    
    for h_idx, (hA, hB, hC) in enumerate(high_level_terms):
        for l_idx, (lA, lB, lC) in enumerate(s2_terms):
            col = h_idx * 7 + l_idx
            
            for (bA, cA_h) in hA:
                biA = bA // 2
                bjA = bA % 2
                for (lA_idx, cA_l) in lA:
                    liA = lA_idx // 2
                    ljA = lA_idx % 2
                    i = biA * 2 + liA
                    j = bjA * 2 + ljA
                    U[i * 4 + j][col] += cA_h * cA_l
            
            for (bB, cB_h) in hB:
                biB = bB // 2
                bjB = bB % 2
                for (lB_idx, cB_l) in lB:
                    liB = lB_idx // 2
                    ljB = lB_idx % 2
                    j = biB * 2 + liB
                    k = bjB * 2 + ljB
                    V[j * 4 + k][col] += cB_h * cB_l
            
            for (bC, cC_h) in hC:
                biC = bC // 2
                bjC = bC % 2
                for (lC_idx, cC_l) in lC:
                    liC = lC_idx // 2
                    ljC = lC_idx % 2
                    k = biC * 2 + liC
                    i = bjC * 2 + ljC
                    W[k * 4 + i][col] += cC_h * cC_l
    
    return {"U": U, "V": V, "W": W}

def _schoolbook_4x4x1():
    """
    Construct a rank-16 decomposition for 4x4x1.
    """
    n = m = p = 1
    # Wait, p is 1 here, but in the context of 4x4x9, p_local=1
    # U shape (16, 16), V shape (4, 16), W shape (4, 16)
    rank = 16
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    col = 0
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j][col] = 1
            W[i][col] = 1
            col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4x9 matrix multiplication.
    Returns a dictionary with U, V, W matrices.
    """
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = 4, 4, 9
    rank_s = 49
    rank_s2 = 49
    rank_1 = 16
    total_rank = rank_s + rank_s2 + rank_1
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Block 1: k=0..3
    s4_1 = _strassen_4x4x4()
    for i in range(16):
        for c in range(rank_s):
            U[i][c] = s4_1['U'][i][c]
    for j in range(16):
        for c in range(rank_s):
            # V index j*4 + k_local. Here k_local is 0..3.
            # Global V index is j*9 + k_global. k_global = k_local.
            # s4_1['V'] has shape (16, 49). Row j*4+k_local.
            # We need to map row j*4+k_local of s4_1['V'] to row j*9+k_global of V.
            # This is tricky because s4_1['V'] is flattened as j*4+k.
            # We need to iterate over j and k_local.
            pass
            
    # Let's do it properly.
    # s4_1['V'][j*4 + k_local][c]
    # V[j*9 + k_global][c]
    
    for j in range(4):
        for k_local in range(4):
            k_global = k_local
            for c in range(rank_s):
                V[j * 9 + k_global][c] = s4_1['V'][j * 4 + k_local][c]
    
    for k_global in range(4):
        for i in range(4):
            for c in range(rank_s):
                # s4_1['W'] index k_local*4 + i
                # W index k_global*4 + i
                W[k_global * 4 + i][c] = s4_1['W'][k_global * 4 + i][c]
                
    # Block 2: k=4..7
    s4_2 = _strassen_4x4x4()
    offset = rank_s
    for i in range(16):
        for c in range(rank_s):
            U[i][offset + c] = s4_2['U'][i][c]
            
    for j in range(4):
        for k_local in range(4):
            k_global = 4 + k_local
            for c in range(rank_s):
                V[j * 9 + k_global][offset + c] = s4_2['V'][j * 4 + k_local][c]
                
    for k_global in range(4, 8):
        k_local = k_global - 4
        for i in range(4):
            for c in range(rank_s):
                W[k_global * 4 + i][offset + c] = s4_2['W'][k_local * 4 + i][c]

    # Block 3: k=8
    sb_1 = _schoolbook_4x4x1()
    offset2 = rank_s + rank_s2
    for i in range(16):
        for c in range(rank_1):
            U[i][offset2 + c] = sb_1['U'][i][c]
            
    for j in range(4):
        k_global = 8
        for c in range(rank_1):
            V[j * 9 + k_global][offset2 + c] = sb_1['V'][j][c]
            
    for i in range(4):
        k_global = 8
        for c in range(rank_1):
            W[k_global * 4 + i][offset2 + c] = sb_1['W'][i][c]

    return {"U": U, "V": V, "W": W}
