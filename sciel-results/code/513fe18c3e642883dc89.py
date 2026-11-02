# EVOLVE-BLOCK-START
import time

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x8 matrix.
    
    Strategy: Use Strassen decomposition on 2x2 blocks.
    - The 2x4 * 4x8 multiplication is partitioned into 4 independent 2x4 * 4x2 multiplications (column pairs).
    - Each 2x4 * 4x2 multiplication is A_L B_L + A_R B_R, computed via two Strassen 2x2 * 2x2 multiplications.
    - Each Strassen 2x2 * 2x2 uses rank 7.
    - Total rank per block: 14.
    - Total rank: 4 * 14 = 56.
    
    The output factor follows the official trace convention W[k*n+i,r].
    """
    start_time = time.time()
    deadline = start_time + 10.0 # Safety margin
    
    n, m, p = 2, 4, 8
    
    terms = []
    
    for j in range(4):
        # B1j = B[0:2, 2j:2j+2]
        # B2j = B[2:4, 2j:2j+2]
        
        # A1 = A[:, 0:2], A2 = A[:, 2:4]
        
        # Strassen for A1 * B1j
        terms.append({
            'A': {(0,0): 1, (1,1): 1},
            'B': {(0, 2*j): 1, (1, 2*j+1): 1},
            'C': {}
        })
        terms.append({
            'A': {(1,0): 1, (1,1): 1},
            'B': {(0, 2*j): 1},
            'C': {}
        })
        terms.append({
            'A': {(0,0): 1},
            'B': {(0, 2*j+1): 1, (1, 2*j+1): -1},
            'C': {}
        })
        terms.append({
            'A': {(1,1): 1},
            'B': {(1, 2*j): 1, (0, 2*j): -1},
            'C': {}
        })
        terms.append({
            'A': {(0,0): 1, (0,1): 1},
            'B': {(1, 2*j+1): 1},
            'C': {}
        })
        terms.append({
            'A': {(1,0): 1, (0,0): -1},
            'B': {(0, 2*j): 1, (0, 2*j+1): 1},
            'C': {}
        })
        terms.append({
            'A': {(0,1): 1, (1,1): -1},
            'B': {(1, 2*j): 1, (1, 2*j+1): 1},
            'C': {}
        })
        
        # Strassen for A2 * B2j
        terms.append({
            'A': {(0,2): 1, (1,3): 1},
            'B': {(2, 2*j): 1, (3, 2*j+1): 1},
            'C': {}
        })
        terms.append({
            'A': {(1,2): 1, (1,3): 1},
            'B': {(2, 2*j): 1},
            'C': {}
        })
        terms.append({
            'A': {(0,2): 1},
            'B': {(2, 2*j+1): 1, (3, 2*j+1): -1},
            'C': {}
        })
        terms.append({
            'A': {(1,3): 1},
            'B': {(3, 2*j): 1, (2, 2*j): -1},
            'C': {}
        })
        terms.append({
            'A': {(0,2): 1, (0,3): 1},
            'B': {(3, 2*j+1): 1},
            'C': {}
        })
        terms.append({
            'A': {(1,2): 1, (0,2): -1},
            'B': {(2, 2*j): 1, (2, 2*j+1): 1},
            'C': {}
        })
        terms.append({
            'A': {(0,3): 1, (1,3): -1},
            'B': {(3, 2*j): 1, (3, 2*j+1): 1},
            'C': {}
        })
        
        if time.time() > deadline:
            break

    # Compute C contributions
    for j in range(4):
        base = j * 14
        
        terms[base + 0]['C'] = {4*j: 1, 4*j+3: 1}
        terms[base + 1]['C'] = {4*j+1: 1, 4*j+3: -1}
        terms[base + 2]['C'] = {4*j+2: 1, 4*j+3: 1}
        terms[base + 3]['C'] = {4*j: 1, 4*j+1: 1}
        terms[base + 4]['C'] = {4*j: -1, 4*j+2: 1}
        terms[base + 5]['C'] = {4*j+3: 1}
        terms[base + 6]['C'] = {4*j: 1}
        
        terms[base + 7]['C'] = {4*j: 1, 4*j+3: 1}
        terms[base + 8]['C'] = {4*j+1: 1, 4*j+3: -1}
        terms[base + 9]['C'] = {4*j+2: 1, 4*j+3: 1}
        terms[base + 10]['C'] = {4*j: 1, 4*j+1: 1}
        terms[base + 11]['C'] = {4*j: -1, 4*j+2: 1}
        terms[base + 12]['C'] = {4*j+3: 1}
        terms[base + 13]['C'] = {4*j: 1}
        
        if time.time() > deadline:
            break

    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, term in enumerate(terms):
        for (i, j_idx), coeff in term['A'].items():
            U[i * m + j_idx][c] = coeff
        for (j_idx, k), coeff in term['B'].items():
            V[j_idx * p + k][c] = coeff
        for (idx, coeff) in term['C'].items():
            if coeff != 0:
                W[idx][c] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
