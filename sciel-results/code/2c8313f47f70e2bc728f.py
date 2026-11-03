"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Laderman's algorithm for 2x3x4 matrix multiplication (rank 14)
    # U: 2x3 = 6 entries, V: 3x4 = 12 entries, W: 4x2 = 8 entries
    # Local indices: U[i,j] for i in {0,1}, j in {0,1,2}
    #                V[j,k] for j in {0,1,2}, k in {0,1,2,3}
    #                W[k,i] for k in {0,1,2,3}, i in {0,1}
    
    laderman_terms = [
        { # Term 1
            'U': {0: 1, 3: 1},  # U[0,0]=1, U[1,0]=1
            'V': {0: 1, 6: 1},  # V[0,0]=1, V[1,2]=1
            'W': {0: 1, 4: 1}   # W[0,0]=1, W[2,0]=1
        },
        { # Term 2
            'U': {1: 1, 4: 1},  # U[0,1]=1, U[1,1]=1
            'V': {1: 1, 7: 1},  # V[0,1]=1, V[1,3]=1
            'W': {1: 1, 5: 1}   # W[1,0]=1, W[2,1]=1
        },
        { # Term 3
            'U': {2: 1, 5: 1},  # U[0,2]=1, U[1,2]=1
            'V': {2: 1, 8: 1},  # V[0,2]=1, V[2,0]=1
            'W': {2: 1, 6: 1}   # W[2,0]=1, W[3,0]=1
        },
        { # Term 4
            'U': {0: 1, 5: -1}, # U[0,0]=1, U[1,2]=-1
            'V': {3: 1, 7: -1},# V[0,3]=1, V[1,3]=-1
            'W': {3: 1, 5: -1} # W[3,0]=1, W[2,1]=-1
        },
        { # Term 5
            'U': {1: 1, 4: -1}, # U[0,1]=1, U[1,1]=-1
            'V': {4: 1, 8: -1},# V[1,0]=1, V[2,0]=-1
            'W': {4: 1, 6: -1} # W[1,1]=1, W[3,0]=-1
        },
        { # Term 6
            'U': {2: 1, 3: -1}, # U[0,2]=1, U[1,0]=-1
            'V': {5: 1, 9: -1},# V[1,1]=1, V[2,1]=-1
            'W': {5: 1, 7: -1} # W[2,1]=1, W[3,1]=-1
        },
        { # Term 7
            'U': {0: 1, 1: -1}, # U[0,0]=1, U[0,1]=-1
            'V': {6: 1, 10: -1},# V[1,2]=1, V[2,2]=-1
            'W': {0: -1, 2: 1} # W[0,0]=-1, W[2,0]=1
        },
        { # Term 8
            'U': {1: 1, 2: -1}, # U[0,1]=1, U[0,2]=-1
            'V': {7: 1, 11: -1},# V[1,3]=1, V[2,3]=-1
            'W': {1: -1, 3: 1} # W[1,0]=-1, W[3,0]=1
        },
        { # Term 9
            'U': {3: 1, 4: -1}, # U[1,0]=1, U[1,1]=-1
            'V': {8: 1, 12: -1},# V[2,0]=1, V[3,0]=-1
            'W': {4: -1, 6: 1} # W[1,1]=-1, W[3,0]=1
        },
        { # Term 10
            'U': {4: 1, 5: -1}, # U[1,1]=1, U[1,2]=-1
            'V': {9: 1, 13: -1},# V[2,1]=1, V[3,1]=-1
            'W': {5: -1, 7: 1} # W[2,1]=-1, W[3,1]=1
        },
        { # Term 11
            'U': {0: 1, 2: -1}, # U[0,0]=1, U[0,2]=-1
            'V': {10: 1, 14: -1},# V[2,2]=1, V[3,2]=-1
            'W': {0: 1, 2: -1} # W[0,0]=1, W[2,0]=-1
        },
        { # Term 12
            'U': {1: 1, 3: -1}, # U[0,1]=1, U[1,0]=-1
            'V': {11: 1, 15: -1},# V[2,3]=1, V[3,3]=-1
            'W': {1: 1, 3: -1} # W[1,0]=1, W[3,0]=-1
        },
        { # Term 13
            'U': {5: 1, 0: -1}, # U[1,2]=1, U[0,0]=-1
            'V': {12: 1, 16: -1},# V[3,0]=1, V[4,0]=-1
            'W': {6: 1, 0: -1} # W[3,0]=1, W[0,0]=-1
        },
        { # Term 14
            'U': {4: 1, 1: -1}, # U[1,1]=1, U[0,1]=-1
            'V': {13: 1, 17: -1},# V[3,1]=1, V[4,1]=-1
            'W': {7: 1, 1: -1} # W[3,1]=1, W[1,0]=-1
        }
    ]
    
    # Strategy: 
    # Part 1: Schoolbook for A[0,:] (1x4) * B (4x8): rank 32
    # Part 2: For A[1:3,:] (2x4) * B (4x8):
    #   Split into two 2x2x4 sub-problems:
    #   - A[1:3,0:2] * B[0:2,0:4] -> 2x2x4, use Laderman-like (rank 7 for 2x2x4? No, need 2x2x4)
    #   Actually, let's use a simpler approach:
    #   Split B into 4 blocks of width 2: B[:, 0:2], B[:, 2:4], B[:, 4:6], B[:, 6:8]
    #   For each block, compute A[1:3,:] (2x4) * B_block (4x2) = 2x2
    #   Use Strassen for each 2x2x2: 7 terms x 4 blocks = 28
    #   Total: 32 + 28 = 60
    
    # But we can do better! Let's use a 2x4x4 decomposition
    # A[1:3,:] (2x4) * B (4x8) can be split into:
    # A[1:3,0:2] (2x2) * B[0:2,:] (2x8) + A[1:3,2:4] (2x2) * B[2:4,:] (2x8)
    # Each 2x2x8 can be done with 4 Strassen 2x2x2 blocks = 28 each, total 56
    # That's the same as before.
    
    # Alternative: Use the fact that 2x4x8 = 2*(2x2x4) + 2*(2x2x4)
    # But we need a good 2x2x4 decomposition.
    
    # For now, let's stick with the proven 88-rank approach but optimize the implementation
    # to ensure it runs well within time limits.
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Schoolbook for A[0, :] (1x4) * B (4x8)
    for j in range(m):
        for k in range(p):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
    
    # Strassen terms for 2x2x2
    strassen_terms = [
        {'U': {0: 1, 3: 1}, 'V': {0: 1, 3: 1}, 'W': {0: 1, 3: 1}},
        {'U': {2: 1, 3: 1}, 'V': {0: 1}, 'W': {2: 1, 3: -1}},
        {'U': {0: 1}, 'V': {1: 1, 3: -1}, 'W': {1: 1, 3: 1}},
        {'U': {3: 1}, 'V': {2: 1, 0: -1}, 'W': {0: 1, 2: 1}},
        {'U': {0: 1, 1: 1}, 'V': {3: 1}, 'W': {0: -1, 1: 1}},
        {'U': {2: 1, 0: -1}, 'V': {0: 1, 1: 1}, 'W': {3: 1}},
        {'U': {1: 1, 3: -1}, 'V': {2: 1, 3: 1}, 'W': {0: 1}}
    ]
    
    # Part 2: Strassen for A[1:3, :] (2x4) * B (4x8)
    for j_block in range(4):
        # P1 = A_2a * B_ba: A[1:3,0:2] * B[0:2, 2*j_block:2*j_block+2]
        for term in strassen_terms:
            for local_u, val in term['U'].items():
                lu, lv = divmod(local_u, 2)
                global_i = 1 + lu
                global_j = lv
                U[global_i * m + global_j][col] += val
            for local_v, val in term['V'].items():
                lu, lv = divmod(local_v, 2)
                global_j = lu
                global_k = 2 * j_block + lv
                V[global_j * p + global_k][col] += val
            for local_w, val in term['W'].items():
                lu, lv = divmod(local_w, 2)
                global_i = 1 + lu
                global_k = 2 * j_block + lv
                W[global_k * n + global_i][col] += val
            col += 1
        
        # P2 = A_2b * B_bb: A[1:3,2:4] * B[2:4, 2*j_block:2*j_block+2]
        for term in strassen_terms:
            for local_u, val in term['U'].items():
                lu, lv = divmod(local_u, 2)
                global_i = 1 + lu
                global_j = 2 + lv
                U[global_i * m + global_j][col] += val
            for local_v, val in term['V'].items():
                lu, lv = divmod(local_v, 2)
                global_j = 2 + lu
                global_k = 2 * j_block + lv
                V[global_j * p + global_k][col] += val
            for local_w, val in term['W'].items():
                lu, lv = divmod(local_w, 2)
                global_i = 1 + lu
                global_k = 2 * j_block + lv
                W[global_k * n + global_i][col] += val
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
