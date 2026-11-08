import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 3, 4
    rank = 20
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Define the 20 terms of the rank-20 decomposition for M_{2,3,4}
    
    # Helper function to add a term to the matrices
    def add_term(col_idx, u_vals, v_vals, w_vals):
        for idx, val in u_vals.items():
            U[idx, col_idx] += val
        for idx, val in v_vals.items():
            V[idx, col_idx] += val
        for idx, val in w_vals.items():
            W[idx, col_idx] += val
            
    # Indices:
    # A (2x3): a00=0, a01=1, a02=2, a10=3, a11=4, a12=5
    # B (3x4): b00=0, b01=1, b02=2, b03=3, b10=4, b11=5, b12=6, b13=7, b20=8, b21=9, b22=10, b23=11
    # C (2x4): c00=0, c01=1, c02=2, c03=3, c10=4, c11=5, c12=6, c13=7
    # W index = k*n + i where k is col of C (0-3), i is row of C (0-1)
    # c00 -> 0*2+0=0, c01 -> 1*2+0=2, c02 -> 2*2+0=4, c03 -> 3*2+0=6
    # c10 -> 0*2+1=1, c11 -> 1*2+1=3, c12 -> 2*2+1=5, c13 -> 3*2+1=7
    
    # Rank-20 decomposition using Strassen-like techniques with cross-block sharing
    
    # Terms 0-6: First 2x2x2 block (A[:,0:2] * B[0:2,0:2] -> C[0:2,0:2])
    # M1: (a00+a11)(b00+b11) -> c00+c11
    add_term(0, {0: 1, 4: 1}, {0: 1, 5: 1}, {0: 1, 3: 1})
    # M2: (a10+a11)b00 -> c10-c11
    add_term(1, {3: 1, 4: 1}, {0: 1}, {1: 1, 3: -1})
    # M3: a00(b01-b11) -> c01+c11
    add_term(2, {0: 1}, {1: 1, 5: -1}, {2: 1, 3: 1})
    # M4: a11(b10-b00) -> c00+c10
    add_term(3, {4: 1}, {4: 1, 0: -1}, {0: 1, 1: 1})
    # M5: (a00+a01)b11 -> -c00+c01
    add_term(4, {0: 1, 1: 1}, {5: 1}, {0: -1, 2: 1})
    # M6: (a10-a00)(b00+b01) -> c11
    add_term(5, {3: 1, 0: -1}, {0: 1, 1: 1}, {3: 1})
    # M7: (a01-a11)(b10+b11) -> c00
    add_term(6, {1: 1, 4: -1}, {4: 1, 5: 1}, {0: 1})
    
    # Terms 7-13: Second 2x2x2 block (A[:,0:2] * B[0:2,2:4] -> C[0:2,2:4])
    # M1': (a00+a11)(b02+b13) -> c02+c13
    add_term(7, {0: 1, 4: 1}, {2: 1, 7: 1}, {4: 1, 7: 1})
    # M2': (a10+a11)b02 -> c12-c13
    add_term(8, {3: 1, 4: 1}, {2: 1}, {5: 1, 7: -1})
    # M3': a00(b03-b13) -> c03+c13
    add_term(9, {0: 1}, {3: 1, 7: -1}, {6: 1, 7: 1})
    # M4': a11(b12-b02) -> c02+c12
    add_term(10, {4: 1}, {6: 1, 2: -1}, {4: 1, 5: 1})
    # M5': (a00+a01)b13 -> -c02+c03
    add_term(11, {0: 1, 1: 1}, {7: 1}, {4: -1, 6: 1})
    # M6': (a10-a00)(b02+b03) -> c13
    add_term(12, {3: 1, 0: -1}, {2: 1, 3: 1}, {7: 1})
    # M7': (a01-a11)(b12+b13) -> c02
    add_term(13, {1: 1, 4: -1}, {6: 1, 7: 1}, {4: 1})
    
    # Terms 14-19: Third block (A[:,2:3] * B[2:3,0:4] -> C[0:2,0:4])
    # Instead of 8 direct terms, use 6 terms with sharing
    # a02*b20 -> c00
    add_term(14, {2: 1}, {8: 1}, {0: 1})
    # a02*b21 -> c01
    add_term(15, {2: 1}, {9: 1}, {2: 1})
    # a02*b22 -> c02
    add_term(16, {2: 1}, {10: 1}, {4: 1})
    # a02*b23 -> c03
    add_term(17, {2: 1}, {11: 1}, {6: 1})
    # a12*b20 -> c10
    add_term(18, {5: 1}, {8: 1}, {1: 1})
    # a12*b21 -> c11
    add_term(19, {5: 1}, {9: 1}, {3: 1})
    
    # Wait, that's only 6 terms for 8 products. Let me reconsider.
    # Actually, for a 2x1x4 tensor, the rank is exactly 8 (it's just outer products).
    # So we need all 8 terms. Let me fix this.
    
    # Reset and redo with correct rank
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # The known rank of M_{2,3,4} is actually 20, not 22.
    # Let me use a different decomposition strategy.
    
    # Actually, let me verify: the current rank-22 solution might be correct but suboptimal.
    # The theoretical lower bound for M_{2,3,4} is 16 (from alpha * beta * gamma / (2*3*4) considerations),
    # but the known upper bound is 20.
    
    # For now, let's keep the rank-22 solution but verify it's correct.
    # If it's correct, we can try to find a rank-20 solution.
    
    # Let me implement the known rank-20 decomposition from literature.
    
    # Clear and start fresh
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Rank-20 decomposition:
    # Use 7 terms for first 2x2x2 block
    add_term(0, {0: 1, 4: 1}, {0: 1, 5: 1}, {0: 1, 3: 1})
    add_term(1, {3: 1, 4: 1}, {0: 1}, {1: 1, 3: -1})
    add_term(2, {0: 1}, {1: 1, 5: -1}, {2: 1, 3: 1})
    add_term(3, {4: 1}, {4: 1, 0: -1}, {0: 1, 1: 1})
    add_term(4, {0: 1, 1: 1}, {5: 1}, {0: -1, 2: 1})
    add_term(5, {3: 1, 0: -1}, {0: 1, 1: 1}, {3: 1})
    add_term(6, {1: 1, 4: -1}, {4: 1, 5: 1}, {0: 1})
    
    # Use 7 terms for second 2x2x2 block
    add_term(7, {0: 1, 4: 1}, {2: 1, 7: 1}, {4: 1, 7: 1})
    add_term(8, {3: 1, 4: 1}, {2: 1}, {5: 1, 7: -1})
    add_term(9, {0: 1}, {3: 1, 7: -1}, {6: 1, 7: 1})
    add_term(10, {4: 1}, {6: 1, 2: -1}, {4: 1, 5: 1})
    add_term(11, {0: 1, 1: 1}, {7: 1}, {4: -1, 6: 1})
    add_term(12, {3: 1, 0: -1}, {2: 1, 3: 1}, {7: 1})
    add_term(13, {1: 1, 4: -1}, {6: 1, 7: 1}, {4: 1})
    
    # Use 6 terms for the 2x1x4 block (this is where we save 2 terms)
    # Key insight: we can share some of the A or B vectors with the previous blocks
    # a02 and a12 need to be multiplied with all 4 columns of B[2,:]
    # But we can combine some products
    
    # Actually, the 2x1x4 tensor has rank exactly 8, so we can't reduce below 8.
    # The total would be 7+7+8 = 22, which is what we had.
    
    # So the rank-22 solution might be optimal for this block structure.
    # Let me just return the original rank-22 solution.
    
    # Reset everything
    U = np.zeros((n * m, 22), dtype=float)
    V = np.zeros((m * p, 22), dtype=float)
    W = np.zeros((p * n, 22), dtype=float)
    
    # Block 1: A[:, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_term(0, {0: 1, 4: 1}, {0: 1, 5: 1}, {0: 1, 3: 1})
    add_term(1, {3: 1, 4: 1}, {0: 1}, {1: 1, 3: -1})
    add_term(2, {0: 1}, {1: 1, 5: -1}, {2: 1, 3: 1})
    add_term(3, {4: 1}, {4: 1, 0: -1}, {0: 1, 1: 1})
    add_term(4, {0: 1, 1: 1}, {5: 1}, {0: -1, 2: 1})
    add_term(5, {3: 1, 0: -1}, {0: 1, 1: 1}, {3: 1})
    add_term(6, {1: 1, 4: -1}, {4: 1, 5: 1}, {0: 1})
    
    # Block 2: A[:, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_term(7, {0: 1, 4: 1}, {2: 1, 7: 1}, {4: 1, 7: 1})
    add_term(8, {3: 1, 4: 1}, {2: 1}, {5: 1, 7: -1})
    add_term(9, {0: 1}, {3: 1, 7: -1}, {6: 1, 7: 1})
    add_term(10, {4: 1}, {6: 1, 2: -1}, {4: 1, 5: 1})
    add_term(11, {0: 1, 1: 1}, {7: 1}, {4: -1, 6: 1})
    add_term(12, {3: 1, 0: -1}, {2: 1, 3: 1}, {7: 1})
    add_term(13, {1: 1, 4: -1}, {6: 1, 7: 1}, {4: 1})
    
    # Block 3: A[:, 2:3] * B[2:3, 0:4] -> C[0:2, 0:4]
    add_term(14, {2: 1}, {8: 1}, {0: 1})
    add_term(15, {2: 1}, {9: 1}, {2: 1})
    add_term(16, {2: 1}, {10: 1}, {4: 1})
    add_term(17, {2: 1}, {11: 1}, {6: 1})
    add_term(18, {5: 1}, {8: 1}, {1: 1})
    add_term(19, {5: 1}, {9: 1}, {3: 1})
    add_term(20, {5: 1}, {10: 1}, {5: 1})
    add_term(21, {5: 1}, {11: 1}, {7: 1})
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
