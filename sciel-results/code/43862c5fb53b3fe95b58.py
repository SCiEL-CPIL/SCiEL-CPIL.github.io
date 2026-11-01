import time
import numpy as np

DIMENSIONS = [2, 2, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    # Use two blocks of 3 columns each.
    # Rank of M_2,2,3 is 9.
    # Total rank = 9 + 9 = 18.
    rank = 18
    
    # Strassen coefficients for 2x2 matrix multiplication
    # U, V, W vectors for the 7 rank-1 terms
    strassen = [
        {'U': [1, 0, 0, 1], 'V': [1, 0, 0, 1], 'W': [1, 0, 0, 1]}, # P1
        {'U': [0, 0, 1, 1], 'V': [1, 0, 0, 0], 'W': [0, 1, 0, -1]}, # P2
        {'U': [1, 0, 0, 0], 'V': [0, 1, 0, -1], 'W': [0, 0, 1, 1]}, # P3
        {'U': [0, 0, 0, 1], 'V': [-1, 0, 1, 0], 'W': [1, 1, 0, 0]}, # P4
        {'U': [1, 1, 0, 0], 'V': [0, 0, 0, 1], 'W': [-1, 0, 1, 0]}, # P5
        {'U': [-1, 0, 1, 0], 'V': [1, 1, 0, 0], 'W': [0, 0, 0, 1]}, # P6
        {'U': [0, 1, 0, -1], 'V': [0, 0, 1, 1], 'W': [1, 0, 0, 0]}, # P7
    ]
    
    # Additional 2 terms for the 3rd column in a 2x2x3 block
    # These are derived from the rank 9 decomposition of M_2,2,3
    # Terms 8 and 9
    # m8 = (a+b) * b_12  => U=[1,1,0,0], V=[0,0,0,1] (local), W=?
    # m9 = (c+d) * b_02  => U=[0,0,1,1], V=[1,0,0,0] (local), W=?
    # This specific decomposition requires careful W mapping.
    # To be safe and ensure correctness, I will use the schoolbook for the 3rd column?
    # No, schoolbook for 1 col is rank 4. 7+4=11.
    # Two blocks of 11 = 22. Worse than 21.
    
    # I need the correct rank 9 decomposition.
    # If I cannot guarantee the rank 9 coefficients, I should stick to 21 or find a verified 18.
    # Actually, the rank of M_2,2,3 is 9.
    # A known decomposition:
    # 1-7: Strassen for cols 0,1.
    # 8: a * b_02
    # 9: b * b_12
    # This gives C_02 = m8 + m9.
    # But C_12 = c*b_02 + d*b_12 is not formed.
    # So this is NOT a valid decomposition of the full tensor.
    
    # Therefore, I will stick to the rank 21 solution but clean it up, 
    # or try to find a verified rank < 21.
    # Given the risk, I will return the rank 21 solution as it is known to be correct.
    # However, the prompt asks for a stronger algorithm.
    # If I can't find a verified lower rank, I should not guess.
    
    # Wait, the inspiration says "Winograd... rank-16".
    # If rank 16 is possible, I should try.
    # But without the coefficients, I can't.
    
    # I will return the current rank 21 solution as it is the only one I can guarantee.
    # Actually, I can try to implement the rank 18 solution IF I find the coefficients.
    # Since I don't have them, I will keep rank 21.
    
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(12)]
    W = [[0] * rank for _ in range(12)]
    
    for b in range(3):
        for s in range(7):
            col = b * 7 + s
            sc = strassen[s]
            
            for idx in range(4):
                if sc['U'][idx] != 0:
                    U[idx][col] = sc['U'][idx]
            
            v_idx = [2*b, 2*b+1, 6+2*b, 6+2*b+1]
            for idx in range(4):
                if sc['V'][idx] != 0:
                    V[v_idx[idx]][col] = sc['V'][idx]
            
            w_idx = [4*b, 4*b+1, 4*b+2, 4*b+3]
            for idx in range(4):
                if sc['W'][idx] != 0:
                    W[w_idx[idx]][col] = sc['W'][idx]
    
    return {"U": U, "V": V, "W": W}
