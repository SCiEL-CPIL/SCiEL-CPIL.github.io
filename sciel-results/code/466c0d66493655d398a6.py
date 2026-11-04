import random
import time
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Get sub-decompositions
    # M_232: 2x3 * 3x2 -> 2x2
    # M_231: 2x3 * 3x1 -> 2x1
    
    U_232, V_232, W_232 = _get_232_factors()
    rank_232 = len(U_232[0])
    
    # M_231 is derived from M_232 by setting the second column of B to 0
    # V_231 has shape (3, rank_232)
    # V_232 has shape (6, rank_232) -> rows 0,1,2 are B[:,0], rows 3,4,5 are B[:,1]
    # Wait, V index is j*p + k. For p=2, j in 0..2, k in 0..1.
    # V[j*2 + k].
    # To get M_231, we set V[j*2 + 1] = 0.
    V_231 = [row[:] for row in V_232]
    for j in range(3):
        for r in range(rank_232):
            V_231[j*2 + 1][r] = 0
            
    # W_231 is same as W_232?
    # W index is k*n + i. For p=2, k in 0..1, i in 0..1.
    # W[k*2 + i].
    # M_231 output is 2x1. So only k=0?
    # No, M_231 is 2x3 * 3x1 -> 2x1.
    # The output index is just i.
    # But the interface requires W to have shape (p*n, rank).
    # For M_231, p=1. So W shape (2, rank).
    # W_232 has shape (4, rank). W[k*2+i].
    # For M_231, we only care about k=0?
    # Actually, the tensor for M_231 is a sub-tensor of M_232?
    # M_232: C_{i,k} for k=0,1.
    # M_231: C_{i,0} for k=0?
    # Yes, if we set B[:,1]=0, then C[:,1]=0.
    # So we can just use W_232 and ignore the k=1 part?
    # But we need to return W of shape (1*2, rank) = (2, rank).
    # So we take W_232[0] and W_232[1] (which correspond to k=0, i=0 and k=0, i=1).
    # Wait, W index is k*n + i.
    # k=0, i=0 -> 0.
    # k=0, i=1 -> 1.
    # k=1, i=0 -> 2.
    # k=1, i=1 -> 3.
    # So W_231 = [W_232[0], W_232[1]].
    
    U_231 = U_232
    W_231 = [W_232[0], W_232[1]]
    
    # Now combine for 2x3x3 (cols 0,1,2)
    # Part 1: cols 0,1 (M_232)
    # Part 2: col 2 (M_231)
    
    # We need to map the indices to the full 2x3x3 problem.
    # U is same for all.
    # V for M_232 covers k=0,1.
    # V for M_231 covers k=2.
    # We need to construct V_233 of shape (3*3, rank_232 + rank_231) = (9, 2*rank_232).
    # Rank_231 is same as rank_232.
    
    U_233 = []
    for r in range(rank_232):
        U_233.append([U_232[i][r] for i in range(6)] + [U_231[i][r] for i in range(6)])
        
    V_233 = []
    # V index j*3 + k.
    # For M_232 (k=0,1): V_232 index j*2 + k.
    # For M_231 (k=2): V_231 index j*1 + 0 = j. (Since p=1 for 231)
    # Wait, V_231 is derived from V_232.
    # In V_232, the rows are j*2+k.
    # For k=2 in the full problem, we use the M_231 part.
    # The M_231 part uses V_231[j] which is V_232[j*2+0] (since we zeroed out k=1).
    # So for full V_233:
    # Row j*3+0 (k=0): from V_232[j*2+0] in first half of rank.
    # Row j*3+1 (k=1): from V_232[j*2+1] in first half of rank.
    # Row j*3+2 (k=2): from V_231[j] in second half of rank.
    
    for j in range(3):
        # k=0
        row0 = [V_232[j*2+0][r] for r in range(rank_232)] + [0]*rank_232
        # k=1
        row1 = [V_232[j*2+1][r] for r in range(rank_232)] + [0]*rank_232
        # k=2
        row2 = [0]*rank_232 + [V_231[j][r] for r in range(rank_232)]
        V_233.append(row0)
        V_233.append(row1)
        V_233.append(row2)
        
    W_233 = []
    # W index k*2 + i.
    # For M_232 (k=0,1): W_232 index k*2+i.
    # For M_231 (k=2): W_231 index i. (Since p=1 for 231, W index is just i? No, W index is k*n+i. For p=1, k=0 only. So W_231[i].)
    # But W_231 was derived as [W_232[0], W_232[1]].
    # So W_231[0] is W_232[0] (k=0, i=0).
    # W_231[1] is W_232[1] (k=0, i=1).
    # In the full problem, k=2 corresponds to i=0,1.
    # So W_233[2*2+0] (k=2, i=0) should be W_231[0] in second half.
    # W_233[2*2+1] (k=2, i=1) should be W_231[1] in second half.
    
    for k in range(3):
        for i in range(2):
            if k < 2:
                # From M_232
                val = [W_232[k*2+i][r] for r in range(rank_232)] + [0]*rank_232
            else:
                # From M_231 (k=2)
                # W_231 index is i.
                val = [0]*rank_232 + [W_231[i][r] for r in range(rank_232)]
            W_233.append(val)
            
    # Now for 2x3x4 (cols 3,4,5,6)
    # Split into two M_232: cols 3,4 and 5,6.
    # Same logic as above, but twice.
    
    U_234 = []
    for r in range(rank_232):
        U_234.append([U_232[i][r] for i in range(6)] + [U_232[i][r] for i in range(6)])
        
    V_234 = []
    for j in range(3):
        # k=3 (first 232, k=0)
        row3 = [V_232[j*2+0][r] for r in range(rank_232)] + [0]*rank_232
        # k=4 (first 232, k=1)
        row4 = [V_232[j*2+1][r] for r in range(rank_232)] + [0]*rank_232
        # k=5 (second 232, k=0)
        row5 = [0]*rank_232 + [V_232[j*2+0][r] for r in range(rank_232)]
        # k=6 (second 232, k=1)
        row6 = [0]*rank_232 + [V_232[j*2+1][r] for r in range(rank_232)]
        V_234.append(row3)
        V_234.append(row4)
        V_234.append(row5)
        V_234.append(row6)
        
    W_234 = []
    for k in range(4):
        for i in range(2):
            if k < 2:
                val = [W_232[k*2+i][r] for r in range(rank_232)] + [0]*rank_232
            else:
                val = [0]*rank_232 + [W_232[(k-2)*2+i][r] for r in range(rank_232)]
            W_234.append(val)
            
    # Combine 233 and 234
    U_full = []
    for r in range(rank_232):
        U_full.append(U_233[0][r:r+6] + U_234[0][r:r+6]) # This is wrong, U_233 and U_234 have different ranks?
        # U_233 rank is 2*rank_232.
        # U_234 rank is 2*rank_232.
        # So U_full rank is 4*rank_232.
        
    # Let's just concatenate the factors directly.
    U_full = []
    for r in range(2*rank_232):
        U_full.append(U_233[0][r] + U_234[0][r]) # No, U_233[0] is a row of U_233.
        
    # U_233 is 6 x (2*rank_232).
    # U_234 is 6 x (2*rank_232).
    # U_full is 6 x (4*rank_232).
    
    U_full = []
    for i in range(6):
        row = U_233[i] + U_234[i]
        U_full.append(row)
        
    V_full = []
    for j in range(3):
        for k in range(7):
            if k < 3:
                row = V_233[j*3+k] + V_234[j*4+k] # V_234 index j*4+k? No, V_234 has 4 cols.
                # V_233 index j*3+k.
                # V_234 index j*4+k.
                # But V_234 corresponds to k=3..6.
                # So for k=0,1,2, we use V_233.
                # For k=3,4,5,6, we use V_234.
                pass
            else:
                pass
                
    # This manual concatenation is error prone.
    # Let's use a helper to combine.
    
    U = []
    for i in range(6):
        U.append(U_233[i] + U_234[i])
        
    V = []
    for j in range(3):
        for k in range(7):
            if k < 3:
                V.append(V_233[j*3+k] + V_234[j*4+k]) # V_234 has zeros for k<3? No, V_234 is for k=3..6.
                # We need to pad V_234 with zeros for k<3?
                # No, V_234 only has entries for its own k.
                # So for k<3, V_234 part is 0.
                # For k>=3, V_233 part is 0.
                pass
            else:
                V.append(V_233[j*3+k] + V_234[j*4+k])
                
    # Correct approach:
    V = []
    for j in range(3):
        for k in range(7):
            row = [0] * (4*rank_232)
            if k < 3:
                for r in range(2*rank_232):
                    row[r] = V_233[j*3+k][r]
            else:
                for r in range(2*rank_232):
                    row[2*rank_232 + r] = V_234[j*4+k][r]
            V.append(row)
            
    W = []
    for k in range(7):
        for i in range(2):
            row = [0] * (4*rank_232)
            if k < 3:
                for r in range(2*rank_232):
                    row[r] = W_233[k*2+i][r]
            else:
                for r in range(2*rank_232):
                    row[2*rank_232 + r] = W_234[(k-3)*2+i][r]
            W.append(row)
            
    return {"U": U, "V": V, "W": W}

def _get_232_factors():
    # Rank 7 decomposition for 2x3x2
    # ...
    pass

# EVOLVE-BLOCK-END
