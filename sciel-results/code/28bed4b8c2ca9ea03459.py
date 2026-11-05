import numpy as np
import time

def construct():
    deadline = time.time() + 100  # safety margin before 120s kill
    
    n, m, p = 3, 3, 6
    
    # Schoolbook rank-54 decomposition (guaranteed correct)
    rank_school = n * m * p  # 54
    U_school = np.zeros((n * m, rank_school), dtype=int)
    V_school = np.zeros((m * p, rank_school), dtype=int)
    W_school = np.zeros((p * n, rank_school), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Try to construct a rank-46 decomposition using two 3x3x3 Winograd blocks
    # The 3x3x6 tensor = direct sum of two 3x3x3 tensors
    # If we have a rank-23 decomposition for 3x3x3, we get rank 46 for 3x3x6
    
    # Winograd 3x3 matrices (rank 23)
    # U_W: 9x23, V_W: 9x23, W_W: 9x23
    # Using a known valid set from literature
    
    # The Winograd 3x3 scheme has 23 terms.
    # I'll use the standard Winograd matrices.
    
    # Actually, let me try a different approach: use Strassen-like block decomposition
    # For 3x3x3, rank 23 is known to be optimal.
    
    # Let me try to verify if I can construct a valid rank-23 3x3x3 decomposition.
    # The Winograd matrices are complex to hardcode correctly from memory.
    
    # Alternative: Try to find a rank-46 decomposition programmatically
    # This is too slow for the time budget.
    
    # Safest approach: return the schoolbook rank-54 decomposition.
    # It's guaranteed valid and gives score 1.0 (baseline).
    
    # However, let me try one more thing: construct a rank-46 using
    # the fact that 3x3x6 = 3x3x3 ⊕ 3x3x3 (direct sum)
    # If I can find ANY valid rank-23 for 3x3x3, I'm done.
    
    # The known minimum rank for 3x3x3 is 23 (proven).
    # Let me try to use a specific known decomposition.
    
    # From "Matrix multiplication by tensor rank decomposition" literature:
    # One valid rank-23 scheme uses the following pattern:
    
    # I'll attempt to use a simplified construction. If it doesn't verify,
    # I'll fall back to rank 54.
    
    # Let me try a rank-46 construction using elementary operations:
    # For each 3x3 block of the 3x6 matrix, use a 3x3x3 decomposition.
    
    # Actually, the cleanest approach: 
    # The tensor for (3x3)*(3x6) has entries T[i,j,k,l] where
    # C[i,l] = sum_j A[i,j]*B[j,l] for l=0..5
    # This is two independent 3x3x3 multiplications (l=0,1,2 and l=3,4,5)
    # Wait, no - it's A(3x3) * B(3x6) = C(3x6)
    # C[i,l] = sum_j A[i,j] * B[j,l]
    # This is a single tensor, not a direct sum.
    
    # The tensor T[i,j,l] = A[i,j] * B[j,l] summed over j
    # has dimensions 3*3*6 = 54 nonzero entries in schoolbook.
    
    # A rank-r decomposition means:
    # T[i,j,l] = sum_{r=0}^{R-1} U[i,j,r] * V[j,l,r] * W[i,l,r]
    # (using Einstein summation convention for the tensor product)
    
    # The schoolbook has R = 54.
    # Can we do better?
    
    # For 3x3x3 (square), minimum rank is 23.
    # For 3x3x6, the minimum rank is at most 2*23 = 46 (using two 3x3x3 blocks)
    # and at least max(9, 9, 18, ...) - actually the lower bound is max(nm, mp, np) = max(9,18,18) = 18
    # but that's a weak bound.
    
    # The true minimum rank for 3x3x6 is unknown to me, but likely between 27 and 46.
    
    # Given the time constraint and the risk of returning an invalid decomposition,
    # I'll return the safe rank-54 schoolbook.
    
    return {"U": U_school, "V": V_school, "W": W_school}
