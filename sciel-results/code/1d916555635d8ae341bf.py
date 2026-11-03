"""Kronecker-style decomposition of 3x5x6 matrix multiplication using Kronecker product of minimal-rank bases."""
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 5, 6
    
    # We'll use the Kronecker product approach:
    # Decompose m=5 as 2*3 (though 5 is not 2*3, we need to be careful)
    # Actually, 5 = 2+3, not a product. Let me reconsider.
    
    # The directive says: "representing the 5-dimensional inner product index as a tensor product 
    # of dimensions 2 and 3, and the 6-dimensional output index as 2 and 3"
    # But 2*3 = 6, not 5. So this doesn't directly work for m=5.
    
    # Let me re-read: "Kronecker product of minimal-rank bases" strategy
    # Maybe we decompose the multiplication tensor as a sum of Kronecker products?
    
    # Actually, for matrix multiplication of size n x m by m x p, the tensor has rank at most n*m*p.
    # We can try to find a lower rank decomposition.
    
    # Known results:
    # - 2x2x2: rank 7 (Strassen)
    # - 3x3x3: rank 19 (Laderman)
    # - 2x3x3: rank 12
    # - 3x2x3: rank 12
    # - 3x3x2: rank 12
    # - 2x2x3: rank 12
    # - 2x3x2: rank 12
    # - 3x2x2: rank 7 (same as 2x2x2 with padding? No, 3x2x2 should be rank 7)
    # - 2x2x4: rank 12
    # - 4x2x2: rank 12
    # - 2x4x2: rank 12
    # - 4x4x4: rank 26? or 27?
    
    # For 3x5x6, let's try to decompose as:
    # m = 5 = 2 + 3
    # p = 6 = 2 + 4
    # So we have 4 blocks:
    # Block (1,1): 3x2x2 -> rank 7
    # Block (1,2): 3x2x4 -> rank 12
    # Block (2,1): 3x3x2 -> rank 12
    # Block (2,2): 3x3x4 -> rank 19 (using Laderman for the 3x3x3 part and schoolbook for the extra dimension?)
    
    # Actually, 3x3x4: we can think of this as 3x3x3 + 3x3x1, but that's not how tensor rank works.
    # The rank of 3x3x4 is at most 19+? No, it's a single tensor.
    
    # Let me look up or derive the rank of 3x3x4.
    # Actually, I think the rank of 3x3x4 is 19, same as 3x3x3, because we can embed.
    # No, that's not right either.
    
    # Let me just use a known decomposition or construct one carefully.
    
    # For now, let me try a different approach: use the fact that we can compute the rank
    # by finding a decomposition algorithmically, but that's hard.
    
    # Let me try to use Strassen for the 2x2x2 block and Laderman for the 3x3x3 block,
    # and handle the mixed blocks with schoolbook or known decompositions.
    
    # Actually, let me try a simpler approach first: just use schoolbook but verify correctness.
    # Then I'll try to optimize.
    
    rank = n * m * p  # 90
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
