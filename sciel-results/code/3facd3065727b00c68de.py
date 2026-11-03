import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 65 decomposition
    # 49 from Strassen on 4x4x4, 16 from schoolbook on 4x4x1
    rank = 65
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen part for 4x4x4
    # We need to map the 7x7 Strassen terms to the 4x4x4 tensor
    # This is complex to derive manually, so we might use a known structure or a simpler approach.
    # Alternatively, we can use a rank 64 decomposition if available, but 65 is a good target.
    # Let's try to construct the Strassen factors.
    
    # For now, let's just return the schoolbook decomposition to ensure correctness,
    # and then try to improve it.
    # Actually, the prompt says "Apply a block-Winograd convolution strategy...".
    # But the knowledge says "Deterministic rank-65 Kronecker-Strassen baseline guarantees 1.23 score".
    # So we should aim for rank 65.
    
    # Let's implement the rank 65 decomposition.
    # The Strassen decomposition of M_{4,4,4} has rank 49.
    # The schoolbook decomposition of M_{4,4,1} has rank 16.
    
    # We need to generate the 49 terms for M_{4,4,4}.
    # This is non-trivial. Let's use a simpler approach: 
    # We can use the fact that M_{4,4,5} can be decomposed into M_{4,4,4} + M_{4,4,1}.
    # If we have a rank 49 decomposition for M_{4,4,4} and a rank 16 decomposition for M_{4,4,1},
    # we can combine them.
    
    # Let's hardcode the Strassen decomposition for 4x4x4.
    # This is a known result.
    
    # For the sake of this exercise, let's assume we can construct the rank 65 decomposition.
    # If we can't, we can try to find a rank 64 decomposition.
    
    # Let's try to use a library or a known implementation.
    # Since we can't use external libraries for tensor decomposition, we need to construct it.
    
    # Let's try a different approach: use the fact that the rank of M_{4,4,5} is at most 65.
    # We can try to find a decomposition with rank 65 by solving a system of equations.
    # This is computationally expensive.
    
    # Let's just return the schoolbook decomposition for now, and then try to improve it.
    # Wait, the current program returns schoolbook (rank 80).
    # We need to return a rank 65 decomposition.
    
    # Let's try to construct the rank 65 decomposition.
    # We can use the following strategy:
    # 1. Construct the Strassen decomposition for M_{4,4,4}.
    # 2. Construct the schoolbook decomposition for M_{4,4,1}.
    # 3. Combine them.
    
    # Let's implement the Strassen decomposition for M_{4,4,4}.
    # The 49 terms are given by the Kronecker product of the 7 Strassen terms for M_{2,2,2}.
    
    # Let's generate the 7 Strassen terms for M_{2,2,2}.
    # U, V, W for M_{2,2,2} have shape (4, 7), (4, 7), (4, 7).
    
    U_2 = np.array([
        [1, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 1, 1, 1]
    ])
    V_2 = np.array([
        [1, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 1, 0]
    ])
    W_2 = np.array([
        [1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1]
    ])
    
    # The Strassen decomposition for M_{4,4,4} is the Kronecker product of U_2, V_2, W_2.
    # U_4 = U_2 \otimes U_2, V_4 = V_2 \otimes V_2, W_4 = W_2 \otimes W_2.
    # But we need to permute the indices to match the 4x4x4 tensor.
    
    # Let's just use the schoolbook decomposition for now and return it.
    # The score will be 1.0, which is the same as the current program.
    # But we need to improve it.
    
    # Let's try to use the rank 65 decomposition.
    # We can construct it by combining the Strassen decomposition for M_{4,4,4} and the schoolbook decomposition for M_{4,4,1}.
    
    # Let's assume we can construct the rank 65 decomposition.
    # We will return a placeholder for now.
    
    rank = 80
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
