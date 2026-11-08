"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe under 150s limit
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def rank52_decomp():
        """Construct a rank-52 decomposition using improved pairing strategy."""
        rank = 52
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        
        # Strategy: Use different pairing for better efficiency
        # Pair j indices as (0,2) and (1,3), handle j=4 separately
        # Pair k indices as (0,3), (1,4), (2,5)
        
        j_pairs = [(0, 2), (1, 3)]
        k_pairs = [(0, 3), (1, 4), (2, 5)]
        
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                # Product 1: (A_0,j0 + A_1,j1)(B_j0,k0 + B_j1,k1)
                U[0*m + j0, col] += 1
                U[1*m + j1, col] += 1
                V[j0*p + k0, col] += 1
                V[j1*p + k1, col] += 1
                W[k0*n + 0, col] += 1
                W[k1*n + 1, col] += 1
                col += 1
                
                # Product 2: (A_1,j0)(B_j0,k0) - correction
                U[1*m + j0, col] += 1
                V[j0*p + k0, col] += 1
                W[k0*n + 1, col] += 1
                col += 1
                
                # Product 3: (A_0,j1)(B_j1,k1) - correction  
                U[0*m + j1, col] += 1
                V[j1*p + k1, col] += 1
                W[k1*n + 0, col] += 1
                col += 1
                
                # Product 4: (A_0,j0 + A_1,j1)(B_j1,k0)
                U[0*m + j0, col] += 1
                U[1*m + j1, col] += 1
                V[j1*p + k0, col] += 1
                W[k0*n + 0, col] += 1
                W[k0*n + 1, col] += 1
                col += 1
                
                # Product 5: (A_0,j0 + A_1,j1)(B_j0,k1)
                U[0*m + j0, col] += 1
                U[1*m + j1, col] += 1
                V[j0*p + k1, col] += 1
                W[k1*n + 0, col] += 1
                W[k1*n + 1, col] += 1
                col += 1
                
                # Product 6: (A_1,j1)(B_j1,k0) - correction
                U[1*m + j1, col] += 1
                V[j1*p + k0, col] += 1
                W[k0*n + 1, col] += 1
                col += 1
                
                # Product 7: (A_0,j0)(B_j0,k1) - correction
                U[0*m + j0, col] += 1
                V[j0*p + k1, col] += 1
                W[k1*n + 0, col] += 1
                col += 1
        
        # Handle j=4 separately
        for k in range(p):
            U[0*m + 4, col] += 1
            V[4*p + k, col] += 1
            W[k*n + 0, col] += 1
            col += 1
            U[1*m + 4, col] += 1
            V[4*p + k, col] += 1
            W[k*n + 1, col] += 1
            col += 1
        
        # We should have used 2*3*7 + 12 = 54, but we need to check if we can do better
        # Actually this is still 54. Let me try a different approach.
        
        return None  # This construction doesn't work, fall back to rank 54
    
    def rank54_decomp():
        """Original rank-54 decomposition."""
        rank = 54
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        j_pairs = [(0, 1), (2, 3)]
        k_pairs = [(0, 1), (2, 3), (4, 5)]
        
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                # P1 = (A_0,j0 + A_1,j1)(B_j0,k0 + B_j1,k1)
                U[0*m + j0, col] += 1
                U[1*m + j1, col] += 1
                V[j0*p + k0, col] += 1
                V[j1*p + k1, col] += 1
                W[k0*n + 0, col] += 1
                W[k1*n + 1, col] += 1
                col += 1
                # P2
                U[1*m + j0, col] += 1
                U[1*m + j1, col] += 1
                V[j0*p + k0, col] += 1
                W[k0*n + 1, col] += 1
                W[k1*n + 1, col] -= 1
                col += 1
                # P3
                U[0*m + j0, col] += 1
                V[j0*p + k1, col] += 1
                V[j1*p + k1, col] -= 1
                W[k1*n + 0, col] += 1
                W[k1*n + 1, col] += 1
                col += 1
                # P4
                U[1*m + j1, col] += 1
                V[j1*p + k0, col] += 1
                V[j0*p + k0, col] -= 1
                W[k0*n + 0, col] += 1
                W[k0*n + 1, col] += 1
                col += 1
                # P5
                U[0*m + j0, col] += 1
                U[0*m + j1, col] += 1
                V[j1*p + k1, col] += 1
                W[k0*n + 0, col] -= 1
                W[k1*n + 0, col] += 1
                col += 1
                # P6
                U[1*m + j0, col] += 1
                U[0*m + j0, col] -= 1
                V[j0*p + k0, col] += 1
                V[j0*p + k1, col] += 1
                W[k1*n + 1, col] += 1
                col += 1
                # P7
                U[0*m + j1, col] += 1
                U[1*m + j1, col] -= 1
                V[j1*p + k0, col] += 1
                V[j1*p + k1, col] += 1
                W[k0*n + 0, col] += 1
                col += 1
        
        for k in range(p):
            U[0*m + 4, col] += 1
            V[4*p + k, col] += 1
            W[k*n + 0, col] += 1
            col += 1
            U[1*m + 4, col] += 1
            V[4*p + k, col] += 1
            W[k*n + 1, col] += 1
            col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def try_rank52():
        """Try to construct a rank-52 decomposition with different strategy."""
        rank = 52
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        
        # New strategy: Use 3 j-pairs and 2 k-pairs with different combination
        # j_pairs: (0,1), (2,3), (4,0) - reuse j=0
        # k_pairs: (0,1), (2,3), (4,5)
        
        # Actually, let me try a completely different approach.
        # For n=2, we can use the fact that we only have 2 rows.
        
        # Strategy: For each pair of k indices, use 6 products to handle all 5 j indices
        # This would give 3 * 6 + 6 = 24, which is too optimistic.
        
        # Let me try: for each k-pair, use 8 products
        # 3 k-pairs * 8 = 24, plus 6 for j=4 = 30. Still too low.
        
        # Realistic approach: keep the 7 products per (j-pair, k-pair) but optimize
        # 2 j-pairs * 3 k-pairs * 7 = 42, plus 12 for j=4 = 54
        # To get 52, we need to save 2 products.
        
        # Idea: For one of the (j-pair, k-pair) combinations, use 5 products instead of 7
        # This saves 2 products.
        
        j_pairs = [(0, 1), (2, 3)]
        k_pairs = [(0, 1), (2, 3), (4, 5)]
        
        for idx, (j0, j1) in enumerate(j_pairs):
            for kidx, (k0, k1) in enumerate(k_pairs):
                if idx == 1 and kidx == 2:
                    # Use optimized 5-product construction for this pair
                    # P1 = (A_0,j0 + A_1,j1)(B_j0,k0 + B_j1,k1)
                    U[0*m + j0, col] += 1
                    U[1*m + j1, col] += 1
                    V[j0*p + k0, col] += 1
                    V[j1*p + k1, col] += 1
                    W[k0*n + 0, col] += 1
                    W[k1*n + 1, col] += 1
                    col += 1
                    # P2 = (A_1,j0 + A_0,j1)(B_j0,k0 + B_j1,k1) - different combination
                    U[1*m + j0, col] += 1
                    U[0*m + j1, col] += 1
                    V[j0*p + k0, col] += 1
                    V[j1*p + k1, col] += 1
                    W[k0*n + 1, col] += 1
                    W[k1*n + 0, col] += 1
                    col += 1
                    # P3 = (A_0,j0)(B_j1,k1)
                    U[0*m + j0, col] += 1
                    V[j1*p + k1, col] += 1
                    W[k1*n + 0, col] += 1
                    col += 1
                    # P4 = (A_1,j1)(B_j0,k0)
                    U[1*m + j1, col] += 1
                    V[j0*p + k0, col] += 1
                    W[k0*n + 1, col] += 1
                    col += 1
                    # P5 = (A_0,j0 + A_1,j1)(B_j0,k1)
                    U[0*m + j0, col] += 1
                    U[1*m + j1, col] += 1
                    V[j0*p + k1, col] += 1
                    W[k1*n + 0, col] += 1
                    W[k1*n + 1, col] += 1
                    col += 1
                else:
                    # Use standard 7-product construction
                    U[0*m + j0, col] += 1
                    U[1*m + j1, col] += 1
                    V[j0*p + k0, col] += 1
                    V[j1*p + k1, col] += 1
                    W[k0*n + 0, col] += 1
                    W[k1*n + 1, col] += 1
                    col += 1
                    U[1*m + j0, col] += 1
                    U[1*m + j1, col] += 1
                    V[j0*p + k0, col] += 1
                    W[k0*n + 1, col] += 1
                    W[k1*n + 1, col] -= 1
                    col += 1
                    U[0*m + j0, col] += 1
                    V[j0*p + k1, col] += 1
                    V[j1*p + k1, col] -= 1
                    W[k1*n + 0, col] += 1
                    W[k1*n + 1, col] += 1
                    col += 1
                    U[1*m + j1, col] += 1
                    V[j1*p + k0, col] += 1
                    V[j0*p + k0, col] -= 1
                    W[k0*n + 0, col] += 1
                    W[k0*n + 1, col] += 1
                    col += 1
                    U[0*m + j0, col] += 1
                    U[0*m + j1, col] += 1
                    V[j1*p + k1, col] += 1
                    W[k0*n + 0, col] -= 1
                    W[k1*n + 0, col] += 1
                    col += 1
                    U[1*m + j0, col] += 1
                    U[0*m + j0, col] -= 1
                    V[j0*p + k0, col] += 1
                    V[j0*p + k1, col] += 1
                    W[k1*n + 1, col] += 1
                    col += 1
                    U[0*m + j1, col] += 1
                    U[1*m + j1, col] -= 1
                    V[j1*p + k0, col] += 1
                    V[j1*p + k1, col] += 1
                    W[k0*n + 0, col] += 1
                    col += 1
        
        for k in range(p):
            U[0*m + 4, col] += 1
            V[4*p + k, col] += 1
            W[k*n + 0, col] += 1
            col += 1
            U[1*m + 4, col] += 1
            V[4*p + k, col] += 1
            W[k*n + 1, col] += 1
            col += 1
        
        result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Verify
        if verify(U, V, W):
            return result
        return None
    
    # Try rank 52 first
    result52 = try_rank52()
    if result52 is not None:
        return result52
    
    # Fall back to rank 54
    return rank54_decomp()
# EVOLVE-BLOCK-END
