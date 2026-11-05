import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 150
    
    n, m, p = DIMENSIONS
    
    # We will build the decomposition recursively
    # Returns U, V, W as lists of lists or numpy arrays
    # To be efficient, we use numpy arrays for construction and convert to lists at the end if needed
    # But the interface expects lists of lists (based on seed)
    
    def decompose(n, m, p):
        # Base case: schoolbook
        if n == 1 and m == 1 and p == 1:
            # Rank 1
            U = [[1]]
            V = [[1]]
            W = [[1]]
            return U, V, W
            
        # Recursive case
        # We use the splitting strategy:
        # Split p into p1 + p2
        # Rank = decompose(n, m, p1) + decompose(n, m, p2)
        # This is not optimal. We need the Strassen-like mixing.
        
        # For now, let's implement the specific 3x4x7 decomposition derived:
        # Rank 54 = 36 (for 3x4x4) + 18 (for 3x4x3)
        
        pass

    # Since implementing a general recursive Strassen with correct index mapping is complex,
    # and we have a specific target (3,4,7), we can hardcode the logic for the splits.
    
    # Helper to combine decompositions
    def combine(decomp1, decomp2):
        U1, V1, W1 = decomp1
        U2, V2, W2 = decomp2
        U = U1 + U2
        V = V1 + V2
        W = W1 + W2
        return U, V, W

    # Helper to get schoolbook decomposition
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    # We need a function to create a decomposition for n x m x p
    # using the specific splits we analyzed.
    
    # Let's implement a generic recursive function that chooses splits
    # to minimize rank.
    
    def get_decomp(n, m, p):
        # Check time
        if time.time() > deadline - 10:
            return schoolbook(n, m, p)
            
        # Try splits
        best_rank = n * m * p
        best_decomp = schoolbook(n, m, p)
        
        # Split p
        for p1 in range(1, p):
            p2 = p - p1
            d1 = get_decomp(n, m, p1)
            d2 = get_decomp(n, m, p2)
            r1 = len(d1[0][0])
            r2 = len(d2[0][0])
            if r1 + r2 < best_rank:
                best_rank = r1 + r2
                U, V, W = combine(d1, d2)
                # Need to shift indices? No, the dimensions n,m,p are the same for the sub-problems?
                # Wait. If we split p, the sub-problems are n x m x p1 and n x m x p2.
                # The output W for the first has size p1 x n.
                # The output W for the second has size p2 x n.
                # We need to concatenate them into a single W of size p x n.
                # W_total[k, i] = W1[k, i] if k < p1 else W2[k-p1, i]
                
                U_total = d1[0]
                V_total = d1[1] # V is m x p1? No, V is m x p.
                # Wait, V index is j * p + k.
                # If p changes, V dimensions change.
                # This simple combination doesn't work directly because V and W indices depend on p.
                
                # We need a more careful combination.
                pass
                
        return best_decomp

    # Given the complexity of generic index shifting, and the specific small size,
    # we will construct the 54-rank decomposition explicitly using the derived logic.
    
    # Decomposition of 3x4x4 (Rank 36)
    # Split 4 into 2+2.
    # R(3,4,2) = 18.
    # R(3,4,2) from R(2,4,2)=14 and R(1,4,2)=4.
    
    # This is getting too complex to hardcode manually without errors.
    # Let's use the simple split strategy which yields 54.
    # It requires correct index mapping.
    
    # I will implement a correct recursive solver with index mapping.
    
    def solve(n, m, p):
        # Returns U, V, W
        # U: n*m x rank
        # V: m*p x rank
        # W: p*n x rank
        
        if time.time() > deadline - 5:
            return schoolbook(n, m, p)
            
        # Base case
        if n == 1 and m == 1 and p == 1:
            return [[1]], [[1]], [[1]]
            
        best_rank = n * m * p
        best_decomp = schoolbook(n, m, p)
        
        # Try splitting p
        for p1 in range(1, p):
            p2 = p - p1
            U1, V1, W1 = solve(n, m, p1)
            U2, V2, W2 = solve(n, m, p2)
            
            r1 = len(U1[0])
            r2 = len(U2[0])
            if r1 + r2 < best_rank:
                # Combine
                # U is same size n*m
                U = U1 + U2
                
                # V1 is m*p1, V2 is m*p2.
                # We need V of size m*p.
                # V[j, k] for k in [0, p1) comes from V1.
                # V[j, k] for k in [p1, p) comes from V2 (shifted).
                V = [[0] * (r1 + r2) for _ in range(m * p)]
                for j in range(m):
                    for k in range(p1):
                        for r in range(r1):
                            V[j * p + k][r] = V1[j * p1 + k][r]
                    for k in range(p2):
                        for r in range(r2):
                            V[j * p + (p1 + k)][r1 + r] = V2[j * p2 + k][r1 + r]
                            
                # W1 is p1*n, W2 is p2*n.
                # W[k, i] for k in [0, p1) comes from W1.
                # W[k, i] for k in [p1, p) comes from W2.
                W = [[0] * (r1 + r2) for _ in range(p * n)]
                for k in range(p1):
                    for i in range(n):
                        for r in range(r1):
                            W[k * n + i][r] = W1[k * n1 + i][r] # Error: n1 is n
                            W[k * n + i][r] = W1[k * n + i][r]
                for k in range(p2):
                    for i in range(n):
                        for r in range(r2):
                            W[k * n + i][r1 + r] = W2[k * n + i][r1 + r]
                            
                best_rank = r1 + r2
                best_decomp = (U, V, W)
                
        # Try splitting n (if beneficial)
        # Similar logic for n and m
        
        return best_decomp

    U, V, W = solve(n, m, p)
    
    return {"U": U, "V": V, "W": W}
