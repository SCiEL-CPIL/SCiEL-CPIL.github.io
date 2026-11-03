import numpy as np
import random
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Try to find a decomposition for M_{3,3,2} with rank 5
    # If found, combine with M_{3,3,3} (rank 23) to get rank 33
    # M_{3,3,7} = M_{3,3,3} + M_{3,3,4} = M_{3,3,3} + 2 * M_{3,3,2}
    
    def find_decomp_332():
        # n=3, m=3, p=2
        n2, m2, p2 = 3, 3, 2
        rank2 = 5
        # U: 9x5, V: 6x5, W: 6x5
        # Indices:
        # U[i*m2+j, c]
        # V[j*p2+k, c]
        # W[k*n2+i, c]
        
        # Random search
        for _ in range(100000):
            if time.time() > deadline:
                return None
            
            # Initialize with random small integers
            U = np.array(random.choices([-1, 0, 1], k=n2*m2*rank2)).reshape(n2*m2, rank2)
            V = np.array(random.choices([-1, 0, 1], k=m2*p2*rank2)).reshape(m2*p2, rank2)
            W = np.array(random.choices([-1, 0, 1], k=p2*n2*rank2)).reshape(p2*n2, rank2)
            
            # Check constraints
            valid = True
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        s = 0
                        for c in range(rank2):
                            s += U[i*m2+j, c] * V[j*p2+k, c] * W[k*n2+i, c]
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                return U, V, W
        return None

    # Hardcoded rank 23 decomposition for M_{3,3,3}
    # This is a known result. I will use a placeholder structure 
    # and fill it with a valid decomposition if I can recall one, 
    # otherwise I will fall back to schoolbook for that part.
    # For now, I will use schoolbook for M_{3,3,3} (rank 27) 
    # if I can't find the rank 23 one quickly.
    # Actually, 27 is the schoolbook rank.
    
    # Let's try to find a decomposition for M_{3,3,3} with rank < 27.
    # If I find rank 23, great.
    
    # For simplicity and reliability within time, I will try to find 
    # a decomposition for the whole M_{3,3,7} with rank < 63.
    
    # Strategy:
    # 1. Try to find a decomposition for M_{3,3,2} with rank 5.
    # 2. If found, use it to build M_{3,3,4} with rank 10.
    # 3. Use schoolbook for M_{3,3,3} (rank 27).
    # Total rank = 27 + 10 = 37.
    # Score = 63 / 37 = 1.7.
    
    # Can we do better?
    # Try to find a decomposition for M_{3,3,3} with rank 23.
    # I will not hardcode it due to risk of error.
    
    # Let's try to find a decomposition for M_{3,3,7} directly with rank 30?
    # Unlikely to find by random search.
    
    # I will implement the combination of M_{3,3,3} (schoolbook 27) and M_{3,3,4} (2 * M_{3,3,2}).
    
    U_332, V_332, W_332 = find_decomp_332()
    
    if U_332 is not None:
        # Construct M_{3,3,4} using two copies of M_{3,3,2}
        # M_{3,3,4} = M_{3,3,2} (cols 0,1) + M_{3,3,2} (cols 2,3)
        # U, V, W for M_{3,3,4}
        # U: 9x10, V: 6x10, W: 12x10
        U_334 = np.zeros((9, 10))
        V_334 = np.zeros((6, 10))
        W_334 = np.zeros((12, 10))
        
        # Copy 1: k=0,1
        U_334[:, 0:5] = U_332
        V_334[:, 0:5] = V_332
        W_334[0:6, 0:5] = W_332  # k=0,1 -> indices 0..5 in W (k*n+i)
        
        # Copy 2: k=2,3
        # Need to shift k index by 2
        # V index: j*p2+k. For k=2,3, p2=2? No, p=4.
        # Wait, M_{3,3,2} has p=2.
        # For M_{3,3,4}, p=4.
        # The second copy uses k=2,3.
        # V index for k=2: j*4+2.
        # V index for k=3: j*4+3.
        # In the M_{3,3,2} decomposition, V index was j*2+k.
        # So we need to map j*2+0 -> j*4+2, j*2+1 -> j*4+3.
        
        for j in range(3):
            for k in range(2):
                V_334[j*4+2+k, 5:10] = V_332[j*2+k, 0:5]
        
        # W index: k*n+i.
        # For k=2,3, n=3.
        # W index for k=2: 2*3+i = 6+i.
        # W index for k=3: 3*3+i = 9+i.
        # In M_{3,3,2}, W index was k*3+i.
        # So we map k=0 -> 6+i, k=1 -> 9+i.
        for k in range(2):
            for i in range(3):
                W_334[(2+k)*3+i, 5:10] = W_332[k*3+i, 0:5]
        
        # U is the same for both copies?
        # U index: i*3+j.
        # Yes, U is the same.
        U_334[:, 5:10] = U_332
        
        # Now combine M_{3,3,3} (schoolbook) and M_{3,3,4}
        # M_{3,3,3} schoolbook rank 27.
        U_333 = np.zeros((9, 27))
        V_333 = np.zeros((9, 27))
        W_333 = np.zeros((9, 27))
        
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U_333[i*3+j, col] = 1
                    V_333[j*3+k, col] = 1
                    W_333[k*3+i, col] = 1
                    col += 1
        
        # Total rank = 27 + 10 = 37
        U_total = np.hstack([U_333, U_334])
        V_total = np.hstack([V_333, V_334])
        W_total = np.hstack([W_333, W_334])
        
        # Convert to list of lists
        U_list = U_total.tolist()
        V_list = V_total.tolist()
        W_list = W_total.tolist()
        
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
