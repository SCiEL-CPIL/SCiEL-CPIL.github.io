import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115
    
    def check_333(U, V, W):
        # U, V, W are 9xr
        r = U.shape[1]
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # Sum over alpha
                    s = 0
                    for alpha in range(r):
                        s += U[i*3+j, alpha] * V[j*3+k, alpha] * W[k*3+i, alpha]
                    if s != 1:
                        return False
        return True

    def find_333():
        # Try to find rank 23
        r = 23
        for restart in range(200):
            if time.time() > deadline:
                break
            U = np.random.randint(-2, 3, (9, r))
            V = np.random.randint(-2, 3, (9, r))
            W = np.random.randint(-2, 3, (9, r))
            
            # Local search
            for _ in range(50):
                if time.time() > deadline:
                    break
                if check_333(U, V, W):
                    return U, V, W
                
                # Pick a random equation (i,j,k) that is wrong
                errs = []
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            s = 0
                            for alpha in range(r):
                                s += U[i*3+j, alpha] * V[j*3+k, alpha] * W[k*3+i, alpha]
                            if s != 1:
                                errs.append((i, j, k, s))
                
                if not errs:
                    return U, V, W
                
                i, j, k, s = random.choice(errs)
                # Flip a random entry in U[i,j,:], V[j,k,:], or W[k,i,:]
                choice = random.randint(0, 2)
                if choice == 0:
                    alpha = random.randint(0, r-1)
                    U[i*3+j, alpha] += random.choice([-1, 1])
                elif choice == 1:
                    alpha = random.randint(0, r-1)
                    V[j*3+k, alpha] += random.choice([-1, 1])
                else:
                    alpha = random.randint(0, r-1)
                    W[k*3+i, alpha] += random.choice([-1, 1])
        
        return None

    U333, V333, W333 = find_333()
    
    if U333 is not None:
        # Use 333 part (rank 23) and 332 part (rank 18 schoolbook)
        # Total rank 41
        # Construct full matrices
        rank = 23 + 18
        U = [[0] * rank for _ in range(9)]
        V = [[0] * rank for _ in range(15)]
        W = [[0] * rank for _ in range(15)]
        
        # Fill 333 part (first 23 columns)
        for i in range(3):
            for j in range(3):
                for alpha in range(23):
                    U[i*3+j][alpha] = int(U333[i*3+j, alpha])
                    V[j*3+k][alpha] = int(V333[j*3+k, alpha]) # Wait, V index for 333 is j*3+k
                    # But in full 335, V index is j*5+k.
                    # The 333 part uses k in 0..2.
                    # So V[j*3+k] in 333 corresponds to V[j*5+k] in 335?
                    # Yes, if we map k=0,1,2 to k=0,1,2.
                    W[k*3+i][alpha] = int(W333[k*3+i, alpha])
                    # W index in 333 is k*3+i. In 335 it is k*5+i.
                    # So W[k*5+i] = W333[k*3+i].
        
        # We need to be careful with indices.
        # 333 part: i,j,k in 0..2.
        # U[i*3+j] -> U[i*3+j] in 9x23.
        # V[j*3+k] -> V[j*5+k] in 15x23. (k=0,1,2)
        # W[k*3+i] -> W[k*5+i] in 15x23. (k=0,1,2)
        
        # 332 part: i,j in 0..2, k in 0..1 (for the 2 columns of B2)
        # We use schoolbook for 332.
        # Rank 18.
        col_offset = 23
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    U[i*3+j][col_offset] = 1
                    V[j*5+k][col_offset] = 1
                    W[k*5+i][col_offset] = 1
                    col_offset += 1
        
        return {"U": U, "V": V, "W": W}
    else:
        # Fall back to schoolbook
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
