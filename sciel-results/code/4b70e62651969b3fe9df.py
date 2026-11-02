import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 2, 11]

def construct():
    start_time = time.time()
    DEADLINE = start_time + 110  # 110s safety margin within 120s
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 44
    
    # Schoolbook decomposition as fallback
    def make_schoolbook():
        rank = rank_schoolbook
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
    
    schoolbook_result = make_schoolbook()
    best_result = schoolbook_result
    best_rank = rank_schoolbook
    
    # Try to find a lower rank decomposition using random search + linear algebra
    # Strategy: try random rank-r candidates and check if the tensor can be represented
    
    def verify_decomposition(U, V, W, rank, tol=1e-10):
        """Verify that U, V, W with given rank correctly decompose the multiplication tensor."""
        # The tensor T[i,j,k] should equal sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Wait, the convention is:
        # U: (n*m, rank) -> U[i*m+j, c]
        # V: (m*p, rank) -> V[j*p+k, c]  
        # W: (p*n, rank) -> W[k*n+i, c]
        # The product: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta(i,j,k) for the multiplication tensor
        
        # Actually for matrix multiplication C = A * B where A is n x m, B is m x p:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor is T[i,j,k] = 1 if we're computing C[i,k], with A[i,j]*B[j,k]
        # So the rank-r decomposition means:
        # C[i,k] = sum_c (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k]) ... 
        # No wait. Let me think again.
        
        # Standard tensor decomposition for matmul:
        # The multiplication tensor T has T[i,j,k] = delta (the coefficient of A[i,j]*B[j,k] in C[i,k])
        # For matmul, T[i,j,k] = 1 for all valid (i,j,k) where i in [0,n), j in [0,m), k in [0,p)
        # Actually no - the tensor entries are: T_{i,j,k} = 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
        # Which it always does, so T[i,j,k] = 1 for all i,j,k.
        
        # Rank-r decomposition: there exist U (n*m x r), V (m*p x r), W (p*n x r) such that
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # And since T[i,j,k] = 1 for all i,j,k:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
        
        # Let's verify this
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Try small ranks using random search
    # For 2x2x11, the minimum rank is at most 44 (schoolbook)
    # Known bounds: rank is between ceil(44/2) and 44... 
    # Actually for 2x2xp, rank is at most 2*2*p = 4p... no, 2*p*2 = 4p? 
    # Schoolbook is n*m*p = 2*2*11 = 44.
    # Strassen gives 7 for 2x2x2, so for 2x2x11 we might do better than 44.
    
    # Let's try to construct using a structured approach:
    # For 2x2x11, we can think of it as two 2x2x11 blocks or use tensor methods.
    
    # Try rank values from 30 to 44, using random initialization + optimization
    for target_rank in range(30, min(44, rank_schoolbook)):
        if time.time() > DEADLINE:
            break
        
        found = False
        for attempt in range(50):
            if time.time() > DEADLINE:
                break
            
            np.random.seed(attempt * 1000 + target_rank)
            
            # Random initialization with rational-like values
            U = np.random.randint(-2, 3, size=(n*m, target_rank))
            V = np.random.randint(-2, 3, size=(m*p, target_rank))
            W = np.random.randint(-2, 3, size=(p*n, target_rank))
            
            # Convert to float for verification
            if verify_decomposition(U.tolist(), V.tolist(), W.tolist(), target_rank):
                found = True
                result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = result
                break
            
            # If not found with pure random, try a structured approach
            # Use the fact that we need sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
            # This is a system of equations. Let's set up the problem differently.
        
        if found:
            continue
    
    # Try a more systematic approach: use the structure of matmul
    # For 2x2x11, think of the 11 as a parameter.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # We can try to find a rank-r decomposition where the structure exploits the 2x2 part.
    
    # Alternative: try to verify if rank 39, 40, 41, 42, 43 are achievable
    # using a linear algebra approach
    
    # For a fixed rank r, we need to find U, V, W such that
    # T = U ⊙ V ⊙ W (tensor product) equals the matmul tensor
    # This is equivalent to: for each c, the outer product U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # sums to T.
    
    # Let's try a different random strategy with more structure
    for target_rank in range(max(25, 44 - 20), 44):
        if time.time() > DEADLINE:
            break
        if target_rank >= best_rank:
            break
            
        found = False
        for attempt in range(100):
            if time.time() > DEADLINE:
                break
            
            np.random.seed(attempt * 7919 + target_rank * 31)
            
            # Use a mix of structured and random
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Initialize with some 1s and -1s
            for c in range(target_rank):
                # Random support
                u_support = set(np.random.choice(n*m, size=max(1, (n*m)//3), replace=False))
                v_support = set(np.random.choice(m*p, size=max(1, (m*p)//3), replace=False))
                w_support = set(np.random.choice(p*n, size=max(1, (p*n)//3), replace=False))
                
                for idx in u_support:
                    U[idx, c] = np.random.choice([-1, 1])
                for idx in v_support:
                    V[idx, c] = np.random.choice([-1, 1])
                for idx in w_support:
                    W[idx, c] = np.random.choice([-1, 1])
            
            if verify_decomposition(U.tolist(), V.tolist(), W.tolist(), target_rank, tol=1e-9):
                found = True
                result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = result
                break
    
    return best_result
