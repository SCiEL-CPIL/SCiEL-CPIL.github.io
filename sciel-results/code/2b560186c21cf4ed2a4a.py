import numpy as np
import time
import random

DIMENSIONS = [4, 4, 8]

def _verify(U, V, W):
    """Verify that U, V, W form a valid decomposition."""
    n, m, p = DIMENSIONS
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    # Check: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] should give the multiplication tensor
    # The multiplication tensor T[i,j,k] = 1 if this is a valid (i,j,k) term
    # Actually, the decomposition means: for all i,j,k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if (i,j,k) is a valid term
    # Wait, the multiplication tensor is: T[i,j,k] = 1 for the term A[i,j]*B[j,k] contributing to C[i,k]
    # So the tensor has entries T[i,j,k] = 1 for all valid (i,j,k)
    # The decomposition: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = T[i,j,k] = 1
    # Wait, that's not right either. Let me re-read.
    
    # The tensor for matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # The multiplication tensor T has T[i,j,k] = 1 (it's the structure tensor)
    # A rank-r decomposition: T = sum_{c=1}^r u_c ⊗ v_c ⊗ w_c
    # where u_c is in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # For matrix multiplication, T[i,j,k] = 1 for all i,j,k (since every term A[i,j]*B[j,k] contributes)
    # Wait no. The tensor T is defined such that the bilinear form gives the multiplication.
    # T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B multiply to give (i,k) entry of C
    # So T[i,j,k] = 1 for all valid i,j,k.
    
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Let's verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Start with schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    best = {"U": U_school, "V": V_school, "W": W_school, "rank": rank_school}
    
    # Try to find lower rank decompositions
    # Use random search with integer/half-integer entries
    random.seed(42)
    
    for target_rank in range(100, 50, -1):
        if time.time() > deadline - 20:
            break
        found = False
        for attempt in range(100):
            if time.time() > deadline - 10:
                break
            # Generate random matrices with integer/half-integer entries
            def rand_entry():
                r = random.randint(0, 4)
                sign = random.choice([-1, 1])
                return sign * r / 2
            
            U = [[rand_entry() for _ in range(target_rank)] for _ in range(n*m)]
            V = [[rand_entry() for _ in range(target_rank)] for _ in range(m*p)]
            W = [[rand_entry() for _ in range(target_rank)] for _ in range(p*n)]
            
            if _verify(U, V, W):
                best = {"U": U, "V": V, "W": W, "rank": target_rank}
                found = True
                break
        
        if found:
            continue
        else:
            # If we can't find rank target_rank, try lower
            if target_rank < best["rank"]:
                break
    
    if time.time() > deadline:
        pass
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
