import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def _trivial_decomposition(n, m, p):
    """Construct the trivial rank n*m*p decomposition."""
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
    return {"U": U, "V": V, "W": W}, rank

def _check_decomposition(U, V, W, n, m, p, rank):
    """
    Check if the decomposition is valid.
    """
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for t in range(rank):
                    s += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                if s != 1:
                    return False
    return True

def _winograd_2x3x7():
    """
    Attempt to construct a Winograd-style decomposition for 2x3x7.
    
    Strategy:
    1. Decompose the 3x7 multiplication (j, k) into blocks.
       Since 7 is prime, we can't split it evenly.
       However, we can use Winograd for the 3x3 part (rank 4) and trivial for the rest?
       Or use a global Winograd scheme.
       
    Let's try a block approach with Winograd on the 3x3 sub-blocks.
    
    The multiplication is C[i,k] = sum_j A[i,j] B[j,k].
    n=2, m=3, p=7.
    
    We can split the k dimension into:
    - k in {0, 1, 2}: 3 columns
    - k in {3, 4, 5}: 3 columns
    - k in {6}: 1 column
    
    For the 3x3 blocks (j in 0..2, k in 0..2 and j in 0..2, k in 3..5),
    we can use the Winograd scheme for 3x3 matrix multiplication.
    The rank for 3x3 matrix multiplication using Winograd is 4 (for the 2x2 case it's 4, for 3x3 it's known to be 19 or 20? No, Winograd for 3x3 is rank 19? Actually, the rank of 3x3 matrix multiplication is 19. The trivial is 27.
    Wait, the Winograd scheme for 3x3 matrix multiplication (multiplying two 3x3 matrices) has rank 19.
    But here we are multiplying a 2x3 matrix by a 3x3 matrix.
    This is a sub-tensor of the 3x3x3 multiplication tensor.
    
    Let's look at the tensor T_{i, j, k} for n=2, m=3, p=3.
    Rank of 2x3x3 multiplication tensor?
    Trivial is 18.
    Can we do better?
    
    Let's try to apply Winograd to the j-dimension (size 3) and k-dimension (size 3).
    Winograd 3x3 scheme uses 4 transforms? No, the 2x2 Winograd uses 4 transforms (2+2-1=3? No, 2+2-1=3 is for convolution. For matrix mult it's different).
    
    Actually, let's stick to the known Winograd scheme for 2x2x2 (rank 4) and see if we can tile it.
    But 3 is not a power of 2.
    
    Alternative: Use the trivial decomposition for the 2x3x1 block (rank 6) and try to find a better decomposition for the 2x3x3 blocks.
    
    Let's try a random search for a rank < 18 decomposition of the 2x3x3 tensor.
    If we find rank 17 for 2x3x3, then total rank = 17 + 17 + 6 = 40.
    If we find rank 16, total = 16 + 16 + 6 = 38.
    
    Let's implement a random search for the 2x3x3 tensor.
    """
    n, m, p_block = 2, 3, 3
    trivial_rank = n * m * p_block # 18
    
    # Target tensor T[i, j, k] = 1 if i,j,k match, else 0.
    # We want U (n*m x r), V (m*p_block x r), W (p_block*n x r)
    # such that sum_r U[i*m+j, r] * V[j*p_block+k, r] * W[k*n+i, r] = 1
    
    # Random search
    best_rank = trivial_rank
    best_decomp = None
    
    # We'll try to find a decomposition of rank r for r from 17 down to 10
    # Time limit for search: 10 seconds
    start_time = time.time()
    deadline = start_time + 10.0
    
    for r in range(trivial_rank - 1, 9, -1):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            # Random initialization
            U = np.random.randint(0, 2, (n*m, r))
            V = np.random.randint(0, 2, (m*p_block, r))
            W = np.random.randint(0, 2, (p_block*n, r))
            
            # Check if it's a valid decomposition
            # This is expensive, so we only check a subset or use a smarter check?
            # For small r, checking all n*m*p = 18 entries is fast.
            is_valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p_block):
                        s = 0
                        for t in range(r):
                            s += U[i*m+j, t] * V[j*p_block+k, t] * W[k*n+i, t]
                        if s != 1:
                            is_valid = False
                            break
                    if not is_valid:
                        break
                if not is_valid:
                    break
            if is_valid:
                best_rank = r
                best_decomp = (U.tolist(), V.tolist(), W.tolist())
                break # Found one for this r, try lower r
        
        if best_decomp is not None:
            # We found a decomposition for rank r, but maybe we can find a lower one?
            # The loop continues to lower r.
            pass

    # If we found a better decomposition for the 2x3x3 block, assemble the full solution
    if best_decomp is not None:
        U_block, V_block, W_block = best_decomp
        r_block = best_rank
        
        # Full decomposition
        # Block 1: k=0,1,2 (uses r_block)
        # Block 2: k=3,4,5 (uses r_block)
        # Block 3: k=6 (uses trivial 6)
        
        total_rank = r_block + r_block + 6
        
        U_full = [[0] * total_rank for _ in range(n*m)]
        V_full = [[0] * total_rank for _ in range(m*7)]
        W_full = [[0] * total_rank for _ in range(7*n)]
        
        # Fill Block 1
        offset = 0
        for i in range(n):
            for j in range(m):
                for t in range(r_block):
                    U_full[i*m+j][offset+t] = U_block[i*m+j][t]
        for j in range(m):
            for k_local in range(p_block):
                for t in range(r_block):
                    V_full[j*7 + k_local][offset+t] = V_block[j*p_block+k_local][t]
        for k_local in range(p_block):
            for i in range(n):
                for t in range(r_block):
                    W_full[k_local*n + i][offset+t] = W_block[k_local*n+i][t]
        
        # Fill Block 2
        offset += r_block
        for i in range(n):
            for j in range(m):
                for t in range(r_block):
                    U_full[i*m+j][offset+t] = U_block[i*m+j][t]
        for j in range(m):
            for k_local in range(p_block):
                k_global = 3 + k_local
                for t in range(r_block):
                    V_full[j*7 + k_global][offset+t] = V_block[j*p_block+k_local][t]
        for k_local in range(p_block):
            k_global = 3 + k_local
            for i in range(n):
                for t in range(r_block):
                    W_full[k_global*n + i][offset+t] = W_block[k_local*n+i][t]
        
        # Fill Block 3 (Trivial)
        offset += r_block
        for i in range(n):
            for j in range(m):
                for k_local in range(1): # Only k=6
                    k_global = 6
                    col = offset + (i*m+j)*1 + k_local
                    U_full[i*m+j][col] = 1
                    V_full[j*7 + k_global][col] = 1
                    W_full[k_global*n + i][col] = 1
        
        result = {"U": U_full, "V": V_full, "W": W_full}
        return result, total_rank
    else:
        # No improvement found, return None
        return None, None

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with trivial decomposition (Rank 42)
    best_result, best_rank = _trivial_decomposition(n, m, p)
    
    # Try Winograd-style improvement
    winograd_result, winograd_rank = _winograd_2x3x7()
    if winograd_result is not None and winograd_rank < best_rank:
        # Verify
        if _check_decomposition(winograd_result['U'], winograd_result['V'], winograd_result['W'], n, m, p, winograd_rank):
            best_result = winograd_result
            best_rank = winograd_rank
    
    return best_result

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
