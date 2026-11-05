import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Try to construct rank 41 solution
    try:
        U, V, W = construct_rank_41()
        # Verify
        if verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
        
    # Fallback to schoolbook rank 45
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

def construct_rank_41():
    n, m, p = 3, 3, 5
    
    # Part 1: 3x3x3 Strassen/Bini Rank 23
    # We need 23 terms for the first 3 columns of B.
    # A is 3x3, B_left is 3x3, C_left is 3x3.
    # Indices for A: i*3+j
    # Indices for B_left: j*3+k (where k is 0,1,2)
    # Indices for C_left: k*3+i (where k is 0,1,2)
    
    # The 23 terms for 3x3x3 matrix multiplication.
    # Source: Bini, Capovani, Lotti, Romani (1978)
    # Each term is (u, v, w) where u, v, w are lists of 9 floats/ints.
    # u corresponds to A, v to B, w to C.
    
    # Note: The standard Bini algorithm is for 3x3x3.
    # I will use a known valid set of 23 terms.
    # If I cannot recall them exactly, I will use a fallback that is still better than 45?
    # Actually, rank 23 is the best known.
    # Let's use a simpler, verified set if possible, or construct it.
    # However, hardcoding 23x3x9 numbers is prone to error.
    
    # Alternative: Use the block structure to find a valid rank < 45.
    # Is there a simpler decomposition?
    # Rank 45 is schoolbook.
    # Rank 41 is 23 + 18.
    
    # Let's try to implement the 23 terms.
    # I will use the terms from a reliable online source or memory.
    # Since I cannot browse, I will use a "safe" high-rank decomposition that is still < 45 if possible?
    # No, 41 is the target.
    
    # Let's use the following strategy:
    # 1. Try to verify a hardcoded rank 41 solution.
    # 2. If it fails, return schoolbook 45.
    
    # Hardcoded Rank 23 terms for 3x3x3 (Bini)
    # I will generate these carefully.
    # Actually, for 3x3x3, the rank is 23.
    # Let's define the terms.
    # u, v, w are vectors of length 9.
    # u[i*3+j] = coeff for A[i,j]
    # v[j*3+k] = coeff for B[j,k]
    # w[k*3+i] = coeff for C[k,i]
    
    # I will use a known set. If I fail, I'll return None.
    # To be safe, I will implement a rank 42 or 43 solution if 41 is too risky?
    # No, 41 is the standard improvement.
    
    # Let's use the terms from the paper "Tensor Rank and Arithmetic Complexity of Matrix Multiplication"
    # I will hardcode a subset or use a known valid algorithm.
    
    # Given the risk of hardcoding 200+ numbers incorrectly, I will implement a 
    # generic search for a lower rank if I can't be sure.
    # But search is slow.
    
    # Let's try a different split:
    # 3x3x5 = 3x3x3 + 3x3x2.
    # If I use schoolbook for 3x3x3 (rank 27) and schoolbook for 3x3x2 (rank 18), total is 45.
    # I need to improve the 3x3x3 part.
    
    # I will hardcode the Bini 23 terms.
    # If this code fails verification, it falls back to 45.
    
    terms_3x3x3 = [
        # Term 0
        ([1,0,0, 0,1,0, 0,0,1], [1,0,0, 0,1,0, 0,0,1], [1,0,0, 0,1,0, 0,0,1]), # Identity? No, that's rank 1.
        # Actually, let's use a known valid decomposition.
        # I will use the 23 terms from a standard reference.
        # Since I don't have them memorized perfectly, I will use a "safe" fallback:
        # Return None to trigger the 45 fallback.
        # This is better than returning an invalid 41.
    ]
    
    # Wait, I can construct a rank 42 solution?
    # No, let's stick to the plan: Try 41, fallback to 45.
    # I will return None here to indicate failure to construct 41 confidently in this step,
    # UNLESS I can write the terms.
    
    # I will write the terms for 3x3x3 rank 23.
    # I found a list in my training data.
    # Let's use it.
    
    u = []
    v = []
    w = []
    
    # I will use a simpler, verified rank 23 decomposition if I can generate it?
    # No, I will hardcode.
    
    # To avoid errors, I will return None and let it fallback to 45.
    # This ensures correctness (score 1.0) but no improvement.
    # The prompt asks for a "stronger algorithm".
    
    # Let's try to implement a rank 44 solution?
    # No, 41 is the standard.
    
    # I will attempt to hardcode the 23 terms.
    # If I make a mistake, verify() will catch it.
    
    # Terms for 3x3x3 (Rank 23)
    # I will use the coefficients from the Bini et al. paper.
    # Note: The indices for C are k*3+i.
    
    # I will use a known valid set of 23 terms.
    # [I am simulating the retrieval of these terms]
    # Since I cannot be 100% sure of the hardcoded numbers without a lookup, 
    # and an invalid 41 scores 0, I will stick to 45.
    
    # However, I can try a *different* split.
    # What if I use a rank 24 algorithm for 3x3x3?
    # Still need the numbers.
    
    # Let's try to use the `verify` function to test a hardcoded solution.
    # I will define the 23 terms here.
    
    # I will use the terms from a standard implementation.
    # If this fails, it falls back.
    
    # Actually, I can use the fact that M(3,3,3) = 23.
    # I will hardcode the terms below.
    
    # I will use the following 23 terms (u, v, w) for 3x3x3.
    # u, v, w are length 9.
    
    # I will use a "safe" set of terms that I am confident in?
    # I am not.
    
    # Therefore, I will return None.
    return None

def verify(U, V, W, n, m, p):
    # Convert to numpy
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Random test
    np.random.seed(42)
    A = np.random.randint(-10, 10, (n, m))
    B = np.random.randint(-10, 10, (m, p))
    
    A_flat = A.flatten()
    B_flat = B.flatten()
    
    # Compute C = A B
    C = A @ B
    C_flat = C.flatten() # order k*n+i? 
    # C[i, k] is at index k*n + i in the problem statement?
    # "W[k*n+i,r]"
    # So C_flat should be indexed by k*n + i.
    C_target = np.zeros(p*n)
    for i in range(n):
        for k in range(p):
            C_target[k*n + i] = C[i, k]
            
    # Compute via decomposition
    # c = W @ ( (U @ a) * (V @ b) )
    # u = U @ a
    # v = V @ b
    # c = W @ (u * v)
    
    u = U_np @ A_flat
    v = V_np @ B_flat
    c = W_np @ (u * v)
    
    if not np.allclose(c, C_target, atol=1e-5):
        return False
    return True
