import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = 3, 3, 6
    
    def make_schoolbook(n, m, p):
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
        return U, V, W, rank

    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Winograd/FFT approach using cyclic group C_3 x C_6
    # The idea: use DFT over C_3 for the "row" and "column" indices
    # This transforms the matrix multiplication into pointwise operations
    
    def try_winograd_fft():
        """
        Attempt rank-36 decomposition using FFT/Winograd technique.
        Use characters of C_3 x C_6 to diagonalize the tensor.
        """
        # For 3x3x6, we can view this as 3 independent 1x3x6 multiplications
        # or use the full group structure
        
        # The tensor T[i,j,k] = 1 for C = A*B where A is 3x3, B is 3x6
        # Indices: i in {0,1,2} (row of C), j in {0,1,2} (col of A = row of B), k in {0,1,2,3,4,5} (col of C)
        
        # Winograd approach: use DFT over cyclic groups
        # For the j index (size 3), use DFT_3
        # This should reduce the rank significantly
        
        # Let's use the standard Winograd formulation for 3x3 matrix multiplication
        # which gives rank 23, then extend to 3x6
        
        # Actually, let me try a direct FFT-based approach
        # The key insight: if we use DFT on the j-index, we can diagonalize part of the tensor
        
        # Define DFT_3 matrix
        omega = np.exp(2j * np.pi / 3)
        DFT3 = np.array([
            [1, 1, 1],
            [1, omega, omega**2],
            [1, omega**2, omega]
        ])
        DFT3_inv = np.conj(DFT3) / 3
        
        # For the 3x3x6 problem, we can use:
        # 1. Apply DFT_3 to the j-index (size 3)
        # 2. This gives us 3 independent problems
        # 3. Each problem has rank related to the size of the other indices
        
        # Let's construct the decomposition using the group characters
        # The rank should be |C_3| * |C_3| * (6/3) = 3 * 3 * 2 = 18? No, that's not right.
        
        # Actually, for matrix multiplication A(n,m) * B(m,p), the Winograd rank is:
        # sum over divisors of m of (n/gcd(n,d)) * (p/gcd(p,d)) * phi(d)
        # For m=3, n=3, p=6:
        # d=1: (3/1)*(6/1)*1 = 18
        # d=3: (3/3)*(6/3)*2 = 2*2*1 = 4... wait, phi(3)=2
        # So total = 18 + 4 = 22? That doesn't seem right either.
        
        # Let me just try to construct a rank-36 decomposition using FFT
        # by treating the 6 columns as 2 groups of 3
        
        rank = 36
        U = [[0] * rank for _ in range(n * m)]  # 9 x 36
        V = [[0] * rank for _ in range(m * p)]  # 18 x 36
        W = [[0] * rank for _ in range(p * n)]  # 18 x 36
        
        # Use DFT_3 for the j-index
        # For each frequency f in {0, 1, 2} (DFT_3 frequency):
        #   For each output position (i, k_block, k_offset) where k_block in {0,1}, k_offset in {0,1,2}:
        #     Create a rank-1 term
        
        # Actually, let me use a simpler approach: just use the schoolbook for now
        # and verify it works, since I'm not confident in the FFT construction
        
        return None
    
    # First, create the safe schoolbook fallback
    U_school, V_school, W_school, rank_school = make_schoolbook(n, m, p)
    
    # Try the FFT approach
    result_fft = try_winograd_fft()
    
    if result_fft is not None:
        U, V, W, rank = result_fft
        if verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W, rank = U_school, V_school, W_school, rank_school
    
    if time.time() > deadline - 10:
        return {"U": U, "V": V, "W": W}
        
    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}
