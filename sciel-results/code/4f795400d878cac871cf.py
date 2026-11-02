"""Program for matrix multiplication tensor decomposition 4x4 x 4x6."""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    # Schoolbook rank is n*m*p = 96
    best_rank = n * m * p
    best_result = None
    
    def build_schoolbook():
        """Standard schoolbook decomposition with rank n*m*p."""
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def validate(U, V, W, rank):
        """
        Check if U, V, W form a valid tensor decomposition.
        T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1.
        Uses numpy einsum for speed.
        """
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # Reshape
        U_reshaped = U_arr.reshape(n, m, rank)  # (i, j, c)
        V_reshaped = V_arr.reshape(m, p, rank)  # (j, k, c)
        W_reshaped = W_arr.reshape(p, n, rank)  # (k, i, c)
        
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        return np.allclose(T, 1.0, atol=1e-10)
    
    def strassen_2x2_decomposition():
        """
        Return the Strassen decomposition for 2x2 x 2x2 -> 2x2 multiplication.
        Returns U, V, W arrays with shapes (4, 7), (4, 7), (4, 7).
        """
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # Strassen's 7 products:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # A indices: A11=0, A12=1, A21=2, A22=3
        # B indices: B11=0, B12=1, B21=2, B22=3
        # C indices: C11=0, C12=1, C21=2, C22=3
        
        # M1: A11+A22 times B11+B22
        U[0, 0] = 1; U[3, 0] = 1  # A11, A22
        V[0, 0] = 1; V[3, 0] = 1  # B11, B22
        W[0, 0] = 1; W[3, 0] = 1  # C11, C22
        
        # M2: A21+A22 times B11
        U[2, 1] = 1; U[3, 1] = 1  # A21, A22
        V[0, 1] = 1  # B11
        W[2, 1] = 1; W[3, 1] = 1  # C21, C22
        
        # M3: A11 times B12-B22
        U[0, 2] = 1  # A11
        V[1, 2] = 1; V[3, 2] = -1  # B12, B22
        W[1, 2] = 1; W[3, 2] = 1  # C12, C22
        
        # M4: A22 times B21-B11
        U[3, 3] = 1  # A22
        V[2, 3] = 1; V[0, 3] = -1  # B21, B11
        W[0, 3] = 1; W[2, 3] = 1  # C11, C21
        
        # M5: A11+A12 times B22
        U[0, 4] = 1; U[1, 4] = 1  # A11, A12
        V[3, 4] = 1  # B22
        W[0, 4] = -1; W[1, 4] = 1  # C11, C12
        
        # M6: A21-A11 times B11+B12
        U[2, 5] = 1; U[0, 5] = -1  # A21, A11
        V[0, 5] = 1; V[1, 5] = 1  # B11, B12
        W[3, 5] = 1  # C22
        
        # M7: A12-A22 times B21+B22
        U[1, 6] = 1; U[3, 6] = -1  # A12, A22
        V[2, 6] = 1; V[3, 6] = 1  # B21, B22
        W[0, 6] = 1  # C11
        
        return U, V, W
    
    def block_decomposition_2x2():
        """
        Use block decomposition: 4x4 x 4x6 viewed as 2x2 blocks of 2x2 times 2x3 blocks of 2x2.
        Each 2x2 x 2x2 -> 2x2 block multiplication has rank 7 (Strassen).
        Total blocks: 2*2 * 2*3 = 4 * 6 = 24 block multiplications.
        But we can share Strassen factors across blocks.
        
        Actually, the full 4x4 x 4x6 can be seen as:
        A is 2x2 block matrix of 2x2 blocks
        B is 2x3 block matrix of 2x2 blocks
        C is 2x3 block matrix of 2x2 blocks
        
        C_ab = sum_d A_ad * B_db for a,d in {0,1} and b in {0,1,2}
        
        We can use 7 Strassen products for the 2x2 block structure, but each product
        involves 2x2 x 2x2 -> 2x2 which itself has rank 7.
        So total rank would be 7 * 7 = 49? No, that's not right.
        
        Actually, we need to think of this differently. The tensor for 4x4 x 4x6
        can be decomposed using the tensor product of the 2x2 block structure.
        
        Let's use the fact that M_{4,4,6} can be built from M_{2,2,2} using
        a Kronecker product structure.
        
        Alternatively, we can use a direct construction:
        - Split A into 4 blocks of 2x2: A00, A01, A10, A11
        - Split B into 6 blocks of 2x2: B00, B01, B02, B10, B11, B12
        - C has 6 blocks of 2x2: C00, C01, C02, C10, C11, C12
        
        C00 = A00*B00 + A01*B10
        C01 = A00*B01 + A01*B11
        C02 = A00*B02 + A01*B12
        C10 = A10*B00 + A11*B10
        C11 = A10*B01 + A11*B11
        C12 = A10*B02 + A11*B12
        
        Each C_ab is a 2x2 matrix computed as sum of two 2x2 products.
        Each 2x2 product has rank 7 (Strassen).
        But we can share the Strassen factors.
        
        Let's use 7 Strassen products for the "outer" 2x2 block structure:
        M1 = (A00+A11)(B00+B10)  -- but B has 3 columns, so this doesn't directly work.
        
        This is getting complex. Let me try a different approach.
        """
        pass
    
    # Start with schoolbook
    school = build_schoolbook()
    if validate(school["U"], school["V"], school["W"], best_rank):
        best_result = school
    
    # Try to find better decompositions
    # Strategy: Use the known result that rank of M_{n,m,p} <= n*m*p
    # and try to find structured decompositions with lower rank.
    
    # For 4x4 x 4x4, the best known rank is 49 (from Strassen-like constructions).
    # For 4x4 x 4x6, we can extend this.
    
    # Let's try a direct construction using the tensor product of Strassen decompositions.
    # M_{4,4,6} can be seen as M_{2,2,3} ⊗ M_{2,2,2} in some sense.
    
    # Actually, let's try to construct a decomposition with rank 49 or lower.
    # We'll use the fact that we can decompose the problem into smaller subproblems.
    
    # For now, let's try random search with structured initialization.
    # We'll perturb the schoolbook decomposition slightly.
    
    def random_perturbed_candidate(rank, base_U, base_V, base_W):
        """Generate a candidate by perturbing a base decomposition."""
        if rank >= len(base_U[0]):
            # Need to reduce rank, so we can't just perturb
            return None, None, None
        
        U = [row[:rank] for row in base_U]
        V = [row[:rank] for row in base_V]
        W = [row[:rank] for row in base_W]
        
        # Add small random perturbations
        for i in range(len(U)):
            for j in range(rank):
                if random.random() < 0.1:
                    U[i][j] += random.randint(-1, 1) * 0.5
        for i in range(len(V)):
            for j in range(rank):
                if random.random() < 0.1:
                    V[i][j] += random.randint(-1, 1) * 0.5
        for i in range(len(W)):
            for j in range(rank):
                if random.random() < 0.1:
                    W[i][j] += random.randint(-1, 1) * 0.5
        
        return U, V, W
    
    import random
    random.seed(42)
    
    # Try various ranks
    for target_rank in range(49, best_rank):
        if time.time() > deadline:
            break
        
        # Try a few random candidates at this rank
        for trial in range(50):
            if time.time() > deadline:
                break
            
            U, V, W = random_perturbed_candidate(target_rank, school["U"], school["V"], school["W"])
            if U is None:
                continue
            
            if validate(U, V, W, target_rank):
                best_rank = target_rank
                best_result = {"U": U, "V": V, "W": W}
                break
    
    if best_result is None:
        best_result = school
    
    return best_result
# EVOLVE-BLOCK-END
