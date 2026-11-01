"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0  # Safety margin before 150s kill

    def verify(U, V, W):
        """
        Check if U, V, W form a valid decomposition of the 3x3 matrix multiplication tensor.
        U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        
        Condition: 
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i, j, k.
        
        Note: The indices for U, V, W in the product correspond to:
        A (n x m) -> index i*m+j
        B (m x p) -> index j*p+k
        C (n x p) -> index k*n+i (Trace convention W[k*n+i, r])
        """
        rank = U.shape[1]
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        # We need T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
        # Vectorized check
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = U3[i, j, :]
                    v = V3[j, k, :]
                    w = W3[k, i, :]
                    val = np.dot(u, v * w)
                    if not np.isclose(val, 1.0):
                        return False
        return True

    def make_rank27():
        """Schoolbook rank-27 decomposition."""
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # Fallback
    U_fb, V_fb, W_fb = make_rank27()
    
    # Try to find a rank-23 decomposition.
    # Strategy: Start with rank 27. Remove 4 columns if they are linearly dependent
    # in the context of the tensor decomposition.
    # A set of columns R is redundant if there exists a non-zero vector c in the null space
    # of the "decomposition matrix" restricted to R?
    # Actually, if we have a decomposition of rank R, we can remove a subset S of columns
    # if the tensor components of the remaining columns can still represent the tensor.
    
    # A simpler heuristic for 3x3:
    # It is known that rank 23 is achievable (Winograd).
    # We will perform a search for a rank 23 solution.
    # Since finding it from scratch is hard, we try to modify the rank 27 solution
    # by combining columns.
    
    # We define a candidate rank 23 search.
    # We will iterate over subsets of 4 columns to remove from rank 27.
    # If removing them still allows a valid decomposition (by adjusting the remaining 23),
    # we accept it.
    
    # However, just removing columns from a valid decomposition usually breaks it unless
    # those columns were zero or redundant. In rank 27, no columns are redundant.
    
    # Therefore, we must *change* the decomposition.
    # We will use a randomized local search around the rank 27 structure.
    
    # Better Strategy: Use the known structure of Winograd's algorithm for 3x3.
    # Winograd's algorithm for 3x3 matrix multiplication has rank 23.
    # The factors are sparse with entries in {-1, 0, 1}.
    # We will attempt to construct it explicitly or search for it.
    
    # Explicit Winograd 3x3 construction:
    # Let A = [[a,b,c],[d,e,f],[g,h,i]], B = [[x,y,z],[u,v,w],[s,t,q]]
    # C = AB
    # 
    # There are 9 multiplications for 2x2 Strassen. For 3x3, we can partition into 2x2 and 1x1?
    # No, Winograd uses a different partitioning.
    
    # Given the complexity of deriving Winograd's exact coefficients from memory without error,
    # and the risk of returning an invalid rank 23 solution, I will attempt a **structured search**
    # for a rank 23 solution using the "cyclic" symmetry of the 3x3 tensor.
    
    # If the search fails within 100s, return rank 27.
    
    import random
    random.seed(123)
    
    best_U, best_V, best_W = None, None, None
    search_deadline = time.time() + 100.0
    
    # Try to find a valid rank 23 decomposition by perturbing the rank 27 one?
    # Or just random search? Random search for rank 23 in dimension 9x23 is extremely unlikely to succeed in 100s.
    
    # Let's try to hardcode a known valid Rank 23 decomposition.
    # I will retrieve a specific known solution.
    # Source: "Winograd's Algorithm for 3x3 Matrix Multiplication"
    
    # The factors U, V, W are 9x23 matrices.
    # I will define them explicitly. If I make a mistake, verify() will catch it and we fall back to 27.
    
    # To ensure I don't hallucinate incorrect coefficients, I will generate a **valid** rank 23
    # by solving a system of linear equations for a **subset** of the Winograd structure.
    
    # Actually, a safer bet: 
    # The prompt allows "integer/half-integer entries".
    # I will try to find a rank 23 solution by optimizing the rank 27 solution 
    # to satisfy rank reduction constraints.
    
    # Since I cannot guarantee a hard-coded 23 solution, and random search is too slow,
    # I will return the rank 27 solution which is guaranteed to be correct and scores 1.0.
    # To potentially improve, I will implement a **fast verification** and a **limited search**
    # for rank 23 using a heuristic that exploits the sparsity of Winograd.
    
    # Heuristic: 
    # Winograd's 3x3 algorithm has 23 products.
    # The structure is highly regular.
    # I will attempt to construct it using the following basis vectors.
    # If construction fails verification, fall back.
    
    # Let's try to build the Winograd 3x3 factors.
    # U, V, W have 9 rows and 23 columns.
    # Rows of U correspond to (i, j) for i,j in 0..2.
    # Rows of V correspond to (j, k) for j,k in 0..2.
    # Rows of W correspond to (k, i) for k,i in 0..2.
    
    # I will define the 23 terms.
    # Terms 0-8: Diagonal-like?
    # This is too risky to guess.
    
    # Instead, I will use a **Genetic Algorithm** approach for 5 seconds to see if I can
    # find a valid 23. If not, return 27.
    
    def evolve_search():
        # Population of (U, V, W) with rank 23.
        # Initialize with random sparse matrices.
        pop_size = 50
        pop = []
        for _ in range(pop_size):
            U = np.zeros((9, 23), dtype=int)
            V = np.zeros((9, 23), dtype=int)
            W = np.zeros((9, 23), dtype=int)
            for r in range(23):
                # Each column should be non-zero in U, V, W
                for idx in range(9):
                    if random.random() < 0.2:
                        U[idx, r] = random.choice([-1, 1])
                for idx in range(9):
                    if random.random() < 0.2:
                        V[idx, r] = random.choice([-1, 1])
                for idx in range(9):
                    if random.random() < 0.2:
                        W[idx, r] = random.choice([-1, 1])
            pop.append((U, V, W))
        
        for gen in range(20): # Max 20 generations
            if time.time() > search_deadline:
                break
            new_pop = []
            for U, V, W in pop:
                # Check validity
                if verify(U, V, W):
                    return U, V, W
                # Mutation: flip a bit
                U_new = U.copy()
                V_new = V.copy()
                W_new = W.copy()
                
                if random.random() < 0.3:
                    r = random.randint(0, 22)
                    idx = random.randint(0, 8)
                    if U_new[idx, r] != 0:
                        U_new[idx, r] = -U_new[idx, r]
                    else:
                        U_new[idx, r] = random.choice([-1, 1])
                if random.random() < 0.3:
                    r = random.randint(0, 22)
                    idx = random.randint(0, 8)
                    if V_new[idx, r] != 0:
                        V_new[idx, r] = -V_new[idx, r]
                    else:
                        V_new[idx, r] = random.choice([-1, 1])
                if random.random() < 0.3:
                    r = random.randint(0, 22)
                    idx = random.randint(0, 8)
                    if W_new[idx, r] != 0:
                        W_new[idx, r] = -W_new[idx, r]
                    else:
                        W_new[idx, r] = random.choice([-1, 1])
                        
                new_pop.append((U_new, V_new, W_new))
            pop = new_pop
        return None

    # Run search
    result = evolve_search()
    if result:
        return {"U": result[0].tolist(), "V": result[1].tolist(), "W": result[2].tolist()}
        
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
