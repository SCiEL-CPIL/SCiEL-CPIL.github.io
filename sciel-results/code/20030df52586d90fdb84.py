"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor T of shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Schoolbook decomposition (rank = n*m*p = 50)
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school), dtype=np.float64)
    V_school = np.zeros((mp, rank_school), dtype=np.float64)
    W_school = np.zeros((pn, rank_school), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1

    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin before 150s kill

    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_school

    def verify_decomposition(U, V, W, T):
        """Check if U, V, W exactly decompose T."""
        T_comp = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(T_comp, T, atol=1e-8)

    def try_structured_decomposition(R):
        """
        Try structured decompositions with sparse patterns.
        For 2x5x5, we exploit the structure:
        - U has 10 rows, V has 25 rows, W has 10 rows
        - Each rank-1 term contributes to specific (i,j,k) positions
        """
        nonlocal best_result, best_rank
        
        # Strategy: enumerate which (i,j,k) triples each rank-1 term covers
        # For n=2, m=5, p=5: there are 50 (i,j,k) triples
        # Each rank-1 term U[:,r] V[:,r] W[:,r] creates a rank-1 tensor
        # We need to cover all 50 positions with R rank-1 terms
        
        # For small R, we can try to find which positions each term covers
        # Use a greedy/beam search approach
        
        # First, let's try known constructions
        # For 2x5x5, the Strassen-like algorithm might work
        
        # Try a specific structured approach:
        # Group by i (2 groups), then within each i, handle j,k pairs
        
        # Let's try ranks from 30 to 49
        # Use a more targeted search
        
        # Approach: For each rank R, try to find U, V, W with integer/half-integer entries
        # that are sparse (few non-zero entries per column)
        
        # Key insight: In known low-rank decompositions, each factor matrix has
        # a specific sparsity pattern. Let's try to enumerate these.
        
        # For rank R, each column of U, V, W has some sparsity pattern
        # Let's try patterns where each column has 1-3 non-zero entries
        
        max_nonzero = 3
        
        # Generate candidate sparsity patterns
        def generate_sparse_patterns(size, max_nz, max_val=2):
            """Generate sparse vectors with max_nz non-zero entries from {-2,-1,0,1,2}"""
            patterns = []
            # Start with single non-zero
            for pos in range(size):
                for val in [-2, -1, 1, 2]:
                    vec = np.zeros(size)
                    vec[pos] = val
                    patterns.append(vec)
            
            # Two non-zeros
            if max_nz >= 2:
                for pos1 in range(size):
                    for pos2 in range(pos1 + 1, size):
                        for val1 in [-1, 1]:
                            for val2 in [-1, 1]:
                                vec = np.zeros(size)
                                vec[pos1] = val1
                                vec[pos2] = val2
                                patterns.append(vec)
            
            # Three non-zeros
            if max_nz >= 3:
                for pos1 in range(size):
                    for pos2 in range(pos1 + 1, size):
                        for pos3 in range(pos2 + 1, size):
                            for val1 in [-1, 1]:
                                for val2 in [-1, 1]:
                                    for val3 in [-1, 1]:
                                        vec = np.zeros(size)
                                        vec[pos1] = val1
                                        vec[pos2] = val2
                                        vec[pos3] = val3
                                        patterns.append(vec)
            
            return patterns
        
        # This could be too many patterns, so let's be more strategic
        # Focus on patterns that match the tensor structure
        
        # For the matrix multiplication tensor, natural patterns are:
        # U column: e_{i*m+j} (single basis vector)
        # V column: e_{j*p+k} (single basis vector)  
        # W column: e_{k*n+i} (single basis vector)
        
        # But for lower rank, we need combinations. Let's try:
        # U column: a*e_{i1*m+j1} + b*e_{i2*m+j2}
        # etc.
        
        # Let's use a different approach: exploit the fact that the tensor has
        # a specific algebraic structure and try known algorithms
        
        # For 2x5x5, try the following structured approach:
        # Use 2x2 Strassen ideas extended to 2x5x5
        
        # Actually, let's try a more direct approach:
        # Enumerate possible "types" of rank-1 terms based on which (i,j,k) they cover
        
        # A rank-1 term with U=e_a, V=e_b, W=e_c contributes to position (a,b,c)
        # where a = i*m+j, b = j*p+k, c = k*n+i
        # For consistency, we need: a mod m == b mod p, b div p == c mod n, c div n == a div m
        
        # Let's precompute which (a,b,c) triples are valid
        valid_triples = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    valid_triples.append((a, b, c))
        
        # Now, for lower rank, we need rank-1 terms that cover multiple valid triples
        # A rank-1 term U[:,r] V[:,r] W[:,r] covers all (a,b,c) where
        # U[a,r] * V[b,r] * W[c,r] != 0
        
        # For the term to contribute to exactly the valid triples, we need:
        # For each valid (a,b,c): U[a,r]*V[b,r]*W[c,r] should sum to 1 over all r
        # For each invalid (a,b,c): U[a,r]*V[b,r]*W[c,r] should sum to 0 over all r
        
        # This is a system of equations. For small R, we can try to solve it.
        
        # Let's try a randomized approach with structure:
        # Initialize U, V, W with sparse random integer entries
        # Then use iterative refinement
        
        for attempt in range(50):
            if time.time() > deadline:
                return False
            
            np.random.seed(42 + R * 1000 + attempt)
            
            # Initialize with sparse random entries
            U = np.zeros((nm, R))
            V = np.zeros((mp, R))
            W = np.zeros((pn, R))
            
            # For each column, assign a random sparse pattern
            for r in range(R):
                # Choose which positions to activate
                # U: choose 1-2 positions from nm=10
                num_u = np.random.randint(1, min(3, nm + 1))
                u_positions = np.random.choice(nm, num_u, replace=False)
                for pos in u_positions:
                    U[pos, r] = np.random.choice([-2, -1, 1, 2])
                
                # V: choose 1-2 positions from mp=25
                num_v = np.random.randint(1, min(3, mp + 1))
                v_positions = np.random.choice(mp, num_v, replace=False)
                for pos in v_positions:
                    V[pos, r] = np.random.choice([-2, -1, 1, 2])
                
                # W: choose 1-2 positions from pn=10
                num_w = np.random.randint(1, min(3, pn + 1))
                w_positions = np.random.choice(pn, num_w, replace=False)
                for pos in w_positions:
                    W[pos, r] = np.random.choice([-2, -1, 1, 2])
            
            # Check if this works
            T_comp = np.einsum('ir,jr,kr->ijk', U, V, W)
            if np.allclose(T_comp, T, atol=1e-8):
                best_rank = R
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                return True
        
        # If random didn't work, try a more structured approach
        # Use the fact that the problem has symmetry
        
        # For 2x5x5, try to construct using 2x2 blocks
        # Let's try a specific construction
        
        # Approach: Use the known result that M(2,5,5) <= 34
        # Try to find such a decomposition
        
        if R <= 34:
            # Try a very targeted search
            # Fix the structure: each rank-1 term covers a specific pattern
            
            # Let's try to use the following idea:
            # For i=0: handle (0,j,k) for j,k in 0..4
            # For i=1: handle (1,j,k) for j,k in 0..4
            
            # A rank-1 term can cover multiple (i,j,k) if U has multiple non-zeros
            # Let's try terms that cover 2 positions in U, 2 in V, 2 in W
            
            for sub_attempt in range(20):
                if time.time() > deadline:
                    return False
                
                np.random.seed(1000 + R * 100 + sub_attempt)
                
                U = np.zeros((nm, R))
                V = np.zeros((mp, R))
                W = np.zeros((pn, R))
                
                # Assign each rank-1 term a "type"
                for r in range(R):
                    # Choose a base (i,j,k) and extend
                    i = np.random.randint(0, n)
                    j = np.random.randint(0, m)
                    k = np.random.randint(0, p)
                    
                    # Primary position
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    
                    U[a, r] = 1
                    V[b, r] = 1
                    W[c, r] = 1
                    
                    # Maybe add a secondary position
                    if np.random.random() < 0.3:
                        i2 = np.random.randint(0, n)
                        j2 = np.random.randint(0, m)
                        k2 = np.random.randint(0, p)
                        a2 = i2 * m + j2
                        b2 = j2 * p + k2
                        c2 = k2 * n + i2
                        
                        if a2 != a:
                            U[a2, r] = np.random.choice([-1, 1])
                        if b2 != b:
                            V[b2, r] = np.random.choice([-1, 1])
                        if c2 != c:
                            W[c2, r] = np.random.choice([-1, 1])
                
                T_comp = np.einsum('ir,jr,kr->ijk', U, V, W)
                if np.allclose(T_comp, T, atol=1e-8):
                    best_rank = R
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    return True
        
        return False

    # Try ranks in increasing order
    candidate_ranks = list(range(30, rank_school))
    
    for R in candidate_ranks:
        if time.time() > deadline - 5.0:
            break
        
        success = try_structured_decomposition(R)
        if success:
            break

    return best_result
# EVOLVE-BLOCK-END
