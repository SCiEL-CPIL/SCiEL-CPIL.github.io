import time
import numpy as np

DIMENSIONS = [2, 2, 11]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Rank 39 solution: 5 Strassen blocks (each rank 7) for 2x2 * 2x11
    # Each block handles 2 columns of B
    def get_rank_39():
        rank = 39
        U = [[0] * rank for _ in range(n * m)]  # 4 x 39
        V = [[0] * rank for _ in range(m * p)]  # 22 x 39
        W = [[0] * rank for _ in range(p * n)]  # 22 x 39
        r = 0
        for block in range(5):
            k0 = block * 2
            k1 = block * 2 + 1
            # V indices: row 0 is B[0,*], row 1 is B[1,*]
            # For block columns k0, k1:
            v_e = k0       # B[0, k0]
            v_f = k1       # B[0, k1]
            v_g = 11 + k0  # B[1, k0]
            v_h = 11 + k1  # B[1, k1]
            # W indices: W[k*n + i, r] where k is column of C, i is row of C
            # C is 2 x 11, so W has 22 rows: C[0,0], C[1,0], C[0,1], C[1,1], ...
            w_c00 = 2 * k0    # C[0, k0]
            w_c10 = 2 * k0 + 1  # C[1, k0]
            w_c01 = 2 * k1    # C[0, k1]
            w_c11 = 2 * k1 + 1  # C[1, k1]
            
            # Strassen's 7 multiplications for 2x2 * 2x2:
            # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
            U[0][r] = 1; U[3][r] = 1
            V[v_e][r] = 1; V[v_h][r] = 1
            W[w_c00][r] = 1; W[w_c11][r] = 1
            r += 1
            # M2 = A[1,0] * (B[0,1]+B[1,1])
            U[2][r] = 1; U[3][r] = 1
            V[v_e][r] = 1
            W[w_c10][r] = 1; W[w_c11][r] = -1
            r += 1
            # M3 = (A[0,0]+A[0,1]) * B[1,0]
            U[0][r] = 1
            V[v_f][r] = 1; V[v_h][r] = -1
            W[w_c01][r] = 1; W[w_c11][r] = 1
            r += 1
            # M4 = A[1,1] * (B[1,0]-B[0,0])
            U[3][r] = 1
            V[v_g][r] = 1; V[v_e][r] = -1
            W[w_c00][r] = 1; W[w_c10][r] = 1
            r += 1
            # M5 = (A[0,0]+A[1,0]) * B[0,1]
            U[0][r] = 1; U[1][r] = 1
            V[v_h][r] = 1
            W[w_c00][r] = -1; W[w_c01][r] = 1
            r += 1
            # M6 = A[0,1] * (B[1,1]-B[0,0])
            U[2][r] = 1; U[0][r] = -1
            V[v_e][r] = 1; V[v_f][r] = 1
            W[w_c11][r] = 1
            r += 1
            # M7 = A[1,1] * (B[0,0]+B[0,1])
            U[1][r] = 1; U[3][r] = -1
            V[v_g][r] = 1; V[v_h][r] = 1
            W[w_c00][r] = 1
            r += 1
        
        # Last column (k=10) handled separately with schoolbook (4 mults)
        k2 = 10
        v_e2 = k2
        v_g2 = 11 + k2
        w_c02 = 2 * k2
        w_c12 = 2 * k2 + 1
        
        U[0][r] = 1; V[v_e2][r] = 1; W[w_c02][r] = 1; r += 1
        U[1][r] = 1; V[v_g2][r] = 1; W[w_c02][r] = 1; r += 1
        U[2][r] = 1; V[v_e2][r] = 1; W[w_c12][r] = 1; r += 1
        U[3][r] = 1; V[v_g2][r] = 1; W[w_c12][r] = 1; r += 1
        
        return {"U": U, "V": V, "W": W}
    
    # Try to find rank 38: use Strassen for 4 blocks (8 cols, rank 28) + 
    # a rank 10 decomposition for the remaining 3 columns (cols 8,9,10)
    # Total: 28 + 10 = 38
    
    # Search for rank 10 decomposition of 2x2 * 2x3 tensor
    # U: 4x10, V: 6x10, W: 6x10
    # Try small integer/half-integer coefficients
    
    def search_rank_10_for_2x2x2x3(max_time=30):
        """Search for rank 10 decomposition of 2x2 * 2x3 matrix multiplication."""
        search_start = time.time()
        search_deadline = search_start + max_time
        
        # The tensor T[i,j,k] = 1 if C[i,k] = sum_l A[i,l]*B[l,k]
        # For 2x2 * 2x3: A is 2x2, B is 2x3, C is 2x3
        # T has dimensions 4 x 6 x 6 (flattened)
        # Actually let's think of it as: for each output C[i,k], it's sum over l of A[i,l]*B[l,k]
        # So the tensor is: T[(i,l), (l,k), (i,k)] but that's not quite right.
        # 
        # Standard form: C = A * B where A is n x m, B is m x p, C is n x p
        # Tensor T has indices (i, l, k) where i in [0,n), l in [0,m), k in [0,p)
        # T[i,l,k] = 1 if the product A[i,l]*B[l,k] contributes to C[i,k]
        # 
        # For n=2, m=2, p=3:
        # T has 4*6*6 = 144 entries, but most are 0.
        # Non-zero: T[i,l,k] = 1 for all valid (i,l,k)
        # 
        # Rank r decomposition: T = sum_{r=1}^{rank} u_r * v_r * w_r
        # where u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
        # 
        # For n=2,m=2,p=3: u in R^4, v in R^6, w in R^6
        
        # We want to find U (4x10), V (6x10), W (6x10) such that
        # sum_r U[:,r] * V[:,r] * W[:,r] = T
        
        # Let's try a structured search. Start from a known rank 11 solution
        # (Strassen for 2 cols + schoolbook for 1 col) and try to merge.
        
        # Actually, let's try random search with small coefficients
        coeffs = [-1, -0.5, 0, 0.5, 1]
        
        import random
        random.seed(42)
        
        best_rank = 11  # Known upper bound
        best_solution = None
        
        # Try random initializations and local optimization
        for attempt in range(500):
            if time.time() > search_deadline:
                break
            
            # Random initialization with small coefficients
            U = np.zeros((4, 10))
            V = np.zeros((6, 10))
            W = np.zeros((6, 10))
            
            for r in range(10):
                # Random sparse vectors
                for i in range(4):
                    if random.random() < 0.5:
                        U[i, r] = random.choice(coeffs)
                for i in range(6):
                    if random.random() < 0.5:
                        V[i, r] = random.choice(coeffs)
                for i in range(6):
                    if random.random() < 0.5:
                        W[i, r] = random.choice(coeffs)
            
            # Check if this gives the right tensor
            # Compute the tensor from U, V, W
            T_recon = np.zeros((4, 6, 6))
            for r in range(10):
                T_recon += np.outer(np.outer(U[:, r], V[:, r]), W[:, r]).reshape(4, 6, 6)
            
            # Target tensor
            T_target = np.zeros((4, 6, 6))
            for i in range(2):  # row of A/C
                for l in range(2):  # col of A / row of B
                    for k in range(3):  # col of B/C
                        T_target[i*2+l, l*3+k, i*3+k] = 1
            
            if np.allclose(T_recon, T_target, atol=1e-10):
                best_rank = 10
                best_solution = (U.tolist(), V.tolist(), W.tolist())
                break
        
        return best_solution
    
    # Try the search
    result_10 = search_rank_10_for_2x2x2x3(max_time=30)
    
    if result_10 is not None and time.time() < deadline:
        U10, V10, W10 = result_10
        # Build rank 38 solution: 4 Strassen blocks (cols 0-7) + rank 10 (cols 8-10)
        rank = 38
        U = [[0] * rank for _ in range(4)]
        V = [[0] * rank for _ in range(22)]
        W = [[0] * rank for _ in range(22)]
        r = 0
        
        # 4 Strassen blocks for cols 0-7
        for block in range(4):
            k0 = block * 2
            k1 = block * 2 + 1
            v_e = k0
            v_f = k1
            v_g = 11 + k0
            v_h = 11 + k1
            w_c00 = 2 * k0
            w_c10 = 2 * k0 + 1
            w_c01 = 2 * k1
            w_c11 = 2 * k1 + 1
            
            U[0][r] = 1; U[3][r] = 1
            V[v_e][r] = 1; V[v_h][r] = 1
            W[w_c00][r] = 1; W[w_c11][r] = 1
            r += 1
            U[2][r] = 1; U[3][r] = 1
            V[v_e][r] = 1
            W[w_c10][r] = 1; W[w_c11][r] = -1
            r += 1
            U[0][r] = 1
            V[v_f][r] = 1; V[v_h][r] = -1
            W[w_c01][r] = 1; W[w_c11][r] = 1
            r += 1
            U[3][r] = 1
            V[v_g][r] = 1; V[v_e][r] = -1
            W[w_c00][r] = 1; W[w_c10][r] = 1
            r += 1
            U[0][r] = 1; U[1][r] = 1
            V[v_h][r] = 1
            W[w_c00][r] = -1; W[w_c01][r] = 1
            r += 1
            U[2][r] = 1; U[0][r] = -1
            V[v_e][r] = 1; V[v_f][r] = 1
            W[w_c11][r] = 1
            r += 1
            U[1][r] = 1; U[3][r] = -1
            V[v_g][r] = 1; V[v_h][r] = 1
            W[w_c00][r] = 1
            r += 1
        
        # Add rank 10 block for cols 8,9,10
        # The rank 10 solution is for 2x2 * 2x3, so it handles 3 columns
        # Map cols 8,9,10 to the 3 columns of the sub-problem
        for rr in range(10):
            for i in range(4):
                U[i][r + rr] = U10[i][rr]
            for i in range(6):
                # V indices: cols 8,9,10 of B
                # In the sub-problem, col 0->8, col 1->9, col 2->10
                # V row l*3+k where l is row of B (0 or 1), k is col in sub-problem
                if i < 3:
                    # l=0, k=i -> global col 8+i
                    V[8 + i][r + rr] = V10[i][rr]
                else:
                    # l=1, k=i-3 -> global col 11 + (8 + (i-3)) = 11 + 5 + i - 3
                    # Wait, B is 2x11, so rows 0-10 are B[0,*], rows 11-21 are B[1,*]
                    # For col c in {8,9,10}: B[0,c] is at index c, B[1,c] is at index 11+c
                    k_sub = i - 3
                    V[11 + 8 + k_sub][r + rr] = V10[i][rr]
            for i in range(6):
                # W indices: C is 2x11, W[k*n+i] = W[2*k+i] where k is col of C, i is row
                # In sub-problem, col 0->8, col 1->9, col 2->10
                if i < 3:
                    # k_sub = i, row 0
                    W[2 * (8 + i) + 0][r + rr] = W10[i][rr]
                else:
                    # k_sub = i-3, row 1
                    W[2 * (8 + (i-3)) + 1][r + rr] = W10[i][rr]
        
        return {"U": U, "V": V, "W": W}
    
    # Fallback to rank 39
    return get_rank_39()
