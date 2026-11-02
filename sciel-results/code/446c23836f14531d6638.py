"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n, m, p = DIMENSIONS

    # Build full rank-54 solution
    def schoolbook():
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    # Try to find a rank-27 decomposition via Kronecker / tensor power
    # The rank-27 solution corresponds to decomposing the tensor into 27 rank-1 terms.
    # Use rank-9 for 2x3x3 (if possible).
    # If R(2,3,3) = 9, we can achieve rank 27.

    # First, check R(2,3,3)
    def check_rank(r, a, b, c):
        # Check if the tensor has rank <= r
        # Tensor is of size a*b*c
        # Build tensor and check using iterative approach or rank computation
        if a * b * c <= 26:
            # Small enough for tensor rank
            import sys
            sys.setrecursionlimit(5000)
            
            # For 2x3x3: 18 entries, rank 9
            # Try to find rank-9 decomposition
            # C[i,k] = sum_t u_t[i] v_t[k] w_t[j]
            # where i in 0..a-1, j in 0..b-1, k in 0..c-1
            # C is a x c matrix
            # B is b x c
            # A is a x b
            # C[i,k] = sum_j A[i,j] B[j,k]
            # C = A B, where A is a x b and B is b x c
            
            # We can try to express this as sum_t u_t v_t^T w_t
            # where u_t is in R^a, v_t is in R^b, w_t is in R^c
            # The tensor T[i,j,k] = 1 if j is free, 0 otherwise? No.
            # For matrix mult tensor: T[i,j,k] = 1
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # So T[i,j,k] = A[i,j] * B[j,k] * 1
            # Wait, the tensor for matrix mult is:
            # T = sum_t (U_t A) * (V_t B) where U_t, V_t are...
            # No, standard: C = A B, with T[i,j,k] = 1 for all i,j,k
            # C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
            
            # Actually the tensor is: T[i,j,k] such that C[i,k] = sum_j A[i,j] B[j,k]
            # This can be written as:
            # C[i,k] = sum_t sum_j (alpha_{t,i,j} A[i,j]) (beta_{t,j,k} B[j,k])
            # where alpha and beta are the factorization of the tensor.
            
            # For the matrix multiplication tensor, the tensor entries are:
            # M[i, (j,k)] = A[i,j] * B[j,k]
            # But in the standard tensor decomposition form:
            # M[i, j, k] = 1 (for all valid i, j, k)
            
            # Let's use a more direct approach:
            # For a x b and b x c multiplication:
            # The tensor is in R^{a*b x b*c x b}
            # We want to find rank r such that the tensor can be written as sum of r rank-1 terms.
            
            # Try a constructive approach: if b=2 and c=2, Strassen gives rank 7.
            # For a=2, b=3, c=3, try to see if rank 9 works.
            
            if a == 2 and b == 3 and c == 3:
                # Try to find a rank-9 decomposition for 2x3 * 3x3
                # One approach: group outputs.
                # C has 2x3 = 6 outputs.
                # B has 3x3 = 9 entries.
                # A has 2x3 = 6 entries.
                
                # Try rank 9 by using the fact that the output is 2x3:
                # Each output C[i,k] = sum_j A[i,j] B[j,k]
                # We can try: rank 9 decomposition where each rank-1 term
                # contributes to multiple outputs.
                
                # Constructive: 
                # Let's use the "Strassen-like" approach for 2x3x3.
                # Or simply try: if we can write C = AB as sum of 9 rank-1 terms.
                
                # A simple valid rank-9 decomposition:
                # For each j (0,1,2) and k (0,1,2), define:
                # Term jk: (A[:,j]) * B[j,k] contributes to C[:,k]
                # But each term has a 2-vector (A[:,j]) and a 3-vector... 
                # Actually, for the tensor decomposition:
                # C[i,k] = sum_t U_t[i] V_t[k] W_t[j] (summed over j? No.)
                
                # The correct tensor form:
                # C = A B where C[i,k] = sum_j A[i,j] B[j,k]
                # The tensor is a 3D array T with T[i,j,k] = 1 for all valid indices
                # Wait, that's not right. The "matrix multiplication tensor" in 
                # tensor decomposition context is:
                # T in R^{(nm) x (mp)} such that:
                # M = sum_{i,j,k} A_{i,j} B_{j,k} E_{i,k}
                # Or in factorization form:
                # We want T = sum_t (u_t, v_t, w_t) as rank-1 tensors
                # where the product gives C = AB.
                
                # For the purposes of this code, let's just use the known result:
                # R(2,3,3) is at least 9. If we can't find a decomposition with rank < 18,
                # we use 18.
                
                return 18  # Conservative: use 18 for 2x3x3
            
            return a * b * c  # Fallback: full rank
    
    # For 2x3 * 3x9:
    # Treat as 3 blocks of 2x3 * 3x3
    # Total rank = 3 * R(2,3,3)
    # Best known: R(2,3,3) <= 18, lower bound 9
    # If R(2,3,3) = 9, total rank = 27
    
    r233 = 18  # Conservative
    total_rank = 3 * r233
    
    # Try rank 27 if possible
    # For now, use rank 54 (schoolbook) as it's guaranteed to work
    
    if total_rank < 54 and time.time() < deadline:
        # Try to construct rank-27 if R(2,3,3)=9
        # For now, skip and use 54
        pass
    
    # Use schoolbook
    U, V, W = schoolbook()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
