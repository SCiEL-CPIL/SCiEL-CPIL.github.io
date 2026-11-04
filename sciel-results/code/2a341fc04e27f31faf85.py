import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline for safety
    
    def strassen_block_solution():
        """Original Strassen-based solution with rank 36"""
        rank = 36
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        def add_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            for idx, val in u_coeffs.items():
                if val != 0:
                    U[idx][col] += val
            for idx, val in v_coeffs.items():
                if val != 0:
                    V[idx][col] += val
            for idx, val in w_coeffs.items():
                if val != 0:
                    W[idx][col] += val
            col += 1

        strassen_terms = [
            ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
            ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
            ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
            ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),
            ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
            ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
            ({1:1, 3:-1}, {2:1, 3:1}, {0:1})
        ]
        
        standard_terms_2x2x1 = [
            ({0:1}, {0:1}, {0:1}),
            ({1:1}, {1:1}, {0:1}),
            ({2:1}, {0:1}, {1:1}),
            ({3:1}, {1:1}, {1:1})
        ]

        for q in range(2):
            for s in range(3):
                if s < 2:
                    for u_t, v_t, w_t in strassen_terms:
                        u_global = {}
                        v_global = {}
                        w_global = {}
                        
                        for idx, val in u_t.items():
                            li = idx // 2
                            lj = idx % 2
                            gi = li
                            gj = 2*q + lj
                            u_global[gi * m + gj] = val
                        
                        for idx, val in v_t.items():
                            lj = idx // 2
                            lk = idx % 2
                            gj = 2*q + lj
                            gk = 2*s + lk
                            v_global[gj * p + gk] = val
                        
                        for idx, val in w_t.items():
                            li = idx // 2
                            lk = idx % 2
                            gi = li
                            gk = 2*s + lk
                            w_global[gk * n + gi] = val
                        
                        add_term(u_global, v_global, w_global)
                else:
                    for u_t, v_t, w_t in standard_terms_2x2x1:
                        u_global = {}
                        v_global = {}
                        w_global = {}
                        
                        for idx, val in u_t.items():
                            li = idx // 2
                            lj = idx % 2
                            gi = li
                            gj = 2*q + lj
                            u_global[gi * m + gj] = val
                        
                        for idx, val in v_t.items():
                            lj = idx
                            gk = 4
                            gj = 2*q + lj
                            v_global[gj * p + gk] = val
                        
                        for idx, val in w_t.items():
                            li = idx
                            gk = 4
                            gi = li
                            w_global[gk * n + gi] = val
                        
                        add_term(u_global, v_global, w_global)

        return {"U": U, "V": V, "W": W}

    def verify_decomposition(U, V, W):
        """Verify that U, V, W correctly decompose matrix multiplication"""
        n, m, p = DIMENSIONS
        num_u = n * m
        num_v = m * p
        num_w = p * n
        rank = len(U[0])
        
        # Build the multiplication tensor T[i,j,k] where C[i,k] = sum_j A[i,j]*B[j,k]
        # For each output entry (i,k), we need: sum_r U[i*m+j... wait
        # Actually: C[i,k] = sum_j A[i,j] * B[j,k]
        # In our decomposition: C[i,k] = sum_r W[k*n+i][r] * (sum_j U[i*m+j][r] * V[j*p+k][r])
        # Wait, that's not right. Let me reconsider.
        
        # The tensor is: T[a,b,c] where a indexes A, b indexes B, c indexes C
        # A has n*m entries, B has m*p entries, C has p*n entries
        # T[a,b,c] = 1 if c corresponds to the output that uses a and b, else 0
        
        # Actually for matrix mult: C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor T[A_idx, B_idx, C_idx] = 1 if A_idx=(i,j), B_idx=(j,k), C_idx=(k,n+i)
        
        # For a decomposition: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # i.e., T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        
        # Let's verify by checking all entries
        for i in range(n):
            for k in range(p):
                c_idx = k * n + i
                # T[?, ?, c_idx] should be non-zero only for valid (a,b) pairs
                # For this c_idx, valid a,b are: a = i*m+j, b = j*p+k for j in 0..m-1
                for j in range(m):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    # T[a_idx, b_idx, c_idx] should equal 1
                    val = sum(U[a_idx][r] * V[b_idx][r] * W[c_idx][r] for r in range(rank))
                    if val != 1:
                        return False
                # Check that T is zero for invalid combinations
                # For this c_idx, invalid a are those where a%m != any j, or...
                # Actually, let's check a few invalid ones
                for a_idx in range(n*m):
                    for b_idx in range(m*p):
                        # Check if this is a valid combination for c_idx
                        ai = a_idx // m
                        aj = a_idx % m
                        bj = b_idx // p
                        bk = b_idx % p
                        if ai == i and bk == k and aj == bj:
                            continue  # valid, already checked
                        # This should be 0
                        val = sum(U[a_idx][r] * V[b_idx][r] * W[c_idx][r] for r in range(rank))
                        if val != 0:
                            return False
        return True

    def greedy_rank1_extraction():
        """
        Try to find a lower-rank decomposition using greedy rank-1 extraction
        with exact rational arithmetic.
        """
        if time.time() > deadline:
            return None
        
        n, m, p = DIMENSIONS
        num_u = n * m  # 8
        num_v = m * p  # 20
        num_w = p * n  # 10
        
        # Build the target tensor T as a 3D array using Fraction
        # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,n+i), else 0
        T = [[[Fraction(0) for _ in range(num_w)] for _ in range(num_v)] for _ in range(num_u)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    T[a_idx][b_idx][c_idx] = Fraction(1)
        
        # Greedy extraction: at each step, find the best rank-1 approximation
        # to the residual tensor
        U_list = []
        V_list = []
        W_list = []
        
        max_rank = 40  # schoolbook rank
        current_T = T
        
        for step in range(max_rank):
            if time.time() > deadline:
                return None
            
            # Find the maximum absolute value in the residual
            max_val = Fraction(0)
            max_a, max_b, max_c = 0, 0, 0
            for a in range(num_u):
                for b in range(num_v):
                    for c in range(num_w):
                        if abs(current_T[a][b][c]) > max_val:
                            max_val = abs(current_T[a][b][c])
                            max_a, max_b, max_c = a, b, c
            
            if max_val == 0:
                break
            
            # The rank-1 term should have:
            # u_a = 1 if a == max_a, else 0
            # v_b = 1 if b == max_b, else 0
            # w_c = T[max_a][max_b][max_c]
            # This gives a rank-1 term that matches the maximum entry
            
            u = [Fraction(0)] * num_u
            v = [Fraction(0)] * num_v
            w = [Fraction(0)] * num_w
            
            u[max_a] = Fraction(1)
            v[max_b] = Fraction(1)
            w[max_c] = current_T[max_a][max_b][max_c]
            
            # Subtract this rank-1 term from the residual
            for a in range(num_u):
                for b in range(num_v):
                    for c in range(num_w):
                        current_T[a][b][c] -= u[a] * v[b] * w[c]
            
            U_list.append(u)
            V_list.append(v)
            W_list.append(w)
            
            # Check if residual is zero
            residual_nonzero = False
            for a in range(num_u):
                for b in range(num_v):
                    for c in range(num_w):
                        if current_T[a][b][c] != 0:
                            residual_nonzero = True
                            break
                    if residual_nonzero:
                        break
                if residual_nonzero:
                    break
            
            if not residual_nonzero:
                break
        
        if len(U_list) >= max_rank:
            return None
        
        # Convert to the required format
        rank = len(U_list)
        U = [[0] * rank for _ in range(num_u)]
        V = [[0] * rank for _ in range(num_v)]
        W = [[0] * rank for _ in range(num_w)]
        
        for r in range(rank):
            for i in range(num_u):
                if U_list[r][i] != 0:
                    U[i][r] = float(U_list[r][i])
            for i in range(num_v):
                if V_list[r][i] != 0:
                    V[i][r] = float(V_list[r][i])
            for i in range(num_w):
                if W_list[r][i] != 0:
                    W[i][r] = float(W_list[r][i])
        
        return {"U": U, "V": V, "W": W}

    def improved_strassen_with_cancellation():
        """
        Try to improve the Strassen-based solution by looking for cancellations.
        This uses a more careful block decomposition.
        """
        if time.time() > deadline:
            return None
        
        n, m, p = DIMENSIONS
        # Use a different block structure: 2x4 x 4x5
        # Split the 4 middle dimension into 2+2
        # Split the 5 output dimension into 2+2+1
        
        # For each 2x2x2 block, use Strassen (rank 7)
        # For each 2x2x1 block, use standard (rank 4)
        # Total: 2*3*7 + 2*1*4 = 42 + 8 = 50... that's worse
        
        # Actually, let's try a different approach:
        # Split A into two 2x2 blocks (rows 0-1, cols 0-1 and cols 2-3)
        # Split B into two 2x5 blocks (rows 0-1 and rows 2-3)
        # C = A1*B1 + A2*B2 where A1 is 2x2, B1 is 2x5, etc.
        
        # For 2x2 * 2x5: rank is at most 2*2*5=20, but can be lower
        # Use Strassen for the 2x2 part of B, then multiply by remaining
        
        # Actually, let's just return the original solution if this doesn't help
        return strassen_block_solution()
    
    # First, try the greedy approach
    greedy_result = greedy_rank1_extraction()
    if greedy_result is not None:
        # Verify it
        if verify_decomposition(greedy_result["U"], greedy_result["V"], greedy_result["W"]):
            return greedy_result
    
    # Fall back to Strassen-based solution
    return strassen_block_solution()
# EVOLVE-BLOCK-END
