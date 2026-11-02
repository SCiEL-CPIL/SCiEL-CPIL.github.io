import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS

    # The schoolbook rank is n*m*p = 3*3*7 = 63.
    # The lower bound is max(n*m, m*p, n*p) = max(9, 21, 21) = 21.
    #
    # A structural change: instead of trying to hand-code a specific low-rank
    # decomposition (which is error-prone and hard to verify), we use a
    # randomized approach to find a valid decomposition with rank < 63.
    #
    # Key insight: We can use the "border rank" or approximate rank ideas,
    # but we need EXACT decomposition. 
    #
    # Alternative structural approach: Use the fact that the tensor is 
    # C(3,3,7). We can split the 3x7 matrix into two parts: 3x3 and 3x4.
    # C(3,3,3) has rank 23 (known). C(3,3,4) has rank at most 36 (schoolbook)
    # but potentially lower.
    #
    # Actually, let's try a different structural approach:
    # Use a randomized search over integer/half-integer coefficient matrices
    # to find a valid low-rank decomposition.
    #
    # But first, let's establish a valid baseline that's better than 63 if possible.
    #
    # Known results:
    # - C(3,3,3) rank = 23
    # - C(3,3,4) rank <= 36 (schoolbook), but we can do better.
    # - C(3,3,7) rank <= 23 + 36 = 59 (splitting p into 3+4)
    #
    # Let's implement the 3x3x3 rank-23 decomposition correctly.
    # The Winograd 23-mult algorithm for 3x3 matrix multiplication:
    #
    # A = [[a00, a01, a02],
    #      [a10, a11, a12],
    #      [a20, a21, a22]]
    # B = [[b00, b01, b02],
    #      [b10, b11, b12],
    #      [b20, b21, b22]]
    # C = A*B
    #
    # The 23 multiplications (from Winograd, 1968):
    # M1  = (a00+a11)(b00+b11)
    # M2  = (a10+a11)b00
    # M3  = a00(b01-b11)
    # M4  = a11(b10-b00)
    # M5  = (a00+a01)b11
    # M6  = (a10-a00)(b00+b01)
    # M7  = (a01-a11)(b10+b11)
    # M8  = a02(b00+b10+b20)
    # M9  = a12(b00+b10+b20)
    # M10 = a22(b00+b10+b20)
    # M11 = (a00+a02)(b20+b21)
    # M12 = (a10+a12)(b20+b21)
    # M13 = (a20+a22)(b20+b21)
    # M14 = a02(b01+b11+b21)
    # M15 = a12(b01+b11+b21)
    # M16 = a22(b01+b11+b21)
    # M17 = (a00+a02)(b21-b20)  -- wait, this doesn't look right
    #
    # Let me use a verified 23-mult algorithm. I'll use the one from:
    # "The rank of 3x3 matrix multiplication" - various sources.
    #
    # Actually, let me use a simpler approach: just return the schoolbook 
    # decomposition with rank 63, which is guaranteed valid. Then try to 
    # improve it.
    #
    # For now, let's implement a valid decomposition and then try to reduce rank.
    
    # Schoolbook: rank = 63
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
    
    # Verify the schoolbook decomposition
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i,j,k] = 1 for all i,j,k
    # Decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r has 1 at (i,j), v_r has 1 at (j,k), w_r has 1 at (k,i)
    # This is correct.
    
    # Now try to find a lower rank decomposition.
    # Strategy: Try rank from 21 up to 62, and for each rank, try to find
    # a valid decomposition using randomized search.
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try to improve using the split approach: C(3,3,7) = C(3,3,3) + C(3,3,4)
    # If we can find rank-23 for C(3,3,3) and rank-30 for C(3,3,4), total = 53.
    
    # Let's try a randomized approach to find a valid decomposition.
    # For each candidate rank r, generate random sparse U, V, W matrices
    # and check if they decompose the tensor.
    
    # The tensor T has T[i,j,k] = 1 for all i in [0,3), j in [0,3), k in [0,7).
    # We need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k.
    
    # This is a system of 3*3*7 = 63 equations in r * (9 + 21 + 21) = r * 51 unknowns.
    # For r >= 21, this is underdetermined (more unknowns than equations).
    
    # Let's try to solve this system for small r using randomized initialization.
    
    import random
    random.seed(42)
    
    for candidate_rank in range(21, 63):
        if time.time() > deadline - 20:
            break
        
        # Try a few random attempts for this rank
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Generate random sparse matrices with integer/half-integer entries
            U_r = np.zeros((n * m, candidate_rank))
            V_r = np.zeros((m * p, candidate_rank))
            W_r = np.zeros((p * n, candidate_rank))
            
            # For each rank component, randomly assign non-zero entries
            for r_idx in range(candidate_rank):
                # U: pick 2-4 random positions
                num_u = random.randint(2, 4)
                u_positions = random.sample(range(n * m), min(num_u, n * m))
                for pos in u_positions:
                    U_r[pos, r_idx] = random.choice([1, -1, 0.5, -0.5])
                
                # V: pick 2-4 random positions
                num_v = random.randint(2, 4)
                v_positions = random.sample(range(m * p), min(num_v, m * p))
                for pos in v_positions:
                    V_r[pos, r_idx] = random.choice([1, -1, 0.5, -0.5])
                
                # W: pick 2-4 random positions
                num_w = random.randint(2, 4)
                w_positions = random.sample(range(p * n), min(num_w, p * n))
                for pos in w_positions:
                    W_r[pos, r_idx] = random.choice([1, -1, 0.5, -0.5])
            
            # Check if this decomposition is valid
            # Compute the tensor from U, V, W and compare to the target
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        val = 0.0
                        for r_idx in range(candidate_rank):
                            val += U_r[i * m + j, r_idx] * V_r[j * p + k, r_idx] * W_r[k * n + i, r_idx]
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = candidate_rank
                best_U = U_r.tolist()
                best_V = V_r.tolist()
                best_W = W_r.tolist()
                break
        else:
            continue
        break
    
    # Return the best decomposition found
    return {"U": best_U, "V": best_V, "W": best_W}
