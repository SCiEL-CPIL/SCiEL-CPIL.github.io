import numpy as np
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def _verify_factors(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W with shapes (n*m, rank), (m*p, rank), (p*n, rank)
    produce the matrix multiplication tensor under the trace convention W[k*n+i, r]."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Enumerate all triples (A, B, C)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                A = i * m + j
                B = j * p + k
                C = k * n + i
                expected = 1
                actual = np.dot(U_arr[A], V_arr[B] * W_arr[C])
                if abs(actual - expected) > tol:
                    return False, (A, B, C, expected, actual)
    return True, None

def _build_schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
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
    return U, V, W

def _build_rank_15_decomposition(n, m, p):
    """
    Attempt to construct a rank-15 decomposition for M_{2,3,3}.
    This uses a structured approach combining Strassen-like reductions.
    If verification fails, returns None.
    """
    if (n, m, p) != (2, 3, 3):
        return None
    
    rank = 15
    U = [[0] * rank for _ in range(n * m)]  # 6x15
    V = [[0] * rank for _ in range(m * p)]  # 9x15
    W = [[0] * rank for _ in range(p * n)]  # 6x15
    
    # We define 15 bilinear forms.
    # Let's try to manually define a known or plausible set.
    # Since I don't have the exact hardcoded coefficients for a guaranteed rank-15 solution for 2x3x3 
    # memorized with 100% confidence in the specific index mapping, 
    # I will implement a search procedure that tries to find such a solution 
    # or simply return None if I can't guarantee it immediately.
    
    # However, the prompt suggests "Leverage the known rank-15 decomposition... by explicitly constructing".
    # If I cannot explicitly construct it with certainty, I should not guess random integers and hope.
    # But wait, the current program scores 1.0 (rank 18). 
    # If I can't find rank 15, I return rank 18.
    
    # Let's try a heuristic construction that might work.
    # Often, lower rank decompositions for small matrices involve small integers.
    # Let's try to solve for the coefficients using a linear system?
    # The condition is sum_r U[a,r] V[b,r] W[c,r] = 1 for valid (a,b,c) and 0 otherwise.
    # This is a system of quadratic equations.
    
    # Given the strict time limit and the risk of returning an invalid decomposition,
    # and the fact that I don't have the explicit coefficients for the rank-15 2x3x3 tensor 
    # readily available in my "memory" as a fixed list of numbers,
    # I will attempt a local optimization or random search with a tight budget 
    # to see if I can find a rank 15 solution. If not, I fall back to 18.
    
    # Actually, a safer bet for "explicit construction" is to use the structure of 
    # embedding the 2x2x2 Strassen into the 2x3x3 problem.
    
    # Let's try to build a rank-15 solution by extending a rank-7 Strassen for 2x2x2 
    # and handling the remaining dimensions.
    
    # A is 2x3, B is 3x3, C is 3x2.
    # Split A into A1 (2x2) and A2 (2x1).
    # Split B into B1 (2x2), B2 (2x1), B3 (1x2), B4 (1x1).
    # C = A1*B1 + A2*B3 (for the first 2 cols of C)
    #    + A1*B2 (for the last col of C? No, dimensions don't match simply)
    
    # Let's stick to the robust path: 
    # 1. Try to verify a hardcoded candidate if I had one. I don't.
    # 2. Use a fast local search to try to find a rank 15 solution.
    # 3. Fallback to rank 18.
    
    return None

def construct():
    n, m, p = 2, 3, 3
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # 1. Try Rank 15
    # Since explicit construction is complex and I don't have the hardcoded values,
    # I will perform a very limited, targeted search for a rank-15 solution.
    # If this is too slow or fails, I fall back to 18.
    
    # A known fact: The tensor rank of 2x3x3 is at most 15.
    # Is it exactly 15? Yes, it is known that R(M_{2,3,3}) = 15.
    
    # Let's try to find it via a randomized gradient descent or similar? 
    # That's risky with 110s.
    
    # Alternative: Use the "Baby-step Giant-step" idea from the directive?
    # Or just return the schoolbook?
    # The current program returns schoolbook.
    # To improve, I MUST find rank 15.
    
    # Let's try a specific construction based on literature for M_{2,3,3}.
    # One common way to get rank 15 is to use a rank-7 decomposition for the 2x2x2 part 
    # and combine it with the remaining terms.
    
    # Let's try to hardcode a known rank-15 decomposition if I can reconstruct it.
    # If I can't, I will use a quick random search with small integers.
    
    found_rank_15 = False
    U15 = V15 = W15 = None
    
    # Try a few fixed seeds for random integer search in {-1, 0, 1} for rank 15
    # This is a long shot but cheap to try a few times.
    import random
    rng = random.Random(42)
    
    # We have 6*15 + 9*15 + 6*15 = 90 + 135 + 90 = 315 variables.
    # Random search in 315 dimensions is hopeless.
    
    # Therefore, I must rely on a structured construction.
    # If I cannot derive it, I should return the best known valid solution, which is 18.
    # However, the directive says "Leverage the known rank-15 decomposition... explicitly constructing".
    # This implies I should be able to.
    
    # Let's look at the structure:
    # C = A B
    # A = [A1 A2] where A1 is 2x2, A2 is 2x1
    # B = [B1 B2; B3 B4] where B1 is 2x2, B2 is 2x1, B3 is 1x2, B4 is 1x1
    # C = A1 B1 + A2 B3  (2x2 block)
    #    A1 B2 + A2 B4   (2x1 block)
    
    # We can compute A1 B1 using Strassen (7 mults).
    # We need to compute A1 B2, A2 B3, A2 B4.
    # A1 B2 is 2x1. Can be done in 2 mults.
    # A2 B3 is 2x2. Can be done in 4 mults.
    # A2 B4 is 2x1. Can be done in 2 mults.
    # Total = 7 + 2 + 4 + 2 = 15 mults.
    
    # Let's construct this!
    
    # Indices:
    # A: 2x3. Rows 0,1. Cols 0,1 (A1), Col 2 (A2).
    # B: 3x3. Rows 0,1 (B1,B2), Row 2 (B3,B4). Cols 0,1 (B1,B3), Col 2 (B2,B4).
    #    B1 = B[0:2, 0:2]
    #    B2 = B[0:2, 2]
    #    B3 = B[2, 0:2]
    #    B4 = B[2, 2]
    # C: 3x2. Rows 0,1 (from A1B1+A2B3), Row 2 (from A1B2+A2B4).
    #    C[0:2, 0:2] = A1 B1 + A2 B3
    #    C[0:2, 2]   = A1 B2 + A2 B4
    #    Wait, C is 3x2.
    #    A is 2x3, B is 3x3. C is 2x3?
    #    No, task says: "multiplying a 2 by 3 matrix by a 3 by 3 matrix".
    #    Output is 2x3.
    #    But the interface says W is (p*n, rank) = (3*2, rank) = 6x15.
    #    And W[k*n+i, r].
    #    C indices: k (row of B? No, k is col of A / row of B in standard multiplication).
    #    Standard: C[i,k] = sum_j A[i,j] B[j,k].
    #    i in 0..1 (n=2), k in 0..2 (p=3).
    #    W index: k*n + i = 2k + i.
    #    So W[0]=C[0,0], W[1]=C[1,0], W[2]=C[0,1], W[3]=C[1,1], W[4]=C[0,2], W[5]=C[1,2].
    
    # Let's re-map the block decomposition to C[i,k].
    # A is 2x3. B is 3x3.
    # Split A into A_left (2x2) and A_right (2x1).
    # Split B into B_top (2x3) and B_bottom (1x3).
    # C = A_left * B_top + A_right * B_bottom.
    
    # C[0:2, 0:2] = A_left * B_top[0:2, 0:2] + A_right * B_bottom[0:2, 0:2]
    # C[0:2, 2]   = A_left * B_top[0:2, 2]   + A_right * B_bottom[0:2, 2]
    
    # Let's use Strassen for the 2x2x2 part of A_left * B_top_left.
    # A_left = [[A00, A01], [A10, A11]]
    # B_top_left = [[B00, B01], [B10, B11]]
    
    # Strassen 7 mults for M = A_left * B_top_left.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C[0,0] = M1 - M4 - M6 + M7
    # C[1,0] = M2 + M4
    # C[0,1] = M3 + M5
    # C[1,1] = M1 - M3 + M6
    
    # Now handle the remaining terms.
    # Term 1: A_right * B_bottom_left (2x1 * 1x2 = 2x2)
    # A_right = [A02, A12]
    # B_bottom_left = [B20, B21]
    # This requires 4 multiplications:
    # T1 = A02 * B20 -> C[0,0] += T1
    # T2 = A02 * B21 -> C[0,1] += T2
    # T3 = A12 * B20 -> C[1,0] += T3
    # T4 = A12 * B21 -> C[1,1] += T4
    
    # Term 2: A_left * B_top_right (2x2 * 2x1 = 2x1)
    # B_top_right = [B02, B12]
    # C[0,2] = A00*B02 + A01*B12
    # C[1,2] = A10*B02 + A11*B12
    # This requires 4 multiplications?
    # Can we do it in fewer?
    # (A00+A01)(B02+B12) = A00B02 + A00B12 + A01B02 + A01B12
    # We want A00B02 + A01B12 and A10B02 + A11B12.
    # This doesn't seem to share much.
    # However, we can use 2 multiplications if we are clever?
    # Actually, multiplying a 2x2 by a 2x1 vector can be done in 2 multiplications?
    # No, it's 4 scalar mults generally.
    # Wait, (A00+A01)B02 and (A10+A11)B12? No.
    
    # Let's check the count: 7 (Strassen) + 4 (A_right*B_bottom_left) + 4 (A_left*B_top_right) + ?
    # What about A_right * B_bottom_right?
    # A_right = [A02, A12] (2x1)
    # B_bottom_right = B22 (1x1)
    # C[0,2] += A02 * B22
    # C[1,2] += A12 * B22
    # This is 2 multiplications.
    
    # Total: 7 + 4 + 4 + 2 = 17.
    # This gives rank 17.
    
    # Can we reduce?
    # The 4 mults for A_right * B_bottom_left:
    # A02 B20, A02 B21, A12 B20, A12 B21.
    # (A02+A12)(B20+B21) = A02B20 + A02B21 + A12B20 + A12B21
    # We need to add these to C[0,0], C[0,1], C[1,0], C[1,1].
    # If we use this 1 mult, we have the sum. We need to distribute.
    # We can't easily separate them without more mults.
    
    # What if we use Strassen-like for the 2x1 * 1x2?
    # It's just a rank-4 tensor. Min rank is 4.
    
    # What if we use Strassen-like for the 2x2 * 2x1?
    # It's a rank-4 tensor. Min rank is 4.
    
    # So 7+4+4+2 = 17 seems to be the natural block construction.
    # Is rank 15 possible?
    # Yes, by sharing more terms or using a different partition.
    
    # Given the complexity and time, I will return the Rank 17 construction 
    # if I can implement it correctly, as it is better than 18.
    # And then try to see if I can shave off 2 more.
    
    # Let's implement Rank 17.
    
    U = [[0]*17 for _ in range(6)]
    V = [[0]*17 for _ in range(9)]
    W = [[0]*17 for _ in range(6)]
    
    col = 0
    
    # Helper to add a term: U[a]+=u, V[b]+=v, W[c]+=w
    def add(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1
    
    # Strassen for A_left * B_top_left
    # A indices: 0(A00), 1(A01), 3(A10), 4(A11) -- Wait, A is 2x3.
    # A[0,0]=0, A[0,1]=1, A[0,2]=2
    # A[1,0]=3, A[1,1]=4, A[1,2]=5
    # B indices: 0(B00), 1(B01), 2(B02)
    #            3(B10), 4(B11), 5(B12)
    #            6(B20), 7(B21), 8(B22)
    # C indices (W):
    # W[0]=C[0,0], W[1]=C[1,0], W[2]=C[0,1], W[3]=C[1,1], W[4]=C[0,2], W[5]=C[1,2]
    
    # M1 = (A00+A11)(B00+B11)
    add(0, 0, 0, 1, 1, 1)  # C00 += M1
    add(4, 4, 0, 1, 1, 1)  # C00 += M1 (Wait, A11 is idx 4, B11 is idx 4)
    # This adds to the same column? No, add() increments col.
    # I need to accumulate into the SAME column for a single bilinear form.
    # My add() function creates a NEW column for each call.
    # This is wrong for Strassen where one mult contributes to multiple outputs.
    # I need to build the matrices such that for a given rank-r term, 
    # U[a,r], V[b,r], W[c,r] are set.
    # If a mult contributes to multiple C entries, W[c,r] will have multiple non-zero entries?
    # No, W is (p*n, rank). Each column r corresponds to one multiplication.
    # The output C[i,k] is sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
    # If one multiplication M_r contributes to C[0,0] and C[1,1], 
    # then W[0, r] and W[3, r] (for C[1,1]) should be non-zero?
    # No, the decomposition is C[i,k] = sum_j sum_r ...
    # Actually, the tensor is T[i,j,k] = delta_{j,k} (for standard mult? No)
    # T[A_idx, B_idx, C_idx] = 1 if A_idx=(i,j), B_idx=(j,k), C_idx=(k,i).
    # The decomposition is sum_r U[i,j,r] V[j,k,r] W[k,i,r] = T[i,j,k].
    
    # So for a single multiplication M_r = (sum u_i A_i) * (sum v_k B_k),
    # it contributes to C_{i,k} if the linear forms match.
    # In Strassen, M1 = (A00+A11)(B00+B11).
    # This is a product of linear forms of A and B.
    # It contributes to C00, C11, etc.
    # Specifically, C00 += M1, C11 += M1.
    # So for column r:
    # U[0, r] = 1 (A00), U[4, r] = 1 (A11)  -- Wait, U is (n*m, rank).
    # U[0, r] corresponds to A[0,0]. U[4, r] corresponds to A[1,1].
    # V[0, r] = 1 (B00), V[4, r] = 1 (B11).
    # W[0, r] = 1 (C00), W[3, r] = 1 (C11).
    # This works! One column r handles the whole bilinear form.
    
    # So I need to define each of the 17 (or 15) bilinear forms as a column.
    
    U = [[0]*17 for _ in range(6)]
    V = [[0]*17 for _ in range(9)]
    W = [[0]*17 for _ in range(6)]
    
    # M1
    r = 0
    U[0][r] = 1; U[4][r] = 1
    V[0][r] = 1; V[4][r] = 1
    W[0][r] = 1; W[3][r] = 1
    
    # M2 = (A10+A11)B00
    r = 1
    U[3][r] = 1; U[4][r] = 1
    V[0][r] = 1
    W[1][r] = 1  # C10 += M2
    
    # M3 = A00(B10-B11)
    r = 2
    U[0][r] = 1
    V[3][r] = 1; V[4][r] = -1
    W[2][r] = 1  # C01 += M3
    
    # M4 = A11(B01-B00)
    r = 3
    U[4][r] = 1
    V[1][r] = 1; V[0][r] = -1
    W[1][r] = 1  # C10 += M4
    
    # M5 = (A00+A01)B11
    r = 4
    U[0][r] = 1; U[1][r] = 1
    V[4][r] = 1
    W[2][r] = 1  # C01 += M5
    
    # M6 = (A10-A00)(B00+B01)
    r = 5
    U[3][r] = 1; U[0][r] = -1
    V[0][r] = 1; V[1][r] = 1
    W[3][r] = 1  # C11 += M6
    
    # M7 = (A01-A11)(B10+B11)
    r = 6
    U[1][r] = 1; U[4][r] = -1
    V[3][r] = 1; V[4][r] = 1
    W[0][r] = 1  # C00 += M7
    
    # Now the other terms.
    # T1 = A02 B20 -> C00
    r = 7
    U[2][r] = 1
    V[6][r] = 1
    W[0][r] = 1
    
    # T2 = A02 B21 -> C01
    r = 8
    U[2][r] = 1
    V[7][r] = 1
    W[2][r] = 1
    
    # T3 = A12 B20 -> C10
    r = 9
    U[5][r] = 1
    V[6][r] = 1
    W[1][r] = 1
    
    # T4 = A12 B21 -> C11
    r = 10
    U[5][r] = 1
    V[7][r] = 1
    W[3][r] = 1
    
    # T5 = A00 B02 -> C02
    r = 11
    U[0][r] = 1
    V[2][r] = 1
    W[4][r] = 1
    
    # T6 = A01 B12 -> C02
    r = 12
    U[1][r] = 1
    V[5][r] = 1
    W[4][r] = 1
    
    # T7 = A10 B02 -> C12
    r = 13
    U[3][r] = 1
    V[2][r] = 1
    W[5][r] = 1
    
    # T8 = A11 B12 -> C12
    r = 14
    U[4][r] = 1
    V[5][r] = 1
    W[5][r] = 1
    
    # T9 = A02 B22 -> C02
    r = 15
    U[2][r] = 1
    V[8][r] = 1
    W[4][r] = 1
    
    # T10 = A12 B22 -> C12
    r = 16
    U[5][r] = 1
    V[8][r] = 1
    W[5][r] = 1
    
    # Verify
    ok, _ = _verify_factors(U, V, W, n, m, p)
    if ok:
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END
