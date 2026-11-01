import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def verify_decomposition(U, V, W, n, m, p):
    """
    Verifies if the given matrices U, V, W form a valid tensor decomposition.
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    
    The tensor T is defined such that T[a, b, c, d, i, k] = 1 if a==i, b==d, c==k else 0.
    We check if sum_l U[a*m+b, l] * V[c*p+d, l] * W[k*n+i, l] == T[a, b, c, d, i, k]
    """
    r = U.shape[1]
    # Precompute indices for speed
    for a in range(n):
        for b in range(m):
            u_idx = a * m + b
            for c in range(m):
                for d in range(p):
                    v_idx = c * p + d
                    for i in range(n):
                        for k in range(p):
                            w_idx = k * n + i
                            target = 1.0 if (a == i and b == d and c == k) else 0.0
                            s = 0.0
                            # Vectorized dot product over rank dimension
                            # s = np.dot(U[u_idx], V[v_idx] * W[w_idx])
                            # Since entries are integers/half-integers, we can use exact arithmetic if we scale, 
                            # but float is fine with tolerance.
                            for l in range(r):
                                s += U[u_idx, l] * V[v_idx, l] * W[w_idx, l]
                            if not np.isclose(s, target, atol=1e-8):
                                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s margin for safety
    
    n, m, p = DIMENSIONS
    
    # Target ranks to try, starting from a known good upper bound (schoolbook is n*m*p = 36)
    # Current best known is 34. We try to beat 34.
    # Theoretical lower bound for 2x3x6 is at least 2*3*6 / max(dim) ? No.
    # Rank of 2x3x3 is 7 (Strassen). 2x3x6 is 2x(2x3x3) -> 14? No, that's if they were independent.
    # Actually, 2x3x6 can be viewed as 2x3x3 + 2x3x3. If we can couple them, maybe rank < 14?
    # Wait, the current program uses rank 34. Why 34?
    # 2x3x3 rank is 7. Two blocks -> 14.
    # Why does the current program use 34?
    # Ah, the current program uses "U233" which is 6x17. 
    # 17 is the rank for 2x3x3? No, Strassen is 7. 
    # Let's check the current program's rank.
    # U_comb is (6, 34). So rank is 34.
    # Why 34? 17 * 2 = 34.
    # So the current program assumes rank(2x3x3) = 17? That is very high. 
    # Standard Strassen is 7.
    # Maybe the current program is not using Strassen correctly or I am misinterpreting the dimensions.
    # U233 is (6, 17). 6 = n*m = 2*3. So it's a 2x3x3 problem.
    # Rank 7 is known for 2x3x3. 
    # If we use rank 7 for each block, total rank 14.
    # Let's try to find a rank 14 decomposition.
    
    # Let's try ranks from 14 up to 33.
    # If we find a valid one, we return it.
    
    # We will use a randomized local search.
    # We initialize U, V, W with random values and try to satisfy the equations.
    # This is a system of polynomial equations.
    # For small ranks, we can try to solve it.
    
    # Strategy: 
    # 1. Try rank 14 (2 * 7).
    # 2. If fails, try rank 15, 16...
    # 3. Use a simple iterative solver or random restarts.
    
    # Since exact integer/half-integer solutions are hard to find with random floats,
    # we will search for a solution over Reals, then check if it can be rounded or if it's close to integers.
    # However, the problem requires exact integer/half-integer entries.
    # So we should search in the lattice of half-integers? Or just integers?
    # The prompt says "integer/half-integer".
    
    # A robust approach for small ranks:
    # Generate random integer matrices U, V, W of rank R.
    # Check if they satisfy the condition.
    # This is a needle in a haystack for R=14.
    
    # Alternative: Use the known Strassen decomposition for 2x3x3 and block it.
    # If we can construct a valid 2x3x3 decomposition of rank 7, we can just duplicate it for the two 3x3 blocks of the 3x6 matrix.
    # The 3x6 matrix is [A1 | A2] where A1, A2 are 3x3.
    # B is 2x3.
    # C = B * [A1 | A2] = [B*A1 | B*A2].
    # This is two independent multiplications.
    # So rank(2x3x6) = rank(2x3x3) + rank(2x3x3) = 14.
    # So we just need a rank 7 decomposition for 2x3x3.
    
    # Let's try to find a rank 7 decomposition for 2x3x3.
    # If we find one, we can easily construct the rank 14 solution for 2x3x6.
    
    def find_2x3x3_decomposition(rank_target, time_limit):
        n, m, p = 2, 3, 3
        # Strassen's algorithm for 2x2x2 has rank 7.
        # For 2x3x3, we can pad to 3x3x3? No.
        # We can view 2x3x3 as a sub-tensor of 3x3x3?
        # Or just search.
        
        # Let's try a simple random search for integers in [-2, 2].
        # If R=7, the number of variables is 7*(6+9+9) = 7*24 = 168.
        # The equations are 2*3*3*3*2*3 = 324 constraints.
        # It's underdetermined? No, 168 variables, 324 equations. Overdetermined.
        # But many equations are zero.
        
        # Actually, let's just use the known Strassen decomposition for 2x2x2 and extend?
        # No, 2x3x3 is not directly 2x2x2.
        
        # Let's try to find a solution using a gradient descent on the loss function:
        # Loss = sum (sum_l U V W - Target)^2
        # We can use scipy.optimize or just random restarts.
        
        # Given the time limit, let's try a few random integer initializations for R=7.
        # If that fails, we fall back to the block construction with Rank 14 (using a valid 2x3x3 decomp).
        
        # Wait, do we have a valid 2x3x3 decomp?
        # The current program has U233 of rank 17. That is very inefficient.
        # We can definitely do better.
        
        # Let's try to find a rank 7 decomp for 2x3x3.
        # If we fail, we return None.
        
        for attempt in range(100):
            if time.time() > time_limit:
                break
            U = np.random.randint(-2, 3, (6, rank_target))
            V = np.random.randint(-2, 3, (9, rank_target))
            W = np.random.randint(-2, 3, (9, rank_target))
            
            # Check
            valid = True
            for a in range(2):
                for b in range(3):
                    u_idx = a * 3 + b
                    for c in range(3):
                        for d in range(3):
                            v_idx = c * 3 + d
                            for i in range(2):
                                for k in range(3):
                                    w_idx = k * 2 + i
                                    target = 1.0 if (a == i and b == d and c == k) else 0.0
                                    s = 0.0
                                    for l in range(rank_target):
                                        s += U[u_idx, l] * V[v_idx, l] * W[w_idx, l]
                                    if not np.isclose(s, target, atol=1e-8):
                                        valid = False
                                        break
                                if not valid: break
                            if not valid: break
                        if not valid: break
                    if not valid: break
                if not valid: break
            if valid:
                return U, V, W
        return None

    # Try to find a 2x3x3 decomposition of rank 7
    U233_7, V233_7, W233_7 = None, None, None
    if time.time() < deadline - 10:
        U233_7, V233_7, W233_7 = find_2x3x3_decomposition(7, deadline - 10)
        
    if U233_7 is not None:
        # Construct 2x3x6 decomposition from 2x3x3
        # U_comb (6, 14)
        # V_comb (18, 14)
        # W_comb (12, 14)
        
        U_comb = np.zeros((6, 14))
        V_comb = np.zeros((18, 14))
        W_comb = np.zeros((12, 14))
        
        # First 7 columns for block 1 (cols 0-2 of 3x6)
        U_comb[:, :7] = U233_7
        # V for block 1: rows 0-8 (3x3)
        V_comb[:9, :7] = V233_7
        # W for block 1: rows 0-5 (3x2)
        W_comb[:6, :7] = W233_7
        
        # Next 7 columns for block 2 (cols 3-5 of 3x6)
        U_comb[:, 7:] = U233_7
        # V for block 2: rows 9-17
        V_comb[9:, 7:] = V233_7
        # W for block 2: rows 6-11
        W_comb[6:, 7:] = W233_7
        
        if verify_decomposition(U_comb, V_comb, W_comb, 2, 3, 6):
            return {
                "U": U_comb.tolist(),
                "V": V_comb.tolist(),
                "W": W_comb.tolist()
            }
            
    # If rank 7 search failed, try rank 8, 9... for 2x3x3?
    # Or just use a known valid decomposition for 2x3x3.
    # Let's try to find a valid 2x3x3 decomposition of rank 8 or 9 quickly.
    # Or fall back to the current program's logic but with a better rank if possible.
    
    # Actually, let's just use the schoolbook rank 36 if we can't find better.
    # But we know rank 14 is possible if we have a rank 7 decomp.
    # If we can't find rank 7, we can try rank 8.
    
    for r in range(8, 15):
        if time.time() > deadline - 5:
            break
        U233_r, V233_r, W233_r = find_2x3x3_decomposition(r, deadline - 5)
        if U233_r is not None:
            U_comb = np.zeros((6, 2*r))
            V_comb = np.zeros((18, 2*r))
            W_comb = np.zeros((12, 2*r))
            
            U_comb[:, :r] = U233_r
            V_comb[:9, :r] = V233_r
            W_comb[:6, :r] = W233_r
            
            U_comb[:, r:] = U233_r
            V_comb[9:, r:] = V233_r
            W_comb[6:, r:] = W233_r
            
            if verify_decomposition(U_comb, V_comb, W_comb, 2, 3, 6):
                return {
                    "U": U_comb.tolist(),
                    "V": V_comb.tolist(),
                    "W": W_comb.tolist()
                }
                
    # Fallback: Schoolbook
    U_school = np.zeros((6, 36))
    V_school = np.zeros((18, 36))
    W_school = np.zeros((12, 36))
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(6):
                U_school[i * 3 + j, col] = 1
                V_school[j * 6 + k, col] = 1
                W_school[k * 2 + i, col] = 1
                col += 1
                
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }

if __name__ == "__main__":
    construct()
