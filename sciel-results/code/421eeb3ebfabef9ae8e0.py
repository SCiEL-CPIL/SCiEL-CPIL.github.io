import time
import numpy as np
import random

def construct():
    n, m, p = 3, 4, 5
    deadline = time.time() + 115  # 115s to be safe within 120s

    # Schoolbook baseline
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
                
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # We search for a decomposition where U and V have entries in {-1, 1}
    # and W is solved exactly. If W has half-integer entries, it's valid.
    # We iterate rank from high to low to find the highest possible rank first?
    # No, we want the MINIMUM rank. So we should iterate from low to high.
    # But low ranks are very hard.
    # Let's try a range. Known bounds for 3x4x5?
    # Schoolbook is 60.
    # Let's try to find a rank < 60.
    # Start from 59 down to some lower bound?
    # Actually, if we find a solution at rank r, we should try to find one at r-1.
    # So we should iterate r from low to high.
    # But if r=25 is infeasible, we waste time.
    # Let's assume the minimum rank is close to 60.
    # Let's try ranks 59, 58, ... 30.
    # If we find a solution at 59, we return it? No, we should check 58.
    # So we must iterate from low to high, or keep the best found and check lower.
    # Let's iterate r from 30 to 59.
    # If we find a solution at r, we update best and continue to r+1?
    # No, if we find a solution at r, we know rank <= r.
    # We want the minimum. So we should try to find a solution at r, and if we do,
    # we can stop if we are iterating from low to high?
    # Yes. If we start at r=30 and find a solution, that's likely the minimum (or close).
    # But if r=30 is infeasible, we go to 31, etc.
    
    # Let's use a smarter search.
    # We know that for random U, V, the probability that W is half-integer is low.
    # We can try to construct U, V such that the linear system for W has a specific structure.
    
    # Alternative: Use the "tensor rank" of 3x4x5.
    # Let's just do the random search but more efficiently.
    # We will try ranks from 59 down to 30?
    # No, that finds the MAXIMUM rank we can find, not the minimum.
    # We want the MINIMUM rank.
    # So we must start from low.
    # Let's start from r=40.
    # If we find nothing by r=59, we return 60.
    
    # Actually, let's try to find a solution for r=59 first, just to see if it's easy.
    # If r=59 is easy, maybe r=58 is easy too.
    # Let's try a few ranks.
    
    # Let's try to use the fact that we can use half-integers.
    # This relaxes the constraint on W.
    
    def find_integer_solution(A, b):
        """
        Find an integer solution x to A x = b.
        A is (rows, R), b is (rows,).
        Returns None if no solution found or not integer.
        """
        try:
            x, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return None
            
        if np.allclose(x, np.round(x), atol=1e-6):
            return np.round(x).astype(int)
        
        return None

    def alternating_solve(R, max_iters=20):
        U = np.random.randint(-1, 2, size=(n * m, R))
        V = np.random.randint(-1, 2, size=(m * p, R))
        W = np.random.randint(-1, 2, size=(p * n, R))
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                return False, None, None, None
            
            U_new = np.zeros_like(U)
            success = True
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, R))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    b = np.ones(p)
                    sol = find_integer_solution(A, b)
                    if sol is None:
                        success = False
                        break
                    U_new[i * m + j, :] = sol
                if not success:
                    break
            if not success:
                return False, None, None, None
            U = U_new
            
            V_new = np.zeros_like(V)
            success = True
            for j in range(m):
                for k in range(p):
                    B = np.zeros((n, R))
                    for i in range(n):
                        B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    b = np.ones(n)
                    sol = find_integer_solution(B, b)
                    if sol is None:
                        success = False
                        break
                    V_new[j * p + k, :] = sol
                if not success:
                    break
            if not success:
                return False, None, None, None
            V = V_new
            
            W_new = np.zeros_like(W)
            success = True
            for k in range(p):
                for i in range(n):
                    C = np.zeros((m, R))
                    for j in range(m):
                        C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    b = np.ones(m)
                    sol = find_integer_solution(C, b)
                    if sol is None:
                        success = False
                        break
                    W_new[k * n + i, :] = sol
                if not success:
                    break
            if not success:
                return False, None, None, None
            W = W_new
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for s in range(R):
                            val += U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
                        if abs(val - 1) > 1e-6:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            if valid:
                return True, U, V, W

        return False, None, None, None

    for r in range(30, 61):
        if time.time() > deadline - 10:
            break
            
        attempts = 0
        max_attempts = 1000
        
        while attempts < max_attempts and time.time() < deadline - 5:
            attempts += 1
            found, U, V, W = alternating_solve(r)
            if found:
                best_rank = r
                best_U, best_V, best_W = U, V, W
                break
        
    # Convert to lists with integer/half-integer values
    def to_list(arr):
        # Round to nearest 0.5
        return [[float(round(x * 2) / 2) for x in row] for row in arr]

    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }

if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    print(f"Rank: {U.shape[1]}")
    print(f"U shape: {U.shape}, V shape: {V.shape}, W shape: {W.shape}")
    
    # Verification
    n, m, p = 3, 4, 5
    r = U.shape[1]
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for s in range(r):
                    val += U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
                if abs(val - 1) > 1e-6:
                    valid = False
                    print(f"Failed at i={i}, j={j}, k={k}, val={val}")
                    break
            if not valid: break
        if not valid: break
    print(f"Valid: {valid}")
