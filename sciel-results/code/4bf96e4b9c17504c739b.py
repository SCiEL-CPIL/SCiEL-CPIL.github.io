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
    
    # Strategy: Random U, V; Solve for W linearly.
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
    
    # Let's try ranks from 59 down to 30, but keep track of the best (lowest) rank found.
    # If we find a solution at r=50, we continue to r=49, 48, etc.
    # If we find a solution at r=59, we continue to 58.
    # This way we find the minimum rank.
    
    # However, if r=59 is infeasible, we waste time.
    # Let's assume it's feasible for r close to 60.
    
    # Let's try a different approach:
    # Fix U and V to be sparse?
    # Or use the identity matrix for U?
    # If U is identity-like, the problem simplifies.
    
    # Let's stick to random search but optimize the loop.
    # We will try r from 59 down to 30.
    # For each r, we try a few random instances.
    # If we find a solution, we update best_rank.
    # We continue to lower r.
    # If we find a solution at r, we know best_rank <= r.
    # We should stop if we reach r where time is running out.
    
    # Actually, if we iterate r from 59 down to 30, and find a solution at 50,
    # we should continue to 49.
    # If we find a solution at 49, we continue to 48.
    # This ensures we find the minimum rank (within the range we search).
    
    # Let's do that.
    
    for r in range(59, 29, -1):
        if time.time() > deadline - 5:
            break
            
        # If we have already found a solution at a lower rank, we can stop?
        # No, we are going down. If we found a solution at 50, we are now at 49.
        # We want to see if 49 works.
        # If we find a solution at 49, we update best_rank to 49.
        # We continue to 48.
        
        attempts = 0
        # More attempts for lower ranks? Or more attempts for higher ranks?
        # Higher ranks are more likely to have a solution.
        # Let's give more attempts to higher ranks to find a "good" starting point?
        # No, we just need ANY solution.
        # Let's use a fixed number of attempts per rank.
        max_attempts = 2000 if r > 50 else 5000
        
        while attempts < max_attempts and time.time() < deadline - 5:
            attempts += 1
            
            U_flat = np.random.choice([-1, 1], size=(n * m, r))
            V_flat = np.random.choice([-1, 1], size=(m * p, r))
            
            valid_solution = True
            W_solution = np.zeros((p * n, r))
            
            for i in range(n):
                for k in range(p):
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = U_flat[i * m + j, :] * V_flat[j * p + k, :]
                    
                    try:
                        # Solve A w = 1
                        # We want w to be half-integers.
                        # Let w = 0.5 * z, where z is integer.
                        # Then A (0.5 z) = 1 => 2 A z = 2 => A z = 1.
                        # Wait, if w is half-integer, 2w is integer.
                        # Let z = 2w. Then A (z/2) = 1 => A z = 2.
                        # So we solve A z = 2.
                        # If z is integer, then w is half-integer.
                        
                        # Let's solve A z = 2.
                        # If the solution z is close to an integer, we have a valid w.
                        
                        # Using lstsq to find z that minimizes ||Az - 2||.
                        # If the residual is small, check if z is integer-like.
                        
                        z, residuals, rank_A, sv = np.linalg.lstsq(A, np.full(m, 2.0), rcond=None)
                        
                        # Check exactness
                        if np.allclose(A @ z, np.full(m, 2.0), atol=1e-9):
                            # Check if z is close to integer
                            if np.allclose(z, np.round(z), atol=1e-6):
                                W_solution[k * n + i, :] = 0.5 * np.round(z)
                            else:
                                valid_solution = False
                                break
                        else:
                            valid_solution = False
                            break
                            
                    except np.linalg.LinAlgError:
                        valid_solution = False
                        break
                        
                if not valid_solution:
                    break
            
            if valid_solution:
                best_rank = r
                best_U, best_V, best_W = U_flat, V_flat, W_solution
                break  # Found a solution for this r, move to lower r
        
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
