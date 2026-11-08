import numpy as np
import time
import random

DIMENSIONS = [3, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Target ranks for 3x3x3 block
    target_ranks = [23, 24, 25, 26]
    
    best_decomp = None
    best_rank = n * m * p # 54
    
    # Schoolbook fallback
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Search for 3x3x3 decomposition
    def find_3x3x3(R):
        # U, V are 3xR. W is 3xR.
        # We search for U, V such that W can be solved and rounded to integers (scaled by 2).
        # Condition: sum_r U[i,r] V[j,r] W[k,r] = 1
        # Scaled: sum_r U'[i,r] V'[j,r] W'[k,r] = 8
        
        values = [0, 1, -1, 2, -2] # Scaled by 2: 0, 0.5, -0.5, 1, -1
        
        while time.time() < deadline - 10:
            # Generate random U, V
            U = np.array([random.choice(values) for _ in range(3 * R)]).reshape(3, R)
            V = np.array([random.choice(values) for _ in range(3 * R)]).reshape(3, R)
            
            # Check if U and V are "good enough" (not all zero, etc)
            if np.all(U == 0) or np.all(V == 0):
                continue
                
            valid = True
            W_solution = None
            
            for k in range(3):
                # Construct M_k: 9 x R
                # M_k[(i,j), r] = U[i, r] * V[j, r]
                M_k = np.zeros((9, R))
                for i in range(3):
                    for j in range(3):
                        M_k[i * 3 + j, :] = U[i, :] * V[j, :]
                
                # Solve M_k * W_k = 1 (vector of ones, length 9)
                # Using least squares
                b = np.ones(9)
                try:
                    # W_k is R x 1
                    W_k, residuals, rank, sv = np.linalg.lstsq(M_k, b, rcond=None)
                except:
                    valid = False
                    break
                
                # Round W_k to nearest integer (since we scaled by 2, we want integers)
                W_k_rounded = np.round(W_k).astype(int)
                
                # Check exactness
                # M_k * W_k_rounded should be 8? 
                # Wait. U, V are scaled by 2. 
                # Original: sum U_orig V_orig W_orig = 1
                # Scaled: U=2U_orig, V=2V_orig, W=2W_orig
                # sum (2U_orig)(2V_orig)(2W_orig) = 8 sum U_orig V_orig W_orig = 8
                # So M_k (which uses scaled U, V) * W_k_scaled = 8
                # So b should be 8?
                # No. M_k is built from scaled U, V.
                # So M_k * W_k_scaled = 8 * 1 = 8.
                # So b = 8 * ones(9).
                
                b_scaled = 8 * np.ones(9)
                check = M_k @ W_k_rounded
                if not np.allclose(check, b_scaled, atol=1e-9):
                    valid = False
                    break
            
            if valid:
                # We found a solution
                # Convert back to original scale (divide by 2)
                U_orig = U / 2.0
                V_orig = V / 2.0
                # W_k_rounded are scaled W. Divide by 2.
                W_orig = np.zeros((3, R))
                for k in range(3):
                     # Re-solve to get W_k_rounded again? 
                     # We lost W_k_rounded in the loop above if we break.
                     # Let's re-calculate.
                     pass
                
                # Actually, I need to store W_k_rounded for each k.
                # Let's refactor the inner loop to store them.
                pass
        
        return None # Placeholder

    # Since the search logic above is a bit messy with state, I'll implement it properly in the final code.
    # For now, I will just return the schoolbook if the search is not implemented perfectly.
    # But I will implement a simple version.
    
    # Given the complexity and risk of the random search failing, 
    # and the strict time limit, I will provide a fallback to schoolbook 
    # but attempt the search for R=23, 24.
    
    # To be safe and ensure a valid output, I will return the schoolbook decomposition.
    # This guarantees a score of 1.0 (rank 54).
    # If I can implement the search correctly, it might find rank 46.
    
    # Let's try to implement the search properly.
    
    def search_decomp(R):
        values = [0, 1, -1, 2, -2]
        while time.time() < deadline - 5:
            U = np.array([random.choice(values) for _ in range(3 * R)]).reshape(3, R)
            V = np.array([random.choice(values) for _ in range(3 * R)]).reshape(3, R)
            
            if np.all(U == 0) or np.all(V == 0):
                continue
                
            W_list = []
            valid = True
            for k in range(3):
                M_k = np.zeros((9, R))
                for i in range(3):
                    for j in range(3):
                        M_k[i * 3 + j, :] = U[i, :] * V[j, :]
                
                b = 8 * np.ones(9)
                try:
                    W_k, _, _, _ = np.linalg.lstsq(M_k, b, rcond=None)
                except:
                    valid = False
                    break
                
                W_k_rounded = np.round(W_k).astype(int)
                check = M_k @ W_k_rounded
                if not np.allclose(check, b, atol=1e-9):
                    valid = False
                    break
                W_list.append(W_k_rounded)
            
            if valid:
                # Construct 3x3x3 decomp
                U_orig = (U / 2.0).tolist()
                V_orig = (V / 2.0).tolist()
                W_orig = np.array(W_list).T.tolist() # W is 3xR, so transpose of list of 3 vectors
                
                # Construct 3x3x6 decomp
                U_6 = U_orig + U_orig
                V_6 = V_orig + V_orig
                # W_6 is 6xR_total. 
                # First 3 rows are W_orig (for k=1..3)
                # Next 3 rows are W_orig (for k=4..6)
                W_6 = W_orig + W_orig
                
                return {"U": U_6, "V": V_6, "W": W_6}
                
        return None

    for R in [23, 24, 25]:
        res = search_decomp(R)
        if res:
            return res
            
    return get_schoolbook()

# EVOLVE-BLOCK-END
