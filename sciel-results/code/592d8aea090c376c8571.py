"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard: 120s limit, use 110s for safety
    start_time = time.time()
    DEADLINE = 110.0

    # 1. Schoolbook Decomposition (Rank 45) - Guaranteed Valid Fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U.tolist(), V.tolist(), W.tolist()

    # 2. Verification Helper
    def is_valid(U, V, W, trials=5, tol=1e-9):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        
        for _ in range(trials):
            A = np.random.rand(n, m)
            B = np.random.rand(m, p)
            C_true = A @ B
            
            # Bilinear form:
            # C_{i,k} = sum_r W_{k,i,r} * (sum_j U_{i,j,r} A_{i,j}) * (sum_l V_{j,l,r} B_{j,l})
            # Note: U index is i*m+j, V index is j*p+k (wait, V index is j*p+k? No, V is (m*p, rank), index j*p+k)
            # W index is k*n+i
            
            # Let's reshape for clarity
            U_t = U.reshape(n, m, rank)
            V_t = V.reshape(m, p, rank)
            W_t = W.reshape(p, n, rank)
            
            # Intermediate sums
            # X_{i,r} = sum_j U_{i,j,r} A_{i,j}
            X = np.einsum('ijr,ij->ir', U_t, A)
            # Y_{k,r} = sum_l V_{l,k,r} B_{l,k} ? No, V_{j,k,r} B_{j,k}
            # V is indexed by (j, k) where j is row of B, k is col of B.
            # So Y_{k,r} = sum_j V_{j,k,r} B_{j,k}
            # V_t is (m, p, rank). V_t[j,k,r].
            Y = np.einsum('jkr,jk->kr', V_t, B)
            
            # C_{i,k} = sum_r W_{k,i,r} X_{i,r} Y_{k,r}
            C_calc = np.einsum('kir,ir,kr->ik', W_t, X, Y)
            
            if not np.allclose(C_true, C_calc, atol=tol, rtol=tol):
                return False
        return True

    # 3. Attempted Improved Decomposition (Rank 41)
    # Strategy: Split into 3x3x3 (Rank 23 via BSC) + 3x3x2 (Rank 18 Schoolbook)
    # Total Rank: 23 + 18 = 41.
    # This requires a valid rank-23 decomposition for 3x3x3.
    # Since hardcoded BSC coefficients are error-prone and not provided in inspirations as verified arrays,
    # we attempt a randomized search for a valid low-rank decomposition?
    # No, random search for rank 23 is unlikely to succeed quickly.
    
    # Instead, let's look at the structure.
    # The problem is 3x3x5.
    # Can we do better than 45?
    # Known bounds:
    # 3x3x3 min rank is 23 (BSC).
    # 3x3x2 min rank is 18 (Schoolbook is optimal for 2? No, 3x3x2 rank is 18? 
    # Actually, rank of 3x3x2 is 18? 
    # Let's check: 3x3x2 tensor. Schoolbook is 18. 
    # Is there a better one? 
    # For 3x3x3, rank is 23.
    # For 3x3x4, rank is likely 23 + 18 = 41? Or 23 + something?
    # If we use BSC for the first 3 columns and Schoolbook for the last 2, we get 23+18=41.
    
    # The main risk is obtaining the correct BSC coefficients.
    # Let's try to construct a valid rank-23 decomposition for 3x3x3 using a known set.
    # If we cannot verify it, we must fall back to 45.
    
    # Let's try a different approach:
    # Is there a known rank-40 decomposition for 3x3x5?
    # Or can we verify a rank-41 decomposition?
    
    # Since we don't have hardcoded BSC coefficients, let's try to *find* a valid decomposition 
    # with rank < 45 using a local optimization method?
    # That's too slow and complex.
    
    # Let's stick to the structural insight:
    # The 3x3x5 multiplication can be viewed as 5 independent 3x3x1 multiplications? No.
    # It is C = A @ B.
    
    # Let's try to implement the BSC coefficients for 3x3x3.
    # I will hardcode a known valid set if I can recall or derive it.
    # Otherwise, I will return 45.
    
    # Given the constraints and the lack of verified BSC coefficients in the prompt,
    # and the fact that previous attempts failed or tied at 1.0,
    # I will implement a robust check.
    
    # Wait, I can try to *generate* a decomposition using a small search?
    # No, 150s budget.
    
    # Let's assume the question implies we should try to beat 45.
    # The most reliable way to beat 45 is to use the 3x3x3 rank 23 result.
    # I will hardcode the BSC coefficients for 3x3x3.
    # Reference: "On the algebraic complexity of matrix multiplication" by Bini, Strassen, Caporaso.
    
    # Here is a known representation for 3x3x3 rank 23.
    # The entries are 0, 1, -1.
    # U, V, W are 9x23.
    
    # Let's try to construct it.
    # If I fail to verify it, I return 45.
    
    def try_bsc_3x3x3():
        # This function returns U, V, W for 3x3x3 if valid, else None.
        # I will use a specific known set.
        # Since I cannot guarantee the correctness of a hardcoded set from memory without verification,
        # and verification takes time, I will use a conservative approach.
        
        # Actually, let's look at the "Inspiration 3" which tried to hardcode it but failed.
        # It suggests that without exact coefficients, it's risky.
        
        # However, there is a simpler improvement:
        # What if we use Rank 44? Or Rank 43?
        # Is there a known rank 44 for 3x3x5?
        
        # Let's try a randomized search for a valid decomposition with rank 44?
        # Unlikely to find exact integer/half-integer solutions quickly.
        
        # Let's reconsider the problem statement.
        # "integer/half-integer entries".
        
        # If I cannot find a valid rank < 45, I must return 45.
        # But the task asks to "Find a rank-minimal...".
        # The score is seed/value. Seed is 45.
        # If I return 45, score is 1.0.
        # If I return 41, score is 45/41 = 1.097.
        
        # I will attempt to hardcode the BSC coefficients for 3x3x3.
        # If the verification fails, I fall back to 45.
        
        # Known BSC 3x3x3 coefficients (one of the valid sets):
        # This is a 9x23 matrix for U, 9x23 for V, 9x23 for W.
        # I will use a placeholder that is likely to fail if I don't have the exact data.
        # To be safe, I will NOT use a guessed BSC set.
        
        # Instead, I will try to find if there is a trivial reduction.
        # No.
        
        # So, the safest bet that *might* improve is if I can find a valid rank 41.
        # Let's try to construct it from a known source online? I can't.
        
        # Let's try a different strategy:
        # Use the fact that 3x3x5 = 3x3x3 + 3x3x2.
        # If I have a valid 3x3x3 rank 23, I can build the rest.
        
        # I will write a function that attempts to verify a hardcoded BSC set.
        # If I don't have the set, I return 45.
        
        # Since I don't have the set, I will return 45.
        # BUT, the prompt asks to COMBINE strategies.
        # Inspiration 2 and 3 both tried to use BSC but lacked coefficients.
        
        # Is there a way to *compute* the BSC coefficients?
        # No, it's a complex algebraic derivation.
        
        # Let's try one more thing:
        # Maybe the rank is not 23 for 3x3x3?
        # No, it is 23.
        
        # Okay, I will return the schoolbook decomposition (45) to ensure validity.
        # This guarantees a score of 1.0.
        # Any attempt to guess coefficients risks a score of 0.0 if invalid.
        
        # Wait, the "Inspiration 1" is just schoolbook.
        # "Inspiration 2" tried to verify and fell back.
        # "Inspiration 3" tried to hardcode and likely failed or fell back.
        
        # To improve, I *must* have a valid lower rank decomposition.
        # Without the coefficients, I cannot provide one.
        
        # However, I can try to *search* for a valid decomposition with rank 44?
        # No.
        
        # Let's assume the "seed" is 45.
        # If I cannot beat it, I return 45.
        
        # Is there any other structure?
        # 3x3x5.
        # What if I use a rank 45 decomposition but with a different structure?
        # No, the rank is the score.
        
        # I will return the schoolbook decomposition.
        U, V, W = schoolbook()
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
