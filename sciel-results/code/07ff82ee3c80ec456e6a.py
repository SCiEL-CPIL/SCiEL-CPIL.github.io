"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Dimensions: A is (n x m) = (2 x 4), B is (m x p) = (4 x 5), C is (n x p) = (2 x 5)
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (20, rank)
    # W shape: (p*n, rank) = (10, rank)
    # Indexing:
    # U[i*m + j, r]
    # V[j*p + k, r]
    # W[k*n + i, r]
    # Tensor equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if i==i', j==j', k==k' else 0?
    # Actually, matrix multiplication tensor T_{i,j,k} = 1.
    # The decomposition is: T_{i,j,k} = sum_r U_{i,j}^r V_{j,k}^r W_{k,i}^r ?
    # Let's check the standard convention.
    # Usually: (A B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # Tensor form: T_{i,j,k} = 1.
    # Decomposition: 1 = sum_r U_{i,j}^r V_{j,k}^r W_{k,i}^r ?
    # The prompt says: "output factor follows the official trace convention W[k*n+i,r]".
    # And "exactly decompose the matrix multiplication tensor".
    # The seed program uses:
    # U[i*m+j][column] = 1
    # V[j*p+k][column] = 1
    # W[k*n+i][column] = 1
    # This corresponds to: sum_r U_{i,j}^r V_{j,k}^r W_{k,i}^r = 1 for all i,j,k.
    # Since U, V, W are 1, the sum is 1 if the indices match the specific triple (i,j,k) associated with that column.
    # So yes, it's a decomposition of the all-ones tensor? No, the multiplication tensor has T_{i,j,k}=1.
    # So we need sum_r U_{i,j}^r V_{j,k}^r W_{k,i}^r = 1 for all i,j,k.

    # Known ranks for 2x4x5:
    # Schoolbook: 2*4*5 = 40.
    # Strassen-like for 2x2x2 is 7.
    # For 2x4x5, we can view it as blocks.
    # A is 2x4, B is 4x5.
    # Split A into A1 (2x2) and A2 (2x2). Split B into B1 (2x5) and B2 (2x5).
    # C = A1 B1 + A2 B2.
    # Rank of 2x2x5 multiplication?
    # Let's try to find a low rank decomposition.
    
    # Strategy: Use a known low-rank algorithm or search.
    # Given the 150s budget, we can do a small search or use a known result.
    # Let's check if rank 35 is possible.
    # The knowledge says: "Block-Strassen structure drops rank 40->36 ... may fold one Strassen term ... to reach 35".
    
    # Let's try to construct a rank 36 decomposition first, then try to optimize to 35.
    # Or just search for the minimal rank directly with a timeout.
    
    # Since we have time, let's try to solve for the minimal rank using a solver or random search with verification.
    # However, solving tensor decomposition is NP-hard.
    # We can try to use the "Strassen" style construction for the specific dimensions.
    
    # Let's define the target tensor T of size 2x4x5.
    # T[i,j,k] = 1.
    
    # We want to find U (8xR), V (20xR), W (10xR) such that:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    
    # This is a system of 40 equations (2*4*5) with 3*R variables (8R + 20R + 10R = 38R).
    # If R=1, 38 vars, 40 eqs. Underdetermined? No, 38 < 40.
    # If R=2, 76 vars, 40 eqs. Overdetermined.
    # We can try to find a solution with small integer entries.
    
    # Let's try a random search for small R.
    # But random search for tensor rank is hard.
    
    # Alternative: Use the fact that we can embed smaller multiplications.
    # 2x4x5 can be seen as two 2x2x5 multiplications? No.
    # A = [A1 A2], B = [B1; B2]. C = A1 B1 + A2 B2.
    # Rank(2x2x5) * 2?
    # What is the rank of 2x2x5?
    # Schoolbook: 2*2*5 = 20.
    # Can we do better?
    # Maybe use Strassen for the 2x2 part?
    # Strassen is for 2x2x2.
    
    # Let's try to construct a specific low-rank solution.
    # Reference: "Matrix multiplication 2,4,5"
    # Known optimal rank for 2x4x5 is likely 35 or 36.
    
    # Let's try to find a rank 35 decomposition.
    # We will use a simple local search or random initialization with gradient descent?
    # No, we need exact integer/half-integer solutions.
    
    # Let's try to use the structure of Strassen for 2x2x2 blocks.
    # If we could decompose 2x4x5 into 2x2x2 blocks, we could use Strassen.
    # But 4 and 5 are not powers of 2.
    
    # Let's try a different approach: use a SAT/SMT solver? Not available.
    # Use random search with verification.
    
    # Verification function:
    def verify(U, V, W):
        n, m, p = DIMENSIONS
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True

    # Let's try to find a rank 35 solution.
    # We can fix U and V to have small entries (0, 1, -1) and solve for W.
    # For a fixed U, V, the equation is:
    # sum_r U_{i,j}^r V_{j,k}^r W_{k,i}^r = 1
    # This is linear in W.
    # Let x_r = W_{k,i}^r.
    # The equations are coupled because W depends on k and i.
    # For each (k,i), we have a variable W_{k,i}^r for each r.
    # Actually, for each (i,j,k), the equation involves W_{k,i}^r.
    # So for a fixed (k,i), we have variables W_{k,i}^1 ... W_{k,i}^R.
    # The equations for all j (j=0..m-1) for a fixed (i,k) are:
    # sum_r U_{i,j}^r V_{j,k}^r W_{k,i}^r = 1 for j=0..m-1.
    # This is a system of m equations for R variables W_{k,i}^r.
    # If R >= m, we might have a solution.
    # Here m=4. So if R >= 4, we can potentially solve for W for each (k,i).
    # But the solution must be consistent across different (i,k) pairs? No, W_{k,i}^r is specific to (k,i).
    # So we can solve for W independently for each (k,i) pair!
    # This is a huge simplification.
    # We need to choose U and V such that for each (k,i), the system:
    # sum_r U_{i,j}^r V_{j,k}^r W_{k,i}^r = 1 for j=0..m-1
    # has a solution W_{k,i}^r with small integer/half-integer entries.
    
    # Let's try to find such U and V.
    # We can fix U and V to be random 0/1 matrices and check if the system is solvable with small W.
    # Or we can choose U and V to make the system trivial.
    
    # Let's try R=35.
    # We have 35 variables W_{k,i}^r for each (k,i).
    # We have 4 equations for each (k,i).
    # So we have many degrees of freedom.
    # We can set 31 of them to 0 and solve for 4.
    # This is easy.
    # The challenge is to find U and V such that the coefficients U_{i,j}^r V_{j,k}^r are non-zero for the chosen r's.
    
    # Let's try a simpler construction.
    # Set U and V to be identity-like matrices.
    # U[i*m+j, r] = 1 if r == i*m+j else 0.
    # V[j*p+k, r] = 1 if r == j*p+k else 0.
    # Then U_{i,j}^r V_{j,k}^r is non-zero only if r == i*m+j AND r == j*p+k.
    # This requires i*m+j == j*p+k.
    # This only happens for specific (i,j,k).
    # This is the schoolbook decomposition. Rank = 40.
    
    # To reduce rank, we need to share r's across different (i,j,k).
    # This is what Strassen does.
    
    # Let's try to use a known low-rank algorithm for 2x4x5.
    # I will implement a search for rank 35.
    
    # Given the time constraint, I will return the schoolbook rank 40 if I don't find a better one quickly.
    # But I have 150s.
    
    # Let's try to find a rank 36 decomposition using the block structure.
    # A = [A1 A2], B = [B1; B2].
    # C = A1 B1 + A2 B2.
    # If we use Strassen for 2x2x2, we need to pad B1, B2 to 2x2.
    # B1 is 2x5. We can split it into B1a (2x2) and B1b (2x3).
    # This seems complicated.
    
    # Let's try a random search for R=35.
    # Initialize U, V with random 0, 1, -1.
    # For each (k,i), solve for W.
    # Check if W has small entries.
    
    # To speed up, we can use the fact that we only need *a* solution.
    # We can set most W to 0 and solve a small system.
    
    # Let's try this approach.
    
    deadline = time.time() + 140
    best_rank = 40
    best_U = None
    best_V = None
    best_W = None
    
    # First, create the schoolbook solution as a fallback.
    U_sch = [[0]*40 for _ in range(8)]
    V_sch = [[0]*40 for _ in range(20)]
    W_sch = [[0]*40 for _ in range(10)]
    col = 0
    for i in range(2):
        for j in range(4):
            for k in range(5):
                U_sch[i*4+j][col] = 1
                V_sch[j*5+k][col] = 1
                W_sch[k*2+i][col] = 1
                col += 1
    best_U, best_V, best_W = U_sch, V_sch, W_sch
    best_rank = 40
    
    # Try to find lower ranks.
    for R in range(35, 40):
        if time.time() > deadline:
            break
        # Try a few random initializations
        for attempt in range(1000):
            if time.time() > deadline:
                break
            U = [[random.choice([0, 1, -1]) for _ in range(R)] for _ in range(8)]
            V = [[random.choice([0, 1, -1]) for _ in range(R)] for _ in range(20)]
            
            # Check if we can solve for W
            valid = True
            W = [[0]*R for _ in range(10)]
            for k in range(5):
                for i in range(2):
                    # System: sum_r U[i*4+j][r] * V[j*5+k][r] * W[k*2+i][r] = 1 for j=0..3
                    # Let A_jr = U[i*4+j][r] * V[j*5+k][r]
                    # sum_r A_jr * W_r = 1
                    # We have 4 equations, R variables.
                    # We can set R-4 variables to 0 and solve for the rest.
                    # Choose 4 indices where the matrix is invertible.
                    # For simplicity, try to find a solution with small W.
                    # We can use Gaussian elimination on a subset of columns.
                    
                    # Let's try to find 4 columns that form an invertible 4x4 matrix.
                    found = False
                    # Try all combinations of 4 columns? C(R,4) is large if R=35.
                    # C(35,4) = 52365. Too many.
                    # Instead, pick 4 random columns and check.
                    cols = random.sample(range(R), 4)
                    M = [[U[i*4+j][r] * V[j*5+k][r] for r in cols] for j in range(4)]
                    # Check if M is invertible
                    det = 1
                    # Simple determinant check for 4x4? Or just try to solve.
                    # Let's use fractions to be exact.
                    M_frac = [[Fraction(x) for x in row] for row in M]
                    # Gaussian elimination
                    try:
                        # Solve M * w = 1
                        # We can do this with simple elimination.
                        # If it fails, try another set of columns.
                        w = [Fraction(0) for _ in range(4)]
                        # Augmented matrix
                        Aug = [row[:] + [Fraction(1)] for row in M_frac]
                        # Eliminate
                        for c in range(4):
                            # Find pivot
                            pivot = -1
                            for r in range(c, 4):
                                if Aug[r][c] != 0:
                                    pivot = r
                                    break
                            if pivot == -1:
                                raise ZeroDivisionError
                            # Swap
                            Aug[c], Aug[pivot] = Aug[pivot], Aug[c]
                            # Normalize
                            pivot_val = Aug[c][c]
                            for cc in range(c, 5):
                                Aug[c][cc] /= pivot_val
                            # Eliminate
                            for r in range(4):
                                if r != c:
                                    factor = Aug[r][c]
                                    if factor != 0:
                                        for cc in range(c, 5):
                                            Aug[r][cc] -= factor * Aug[c][cc]
                        # Solution
                        w = [Aug[i][4] for i in range(4)]
                        # Check if w has small denominator
                        valid_w = True
                        for val in w:
                            if val.denominator > 2:
                                valid_w = False
                                break
                        if valid_w:
                            # Assign W
                            for idx, r in enumerate(cols):
                                W[k*2+i][r] = w[idx]
                            found = True
                            break
                    except ZeroDivisionError:
                        continue
                    
                    if not found:
                        valid = False
                        break
            
            if valid:
                # Verify the solution
                if verify(U, V, W):
                    best_rank = R
                    best_U, best_V, best_W = U, V, W
                    break # Found a solution for this R, try lower R
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
