import numpy as np
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin before 120s limit
    
    # 1. Schoolbook Solution (Rank = n*m*p = 84)
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1

    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    # 2. Pre-compute the Linear System Structure
    # We seek U, V, W such that for all A, B:
    # (A B)_{ik} = sum_j A_{ij} B_{jk}
    # Decomposition:
    # (A B)_{ik} = sum_r (sum_j U_{ij,r} A_{ij}) (sum_k V_{jk,r} B_{jk}) W_{ik,r}
    # Note: The problem statement says W[k*n+i, r].
    # Let's stick to the standard tensor decomposition form where the output index is (i, k).
    # The term for output (i, k) is sum_r (U^T A)_r (V^T B)_r W_{ik, r}.
    
    # We fix U and V (random integers) and solve for W.
    # The condition is that the tensor T_{ik, ij, jk} = delta_{ij, ij} delta_{jk, jk} (conceptually)
    # is represented as sum_r U_{ij, r} V_{jk, r} W_{ik, r}.
    
    # This is a linear system for W given U, V.
    # Variables: W_{ik, r} for all i, k, r. Total variables: n * p * R.
    # Equations: For each triple (i, j, k), we have a constraint.
    # Wait, the "equations" are actually the definition of the multiplication.
    # The multiplication tensor T is fixed.
    # T_{(i,j), (j,k), (i,k)} = 1 if the indices match the multiplication rule, else 0?
    # No, the multiplication tensor T has T_{(i,j), (j,k), (i,k)} = 1 if the indices are valid for the product term A_{ij}B_{jk} contributing to C_{ik}.
    # Actually, the tensor is defined such that T(A, B) = C.
    # T is a 3rd order tensor with indices (i, j), (j, k), (i, k).
    # T_{(i,j), (j,k), (i,k)} = 1.
    
    # We want to approximate T by sum_r u_r (tensor) v_r (tensor) w_r.
    # u_r has indices (i, j), v_r has indices (j, k), w_r has indices (i, k).
    # (sum_r u_r v_r w_r)_{(i,j), (j,k), (i,k)} = sum_r U_{ij, r} V_{jk, r} W_{ik, r}.
    # We require this sum to equal T_{(i,j), (j,k), (i,k)} for all i, j, k.
    
    # So for each (i, j, k), we have the equation:
    # sum_r U_{ij, r} V_{jk, r} W_{ik, r} = 1
    
    # This is a system of n * m * p equations (3*4*7 = 84 equations).
    # Unknowns: W_{ik, r} for all i, k, r.
    # Number of unknowns: n * p * R = 3 * 7 * R = 21 * R.
    
    # If R > 4 (since 84/21 = 4), the system is underdetermined.
    # We can find a solution W for any U, V?
    # Not any U, V. The matrix of the system must have full row rank (84).
    # If it does, there are infinitely many solutions.
    # We want a solution with integer/half-integer entries.
    
    # Let's construct the matrix M of the system.
    # Rows indexed by (i, j, k).
    # Columns indexed by (i, k, r).
    # M_{(i,j,k), (i', k', r)} = 1 if (i', k') == (i, k) and U_{ij, r} V_{jk, r} != 0?
    # No.
    # Equation for (i, j, k):
    # sum_{i', k', r} [ (i', k') == (i, k) ] * U_{ij, r} V_{jk, r} * W_{i' k', r} = 1
    # Since W is only non-zero for the specific i, k in the sum term?
    # No, W is a variable for all i, k.
    # But in the sum for a specific (i, j, k), only the term where the output index is (i, k) matters?
    # Yes, because T_{(i,j), (j,k), (i', k')} is 0 unless (i', k') = (i, k).
    # So the equation is simply:
    # sum_r U_{ij, r} V_{jk, r} W_{ik, r} = 1
    
    # This decouples!
    # For each (i, k), we have a subset of equations?
    # No, W_{ik, r} appears in equations for all j.
    # For fixed (i, k), the equations are for all j in [0, m).
    # sum_r U_{ij, r} V_{jk, r} W_{ik, r} = 1  for all j.
    
    # So for each pair (i, k), we have a system of m equations (4 equations)
    # with R unknowns (W_{ik, 0}, ..., W_{ik, R-1}).
    # Let A_{ik} be the m x R matrix with entries A_{ik, j, r} = U_{ij, r} V_{jk, r}.
    # Let b_{ik} be the vector of ones of length m.
    # We need to solve A_{ik} w_{ik} = b_{ik}.
    
    # This is much simpler than the global system!
    # We just need to find U, V such that for all (i, k), the system A_{ik} w = 1 has a solution
    # where w can be chosen to be half-integers.
    # Actually, since we can choose W, we just need A_{ik} to have full row rank (rank m=4).
    # If rank(A_{ik}) = 4, there are solutions.
    # We need the solution to be half-integers.
    
    # Strategy:
    # Iterate R from 4 upwards? No, start from 83 downwards?
    # Wait, if R is large, it's easier to find U, V with full rank sub-blocks.
    # But we want MINIMAL rank.
    # Theoretical lower bound:
    # The rank of the matrix multiplication tensor is at least max(n*m, m*p, p*n) = max(12, 28, 21) = 28.
    # Also, it's known that rank(3x4x7) is likely > 28.
    # Schoolbook is 84.
    # Let's search R from 28 to 83.
    
    # However, checking if a solution is half-integer is tricky.
    # If A_{ik} is 4xR with integer entries, and we solve A w = 1.
    # The solution w is in the null space of A^T A - A^T 1? No.
    # w = A^+ 1 + (I - A^+ A) z.
    # We want w to be half-integer.
    
    # Simpler approach:
    # Try random U, V with small integer entries (-1, 0, 1).
    # For each (i, k), check if rank(A_{ik}) == 4.
    # If yes, find a particular solution w_part.
    # If w_part is half-integer (or close to it), we are good.
    # If not, we might need to adjust.
    
    # But wait, if we fix U and V to be 0/1 matrices, the entries of A_{ik} are integers.
    # The solution w to A w = 1 will be rational.
    # We need w to be half-integer.
    
    # Let's try to find such U, V.
    
    def check_and_solve(U, V, R):
        """
        Checks if for given U, V, we can find W with half-integer entries.
        Returns W if found, else None.
        """
        W = np.zeros((n * p, R))
        
        for i in range(n):
            for k in range(p):
                # Construct A_ik: (m, R)
                A_ik = np.zeros((m, R))
                for j in range(m):
                    for r in range(R):
                        A_ik[j, r] = U[i * m + j, r] * V[j * p + k, r]
                
                # Solve A_ik w = 1
                # b is ones
                b = np.ones(m)
                
                # Check rank
                if np.linalg.matrix_rank(A_ik) < m:
                    return None
                
                # Find a solution
                # We want a solution that is half-integer.
                # Let's first find the least squares solution (which is exact if consistent)
                w, residuals, rank, s = np.linalg.lstsq(A_ik, b, rcond=None)
                
                # Check if residual is small
                if len(residuals) > 0 and residuals[0] > 1e-9:
                    # This shouldn't happen if rank is full and we're in consistent case?
                    # Actually, if rank(A) < m, we returned None.
                    # If rank(A) == m, and b is in col(A), residual is 0.
                    # Is b always in col(A)?
                    # Not necessarily. The system might be inconsistent.
                    # If inconsistent, no solution exists for this U, V.
                    return None
                
                # w is a solution.
                # Check if w is half-integer
                w2 = w * 2
                w2_rounded = np.round(w2)
                if np.allclose(w2, w2_rounded, atol=1e-9):
                    # It's a half-integer solution
                    W[k * n + i, :] = w
                else:
                    # The particular solution is not half-integer.
                    # Can we find another solution in the null space that is?
                    # w = w_part + N z.
                    # This is a complex Diophantine problem.
                    # For now, reject this U, V pair.
                    return None
        
        return W

    # Search for R
    # Start from 28 (lower bound) up to 83.
    # Actually, starting from 28 might be slow if many fail.
    # Starting from 83 down might find a solution faster but higher rank.
    # We want MINIMAL. So we must check 28, 29, ...
    
    # However, finding a valid U, V for low R is hard.
    # Let's try a few random trials for each R.
    
    best_found = None
    
    # Heuristic: R needs to be at least m=4 for each block.
    # Let's iterate R from 28 to 83.
    
    for R in range(28, 84):
        if time.time() > deadline - 5.0:
            break
            
        # Number of trials depends on time remaining
        # Estimate time per trial:
        # n*p = 21 blocks.
        # Each block: 4xR matrix. Rank check + lstsq.
        # Very fast.
        # Let's do 100 trials per R.
        
        found = False
        for trial in range(100):
            if time.time() > deadline - 5.0:
                break
                
            # Random U, V with entries in {-1, 0, 1}
            # Using 0/1 might be better for rank?
            # Let's use {-1, 1} to avoid zero rows/cols which might reduce rank.
            U = np.random.choice([-1, 1], size=(n * m, R))
            V = np.random.choice([-1, 1], size=(m * p, R))
            
            W_sol = check_and_solve(U, V, R)
            
            if W_sol is not None:
                # Found a valid decomposition for rank R
                # Verify it just in case
                # (Optional: quick check)
                
                best_rank = R
                best_U = U
                best_V = V
                best_W = W_sol
                found = True
                break # Found one for this R, try next R (lower? No, R is increasing)
        
        if found:
            # We found a solution for R.
            # Since we iterate R upwards, this is the minimal R found so far.
            # Can we find a smaller R? No, we started at 28.
            # So this is the minimal possible if 28 was the lower bound.
            # Wait, if we found it at R=30, we should check if R=29 works?
            # No, the loop goes 28, 29, 30...
            # If it failed at 28, 29, and succeeded at 30, then 30 is the minimal found.
            # We can break because we are looking for the MINIMAL rank.
            # If we found one at R, any R' < R already failed (or we didn't try enough trials).
            # To be safe, we might want to try a few more trials for lower R if we skipped them?
            # No, the loop structure ensures we try lower R first.
            # If we found a solution at R, it is the best we can get with the current search strategy.
            # We should break to save time.
            break

    # If no better found, return schoolbook
    if best_rank == rank_sb:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
# EVOLVE-BLOCK-END
