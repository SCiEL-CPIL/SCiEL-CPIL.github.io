"""Matrix multiplication 2x3 by 3x6: rank-minimal exact tensor decomposition search."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x6 matrix.
    Output factor follows W[k*n+i, r] convention.
    
    Strategy:
    1. Start with schoolbook rank 36.
    2. Try to reduce rank by finding linear dependencies among the rank-1 terms.
    3. Use a structured search over potential rank reductions.
    4. For small n*m*p, we can try all subsets or use algebraic methods.
    
    For 2x3x6, the tensor has 36 entries. We seek the minimum rank.
    
    Known bounds: 
    - Schoolbook: 36
    - Strassen-like improvements possible for matrix multiplication.
    
    For small matrices, we can try to find the exact minimal rank by:
    1. Setting up the tensor as a 3-tensor T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to (i,k) of C.
    2. The tensor is T[i,j,k] = 1 for all valid (i,j,k) where 0<=i<2, 0<=j<3, 0<=k<6.
    3. We want to write T as sum of r rank-1 tensors: T = sum_{c=0}^{r-1} u_c ⊗ v_c ⊗ w_c
    
    Where u_c has entries indexed by (i,j), v_c by (j,k), w_c by (k,i).
    
    The constraint is: for all (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    This is a system of polynomial equations. We can try to find small-rank solutions.
    
    Given the small size, let's try a direct approach:
    - For rank r, we have r unknowns per factor (well, r columns in each factor).
    - U has 6*r entries, V has 18*r entries, W has 12*r entries.
    - We have 36 equations (one for each (i,j,k) triple).
    
    For r < 36, the system is overdetermined in terms of unknowns vs equations, but the nonlinearity makes it complex.
    
    Let's try a practical approach:
    1. Start with the schoolbook solution (rank 36).
    2. Try to merge pairs of terms if possible.
    3. Use a greedy or local search approach.
    
    Actually, for this specific problem, let me think about whether there's a known optimal rank.
    
    For 2x3 * 3x6, this is equivalent to multiplying a 2x6 matrix by... no, it's 2x3 times 3x6.
    
    Let me try a different approach: use the fact that for small matrices, we can try all possible rank values from some lower bound up to 36, and check if a solution exists.
    
    A lower bound: The tensor T has border rank at least max(n, m, p) = 6, but the actual rank could be higher.
    
    Let me try to implement a search that:
    1. Starts with rank 36 (schoolbook).
    2. Attempts to reduce rank by finding linear combinations of the basis terms.
    
    Actually, let me try a more direct algebraic approach for small ranks.
    
    For rank r, we need to find U (6xr), V (18xr), W (12xr) such that:
    sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all (i,j,k)
    
    This is a system of 36 polynomial equations in 36r unknowns (6r + 18r + 12r = 36r).
    
    For r = 36, we have 1296 unknowns and 36 equations - very underdetermined.
    For r = 18, we have 648 unknowns and 36 equations.
    For r = 12, we have 432 unknowns and 36 equations.
    
    The system is always underdetermined for r >= 1, so solutions likely exist for small r, but we need to find them.
    
    Let me try a practical search:
    - Fix one factor (say W) to simplify, then solve for U and V.
    - Or use a random initialization and gradient descent.
    
    Given the time constraint, let me try a hybrid approach:
    1. Use the schoolbook solution as a starting point.
    2. Try to reduce rank by identifying redundant terms.
    3. Use a local search to find better solutions.
    
    For now, let me return the schoolbook solution and add a deadline guard.
    """
    start_time = time.time()
    deadline = start_time + 120  # 120 seconds max
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Schoolbook solution
    rank_sb = nm * mp * pn // (m * m)  # Wait, this isn't right.
    # Actually, schoolbook rank is n*m*p = 2*3*6 = 36
    rank_sb = n * m * p
    
    U = np.zeros((nm, rank_sb))
    V = np.zeros((mp, rank_sb))
    W = np.zeros((pn, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce rank
    # Strategy: try rank values from some lower bound up to rank_sb - 1
    # For each rank, try to find a solution
    
    # Lower bound: at least max(n, m, p) = 6, but likely higher
    # Let's try from 18 down to 10, since 36 is the schoolbook rank
    
    def verify(U_mat, V_mat, W_mat, tol=1e-9):
        """Check if the decomposition is valid."""
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(U_mat.shape[1]):
                        val += U_mat[i * m + j, c] * V_mat[j * p + k, c] * W_mat[k * n + i, c]
                    if abs(val - 1) > tol:
                        return False
        return True
    
    # Try to find a solution with reduced rank
    # Use a simple approach: try rank = 18, 16, 14, 12, 10
    # For each, use random initialization and local search
    
    import random
    
    def try_rank(r, max_iters=10000, seed=42):
        """Try to find a rank-r decomposition."""
        if time.time() > deadline - 5:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize with random values
        U_try = rng.randn(nm, r)
        V_try = rng.randn(mp, r)
        W_try = rng.randn(pn, r)
        
        # Scale to get values closer to 1
        U_try /= np.sqrt(np.sum(U_try**2, axis=0))
        V_try /= np.sqrt(np.sum(V_try**2, axis=0))
        W_try /= np.sqrt(np.sum(W_try**2, axis=0))
        
        # Use gradient descent on the loss function
        loss = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(r):
                        val += U_try[i * m + j, c] * V_try[j * p + k, c] * W_try[k * n + i, c]
                    loss += (val - 1) ** 2
        
        # Simple gradient descent
        lr = 0.01
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Compute gradients
            dU = np.zeros_like(U_try)
            dV = np.zeros_like(V_try)
            dW = np.zeros_like(W_try)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(r):
                            val += U_try[i * m + j, c] * V_try[j * p + k, c] * W_try[k * n + i, c]
                        err = val - 1
                        for c in range(r):
                            dU[i * m + j, c] += 2 * err * V_try[j * p + k, c] * W_try[k * n + i, c]
                            dV[j * p + k, c] += 2 * err * U_try[i * m + j, c] * W_try[k * n + i, c]
                            dW[k * n + i, c] += 2 * err * U_try[i * m + j, c] * V_try[j * p + k, c]
            
            # Update
            U_try -= lr * dU
            V_try -= lr * dV
            W_try -= lr * dW
            
            # Recompute loss
            loss_new = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(r):
                            val += U_try[i * m + j, c] * V_try[j * p + k, c] * W_try[k * n + i, c]
                        loss_new += (val - 1) ** 2
            
            if abs(loss_new - loss) < 1e-12:
                break
            loss = loss_new
        
        if loss < 1e-6:
            # Verify with higher precision
            if verify(U_try, V_try, W_try, tol=1e-8):
                return U_try, V_try, W_try
        
        return None
    
    # Try different ranks
    for r in range(18, 9, -2):  # 18, 16, 14, 12, 10
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r, max_iters=5000, seed=42 + r)
        if result is not None:
            U_try, V_try, W_try = result
            best_rank = r
            best_U, best_V, best_W = U_try, V_try, W_try
            break
    
    # Convert to lists of integers/half-integers if possible
    def to_half_integer_list(mat):
        """Convert numpy array to list of lists with half-integer entries."""
        result = []
        for row in mat:
            row_list = []
            for val in row:
                # Round to nearest half-integer
                half_int = round(val * 2) / 2
                row_list.append(half_int)
            result.append(row_list)
        return result
    
    # Check if the solution has half-integer entries
    if best_rank < rank_sb:
        U_out = to_half_integer_list(best_U)
        V_out = to_half_integer_list(best_V)
        W_out = to_half_integer_list(best_W)
        
        # Verify the half-integer solution
        U_check = np.array(U_out)
        V_check = np.array(V_out)
        W_check = np.array(W_out)
        
        if verify(U_check, V_check, W_check, tol=1e-9):
            return {"U": U_out, "V": V_out, "W": W_out}
    
    # Return schoolbook solution
    U_out = [[float(x) for x in row] for row in U]
    V_out = [[float(x) for x in row] for row in V]
    W_out = [[float(x) for x in row] for row in W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
