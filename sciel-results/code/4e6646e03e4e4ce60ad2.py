"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8
    
    def naive_decomp():
        rank = n * m * p  # 16
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Verify the tensor decomposition is exact."""
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def get_rank14_factors():
        """Return the known rank-14 decomposition factors as numpy arrays."""
        R = 14
        U = np.zeros((4, R))
        V = np.zeros((8, R))
        W = np.zeros((8, R))
        
        # Left block (Strassen for 2x2 * 2x2 -> top-left 2x2 of output)
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[5,0]=1
        W[0,0]=1; W[3,0]=1
        
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        W[1,1]=1; W[3,1]=-1
        
        U[0,2]=1
        V[1,2]=1; V[5,2]=-1
        W[2,2]=1; W[3,2]=1
        
        U[3,3]=1
        V[4,3]=1; V[0,3]=-1
        W[0,3]=1; W[1,3]=1
        
        U[0,4]=1; U[1,4]=1
        V[5,4]=1
        W[0,4]=-1; W[2,4]=1
        
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[1,5]=1
        W[3,5]=1
        
        U[1,6]=1; U[3,6]=-1
        V[4,6]=1; V[5,6]=1
        W[0,6]=1
        
        # Right block (Strassen for 2x2 * 2x2 -> bottom-right 2x2 of output)
        U[0,7]=1; U[3,7]=1
        V[2,7]=1; V[7,7]=1
        W[4,7]=1; W[7,7]=1
        
        U[2,8]=1; U[3,8]=1
        V[2,8]=1
        W[5,8]=1; W[7,8]=-1
        
        U[0,9]=1
        V[3,9]=1; V[7,9]=-1
        W[6,9]=1; W[7,9]=1
        
        U[3,10]=1
        V[6,10]=1; V[2,10]=-1
        W[4,10]=1; W[5,10]=1
        
        U[0,11]=1; U[1,11]=1
        V[7,11]=1
        W[4,11]=-1; W[6,11]=1
        
        U[2,12]=1; U[0,12]=-1
        V[2,12]=1; V[3,12]=1
        W[7,12]=1
        
        U[1,13]=1; U[3,13]=-1
        V[6,13]=1; V[7,13]=1
        W[4,13]=1
        
        return U, V, W

    def try_peeling_rank13():
        """
        Apply peeling algorithm to try to find a rank-13 decomposition.
        Strategy: 
        1. Start with the known rank-14 decomposition.
        2. Check if any single rank-1 term can be expressed as a linear combination 
           of the others while maintaining the sum.
        3. If a term t_k can be written as sum_j c_j * t_j (j != k), then we can 
           potentially merge, but only if the resulting combined term is still rank-1.
        4. More practically: check if the 14 flattened tensors are linearly dependent.
           If T_total = sum(t_i), and there exists a relation sum(a_i * t_i) = 0,
           then T_total = sum(t_i) - sum(a_i * t_i) = sum((1-a_i) * t_i).
           This doesn't directly reduce rank unless we can re-factor.
        
        Alternative peeling approach:
        - The multiplication tensor T has flattening matrices.
        - If we can find a rank-1 component that is "trivial" (e.g., corresponds to 
          a diagonal entry that can be absorbed), we remove it.
        
        For 2x2 * 2x4, the known lower bound is 13 (from literature).
        Let's try a direct algebraic search for a rank-13 solution by fixing 
        the structure of the two Strassen blocks and allowing one shared term.
        """
        if time.time() > deadline - 20:
            return None
            
        U14, V14, W14 = get_rank14_factors()
        
        # Flatten each rank-1 tensor
        tensors = np.zeros((14, 4*8*8))
        for r in range(14):
            tensors[r] = np.einsum('i,j,k->ijk', U14[:,r], V14[:,r], W14[:,r]).flatten()
        
        T_total = tensors.sum(axis=0)
        
        # Check linear dependencies among the 14 tensors
        # If there is a null vector c such that sum(c_i * t_i) = 0, 
        # then T_total = sum(t_i) = sum(t_i) - sum(c_i * t_i) = sum((1-c_i)*t_i)
        # This doesn't help directly. We need to find if T_total can be written 
        # as a sum of 13 rank-1 tensors.
        
        # Try: for each pair of terms, see if their sum can be replaced by a 
        # single rank-1 term that, together with the remaining 12, sums to T_total.
        # This is equivalent to checking if t_i + t_j is "rank-1 extendable" 
        # in a way that allows merging.
        
        # A more direct approach: use the fact that for matrix multiplication,
        # the rank is at least max of the ranks of the flattening matrices.
        # Flattening 1: T as (4) x (8*8) -> rank <= 4
        # Flattening 2: T as (8) x (4*8) -> rank <= 8
        # Flattening 3: T as (8) x (4*8) -> rank <= 8
        # These don't give a tight lower bound.
        
        # Let's try a randomized local search: perturb the rank-14 solution 
        # and try to drive one coefficient to zero.
        # This is heuristic and may not work, but worth trying within time budget.
        
        # Actually, a known result: 2x2 * 2x4 has rank 13.
        # Let's try to construct it explicitly by modifying the Strassen blocks.
        
        # Idea: In the standard 2x2 * 2x4, we can use 7 terms for the left 2x2*2x2 
        # and 6 terms for the right, sharing one term.
        
        # Let's try a specific construction:
        # Left block: standard Strassen 7 terms
        # Right block: Strassen 7 terms, but one term is shared with left block.
        
        # The shared term must satisfy:
        # U_shared, V_shared, W_shared such that:
        # - In left block, it contributes to left 2x2 output
        # - In right block, it contributes to right 2x2 output
        
        # This requires the shared term to have non-zero entries in both 
        # left and right parts of V and W, which is possible if we choose 
        # V_shared and W_shared to have support in both halves.
        
        # Let's try:
        # Term 0: U=[1,0,0,1], V=[1,0,0,0,0,1,0,0], W=[1,0,0,1,0,0,0,0]  (left M1)
        # Term 1: U=[0,0,1,1], V=[1,0,0,0,0,0,0,0], W=[0,1,0,-1,0,0,0,0]  (left M2)
        # ... etc for left block
        
        # For the right block, instead of 7 new terms, use 6 new + 1 shared.
        # The shared term must be one of the left block terms, modified to also 
        # affect the right block.
        
        # This is complex. Let's try a numerical approach:
        # Set up the system for rank 13 and solve using nonlinear least squares.
        
        # Define variables for U, V, W with 13 columns.
        # Minimize ||sum_r U_r V_r W_r - T_mult||^2
        # T_mult is the true multiplication tensor.
        
        # Build T_mult
        T_mult = np.zeros((4, 8, 8))
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    T_mult[i*2+j, j*4+k, k*2+i] = 1.0
        
        # Initialize with the rank-14 solution, drop one column, and perturb.
        # Try dropping each column and see if we can converge to a solution.
        
        best_result = None
        best_norm = np.inf
        
        for drop_idx in range(14):
            if time.time() > deadline - 10:
                break
                
            # Create initial guess for rank 13
            U13 = np.delete(U14, drop_idx, axis=1)
            V13 = np.delete(V14, drop_idx, axis=1)
            W13 = np.delete(W14, drop_idx, axis=1)
            
            # Perturb slightly
            np.random.seed(42 + drop_idx)
            U13 = U13 + 0.1 * np.random.randn(4, 13)
            V13 = V13 + 0.1 * np.random.randn(8, 13)
            W13 = W13 + 0.1 * np.random.randn(8, 13)
            
            # Use simple gradient descent to minimize the residual
            lr = 0.01
            for step in range(100):
                if time.time() > deadline - 5:
                    break
                    
                # Compute current tensor
                T_curr = np.einsum('ir,jr,kr->ijk', U13, V13, W13)
                residual = T_curr - T_mult
                
                # Compute gradients
                dU = np.einsum('ijk,jr,kr->ir', residual, V13, W13)
                dV = np.einsum('ijk,ir,kr->jr', residual, U13, W13)
                dW = np.einsum('ijk,ir,jr->kr', residual, U13, V13)
                
                # Update
                U13 -= lr * dU
                V13 -= lr * dV
                W13 -= lr * dW
                
                # Check convergence
                if step % 10 == 0:
                    norm = np.linalg.norm(residual)
                    if norm < best_norm:
                        best_norm = norm
                        best_U = U13.copy()
                        best_V = V13.copy()
                        best_W = W13.copy()
                    if norm < 1e-8:
                        break
            
            # Final check
            T_final = np.einsum('ir,jr,kr->ijk', U13, V13, W13)
            final_norm = np.linalg.norm(T_final - T_mult)
            if final_norm < 1e-7:
                # Convert to lists
                U_list = [[float(U13[i,r]) for r in range(13)] for i in range(4)]
                V_list = [[float(V13[j,r]) for r in range(13)] for j in range(8)]
                W_list = [[float(W13[k,r]) for r in range(13)] for k in range(8)]
                
                if verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    def try_strassen_rectangular():
        """
        Rank-14 decomposition for 2x2 * 2x4 using two Strassen 2x2 blocks.
        """
        R = 14
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # Left block
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[5][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        U[0][2] = 1
        V[1][2] = 1; V[5][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        U[3][3] = 1
        V[4][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        U[0][4] = 1; U[1][4] = 1
        V[5][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        U[1][6] = 1; U[3][6] = -1
        V[4][6] = 1; V[5][6] = 1
        W[0][6] = 1
        
        # Right block
        U[0][7] = 1; U[3][7] = 1
        V[2][7] = 1; V[7][7] = 1
        W[4][7] = 1; W[7][7] = 1
        
        U[2][8] = 1; U[3][8] = 1
        V[2][8] = 1
        W[5][8] = 1; W[7][8] = -1
        
        U[0][9] = 1
        V[3][9] = 1; V[7][9] = -1
        W[6][9] = 1; W[7][9] = 1
        
        U[3][10] = 1
        V[6][10] = 1; V[2][10] = -1
        W[4][10] = 1; W[5][10] = 1
        
        U[0][11] = 1; U[1][11] = 1
        V[7][11] = 1
        W[4][11] = -1; W[6][11] = 1
        
        U[2][12] = 1; U[0][12] = -1
        V[2][12] = 1; V[3][12] = 1
        W[7][12] = 1
        
        U[1][13] = 1; U[3][13] = -1
        V[6][13] = 1; V[7][13] = 1
        W[4][13] = 1
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        else:
            return None
    
    # Try to find rank-13 via peeling/local search first
    result = try_peeling_rank13()
    if result is not None:
        return result
    
    # Fall back to rank-14
    result = try_strassen_rectangular()
    if result is not None:
        return result
    
    # Fallback to naive rank 16
    return naive_decomp()
# EVOLVE-BLOCK-END
