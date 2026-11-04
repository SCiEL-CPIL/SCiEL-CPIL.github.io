"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Build the rank-42 Strassen decomposition
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    def add_strassen_block(a_idx, b_idx, w_idx, start_col):
        """
        a_idx: [a00, a01, a10, a11] indices into U
        b_idx: [b00, b01, b10, b11] indices into V
        w_idx: [w00, w01, w10, w11] indices into W for the 2x2 output block
        """
        c = start_col
        # M1 = (a00+a11)(b00+b11)
        U[a_idx[0]][c] += 1; U[a_idx[3]][c] += 1
        V[b_idx[0]][c] += 1; V[b_idx[3]][c] += 1
        W[w_idx[0]][c] += 1; W[w_idx[3]][c] += 1
        # M2 = (a10+a11)b00
        U[a_idx[2]][c+1] += 1; U[a_idx[3]][c+1] += 1
        V[b_idx[0]][c+1] += 1
        W[w_idx[2]][c+1] += 1; W[w_idx[3]][c+1] -= 1
        # M3 = a00(b01-b11)
        U[a_idx[0]][c+2] += 1
        V[b_idx[1]][c+2] += 1; V[b_idx[3]][c+2] -= 1
        W[w_idx[1]][c+2] += 1; W[w_idx[3]][c+2] += 1
        # M4 = a11(b10-b00)
        U[a_idx[3]][c+3] += 1
        V[b_idx[2]][c+3] += 1; V[b_idx[0]][c+3] -= 1
        W[w_idx[0]][c+3] += 1; W[w_idx[2]][c+3] += 1
        # M5 = (a00+a01)b11
        U[a_idx[0]][c+4] += 1; U[a_idx[1]][c+4] += 1
        V[b_idx[3]][c+4] += 1
        W[w_idx[0]][c+4] -= 1; W[w_idx[1]][c+4] += 1
        # M6 = (a10-a00)(b00+b01)
        U[a_idx[2]][c+5] += 1; U[a_idx[0]][c+5] -= 1
        V[b_idx[0]][c+5] += 1; V[b_idx[1]][c+5] += 1
        W[w_idx[3]][c+5] += 1
        # M7 = (a01-a11)(b10+b11)
        U[a_idx[1]][c+6] += 1; U[a_idx[3]][c+6] -= 1
        V[b_idx[2]][c+6] += 1; V[b_idx[3]][c+6] += 1
        W[w_idx[0]][c+6] += 1
    
    # A is 2x4: indices i*m+j
    # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3
    # A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7
    a_L = [0, 1, 4, 5]  # a00, a01, a10, a11
    a_R = [2, 3, 6, 7]
    
    # B is 4x6: indices j*p+k
    # B col block 0: cols 0,1
    b_1_T = [0, 1, 6, 7]
    b_1_B = [12, 13, 18, 19]
    w_1 = [0, 2, 1, 3]
    
    add_strassen_block(a_L, b_1_T, w_1, 0)
    add_strassen_block(a_R, b_1_B, w_1, 7)
    
    # B col block 1: cols 2,3
    b_2_T = [2, 3, 8, 9]
    b_2_B = [14, 15, 20, 21]
    w_2 = [4, 6, 5, 7]
    
    add_strassen_block(a_L, b_2_T, w_2, 14)
    add_strassen_block(a_R, b_2_B, w_2, 21)
    
    # B col block 2: cols 4,5
    b_3_T = [4, 5, 10, 11]
    b_3_B = [16, 17, 22, 23]
    w_3 = [8, 10, 9, 11]
    
    add_strassen_block(a_L, b_3_T, w_3, 28)
    add_strassen_block(a_R, b_3_B, w_3, 35)
    
    # Verify the decomposition
    def verify_decomp(U, V, W, rank):
        """Check if sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    if not verify_decomp(U, V, W, rank):
        # Fallback: return schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
        return {"U": U, "V": V, "W": W}
    
    # Convert to numpy for linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Support pruning: try to find rank-40 or rank-38 decompositions
    # by removing pairs of columns and solving for coefficients
    
    # The tensor equation: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We know T[i,j,k] = 1 for all i,j,k
    
    # For a given set of active columns S (size R), we need:
    # For each (i,j,k): sum_{r in S} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is nonlinear in the coefficients. But if we fix U and V and solve for W,
    # or fix two of them, it becomes linear.
    
    # Strategy: try removing 2 columns (rank 40) and see if we can adjust
    # Actually, a simpler approach: check if any 2 columns are linearly dependent
    # in the tensor sense, i.e., if removing them and rescaling others works.
    
    # Better approach: use the fact that the decomposition is overdetermined.
    # We have 2*4*6 = 48 equations and 42 unknowns (if we fix one factor).
    
    # Let's try a different approach: check if the rank can be reduced by
    # finding if any subset of 40 columns still spans the tensor.
    
    # For each pair of columns to remove, check if the remaining 40 columns
    # can be scaled to still satisfy the equations.
    
    # This is equivalent to checking if the removed columns are in the span
    # of the remaining ones, with appropriate scaling.
    
    # Simpler: just try all pairs of columns to remove and check if the
    # resulting system has a solution.
    
    # For rank 40: remove 2 columns
    # For rank 38: remove 4 columns
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try rank 40 first
    if time.time() < deadline - 20:
        for remove_pair in combinations(range(rank), 2):
            if time.time() > deadline - 15:
                break
            
            active = [r for r in range(rank) if r not in remove_pair]
            R = len(active)
            
            # Check if the remaining columns can be scaled to satisfy the tensor
            # We'll use a least-squares approach
            # For each (i,j,k), we need sum_{r in active} U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            # This is nonlinear, but if we fix U and V, it's linear in W
            # Or fix U and W, linear in V, etc.
            
            # Let's fix U and V, solve for W
            # For each k*n+i (12 values), we have 48 equations (one per i,j,k... wait)
            # Actually, for each (i,j,k) triple, we have one equation
            # The unknowns are W[k*n+i, r] for r in active
            
            # Let's set up the linear system
            # For a fixed (i,j,k): sum_{r in active} U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # Let c_r = U[i*m+j,r] * V[j*p+k,r]
            # Then sum_{r in active} c_r * W[k*n+i,r] = 1
            
            # For different (i,j,k) with the same k*n+i, we get different equations
            # for the same W row
            
            # Actually, let's just verify if the current decomposition works with
            # those columns zeroed out (i.e., if the removed columns contribute 0)
            
            U_test = U_np.copy()
            V_test = V_np.copy()
            W_test = W_np.copy()
            
            U_test[:, remove_pair] = 0
            V_test[:, remove_pair] = 0
            W_test[:, remove_pair] = 0
            
            # Check if this still satisfies the tensor
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0
                        for r in active:
                            total += U_test[i*m+j, r] * V_test[j*p+k, r] * W_test[k*n+i, r]
                        if abs(total - 1) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = R
                best_U = U_test.tolist()
                best_V = V_test.tolist()
                best_W = W_test.tolist()
                break  # Found rank 40, stop
    
    # If we found rank 40, try rank 38
    if best_rank == 40 and time.time() < deadline - 20:
        for remove_pair in combinations(range(best_rank), 2):
            if time.time() > deadline - 15:
                break
            
            active = [r for r in range(best_rank) if r not in remove_pair]
            R = len(active)
            
            U_test = np.array(best_U, dtype=float)
            V_test = np.array(best_V, dtype=float)
            W_test = np.array(best_W, dtype=float)
            
            U_test[:, remove_pair] = 0
            V_test[:, remove_pair] = 0
            W_test[:, remove_pair] = 0
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0
                        for r in active:
                            total += U_test[i*m+j, r] * V_test[j*p+k, r] * W_test[k*n+i, r]
                        if abs(total - 1) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = R
                best_U = U_test.tolist()
                best_V = V_test.tolist()
                best_W = W_test.tolist()
                break
    
    if best_rank < rank:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
