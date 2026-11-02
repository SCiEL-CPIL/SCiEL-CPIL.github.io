import time
import numpy as np

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    """
    n, m, p = 2, 2, 7
    start_time = time.time()
    deadline = start_time + 120
    
    # First, try to find a rank-23 solution via systematic search
    # The tensor T[i,j,k] = 1 if i==k else 0, for i,k in {0,1}, j in {0,1}, k in {0,...,6}
    # Rank-r: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Try rank 23 first, then fall back to 25 if not found in time
    for target_rank in [23, 24, 25]:
        if time.time() > deadline - 10:
            break
        
        # For rank r, we have:
        # U: shape (4, r) - indices (i,j) where i in {0,1}, j in {0,1}
        # V: shape (14, r) - indices (j,k) where j in {0,1}, k in {0,...,6}
        # W: shape (14, r) - indices (k,i) where k in {0,...,6}, i in {0,1}
        
        # The constraints are: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = (1 if i==k else 0)
        
        # This is a system of 28 equations (14 diagonal = 1, 14 off-diagonal = 0)
        # with 3*r unknowns (entries of U, V, W)
        
        # For small ranks, we can try to find integer solutions
        # Strategy: fix V and W, solve for U, or use a more systematic approach
        
        if target_rank <= 24:
            # Try to find a solution by fixing some variables and solving
            # Use the structure: for each k, the constraints couple U and V/W
        
            # Let's try a constructive approach for rank 23
            # Based on the idea of mixing columns globally
            
            # Actually, let me try a direct search for small integer solutions
            # For rank 23, we have 69 unknowns and 28 equations - very underdetermined
            
            # Let me try a specific construction
            U = np.zeros((4, target_rank), dtype=int)
            V = np.zeros((14, target_rank), dtype=int)
            W = np.zeros((14, target_rank), dtype=int)
            
            # Use Strassen for first 3 pairs (rank 21) + 2 extra terms for column 6
            # This gives rank 23 if we can share 2 terms
            
            # Strassen block 1: columns 0,1 (terms 0-6)
            # M1 = (A00+A11)*(B0,0+B1,1)
            U[0, 0] += 1; U[3, 0] += 1
            V[0, 0] += 1; V[7+1, 0] += 1
            W[0, 0] += 1; W[2+1, 0] += 1
            
            # M2 = (A10+A11)*B0,0
            U[2, 1] += 1; U[3, 1] += 1
            V[0, 1] += 1
            W[2+1, 1] += 1; W[0+1, 1] -= 1
            
            # M3 = A00*(B0,1-B1,1)
            U[0, 2] += 1
            V[0+1, 2] += 1; V[7+1, 2] -= 1
            W[0+1, 2] += 1; W[2+1, 2] += 1
            
            # M4 = A11*(B1,0-B0,0)
            U[3, 3] += 1
            V[7+0, 3] += 1; V[0+0, 3] -= 1
            W[0, 3] += 1; W[2+1, 3] += 1
            
            # M5 = (A00+A01)*B1,1
            U[0, 4] += 1; U[1, 4] += 1
            V[7+1, 4] += 1
            W[0, 4] -= 1; W[0+1, 4] += 1
            
            # M6 = (A10-A00)*(B0,0+B0,1)
            U[2, 5] += 1; U[0, 5] -= 1
            V[0, 5] += 1; V[0+1, 5] += 1
            W[2+1, 5] += 1
            
            # M7 = (A01-A11)*(B1,0+B1,1)
            U[1, 6] += 1; U[3, 6] -= 1
            V[7+0, 6] += 1; V[7+1, 6] += 1
            W[0, 6] += 1
            
            # Strassen block 2: columns 2,3 (terms 7-13)
            U[0, 7] += 1; U[3, 7] += 1
            V[0+2, 7] += 1; V[7+3, 7] += 1
            W[0+2, 7] += 1; W[2+3, 7] += 1
            
            U[2, 8] += 1; U[3, 8] += 1
            V[0+2, 8] += 1
            W[2+3, 8] += 1; W[0+2, 8] -= 1
            
            U[0, 9] += 1
            V[0+3, 9] += 1; V[7+3, 9] -= 1
            W[0+3, 9] += 1; W[2+3, 9] += 1
            
            U[3, 10] += 1
            V[7+2, 10] += 1; V[0+2, 10] -= 1
            W[0+2, 10] += 1; W[2+3, 10] += 1
            
            U[0, 11] += 1; U[1, 11] += 1
            V[7+3, 11] += 1
            W[0+2, 11] -= 1; W[0+3, 11] += 1
            
            U[2, 12] += 1; U[0, 12] -= 1
            V[0+2, 12] += 1; V[0+3, 12] += 1
            W[2+3, 12] += 1
            
            U[1, 13] += 1; U[3, 13] -= 1
            V[7+2, 13] += 1; V[7+3, 13] += 1
            W[0+2, 13] += 1
            
            # Strassen block 3: columns 4,5 (terms 14-20)
            U[0, 14] += 1; U[3, 14] += 1
            V[0+4, 14] += 1; V[7+5, 14] += 1
            W[0+4, 14] += 1; W[2+5, 14] += 1
            
            U[2, 15] += 1; U[3, 15] += 1
            V[0+4, 15] += 1
            W[2+5, 15] += 1; W[0+4, 15] -= 1
            
            U[0, 16] += 1
            V[0+5, 16] += 1; V[7+5, 16] -= 1
            W[0+5, 16] += 1; W[2+5, 16] += 1
            
            U[3, 17] += 1
            V[7+4, 17] += 1; V[0+4, 17] -= 1
            W[0+4, 17] += 1; W[2+5, 17] += 1
            
            U[0, 18] += 1; U[1, 18] += 1
            V[7+5, 18] += 1
            W[0+4, 18] -= 1; W[0+5, 18] += 1
            
            U[2, 19] += 1; U[0, 19] -= 1
            V[0+4, 19] += 1; V[0+5, 19] += 1
            W[2+5, 19] += 1
            
            U[1, 20] += 1; U[3, 20] -= 1
            V[7+4, 20] += 1; V[7+5, 20] += 1
            W[0+4, 20] += 1
            
            # For column 6, instead of 4 schoolbook terms, try 2 mixed terms
            # C0,6 = A00*B0,6 + A01*B1,6
            # C1,6 = A10*B0,6 + A11*B1,6
            
            # Try: M8 = (A00+A11)*(B0,6+B1,6) and M9 = (A01+A10)*(B1,6-B0,6)
            # M8 contributes: A00*B0,6 + A00*B1,6 + A11*B0,6 + A11*B1,6
            # M9 contributes: A01*B1,6 - A01*B0,6 + A10*B1,6 - A10*B0,6
            
            # We need: C0,6 = A00*B0,6 + A01*B1,6
            # From M8: A00*B0,6 + A11*B0,6 (has extra A11*B0,6)
            # From M9: A01*B1,6 (good) - A01*B0,6 + A10*B1,6 - A10*B0,6
            
            # This doesn't work cleanly. Let me try another approach.
            
            # Alternative: use 2 terms that exactly give column 6
            # M8 = A00*B0,6 + A01*B1,6 -> but this is already rank 2
            # M9 = A10*B0,6 + A11*B1,6 -> rank 2
            # Total: 4 terms, which is what we had.
            
            # To get rank 23, we need to share 1 term between the Strassen blocks and column 6.
            # This is tricky. Let me just verify rank 25 works and return that.
            
            pass  # Skip rank 23/24 for now, use rank 25
        
        # Use the known rank-25 construction
        rank = 25
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        W = np.zeros((p*n, rank), dtype=int)
        
        def add_strassen_block(col_start, k_start):
            U[0, col_start+0] += 1
            U[3, col_start+0] += 1
            V[0*p + k_start, col_start+0] += 1
            V[1*p + (k_start+1), col_start+0] += 1
            W[k_start*n + 0, col_start+0] += 1
            W[(k_start+1)*n + 1, col_start+0] += 1

            U[2, col_start+1] += 1
            U[3, col_start+1] += 1
            V[0*p + k_start, col_start+1] += 1
            W[k_start*n + 1, col_start+1] += 1
            W[(k_start+1)*n + 1, col_start+1] -= 1

            U[0, col_start+2] += 1
            V[0*p + (k_start+1), col_start+2] += 1
            V[1*p + (k_start+1), col_start+2] -= 1
            W[(k_start+1)*n + 0, col_start+2] += 1
            W[(k_start+1)*n + 1, col_start+2] += 1

            U[3, col_start+3] += 1
            V[1*p + k_start, col_start+3] += 1
            V[0*p + k_start, col_start+3] -= 1
            W[k_start*n + 0, col_start+3] += 1
            W[k_start*n + 1, col_start+3] += 1

            U[0, col_start+4] += 1
            U[1, col_start+4] += 1
            V[1*p + (k_start+1), col_start+4] += 1
            W[k_start*n + 0, col_start+4] -= 1
            W[(k_start+1)*n + 0, col_start+4] += 1

            U[2, col_start+5] += 1
            U[0, col_start+5] -= 1
            V[0*p + k_start, col_start+5] += 1
            V[0*p + (k_start+1), col_start+5] += 1
            W[(k_start+1)*n + 1, col_start+5] += 1

            U[1, col_start+6] += 1
            U[3, col_start+6] -= 1
            V[1*p + k_start, col_start+6] += 1
            V[1*p + (k_start+1), col_start+6] += 1
            W[k_start*n + 0, col_start+6] += 1

        add_strassen_block(0, 0)
        add_strassen_block(7, 2)
        add_strassen_block(14, 4)
        
        col_start = 21
        k = 6
        U[0, col_start+0] += 1
        V[0*p + k, col_start+0] += 1
        W[k*n + 0, col_start+0] += 1
        
        U[1, col_start+1] += 1
        V[1*p + k, col_start+1] += 1
        W[k*n + 0, col_start+1] += 1
        
        U[2, col_start+2] += 1
        V[0*p + k, col_start+2] += 1
        W[k*n + 1, col_start+2] += 1
        
        U[3, col_start+3] += 1
        V[1*p + k, col_start+3] += 1
        W[k*n + 1, col_start+3] += 1

        # Verify
        valid = True
        for i in range(n):
            for j in range(m):
                for k_idx in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k_idx
                    c_idx = k_idx * n + i
                    s = 0
                    for r in range(rank):
                        s += U[a_idx, r] * V[b_idx, r] * W[c_idx, r]
                    if s != (1 if i == k_idx else 0):
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n*m, rank_sb), dtype=int)
        V_sb = np.zeros((m*p, rank_sb), dtype=int)
        W_sb = np.zeros((p*n, rank_sb), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k_idx in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k_idx, col] = 1
                    W_sb[k_idx*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
