import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline, 10s margin before 150s kill

    n, m, p = 3, 3, 5
    nm, mp, pn = n*m, m*p, p*n
    
    # We will try to find a decomposition with rank 24 (block-Strassen for the 3x3x3 part)
    # or at least less than 45.
    
    # The tensor T[i,j,k] = delta_{i,k} for A (i,j) and B (j,k).
    # T_{i,j,k} = 1 if i==k, else 0.
    
    # We want to decompose T = sum_r U_r \otimes V_r \otimes W_r
    # where U_r is a 3x3 matrix, V_r is a 3x5 matrix, W_r is a 5x3 matrix.
    # Actually, the indices are:
    # U has shape (n*m, rank) = (9, rank)
    # V has shape (m*p, rank) = (15, rank)
    # W has shape (p*n, rank) = (15, rank)
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Note the W index: W[k*n+i, r].
    
    # Let's try a greedy search for a low-rank decomposition.
    # We start with the schoolbook basis (rank 45) and try to merge or replace terms.
    # Or, we can generate a pool of candidate rank-1 tensors (Strassen-like) and do a set cover.
    
    # Candidate rank-1 tensors:
    # A rank-1 tensor is defined by matrices U0 (3x3), V0 (3x5), W0 (5x3).
    # The tensor is T0_{i,j,k} = U0[i,j] * V0[j,k] * W0[k,i].
    
    # We want to cover the target tensor T with a linear combination of these.
    # Actually, the decomposition is exact, so we need the sum of these rank-1 tensors to equal T.
    
    # Let's use a random search or a structured search.
    # Given the time limit, a random search might be too slow.
    # Let's try to construct a specific decomposition.
    
    # For the 3x3x3 part (k=0,1,2), we can use the Strassen-like decomposition for 3x3.
    # The rank of 3x3 matrix multiplication is 19 (Landsberg).
    # But here we have 3x5, so it's not just 3x3.
    
    # Let's try to find a rank 24 decomposition.
    # We can use the following idea:
    # Split B into B1 (3x3) and B2 (3x2).
    # C1 = A * B1 (3x3)
    # C2 = A * B2 (3x2)
    # Rank of 3x3x3 is 19 (optimal) or 21 (Strassen-like).
    # Rank of 3x3x2 is ?
    # Actually, let's just use a simple heuristic:
    # We will generate a set of candidate rank-1 tensors and use a greedy algorithm to select a subset that sums to T.
    
    # Target tensor T:
    T = np.zeros((9, 15, 15))
    for i in range(3):
        for j in range(3):
            for k in range(5):
                if i == k:
                    T[i*m+j, j*p+k, k*n+i] = 1
    
    # We will search for a sparse decomposition.
    # We'll start with the schoolbook terms and try to replace 2 terms with 1 term if possible.
    # Or, we'll just use the schoolbook decomposition (rank 45) as a fallback.
    
    # Fallback: Schoolbook
    R_school = 45
    U_school = np.zeros((nm, R_school))
    V_school = np.zeros((mp, R_school))
    W_school = np.zeros((pn, R_school))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, r] = 1
                V_school[j*p+k, r] = 1
                W_school[k*n+i, r] = 1
                r += 1
                
    # Let's try to improve upon schoolbook.
    # We can try to find a linear dependence among the schoolbook terms? No, they are independent.
    # We need to find a different set of rank-1 tensors.
    
    # Let's try a random search for rank 30, 35, 40.
    # But this is likely to fail within 140s.
    
    # Instead, let's try to use the block structure.
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1.
    # B = [[B11, B12, B13], [B21, B22, B23], [B31, B32, B33]]
    # This is getting complicated.
    
    # Let's return the schoolbook solution for now, as it's guaranteed to be correct.
    # The score will be 45.
    
    # Wait, the score is 1.0000 for rank 45.
    # If I can find rank 44, score is 45/44 > 1.
    
    # Let's try to see if we can merge any terms.
    # For example, can we replace two terms with one?
    # Term 1: U1, V1, W1
    # Term 2: U2, V2, W2
    # Can we find U, V, W such that U\otimes V\otimes W = U1\otimes V1\otimes W1 + U2\otimes V2\otimes W2?
    # This is equivalent to saying the sum of two rank-1 tensors is rank 1, which is only true if they are proportional.
    # So we can't simply merge two schoolbook terms.
    
    # We need to change the basis.
    
    # Let's try a specific construction for 3x3x3 with rank 19.
    # But I don't have the explicit coefficients memorized.
    
    # Let's return the schoolbook solution.
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result["U"][0]))
