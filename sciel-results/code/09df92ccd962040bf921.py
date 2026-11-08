import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Strategy: Split the 4-dimensional inner product into two 2-dimensional blocks.
    # This creates two independent sub-problems: 2x2x6 and 2x2x6.
    # We then decompose each 2x2x6 problem.
    #
    # For 2x2x6, we can split the 6-dimensional output into three 2-dimensional blocks.
    # This creates three 2x2x2 sub-problems.
    #
    # However, a more efficient approach is to recognize that 2x2x6 can be seen as
    # multiplying a 2x2 matrix by a 2x6 matrix.
    # We can use a block decomposition where we treat the 2x6 matrix as three 2x2 blocks.
    # So 2x2x6 = 2x2x2 + 2x2x2 + 2x2x2.
    # Each 2x2x2 can be done with Strassen (rank 7).
    # So 2x2x6 has rank 3 * 7 = 21? No, that's if they are independent.
    # But the 2x2 matrix A is shared. In tensor decomposition, the rank is the sum of the ranks
    # of the sub-tensors if they are disjoint in the tensor product space.
    # The tensor for 2x2x6 is T[i, j, k] where i in [0,2), j in [0,2), k in [0,6).
    # Splitting k into k1, k2, k3 (each in [0,2)) gives three tensors T1, T2, T3.
    # T1[i, j, k1] = 1, T2[i, j, k2] = 1, T3[i, j, k3] = 1.
    # These are three separate 2x2x2 tensors.
    # The rank of the direct sum is the sum of the ranks.
    # So rank(2x2x6) <= 3 * rank(2x2x2) = 3 * 7 = 21.
    #
    # Wait, is there a better way?
    # The rank of 2x2x2 is 7 (Strassen).
    # The rank of 2x2x3 is 13 (known result? Let's check).
    # If rank(2x2x3) = 13, then rank(2x2x6) = 2 * 13 = 26?
    # Or is it 2 * rank(2x2x3) if we split the 6 into two 3s?
    # Yes, splitting k into two blocks of 3 gives two 2x2x3 tensors.
    # So rank(2x2x6) <= 2 * rank(2x2x3).
    #
    # What is the rank of 2x2x3?
    # The schoolbook rank is 2*2*3 = 12.
    # Can we do better?
    # Let's try to find a rank-12 decomposition for 2x2x3.
    # Actually, for small sizes, the schoolbook rank is often optimal or close to it.
    # But for 2x2x2, Strassen gives 7 < 8.
    # For 2x2x3, let's see if we can do better than 12.
    #
    # Let's assume for now that we use the block decomposition into 2x2x2s.
    # Total rank = 2 * (3 * 7) = 42.
    #
    # But the prompt suggests a total rank of 26.
    # This implies rank(2x2x6) = 13.
    # Is rank(2x2x6) = 13?
    # If we split 6 into 3+3, and rank(2x2x3) = 6.5? No, rank is integer.
    # Maybe rank(2x2x3) is 7? Then 2*7 = 14. Total 28.
    # Maybe rank(2x2x3) is 6? Then 2*6 = 12. Total 24.
    #
    # Let's look for a rank-13 decomposition of 2x2x6 directly.
    # Or maybe the prompt means something else.
    # "decompose each 2x2x6 tensor using the minimal rank of 13 (derived from the 2x2x3 tensor rank of 7 via block diagonalization or known bounds)"
    # This phrasing is confusing. If 2x2x3 has rank 7, then 2x2x6 (which is two 2x2x3s) would have rank 14.
    # Unless the two 2x2x3s share some structure? No, they are disjoint in k.
    #
    # Let's try to construct a rank-7 decomposition for 2x2x3.
    # T[i, j, k] for i,j in [0,2), k in [0,3).
    # We want to sum 7 rank-1 terms.
    #
    # Alternatively, let's just implement the Strassen-based rank 42 decomposition first,
    # as it is a known improvement over 48. Then we can try to optimize further.
    #
    # Let's implement the correct Strassen decomposition for 2x2x2.
    # I will use a verified set of coefficients.
    #
    # Verified Strassen decomposition for 2x2x2:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a+b)(h)
    # M4 = (d-b)(f+h)
    # M5 = (a-c)(e+f)
    # M6 = (b+d)(g+e)
    # M7 = (a+c)(g+h)
    #
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    #
    # Let's re-verify C[0,0] = ae + bg.
    # M1 = ae + ah + de + dh
    # M4 = df + dh - bf - bh
    # M5 = ae + af - ce - cf
    # M7 = ag + ah + cg + ch
    #
    # Sum = (ae + ah + de + dh) + (df + dh - bf - bh) - (ae + af - ce - cf) + (ag + ah + cg + ch)
    # = ae + ah + de + dh + df + dh - bf - bh - ae - af + ce + cf + ag + ah + cg + ch
    # = (ae - ae) + (ah + ah) + de + (dh + dh) + df - bf - bh - af + ce + cf + ag + cg + ch
    # = 2ah + de + 2dh + df - bf - bh - af + ce + cf + ag + cg + ch
    #
    # This is definitely not ae + bg.
    # I am struggling with the Strassen coefficients.
    #
    # Let's try a different set.
    # From Wikipedia:
    # M1 = (a + d)(e + h)
    # M2 = (c + d)(e)
    # M3 = (a + b)(h)
    # M4 = (d - b)(f + h)
    # M5 = (a - c)(e + f)
    # M6 = (b + d)(g + e)
    # M7 = (a + c)(g + h)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Let's check C12 = M3 + M5.
    # M3 = (a+b)h = ah + bh
    # M5 = (a-c)(e+f) = ae + af - ce - cf
    # Sum = ah + bh + ae + af - ce - cf
    # We want C12 = af + bh.
    # So we need ah + ae - ce - cf to be 0? No.
    #
    # It seems my manual verification is failing because I might be misremembering the exact formulas or the variable mapping.
    #
    # Let's use a brute-force search for a rank-7 decomposition of 2x2x2 to get the correct coefficients.
    # Or, let's just use the direct decomposition for the 2x2x2 blocks, which has rank 8.
    # Then the total rank would be 2 * 3 * 8 = 48. That's no improvement.
    #
    # Let's try to find the Strassen coefficients programmatically.
    # We want U, V, W of shape (4, 7) such that sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k.
    #
    # This is a system of 8 equations (for the 8 entries of the 2x2x2 tensor) with 3*4*7 = 84 unknowns.
    # We can set up this system and solve it.
    
    # Let's construct the Strassen decomposition for 2x2x2 using a solver.
    # We fix the first row of U to be [1, 0, ..., 0] to break symmetry (if possible).
    # Actually, let's just use a known valid Strassen decomposition found online or in literature.
    #
    # Here is a valid Strassen decomposition (verified by code in previous attempts or known sources):
    # U = np.array([
    #     [1, 0, 1, 0, 1, 0, 1],
    #     [0, 1, 1, -1, 0, 1, 0],
    #     [0, 1, 0, 0, -1, 0, 1],
    #     [1, 1, 0, 1, 0, 1, 0]
    # ])
    # V = np.array([
    #     [1, 0, 0, 0, 1, 1, 0],
    #     [0, 0, 0, 1, 1, 0, 0],
    #     [0, 0, 0, 0, 0, 1, 1],
    #     [1, 0, 1, 1, 0, 0, 1]
    # ])
    # W = np.array([
    #     [1, 0, 0, 0, -1, 0, 1],
    #     [0, 0, 1, 0, 1, 0, 0],
    #     [0, 0, 0, 1, 0, 0, 0],
    #     [1, 0, 1, 0, 0, 1, 0]
    # ])
    
    # Let's verify this one.
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # We want T[i,j,k] = 1 for all i,j,k in [0,2).
    
    # Let's check T[0,0,0] (i=0, j=0, k=0).
    # U[0,0,:] = [1, 0, 1, 0, 1, 0, 1]
    # V[0,0,:] = [1, 0, 0, 0, 1, 1, 0]
    # W[0,0,:] = [1, 0, 0, 0, -1, 0, 1]
    # Sum = 1*1*1 + 0 + 1*0*0 + 0 + 1*1*(-1) + 0 + 1*0*1 = 1 - 1 = 0.
    # We want 1. So this is wrong.
    
    # Let's try another set.
    # I will use a simple greedy search to find a rank-7 decomposition.
    # But that might take too long.
    
    # Let's stick to the direct decomposition for now, but use a smarter block structure.
    # Actually, the prompt says "yielding a total rank of 26".
    # This is a very specific number.
    # 26 = 2 * 13.
    # So we need rank(2x2x6) = 13.
    #
    # Let's assume there is a rank-13 decomposition for 2x2x6.
    # How to find it?
    # 2x2x6 can be seen as two 2x2x3 tensors.
    # If rank(2x2x3) = 6.5, that's impossible.
    # Maybe rank(2x2x3) = 7 and we save 1?
    # Or maybe there is a direct rank-13 decomposition.
    
    # Let's try to construct a rank-13 decomposition for 2x2x6.
    # T[i, j, k] for i,j in [0,2), k in [0,6).
    # We want sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1.
    
    # This is a system of 2*2*6 = 24 equations.
    # Unknowns: 3 * 4 * 13 = 156.
    # We can solve this using least squares or a direct solver if we fix some variables.
    
    # Let's use a randomized approach to find a low-rank decomposition.
    # We'll start with rank 13 and try to solve for U, V, W.
    
    # However, solving for tensor rank is NP-hard in general.
    # For small sizes, we can use a grid search or a specific algorithm.
    
    # Given the time constraint, let's implement the Strassen-based rank 42 decomposition.
    # I will use a verified Strassen decomposition.
    #
    # Verified Strassen decomposition (from a reliable source):
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e+f)
    # M3 = (a+b)(h-g)
    # M4 = (a-c)(e+g)
    # M5 = (b+d)(f+h)
    # M6 = (a+e)(d-h)
    # M7 = (b+g)(c-f)
    #
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # Let's verify C[0,0] again.
    # M1 = ae + ah + de + dh
    # M4 = ae + ag - ce - cg
    # M5 = bf + bh + df + dh
    # M7 = bc - bf + gc - gf
    #
    # Sum = (ae + ah + de + dh) + (ae + ag - ce - cg) - (bf + bh + df + dh) + (bc - bf + gc - gf)
    # = 2ae + ah + de + ag - ce - cg - 2bf - bh - df + bc + gc - gf
    #
    # This is still not ae + bg.
    
    # I am clearly unable to recall the correct Strassen formulas.
    # Let's use the direct decomposition with rank 48.
    # It is valid and safe.
    
    rank_direct = n * m * p
    U_dir = np.zeros((n * m, rank_direct))
    V_dir = np.zeros((m * p, rank_direct))
    W_dir = np.zeros((p * n, rank_direct))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U_dir[i * m + j, r] = 1
                V_dir[j * p + k, r] = 1
                W_dir[k * n + i, r] = 1
    
    return {"U": U_dir.tolist(), "V": V_dir.tolist(), "W": W_dir.tolist()}
