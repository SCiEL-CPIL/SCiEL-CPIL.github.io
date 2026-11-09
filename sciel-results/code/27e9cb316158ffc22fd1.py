"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    schoolbook_rank = n * m * p  # 105
    
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We need U (n*m x r), V (m*p x r), W (p*n x r)
    # such that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Known lower bound: rank >= max(n*m, m*p, p*n) is not quite right for matrix mult.
    # For matrix multiplication, the tensor rank is at least max(n*m, m*p, p*n) is false.
    # Actually, for M_{n,m,p}, the rank is at least max(n*m, m*p, p*n) is not a known bound.
    # However, we know rank >= n*m, rank >= m*p, rank >= p*n is not necessarily true.
    # But we do know that for M_{2,2,2}, rank is 7 (Strassen).
    # For M_{3,5,7}, we don't have a known minimal rank, but it's likely much less than 105.
    
    # Let's try to find a decomposition with rank < 105.
    # We'll use a random search with local optimization.
    
    # First, let's check if we can find a solution with rank close to theoretical bounds.
    # A simple bound: rank >= n*m = 15, rank >= m*p = 35, rank >= p*n = 21.
    # So rank >= 35.
    
    # Let's try ranks from 35 upwards.
    # We'll use a randomized approach: initialize U, V, W randomly, then use ALS to minimize
    # the error ||T - sum_c U_c V_c W_c||^2.
    
    # But the tensor is all ones. Let's exploit symmetry.
    # If we set U, V, W to have certain structure...
    
    # Actually, let's think about this differently.
    # The tensor T[i,j,k] = 1 for all i,j,k.
    # This is a "constant" tensor.
    
    # One approach: use the fact that 1 = sum_c a_c b_c c_c where a, b, c are vectors.
    # But we need the tensor structure.
    
    # Let's try a simple construction:
    # If we can find vectors u_i in R^r, v_j in R^r, w_k in R^r such that
    # sum_c u_i[c] v_j[c] w_k[c] = 1 for all i,j,k.
    
    # This is equivalent to finding a rank-r decomposition of the all-ones tensor.
    
    # The all-ones tensor of size n x m x p has rank 1! Because T[i,j,k] = 1 = 1*1*1.
    # So U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1.
    # This gives rank 1!
    
    # Wait, let me verify:
    # U is (n*m) x 1, V is (m*p) x 1, W is (p*n) x 1.
    # U[i*m+j, 0] = 1 for all i,j.
    # V[j*p+k, 0] = 1 for all j,k.
    # W[k*n+i, 0] = 1 for all k,i.
    
    # Then sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1*1*1 = 1.
    # Yes! This works!
    
    # So the rank is 1.
    
    U = [[1] * 1 for _ in range(n * m)]
    V = [[1] * 1 for _ in range(m * p)]
    W = [[1] * 1 for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
