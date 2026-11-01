import time
import numpy as np
from scipy.optimize import linprog

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Known lower bound: for 2x2x2, rank is 7 (Strassen).
    # For 2x2x7, we can decompose into 3 blocks of 2x2x2 (rank 7 each) and 1 block of 2x2x1 (rank 4).
    # Total rank = 3*7 + 4 = 25.
    
    # Can we do better?
    # A tensor decomposition of rank r for [2,2,7] implies a decomposition of [2,2,2] x [2,2,1] tensor product structure? No, not necessarily.
    # However, the matrix multiplication tensor is a sum of Kronecker products if we view it as a block matrix.
    # The tensor T_{2,2,7} = T_{2,2,2} \otimes T_{2,2,1} is not correct.
    # Actually, T_{2,2,7} corresponds to multiplying A (2x2) by B (2x7).
    # B can be viewed as 3 blocks of 2x2 and 1 block of 2x1? No, B is 2x7.
    # Let's partition the columns of B into groups.
    # If we group columns into sizes, say 3, 2, 2?
    # Or 2, 2, 2, 1.
    # The rank of a Kronecker product of tensors is the product of their ranks.
    # But matrix multiplication is not just a Kronecker product of two matrix multiplications.
    # It is a block diagonal structure if we partition the output columns.
    # Let B = [B1, B2, B3] where B1 is 2x2, B2 is 2x2, B3 is 2x3?
    # Then AB = [AB1, AB2, AB3].
    # The tensor for AB1 is T_{2,2,2} (rank 7).
    # The tensor for AB2 is T_{2,2,2} (rank 7).
    # The tensor for AB3 is T_{2,2,3}. What is the rank of T_{2,2,3}?
    # T_{2,2,3} can be decomposed into T_{2,2,2} + T_{2,2,1}.
    # Rank(T_{2,2,1}) = 4 (since it's just 4 scalar multiplications, and rank of [2,2,1] is 4).
    # Is rank(T_{2,2,3}) = 7 + 4 = 11?
    # Or can we do better for [2,2,3]?
    # Lower bound for [2,2,3]:
    # Using the method of substitution or other bounds.
    # Known ranks:
    # [1,1,1] -> 1
    # [1,1,2] -> 2
    # [1,2,2] -> 4
    # [2,2,2] -> 7
    # [2,2,3] -> ?
    # A decomposition of [2,2,3] with rank 10?
    # Let's check if rank 10 is possible for [2,2,3].
    # If rank([2,2,3]) = 10, then total rank = 7 + 7 + 10 = 24.
    # If rank([2,2,3]) = 9, total = 23.
    
    # Let's try to construct a rank-10 decomposition for [2,2,3].
    # If we can find one, we get 24.
    # If not, maybe 25 is optimal or close.
    
    # Let's attempt an optimization for rank R.
    # We will start with R=24 and try to find a valid decomposition.
    # If that fails, we fall back to the known R=25.
    
    # The verification step is O(n*m*p^2 * R) which is small.
    # The construction via random restarts + local search might be too slow for 120s if R is high.
    # However, we have a strong structural prior: the block diagonal structure.
    
    # Strategy:
    # 1. Return the known rank-25 solution immediately to ensure a valid score > 1.0 (since 28/25 = 1.12).
    # 2. If we have time, try to optimize to rank 24 by searching for a better [2,2,3] decomposition.
    
    # Construct Rank 25 Solution
    def get_rank_25_solution():
        # Strassen for 2x2x2
        U_s = [
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1,  0, 1],
            [0, 1, 0, 0, 0,  1, 0],
            [1, 1, 0, 1, 0,  0,-1]
        ]
        V_s = [
            [1, 1, 0, -1, 0, 1, 0],
            [0, 0, 1,  0, 0, 1, 0],
            [0, 0, 0,  1, 0, 0, 1],
            [1, 0,-1,  0, 1, 0, 1]
        ]
        W_s = [
            [1, 0, 0, 1, -1, 0, 1],
            [0, 0, 1, 0,  1, 0, 0],
            [0, 1, 0, 1,  0, 0, 0],
            [1,-1, 1, 0,  0, 1, 0]
        ]
        
        rank = 25
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        # 3 Strassen blocks (for columns 0-1, 2-3, 4-5)
        for block in range(3):
            k_base = block * 2
            for r in range(7):
                for i in range(n):
                    for j in range(m):
                        U[i * m + j][col] = U_s[i * m + j][r]
                for j in range(m):
                    for kk in range(2):
                        V[j * p + (k_base + kk)][col] = V_s[j * 2 + kk][r]
                for i in range(n):
                    for kk in range(2):
                        W[(k_base + kk) * n + i][col] = W_s[i * 2 + kk][r]
                col += 1
            
        # 1 Schoolbook block for column 6 (2x2x1 effectively, but only 1 col of B)
        # Wait, the last block is 2x1 of B? No, B is 2x7.
        # The partition was 2,2,2,1.
        # The last block corresponds to B[:, 6:7], which is 2x1.
        # The multiplication A (2x2) * B_last (2x1) results in C_last (2x1).
        # This is a [2,2,1] tensor. Rank is 4.
        # U part: A (2x2) -> 4 entries.
        # V part: B_last (2x1) -> 2 entries.
        # W part: C_last (2x1) -> 2 entries.
        # Decomposition: C_{i,k} = sum_j A_{i,j} B_{j,k}.
        # For fixed k=6, C_{i,6} = A_{i,0}B_{0,6} + A_{i,1}B_{1,6}.
        # We need 4 terms:
        # 1. A00 * B06 -> C06
        # 2. A10 * B06 -> C16
        # 3. A01 * B16 -> C06
        # 4. A11 * B16 -> C16
        
        # Indices in U: i*m+j
        # Indices in V: j*p+k
        # Indices in W: k*n+i
        
        # Term 1: U[0,0]=1, V[0,6]=1, W[6,0]=1
        U[0][col] = 1
        V[0 * p + 6][col] = 1
        W[6 * n + 0][col] = 1
        col += 1
        
        # Term 2: U[2,0]=1 (i=1,j=0), V[0,6]=1, W[6,1]=1
        U[1 * m + 0][col] = 1
        V[0 * p + 6][col] = 1
        W[6 * n + 1][col] = 1
        col += 1
        
        # Term 3: U[0,1]=1 (i=0,j=1), V[1,6]=1, W[6,0]=1
        U[0 * m + 1][col] = 1
        V[1 * p + 6][col] = 1
        W[6 * n + 0][col] = 1
        col += 1
        
        # Term 4: U[1,1]=1 (i=1,j=1), V[1,6]=1, W[6,1]=1
        U[1 * m + 1][col] = 1
        V[1 * p + 6][col] = 1
        W[6 * n + 1][col] = 1
        col += 1
        
        return {"U": U, "V": V, "W": W}
    
    # Check if we have time to try rank 24
    # Rank 24 would imply rank([2,2,3]) <= 10.
    # Let's try a random search for a rank-10 decomposition of [2,2,3].
    # If found, we can combine with two Strassen blocks to get rank 24.
    
    def try_find_rank_10_223():
        # T_{2,2,3}
        # n=2, m=2, p=3
        R = 10
        # Equations:
        # For each (i,j,k,kp,ip) ? No.
        # Tensor entries T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} ? No.
        # Convention: W[k*n+i, r].
        # T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We want T_{i,j,k} = delta_{i,i} delta_{j,j} delta_{k,k}? No.
        # The matrix multiplication tensor is defined such that
        # sum_k A_{i,k} B_{k,j} = C_{i,j}.
        # Here our dims are n=2 (rows of A), m=2 (cols of A/rows of B), p=3 (cols of B).
        # A is 2x2. B is 2x3. C is 2x3.
        # C_{i,k} = sum_j A_{i,j} B_{j,k}.
        # So the tensor T has indices (i, j, k) where i is row of C, j is col of A/row of B, k is col of B/col of C.
        # T_{i,j,k} = 1 if it contributes to C_{i,k} from A_{i,j}B_{j,k}.
        # Actually, the tensor is T_{i,j,k} = delta_{i,i} ... no.
        # The tensor T is the structure tensor.
        # T_{i,j,k} = 1 if (i,j,k) corresponds to a valid multiplication term?
        # No, T_{i,j,k} is 1 if the term A_{i,j} B_{j,k} is present.
        # Since all terms are present, T_{i,j,k} = 1 for all valid i,j,k?
        # No, that would be a rank-1 tensor (all ones).
        # The tensor T is defined by T_{(i,j), (j,k), (k,i)} = 1? No.
        
        # Let's stick to the equation:
        # For all i, j, k:
        # delta_{i, i} delta_{j, j} delta_{k, k} is not the right way.
        # The equation is:
        # For each output C_{i,k}, it is sum_j A_{i,j} B_{j,k}.
        # This means the coefficient of A_{i,j} B_{j,k} in C_{i,k} is 1.
        # So the tensor T_{i,j,k} is actually a 3-index object where:
        # T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
        # This is always 1 for valid indices.
        # But the decomposition is:
        # C_{i,k} = sum_r (sum_j U_{i,j,r} A_{i,j}) * (sum_k' V_{j,k',r} B_{j,k'}) ... No.
        
        # Correct decomposition form:
        # Let U_{i,j,r}, V_{j,k,r}, W_{k,i,r}.
        # The product is sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} ? No.
        
        # Standard tensor decomposition for matrix mult [n,m,p]:
        # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
        # We require T_{i,j,k} = 1 if i,j,k are valid indices for the term A_{i,j}B_{j,k} contributing to C_{i,k}.
        # Wait, the indices of T are (i, j, k) corresponding to (row_C, col_A/row_B, col_B).
        # The term A_{i,j} B_{j,k} contributes to C_{i,k}.
        # So T_{i,j,k} = 1.
        # And the decomposition must satisfy:
        # sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1 for all i,j,k?
        # No, that would mean all entries are 1.
        
        # The correct condition is:
        # The bilinear map is M(A,B) = sum_r (U_r * A) * (V_r * B) * W_r? No.
        
        # Let's use the explicit equation from the seed program:
        # lhs = 1.0 if (i == ip and j == jp and k == kp) else 0.0
        # rhs = sum_r U[i*m+j][r] * V[jp*p+kp][r] * W[kp*n+ip][r]
        
        # This implies that the tensor T is defined on indices (i, j, k, jp, kp, ip)? No.
        # The seed program iterates over:
        # i, j (indices of A)
        # jp, kp (indices of B? No, jp is row of B, kp is col of B)
        # ip (row of C)
        
        # lhs is 1 if i==ip, j==jp, k==kp.
        # This means the tensor T has non-zero entries T_{i,j,k, jp,kp,ip} ?
        
        # Actually, the standard matrix multiplication tensor T_{[n,m,p]} has 3 modes of sizes n*m, m*p, p*n.
        # Entry T_{(i,j), (j',k'), (k'',i'')} is 1 if i=i'', j=j', k=k'?
        # Let's check the seed:
        # U index: i*m+j
        # V index: jp*p+kp
        # W index: kp*n+ip
        
        # The condition is:
        # sum_r U[i*m+j, r] V[jp*p+kp, r] W[kp*n+ip, r] = delta_{i,ip} delta_{j,jp} delta_{k,kp}
        
        # This means:
        # If we fix i, j, k, jp, kp, ip, the sum is 1 only if i=ip, j=jp, k=kp.
        # Otherwise 0.
        
        # This is the definition of the matrix multiplication tensor.
        
        # So for [2,2,3]:
        # n=2, m=2, p=3.
        # U: 4xR
        # V: 6xR
        # W: 6xR
        
        # We want to find U, V, W such that:
        # sum_r U[i*m+j, r] V[jp*p+kp, r] W[kp*n+ip, r] = delta_{i,ip} delta_{j,jp} delta_{k,kp}
        
        # This is a system of linear equations in the products? No, it's trilinear.
        # We can fix one matrix, say W, and solve for U and V? No, still bilinear.
        
        # We can iterate over one variable?
        # Let's try a random search with local optimization.
        # Given R=10, we have 4*10 + 6*10 + 6*10 = 160 variables.
        # The constraints are for all i,j,k,jp,kp,ip.
        # Number of constraints: n * m * p * m * p * n = 2 * 2 * 3 * 2 * 3 * 2 = 144.
        # But many are zero.
        # Non-zero constraints are when i=ip, j=jp, k=kp.
        # Number of non-zero constraints: n * m * p = 2 * 2 * 3 = 12.
        # Each such constraint requires sum_r ... = 1.
        # The other 132 constraints require sum_r ... = 0.
        
        # This is a system of 144 polynomial equations of degree 3.
        
        # We can use a simple random restart + gradient descent?
        # Or we can fix W to a random matrix and solve for U and V?
        # If W is fixed, the equations are linear in U and V?
        # sum_r U[i*m+j, r] V[jp*p+kp, r] W[kp*n+ip, r] = C_{i,j,k,jp,kp,ip}
        
        # This is not linear in U and V jointly.
        
        # However, if we fix V and W, it is linear in U.
        # If we fix U and W, it is linear in V.
        # If we fix U and V, it is linear in W.
        
        # We can do alternating least squares (ALS) on the tensors?
        
        # Let's try a few random restarts.
        
        np.random.seed(42)
        
        for restart in range(50):
            # Random initialization
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            # Optimize?
            # For simplicity, we just check if a random solution exists? Unlikely.
            # We need a better search.
            
            # Let's use the fact that we can scale columns of U, V, W.
            
            # Given time constraints, I will not implement a full ALS solver here.
            # I will stick to the rank 25 solution which is guaranteed.
            
            pass
            
        return None
    
    # For now, return rank 25.
    # If time permits, I could try to verify if rank 24 is possible with a more sophisticated search,
    # but the risk of timeout is high.
    
    solution = get_rank_25_solution()
    
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            lhs = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            rhs = 0.0
                            for r in range(25):
                                rhs += solution["U"][i * m + j][r] * solution["V"][jp * p + kp][r] * solution["W"][kp * n + ip][r]
                            if abs(lhs - rhs) > 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if not valid: break
            if not valid: break
        if not valid: break
        
    if valid:
        return solution
        
    # Fallback
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
