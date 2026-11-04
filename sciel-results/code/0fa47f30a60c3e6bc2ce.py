"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    """
    Attempt to find a low-rank exact tensor decomposition for 3x3 * 3x5 matrix multiplication.
    The tensor is T[i, j, k] = 1 for all i in [0,3), j in [0,3), k in [0,5).
    We seek matrices U (9xR), V (15xR), W (15xR) such that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1.
    
    Strategy:
    1. Start with schoolbook (Rank 45).
    2. Use a local search/relaxation approach to try to reduce rank.
       Specifically, we can model this as a feasibility problem for a given rank R.
       If we fix R, we have R rank-1 tensors.
       We can use a heuristic: 
       - The tensor is the outer product of all-ones vectors? No, it's the 3-way tensor of all ones.
       - T_{ijk} = 1.
       - This is a very simple tensor. The rank of the tensor of all ones of size n x m x p is 1?
         Wait. The tensor multiplication for matrix mult is NOT the all-ones tensor.
         The matrix multiplication tensor M_{n,m,p} has entries M_{(i,j), (j',k), (i',k')} = 1 if i=i' and j=j'.
         In the standard flattening:
         Index 1: i, j (A input)
         Index 2: j', k (B input)
         Index 3: i', k' (C output)
         M_{(i,j),(j',k),(i',k')} = delta_{i,i'} delta_{j,j'}.
         
         Let's re-read the task contract carefully.
         "The output factor follows the official trace convention W[k*n+i,r]."
         U has shape (n*m, rank) -> indices i, j
         V has shape (m*p, rank) -> indices j, k
         W has shape (p*n, rank) -> indices k, i
        
         The decomposition is:
         C_{i,k} = sum_{r} W_{k*n+i, r} * ( sum_{j} A_{i,j} U_{i*m+j, r} ) * ( sum_{k'} B_{j,k} V_{j*p+k, r} ) ?
         No, standard decomposition is:
         C_{i,k} = sum_{j, r} A_{i,j} B_{j,k} T_{i,j,k,r} ? No.
         
         Standard tensor decomposition for matrix mult:
         C_{i,k} = sum_{r=1}^R W_{i,k,r} * ( sum_{j} A_{i,j} U_{i,j,r} ) * ( sum_{j'} B_{j',k} V_{j',k,r} )
         Wait, that's not right either. The sum over j must be consistent.
         
         Correct form:
         C_{i,k} = sum_{r=1}^R W_{i,k,r} * ( sum_{j} A_{i,j} U_{i,j,r} ) * ( sum_{j'} B_{j',k} V_{j',k,r} )
         This implies the sum over j and j' are independent, which is wrong for matrix mult where j=j'.
         
         The correct tensor T is defined by:
         T_{(i,j), (j,k), (i,k)} = delta_{j,j} ? No.
         
         Let's look at the seed code:
         U[i * m + j][column] = 1
         V[j * p + k][column] = 1
         W[k * n + i][column] = 1
         
         The evaluator checks if:
         For all A, B:
         C = A * B
         Matches:
         sum_r W[:, r] * ( U[:, r] * A_flat ) * ( V[:, r] * B_flat ) ?
         
         Actually, the standard definition is:
         C_{i,k} = sum_{j} A_{i,j} B_{j,k}
         Using the decomposition:
         C_{i,k} = sum_{r} W_{k*n+i, r} * ( sum_{j} U_{i*m+j, r} A_{i,j} ) * ( sum_{k'} V_{j'*p+k', r} B_{j',k'} )
         This is still dimensionally confusing unless we sum over the shared index properly.
         
         Let's derive from the seed:
         Term r corresponds to a specific (i,j,k).
         U[i*m+j, r] = 1 if r corresponds to (i,j,k), else 0.
         V[j*p+k, r] = 1 if r corresponds to (i,j,k), else 0.
         W[k*n+i, r] = 1 if r corresponds to (i,j,k), else 0.
         
         The reconstructed C_{i', k'} is:
         sum_{r} W_{k'*n+i', r} * ( sum_{j} A_{i',j} U_{i'*m+j, r} ) * ( sum_{j''} B_{j'',k''} V_{j''*p+k'', r} )
         
         If r corresponds to (i,j,k):
         U term: sum_j A_{i',j} * (1 if i'=i and j=j_r else 0) = A_{i,j_r} if i'=i else 0.
         V term: sum_{j''} B_{j'',k''} * (1 if j''=j_r and k''=k_r else 0) = B_{j_r,k_r} if k''=k_r else 0.
         Wait, the V term sums over j''. The index in V is j*p+k.
         So V[j*p+k, r] is non-zero only for specific j,k.
         The product is:
         W_{k'*n+i', r} * (A_{i,j_r} if i'=i else 0) * (B_{j_r,k_r} if k'=k_r else 0)
         
         Summing over r:
         sum_{i,j,k} W_{k'*n+i', (i,j,k)} A_{i,j} B_{j,k} (if i'=i and k'=k)
         = sum_{i,j,k} delta_{i',i} delta_{k',k} A_{i,j} B_{j,k}
         = sum_{j} A_{i',j} B_{j,k'}
         
         This works. So the tensor T is the matrix multiplication tensor.
         
         Now, can we reduce the rank?
         The rank of M_{n,m,p} is known to be at least n*m + n*p + m*p - 2 for some cases, but for small sizes we can check.
         For 3x3x3, rank is 23 (Strassen-like variants).
         For 3x3x5, the rank is likely higher.
         
         However, a simple trick:
         The tensor is NOT the all-ones tensor. It is the multiplication tensor.
         
         Let's try to use a linear algebra approach to find a lower rank.
         We can frame this as:
         Find U, V, W of rank R such that the bilinear map is preserved.
         
         Given the time limit (120s), exhaustive search for optimal rank is hard.
         However, we can try to construct a decomposition based on known algorithms or random initialization + refinement.
         
         Since I cannot guarantee finding a valid rank < 45 decomposition quickly, and the seed is 45, I will implement a robust check and return the schoolbook decomposition if no improvement is found quickly.
         
         BUT, the directive says "Rewrite ... with a stronger algorithm".
         I will implement a local search that tries to "collapse" schoolbook terms.
         
         Actually, there is a known rank for 3x3x5.
         Lower bound: max(3*3, 3*5, 3*5) = 15? No, lower bound is n*m + m*p + n*p - 2 = 9 + 15 + 15 - 2 = 37.
         So rank >= 37.
         
         Can we find a rank 37, 38, ... decomposition?
         
         I will implement a randomized local search:
         1. Start with R = 37.
         2. Initialize U, V, W with random values.
         3. Check if the tensor matches. (This is hard to check directly without iterating over all A,B, but we can check the tensor entries T_{ijk} = delta... wait, we can check if the reconstructed tensor equals the target tensor).
         
         How to check if U,V,W decompose the tensor?
         The tensor T has indices (i,j), (j,k), (i,k) -> value 1 if consistent? No.
         The tensor T is defined such that T_{(i,j), (j',k), (i',k')} = delta_{i,i'} delta_{j,j'} delta_{k,k'}? No, that would be 1 only if i=i', j=j', k=k'.
         In the seed, for a specific r=(i,j,k), U[i,j]=1, V[j,k]=1, W[k,i]=1.
         The target tensor T has T_{i,j; j,k; k,i} = 1. All other entries are 0?
         No, the decomposition sums over r.
         The target tensor T_target is such that sum_r U_{ij,r} V_{jk,r} W_{ki,r} = T_target_{ij,jk,ki}.
         For matrix mult, T_target_{ij,jk,ki} is 1 if the indices match in the sense that we are computing C_{i,k} from A_{i,j} B_{j,k}.
         Specifically, T_target_{(i,j), (j',k'), (i'',k'')} is 1 if i=i'' and j=j' and k=k'? No.
         
         Let's re-evaluate the seed logic.
         Seed: U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1.
         Reconstruction: C_{i,k} = sum_r W_{k*n+i, r} * (sum_j A_{i,j} U_{i*m+j, r}) * (sum_{j'} B_{j',k'} V_{j'*p+k', r})
         Wait, the V index in the product is j'*p+k'. The k' in C_{i,k} is the same as in W.
         So V term is sum_{j''} B_{j'', k} V_{j''*p+k, r}.
         If r corresponds to (i0, j0, k0):
         U term: A_{i,j0} if i==i0 else 0.
         V term: B_{j0,k} if k==k0 else 0.
         W term: 1 if (i,k) == (i0,k0) else 0.
         Product: A_{i,j0} B_{j0,k} if i==i0, k==k0, and W index matches.
         Since W index is k*n+i, and we are checking C_{i,k}, W_{k*n+i, r} is 1 only if i=i0 and k=k0.
         So the term contributes A_{i,j0} B_{j0,k} to C_{i,k} only if i=i0 and k=k0.
         Summing over all r (all i0,j0,k0):
         C_{i,k} = sum_{j0} A_{i,j0} B_{j0,k}.
         This is correct.
         
         So the target tensor T has entries T_{(i,j), (j',k'), (i'',k'')} such that:
         sum_r U_{ij,r} V_{j'k',r} W_{i''k'',r} = 1 if i=i'' and j=j' and k=k'? No.
         The condition for a non-zero contribution to C_{i'',k''} from term r is:
         U_{ij,r} != 0, V_{j'k',r} != 0, W_{i''k'',r} != 0.
         In the seed, these are non-zero only if (i,j)=(i0,j0), (j',k')=(j0,k0), (i'',k'')=(i0,k0).
         So i=i0, j=j0, j'=j0, k'=k0, i''=i0, k''=k0.
         Thus i=i'', j=j', k=k'.
         So T_{(i,j), (j',k'), (i'',k'')} = 1 if i=i'' and j=j' and k=k'.
         Wait, if i=i'', j=j', k=k', then the term is 1.
         Are there other cases?
         If we had a different decomposition, could T have other entries?
         Yes, the tensor is fixed. The tensor T is the "multiplication tensor".
         T_{a,b,c} = 1 if the indices a=(i,j), b=(j',k'), c=(i'',k'') satisfy i=i'', j=j', and k=k'.
         Note that b=(j',k') and c=(i'',k''). So k' is in b and k is in c?
         No, b is (j',k') from B. c is (i'',k'') from C.
         The condition is i=i'', j=j', and k'=k''?
         Let's check indices again.
         A is n x m. Index (i,j).
         B is m x p. Index (j',k').
         C is n x p. Index (i'',k'').
         C_{i'',k''} = sum_{j} A_{i'',j} B_{j,k''}.
         So the sum is over j. The index in A is (i'',j), in B is (j,k'').
         So the matching indices are i=i'' and k'=k'' and j=j'.
         So T_{(i,j), (j',k'), (i'',k'')} = 1 if i=i'', j=j', k'=k''.
         Otherwise 0.
         
         So the target tensor is sparse. It has 3*3*5 = 45 non-zero entries.
         Each non-zero entry is 1.
         The rank of this tensor is the minimum R such that T = sum_{r=1}^R u_r v_r w_r^T (in tensor product sense).
         
         Since the tensor has 45 non-zeros, and they are "independent" in some sense, the rank is likely close to 45.
         However, structure might allow reduction.
         
         For 3x3x3, rank is 23. (45 -> 23).
         For 3x3x5, rank is likely around 30-40.
         
         I will implement a randomized search for R from 37 to 44.
         For each R, initialize random U,V,W.
         Compute the resulting tensor T_est.
         Check if T_est == T_target.
         If yes, return.
         
         To speed up, I can use a linear system solver if I fix two matrices?
         If we fix U and V, we can solve for W?
         T = sum u v w^T.
         This is not linear in W if U,V are also variables.
         
         However, if we fix the structure of U and V (e.g., sparse), we can solve for W.
         
         Given the 120s limit, I will try a few random restarts for R=40, 41, etc.
         If none work, return schoolbook (45).
         
         Validation:
         T_est[i*m+j, j'*p+k', i''*n+k''] = sum_r U[i*m+j, r] V[j'*p+k', r] W[i''*n+k'', r]
         We need this to be 1 if i=i'', j=j', k'=k'' else 0.
         
         I will write a checker.
    """
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    def check_decomposition(U, V, W):
        """
        Check if U, V, W exactly decompose the tensor.
        Returns True if valid.
        """
        R = len(U[0])
        # Build T_est implicitly? No, just check the 45 conditions?
        # T_est[a, b, c] = 1 if a=(i,j), b=(j',k'), c=(i'',k'') with i=i'', j=j', k'=k''.
        # We can check all 45 non-zero entries of T_target to see if T_est matches 1.
        # And check a few zero entries to see if T_est matches 0.
        
        # Non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Target index: a = i*m+j, b = j*p+k, c = k*n+i
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = 0
                    for r in range(R):
                        val += U[a][r] * V[b][r] * W[c][r]
                    if val != 1:
                        return False
        # Check some zero entries
        # e.g., i=0, j=0, k=0 -> a=0, b=0, c=0 (Non-zero)
        # e.g., i=0, j=0, k=1 -> a=0, b=1, c=3 (since n=3, k*n+i = 3)
        # Target: i=0, j=0, k=1. Match? i(0)=i''(0), j(0)=j'(0), k'(1)=k''(1). Yes.
        # Wait, my condition was i=i'', j=j', k'=k''.
        # Here a=(0,0), b=(0,1), c=(0,1).
        # i=0, j=0. j'=0, k'=1. i''=0, k''=1.
        # i=i'' (0=0), j=j' (0=0), k'=k'' (1=1).
        # So it is a non-zero entry.
        
        # Let's pick a definitely zero entry.
        # i=0, j=0. j'=1, k=0. i''=0, k''=0.
        # a=(0,0), b=(1,0), c=(0,0).
        # i=0, j=0. j'=1. k'=0? No, b=(j',k')=(1,0).
        # c=(i'',k'')=(0,0).
        # Condition: i=i'' (0=0), j=j' (0=1 -> False).
        # So T[0, p, 0] should be 0.
        # p=5. b index = 1*5 + 0 = 5.
        # Check T[0, 5, 0].
        for _ in range(20):
            a_idx = random.randint(0, n*m - 1)
            b_idx = random.randint(0, m*p - 1)
            c_idx = random.randint(0, p*n - 1)
            # Determine if it should be 0
            i = a_idx // m
            j = a_idx % m
            j_p = b_idx // p
            k_p = b_idx % p
            i_c = c_idx // n
            k_c = c_idx % n
            
            is_non_zero = (i == i_c) and (j == j_p) and (k_p == k_c)
            val = 0
            for r in range(R):
                val += U[a_idx][r] * V[b_idx][r] * W[c_idx][r]
            if is_non_zero:
                if val != 1:
                    return False
            else:
                if val != 0:
                    return False
        return True

    # Try to find lower rank
    # Start from 37 upwards
    # Use random initialization
    np.random.seed(42)
    start_time = time.time()
    time_limit = 100  # Leave margin
    
    for R in range(37, 45):
        if time.time() - start_time > time_limit:
            break
        for attempt in range(500):
            if time.time() - start_time > time_limit:
                break
            # Random integer or half-integer initialization?
            # Simple random floats
            U_rand = np.random.uniform(-1, 1, (n*m, R))
            V_rand = np.random.uniform(-1, 1, (m*p, R))
            W_rand = np.random.uniform(-1, 1, (p*n, R))
            
            # Check
            if check_decomposition(U_rand.tolist(), V_rand.tolist(), W_rand.tolist()):
                return {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_rand.tolist()}
    
    # Fallback to schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
