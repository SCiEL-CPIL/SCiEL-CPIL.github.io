import time
import random
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = 1 if i, j, k are valid indices for the multiplication
    # The tensor T represents the bilinear map (A, B) -> C
    # T[i, j, k] corresponds to the coefficient of A[i, j] * B[j, k] in C[i, k]
    # Wait, the standard tensor for matrix mult C = A * B is defined by:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # The tensor T has indices (i, j, k) where i is row of A, j is col of A / row of B, k is col of B.
    # T[i, j, k] = 1 if the term A[i, j]B[j, k] contributes to C[i, k].
    # Actually, the tensor is usually indexed as T[i, j, k] such that the bilinear form is sum T[i, j, k] A[i, j] B[j, k] = C[i, k].
    # This implies T[i, j, k] is 1 only if the output index (i, k) matches the input indices.
    # Let's define the tensor T of shape (n, m, p) where T[i, j, k] = 1 if i, j, k form a valid multiplication term.
    # In the decomposition U, V, W:
    # C[i, k] = sum_r U[i, r] V[r, k] ... no, the factors are usually:
    # C[i, k] = sum_r U[i, m, r] V[m, p, r] W[k, n, r] ? 
    # The prompt says: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank).
    # Let's unpack the indices.
    # U index: i*m + j. V index: j*p + k. W index: k*n + i.
    # The product sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] must equal 1 if (i, j, k) is a valid term, else 0?
    # No, the tensor decomposition identity is:
    # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # For matrix multiplication, T[i, j, k] = 1 if the indices are consistent? 
    # Actually, the multiplication tensor T_mul has T_mul[i, j, k] = 1 for all valid i, j, k? 
    # No, that's not right. The tensor is defined by the bilinear map.
    # The tensor T is a 3-tensor. The decomposition is T = sum_r U_r \otimes V_r \otimes W_r.
    # U_r is a vector of size n*m. V_r is a vector of size m*p. W_r is a vector of size p*n.
    # The entry T[i, j, k] is the sum over r of U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i].
    # For matrix multiplication C = A * B, the coefficient of A[i, j] * B[j, k] in C[i, k] is 1.
    # So T[i, j, k] = 1 for all i in [0, n), j in [0, m), k in [0, p).
    # Wait, is that true? 
    # C[i, k] = sum_j A[i, j] B[j, k].
    # The term A[i, j] B[j, k] appears in C[i, k].
    # So yes, T[i, j, k] = 1 for all valid i, j, k.
    # This means the tensor is the "all-ones" tensor of shape (n, m, p)?
    # No, that would mean C[i, k] = sum_{j} A[i, j] B[j, k] is represented by a tensor where every entry is 1.
    # Let's check the schoolbook decomposition.
    # Rank = n*m*p.
    # For each (i, j, k), we have a rank-1 term.
    # U has 1 at [i*m+j], V has 1 at [j*p+k], W has 1 at [k*n+i].
    # Sum_r ... = 1 if (i, j, k) matches the specific term, else 0.
    # So T[i, j, k] = 1 for all i, j, k.
    
    # So we are looking for a rank-r decomposition of the all-ones tensor of shape (n, m, p).
    # This is equivalent to finding the rank of the tensor T where T[i, j, k] = 1.
    
    # However, the "all-ones" tensor is rank 1! 
    # If T[i, j, k] = 1 for all i, j, k, then T = u \otimes v \otimes w where u, v, w are all-ones vectors.
    # Why is the schoolbook rank n*m*p then?
    # Ah, the indices in U, V, W are coupled.
    # U is indexed by (i, j). V by (j, k). W by (k, i).
    # The term is U[i, j] * V[j, k] * W[k, i].
    # The tensor entry T[i, j, k] is the sum over r of these products.
    # For the multiplication tensor, we want T[i, j, k] = 1 if the indices form a valid chain?
    # No, the tensor is defined on the space of matrices A (n x m), B (m x p), and the output C (n x p).
    # The bilinear form is <C, A \otimes B> = trace(C^T A B).
    # The tensor T has indices (i, k) for C, and (i, j) for A, (j, k) for B.
    # So T[i, k, i, j, j, k] = 1? No, that's a 6-index tensor.
    # The standard "matrix multiplication tensor" T_{n,m,p} is a 3-tensor with indices (i, j, k) where:
    # i indexes rows of A (and C), j indexes cols of A (and rows of B), k indexes cols of B (and C).
    # The entry T[i, j, k] is 1 if the indices are consistent? 
    # Actually, T[i, j, k] = 1 for all i, j, k is NOT the matrix multiplication tensor.
    # The matrix multiplication tensor is the tensor of the bilinear map.
    # The map is (A, B) -> C.
    # C_{i,k} = \sum_j A_{i,j} B_{j,k}.
    # The tensor T is defined such that C_{i,k} = \sum_{j} T_{i,k,j} A_{i,j} B_{j,k}? No.
    # C_{i,k} = \sum_{j} A_{i,j} B_{j,k}.
    # This is a sum over j.
    # The tensor T has entries T[i, j, k] = 1.
    # Wait, if T[i, j, k] = 1 for all i, j, k, then the rank is 1?
    # Let's check the decomposition formula again.
    # T[i, j, k] = \sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)}
    # If we pick U_{i,j}^{(1)} = 1, V_{j,k}^{(1)} = 1, W_{k,i}^{(1)} = 1, then sum is 1.
    # So rank 1 would work?
    # Why is the schoolbook rank n*m*p?
    # Because the factors U, V, W are constrained to be "diagonal" or sparse in the schoolbook method?
    # No, the schoolbook method uses n*m*p rank-1 terms.
    # Each term corresponds to a specific (i, j, k) triple.
    # U has 1 at (i, j), 0 elsewhere.
    # V has 1 at (j, k), 0 elsewhere.
    # W has 1 at (k, i), 0 elsewhere.
    # This constructs the tensor where T[i, j, k] = 1.
    # So the tensor IS the all-ones tensor.
    # And the all-ones tensor has rank 1.
    # Why do we need rank n*m*p?
    # Ah, the indices of U, V, W in the decomposition sum over r.
    # The indices i, j, k are FIXED for the tensor entry T[i, j, k].
    # But U is indexed by (i, j). V by (j, k). W by (k, i).
    # In the sum \sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)}, the indices i, j, k are the same for all three factors.
    # This is NOT a standard tensor product decomposition.
    # A standard tensor product decomposition of a 3-tensor T[a, b, c] is \sum_r U[a, r] V[b, r] W[c, r].
    # Here, the indices are coupled: a=(i,j), b=(j,k), c=(k,i).
    # This is a specific structure known as the "matrix multiplication tensor".
    # It is NOT the all-ones tensor in the standard sense because the indices of the factors are not independent.
    # The factor U lives in R^{n*m}, V in R^{m*p}, W in R^{p*n}.
    # The tensor T is in R^{n*m} \otimes R^{m*p} \otimes R^{p*n}.
    # The entry T[a, b, c] is non-zero only if the indices in a, b, c are consistent (i.e., the j in a matches the j in b, the k in b matches the k in c, the i in c matches the i in a).
    # Specifically, if a = i*m + j, b = j*p + k, c = k*n + i, then T[a, b, c] = 1.
    # Otherwise T[a, b, c] = 0.
    # So T is a sparse tensor with n*m*p non-zero entries.
    # The schoolbook decomposition sets U, V, W to be basis vectors corresponding to these non-zero entries.
    # This gives rank n*m*p.
    # Can we do better?
    # Yes, if we can find a set of rank-1 tensors (u_r, v_r, w_r) such that their sum equals T.
    # This is the tensor rank of the matrix multiplication tensor.
    # For 2x2x2, the rank is 7 (Strassen).
    # For 2x2x8, the rank is at most 7*8 = 56? Or can we do better?
    # The tensor is a "block" tensor. It consists of 8 blocks of 2x2x2 tensors?
    # No, the indices are coupled.
    # However, if we fix k, the slice T[:, :, k] is a 2x2x2 tensor?
    # Let's check.
    # a = i*m + j. b = j*p + k. c = k*n + i.
    # If we fix k, then b depends on j, and c depends on i.
    # The non-zero entries for fixed k are (i, j, k).
    # This looks like 8 independent 2x2x2 tensors?
    # Let's check the coupling.
    # For a fixed k, the entries are (i, j, k).
    # The indices i, j vary from 0 to 1.
    # The value is 1.
    # So for each k, we have a tensor T_k[i, j] = 1? No, it's a 3-tensor slice.
    # Actually, the tensor T can be viewed as a sum of 8 tensors T^{(k)} where T^{(k)} is non-zero only when the index k in b and c matches.
    # But the indices i and j are shared.
    # This structure suggests that the rank might be related to the rank of the 2x2x2 tensor multiplied by 8, or less if there is sharing.
    # Strassen's algorithm for 2x2x2 has rank 7.
    # If we can reuse the Strassen factors for each k, we might get rank 7*8 = 56.
    # But maybe we can do better?
    # The task asks for rank-minimal.
    # Schoolbook is 32.
    # 56 is worse than 32.
    # So we need a rank < 32.
    # Is it possible to have rank < 32?
    # Yes, if the blocks share factors.
    # But the indices i, j are the same for all k.
    # This implies that the factors U and V are the same for all k?
    # No, V depends on k. U depends on i, j. W depends on k, i.
    # If we use Strassen for each k, we need 7 factors for each k.
    # Total 56 factors.
    # Can we mix them?
    # This is a hard problem.
    # However, random search might find a rank < 32 if one exists.
    # Let's try to find a rank < 32 using random search.
    
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    # Target tensor T
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i, else 0.
    # We represent T as a dense array of shape (n*m, m*p, p*n).
    T = np.zeros((n*m, m*p, p*n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
                
    # We want to find U, V, W of rank r such that sum_r U[:, r] V[:, r] W[:, r] == T.
    # Note: The outer product of U[:, r], V[:, r], W[:, r] is a tensor of shape (n*m, m*p, p*n).
    
    # We will search for r from 1 upwards? No, start from schoolbook and go down?
    # Or just random search for r < 32.
    # If we find a valid r, we return it.
    # We should try small r first? Or random r?
    # If rank 1 is impossible, we try 2, etc.
    # But random search for low rank is very hard.
    # Let's try r = 31, 30, ... down to 1?
    # Or just random search for any r < 32.
    
    # Strategy:
    # 1. Try r = 31 down to 1.
    # 2. For each r, do a few random trials.
    # 3. If a valid decomposition is found, return it.
    # 4. If time runs out, return the schoolbook decomposition.
    
    # Schoolbook decomposition
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = get_schoolbook()
    best_rank = n * m * p
    
    # Search for lower rank
    # We'll try ranks from best_rank - 1 down to 1.
    # For each rank, we'll do a limited number of random trials.
    
    # To make the search faster, we can use numpy.
    # T is small (4 * 16 * 16 = 1024 entries).
    
    for r in range(best_rank - 1, 0, -1):
        if time.time() > deadline - 10:
            break
            
        # Number of trials per rank
        # If r is close to best_rank, fewer trials?
        # Let's do 100 trials per rank.
        trials = 100
        
        for _ in range(trials):
            if time.time() > deadline - 2:
                break
                
            # Generate random U, V, W
            # Entries from {-1, 0, 1}
            U_rand = np.random.randint(-1, 2, (n*m, r))
            V_rand = np.random.randint(-1, 2, (m*p, r))
            W_rand = np.random.randint(-1, 2, (p*n, r))
            
            # Compute tensor sum
            # T_sum[a, b, c] = sum_r U_rand[a, r] * V_rand[b, r] * W_rand[c, r]
            # We can use einsum
            # T_sum = np.einsum('ar,br,cr->abc', U_rand, V_rand, W_rand)
            
            # Check if T_sum == T
            # Since T is 0/1, and U,V,W are -1,0,1, the sum can be -r to r.
            # We need exact match.
            
            T_sum = np.einsum('ar,br,cr->abc', U_rand, V_rand, W_rand)
            
            if np.array_equal(T_sum, T):
                # Found a valid decomposition
                best_rank = r
                best_result = {
                    "U": U_rand.tolist(),
                    "V": V_rand.tolist(),
                    "W": W_rand.tolist()
                }
                # We can break out of the trials loop, but maybe try to find even lower rank?
                # The loop structure already handles lower ranks.
                break
        
        # If we found a valid rank, we can continue to lower ranks.
        # But if we didn't, we continue.
        
    return best_result
# EVOLVE-BLOCK-END
