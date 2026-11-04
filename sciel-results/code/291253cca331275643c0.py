import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Target tensor T[i, j, k] = 1
    # We want sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Try ranks from low to high
    # Schoolbook is 42.
    # Let's try to find a rank < 42.
    
    # If we don't find a better rank, return schoolbook.
    schoolbook_rank = n * m * p
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    # Convert to list of lists for output
    def to_list(mat):
        return mat.tolist()

    # Search for better rank
    # We will use a simple gradient descent or random local search on the squared error.
    # Error E = sum_{i,j,k} (1 - sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r])^2
    
    # To make it faster, we can fix one matrix, e.g., W, to be sparse or structured?
    # Or just random search.
    
    # Given the time constraint, a full global optimization might be too slow.
    # However, for small dimensions, rank 14-20 might be findable.
    # Known results: 
    # 2x3x3 rank is 16? No, 2x2x2 is 7. 2x3x3 is at most 18.
    # Let's just try to find any rank < 42.
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = schoolbook_rank
    
    # Try ranks from 1 to schoolbook_rank - 1
    # We'll try a few random restarts for each rank.
    # To be safe, we'll prioritize lower ranks.
    
    # Actually, for 2x3x7, it is highly likely that the rank is close to schoolbook or slightly less.
    # Let's try rank 41, 40, etc.
    # Or maybe there's a known decomposition.
    
    # Let's try a heuristic: 
    # If we can't find a better one quickly, we return schoolbook.
    
    # We will implement a simple local search.
    # Variables: U, V, W
    # Objective: minimize sum of squares of residuals.
    
    # Since we need integer/half-integer, we can search in that space directly?
    # That's a huge space.
    # Better to find float solution then round.
    
    # Let's try rank 14 (half of 28? no).
    # Let's just iterate ranks 1 to 41.
    
    # To save time, we'll only do a few iterations per rank.
    
    # We'll use numpy for speed.
    
    # Precompute indices
    # i in 0..n-1, j in 0..m-1, k in 0..p-1
    # u_idx = i*m + j
    # v_idx = j*p + k
    # w_idx = k*n + i
    
    # We'll try to find a rank r solution.
    
    # For efficiency, we'll only try a subset of ranks if time is tight.
    # But 42 is small.
    
    # Let's try to find a rank 21 solution?
    # Actually, let's just return schoolbook if we don't find a better one fast.
    # The score is 1.0 for schoolbook.
    # If we find rank 41, score is 42/41 > 1.0.
    
    # Let's try to find a rank < 42.
    
    # We'll use a simple randomized local search.
    
    # For each rank r:
    #   for restart in range(num_restarts):
    #     U, V, W = random
    #     for step in range(num_steps):
    #       pick random i,j,k
    #       compute residual
    #       update U, V, W slightly
    #     check if error < tol
    #     if yes, round and verify
    
    # This might be too slow in Python.
    
    # Alternative: Use the fact that T is all 1s.
    # If we set U[i*m+j, r] = a_{i,j,r}, V[j*p+k, r] = b_{j,k,r}, W[k*n+i, r] = c_{k,i,r}
    # sum_r a b c = 1
    
    # This is equivalent to finding a tensor rank decomposition of the all-ones tensor.
    # The all-ones tensor of size n x m x p has rank 1!
    # Wait.
    # T_{i,j,k} = 1.
    # If we set U[i*m+j, 1] = 1, V[j*p+k, 1] = 1, W[k*n+i, 1] = 1.
    # Then sum_r U V W = 1 * 1 * 1 = 1.
    # Is this a valid matrix multiplication decomposition?
    
    # Let's check the definition.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The decomposition gives:
    # C_{i,k} = sum_r (sum_j U[i*m+j, r] A_{i,j}) * (sum_k V[j*p+k, r] B_{j,k}) ... NO.
    
    # The standard decomposition is:
    # C_{i,k} = sum_r U[i*m+j, r] ... wait.
    # The tensor T has indices i, j, k.
    # T_{i,j,k} = 1.
    # The decomposition is T_{i,j,k} = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
    # If this holds, then for any A, B:
    # sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} = sum_{i,j,k} A_{i,j} B_{j,k}
    # = sum_r (sum_{i,j} U[i*m+j, r] A_{i,j}) (sum_{j,k} V[j*p+k, r] B_{j,k}) (sum_{k,i} W[k*n+i, r] ??)
    
    # No, the multiplication is:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The bilinear form is sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} C^*_{i,k}
    # The decomposition of the tensor T allows us to write the product as:
    # C_{i,k} = sum_r (sum_j U[i*m+j, r] A_{i,j}) (sum_j V[j*p+k, r] B_{j,k}) ... this doesn't match indices.
    
    # Let's look at the standard definition again.
    # U is (nm x r), V is (mp x r), W is (pn x r).
    # The product is C = A B.
    # The decomposition is valid if:
    # C_{i,k} = sum_r (sum_j U[i*m+j, r] A_{i,j}) (sum_j V[j*p+k, r] B_{j,k}) ... NO.
    
    # Correct expansion:
    # The tensor T_{i,j,k} represents the coefficient of A_{i,j} B_{j,k} in the product.
    # Actually, the product C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # So the tensor is T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} appears in C_{i,k}.
    # But C_{i,k} depends on i, k. The term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # So T_{i,j,k} = 1 for all i, j, k.
    
    # The rank of this tensor is the minimum number of multiplications.
    # If T_{i,j,k} = 1 for all i, j, k, then the tensor is the all-ones tensor.
    # The rank of the all-ones tensor is 1.
    # Decomposition: U[i*m+j, 1] = 1, V[j*p+k, 1] = 1, W[k*n+i, 1] = 1.
    # Check: sum_r U V W = 1 * 1 * 1 = 1.
    # So rank 1 works?
    
    # Let's verify if this corresponds to a valid matrix multiplication algorithm.
    # If rank is 1, then C_{i,k} = (sum_j A_{i,j} * 1) * (sum_j B_{j,k} * 1) * 1 ?
    # No. The algorithm is:
    # m_1 = sum_j U[i*m+j, 1] A_{i,j}
    # n_1 = sum_j V[j*p+k, 1] B_{j,k}
    # C_{i,k} = sum_r W[k*n+i, r] m_r n_r ?
    
    # Let's check the indices in the trace convention.
    # W[k*n+i, r].
    # The output is C_{i,k}.
    # The standard form is:
    # C_{i,k} = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A_{i,j}) * (sum_j V[j*p+k, r] B_{j,k})
    
    # If U=1, V=1, W=1:
    # C_{i,k} = 1 * (sum_j A_{i,j}) * (sum_j B_{j,k})
    # This is NOT equal to sum_j A_{i,j} B_{j,k} in general.
    
    # So the tensor T_{i,j,k} is NOT the all-ones tensor in the sense of just 1s.
    # The tensor T is defined such that:
    # sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} = sum_{i,k} C_{i,k} C^*_{i,k} ?
    
    # The bilinear form is:
    # <T, A, B, C> = sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} C_{k,i} (using trace convention for W)
    # We want this to be zero for all A, B, C such that C = AB.
    # Or rather, the decomposition computes C = AB.
    
    # The condition for the decomposition to be correct is:
    # sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} C_{k,i} = sum_{i,k} (sum_j A_{i,j} B_{j,k}) C_{k,i}
    # LHS = sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} C_{k,i}
    # RHS = sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i}
    # So we need T_{i,j,k} = 1 for all i, j, k.
    
    # So the tensor IS the all-ones tensor.
    # But my previous check showed that rank 1 decomposition gives wrong result.
    # Why?
    
    # Let's re-read the decomposition formula.
    # C_{i,k} = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A_{i,j}) * (sum_j V[j*p+k, r] B_{j,k})
    
    # If U=1, V=1, W=1:
    # C_{i,k} = (sum_j A_{i,j}) (sum_j B_{j,k})
    # This is not AB.
    
    # So the tensor T is NOT just 1s?
    # The tensor T represents the multiplication map.
    # The map is linear in A and B.
    # The coefficients are 1.
    # But the structure of the decomposition imposes constraints.
    
    # Ah, the indices of U, V, W in the decomposition formula are fixed by the convention.
    # U[i*m+j, r] multiplies A_{i,j}.
    # V[j*p+k, r] multiplies B_{j,k}.
    # W[k*n+i, r] multiplies the product term.
    
    # So the condition is:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i, j, k.
    
    # This IS the condition I derived.
    # And I found that U=1, V=1, W=1 satisfies this.
    # But it gives wrong matrix product.
    
    # Contradiction.
    # Let's check the trace convention again.
    # "output factor follows the official trace convention W[k*n+i,r]"
    
    # If the convention is different, maybe the indices are permuted.
    # But the task says "W[k*n+i,r]".
    
    # Let's test with 2x2x2.
    # n=2, m=2, p=2.
    # Schoolbook rank 8.
    # Strassen rank 7.
    # If all-ones tensor rank is 1, then 2x2x2 rank would be 1.
    # But it's 7.
    
    # So the tensor is NOT the all-ones tensor.
    
    # What is the tensor?
    # The multiplication tensor for C = AB is defined by:
    # T_{i,j,k} = 1 if the product A_{i,j} B_{j,k} contributes to C_{i,k}.
    # It contributes if the indices match.
    # A_{i,j} B_{j,k} -> C_{i,k}.
    # So for every i, j, k, there is exactly one term A_{i,j} B_{j,k} that contributes to C_{i,k}.
    # So T_{i,j,k} = 1 for all i, j, k.
    
    # Why does rank 1 fail?
    # Because the decomposition formula I wrote might be wrong for the trace convention.
    
    # Let's look at the standard definition of matrix multiplication tensor.
    # M_{n,m,p} = sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{k,i}
    # where e_{i,j} is the matrix unit.
    # The coefficients are 1.
    
    # The rank is the minimum r such that M = sum_{r=1}^r u_r \otimes v_r \otimes w_r.
    
    # If M = 1 \otimes 1 \otimes 1, then rank is 1.
    # But M_{2,2,2} has rank 7.
    
    # So M is NOT 1 \otimes 1 \otimes 1.
    
    # Ah, the indices!
    # In M = sum e_{i,j} \otimes e_{j,k} \otimes e_{k,i}, the first factor is indexed by (i,j), second by (j,k), third by (k,i).
    # The tensor entry T_{(i,j), (j,k), (k,i)} = 1.
    # All other entries are 0.
    
    # In our case, the factors are vectors U, V, W.
    # U has index i*m+j.
    # V has index j*p+k.
    # W has index k*n+i.
    
    # So the tensor T is sparse?
    # No, for every i, j, k, there is an entry T_{i*m+j, j*p+k, k*n+i} = 1.
    # But these are different positions in the full tensor space of size (nm) x (mp) x (pn).
    # The total number of non-zero entries is n*m*p.
    # The total size of the tensor is (nm)(mp)(pn).
    # So it is very sparse.
    
    # My previous assumption that T_{i,j,k}=1 for all i,j,k in a dense array was wrong.
    # The indices i, j, k determine specific rows in U, V, W.
    
    # So the condition is:
    # For all i in 0..n-1, j in 0..m-1, k in 0..p-1:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    
    # This is a system of n*m*p equations.
    # The unknowns are the entries of U, V, W.
    
    # For n=2, m=3, p=7, there are 42 equations.
    # We want to find a rank r solution.
    
    # If we set U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1 for r=1:
    # Then for a specific i,j,k, the term is 1*1*1 = 1.
    # But wait, do the indices match?
    # For a fixed i,j,k, we are looking at row i*m+j of U, row j*p+k of V, row k*n+i of W.
    # If we set ALL entries of U, V, W to 1, then for ANY i,j,k, the sum is 1.
    # So rank 1 satisfies the equations!
    
    # Why did I think it gives wrong matrix product?
    # Let's re-evaluate C_{i,k} = sum_r W[k*n+i, r] (sum_j U[i*m+j, r] A_{i,j}) (sum_j V[j*p+k, r] B_{j,k})
    
    # If U=1, V=1, W=1:
    # C_{i,k} = 1 * (sum_j A_{i,j}) * (sum_j B_{j,k})
    # This is NOT sum_j A_{i,j} B_{j,k}.
    
    # So the decomposition formula must be different.
    
    # The correct formula for tensor decomposition of multiplication is:
    # C_{i,k} = sum_r (sum_j U_{i,j,r} A_{i,j}) (sum_j V_{j,k,r} B_{j,k}) ... NO.
    
    # The standard form is:
    # C_{i,k} = sum_r W_{i,k,r} * (sum_j U_{i,j,r} A_{i,j}) * (sum_j V_{j,k,r} B_{j,k})
    
    # Let's check indices.
    # U is indexed by (i,j).
    # V is indexed by (j,k).
    # W is indexed by (i,k).
    
    # In the problem, W is indexed by k*n+i.
    # So W_{k*n+i, r} corresponds to W_{i,k,r} if we map k*n+i to (i,k).
    # Note that k*n+i is the index in a (p*n) vector.
    # If we reshape W to (p, n), then W[k, i].
    # So W_{k*n+i, r} is W[k, i, r].
    
    # So the formula is:
    # C_{i,k} = sum_r W[k, i, r] * (sum_j U[i*m+j, r] A_{i,j}) * (sum_j V[j*p+k, r] B_{j,k})
    
    # Let's test with U=1, V=1, W=1.
    # C_{i,k} = 1 * (sum_j A_{i,j}) * (sum_j B_{j,k})
    # Still wrong.
    
    # What is the correct tensor?
    # The multiplication tensor is sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{i,k}.
    # So the third factor is indexed by (i,k).
    # In our case, the third factor is W[k*n+i, r].
    # If we map k*n+i to (i,k), then it matches.
    
    # So the condition is:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k.
    
    # And the product is:
    # C_{i,k} = sum_r W[k*n+i, r] (sum_j U[i*m+j, r] A_{i,j}) (sum_j V[j*p+k, r] B_{j,k})
    
    # If this condition holds, does it imply C=AB?
    # Let's expand the RHS:
    # sum_r W_{k,i,r} sum_j U_{i,j,r} A_{i,j} sum_j' V_{j',k,r} B_{j',k}
    # = sum_{j,j'} (sum_r U_{i,j,r} V_{j',k,r} W_{k,i,r}) A_{i,j} B_{j',k}
    
    # We want this to be sum_j A_{i,j} B_{j,k}.
    # So we need sum_r U_{i,j,r} V_{j',k,r} W_{k,i,r} = delta_{j,j'}.
    
    # But the tensor condition was sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1.
    # This is NOT delta_{j,j'}.
    
    # So the tensor condition I was using is WRONG.
    
    # The correct condition for the decomposition to compute C=AB is:
    # sum_r U[i*m+j, r] V[j'*p+k, r] W[k*n+i, r] = delta_{j,j'}
    
    # This is a system of n * m * m * p equations?
    # No, for each i, k, and pair (j, j'), the sum must be delta_{j,j'}.
    
    # So for fixed i, k:
    # sum_r U[i*m+j, r] V[j'*p+k, r] W[k*n+i, r] = delta_{j,j'}
    
    # This means that for each i, k, the matrices U_{i,:} and V_{:,k} (reshaped) with weights W_{k,i} form an identity.
    
    # Specifically, let U_i be the vector of U[i*m+j, r] for j=0..m-1.
    # Let V_k be the vector of V[j*p+k, r] for j=0..m-1.
    # Then sum_r W[k*n+i, r] U_i[j, r] V_k[j', r] = delta_{j,j'}.
    
    # This is equivalent to:
    # sum_r W[k*n+i, r] (U_i \otimes V_k)_r = I_m
    
    # Where U_i \otimes V_k is the outer product? No, it's a sum of outer products weighted by W.
    # Let M_{i,k} = sum_r W[k*n+i, r] U_i[:, r] V_k[:, r]^T
    # Then M_{i,k} = I_m.
    
    # So for every i, k, we need to form the identity matrix I_m using a sum of r rank-1 matrices, where the coefficients are W[k*n+i, r].
    
    # This is a much stronger condition.
    
    # For schoolbook:
    # r = j (for U), r = j (for V), r = k (for W)?
    # No, schoolbook uses r = (i,j,k) triple.
    # U[i*m+j, r] = 1 if r corresponds to (i,j).
    # V[j*p+k, r] = 1 if r corresponds to (j,k).
    # W[k*n+i, r] = 1 if r corresponds to (i,k).
    
    # Then sum_r U V W = 1 if r=(i,j) and r=(j,k) and r=(i,k).
    # This happens only if the indices match.
    # For fixed i,k, sum_j delta_{j,j'} = delta_{j,j'}.
    
    # So the condition is indeed sum_r U[i*m+j, r] V[j'*p+k, r] W[k*n+i, r] = delta_{j,j'}.
    
    # We need to find a rank r decomposition satisfying this.
    
    # This is equivalent to finding a set of r matrices U_i, V_k, W_{i,k} such that
    # sum_r W_{i,k,r} U_i[:, r] V_k[:, r]^T = I_m for all i, k.
    
    # We can search for this.
    
    # For m=3, we need to form I_3.
    
    # Let's implement a search for this condition.
    
    # We'll use the same random restart approach, but with the correct objective function.
    # Objective: sum_{i,k,j,j'} (sum_r U[i*m+j, r] V[j'*p+k, r] W[k*n+i, r] - delta_{j,j'})^2
    
    # This has n * p * m * m = 2 * 7 * 3 * 3 = 126 terms.
    
    # We'll try to find a rank r solution.
    
    # If we find one, we round to nearest 0.5 and verify.
    
    # Let's code this.

    def check_solution(U, V, W, r):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        err = 0.0
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for j2 in range(m):
                        s = 0.0
                        for r_idx in range(r):
                            s += U[i*m+j, r_idx] * V[j2*p+k, r_idx] * W[k*n+i, r_idx]
                        target = 1.0 if j == j2 else 0.0
                        err += (s - target) ** 2
        return err

    # Search
    for r in range(1, schoolbook_rank):
        if time.time() > deadline:
            break
            
        # Try a few restarts
        for restart in range(5):
            if time.time() > deadline:
                break
                
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Local search
            best_err = check_solution(U, V, W, r)
            for step in range(50):
                if time.time() > deadline:
                    break
                # Pick random i, k, j, j2
                i = np.random.randint(n)
                k = np.random.randint(p)
                j = np.random.randint(m)
                j2 = np.random.randint(m)
                
                # Compute gradient-like update
                # This is complex, so we'll just do a random perturbation.
                U_new = U.copy()
                V_new = V.copy()
                W_new = W.copy()
                
                # Perturb
                U_new[i*m+j] += np.random.randn(r) * 0.1
                V_new[j2*p+k] += np.random.randn(r) * 0.1
                W_new[k*n+i] += np.random.randn(r) * 0.1
                
                err = check_solution(U_new, V_new, W_new, r)
                if err < best_err:
                    U, V, W = U_new, V_new, W_new
                    best_err = err
            
            if best_err < 1e-6:
                # Round to nearest 0.5
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                
                if check_solution(U_round, V_round, W_round, r) < 1e-9:
                    return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
    
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
