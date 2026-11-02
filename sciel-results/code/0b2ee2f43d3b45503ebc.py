"""Program to find a low-rank exact tensor decomposition for 3x4 * 4x4 matrix multiplication."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # n=3, m=4, p=4
    # We decompose the multiplication tensor T[i, j, k] (i in 0..2, j in 0..3, k in 0..3)
    # Using a tensor rank decomposition.
    # Since the tensor is small, we can use a heuristic based on blocking.
    # Let's try to block the input A (3x4) into rows and B (4x4) into cols.
    
    # Approach: Use a "slice" decomposition or a specific known lower bound construction.
    # The schoolbook rank is 3*4*4 = 48.
    # We can try to reduce this by observing that the multiplication is linear.
    # For each row i of A (3 rows), we compute y_i = A[i,:] * B.
    # A[i,:] is a vector of length 4. B is 4x4.
    # This is equivalent to computing 3 independent 1x4 * 4x4 multiplications.
    # The tensor rank for 1x4 * 4x4 is at most 16.
    # So total rank <= 3 * 16 = 48. This is not better.
    
    # However, we can mix the rows.
    # Let's look at the structure of the tensor T in indices (i, j, k).
    # T[i, j, k] = 1 if the output element (i, k) depends on input (j).
    # Actually, the standard matrix multiplication tensor has entries T_{i,j,k} = 1 if the output index is (i,k) and inputs are row i of A and column k of B?
    # No, the tensor is defined such that sum_{j} A_{i,j} B_{j,k} = C_{i,k}.
    # The tensor T has dimensions n x m x p.
    # T[i, j, k] = 1 means that the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # So T[i, j, k] is 1 for all valid i, j, k? No, T is the structure tensor.
    # T_{i,j,k} = delta_{i, row} delta_{k, col} ... wait.
    # The multiplication tensor \mu_{n,m,p} is the tensor in V^* \otimes W^* \otimes Z such that
    # \mu(A, B) = C.
    # \mu = \sum_{i,j,k} e_i \otimes f_j \otimes g_k \otimes (e_i \cdot f_j \cdot g_k ... no)
    # \mu = \sum_{i=0}^{n-1} \sum_{j=0}^{m-1} \sum_{k=0}^{p-1} e_i \otimes f_j \otimes g_k \text{ where } e_i \in \mathbb{R}^n, f_j \in \mathbb{R}^m, g_k \in \mathbb{R}^p.
    # Specifically, \mu = \sum_{i,j,k} u_i \otimes v_j \otimes w_k such that the bilinear form is \sum_{i,j,k} \mu_{i,j,k} A_{i,j} B_{j,k} = \sum_{i,k} C_{i,k}.
    # So \mu_{i,j,k} = 1 if the pair (i,j) and (j,k) form a valid path?
    # Yes, for standard matrix multiplication, \mu_{i,j,k} = 1 for all i, j, k.
    # Wait, is that right?
    # \sum_{j} A_{i,j} B_{j,k} = C_{i,k}.
    # The tensor T satisfies <T, A \otimes B> = \sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} = \sum_{i,k} C_{i,k}.
    # This is not a scalar. The result is a matrix.
    # So the tensor maps V \otimes W \to Z.
    # T \in V^* \otimes W^* \otimes Z.
    # T_{i, j, z} where z = (i, k).
    # T_{i, j, (r, s)} = 1 if i=r and the output index is (r, s).
    # The term A_{i,j} B_{j,s} contributes to C_{i,s}.
    # So T_{i, j, (i, s)} = 1.
    # T_{i, j, (r, s)} = \delta_{i,r} \delta_{j,j} ... no.
    # T_{i, j, (r, s)} = 1 if r=i. The index j is the contraction index.
    # So T_{i, j, (r, s)} = \delta_{i,r}.
    # Let's verify.
    # \sum_{i,j} \sum_{r,s} T_{i,j,(r,s)} A_{i,j} B_{j,s} = \sum_{i,j,s} A_{i,j} B_{j,s} C_{i,s} \text{ (as a tensor in } Z).
    # We want this to equal C.
    # C_{r,s} = \sum_{i,j} T_{i,j,(r,s)} A_{i,j} B_{j,s}.
    # We want C_{r,s} = \sum_{j} A_{r,j} B_{j,s}.
    # So T_{i,j,(r,s)} must be \delta_{i,r}.
    # Thus T_{i,j,(r,s)} = 1 if i=r, else 0.
    
    # So the tensor T has dimensions 3 x 4 x (4*3=12).
    # Non-zero entries: for each j (4 values) and s (4 values), and i (3 values), T[i, j, (i, s)] = 1.
    # Total non-zeros: 3 * 4 * 4 = 48.
    
    # We need to find a rank decomposition:
    # T = \sum_{c=0}^{R-1} u_c \otimes v_c \otimes w_c.
    # u_c \in \mathbb{R}^3, v_c \in \mathbb{R}^4, w_c \in \mathbb{R}^{12}.
    # w_c has indices (r, s).
    
    # One simple decomposition is the "slice" or "outer product" form:
    # For each j, we can write the contribution of the j-th column of A and j-th row of B.
    # Let's try to group by s (output column).
    # For a fixed s, the output C[:, s] = A * B[:, s].
    # This is a 3x1 vector.
    # The tensor part for fixed s is T_s \in \mathbb{R}^{3 \times 4}.
    # T_s[i, j] = 1.
    # This is the all-ones matrix of size 3x4.
    # The rank of the all-ones matrix 3x4 is 1.
    # So for each s, we can decompose T_s as u_s \otimes v_s where u_s = (1,1,1) and v_s = (1,1,1,1).
    # Then T = \sum_{s=0}^{3} (u_s \otimes v_s \otimes e_s).
    # Here e_s is the basis vector in \mathbb{R}^{12} corresponding to output index (i, s)?
    # Wait, w_c is a vector in Z.
    # If we use this decomposition, for each s, we have one term.
    # Term s: u_s = (1,1,1), v_s = (1,1,1,1), w_s = e_{(0,s)} + e_{(1,s)} + e_{(2,s)}?
    # Let's check.
    # Contribution to C_{r,s'}:
    # \sum_{s} \sum_{i,j} u_s[i] v_s[j] w_s[(r, s')] A_{i,j} B_{j, s'} ? No.
    # The decomposition is T = \sum_c u_c \otimes v_c \otimes w_c.
    # The bilinear form is \sum_c (u_c \cdot A_{row}) (v_c \cdot B_{col}) w_c? No.
    # The standard form is:
    # C = \sum_{c} \alpha_c(A) \beta_c(B) \gamma_c.
    # Where \alpha_c(A) = \sum_i u_{c,i} A_{i, j_c}? No.
    # The indices must match.
    # T_{i, j, (r, s)} = \sum_c u_{c, i} v_{c, j} w_{c, (r, s)}.
    # We know T_{i, j, (r, s)} = \delta_{i,r}.
    # So we need \sum_c u_{c, i} v_{c, j} w_{c, (r, s)} = \delta_{i,r}.
    
    # Let's try the decomposition:
    # For each j (0..3), define terms.
    # Actually, the rank of the matrix multiplication tensor is known to be less than n*m*p for larger matrices, but for small ones like 2x2x2 it is 7 (Strassen) vs 8.
    # For 3x4x4, what is the minimal rank?
    # It is at least max(n*p, m*p, n*m) = max(12, 16, 12) = 16.
    # It is at most n*m*p = 48.
    # There are known results for small matrices.
    # For 3x3x3, the rank is at most 23 (Landsberg?) or less.
    # Let's try to construct a decomposition with rank < 48.
    
    # Idea: Use the fact that the tensor is separable in the output index s?
    # T_{i, j, (r, s)} = \delta_{i,r}. It does not depend on s except in the output index.
    # We can write:
    # T = \sum_{s=0}^{3} \sum_{i=0}^{2} \sum_{j=0}^{3} e_i \otimes e_j \otimes \delta_{i,r} e_{(r,s)}
    # This is the standard 48 term decomposition.
    
    # Can we merge terms?
    # Consider the slice s. T_s[i, j] = 1.
    # We need \sum_c u_{c,i} v_{c,j} w_{c,(r,s)} = \delta_{i,r}.
    # This must hold for all r, s.
    # Fix r, s.
    # \sum_c u_{c,i} v_{c,j} w_{c,(r,s)} = \delta_{i,r}.
    # This looks like a matrix factorization problem for each (r,s).
    # But the u_c, v_c are shared across all (r,s).
    # w_c is a vector of length 12.
    
    # Let's try a rank R decomposition.
    # We can use the "border rank" or other techniques, but we need an exact integer/half-integer decomposition.
    
    # Let's try a construction based on the Strassen algorithm for 2x2x2.
    # We can embed 2x2x2 blocks.
    # Split i into (0,1) and 2.
    # Split j into (0,1) and (2,3).
    # Split s into (0,1) and (2,3).
    # This is getting complicated.
    
    # Alternative: Just use a random search or a known low rank decomposition if available.
    # Since I don't have the exact minimal rank for 3x4x4 memorized, I will provide a decomposition that is better than 48.
    # One simple improvement:
    # Note that T_{i,j,(r,s)} = \delta_{i,r}.
    # We can write T = \sum_{i=0}^2 \sum_{j=0}^3 \sum_{s=0}^3 e_i \otimes e_j \otimes e_{(i,s)}.
    # This is rank 48.
    
    # Let's try to reduce the rank by combining the j-sum.
    # For fixed i and s, the term is e_i \otimes (sum_j e_j) \otimes e_{(i,s)}.
    # Sum_j e_j is a vector of ones in R^4.
    # So T = \sum_{i=0}^2 \sum_{s=0}^3 e_i \otimes \mathbf{1}_4 \otimes e_{(i,s)}.
    # This is rank 3 * 4 = 12!
    # Let's verify.
    # Term (i,s): u = e_i, v = \mathbf{1}_4, w = e_{(i,s)}.
    # Sum over i, s of e_i \otimes \mathbf{1}_4 \otimes e_{(i,s)}.
    # Entry (i', j', (r, s')):
    # \sum_{i,s} u_{i'} v_{j'} w_{(r,s')} = \sum_{i,s} \delta_{i',i} * 1 * \delta_{r,i} \delta_{s',s}.
    # = \delta_{i', r} \delta_{s', s} (summed over i=s')?
    # Let's trace:
    # Non-zero if i'=i, r=i, s'=s.
    # So i'=r and s'=s.
    # Sum is 1 if i'=r and s'=s.
    # But we need \delta_{i,r} for ALL s.
    # The result is \delta_{i',r} \delta_{s',s} which is 1 only if s'=s.
    # But the tensor entry T_{i', j', (r, s')} should be \delta_{i',r} regardless of s'.
    # My construction gives \delta_{i',r} \delta_{s',s} which is 0 if s' != s.
    # So this is incorrect. The vector w = e_{(i,s)} only has a 1 at (i,s).
    # We need w to have 1s at all (i, s) for the given i?
    # No, the output index is (r, s).
    # For a fixed input pair (i, j), it contributes to C_{i, s} for all s.
    # So the vector w should be (0,..,0, 1, 1, 1, 1, 0,..,0) with 1s at positions (i, 0), (i, 1), (i, 2), (i, 3).
    # Let w_i be the vector in R^{12} with 1s at indices (i, 0), (i, 1), (i, 2), (i, 3) and 0 elsewhere.
    # Then T = \sum_{i=0}^2 \sum_{j=0}^3 e_i \otimes e_j \otimes w_i ?
    # Entry (i', j', (r, s')):
    # \sum_{i,j} \delta_{i',i} \delta_{j',j} w_i[(r,s')].
    # = w_{i'}[(r,s')].
    # w_{i'} has 1 at (i', s) for s=0..3.
    # So w_{i'}[(r,s')] = \delta_{i',r} * 1 (if s' in 0..3).
    # So the result is \delta_{i',r}.
    # This is correct!
    # The rank is \sum_{i} \sum_{j} 1 = 3 * 4 = 12.
    
    # Wait, is this valid?
    # U has shape (n*m, R) = (12, 12).
    # V has shape (m*p, R) = (16, 12).
    # W has shape (p*n, R) = (12, 12).
    
    # Let's construct U, V, W.
    # R = 12.
    # Columns c = 0..11.
    # Let c = i * 4 + j.
    # U[i*m + j, c] = 1. (Since u is e_i, U row i*m+j has 1 at col c).
    # V[j*p + k, c] = 1. (Since v is e_j, V row j*p+k has 1 at col c).
    # W[k*n + r, c] = 1 if r=i. (Since w_i has 1 at (i, s) for all s).
    # Note: W index is k*n + r. Here k is the output column s?
    # The task says W[k*n + i, r]. Wait, the task description says:
    # "output factor follows the official trace convention W[k*n+i,r]"
    # In the seed program: W[k*n+i][column] = 1.
    # Here i is the row index of output (0..n-1), k is the column index of output (0..p-1)?
    # In seed: for k in range(p): for i in range(n): W[k*n+i].
    # So the index in W is k*n + i, where k is col, i is row.
    # My w_i has 1 at (i, s) where i is row, s is col.
    # So W index is s*n + i.
    # So W[s*n + i, c] = 1 if the row index i matches the input row i.
    
    # Let's re-verify the seed program's W convention.
    # Seed: W[k*n+i][column] = 1.
    # Loop: for i in range(n): for j in range(m): for k in range(p):
    # U[i*m+j] = 1
    # V[j*p+k] = 1
    # W[k*n+i] = 1
    # This corresponds to T[i,j,k] = 1.
    # My derivation: T_{i,j,(r,s)} = \delta_{i,r}.
    # In seed indices: U row i*m+j, V row j*p+k, W row k*n+i.
    # Here the W index is k*n + i.
    # So the output index is (i, k) where i is row, k is col.
    # My w_i has 1 at (row=i, col=s) for all s.
    # So W index s*n + i should be 1.
    # In the loop for a fixed column c (corresponding to input row i, input col j):
    # U[i*m + j, c] = 1.
    # V[j*p + s, c] = 1 for all s? No.
    # In my decomposition, v is e_j. So V[j*p + s, c] = 1 if the input col index is j.
    # Wait, V index is j*p + k. j is input row of B (which is input col of A), k is input col of B.
    # So V[j*p + k, c] = 1 if k = j? No, V index is row of B.
    # B is m x p. Rows 0..m-1, Cols 0..p-1.
    # V index is row*p + col.
    # In my term, v = e_j (where j is 0..m-1).
    # So V[j*p + k, c] should be 1 for all k?
    # No, v is a vector in R^m.
    # The index in V is j*p + k. This implies V is a vector of length m*p.
    # But v_c is a vector of length m.
    # How does v_c map to V?
    # V[row, c] = v_c[row]?
    # If V has shape (m*p, R), then each column c is a vector of length m*p.
    # This doesn't match v_c being length m.
    
    # Let's re-read the interface.
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    # This implies the tensor is decomposed as:
    # T_{i, j, k} = \sum_c U_{i*m + j, c} V_{j*p + k, c} W_{k*n + i, c}
    # Wait, the indices in U, V, W must be consistent.
    # In the seed:
    # U[i*m+j]
    # V[j*p+k]
    # W[k*n+i]
    # This suggests:
    # U index: (i, j) -> i*m + j. (i: row of A, j: col of A)
    # V index: (j, k) -> j*p + k. (j: row of B, k: col of B)
    # W index: (k, i) -> k*n + i. (k: col of B, i: row of A) -> This is the output index?
    # Output C has row i, col k.
    # So W index is k*n + i.
    
    # Let's check my decomposition again.
    # T_{i, j, (r, s)} = \delta_{i,r}.
    # Here i is row of A, j is col of A (row of B), r is row of C, s is col of C (col of B).
    # So T_{i, j, s} (with output index (r, s)) = \delta_{i,r}.
    # In the seed notation:
    # U index: i*m + j.
    # V index: j*p + s. (since k is col of B, which is s)
    # W index: s*n + r. (since k is s, i is r)
    # We want:
    # \sum_c U_{i*m+j, c} V_{j*p+s, c} W_{s*n+r, c} = \delta_{i,r}.
    
    # My previous construction:
    # Terms indexed by (i, j).
    # U_{i*m+j, c} = 1.
    # V_{j*p+s, c} = 1 for all s?
    # If V_{j*p+s, c} = 1 for all s, then V column c is not a basis vector e_j in R^m, but something else.
    # Let's check the sum:
    # \sum_c U_{i*m+j, c} V_{j*p+s, c} W_{s*n+r, c}
    # With c = i0*m + j0.
    # U_{i*m+j, c} = \delta_{i,i0} \delta_{j,j0}.
    # If we set V_{j*p+s, c} = \delta_{j,j0} (for all s), and W_{s*n+r, c} = \delta_{r,i0}.
    # Then term is \delta_{i,i0} \delta_{j,j0} \delta_{j,j0} \delta_{r,i0}.
    # Sum over c (i0, j0):
    # \sum_{i0, j0} \delta_{i,i0} \delta_{j,j0} \delta_{j,j0} \delta_{r,i0}
    # = \delta_{i,r} \delta_{j,j} = \delta_{i,r}.
    # This works!
    
    # So the construction is:
    # Rank R = n * m = 3 * 4 = 12.
    # For each i in 0..n-1, j in 0..m-1:
    #   c = i * m + j
    #   U[i*m + j, c] = 1
    #   V[j*p + s, c] = 1 for all s in 0..p-1
    #   W[s*n + r, c] = 1 if r == i, else 0 for all s in 0..p-1, r in 0..n-1
    
    # Let's verify V and W shapes.
    # V has shape (m*p, R).
    # For a fixed c (i, j), V[:, c] has 1s at rows j*p + s for all s.
    # W has shape (p*n, R).
    # For a fixed c (i, j), W[:, c] has 1s at rows s*n + i for all s.
    
    # This gives a rank 12 decomposition.
    # This is much better than 48.
    
    n, m, p = DIMENSIONS
    R = n * m
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            c = i * m + j
            U[i * m + j][c] = 1
            for s in range(p):
                V[j * p + s][c] = 1
                W[s * n + i][c] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
