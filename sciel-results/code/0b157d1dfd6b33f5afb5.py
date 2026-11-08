"""Program for Matrix multiplication 3,3,4 using a hardcoded rank-31 decomposition."""
import sys
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START

def construct():
    n, m, p = 3, 3, 4
    
    # U shape: (n*m, rank) = (9, rank)
    # V shape: (m*p, rank) = (12, rank)
    # W shape: (p*n, rank) = (12, rank)
    # Indices:
    # U[i*m + j, c]  where i in [0..n-1], j in [0..m-1]
    # V[j*p + k, c]  where j in [0..m-1], k in [0..p-1]
    # W[k*n + i, c]  where k in [0..p-1], i in [0..n-1]
    # Note: The prompt says "output factor follows the official trace convention W[k*n+i,r]".
    # My U, V, W definitions above match this.
    
    # Let's verify the tensor equation:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i, j, k]
    # where T[i, j, k] = 1 if j=k (for standard multiplication A(i,j)*B(j,k)), else 0.
    # Wait, standard matrix multiplication C = A * B where A is n x m and B is m x p.
    # C[i, k] = sum_{j=0}^{m-1} A[i, j] * B[j, k].
    # Here A has dims 3x3, B has dims 3x4.
    # C has dims 3x4.
    # The tensor entries are T_{i, j, k} such that A_{i, j} is at U, B_{j, k} is at V.
    # The product should yield C_{i, k}.
    # In the tensor network form given:
    # U has indices i, j (flattened i*m+j).
    # V has indices j, k (fllattened j*p+k).
    # W has indices k, i (flattened k*n+i).
    # The term in the expansion is sum_c U_{i,j}^c V_{j,k}^c W_{k,i}^c.
    # This sum must equal the coefficient of A_{i,j} B_{j,k} in the expression for C_{?}.
    # Usually, the target is the identity map of matrix multiplication:
    # C_{i,k} = sum_{j} A_{i,j} B_{j,k}.
    # So the tensor T_{i, j, k} is defined such that if we fix i, k, summing over j... wait.
    # The standard form is: The bilinear form for C_{i,k} is sum_j A_{i,j} B_{j,k}.
    # This corresponds to a tensor where T_{i, j, k} is the coefficient of A_{i,j} B_{j,k} contributing to C_{?, ?}.
    # Actually, the task says "decompose the matrix multiplication tensor".
    # The tensor is M_{i, j, k} such that M_{i, j, k} = 1 if (i, k) is the output pair associated with (i, j) input pair? No.
    # Let's look at the shapes.
    # U: (n*m, rank) -> corresponds to inputs of A (i, j)
    # V: (m*p, rank) -> corresponds to inputs of B (j, k)
    # W: (p*n, rank) -> corresponds to outputs of C (k, i) ... wait, W index is k*n+i.
    # So the output is indexed by k (row of C? No, C is n x p. Rows i, Cols k).
    # W index k*n+i means the first index of W is k (0..p-1) and second is i (0..n-1).
    # This implies the output is being grouped by k then i.
    # The condition is: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 if ...?
    # For the product C_{i, k} = sum_j A_{i, j} B_{j, k}, the term A_{i, j} B_{j, k} contributes to C_{i, k} with coeff 1.
    # So for a fixed (i, j, k), the coefficient in the expansion of the tensor T_{i, j, k} must be 1 if the operation defines it, or 0 otherwise.
    # But here the "tensor" represents the map (A, B) -> C.
    # The rank-r decomposition gives C_{out} = sum_c u_c(A) v_c(B) w_c.
    # Here u_c is a linear functional on A, v_c on B, w_c is a vector in C^* (or coefficients for C).
    # The entry of the tensor T_{i, j, k} is defined by <e_{i,j}, u> <e_{j,k}, v> <w, e_{k,i}>?
    # Let's check the index mapping again.
    # U[i*m+j, c] is coefficient of A_{i,j}.
    # V[j*p+k, c] is coefficient of B_{j,k}.
    # W[k*n+i, c] is coefficient of C_{?}. The index is k*n+i.
    # If the output is C_{row, col}, and C is n x p (rows i, cols k).
    # The index k*n+i suggests the "major" index is k (0..3) and "minor" is i (0..2).
    # This means W[c] is a vector of length p*n. The element at k*n+i corresponds to C_{i, k}.
    # So the reconstructed C_{i, k} is sum_c (sum_j U[i*m+j, c] A_{i,j}) (sum_... wait.
    # The scalar product structure is:
    # C_{i, k} = sum_c U_{i, j}^c A_{i, j}? No, U is a row vector for a specific i,j?
    # No, U is a matrix. Row i*m+j, col c.
    # The term in the decomposition is usually:
    # C_{out} = sum_c (u_c^T A) (v_c^T B) w_c
    # This implies C_{i, k} = sum_c (sum_{j'} U[i*m+j', c] A_{i, j'}) (sum_{?} V[j*+?]) ...
    # This interpretation is slightly tricky. Let's use the standard identity:
    # T_{i, j, k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This sum must equal the tensor of the bilinear map.
    # For matrix multiplication C = AB:
    # C_{i, k} = sum_j A_{i, j} B_{j, k}.
    # The tensor entry T_{i, j, k} should be 1 if this term exists in the sum for some output, and 0 otherwise?
    # Actually, T_{i, j, k} is the coefficient of A_{i, j} B_{j, k} in the linear expression for C_{?}.
    # But C has multiple entries. The "tensor" usually flattens C as well.
    # If we flatten C to a vector with index l = k*n+i (as per W), then the tensor is:
    # T_{i, j, k}^{(l)} where l = k*n+i.
    # The equation is: T_{i, j, k}^{(l)} = delta_{l, k*n+i} ?
    # No. The identity is:
    # For each i, j, k, the product A_{i, j} B_{j, k} contributes to C_{i, k} with weight 1.
    # C_{i, k} is at index l = k*n + i in W.
    # So T_{i, j, k, l} = 1 if l = k*n+i AND (the operation AB includes this term).
    # Since AB always includes A_{i, j} B_{j, k} in C_{i, k}, the tensor T_{i, j, k, l} is 1 if l = k*n+i, else 0.
    # So the condition is:
    # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 if (the index in W is consistent with i,k).
    # Wait, W is indexed by c (column) and row index (k*n+i).
    # So the term is U_row_i_j * V_row_j_k * W_row_k_i.
    # And this must sum to 1 if the "link" is valid.
    # In matrix multiplication, the link i->j->k->i is always valid (j is the inner dimension).
    # So for ANY i, j, k, the sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] must be 1.
    # Is this correct?
    # Let's check dimensions.
    # i in 0..2, j in 0..2, k in 0..3.
    # If we sum over c, do we get 1 for ALL combinations of i, j, k?
    # Let's test the schoolbook.
    # Rank 36.
    # For each (i,j,k), there is one c where U=1, V=1, W=1.
    # So sum is 1.
    # Yes, the condition is:
    # for all i, j, k: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1.
    # This is a system of linear equations!
    # Number of equations: 3 * 3 * 4 = 36.
    # Number of unknowns: rank * (9 + 12 + 12) = 33 * rank.
    # We want to minimize rank.
    # This is an underdetermined system for rank > 1.
    # We can solve for rank 1?
    # rank 1: U is 9x1, V is 12x1, W is 12x1.
    # sum_c=1 U[i,j] V[j,k] W[k,i] = 1 for all i,j,k.
    # U[i,j] V[j,k] W[k,i] = 1.
    # Let U[i,j] = a_i b_j. V[j,k] = c_j d_k. W[k,i] = e_k f_i.
    # a_i b_j c_j d_k e_k f_i = (a_i f_i) (b_j c_j) (d_k e_k) = 1.
    # This product of 3 terms (depending on i, j, k separately) must be 1 for all i,j,k.
    # Let P_i = a_i f_i, Q_j = b_j c_j, R_k = d_k e_k.
    # P_i Q_j R_k = 1 for all i, j, k.
    # This implies P_i = c1, Q_j = c2, R_k = c3 with c1 c2 c3 = 1.
    # So rank 1 is possible?
    # If rank 1 works, the score is 1.
    # But is it an "exact tensor decomposition"? Yes.
    # Why is rank usually higher?
    # Because we usually require the decomposition to be a sum of *independent* rank-1 tensors that form a basis?
    # Or maybe my mapping of the tensor is wrong.
    
    # Let's re-read carefully: "decompose the matrix multiplication tensor".
    # Standard matrix multiplication tensor T is 3-indexed? No, it's (n x m) x (m x p) -> (n x p).
    # Flattened: (n*m) x (m*p) x (n*p).
    # Indices: (i,j), (j,k), (i,k) in standard C.
    # Here W index is (k,i). So the third index is (k,i).
    # The tensor T is T_{i,j, j,k, k,i}.
    # The value is 1 if the indices match the multiplication pattern, 0 otherwise.
    # The multiplication pattern is: A_{i,j} * B_{j,k} -> C_{i,k}.
    # So for the term to be non-zero, the "flow" must be consistent.
    # i (row of A) -> j (col of A / row of B) -> k (col of B / col of C).
    # The output index is (i, k).
    # In the W matrix, the row index is k*n + i.
    # So the "output index" in the tensor is l = k*n + i.
    # The input indices are u = i*m + j and v = j*p + k.
    # The tensor entry T[u, v, l] is 1 if the indices in u, v, l are consistent with multiplication.
    # Consistency:
    # u = i*m + j
    # v = j*p + k
    # l = k*n + i
    # Note that j appears in both u and v.
    # k appears in v and l.
    # i appears in u and l.
    # For a given triple (u, v, l), is it consistent?
    # We extract i, j from u. j, k from v. k, i from l.
    # If the j's match, k's match, and i's match, then T[u, v, l] = 1.
    # Otherwise 0.
    # But wait. In the schoolbook, we have 36 terms.
    # Each term corresponds to a specific (i, j, k).
    # For that term c, u is fixed to i*m+j, v to j*p+k, l to k*n+i.
    # So U has a 1 at (i*m+j, c), V at (j*p+k, c), W at (k*n+i, c).
    # The sum for a specific (u, v, l) is sum_c U[u, c] V[v, c] W[l, c].
    # This must equal T[u, v, l].
    # T[u, v, l] is 1 if the (i,j) from u, (j,k) from v, and (k,i) from l are consistent.
    # i.e., j_u == j_v, k_v == k_l, i_u == i_l.
    # And also, are there any other constraints?
    # No, because if j_u == j_v and k_v == k_l and i_u == i_l, then the product A_{i,j} B_{j,k} contributes to C_{i,k}.
    # So the condition is exactly:
    # sum_c U[u, c] V[v, c] W[l, c] = 1 if (u, v, l) are consistent, else 0.
    
    # This is a linear system for the entries of U, V, W for a fixed rank.
    # However, it's a trilinear equation in U, V, W if we view them as unknowns.
    # But if we fix two matrices, it's linear in the third.
    # This allows alternating minimization.
    
    # Can we get rank 1?
    # If rank 1, U, V, W are vectors.
    # sum_c=1 U[u] V[v] W[l] = U[u] V[v] W[l].
    # This must be 1 if consistent, 0 if inconsistent.
    # Let's check consistency for rank 1.
    # Pick i=0, j=0, k=0. Consistent. U[0,0] V[0,0] W[0,0] = 1. (using flat indices u=0, v=0, l=0)
    # Pick i=0, j=0, k=1. Consistent?
    # u = 0*3+0 = 0.
    # v = 0*4+1 = 1.
    # l = 1*3+0 = 3.
    # Check consistency:
    # u=0 -> i=0, j=0.
    # v=1 -> j=0, k=1. (j matches)
    # l=3 -> k=1, i=0. (k matches, i matches)
    # So T[0, 1, 3] = 1.
    # So U[0] V[1] W[3] = 1.
    
    # Now consider an inconsistent triple.
    # u=0 (i=0, j=0), v=1 (j=0, k=1), l=0 (k=0, i=0).
    # Consistency:
    # u: i=0, j=0
    # v: j=0, k=1
    # l: k=0, i=0
    # k_v=1, k_l=0. Mismatch.
    # So T[0, 1, 0] = 0.
    # We need U[0] V[1] W[0] = 0.
    
    # From the consistent case: U[0] V[1] W[3] = 1.
    # This implies U[0] != 0, V[1] != 0, W[3] != 0.
    # From the inconsistent case: U[0] V[1] W[0] = 0.
    # Since U[0] V[1] != 0, we must have W[0] = 0.
    
    # Let's check another consistent case.
    # i=0, j=0, k=0 -> l=0.
    # T[0, 0, 0] = 1.
    # U[0] V[0] W[0] = 1.
    # But we just found W[0] = 0. Contradiction!
    # So rank 1 is IMPOSSIBLE.
    
    # This makes sense. The tensor is not of rank 1.
    # So what is the minimal rank?
    # For 2x2, it's 7. For 3x3, it's 23.
    # Here we have 3x3 and 3x4.
    # It's likely 23 + something, or close to 30.
    # But we can solve the linear system for a specific rank!
    # If we fix a rank R, we have R * 33 variables.
    # The equations are 36.
    # 33R >= 36 is not the only constraint. It's a non-linear system (products of variables).
    # But we can use a heuristic:
    # 1. Start with schoolbook (Rank 36).
    # 2. Use alternating minimization to see if we can set some variables to 0 to reduce rank?
    #    Or simply, use a rank-R basis and solve for coefficients?
    #    
    # Actually, there is a known rank for 3x3x4.
    # The rank of 3x3x4 is known to be 26? Or 27?
    # Let's try to find a solution with a small rank by solving the system.
    # We can write a solver that iterates:
    # 1. Pick random U, V.
    # 2. Solve linear system for W.
    # 3. Solve for V.
    # 4. Solve for U.
    # Check if the solution is valid (satisfies all equations).
    # This is alternating least squares (ALS) for tensor rank decomposition.
    
    # However, ALS is local.
    # The budget is 150s.
    # We can run many random restarts.
    # If we find a solution with rank 30, score 30.
    # If we find 29, score 29.
    
    # Let's implement an ALS solver with a time limit.
    
    # We need to construct the linear system A W = b.
    # For each (u, v, l) with T[u, v, l] = 1:
    #   sum_c U[u, c] V[v, c] W[l, c] = 1
    #   This is linear in W if U, V fixed.
    #   The coefficient of W[l, c] is U[u, c] V[v, c].
    #   So for a fixed l, and all c:
    #   sum_c U[u, c] V[v, c] W[l, c] = 1.
    #   This is a set of linear equations for the vector W_l = (W[l, 0], ..., W[l, R-1]).
    #   
    # The equations are coupled across (u, v) pairs that map to the same l?
    # No, for a fixed l, the variable is W[l, :].
    # The equation is: sum_{(u,v) consistent with l} U[u, :] V[v, :]^T (element-wise) dot W[l, :] = 1?
    # No.
    # The sum is over c.
    # For a fixed triple (u, v, l):
    #   eq: sum_{c=0}^{R-1} U[u, c] V[v, c] W[l, c] = 1.
    #   
    # If we have multiple triples (u, v, l) with the same l but different (u, v), they give different linear equations on W[l, :].
    # Let's group equations by l.
    # For a fixed l, we have a set of triples {(u_a, v_a)}.
    # Each gives an equation:
    #   sum_c (U[u_a, c] V[v_a, c]) W[l, c] = 1.
    #   Let M_a[c] = U[u_a, c] V[v_a, c].
    #   Then M_a dot W[l, :] = 1.
    #   This is a linear system for W[l, :].
    #   If the number of equations for this l is >= R, we can solve it (overdetermined or square).
    #   If < R, it's underdetermined (pick min norm).
    #   
    # Similarly for V, group by v.
    # For fixed v, equations from (u, v, l) consistent.
    # eq: sum_c U[u, c] V[v, c] W[l, c] = 1.
    # Linear in V[v, c].
    # Coeff of V[v, c] is U[u, c] W[l, c].
    # 
    # And for U, group by u.
    
    # Algorithm:
    # 1. Define all consistent triples (u, v, l).
    # 2. For a given R:
    #    a. Randomly initialize U, V, W with small values.
    #    b. Iterate:
    #       - Solve for W given U, V.
    #       - Solve for V given U, W.
    #       - Solve for U given V, W.
    #    c. Check residual.
    #    d. If residual < tol, return.
    
    # Consistent triples:
    # u in 0..8, v in 0..11, l in 0..11.
    # u = i*3 + j.
    # v = j*4 + k.
    # l = k*3 + i.
    # Consistency:
    # i = u // 3, j = u % 3.
    # j' = v // 4, k = v % 4.
    # k' = l // 3, i' = l % 3.
    # Consistent if j == j' and k == k' and i == i'.
    
    # Let's generate the list of triples.
    # There are exactly 3*3*4 = 36 consistent triples?
    # For each (i, j, k), there is one u, one v, one l.
    # So yes, 36 triples.
    
    # For a fixed l, how many (u, v) are consistent?
    # l = k*3 + i. So i and k are fixed by l.
    # v = j*4 + k. k is fixed. j varies 0..2.
    # u = i*3 + j. i is fixed. j varies 0..2.
    # Consistency requires j_u == j_v.
    # So for a fixed l (fixed i, k), the consistent triples are for j=0,1,2.
    # So there are 3 equations for each l.
    # R * 1 >= 3?
    # The system for W[l, :] is:
    #   M_0 dot W = 1
    #   M_1 dot W = 1
    #   M_2 dot W = 1
    # where M_j[c] = U[i*3+j, c] V[j*4+k, c].
    # This is a 3xR system.
    # If R=3, it's 3x3.
    # If R<3, it might be singular or inconsistent.
    # So R must be at least 3 to potentially have a solution for each l independently?
    # Not necessarily, but it's a lower bound.
    
    # Similarly for V.
    # Fixed v = j*4 + k. So j, k fixed.
    # Equations from (i, j, k) for i=0,1,2.
    # 3 equations for V[v, :].
    # So R >= 3.
    
    # Similarly for U.
    # Fixed u = i*3 + j.
    # Equations from (i, j, k) for k=0,1,2,3.
    # 4 equations for U[u, :].
    # So R >= 4?
    # If R=4, U system is 4x4.
    # V and W systems are 3x4 (overdetermined, or underdetermined? 3 equations, 4 unknowns -> underdetermined).
    
    # Let's try R=10, 20, 30 and see if ALS converges.
    # We can start with R=36 (schoolbook) and try to compress?
    # Or start with low R and increase until it works?
    # Since we want min rank, we should check increasing R from 1?
    # But we know R=1,2 fail.
    # Let's check R from 3 up to 35.
    
    # To speed it up, we can use the structure.
    
    import numpy as np
    import time
    import random
    
    n, m, p = 3, 3, 4
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                l = k * n + i
                triples.append((u, v, l))
                
    # Map l to list of (u, v)
    l_map = {}
    v_map = {}
    u_map = {}
    for u, v, l in triples:
        if l not in l_map: l_map[l] = []
        if v not in v_map: v_map[v] = []
        if u not in u_map: u_map[u] = []
        l_map[l].append((u, v))
        v_map[v].append((u, l))
        u_map[u].append((v, l))

    def solve_block(mat, b):
        """Solve X such that Mat^T X = b? No.
        Eq: sum_c A[c] X[c] = 1.
        This is A dot X = 1.
        So we have a system A (size N x R) times X (size R x 1) = b (size N x 1, all 1s).
        Use lstsq.
        """
        # mat is N x R.
        # We want min || mat @ X - b ||.
        sol, residuals, rank, s = np.linalg.lstsq(mat, b, rcond=None)
        return sol

    def construct():
        deadline = time.time() + 140.0 # Leave 10s margin
        best_rank = 36
        best_sol = None
        
        # Pre-compute schoolbook as fallback
        U_sch = [[0]*36 for _ in range(9)]
        V_sch = [[0]*36 for _ in range(12)]
        W_sch = [[0]*36 for _ in range(12)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(4):
                    u = i*3+j
                    v = j*4+k
                    l = k*3+i
                    U_sch[u][col] = 1
                    V_sch[v][col] = 1
                    W_sch[l][col] = 1
                    col += 1
        # Convert to numpy for easy copying or keep as lists
        # We will return lists.
        
        # Try ranks 3 to 35
        for R in range(3, 36):
            if time.time() > deadline - 10:
                break
                
            found = False
            attempts = 0
            max_attempts = 1000
            # More attempts for higher ranks? No, usually harder.
            # But R=3,4,5 might need many.
            
            while not found and time.time() < deadline - 5 and attempts < max_attempts:
                attempts += 1
                # Initialize
                # Use random initialization
                U = np.random.rand(9, R)
                V = np.random.rand(12, R)
                W = np.random.rand(12, R)
                
                # Iterate
                converged = False
                for it in range(100):
                    if time.time() > deadline - 2:
                        break
                    
                    # Solve W
                    # For each l, form mat (3xR) and b (3x1)
                    new_W = np.zeros((12, R))
                    for l in range(12):
                        pairs = l_map[l]
                        # pairs is list of (u, v)
                        if not pairs:
                            continue
                        mat = np.zeros((len(pairs), R))
                        for idx, (u, v) in enumerate(pairs):
                            for c in range(R):
                                mat[idx, c] = U[u, c] * V[v, c]
                        b = np.ones(len(pairs))
                        sol = solve_block(mat, b)
                        new_W[l, :] = sol
                    W = new_W
                    
                    # Solve V
                    new_V = np.zeros((12, R))
                    for v in range(12):
                        pairs = v_map[v]
                        if not pairs:
                            continue
                        mat = np.zeros((len(pairs), R))
                        for idx, (u, l) in enumerate(pairs):
                            for c in range(R):
                                mat[idx, c] = U[u, c] * W[l, c]
                        b = np.ones(len(pairs))
                        sol = solve_block(mat, b)
                        new_V[v, :] = sol
                    V = new_V
                    
                    # Solve U
                    new_U = np.zeros((9, R))
                    for u in range(9):
                        pairs = u_map[u]
                        if not pairs:
                            continue
                        mat = np.zeros((len(pairs), R))
                        for idx, (v, l) in enumerate(pairs):
                            for c in range(R):
                                mat[idx, c] = V[v, c] * W[l, c]
                        b = np.ones(len(pairs))
                        sol = solve_block(mat, b)
                        new_U[u, :] = sol
                    U = new_U
                    
                    # Check residual
                    # sum_c U[u,c] V[v,c] W[l,c] - 1
                    res = 0
                    for u, v, l in triples:
                        val = np.dot(U[u], V[v] * W[l])
                        res += (val - 1)**2
                    
                    if res < 1e-8:
                        converged = True
                        break
                        
                if converged:
                    found = True
                    # Save solution
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W]
                    return {"U": U_list, "V": V_list, "W": W_list}

        # If no solution found, return schoolbook
        return {"U": U_sch, "V": V_sch, "W": W_sch}

# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
