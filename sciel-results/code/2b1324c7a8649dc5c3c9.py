import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # The matrix multiplication tensor T_{ijk} = delta_{ij} * delta_{jk} (in a suitable basis)
    # We want to find a decomposition of rank r < rank_schoolbook.
    # This is a hard problem. For small dimensions, we can use known results or search.
    # However, 120s is a tight constraint for a general search.
    # Let's try to implement a randomized search for a lower rank decomposition.
    # If we find one, great. If not, return schoolbook.
    
    # Strategy:
    # 1. Start with a random rank r candidate.
    # 2. Try to solve for U, V, W such that T = U V^T W^T (element-wise product of outer products).
    #    Specifically, T_{ijk} = sum_c U_{ij,c} V_{jk,c} W_{ik,c} (note indices: W is p*n, so W_{k,i} or W_{i,k}?)
    #    The task says W[k*n+i, r]. So W is a matrix of size (p*n) x rank.
    #    The term is U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    #    
    #    This is a system of polynomial equations in the entries of U, V, W.
    #    We can fix a gauge to reduce symmetry. E.g., fix some entries of U to 1.
    #    
    #    Let's try a simpler approach:
    #    We know that rank(MN) <= min(rank(M), rank(N)) is not directly applicable here as this is a tensor.
    #    
    #    Let's try to use the fact that for 2x3x10, the rank might be reducible.
    #    Actually, 2x3x10 is a specific case.
    #    Let's try to search for rank 59, 58, etc.
    #    
    #    Given the time limit, a random search for a specific rank is unlikely to succeed unless we have a good initialization.
    #    
    #    Let's try to use a known algorithm or a heuristic.
    #    One heuristic is to use the "border rank" or "approximate rank" and then round.
    #    But we need *exact* integer/half-integer entries.
    #    
    #    Let's try to construct a decomposition using Strassen-like ideas if possible, but 2x3x10 is not a power of 2.
    #    
    #    Alternative: Use the fact that the tensor is sparse.
    #    The tensor T has n*m*p = 60 non-zero entries (all 1s).
    #    Wait, T_{ijk} is 1 if i=j and j=k? No.
    #    Matrix multiplication: C_{ik} = sum_j A_{ij} B_{jk}.
    #    The tensor is T_{ijk} = A_{ij} B_{jk} C_{ik}.
    #    The structure is: for each (i,j,k), the coefficient is 1.
    #    So T is the "matrix multiplication tensor" of size 2x3x10.
    #    It has 2*3*10 = 60 entries, all 1s? No.
    #    The tensor is defined on indices i in {0,1}, j in {0,1,2}, k in {0,...,9}.
    #    The entry T_{ijk} is 1 for all i,j,k.
    #    Wait, no. The tensor decomposes the multiplication.
    #    The "multiplication tensor" M_{n,m,p} has entries M_{ijk} = 1 if the indices correspond to a valid multiplication step?
    #    Actually, the tensor is usually defined such that <T, A \otimes B \otimes C> = Tr(AB C^T) or similar.
    #    For standard matrix multiplication C = AB:
    #    C_{ik} = sum_j A_{ij} B_{jk}.
    #    This can be written as sum_{i,j,k} A_{ij} B_{jk} C_{ik}.
    #    So the tensor T has entries T_{ijk} = 1 for all valid i,j,k.
    #    Yes, all 60 entries are 1.
    #    
    #    So we want to decompose the all-ones tensor of size 2x3x10.
    #    Rank of all-ones tensor of size a x b x c is min(a,b,c) if we allow real entries? No, that's for the sum of outer products.
    #    Rank of the all-ones tensor J_{abc} is 1? No.
    #    J_{abc} = u v^T w^T where u is all-ones of size a, v of size b, w of size c.
    #    Wait, (u v^T w^T)_{ijk} = u_i v_j w_k.
    #    If u_i=1, v_j=1, w_k=1, then (u v^T w^T)_{ijk} = 1.
    #    So the rank of the all-ones tensor is 1!
    #    
    #    BUT, the matrix multiplication tensor is NOT the all-ones tensor.
    #    The matrix multiplication tensor T_{m,n,p} is defined by:
    #    T_{ijk} = 1 if i=j and j=k? No.
    #    Let's re-read the standard definition.
    #    The tensor is an element of R^{n*m} \otimes R^{m*p} \otimes R^{p*n}.
    #    The indices are (i,j), (j,k), (k,i).
    #    The entry is 1 if the indices match?
    #    Actually, the tensor is defined by the bilinear map (A,B) -> AB.
    #    T(A,B) = sum_{i,j,k} A_{ij} B_{jk} E_{ik}.
    #    So T = sum_{i,j,k} e_{ij} \otimes e_{jk} \otimes e_{ik}.
    #    Here e_{ij} is the basis vector for A, e_{jk} for B, e_{ik} for C.
    #    So the tensor has 1s at positions (i,j; j,k; k,i) and 0s elsewhere.
    #    Total non-zero entries: n*m*p.
    #    
    #    So it is NOT the all-ones tensor. It is a "permutation-like" tensor.
    #    For n=2, m=3, p=10:
    #    Indices: i in {0,1}, j in {0,1,2}, k in {0..9}.
    #    Non-zero at (i*m+j, j*p+k, k*n+i).
    #    
    #    So the rank is at most n*m*p = 60.
    #    Can it be lower?
    #    
    #    This is a known open problem for many sizes, but for small sizes, exact ranks are known.
    #    For 2x3x10, I don't know the exact rank off the top of my head.
    #    However, 60 is the schoolbook rank.
    #    
    #    Let's try to search for a rank < 60 using a random linear algebra approach.
    #    We can fix a random U, V and try to solve for W, or vice versa.
    #    
    #    Let's try to use the "subspace" method.
    #    If the rank is r, then the tensor lies in the span of r rank-1 tensors.
    #    
    #    Given the constraints, I will implement a quick random search for rank 59.
    #    If it fails, I return 60.
    
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    def try_rank(r):
        if r >= 60:
            return None
        
        # Try to find U, V, W of rank r
        # We can fix U and V to random values and solve for W?
        # T_{ijk} = sum_c U_{ij,c} V_{jk,c} W_{ki,c}
        # This is linear in W if U and V are fixed.
        # For each (i,k), we have:
        # T_{ik} (which is 1 for all i,k? No, T is 3D)
        # Wait, the tensor is T_{(i,j),(j,k),(k,i)}.
        # Let's flatten indices.
        # A_idx = i*m + j
        # B_idx = j*p + k
        # C_idx = k*n + i
        # T[A_idx, B_idx, C_idx] = 1 if indices match, else 0.
        # Actually, for a given c, the term is U[A_idx, c] * V[B_idx, c] * W[C_idx, c].
        # We want sum_c U[A_idx, c] * V[B_idx, c] * W[C_idx, c] = T[A_idx, B_idx, C_idx].
        
        # If we fix U and V, the equation for W[C_idx, c] is:
        # sum_c W[C_idx, c] * (sum over A_idx, B_idx such that T[A_idx, B_idx, C_idx] != 0? No.)
        # The equation is:
        # For all (A_idx, B_idx, C_idx):
        # sum_c U[A_idx, c] V[B_idx, c] W[C_idx, c] = T[A_idx, B_idx, C_idx]
        
        # This is a system of equations in W.
        # Let M_c be the matrix M_c[A_idx, B_idx] = U[A_idx, c] V[B_idx, c].
        # Then sum_c W[C_idx, c] * (M_c[A_idx, B_idx]) = T[A_idx, B_idx, C_idx].
        # This doesn't look linear in W because W[C_idx, c] multiplies a term that depends on c.
        # Actually, for a fixed C_idx, the equation is:
        # sum_c W[C_idx, c] * U[A_idx, c] V[B_idx, c] = T[A_idx, B_idx, C_idx]
        # This is linear in W[C_idx, :].
        # Let x_c = W[C_idx, c].
        # sum_c x_c * (U[A_idx, c] V[B_idx, c]) = T[A_idx, B_idx, C_idx]
        # Let y_c(A_idx, B_idx) = U[A_idx, c] V[B_idx, c].
        # We need to find x such that sum_c x_c y_c(A_idx, B_idx) = T[A_idx, B_idx, C_idx].
        # This is a linear system: Y * x = T_C
        # Where Y is a matrix of size (n*m * m*p) x r? No.
        # The indices (A_idx, B_idx) must be such that T[A_idx, B_idx, C_idx] is defined.
        # For a fixed C_idx = k*n + i, the non-zero T entries are those where (A_idx, B_idx) = (i*m+j, j*p+k) for some j.
        # There are m such pairs for each (i,k).
        # So for each C_idx, we have m equations.
        # Total equations: n * p * m = 60.
        # Unknowns: r (for each C_idx, we have r unknowns W[C_idx, c]).
        # But W has p*n rows, each of length r.
        # So total unknowns in W: p*n*r.
        # The system is decoupled for each C_idx.
        # For each C_idx, we have m equations and r unknowns.
        # If r > m, the system is underdetermined. We can find a solution if the RHS is in the column space.
        
        # So, algorithm:
        # 1. Generate random U (n*m x r) and V (m*p x r).
        # 2. For each C_idx, solve the linear system Y_C * x = T_C for x = W[C_idx, :].
        #    If a solution exists, set W[C_idx, :] = x.
        # 3. Check if the resulting decomposition is exact.
        
        # Note: U and V must be such that the system is solvable.
        # This is a random search.
        
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        
        W = np.zeros((p * n, r))
        
        for k in range(p):
            for i in range(n):
                c_idx = k * n + i
                # Equations for this c_idx:
                # For each j in range(m):
                # A_idx = i * m + j
                # B_idx = j * p + k
                # T[A_idx, B_idx, c_idx] = 1
                # Equation: sum_c U[A_idx, c] * V[B_idx, c] * W[c_idx, c] = 1
                
                # Build matrix Y of size m x r
                Y = np.zeros((m, r))
                for j in range(m):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    Y[j, :] = U[a_idx, :] * V[b_idx, :]
                
                # RHS is vector of 1s of size m
                rhs = np.ones(m)
                
                # Solve Y * x = rhs
                # Use least squares
                x, residuals, rank, sv = np.linalg.lstsq(Y, rhs, rcond=None)
                
                # Check if the solution is exact
                # We need Y * x == rhs
                if np.allclose(Y @ x, rhs, atol=1e-9):
                    W[c_idx, :] = x
                else:
                    # If not solvable, this U, V pair fails
                    return None
        
        # Verify the full decomposition
        # T_reconstructed = sum_c U[:, c] * V[:, c] * W[:, c]
        # But we only need to check the non-zero entries.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    val = np.dot(U[a_idx, :], V[b_idx, :] * W[c_idx, :])
                    if abs(val - 1) > 1e-9:
                        return None
        
        # If we get here, we have a valid decomposition
        return U, V, W
    
    # Try ranks from 59 down to, say, 50
    # Given time limit, we can only try a few.
    for r in range(59, 49, -1):
        if time.time() > deadline:
            break
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            # Convert to lists of Fractions for exactness?
            # The problem allows integer/half-integer.
            # Our U, V, W are floats. We need to round them.
            # If they are close to integers or half-integers, we can round.
            # But they might be arbitrary reals.
            # The problem says "integer/half-integer entries".
            # This implies the solution space is discrete?
            # Or that we must output such entries.
            # If our random solution has arbitrary reals, we cannot output them.
            # We need a solution with integer/half-integer entries.
            # This makes the search much harder.
            
            # Let's check if the entries are close to integers or half-integers.
            def is_half_int(x):
                return abs(x * 2 - round(x * 2)) < 1e-9
            
            if all(is_half_int(x) for x in U.flatten()) and \
               all(is_half_int(x) for x in V.flatten()) and \
               all(is_half_int(x) for x in W.flatten()):
                
                U_frac = np.round(U * 2) / 2
                V_frac = np.round(V * 2) / 2
                W_frac = np.round(W * 2) / 2
                
                # Convert to lists
                U_list = U_frac.tolist()
                V_list = V_frac.tolist()
                W_list = W_frac.tolist()
                
                return {"U": U_list, "V": V_list, "W": W_list}
        
    # Fallback: Schoolbook
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

# EVOLVE-BLOCK-START
# (The construct function is defined above, but the block structure must be maintained)
# Wait, the prompt says "Keep the program's entry-point function name(s) and their return interface EXACTLY as in the current program above".
# And "Output the COMPLETE program in a single python code block."
# The current program has:
# DIMENSIONS = [2, 3, 10]
# # EVOLVE-BLOCK-START
# def construct(): ...
# # EVOLVE-BLOCK-END

# So I should wrap my code in the same structure.

# EVOLVE-BLOCK-END
