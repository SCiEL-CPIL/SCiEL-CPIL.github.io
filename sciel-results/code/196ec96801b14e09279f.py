import numpy as np
import time
import sys

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Theoretical lower bound is ceil(nmp / max(n,m,p)) ? No, lower bound is usually related to border rank.
    # Simple lower bound: max(n*m, m*p, p*n) is not correct. 
    # Lower bound is at least max(n, m, p) * something? 
    # Actually, for matrix multiplication, rank is at least n*m*p / max(n,m,p) is a heuristic, 
    # but the true lower bound is often close to n*m*p / 4 for large matrices.
    # For 2x4x7, let's try ranks from 14 up to 55.
    
    start_time = time.time()
    time_limit = 140.0 # Safety margin
    
    # Pre-compute the target tensor T as a 3D array
    # T[i, j, k] = 1 if this term contributes to the product, else 0
    # Wait, the tensor for matrix multiplication C = AB is:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # So the tensor T has entries T_{i,j,k} = 1. 
    # The decomposition is sum_c U_{i,c} V_{j,c} W_{k,c} = T_{i,j,k}
    # Here, the indices are:
    # U is (n*m, r) -> Index is i*m + j ? No, the standard tensor T_{ijk} has i in [0,n), j in [0,m), k in [0,p).
    # The problem statement says:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    # And W[k*n+i, r].
    # This implies the tensor being decomposed is not the standard T_{ijk}=1.
    # Let's map the indices carefully.
    # Standard: C_{ik} = sum_j A_{ij} B_{jk}.
    # The tensor T has dimensions (n, m, p).
    # The factors are usually U (n, r), V (m, r), W (p, r).
    # Here, U is (n*m, r). This suggests the input is flattened.
    # Let's re-read: "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "W[k*n+i, r]"
    # This looks like the factors are acting on the flattened indices?
    # No, matrix multiplication is a trilinear map.
    # If U is (n*m, r), it mixes i and j? That would break the bilinearity in A and B separately.
    # UNLESS the "matrix multiplication" is defined differently or the indices in the prompt are specific to a flattened representation where the tensor is not the standard one?
    # Let's look at the schoolbook code in the seed:
    # U[i * m + j][column] = 1
    # V[j * p + k][column] = 1
    # W[k * n + i][column] = 1
    # This means:
    # U has row index i*m+j.
    # V has row index j*p+k.
    # W has row index k*n+i.
    # The product sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] should equal the tensor entry.
    # In the schoolbook case, for a specific (i,j,k), only the term where c corresponds to (i,j,k) is non-zero.
    # U[i*m+j, c_ijk] = 1
    # V[j*p+k, c_ijk] = 1
    # W[k*n+i, c_ijk] = 1
    # Sum = 1.
    # For any other (i',j',k'), if i'!=i or j'!=j or k'!=k, at least one factor is 0.
    # So the tensor T_{i,j,k} = 1 for all i,j,k?
    # Yes, the standard matrix multiplication tensor has T_{i,j,k} = 1.
    # The "flattening" in U and V is just a way to pack the indices into the row indices of the factor matrices, but the tensor structure itself is the standard one: T[i, j, k] = 1.
    # Wait, if U depends on i and j, and V depends on j and k, and W depends on k and i...
    # This is NOT the standard matrix multiplication tensor decomposition form.
    # Standard form: U(i, c), V(j, c), W(k, c).
    # Here: U(i*m+j, c). This implies the factor U is not linear in A alone?
    # If A is a 2x4 matrix, it has 8 elements. U is 8xR.
    # If we form the product, do we get C?
    # The problem says "exactly decompose the matrix multiplication tensor".
    # And "score is the rank".
    # The seed program works with score 1.0 (rank 56).
    # So the definition of "rank" here is the dimension of the intermediate space in this specific factorization format.
    # The tensor is T[i, j, k] = 1.
    # The decomposition is: T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    # This is a valid tensor decomposition, just with a specific index mapping.
    
    # Let's verify the tensor entries.
    # T[i, j, k] should be 1 for all valid i,j,k.
    # i in [0,2), j in [0,4), k in [0,7).
    
    # Strategy: 
    # We want to find U, V, W with rank R such that for all i,j,k:
    # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1
    
    # This is a system of quadratic equations in the entries of U, V, W.
    # We can use a random initialization and gradient descent / Newton steps.
    # However, the equations are cubic in the variables if we consider the product of 3 variables?
    # No, if we fix U and V, the equation is linear in W.
    # If we fix U and W, it is linear in V.
    # If we fix V and W, it is linear in U.
    # This allows for a coordinate descent or alternating least squares (ALS) approach.
    
    # Let's implement a simple ALS with random restarts.
    
    def try_decompose(rank, max_iter=200, seed=0):
        np.random.seed(seed)
        # Initialize U, V, W
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Precompute indices for speed
        # We need to solve for one factor at a time.
        # Equation: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1
        
        # Let's reshape for easier linear solving.
        # For a fixed i, j:
        # sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all k.
        # This is a vector equation for each (i,j).
        # Let u_idx = i*m + j.
        # Let v_idx = j*p + k.
        # Let w_idx = k*n + i.
        
        # This coupling is tricky for simple ALS because the indices are entangled.
        # Let's try a different approach: 
        # Since the tensor is all 1s, maybe there is a low-rank structure?
        # T[i,j,k] = 1.
        # Can we write 1 = sum_c a_c(i,j) b_c(j,k) c_c(k,i)?
        
        # Let's try small ranks first.
        # If we can't find a solution, return None.
        
        # We will use a simple iterative solver:
        # 1. Fix V and W, solve for U.
        # 2. Fix U and W, solve for V.
        # 3. Fix U and V, solve for W.
        
        # For step 1:
        # For each i, j:
        # sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1 for all k.
        # This is not a standard linear system because the coefficient of U depends on k.
        # Actually, for a fixed i, j, we have p equations (one for each k) and rank unknowns (U[i*m+j, 0..r-1]).
        # If rank < p, it might be underdetermined or inconsistent.
        # The system is:
        # M[i,j] * U[i*m+j, :] = ones(p)
        # Where M[i,j] is a p x rank matrix with M[i,j][k, c] = V[j*p+k, c] * W[k*n+i, c].
        
        # So we can solve this least squares problem for each (i,j).
        
        def solve_U(V, W):
            new_U = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    # Build M
                    M = np.zeros((p, rank))
                    for k in range(p):
                        v_idx = j * p + k
                        w_idx = k * n + i
                        M[k, :] = V[v_idx, :] * W[w_idx, :]
                    # Solve M * u = ones(p)
                    try:
                        u, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        new_U[i * m + j, :] = u
                    except:
                        return None
            return new_U

        def solve_V(U, W):
            new_V = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, rank))
                    for i in range(n):
                        u_idx = i * m + j
                        w_idx = k * n + i
                        M[i, :] = U[u_idx, :] * W[w_idx, :]
                    try:
                        v, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
                        new_V[j * p + k, :] = v
                    except:
                        return None
            return new_V

        def solve_W(U, V):
            new_W = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, rank))
                    for j in range(m):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        M[j, :] = U[u_idx, :] * V[v_idx, :]
                    try:
                        w, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        new_W[k * n + i, :] = w
                    except:
                        return None
            return new_W

        # Iterate
        for it in range(max_iter):
            U_new = solve_U(V, W)
            if U_new is None: return None
            V_new = solve_V(U_new, W)
            if V_new is None: return None
            W_new = solve_W(U_new, V_new)
            if W_new is None: return None
            
            # Check error
            error = 0.0
            # Sample some points to check error quickly
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(rank):
                            val += U_new[i*m+j, c] * V_new[j*p+k, c] * W_new[k*n+i, c]
                        error += (val - 1.0)**2
            if error < 1e-8:
                return U_new, V_new, W_new
            
            U, V, W = U_new, V_new, W_new
            
            if time.time() - start_time > time_limit:
                return None

        return None

    # Try ranks from lower bound to schoolbook
    # Lower bound: 
    # The rank of the tensor T (all 1s) in this specific format.
    # Let's just try a range.
    # 14 is ceil(56/4).
    # We will try ranks 14 to 55.
    # To save time, we will try a few random seeds for each rank.
    
    best_result = None
    best_rank = rank_schoolbook
    
    # We know 56 works.
    # Let's try to beat it.
    
    # Optimization: 
    # If we find a solution with rank R, we can try to compress it? 
    # No, just search for lower R.
    
    # We will iterate R from 14 to 55.
    # To avoid spending too much time on low ranks that are unlikely to work, 
    # we can limit the number of iterations or seeds.
    
    # However, the ALS might converge to a local minimum.
    # Given the small size (2x4x7), the search space is small enough that random restarts might find it.
    
    # Let's try ranks 20 to 55 first, then 14 to 19 if time permits.
    # Actually, let's just go 14 to 55.
    
    for r in range(14, 56):
        if time.time() - start_time > time_limit - 10:
            break
            
        # Try 5 random restarts for this rank
        found = False
        for seed in range(5):
            if time.time() - start_time > time_limit - 5:
                break
            res = try_decompose(r, max_iter=100, seed=seed)
            if res is not None:
                U, V, W = res
                # Snap to half-integers?
                # The problem allows integer/half-integer.
                # Our solution is floating point.
                # We should check if it can be represented as half-integers.
                # Multiply by 2, round, divide by 2.
                U_h = np.round(U * 2) / 2
                V_h = np.round(V * 2) / 2
                W_h = np.round(W * 2) / 2
                
                # Verify the snapped solution
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(r):
                                val += U_h[i*m+j, c] * V_h[j*p+k, c] * W_h[k*n+i, c]
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                
                if valid:
                    # Convert to list of lists
                    U_list = [[float(U_h[i*m+j, c]) for c in range(r)] for i in range(n) for j in range(m)]
                    V_list = [[float(V_h[j*p+k, c]) for c in range(r)] for j in range(m) for k in range(p)]
                    W_list = [[float(W_h[k*n+i, c]) for c in range(r)] for k in range(p) for i in range(n)]
                    return {"U": U_list, "V": V_list, "W": W_list}
        # If not found for this rank, continue to next rank

    # If no better rank found, return schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
