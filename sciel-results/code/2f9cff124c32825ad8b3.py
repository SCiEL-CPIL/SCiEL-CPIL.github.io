"""Find a low-rank exact tensor decomposition for 3x3 * 3x4 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    # Tensor T[k*n + i, j*m + k] = 1 if i == i (always) ... wait
    # The tensor for matmul (A_{n x m}, B_{m x p}) -> C_{n x p}
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # Tensor indices: T[i, j, k] = delta_{i,i} delta_{j,j} delta_{k,k} = 1
    # But the factorization convention is:
    # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
    # The task says: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # Wait, the indices in the task description:
    # U has shape (n*m, rank) -> indexed by (i, j)? or (i, j) flattened?
    # "U, V, W': factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "W[k*n+i,r]" -> W is indexed by (k, i) where k is in [0,p), i is in [0,n)
    # So W row index = k*n + i. This corresponds to the output indices (k, i) i.e. (column, row) of C.
    # So C_{i,k} = sum_{j,r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Let's verify the schoolbook:
    # U[i*m+j, r] = 1 if r = (i*m+j)*p + k ... no.
    # Schoolbook: r indexes (i,j,k).
    # U[i*m+j, (i,j,k)] = 1
    # V[j*p+k, (i,j,k)] = 1
    # W[k*n+i, (i,j,k)] = 1
    # Product: sum_{(i',j',k')} U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # = sum_{r} [r corresponds to (i',j',k')] delta_{i',i} delta_{j',j} delta_{k',k}
    # Wait, U depends on i',j'. V on j',k'. W on k',i'.
    # So U[i*m+j, r]=1 iff i'=i, j'=j.
    # V[j*p+k, r]=1 iff j'=j, k'=k.
    # W[k*n+i, r]=1 iff k'=k, i'=i.
    # Sum is 1 if there exists r=(i',j',k') such that i'=i, j'=j, k'=k.
    # This gives 1 for C[i,k] for any j? No.
    # The sum over r is just 1.
    # But C[i,k] should be sum_j A[i,j]B[j,k].
    # The tensor T[i,j,k] = 1.
    # C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k] = sum_j A[i,j]B[j,k].
    # So the tensor is all ones.
    
    # We need to find U (nm x r), V (mp x r), W (pn x r) such that
    # T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    # for all i in [0,n), j in [0,m), k in [0,p).
    # Note: The tensor is indexed by (i,j,k) but the factors are flattened differently.
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # Target: T is a 3D tensor of size (n*m) x (m*p) x (p*n) ?
    # No, the tensor is defined on the triplet (i,j,k).
    # Let's flatten the tensor T to a matrix or use a direct residual function.
    
    # Residual function for given U, V, W flattened to vectors u, v, w
    # We have variables: u (nm*r), v (mp*r), w (pn*r).
    # Constraints: for each (i,j,k), sum_r u[i*m+j, r] * v[j*p+k, r] * w[k*n+i, r] - 1 = 0.
    
    # This is a system of n*m*p equations.
    # For n=3, m=3, p=4, that's 36 equations.
    
    # We want to minimize rank r.
    # Known lower bound: max(n*m, m*p, p*n) = max(9, 12, 12) = 12.
    # Schoolbook rank: 36.
    # Let's try r from 12 to 36.
    # For each r, we have 3 * (nm+mp+pn) * r variables.
    # nm=9, mp=12, pn=12. Sum = 33.
    # Variables = 33 * r.
    # For r=12, vars = 396. Equations = 36.
    # This is highly underdetermined. A solution likely exists for small r.
    
    # However, the entries must be integer or half-integer.
    # This suggests the solution has a specific structure.
    
    # Let's define the residual vector.
    # We iterate over all i, j, k.
    def make_residual_func(n, m, p):
        def residual(params):
            # params: [u, v, w]
            # u size: n*m
            # v size: m*p
            # w size: p*n
            # r is implicit in the shape of u[0]
            u = params[:n*m]
            v = params[n*m:n*m+m*p]
            w = params[n*m+m*p:n*m+m*p+p*n]
            
            # r = len(u[0])
            r = len(u[0])
            
            # Residual vector of size n*m*p
            res = np.zeros(n*m*p)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        val = np.dot(u[i*m+j] * v[j*p+k], w[k*n+i])
                        res[idx] = val - 1.0
                        idx += 1
            return res
        return residual
    
    # We need to handle the variable count changing with r.
    # least_squares expects a fixed shape.
    # So we define a solver function for a specific r.
    
    def solve_for_rank(r, n, m, p, num_trials=5, max_iter=2000):
        size_u = n * m
        size_v = m * p
        size_w = p * n
        total_size = size_u + size_v + size_w
        
        for trial in range(num_trials):
            # Initialize with random values
            # Since the solution might have small integers, start with small random numbers
            np.random.seed(trial + 12345)
            params = np.random.randn(total_size * r) * 0.5
            
            # Reshape to (total_size, r) for clarity, but least_squares uses flat
            # Let's create a residual function that reshapes
            def residual_flat(x):
                u_v = x[:size_u * r].reshape(size_u, r)
                v_v = x[size_u * r : size_u * r + size_v * r].reshape(size_v, r)
                w_v = x[size_u * r + size_v * r :].reshape(size_w, r)
                
                # Compute residual
                res = np.zeros(n*m*p)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # sum_r u[i*m+j, r] * v[j*p+k, r] * w[k*n+i, r]
                            # np.einsum('r,r,r->', u[i*m+j], v[j*p+k], w[k*n+i])
                            val = np.einsum('r,r,r->', u_v[i*m+j], v_v[j*p+k], w_v[k*n+i])
                            res[idx] = val - 1.0
                            idx += 1
                return res
            
            # Try to solve
            try:
                res = least_squares(residual_flat, params, max_nfev=max_iter, xtol=1e-12, ftol=1e-12, gtol=1e-12)
                if res.success or res.cost < 1e-8:
                    # Check if we can round to half-integers
                    u_final = res.x[:size_u * r].reshape(size_u, r)
                    v_final = res.x[size_u * r : size_u * r + size_v * r].reshape(size_v, r)
                    w_final = res.x[size_u * r + size_v * r :].reshape(size_w, r)
                    
                    # Round to nearest 0.5
                    u_round = np.round(u_final * 2) / 2
                    v_round = np.round(v_final * 2) / 2
                    w_round = np.round(w_final * 2) / 2
                    
                    # Verify exactness (within tight tolerance)
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.einsum('r,r,r->', u_round[i*m+j], v_round[j*p+k], w_round[k*n+i])
                                err = abs(val - 1.0)
                                if err > max_err:
                                    max_err = err
                            
                    if max_err < 1e-9:
                        # Convert to list of lists
                        U = u_round.tolist()
                        V = v_round.tolist()
                        W = w_round.tolist()
                        # Clean up floats to int if possible
                        def clean_matrix(mat):
                            cleaned = []
                            for row in mat:
                                new_row = []
                                for val in row:
                                    if abs(val - round(val)) < 1e-9:
                                        new_row.append(int(round(val)))
                                    else:
                                        new_row.append(val)
                                cleaned.append(new_row)
                            return cleaned
                        U = clean_matrix(U)
                        V = clean_matrix(V)
                        W = clean_matrix(W)
                        return U, V, W, r
            except Exception as e:
                pass
            
            if time.time() - start_time > deadline - 5:
                break
                
        return None

    # Try ranks from low to high
    # Lower bound is 12. Let's start from 12.
    # But maybe the structure requires higher rank.
    # Schoolbook is 36.
    # Let's try 12 to 36.
    for r in range(12, 37):
        if time.time() - start_time > deadline - 5:
            break
        result = solve_for_rank(r, n, m, p, num_trials=3 if r > 20 else 10, max_iter=1000 if r < 20 else 500)
        if result is not None:
            U, V, W, rank = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
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
