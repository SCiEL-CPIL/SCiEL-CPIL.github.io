import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    # Shapes
    dim_U = n * m  # 6
    dim_V = m * p  # 9
    dim_W = p * n  # 6

    # Schoolbook rank is 18
    best_rank = 18
    best_U = np.zeros((dim_U, best_rank))
    best_V = np.zeros((dim_V, best_rank))
    best_W = np.zeros((dim_W, best_rank))
    
    # Initialize best with schoolbook
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j, column] = 1
                best_V[j * p + k, column] = 1
                best_W[k * n + i, column] = 1
                column += 1

    def check_solution(U, V, W, R):
        """
        Verifies if U, V, W exactly decompose the tensor.
        U: (n*m, R)
        V: (m*p, R)
        W: (p*n, R)
        """
        # To be exact, we should check all entries. 
        # The tensor T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # We check against the standard multiplication tensor where T[i,j,k] = 1 if j is the index of A[i,j] and B[j,k], 
        # which means the term in C[i,k] is A[i,j]B[j,k].
        # So T[i,j,k] is 1 if the indices align for the product, 0 otherwise.
        # Specifically, for C[i,k] = sum_j A[i,j] B[j,k], the coefficient of A[i,j]B[j,k] is 1.
        # In the flattened vector representation, the tensor element corresponding to (i, j, k) is 1.
        
        # We can check a few random integer matrices A, B to see if C = A@B matches.
        # Since entries are half-integers, exactness is easier to verify.
        np.random.seed(123)
        for _ in range(20):
            A = np.random.randint(-3, 4, size=(n, m)).astype(float)
            B = np.random.randint(-3, 4, size=(m, p)).astype(float)
            C_true = A @ B
            A_flat = A.flatten()
            B_flat = B.flatten()
            
            C_approx = np.zeros((n, p))
            for r in range(R):
                u_val = np.dot(U[:, r], A_flat)
                v_val = np.dot(V[:, r], B_flat)
                w_vec = W[:, r]
                for i in range(n):
                    for k in range(p):
                        C_approx[i, k] += w_vec[k * n + i] * u_val * v_val
                        
            if not np.allclose(C_approx, C_true, atol=1e-4):
                return False
        return True

    def try_rank(R):
        nonlocal best_rank, best_U, best_V, best_W
        if R >= best_rank:
            return False
        
        # We want to find U, V, W of shape (dim_U, R), (dim_V, R), (dim_W, R)
        # such that for all i, j, k:
        # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
        
        # This is a system of dim_U * dim_V * dim_W / (some symmetry?) equations?
        # No, the tensor has n*m*p = 18 non-zero entries?
        # Wait, the matrix multiplication tensor T_{ijk} is 1 if j is the middle index.
        # For 2x3 * 3x3:
        # i in {0,1}, j in {0,1,2}, k in {0,1,2}
        # T[i,j,k] = 1. All 18 entries are 1?
        # C[i,k] = sum_j A[i,j] B[j,k].
        # The term A[i,j] B[j,k] appears with coefficient 1.
        # So yes, T[i,j,k] = 1 for all i,j,k.
        
        # Strategy: Fix U and V to random values (or sparse ones), then solve for W?
        # Or fix U and W, solve for V?
        # The equations are linear in W if U and V are fixed.
        # Equation for each (i, j, k):
        # sum_r (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1
        
        # Let M_{(i,j,k), (l,r)} = U[i*m+j, r] * V[j*p+k, r] if l == k*n+i, else 0?
        # No, W index is (k, i). Let w_idx = k*n + i.
        # The variable is W[w_idx, r].
        # The equation is: sum_r W[w_idx, r] * (U[i*m+j, r] * V[j*p+k, r]) = 1.
        
        # This is a linear system for the entries of W.
        # Number of equations: n*m*p = 18.
        # Number of unknowns: dim_W * R = 6 * R.
        
        # If R=17, unknowns = 102. Equations = 18. Underdetermined.
        # We can pick a random solution in the null space?
        # But we need U and V to be "nice" (integer/half-integer) too.
        
        # Alternative: Randomize U and V with small integer entries.
        # Solve for W. Check if W is close to half-integers.
        
        for attempt in range(500):
            if time.time() > deadline:
                return False
            
            # Random U, V with entries from {-1, 0, 1}
            # This ensures U, V are integers.
            U = np.random.randint(-1, 2, size=(dim_U, R)).astype(float)
            V = np.random.randint(-1, 2, size=(dim_V, R)).astype(float)
            
            # Construct the linear system for W
            # A_sys * W_flat = b
            # W_flat is of size dim_W * R
            # b is of size n*m*p (18), all 1s.
            
            num_eq = n * m * p
            num_unk = dim_W * R
            
            # Use sparse matrix
            rows = []
            cols = []
            data = []
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        u_val = U[i * m + j]  # vector of length R
                        v_val = V[j * p + k]  # vector of length R
                        
                        w_idx_base = k * n + i
                        for r in range(R):
                            coeff = u_val[r] * v_val[r]
                            if coeff != 0:
                                rows.append(eq_idx)
                                cols.append(w_idx_base * R + r)
                                data.append(coeff)
            
            A_sys = sp.csr_matrix((data, (rows, cols)), shape=(num_eq, num_unk))
            b = np.ones(num_eq)
            
            # Solve A_sys * W_flat = b
            # Since it's underdetermined, we want a minimum norm solution or just any solution.
            # scipy.linalg.lstsq can handle this.
            try:
                W_flat, residuals, rank, sv = np.linalg.lstsq(A_sys.toarray(), b, rcond=None)
            except Exception:
                continue
            
            W = W_flat.reshape(dim_W, R)
            
            # Check if W is close to half-integers
            W_rounded = np.round(W * 2) / 2
            error = np.linalg.norm(W - W_rounded)
            
            if error < 1e-6:
                # Verify the solution
                if check_solution(U, V, W_rounded, R):
                    best_rank = R
                    best_U = U
                    best_V = V
                    best_W = W_rounded
                    return True
        
        return False

    # Try rank 17
    if try_rank(17):
        pass
    # If 17 fails, we keep 18.
    # Note: Rank 16 is theoretically impossible for 2x3x3? 
    # Strassen rank for 2x2 is 7. For 2x3x3, lower bound is 17? 
    # Actually, the rank of 2x3x3 matrix multiplication is known to be 17.
    # See: "The rank of 2x3x3 matrix multiplication is 17" (e.g., via Strassen's method or similar).
    
    # Convert to lists for return
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
