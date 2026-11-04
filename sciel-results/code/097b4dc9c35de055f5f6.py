"""Matrix multiplication 3,3,5 — rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=5
    
    # Build the target tensor: T[i,j,k] = 1 for matrix mult A(i,m)*B(m,p)=C(i,p)
    # U index: i*m+j, V index: j*p+k, W index: k*n+i
    # Tensor entry: T[U_idx, V_idx, W_idx] = 1 if U_idx=i*m+j, V_idx=j*p+k, W_idx=k*n+i
    # i.e., T[i*m+j, j*p+k, k*n+i] = 1
    
    # The decomposition: C[i,k] = sum_t W[k*n+i, t] * (sum_j U[i*m+j,t]*A[i,j]) * (sum_j V[j*p+k,t]*B[j,k])
    # This equals sum_t W[k*n+i,t] * (U_vec_t . A_row_i) * (V_vec_t . B_col_k)
    # where U_vec_t is the vector of U[:,t] reshaped, etc.
    
    # For this to equal C[i,k] = sum_j A[i,j]*B[j,k], we need:
    # sum_t W[k*n+i,t] * U[i*m+j,t] * V[j*p+k,t] = 1 for all i,j,k (the tensor condition)
    # Wait, that's not quite right. Let me re-derive.
    
    # C[i,k] = sum_t W[k*n+i,t] * (sum_j U[i*m+j,t]*A[i,j]) * (sum_j' V[j'*p+k,t]*B[j',k])
    # = sum_t W[k*n+i,t] * sum_j sum_j' U[i*m+j,t]*V[j'*p+k,t]*A[i,j]*B[j',k]
    # For this to equal sum_j A[i,j]*B[j,k] for all A,B:
    # sum_t W[k*n+i,t] * U[i*m+j,t] * V[j*p+k,t] = delta_{j,j'} ... 
    # Actually: sum_t W[k*n+i,t] * U[i*m+j,t] * V[j*p+k,t] = 1 if j=j', 0 otherwise
    # i.e., sum_t W[k*n+i,t] * U[i*m+j,t] * V[j*p+k,t] = 1 for all i,j,k
    
    # So the tensor condition is: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_t W[k*n+i, t] * U[i*m+j, t] * V[j*p+k, t] = 1
    
    # Let me try to find a decomposition with rank r using a randomized approach.
    
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    # Try ranks from low to high
    # Lower bound: at least max(n*m, m*p, p*n) / something... 
    # Actually, the tensor has 45 nonzero entries, so rank >= 45/9 = 5? No, that's not right.
    # The rank is at least the max of the "slice ranks". 
    
    # Let me try a practical approach: use a known structure.
    # Split into 3x3x3 (cols 0-2) and 3x3x2 (cols 3-4).
    
    # For 3x3x3, use the Lader rank-19 algorithm.
    # For 3x3x2, find a good decomposition.
    
    # Actually, let me just try to find ANY valid decomposition with rank < 45.
    # I'll use a randomized search with local improvement.
    
    def verify(U, V, W):
        """Check if U,V,W form a valid decomposition."""
        r = U.shape[1]
        # For each (i,j,k), check: sum_t W[k*n+i,t] * U[i*m+j,t] * V[j*p+k,t] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for t in range(r):
                        s += W[k*n+i, t] * U[i*m+j, t] * V[j*p+k, t]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def is_half_integer(arr):
        """Check if all entries are half-integers (i.e., 2*x is integer)."""
        return np.allclose(2*np.array(arr), np.round(2*np.array(arr)), atol=1e-9)
    
    # Try to find a decomposition using randomized local search
    best_rank = None
    best_decomp = None
    
    # Start with schoolbook as fallback
    schoolbook_rank = n * m * p
    U_sb = np.zeros((n*m, schoolbook_rank))
    V_sb = np.zeros((m*p, schoolbook_rank))
    W_sb = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    # Try randomized search for lower rank
    # Strategy: for each rank r, try random initializations and local search
    
    for r in range(25, schoolbook_rank):  # Try ranks from 25 to 44
        if time.time() > deadline - 10:
            break
        
        max_attempts = 20
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                    break
            
            # Random initialization with half-integers in {-1, -0.5, 0, 0.5, 1}
            vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
            U = np.array([[np.random.choice(vals) for _ in range(r)] for _ in range(n*m)])
            V = np.array([[np.random.choice(vals) for _ in range(r)] for _ in range(m*p)])
            W = np.array([[np.random.choice(vals) for _ in range(r)] for _ in range(p*n)])
            
            # Local search: try to fix violations
            improved = True
            max_iters = 50
            for it in range(max_iters):
                if time.time() > deadline - 3:
                    break
                # Find violations
                violations = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = sum(W[k*n+i, t] * U[i*m+j, t] * V[j*p+k, t] for t in range(r))
                            if abs(s - 1.0) > 1e-9:
                                violations.append((i, j, k, s))
                if not violations:
                    # Found a valid decomposition!
                    if is_half_integer(U) and is_half_integer(V) and is_half_integer(W):
                        best_rank = r
                        best_decomp = (U, V, W)
                        improved = False
                        break
                    else:
                        # Snap to half-integers and re-verify
                        U_snapped = np.round(2*U)/2
                        V_snapped = np.round(2*V)/2
                        W_snapped = np.round(2*W)/2
                        if verify(U_snapped, V_snapped, W_snapped):
                            best_rank = r
                            best_decomp = (U_snapped, V_snapped, W_snapped)
                            improved = False
                            break
                
                if not improved:
                    break
                
                # Try to fix one violation by changing one entry
                if violations:
                    i, j, k, s = violations[np.random.randint(len(violations))]
                    target = 1.0
                    # Try changing U[i*m+j, t] for some t
                    t = np.random.randint(r)
                    current = U[i*m+j, t]
                    # What change in U[i*m+j, t] would help?
                    # d(s)/d(U[i*m+j,t]) = W[k*n+i,t] * V[j*p+k,t]
                    grad = W[k*n+i, t] * V[j*p+k, t]
                    if abs(grad) > 1e-9:
                        # New value to make s closer to 1
                        new_val = current + (target - s) / grad
                        # Snap to nearest half-integer
                        new_val = np.round(2*new_val)/2
                        if new_val != current:
                            U[i*m+j, t] = new_val
                            # Re-snap all to half-integers
                            U = np.round(2*U)/2
                            V = np.round(2*V)/2
                            W = np.round(2*W)/2
                    else:
                        # Try changing V or W
                        if np.random.random() < 0.5:
                            t2 = np.random.randint(r)
                            current = V[j*p+k, t2]
                            grad = W[k*n+i, t2] * U[i*m+j, t2]
                            if abs(grad) > 1e-9:
                                new_val = current + (target - s) / grad
                                new_val = np.round(2*new_val)/2
                                if new_val != current:
                                    V[j*p+k, t2] = new_val
                                    V = np.round(2*V)/2
                        else:
                            t2 = np.random.randint(r)
                            current = W[k*n+i, t2]
                            grad = U[i*m+j, t2] * V[j*p+k, t2]
                            if abs(grad) > 1e-9:
                                new_val = current + (target - s) / grad
                                new_val = np.round(2*new_val)/2
                                if new_val != current:
                                    W[k*n+i, t2] = new_val
                                    W = np.round(2*W)/2
            improved = False
        
        if best_rank is not None:
            break
    
    if best_rank is not None:
        U, V, W = best_decomp
        # Convert to lists
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
        U_list = [[float(x) for x in row] for row in U_sb]
        V_list = [[float(x) for x in row] for row in V_sb]
        W_list = [[float(x) for x in row] for row in W_sb]
        return {"U": U_list, "V": V_list, "W": W_list}
