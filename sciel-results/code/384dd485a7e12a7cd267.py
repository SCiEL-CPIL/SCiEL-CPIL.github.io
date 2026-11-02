"""Matrix multiplication 3x4x4 tensor decomposition search."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 4]

# The schoolbook rank for (n,m,p) is n*m*p = 48.
# We try to find a lower-rank decomposition.
# The tensor T has indices (i,j,k) -> (i*m+j, j*p+k, k*n+i) with value 1.
# A rank-r decomposition gives U (n*m x r), V (m*p x r), W (p*n x r) such that
# for all (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
# and for all "forbidden" triples (where the product should be 0 in the tensor),
# the sum should be 0.

# Actually, the matrix multiplication tensor is defined as:
# T[i, j, k] = 1 if the output index is k*n+i and input indices are i*m+j, j*p+k.
# The bilinear form: (A*B)[k, i] = sum_j A[i, j] * B[j, k]
# So the tensor has a 1 at (i*m+j, j*p+k, k*n+i) for all valid (i,j,k).
# All other entries are 0.

def _verify(U, V, W, n=3, m=4, p=4):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Check all valid (i, j, k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check all forbidden triples
    # A triple (a, b, c) is valid if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    # i.e., a mod m == b div p and b mod p == c div n and c mod n == a div m
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                if j == j2 and k == k2 and i == i2:
                    continue  # valid triple, already checked
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val) > 1e-9:
                    return False
    return True

def _verify_fast(U, V, W, n=3, m=4, p=4):
    """Faster verification using tensor contraction."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Build the full tensor T[a, b, c] = sum_c U[a,c]*V[b,c]*W[c,c]... wait
    # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    # We need T to match the matrix multiplication tensor.
    
    # Efficiently: for each r, the contribution is outer(U[:,r], V[:,r], W[:,r])
    # T = sum_r np.einsum('ar,br,cr->abc', U, V, W)
    # But that's expensive. Let's use a different approach.
    
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # = (U @ V.T) ... no, that's not right.
    # T[a,b,c] = sum_r U[a,r] * (V[b,r] * W[c,r])
    
    # Let's just check valid and forbidden separately with vectorized ops
    # Valid triples:
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                val = np.dot(U[a], V[b] * W[c])
                if abs(val - 1.0) > 1e-8:
                    return False
    
    # Forbidden triples - vectorized check
    # Build all triples and check
    # This is n*m * m*p * p*n = 12*16*12 = 2304 triples total, minus 48 valid = 2256 forbidden
    # For each forbidden (a,b,c), check sum_r U[a,r]*V[b,r]*W[c,r] == 0
    
    # Vectorized: for each r, we have U[:,r] (12,), V[:,r] (16,), W[:,r] (12,)
    # T[:, :, r] = U[:, r, None, None] * V[None, :, r, None] * W[None, None, r, :]
    # This is 12*16*12*rank which might be large.
    
    # Alternative: iterate over forbidden triples but vectorize over r
    forbidden = []
    valid_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid_set.add((i*m+j, j*p+k, k*n+i))
    
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                if (a, b, c) not in valid_set:
                    forbidden.append((a, b, c))
    
    # Check forbidden triples
    for (a, b, c) in forbidden:
        val = np.dot(U[a], V[b] * W[c])
        if abs(val) > 1e-8:
            return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    schoolbook_rank = nm * mp * pn  # 48... wait no
    # Schoolbook rank is n*m*p = 3*4*4 = 48
    schoolbook_rank = n * m * p
    
    # Start with schoolbook decomposition
    best_rank = schoolbook_rank
    best_U = [[0]*best_rank for _ in range(nm)]
    best_V = [[0]*best_rank for _ in range(mp)]
    best_W = [[0]*best_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j][col] = 1
                best_V[j*p+k][col] = 1
                best_W[k*n+i][col] = 1
                col += 1
    
    # Try to find lower rank decompositions
    # Strategy: random initialization + gradient descent / iterative refinement
    
    # The problem is to find U (nm x r), V (mp x r), W (pn x r) such that:
    # sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
    # where T is the matrix multiplication tensor.
    
    # This is a non-convex optimization problem. We can use:
    # 1. Random initialization and alternating least squares
    # 2. Gradient descent
    # 3. Try small ranks first
    
    # Let's try ranks from lower to higher
    # The minimum possible rank is at least max(nm, mp, pn) = 16
    # Actually, the rank is at least the "tensor rank" which has lower bounds.
    # For matrix multiplication, the rank is at least n*m*p / (something)...
    # Let's just try ranks from 16 upward.
    
    # Precompute the target tensor values
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[(i*m+j, j*p+k, k*n+i)] = 1.0
    
    def compute_residual(U_arr, V_arr, W_arr, r):
        """Compute the residual (squared error) of the decomposition."""
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # residual = sum over all (a,b,c) of (T[a,b,c] - sum_r U[a,r]V[b,r]W[c,r])^2
        # Only non-zero entries of T matter for the "should be 1" part
        # For "should be 0" entries, we need them to be 0
        
        # Efficient computation:
        # For each r, the contribution to the tensor is the outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # We need sum_r of these to equal T.
        
        # residual = ||sum_r U[:,r]⊗V[:,r]⊗W[:,r] - T||^2
        # = ||sum_r ...||^2 - 2 <sum_r ..., T> + ||T||^2
        # ||T||^2 = n*m*p = 48
        
        # <sum_r U[:,r]⊗V[:,r]⊗W[:,r], T> = sum_r sum_{(a,b,c) in valid} U[a,r]V[b,r]W[c,r]
        
        # ||sum_r ...||^2 = sum_{r1, r2} sum_{a,b,c} U[a,r1]V[b,r1]W[c,r1] U[a,r2]V[b,r2]W[c,r2]
        # = sum_{r1,r2} (sum_a U[a,r1]U[a,r2]) (sum_b V[b,r1]V[b,r2]) (sum_c W[c,r1]W[c,r2])
        
        # This is getting complex. Let's just compute it directly for small sizes.
        
        residual = 0.0
        # Valid entries
        for (a, b, c), tval in target.items():
            s = 0.0
            for r_idx in range(r):
                s += U_arr[a, r_idx] * V_arr[b, r_idx] * W_arr[c, r_idx]
            residual += (s - tval) ** 2
        
        # Forbidden entries
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if (a, b, c) not in target:
                        s = 0.0
                        for r_idx in range(r):
                            s += U_arr[a, r_idx] * V_arr[b, r_idx] * W_arr[c, r_idx]
                        residual += s * s
        
        return residual
    
    def compute_residual_fast(U_arr, V_arr, W_arr, r):
        """Faster residual computation using matrix operations."""
        # Build the full tensor and compare
        # T_computed[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
        
        # Using einsum
        T_comp = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        
        # Target tensor
        T_target = np.zeros((nm, mp, pn))
        for (a, b, c), tval in target.items():
            T_target[a, b, c] = tval
        
        return np.sum((T_comp - T_target) ** 2)
    
    def random_decomposition(r, seed=None):
        """Create a random decomposition of rank r."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        # Initialize with small random values
        U = np.array([[rng.gauss(0, 0.5) for _ in range(r)] for _ in range(nm)])
        V = np.array([[rng.gauss(0, 0.5) for _ in range(r)] for _ in range(mp)])
        W = np.array([[rng.gauss(0, 0.5) for _ in range(r)] for _ in range(pn)])
        return U, V, W
    
    def als_step(U, V, W, r, lr=0.01, max_iter=1000):
        """Alternating least squares to minimize the residual."""
        # This is a non-convex problem, so ALS may not converge well.
        # Instead, let's use a simple gradient descent approach.
        pass
    
    def gradient_descent(U, V, W, r, lr=0.001, max_iter=5000, tol=1e-10):
        """Simple gradient descent to minimize the residual."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Compute gradients
            # residual = sum_{a,b,c} (sum_r U[a,r]V[b,r]W[c,r] - T[a,b,c])^2
            # d/dU[a,r'] = 2 * sum_{b,c} (sum_r U[a,r]V[b,r]W[c,r] - T[a,b,c]) * V[b,r'] * W[c,r']
            
            # This is expensive to compute naively. Let's use a simplified approach.
            # For each r', the gradient of U[:, r'] is:
            # 2 * (V[:, r'] * W[:, r']).T @ (T_comp - T_target)[:, :, :]
            # where the contraction is over b and c.
            
            T_comp = np.einsum('ar,br,cr->abc', U, V, W)
            T_target = np.zeros((nm, mp, pn))
            for (a, b, c), tval in target.items():
                T_target[a, b, c] = tval
            
            diff = T_comp - T_target
            
            # Gradient for U[:, r']
            # dU[a, r'] = 2 * sum_{b,c} diff[a,b,c] * V[b,r'] * W[c,r']
            # = 2 * sum_c (sum_b diff[a,b,c] * V[b,r']) * W[c,r']
            # Let's compute this efficiently
            # temp1[a, c, r'] = sum_b diff[a,b,c] * V[b,r']
            # Then dU[a, r'] = 2 * sum_c temp1[a, c, r'] * W[c, r']
            
            # Using einsum
            # temp1[a, c, r'] = einsum('abc,br->acr', diff, V)
            # dU[a, r'] = 2 * einsum('acr,cr->ar', temp1, W)
            
            temp1 = np.einsum('abc,br->acr', diff, V)
            dU = 2.0 * np.einsum('acr,cr->ar', temp1, W)
            
            # Similarly for V
            # dV[b, r'] = 2 * sum_{a,c} diff[a,b,c] * U[a,r'] * W[c,r']
            temp2 = np.einsum('abc,ar->bcr', diff, U)
            dV = 2.0 * np.einsum('bcr,cr->br', temp2, W)
            
            # And for W
            # dW[c, r'] = 2 * sum_{a,b} diff[a,b,c] * U[a,r'] * V[b,r']
            temp3 = np.einsum('abc,ar->bcr', diff, U)
            dW = 2.0 * np.einsum('bcr,br->cr', temp3, V)
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            if it % 100 == 0:
                res = np.sum(diff**2)
                if res < tol:
                    break
        
        return U, V, W
    
    # Try different ranks
    # The theoretical lower bound for the rank of matrix multiplication tensor
    # For (n,m,p), the rank is at least max(n*m, m*p, p*n) = max(12, 16, 12) = 16
    # But in practice, it's much higher.
    # Let's try ranks from 20 to 47.
    
    best_result = None
    for r in range(20, schoolbook_rank):
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            U, V, W = random_decomposition(r, seed=42 + r*10 + attempt)
            U, V, W = gradient_descent(U, V, W, r, lr=0.005, max_iter=3000)
            
            if time.time() > deadline:
                break
            
            # Verify
            if _verify_fast(U, V, W):
                # Convert to lists
                best_result = (r, U.tolist(), V.tolist(), W.tolist())
                found = True
                break
        
        if found:
            break
    
    if best_result is not None:
        r, U_list, V_list, W_list = best_result
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Return schoolbook if nothing better found
    return {"U": best_U, "V": best_V, "W": best_W}
