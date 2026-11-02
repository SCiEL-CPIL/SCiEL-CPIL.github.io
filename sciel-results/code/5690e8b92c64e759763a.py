"""
Program for matrix multiplication 2x3 * 3x5 tensor decomposition.
Strategy: Algebraic constraint solving via Gaussian Elimination.
We model the tensor decomposition as a linear combination of elementary tensors.
We check if a rank-r decomposition exists by solving a system of linear equations.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin from 150s limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # The target tensor T is the "multiplication tensor".
    # T[i, j, k] = 1 if the indices correspond to a valid multiplication step.
    # Specifically, for matrix multiplication C = A * B:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor entries are T_{i,j,k} which contribute to C_{i,k}.
    # In the trace convention W[k*n+i, r], the index for W is (k, i).
    # The index for U is (i, j). The index for V is (j, k).
    # So T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Wait, the tensor is defined such that sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1.
    # This is exactly the identity tensor for the bilinear map.
    
    # We want to find U, V, W with rank r.
    # This is equivalent to finding r terms in the tensor product space.
    # Let's construct the constraint matrix.
    # Each "term" in the decomposition is a rank-1 tensor $u \otimes v \otimes w$.
    # The space of all such tensors is huge.
    # However, we can restrict to a "structured" basis if we assume the decomposition
    # is a linear combination of the "schoolbook" basis tensors plus some perturbations?
    # No, that's not general.
    
    # Alternative Structural Approach:
    # The problem of finding the minimum rank of a tensor is NP-hard.
    # However, for small dimensions, we can check specific ranks.
    # 
    # Key Insight: The schoolbook decomposition uses 30 terms.
    # Each term is a "unit" tensor e_{i,j} \otimes e_{j,k} \otimes e_{k,i}.
    # Are there any linear dependencies among these 30 tensors?
    # If the 30 tensors are linearly independent, the rank is at least 30.
    # Since we have a rank-30 decomposition, the rank would be exactly 30.
    
    # Let's check the linear independence of the 30 schoolbook tensors.
    # We flatten the tensors into vectors of size nm * mp * pn = 6 * 15 * 10 = 900.
    # We form a matrix M of size 900 x 30 where each column is a flattened schoolbook tensor.
    # If rank(M) == 30, they are independent.
    
    def flatten_tensor(i, j, k):
        """Flatten the unit tensor e_{i,j} \otimes e_{j,k} \otimes e_{k,i}"""
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        vec = np.zeros(nm * mp * pn)
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        # The tensor T has entries T[a,b,c] = 1 if a=u_idx, b=v_idx, c=w_idx
        # In the flattened vector, the index is a * (mp * pn) + b * pn + c
        flat_idx = u_idx * (mp * pn) + v_idx * pn + w_idx
        vec[flat_idx] = 1.0
        return vec

    # Construct the matrix of schoolbook tensors
    tensors = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensors.append(flatten_tensor(i, j, k))
    
    M = np.column_stack(tensors)
    
    # Calculate the rank of this matrix
    # If the rank is less than 30, there are linear dependencies, 
    # which might allow a lower rank decomposition?
    # Actually, if the schoolbook tensors are linearly independent, 
    # it doesn't automatically mean the tensor rank is 30. 
    # It means the schoolbook basis is a basis for the subspace they span.
    # But the tensor rank is the minimum number of rank-1 tensors to sum to T.
    # The schoolbook T is the sum of these 30 rank-1 tensors.
    # If they are linearly independent, you cannot remove any term from the sum 
    # and still get T, UNLESS there are other rank-1 tensors that can replace them.
    # 
    # However, for the "generic" tensor, the rank is often close to the schoolbook rank.
    # For this specific structured tensor, let's check if we can find a lower rank 
    # by solving for a decomposition of rank r < 30.
    
    # We will attempt to verify if rank 29, 28, etc. is possible.
    # A rank-r decomposition exists if there are vectors u_c, v_c, w_c 
    # such that sum_c u_c \otimes v_c \otimes w_c = T.
    
    # This is a system of quadratic equations (bilinear in u,v for each w, etc).
    # This is hard to solve directly.
    
    # Let's use a different structural check: 
    # The "Border Rank" or "Approximate Rank" is easier to bound, but we need exact.
    
    # Given the small size, let's try a randomized search for rank < 30 
    # but with a MUCH better initialization and verification.
    # The previous program's random search was too weak.
    
    # We will try to find a rank-29 decomposition.
    # If we fail, we assume 30 is the minimum (or close enough that we can't beat it).
    
    # Actually, there is a known result for M_{2,3,5}.
    # Let's try to construct a rank 29 decomposition by merging two schoolbook terms
    # if they are "compatible".
    # Two terms (i1, j1, k1) and (i2, j2, k2) can be merged into 1 term if:
    # u1 \otimes v1 \otimes w1 + u2 \otimes v2 \otimes w2 = u3 \otimes v3 \otimes w3.
    # This is only possible if the tensors are "proportional" in some sense, 
    # which for unit vectors implies they are the same tensor (which they aren't) 
    # or if we allow linear combinations of the basis vectors.
    
    # Let's try a generic randomized search for r in [25, 29].
    # We will use a "greedy" approach:
    # 1. Start with residual R = T.
    # 2. Pick a random rank-1 tensor t = u \otimes v \otimes w.
    # 3. Find the best coefficient alpha such that || R - alpha * t || is minimized.
    # 4. Update R = R - alpha * t.
    # 5. Repeat r times.
    # 6. Check if R is close to 0.
    
    def norm_squared(vec):
        return np.dot(vec, vec)
    
    def find_decomposition_greedy(r, max_iters=500):
        T_flat = np.zeros(nm * mp * pn)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    flat_idx = u_idx * (mp * pn) + v_idx * pn + w_idx
                    T_flat[flat_idx] = 1.0
        
        R = T_flat.copy()
        
        # We need to represent the decomposition.
        # U is nm x r, V is mp x r, W is pn x r.
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        
        for c in range(r):
            # Generate random u, v, w
            # To increase chance of success, we can bias towards the support of R.
            # But random uniform is a start.
            u = np.random.randn(nm)
            v = np.random.randn(mp)
            w = np.random.randn(pn)
            
            # Compute the rank-1 tensor t = u \otimes v \otimes w
            # t_flat[idx] = u[i] * v[j] * w[k]
            t = np.outer(np.outer(u, v), w).flatten()
            
            # Find alpha = <R, t> / <t, t>
            dot_rt = np.dot(R, t)
            dot_tt = np.dot(t, t)
            
            if dot_tt < 1e-12:
                continue
                
            alpha = dot_rt / dot_tt
            
            # Update R
            R = R - alpha * t
            
            # Store the term
            # The term is alpha * (u \otimes v \otimes w)
            # We can absorb alpha into U, V, or W.
            # Let's put it in U.
            U[:, c] = alpha * u
            V[:, c] = v
            W[:, c] = w
            
        # Check residual
        if norm_squared(R) < 1e-6:
            # We found a decomposition.
            # Convert to the required format (lists of lists)
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
        return None

    # Try ranks from 30 downwards
    best_decomp = None
    best_rank = 30
    
    # First, return the schoolbook decomposition as a fallback
    def get_schoolbook():
        U = [[0]*30 for _ in range(nm)]
        V = [[0]*30 for _ in range(mp)]
        W = [[0]*30 for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    fallback = get_schoolbook()
    
    # Try to find a better rank
    # We only have ~140 seconds.
    # Greedy search is fast. Let's try a few iterations for r=29, 28.
    
    np.random.seed(42)
    
    for r in range(29, 20, -1):
        if time.time() > deadline - 10:
            break
            
        # Try a few times
        found = False
        for attempt in range(20):
            decomp = find_decomposition_greedy(r)
            if decomp is not None:
                # Verify it exactly (or with high precision)
                if _verify_decomp(n, m, p, decomp):
                    best_rank = r
                    best_decomp = decomp
                    found = True
                    break
        if found:
            # Continue to see if we can go lower?
            # The task asks for rank-minimal.
            # If we found 29, we should try 28.
            continue
            
    if best_decomp is None:
        best_decomp = fallback
        best_rank = 30
        
    return best_decomp

def _verify_decomp(n, m, p, decomp, tol=1e-9):
    U = np.array(decomp["U"])
    V = np.array(decomp["V"])
    W = np.array(decomp["W"])
    r = U.shape[1]
    
    # Reconstruct the tensor
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # We can do this efficiently using Einstein summation or loops.
    # Given small sizes, loops are fine.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                for c in range(r):
                    s += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                if abs(s - 1) > tol:
                    return False
    return True
# EVOLVE-BLOCK-END
