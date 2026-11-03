"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import itertools
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, rank
    
    best_U, best_V, best_W, best_rank = schoolbook()
    
    def verify_decomposition(U, V, W):
        """Verify that the decomposition is correct."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(len(U[0])):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        return False
        return True
    
    # Generate candidate rank-one terms
    # Each term is a triple (u_vec, v_vec, w_vec) where each is a sparse vector
    # u_vec has shape (n*m,), v_vec has shape (m*p,), w_vec has shape (p*n,)
    # For efficiency, we only consider terms with small integer coefficients
    
    candidates = []
    
    # Generate all possible sparse vectors with entries in {-1, 0, 1}
    # But limit to vectors with at most 2 non-zero entries for efficiency
    
    def generate_sparse_vectors(dim, max_nonzero=2, max_val=1):
        """Generate sparse vectors of given dimension with limited non-zero entries."""
        vecs = set()
        # All-zero vector
        vecs.add(tuple([0]*dim))
        
        if max_nonzero >= 1:
            for pos in range(dim):
                for val in range(-max_val, max_val+1):
                    if val != 0:
                        v = [0]*dim
                        v[pos] = val
                        vecs.add(tuple(v))
        
        if max_nonzero >= 2:
            for pos1 in range(dim):
                for pos2 in range(pos1+1, dim):
                    for val1 in range(-max_val, max_val+1):
                        for val2 in range(-max_val, max_val+1):
                            if val1 != 0 and val2 != 0:
                                v = [0]*dim
                                v[pos1] = val1
                                v[pos2] = val2
                                vecs.add(tuple(v))
        
        return list(vecs)
    
    # For 2x3*3x4: n*m=6, m*p=12, p*n=8
    # Generate candidates with limited sparsity
    u_vecs = generate_sparse_vectors(n*m, max_nonzero=2, max_val=1)
    v_vecs = generate_sparse_vectors(m*p, max_nonzero=2, max_val=1)
    w_vecs = generate_sparse_vectors(p*n, max_nonzero=2, max_val=1)
    
    # Filter to only keep vectors that could contribute to the tensor
    # The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1
    
    # For each candidate term, check if it's "useful" - i.e., has non-zero contribution
    # to at least one tensor entry
    
    useful_candidates = []
    for u in u_vecs:
        for v in v_vecs:
            for w in w_vecs:
                # Check if this term contributes to any tensor entry
                contributes = False
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            prod = u[i*m+j] * v[j*p+k] * w[k*n+i]
                            if prod != 0:
                                contributes = True
                                break
                        if contributes:
                            break
                    if contributes:
                        break
                if contributes:
                    useful_candidates.append((u, v, w))
    
    # Now use LP to find the minimum number of terms that sum to the target tensor
    # This is a set cover / integer programming problem
    
    # For small instances, we can try greedy or exhaustive search
    # The target tensor has n*m*p = 24 entries, all equal to 1
    
    # Let's try to find a decomposition with rank < 24
    
    def try_find_decomposition(target_rank):
        """Try to find a decomposition with exactly target_rank terms."""
        if time.time() > deadline:
            return None
        
        # Use a greedy approach: iteratively add terms that cover the most uncovered entries
        # with the correct sign
        
        # First, let's identify which tensor entries need to be covered
        # Each entry (i,j,k) needs sum of contributions = 1
        
        # For simplicity, let's try a randomized local search
        import random
        
        best_solution = None
        max_iterations = 1000
        iteration = 0
        
        while iteration < max_iterations and time.time() < deadline:
            iteration += 1
            
            # Start with a random subset of candidates
            if len(useful_candidates) < target_rank:
                return None
            
            selected = random.sample(useful_candidates, target_rank)
            
            # Check if this selection works
            # Build the contribution matrix
            contributions = {}
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        contributions[(i,j,k)] = 0
            
            valid = True
            for u, v, w in selected:
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            prod = u[i*m+j] * v[j*p+k] * w[k*n+i]
                            contributions[(i,j,k)] += prod
            
            # Check if all contributions equal 1
            for key, val in contributions.items():
                if val != 1:
                    valid = False
                    break
            
            if valid:
                # Build the decomposition
                U = [[0]*target_rank for _ in range(n*m)]
                V = [[0]*target_rank for _ in range(m*p)]
                W = [[0]*target_rank for _ in range(p*n)]
                
                for c, (u, v, w) in enumerate(selected):
                    for idx in range(n*m):
                        U[idx][c] = u[idx]
                    for idx in range(m*p):
                        V[idx][c] = v[idx]
                    for idx in range(p*n):
                        W[idx][c] = w[idx]
                
                return U, V, W
        
        return None
    
    # Try ranks from 23 down to 15 (reasonable lower bound)
    for rank in range(23, 14, -1):
        if time.time() > deadline:
            break
        
        result = try_find_decomposition(rank)
        if result is not None:
            U, V, W = result
            if verify_decomposition(U, V, W):
                return {"U": U, "V": V, "W": W}
    
    # If no better decomposition found, return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
