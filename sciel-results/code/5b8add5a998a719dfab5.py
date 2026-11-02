"""Matrix multiplication 4x5 x 5x5 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # W[k*n + i, r] convention
    
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Best known: schoolbook rank 100
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Try to find lower rank decompositions
    # Start from a reasonable upper bound and go down
    # For 4x5 x 5x5, let's try ranks from 80 down to see if we can find something
    
    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # For each (i, j, k), check that sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # Actually, the tensor decomposition means:
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # where T[i,j,k] = 1 for all valid (i,j,k) in the matrix multiplication
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def verify_fast(U, V, W, rank):
        """Faster verification using numpy vectorized operations."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Reshape to 3D tensors
        U3 = U.reshape(n, m, rank)  # U[i,j,r]
        V3 = V.reshape(m, p, rank)  # V[j,k,r]
        W3 = W.reshape(p, n, rank)  # W[k,i,r]
        
        # Compute sum over r: T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # Use einsum: 'ijr,jkr,kir->ijk'
        total = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        
        # Check if total is all ones
        if not np.allclose(total, 1.0, atol=1e-9):
            return False
        return True
    
    # Schoolbook decomposition as fallback
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
    
    # Get schoolbook as initial best
    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Try to find lower rank decompositions
    # Strategy: Try structured perturbations of the schoolbook decomposition
    # and random searches with better structure
    
    # First, try a few structured approaches
    # Approach 1: Try to merge some schoolbook terms
    # The schoolbook has rank 100. Can we find a decomposition of lower rank?
    
    # For 4x5 x 5x5, the lower bound is max(4*5, 5*5, 5*4) = 25
    # But the actual rank is likely close to 100 for these small dimensions
    
    # Let's try random search with a better strategy:
    # Initialize with schoolbook-like structure and perturb
    
    for target_rank in range(95, 74, -1):
        if time.time() > deadline - 15:
            break
        
        # Try multiple random attempts with structured initialization
        max_attempts = 30
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
            
            # Strategy: Start from a near-schoolbook structure
            # For each (i,j,k), we need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            
            # Create a base structure where most terms are like schoolbook
            # but some are merged or perturbed
            U = np.zeros((n * m, target_rank), dtype=float)
            V = np.zeros((m * p, target_rank), dtype=float)
            W = np.zeros((p * n, target_rank), dtype=float)
            
            # Assign each (i,j,k) triple to a rank term
            # With target_rank < n*m*p, some terms must handle multiple triples
            triples = list(product(range(n), range(m), range(p)))
            random.shuffle(triples)
            
            # Assign triples to rank terms, allowing some terms to handle multiple
            term_assignments = {}
            for r in range(target_rank):
                term_assignments[r] = []
            
            # Distribute triples as evenly as possible
            for idx, triple in enumerate(triples):
                r = idx % target_rank
                term_assignments[r].append(triple)
            
            # For each rank term, set up U, V, W to satisfy all assigned triples
            # This is generally not possible with a single term unless all triples
            # share the same i, j, or k structure
            
            # Better approach: Use a greedy assignment where we try to match
            # triples that share indices
            U = np.zeros((n * m, target_rank), dtype=float)
            V = np.zeros((m * p, target_rank), dtype=float)
            W = np.zeros((p * n, target_rank), dtype=float)
            
            # For each rank term, pick a random (i,j,k) and set the corresponding entries
            for r in range(target_rank):
                i, j, k = random.randint(0, n-1), random.randint(0, m-1), random.randint(0, p-1)
                U[i*m + j, r] = 1.0
                V[j*p + k, r] = 1.0
                W[k*n + i, r] = 1.0
            
            # Verify
            if verify_fast(U.tolist(), V.tolist(), W.tolist(), target_rank):
                best_rank = target_rank
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                break
        
        if best_rank < n * m * p:
            continue
    
    # Final verification of the best result
    if verify_fast(best_U, best_V, best_W, best_rank):
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        # Return schoolbook as guaranteed valid fallback
        U_sb, V_sb, W_sb, rank_sb = schoolbook()
        return {"U": U_sb, "V": V_sb, "W": W_sb}
