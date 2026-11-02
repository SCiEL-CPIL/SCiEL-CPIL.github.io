import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target multiplication tensor T_mult as a dict of nonzero entries
    # T_mult[a, b, c] = 1 where a = i*m+j, b = j*p+k, c = k*n+i
    # All other entries are 0
    target_nonzeros = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_nonzeros[(a, b, c)] = 1
    
    # Trivial decomposition (Rank 45)
    rank_trivial = n * m * p
    U_triv = np.zeros((nm, rank_trivial))
    V_triv = np.zeros((mp, rank_trivial))
    W_triv = np.zeros((pn, rank_trivial))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_triv[i * m + j, col] = 1
                V_triv[j * p + k, col] = 1
                W_triv[k * n + i, col] = 1
                col += 1
    
    def verify_decomposition(U, V, W, rank):
        """Verify if U,V,W with given rank exactly decompose the multiplication tensor."""
        # Check all target nonzero positions
        for (a, b, c) in target_nonzeros.keys():
            val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
            if abs(val - 1) > 1e-9:
                return False
        # Check a sample of zero positions to ensure no cross-talk
        # We need to check that positions NOT in target_nonzeros give 0
        # Sample some random zero positions
        all_positions = set()
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    all_positions.add((a, b, c))
        zero_positions = all_positions - set(target_nonzeros.keys())
        zero_list = list(zero_positions)
        # Check up to 100 random zero positions
        random.shuffle(zero_list)
        for pos in zero_list[:100]:
            a, b, c = pos
            val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
            if abs(val) > 1e-9:
                return False
        return True
    
    # Try to find lower rank decompositions using structured random search
    # Strategy: use small integer coefficients and try to satisfy the equations
    
    best_result = {"U": U_triv, "V": V_triv, "W": W_triv}
    best_rank = rank_trivial
    
    # Try ranks from 20 up to 44
    for rank in range(20, rank_trivial):
        if time.time() > deadline:
            break
        
        # Try multiple random attempts for this rank
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Generate random factor matrices with small integer entries
            # Use a structured approach: random sparse matrices with small coefficients
            U = np.zeros((nm, rank))
            V = np.zeros((mp, rank))
            W = np.zeros((pn, rank))
            
            # Fill with random small integers (-1, 0, 1, 2)
            for r in range(rank):
                # Each column is a random vector
                U[:, r] = np.random.randint(-2, 3, size=nm)
                V[:, r] = np.random.randint(-2, 3, size=mp)
                W[:, r] = np.random.randint(-2, 3, size=pn)
            
            # Verify
            if verify_decomposition(U, V, W, rank):
                best_rank = rank
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break  # Found valid decomposition for this rank
        
        # If we found a valid decomposition at this rank, we can stop
        # since we're going from low to high rank
        if best_rank == rank:
            break
    
    return best_result
