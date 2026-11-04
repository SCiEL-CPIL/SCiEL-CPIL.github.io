"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 12, 32, 24
    
    # Construct the matrix multiplication tensor T of shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Schoolbook baseline
    rank_school = n * m * p  # 96
    U_school = np.zeros((nm, rank_school), dtype=np.int8)
    V_school = np.zeros((mp, rank_school), dtype=np.int8)
    W_school = np.zeros((pn, rank_school), dtype=np.int8)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    def validate(U, V, W):
        """Check if U, V, W correctly decompose T."""
        rank = U.shape[1]
        if rank == 0:
            return np.all(T == 0)
        T_recon = np.zeros((nm, mp, pn), dtype=np.int64)
        for r in range(rank):
            T_recon += np.einsum('i,j,k->ijk', U[:,r].astype(np.int64), 
                                  V[:,r].astype(np.int64), W[:,r].astype(np.int64))
        return np.array_equal(T_recon, T.astype(np.int64))
    
    # Try to find a lower rank decomposition using structured approaches
    
    # Strategy 1: Split p=8 into two groups of 4
    # R(3,4,8) <= 2 * R(3,4,4)
    # If we can find R(3,4,4) < 48, we get improvement
    
    def try_find_344_decomposition(time_limit=30.0):
        """Try to find a decomposition of 3x4x4 matrix multiplication with rank < 48."""
        n2, m2, p2 = 3, 4, 4
        nm2, mp2, pn2 = n2*m2, m2*p2, p2*n2  # 12, 16, 12
        
        # Build tensor for 3x4x4
        T2 = np.zeros((nm2, mp2, pn2), dtype=np.int8)
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    T2[i*m2+j, j*p2+k, k*n2+i] = 1
        
        # Schoolbook for 3x4x4
        rank2_school = n2 * m2 * p2  # 48
        U2_school = np.zeros((nm2, rank2_school), dtype=np.int8)
        V2_school = np.zeros((mp2, rank2_school), dtype=np.int8)
        W2_school = np.zeros((pn2, rank2_school), dtype=np.int8)
        
        col = 0
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    U2_school[i*m2+j, col] = 1
                    V2_school[j*p2+k, col] = 1
                    W2_school[k*n2+i, col] = 1
                    col += 1
        
        best_U2, best_V2, best_W2 = U2_school, V2_school, W2_school
        best_rank2 = rank2_school
        
        def validate2(U, V, W):
            rank = U.shape[1]
            if rank == 0:
                return np.all(T2 == 0)
            T2_recon = np.zeros((nm2, mp2, pn2), dtype=np.int64)
            for r in range(rank):
                T2_recon += np.einsum('i,j,k->ijk', U[:,r].astype(np.int64), 
                                       V[:,r].astype(np.int64), W[:,r].astype(np.int64))
            return np.array_equal(T2_recon, T2.astype(np.int64))
        
        # Try greedy elimination: start with schoolbook, try to remove terms
        # that are linear combinations of others
        # This is a heuristic - we try random subsets to see if they span the same space
        
        if time.time() > start_time + time_limit:
            return best_U2, best_V2, best_W2, best_rank2
        
        # Try to find linear dependencies among the rank-1 terms
        # Each term is a vector in R^(nm2 * mp2 * pn2)
        # Flatten each rank-1 tensor
        terms = []
        for r in range(rank2_school):
            term = np.einsum('i,j,k->ijk', U2_school[:,r].astype(np.int64), 
                             V2_school[:,r].astype(np.int64), W2_school[:,r].astype(np.int64))
            terms.append(term.flatten())
        
        terms_np = np.array(terms, dtype=np.int64)
        
        # Check for linear dependencies using rank
        # If rank of terms_np < rank2_school, there are dependencies
        rank_of_terms = np.linalg.matrix_rank(terms_np.astype(np.float64))
        
        if rank_of_terms < rank2_school:
            # Find a minimal spanning set
            # Use greedy algorithm to find minimal set of terms that span the same space
            current_basis = []
            current_indices = []
            remaining = list(range(rank2_school))
            
            while remaining and len(current_basis) < rank_of_terms:
                best_idx = None
                best_rank = -1
                for idx in remaining:
                    if best_rank >= rank_of_terms:
                        break
                    test_basis = current_basis + [terms_np[idx].astype(np.float64)]
                    test_rank = np.linalg.matrix_rank(np.array(test_basis))
                    if test_rank > best_rank:
                        best_rank = test_rank
                        best_idx = idx
                
                if best_idx is not None:
                    current_basis.append(terms_np[best_idx].astype(np.float64))
                    current_indices.append(best_idx)
                    remaining.remove(best_idx)
            
            if len(current_indices) < rank2_school:
                # We found a smaller set - but this doesn't directly give us a decomposition
                # We need to express the removed terms as linear combinations
                # This is complex, so for now just note the potential rank reduction
                pass
        
        # Try random search: randomly perturb the decomposition and see if we can reduce rank
        # This is a heuristic approach
        
        # For now, return schoolbook if no improvement found
        return best_U2, best_V2, best_W2, best_rank2
    
    # Try to find improved 3x4x4 decomposition
    if time.time() < deadline - 20:
        U_344, V_344, W_344, rank_344 = try_find_344_decomposition(time_limit=25.0)
        
        if rank_344 < 48:
            # Combine two 3x4x4 decompositions for 3x4x8
            U_full = np.zeros((nm, 2 * rank_344), dtype=np.int8)
            V_full = np.zeros((mp, 2 * rank_344), dtype=np.int8)
            W_full = np.zeros((pn, 2 * rank_344), dtype=np.int8)
            
            # First half: k=0..3
            for r in range(rank_344):
                U_full[:, r] = U_344[:, r]
                V_full[:, r] = V_344[:, r]
                W_full[:, r] = W_344[:, r]
            
            # Second half: k=4..7, shift W indices
            for r in range(rank_344):
                U_full[:, rank_344 + r] = U_344[:, r]
                V_full[:, rank_344 + r] = V_344[:, r]
                W_shifted = np.zeros(pn, dtype=np.int8)
                for idx in range(pn):
                    if W_344[idx, r] != 0:
                        k_sub = idx // n
                        i_sub = idx % n
                        new_idx = (k_sub + 4) * n + i_sub
                        if new_idx < pn:
                            W_shifted[new_idx] = W_344[idx, r]
                W_full[:, rank_344 + r] = W_shifted
            
            if validate(U_full, V_full, W_full):
                best_U, best_V, best_W = U_full, V_full, W_full
                best_rank = 2 * rank_344
    
    # Strategy 2: Try to use structure in the 3x4x8 problem directly
    # The output has 24 entries (3*8), input has 12 (3*4) and 32 (4*8)
    # Maybe we can find a decomposition that exploits the rectangular shape
    
    # For now, return the best decomposition found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
