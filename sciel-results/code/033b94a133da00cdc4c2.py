import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 7]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    T_comp = np.einsum('ar,br,cr->abc', U, V, W)
    T_target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0
    return np.allclose(T_comp, T_target, atol=tol)

def construct():
    start_time = time.time()
    deadline = 135  # safety margin
    
    n, m, p = DIMENSIONS
    A_dim = n * m  # 8
    B_dim = m * p  # 28
    C_dim = p * n  # 14
    
    # Build target tensor as a sparse set of (a, b, c) indices that should be 1
    target_indices = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.add((a, b, c))
    
    # Strategy: Greedy search for rank terms.
    # Each rank term is defined by:
    #   U_support: subset of {0..7}
    #   V_support: subset of {0..27}
    #   W_support: subset of {0..13}
    # The contribution to the tensor at (a,b,c) is:
    #   prod_{r} U[a,r]*V[b,r]*W[c,r]
    # where for a given rank term r, U[a,r] is 1 if a in U_support, etc.
    # But we need signs too. Let's allow coefficients in {-1, 0, 1} for simplicity,
    # but start with binary supports and assign signs later? 
    # Actually, the tensor entries are all +1. 
    # A rank term contributes to (a,b,c) iff a in U_sup, b in V_sup, c in W_sup.
    # The value contributed is U[a]*V[b]*W[c].
    # If we restrict to binary supports with value 1, then the contribution is 1 if (a,b,c) is in the Cartesian product of the supports.
    # We want the sum over all rank terms of these contributions to be 1 for target indices and 0 otherwise.
    # This is a set cover / packing problem with signs.
    
    # Simplified approach: 
    # Let's try to find a set of rank terms such that for each target (a,b,c), 
    # the number of rank terms covering it (with sign +) minus those covering it (with sign -) is 1.
    # And for non-target (a,b,c), the sum is 0.
    
    # Greedy algorithm:
    # 1. Maintain a residual tensor R, initialized to target (1s at target indices, 0 elsewhere).
    # 2. At each step, find a rank term (U_sup, V_sup, W_sup, sign) that minimizes the L1 norm of R - sign * indicator(Cartesian product).
    # 3. Add it to the decomposition and update R.
    # 4. Stop when R is zero or time limit reached.
    
    # To make this efficient, we precompute potential rank terms.
    # A rank term is defined by 3 subsets. 
    # Since dimensions are small (8, 28, 14), we can't enumerate all subsets.
    # Instead, we use a heuristic: 
    # For each target index (a,b,c) that is still unsatisfied (R[a,b,c] != 0),
    # try to find a rank term that covers it and many other unsatisfied indices, while covering few satisfied ones.
    
    # Let's define a score for a candidate rank term (U_sup, V_sup, W_sup):
    # gain = sum_{(a,b,c) in Cartesian product} (R[a,b,c] if sign=1 else -R[a,b,c])
    # We want to maximize gain for sign=1, or minimize gain for sign=-1 (i.e., maximize -gain).
    # Actually, we just want to maximize the reduction in L1 norm.
    # delta_L1 = sum_{(a,b,c) in Cartesian product} (|R[a,b,c]| - |R[a,b,c] - sign|)
    # For sign=1:
    #   if R=1, |1|-|0|=1
    #   if R=0, |0|-|-1|=1
    #   if R=-1, |-1|-|-2|=1
    # So delta_L1 is always the size of the Cartesian product! That doesn't help distinguish.
    # We need to look at the L2 norm or a different metric.
    # Let's use L2 norm squared:
    # delta_L2sq = sum_{(a,b,c) in Cartesian product} (R^2 - (R-sign)^2) = sum (2*R*sign - 1)
    # For sign=1: delta = 2*R - 1.
    #   if R=1, delta=1 (good)
    #   if R=0, delta=-1 (bad)
    #   if R=-1, delta=-3 (very bad)
    # So we want to maximize sum (2*R - 1) over the Cartesian product.
    
    # Algorithm:
    # 1. R = target tensor (sparse).
    # 2. Repeat until R is zero or time limit:
    #    a. For each unsatisfied target index (a,b,c) where R[a,b,c] != 0:
    #       - Generate candidate U_sup, V_sup, W_sup that contain a, b, c respectively.
    #       - To keep it simple, let's try small supports.
    #       - Actually, let's try a different heuristic: 
    #         Pick a random unsatisfied index (a,b,c).
    #         Try U_sup = {a}, V_sup = {b}, W_sup = {c}. This is a "unit" rank term.
    #         This will satisfy (a,b,c) but might mess up others if they were satisfied.
    #         But since we start with R=1 at targets and 0 elsewhere, a unit term at a target will:
    #           - Change R[a,b,c] from 1 to 0 (good)
    #           - Change R[a',b',c'] for other (a',b',c') in {a}x{b}x{c} (only one point) from 0 to -1 (bad if it was a target, but it's the same point).
    #         So a unit term at a target index is always a good move? 
    #         Wait, if I use a unit term at (a,b,c), it only affects that one point.
    #         So I can just use n*m*p unit terms to solve it. That's the trivial solution (rank 56).
    #         To get a lower rank, I need larger supports that cover multiple target indices.
    
    # Let's try a greedy set cover approach:
    # 1. R = target tensor.
    # 2. While R is not zero:
    #    - Find the rank term (U_sup, V_sup, W_sup) that maximizes the number of target indices it covers that are currently "positive" in R, minus the number of non-target indices it covers that are currently "negative" in R.
    #    - Actually, let's just maximize the dot product with R.
    #    - To find such a term, we can iterate over all possible U_sup, V_sup, W_sup? No, too many.
    #    - Heuristic: 
    #      * Pick a random target index (a,b,c) that is still positive in R.
    #      * Grow U_sup, V_sup, W_sup around a, b, c to maximize the score.
    
    # This is getting complex. Let's try a simpler structural change:
    # Use the known rank-50 solution but verify it works. If it does, return it.
    # If not, fall back to trivial.
    # But the task asks for a DIFFERENT strategy.
    
    # Let's implement a "randomized greedy" approach:
    # 1. R = target tensor.
    # 2. For rank in range(40, 60):
    #    - Try to find a decomposition of rank `rank` using greedy selection.
    #    - If found, return it.
    
    # Greedy selection for a fixed rank:
    # 1. R = target tensor.
    # 2. For r in range(rank):
    #    - Find the best rank term (U_sup, V_sup, W_sup, sign) that maximizes the reduction in L2 norm.
    #    - To find the best term, sample random supports of size 1-4 and evaluate.
    #    - Add the best term to the decomposition.
    # 3. If R is zero, return success.
    
    # This is a heuristic search. Let's implement it.
    
    def _greedy_search(max_rank, time_limit):
        """
        Try to find a decomposition of rank <= max_rank using greedy search.
        Returns (U, V, W, success)
        """
        start = time.time()
        R = np.zeros((A_dim, B_dim, C_dim))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    R[i*m+j, j*p+k, k*n+i] = 1.0
        
        U_terms = []
        V_terms = []
        W_terms = []
        
        for r in range(max_rank):
            if time.time() - start > time_limit:
                return None, None, None, False
            
            best_score = -np.inf
            best_U = None
            best_V = None
            best_W = None
            best_sign = 1
            
            # Sample random rank terms
            for _ in range(5000):
                # Random support sizes
                u_size = np.random.randint(1, 5)
                v_size = np.random.randint(1, 5)
                w_size = np.random.randint(1, 5)
                
                U_sup = set(np.random.choice(A_dim, u_size, replace=False))
                V_sup = set(np.random.choice(B_dim, v_size, replace=False))
                W_sup = set(np.random.choice(C_dim, w_size, replace=False))
                
                # Create the rank term tensor
                T_term = np.zeros((A_dim, B_dim, C_dim))
                for a in U_sup:
                    for b in V_sup:
                        for c in W_sup:
                            T_term[a, b, c] = 1.0
                
                # Evaluate score: dot product with R
                score = np.sum(R * T_term)
                
                # Try both signs
                if score > best_score:
                    best_score = score
                    best_U = U_sup
                    best_V = V_sup
                    best_W = W_sup
                    best_sign = 1
                elif -score > best_score:
                    best_score = -score
                    best_U = U_sup
                    best_V = V_sup
                    best_W = W_sup
                    best_sign = -1
            
            if best_U is None:
                return None, None, None, False
            
            # Add the best term
            U_term = np.zeros(A_dim)
            for a in best_U:
                U_term[a] = best_sign
            V_term = np.zeros(B_dim)
            for b in best_V:
                V_term[b] = 1.0
            W_term = np.zeros(C_dim)
            for c in best_W:
                W_term[c] = 1.0
            
            U_terms.append(U_term)
            V_terms.append(V_term)
            W_terms.append(W_term)
            
            # Update R
            R -= best_sign * np.einsum('a,b,c->abc', U_term, V_term, W_term)
            
            # Check if R is zero
            if np.allclose(R, 0, atol=1e-9):
                break
        
        if not np.allclose(R, 0, atol=1e-9):
            return None, None, None, False
        
        # Build the final matrices
        U = np.column_stack(U_terms)
        V = np.column_stack(V_terms)
        W = np.column_stack(W_terms)
        
        return U, V, W, True
    
    # Try greedy search for rank 40-55
    for rank in range(40, 56):
        if time.time() > deadline:
            break
        U, V, W, success = _greedy_search(rank, 10)
        if success:
            if _verify_decomposition(U, V, W, n, m, p):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    # Fall back to trivial
    rank = n * m * p
    U_triv = np.zeros((A_dim, rank))
    V_triv = np.zeros((B_dim, rank))
    W_triv = np.zeros((C_dim, rank))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_triv[i * m + j, r] = 1
                V_triv[j * p + k, r] = 1
                W_triv[k * n + i, r] = 1
                r += 1
    
    return {
        "U": U_triv.tolist(),
        "V": V_triv.tolist(),
        "W": W_triv.tolist()
    }

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    T = np.einsum('ar,br,cr->abc', U, V, W)
    Expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Expected[i*m + j, j*p + k, k*n + i] = 1
    
    if np.allclose(T, Expected, atol=1e-9):
        print(f"Valid decomposition with rank {U.shape[1]}")
    else:
        print("Invalid decomposition")
        diff = np.abs(T - Expected)
        print(f"Max diff: {diff.max()}")
