"""Matrix multiplication 2x3x4 rank-minimal decomposition search."""
import time
import itertools
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 3, 4]

def build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array."""
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    return T

def verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    T = build_tensor(n, m, p)
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T_decomp = np.zeros_like(T)
    for r in range(rank):
        T_decomp += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.allclose(T_decomp, T, atol=1e-9)

def search_rank_decomposition(n, m, p, target_rank, deadline):
    """
    Search for a rank-target_rank decomposition using F_5 modular arithmetic.
    Returns (U, V, W) as lists of lists with integer/half-integer entries, or None.
    """
    # Work in F_5: values are 0, 1, 2, 3, 4 (mod 5)
    # Half-integers: 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4
    # In F_5, we can represent x/2 as x * 3 (since 2*3 = 6 = 1 mod 5)
    
    U_size = n * m  # 6
    V_size = m * p  # 12
    W_size = p * n  # 8
    rank = target_rank
    
    # Build the tensor constraints
    # T[a,b,c] = 1 if there exists (i,j,k) such that a=i*m+j, b=j*p+k, c=k*n+i
    # Otherwise T[a,b,c] = 0
    
    # For each (a,b,c), we need: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] (mod 5)
    
    # Precompute which (a,b,c) triples are "active" (T=1)
    active_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                active_triples.append((a, b, c))
    
    # We'll use a randomized local search approach
    # Start with random assignments and try to minimize the number of violated constraints
    
    best_U = None
    best_V = None
    best_W = None
    best_violations = float('inf')
    
    np.random.seed(42)
    
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        # Initialize randomly in F_5
        U = np.random.randint(0, 5, size=(U_size, rank), dtype=np.int32)
        V = np.random.randint(0, 5, size=(V_size, rank), dtype=np.int32)
        W = np.random.randint(0, 5, size=(W_size, rank), dtype=np.int32)
        
        # Local search: iteratively update one entry at a time to reduce violations
        for iteration in range(2000):
            if time.time() > deadline:
                break
                
            # Count violations
            violations = 0
            for (a, b, c) in active_triples:
                s = 0
                for r in range(rank):
                    s = (s + U[a, r] * V[b, r] * W[c, r]) % 5
                if s != 1:
                    violations += 1
            
            # Also check inactive triples (should sum to 0)
            # This is expensive, so we'll sample
            for _ in range(50):
                a = np.random.randint(0, U_size)
                b = np.random.randint(0, V_size)
                c = np.random.randint(0, W_size)
                # Check if this is an active triple
                is_active = (a, b, c) in active_triples
                s = 0
                for r in range(rank):
                    s = (s + U[a, r] * V[b, r] * W[c, r]) % 5
                if is_active:
                    if s != 1:
                        violations += 1
                else:
                    if s != 0:
                        violations += 1
            
            if violations < best_violations:
                best_violations = violations
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if violations == 0:
                break
            
            # Try to improve: pick a random matrix, row, and column, change the value
            mat_choice = np.random.randint(0, 3)
            if mat_choice == 0:
                row = np.random.randint(0, U_size)
                col = np.random.randint(0, rank)
                old_val = U[row, col]
                new_val = (old_val + 1) % 5
                U[row, col] = new_val
                
                # Check if this reduces violations (spot check)
                local_violations = 0
                for (a, b, c) in active_triples:
                    if a == row:
                        s = 0
                        for r in range(rank):
                            s = (s + U[a, r] * V[b, r] * W[c, r]) % 5
                        if s != 1:
                            local_violations += 1
                
                if local_violations >= violations:
                    U[row, col] = old_val
            elif mat_choice == 1:
                row = np.random.randint(0, V_size)
                col = np.random.randint(0, rank)
                old_val = V[row, col]
                new_val = (old_val + 1) % 5
                V[row, col] = new_val
                
                local_violations = 0
                for (a, b, c) in active_triples:
                    if b == row:
                        s = 0
                        for r in range(rank):
                            s = (s + U[a, r] * V[b, r] * W[c, r]) % 5
                        if s != 1:
                            local_violations += 1
                
                if local_violations >= violations:
                    V[row, col] = old_val
            else:
                row = np.random.randint(0, W_size)
                col = np.random.randint(0, rank)
                old_val = W[row, col]
                new_val = (old_val + 1) % 5
                W[row, col] = new_val
                
                local_violations = 0
                for (a, b, c) in active_triples:
                    if c == row:
                        s = 0
                        for r in range(rank):
                            s = (s + U[a, r] * V[b, r] * W[c, r]) % 5
                        if s != 1:
                            local_violations += 1
                
                if local_violations >= violations:
                    W[row, col] = old_val
        
        if best_violations == 0:
            break
    
    if best_violations != 0:
        return None
    
    # Convert from F_5 to rational numbers
    # In F_5, we used values 0-4. To get half-integers, we can interpret:
    # 0 -> 0, 1 -> 1/2, 2 -> 1, 3 -> 3/2, 4 -> 2
    # But actually, we need the product to work out correctly.
    # Let's just use the values as integers first and verify.
    
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    # Verify with rational arithmetic
    if verify_decomposition(U_list, V_list, W_list, n, m, p):
        return U_list, V_list, W_list
    
    # Try interpreting values as half-integers: val -> val/2
    U_list_half = [[v/2 for v in row] for row in best_U.tolist()]
    V_list_half = [[v/2 for v in row] for row in best_V.tolist()]
    W_list_half = [[v/2 for v in row] for row in best_W.tolist()]
    
    if verify_decomposition(U_list_half, V_list_half, W_list_half, n, m, p):
        return U_list_half, V_list_half, W_list_half
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # First, try to find a lower-rank decomposition
    # Known minimal rank for 2x3x4 is 20 (Landsberg)
    # Try ranks from 20 upward, but we want to find the minimal
    
    best_result = None
    
    # Try rank 20 first (known minimal)
    for target_rank in [20, 21, 22, 23]:
        if time.time() > deadline - 10:
            break
        
        result = search_rank_decomposition(n, m, p, target_rank, deadline - 5)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            break
    
    # Fallback: schoolbook decomposition (rank 24)
    if best_result is None:
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
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
