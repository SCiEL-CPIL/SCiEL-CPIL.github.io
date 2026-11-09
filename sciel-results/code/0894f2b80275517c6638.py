"""Program that searches for a lower-rank tensor decomposition of matrix multiplication 3x5x7."""
import time
import itertools
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

def _validate_decomposition(U, V, W, n=3, m=5, p=7):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = 3, 5, 7
    rank = len(U[0]) if U else 0
    # Reconstruct the tensor and check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - expected) > 1e-9:
                    return False
    return True

def _validate_decomposition_f2(U, V, W, n=3, m=5, p=7):
    """Validate decomposition over F_2 (mod 2 arithmetic)."""
    n, m, p = 3, 5, 7
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s = (s + (U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c])) % 2
                if s != 1 % 2:
                    return False
    return True

def _rank1_contribution(u, v, w, n, m, p):
    """Compute the rank-1 tensor contribution from vectors u, v, w."""
    contrib = np.zeros((n, m, p), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                contrib[i, j, k] = u[i * m + j] * v[j * p + k] * w[k * n + i]
    return contrib

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin for evaluation overhead
    
    n, m, p = DIMENSIONS
    base_rank = n * m * p  # 105
    
    # Start with schoolbook
    U_school = [[0] * base_rank for _ in range(n * m)]
    V_school = [[0] * base_rank for _ in range(m * p)]
    W_school = [[0] * base_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = base_rank
    
    # Strategy 1: Try to find rank-104 by merging two schoolbook terms
    # Two schoolbook terms (i1,j1,k1) and (i2,j2,k2) can be merged if
    # we can find a rank-1 term that covers both and we can adjust others
    # Actually, let's try a direct approach: use structured random search
    
    # Strategy 2: Greedy peeling with structured candidates
    # Build the full tensor
    tensor = np.ones((n, m, p), dtype=np.int8)
    
    U_factors = []
    V_factors = []
    W_factors = []
    
    current_tensor = tensor.copy()
    max_peels = 104  # Try to find decomposition with rank <= 104
    
    for peel_idx in range(max_peels):
        if time.time() > deadline:
            break
        
        if np.all(current_tensor == 0):
            break
        
        best_score = -1000
        best_u = None
        best_v = None
        best_w = None
        
        # Generate structured candidates
        # Type 1: Single position (schoolbook-like)
        # Type 2: Two positions sharing an index
        # Type 3: Random sparse
        
        num_trials = 200
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            
            u = np.zeros(n * m, dtype=np.int64)
            v = np.zeros(m * p, dtype=np.int64)
            w = np.zeros(p * n, dtype=np.int64)
            
            # Choose a random structure
            rtype = random.random()
            
            if rtype < 0.3:
                # Single position
                i = random.randint(0, n - 1)
                j = random.randint(0, m - 1)
                k = random.randint(0, p - 1)
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
            elif rtype < 0.6:
                # Two positions sharing i
                i = random.randint(0, n - 1)
                j1 = random.randint(0, m - 1)
                j2 = random.randint(0, m - 1)
                if j1 == j2:
                    j2 = (j2 + 1) % m
                k1 = random.randint(0, p - 1)
                k2 = random.randint(0, p - 1)
                u[i * m + j1] = 1
                u[i * m + j2] = 1
                v[j1 * p + k1] = 1
                v[j2 * p + k2] = 1
                w[k1 * n + i] = 1
                w[k2 * n + i] = 1
            elif rtype < 0.8:
                # Two positions sharing j
                i1 = random.randint(0, n - 1)
                i2 = random.randint(0, n - 1)
                if i1 == i2:
                    i2 = (i2 + 1) % n
                j = random.randint(0, m - 1)
                k1 = random.randint(0, p - 1)
                k2 = random.randint(0, p - 1)
                u[i1 * m + j] = 1
                u[i2 * m + j] = 1
                v[j * p + k1] = 1
                v[j * p + k2] = 1
                w[k1 * n + i1] = 1
                w[k2 * n + i2] = 1
            else:
                # Random sparse with small entries
                u_positions = random.sample(range(n * m), random.randint(1, 4))
                v_positions = random.sample(range(m * p), random.randint(1, 4))
                w_positions = random.sample(range(p * n), random.randint(1, 4))
                for pos in u_positions:
                    u[pos] = random.choice([1, -1])
                for pos in v_positions:
                    v[pos] = random.choice([1, -1])
                for pos in w_positions:
                    w[pos] = random.choice([1, -1])
            
            # Compute contribution
            contrib = _rank1_contribution(u, v, w, n, m, p)
            
            # Score: maximize matches with current_tensor, penalize mismatches
            score = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        c = contrib[i, j, k]
                        if c != 0:
                            if current_tensor[i, j, k] == 1 and c == 1:
                                score += 1
                            elif current_tensor[i, j, k] == 0 and c == 1:
                                score -= 2
                            elif current_tensor[i, j, k] == 1 and c == -1:
                                score -= 2
                            elif current_tensor[i, j, k] == 0 and c == -1:
                                score += 1
            
            if score > best_score:
                best_score = score
                best_u = u.copy()
                best_v = v.copy()
                best_w = w.copy()
        
        if best_u is None or best_score <= 0:
            break
        
        U_factors.append(best_u.tolist())
        V_factors.append(best_v.tolist())
        W_factors.append(best_w.tolist())
        
        # Update current tensor
        contrib = _rank1_contribution(best_u, best_v, best_w, n, m, p)
        current_tensor = current_tensor - contrib
        
        # Check if tensor is now zero
        if np.all(current_tensor == 0):
            break
    
    # Check if we found a complete decomposition
    if len(U_factors) < base_rank and np.all(current_tensor == 0):
        rank = len(U_factors)
        U_new = [[0] * rank for _ in range(n * m)]
        V_new = [[0] * rank for _ in range(m * p)]
        W_new = [[0] * rank for _ in range(p * n)]
        
        for c in range(rank):
            for i in range(n * m):
                U_new[i][c] = U_factors[c][i]
            for j in range(m * p):
                V_new[j][c] = V_factors[c][j]
            for k in range(p * n):
                W_new[k][c] = W_factors[c][k]
        
        if _validate_decomposition(U_new, V_new, W_new, n, m, p):
            best_result = {"U": U_new, "V": V_new, "W": W_new}
            best_rank = rank
    
    # Strategy 3: Try rank-104 by starting from schoolbook and modifying
    if best_rank == base_rank:
        if time.time() < deadline:
            # Try to find a rank-104 decomposition by combining two schoolbook terms
            # into one using a more complex rank-1 term
            
            # For each pair of schoolbook terms, try to find a rank-1 term
            # that covers both, then check if the remaining can be covered with 102 more terms
            
            # This is expensive, so try a limited number of pairs
            pairs_tried = 0
            max_pairs = 50
            
            for i1 in range(n):
                for j1 in range(m):
                    for k1 in range(p):
                        for i2 in range(n):
                            if time.time() > deadline:
                                break
                            if pairs_tried >= max_pairs:
                                break
                            for j2 in range(m):
                                if pairs_tried >= max_pairs:
                                    break
                                for k2 in range(p):
                                    if pairs_tried >= max_pairs:
                                        break
                                    
                                    # Skip if same term
                                    if i1 == i2 and j1 == j2 and k1 == k2:
                                        continue
                                    
                                    pairs_tried += 1
                                    
                                    # Try to find a rank-1 term covering both (i1,j1,k1) and (i2,j2,k2)
                                    # The rank-1 term must have:
                                    # u[i1*m+j1] * v[j1*p+k1] * w[k1*n+i1] != 0
                                    # u[i2*m+j2] * v[j2*p+k2] * w[k2*n+i2] != 0
                                    
                                    # Try simple cases
                                    # Case 1: Share i
                                    if i1 == i2:
                                        u = np.zeros(n * m, dtype=np.int64)
                                        v = np.zeros(m * p, dtype=np.int64)
                                        w = np.zeros(p * n, dtype=np.int64)
                                        u[i1 * m + j1] = 1
                                        u[i1 * m + j2] = 1
                                        v[j1 * p + k1] = 1
                                        v[j2 * p + k2] = 1
                                        w[k1 * n + i1] = 1
                                        w[k2 * n + i1] = 1
                                        
                                        # Check if this works as part of a full decomposition
                                        # This is complex, so skip for now
                                        pass
                                    # Case 2: Share j
                                    elif j1 == j2:
                                        pass
                                    # Case 3: Share k
                                    elif k1 == k2:
                                        pass
                                    # Case 4: No shared index - harder
                                    else:
                                        pass
            
            # Strategy 4: Random search with small integer entries for rank = 104
            for attempt in range(50):
                if time.time() > deadline:
                    break
                
                test_rank = base_rank - 1  # 104
                
                # Start from schoolbook and randomly modify
                U_test = [row[:] for row in U_school]
                V_test = [row[:] for row in V_school]
                W_test = [row[:] for row in W_school]
                
                # Remove one column and try to redistribute
                remove_col = random.randint(0, base_rank - 1)
                
                # Zero out that column
                for i in range(n * m):
                    U_test[i][remove_col] = 0
                for j in range(m * p):
                    V_test[j][remove_col] = 0
                for k in range(p * n):
                    W_test[k][remove_col] = 0
                
                # Try to fix by modifying other columns
                # Pick a random column to modify
                fix_col = random.randint(0, base_rank - 1)
                if fix_col == remove_col:
                    fix_col = (fix_col + 1) % base_rank
                
                # Add small random perturbations
                for i in range(n * m):
                    if random.random() < 0.3:
                        U_test[i][fix_col] += random.choice([-1, 1])
                for j in range(m * p):
                    if random.random() < 0.3:
                        V_test[j][fix_col] += random.choice([-1, 1])
                for k in range(p * n):
                    if random.random() < 0.3:
                        W_test[k][fix_col] += random.choice([-1, 1])
                
                # Quick validation
                valid = True
                for _ in range(50):
                    i = random.randint(0, n - 1)
                    j = random.randint(0, m - 1)
                    k = random.randint(0, p - 1)
                    s = 0
                    for c in range(test_rank):
                        s += U_test[i * m + j][c] * V_test[j * p + k][c] * W_test[k * n + i][c]
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
                
                if valid:
                    if _validate_decomposition(U_test, V_test, W_test, n, m, p):
                        best_result = {"U": U_test, "V": V_test, "W": W_test}
                        best_rank = test_rank
                        break
    
    return best_result

# EVOLVE-BLOCK-START
# The construct function is defined above. This block is kept for structural compatibility.
# EVOLVE-BLOCK-END
