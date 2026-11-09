"""Matrix multiplication 3x4x6 - rank minimization via block structure + search."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 145.0  # safety margin before 150s kill

    n, m, p = 3, 4, 6
    # Schoolbook: rank = n*m*p = 72
    rank_school = n * m * p

    def verify_decomposition(U, V, W, rank):
        """Check if U, V, W correctly decompose the multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True

    def count_errors(U, V, W, rank):
        """Count how many (i,j,k) triples are incorrectly decomposed."""
        errors = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    if abs(actual - 1.0) > 1e-9:
                        errors += 1
        return errors

    # Start with schoolbook
    U_sb = [[0] * rank_school for _ in range(n * m)]
    V_sb = [[0] * rank_school for _ in range(m * p)]
    W_sb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    best_rank = rank_school
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}

    # Try block structure: split p=6 into 3 blocks of size 2
    # For each block, use a decomposition of 3x4x2
    # If we can find rank r for 3x4x2, total rank = 3*r
    # Schoolbook for 3x4x2 is 24, so we'd need r < 24 to improve
    
    # Let's try to find a good rank for 3x4x2 via search
    # Known: mu(3,4,2) <= 12 (schoolbook), but can we do better?
    
    def search_3x4x2(target_rank, max_time=30.0):
        """Search for a rank-target_rank decomposition of 3x4x2."""
        t_start = time.time()
        import random
        random.seed(42)
        
        n2, m2, p2 = 3, 4, 2
        
        def verify_3x4x2(U, V, W, rank):
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        expected = 1
                        actual = 0.0
                        for c in range(rank):
                            actual += U[i*m2 + j][c] * V[j*p2 + k][c] * W[k*n2 + i][c]
                        if abs(actual - expected) > 1e-9:
                            return False
            return True
        
        def count_errors_3x4x2(U, V, W, rank):
            errors = 0
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        actual = 0.0
                        for c in range(rank):
                            actual += U[i*m2 + j][c] * V[j*p2 + k][c] * W[k*n2 + i][c]
                        if abs(actual - 1.0) > 1e-9:
                            errors += 1
            return errors
        
        # Try random initialization with local search
        for restart in range(50):
            if time.time() - t_start > max_time:
                break
            
            # Initialize with small integers
            U = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(n2*m2)]
            V = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(m2*p2)]
            W = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(p2*n2)]
            
            errors = count_errors_3x4x2(U, V, W, target_rank)
            
            for step in range(200):
                if time.time() - t_start > max_time:
                    break
                if errors == 0:
                    if verify_3x4x2(U, V, W, target_rank):
                        return U, V, W
                    break
                
                # Pick a random entry to modify
                choice = random.choice(['U', 'V', 'W'])
                if choice == 'U':
                    row = random.randint(0, n2*m2 - 1)
                    col_idx = random.randint(0, target_rank - 1)
                    old_val = U[row][col_idx]
                    new_val = old_val + random.choice([-1, 1])
                    U[row][col_idx] = new_val
                elif choice == 'V':
                    row = random.randint(0, m2*p2 - 1)
                    col_idx = random.randint(0, target_rank - 1)
                    old_val = V[row][col_idx]
                    new_val = old_val + random.choice([-1, 1])
                    V[row][col_idx] = new_val
                else:
                    row = random.randint(0, p2*n2 - 1)
                    col_idx = random.randint(0, target_rank - 1)
                    old_val = W[row][col_idx]
                    new_val = old_val + random.choice([-1, 1])
                    W[row][col_idx] = new_val
                    
                new_errors = count_errors_3x4x2(U, V, W, target_rank)
                if new_errors > errors:
                    # Revert
                    if choice == 'U':
                        U[row][col_idx] = old_val
                    elif choice == 'V':
                        V[row][col_idx] = old_val
                    else:
                        W[row][col_idx] = old_val
                else:
                    errors = new_errors
        
        return None

    # Try to find rank 8, 9, 10 for 3x4x2
    for target_rank in [8, 9, 10, 11]:
        if time.time() > deadline - 10.0:
            break
        
        result = search_3x4x2(target_rank, max_time=20.0)
        if result is not None:
            U2, V2, W2 = result
            # Build full 3x4x6 decomposition
            full_rank = 3 * target_rank
            U_full = [[0] * full_rank for _ in range(n*m)]
            V_full = [[0] * full_rank for _ in range(m*p)]
            W_full = [[0] * full_rank for _ in range(p*n)]
            
            for block in range(3):
                # Block uses columns block*target_rank to (block+1)*target_rank
                for i in range(n):
                    for j in range(m):
                        for c in range(target_rank):
                            U_full[i*m + j][block*target_rank + c] = U2[i*m2 + j][c]
                
                for j in range(m):
                    for k in range(p2):
                        for c in range(target_rank):
                            V_full[j*p + block*p2 + k][block*target_rank + c] = V2[j*p2 + k][c]
                
                for k in range(p2):
                    for i in range(n):
                        for c in range(target_rank):
                            W_full[(block*p2 + k)*n + i][block*target_rank + c] = W2[k*n2 + i][c]
            
            if verify_decomposition(U_full, V_full, W_full, full_rank):
                if full_rank < best_rank:
                    best_rank = full_rank
                    best_result = {"U": U_full, "V": V_full, "W": W_full}
                break

    # If block approach didn't work, try direct search on full problem
    if best_rank == rank_school and time.time() < deadline - 15.0:
        import random
        random.seed(123)
        
        for target_rank in [60, 55, 50, 45]:
            if time.time() > deadline - 10.0:
                break
            
            U = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(p*n)]
            
            errors = count_errors(U, V, W, target_rank)
            
            for step in range(500):
                if time.time() > deadline - 10.0:
                    break
                if errors == 0:
                    if verify_decomposition(U, V, W, target_rank):
                        if target_rank < best_rank:
                            best_rank = target_rank
                            best_result = {"U": U, "V": V, "W": W}
                    break
                
                choice = random.choice(['U', 'V', 'W'])
                if choice == 'U':
                    row = random.randint(0, n*m - 1)
                    col_idx = random.randint(0, target_rank - 1)
                    old_val = U[row][col_idx]
                    new_val = old_val + random.choice([-1, 1])
                    U[row][col_idx] = new_val
                elif choice == 'V':
                    row = random.randint(0, m*p - 1)
                    col_idx = random.randint(0, target_rank - 1)
                    old_val = V[row][col_idx]
                    new_val = old_val + random.choice([-1, 1])
                    V[row][col_idx] = new_val
                else:
                    row = random.randint(0, p*n - 1)
                    col_idx = random.randint(0, target_rank - 1)
                    old_val = W[row][col_idx]
                    new_val = old_val + random.choice([-1, 1])
                    W[row][col_idx] = new_val
                    
                new_errors = count_errors(U, V, W, target_rank)
                if new_errors > errors:
                    if choice == 'U':
                        U[row][col_idx] = old_val
                    elif choice == 'V':
                        V[row][col_idx] = old_val
                    else:
                        W[row][col_idx] = old_val
                else:
                    errors = new_errors
            
            if best_rank < rank_school:
                break

    return best_result
# EVOLVE-BLOCK-END
