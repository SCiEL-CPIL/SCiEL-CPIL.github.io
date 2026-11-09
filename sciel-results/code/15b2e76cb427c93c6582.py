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

    # Strategy: Decompose 3x4x6 into 2x4x4 block (rows 0-1 of A) + 1x4x2 block (row 2 of A)
    # For 2x4x4: use Kronecker product of 2x2x2 (rank 7) with 2x2x2 (rank 7) -> rank 49? No, that's too high.
    # Actually, 2x2x2 has rank 7. 2x4x4 = (2x2) x (2x2) where first 2x2 is for (i,j) with i in {0,1}, j in {0,1}
    # and second 2x2 is for (j',k) with j' in {0,1}, k in {0,1}. But m=4, p=4 for the block.
    # Let's think differently: 2x4x4 can be seen as (2x2x2) tensor (2x2x2) via Kronecker.
    # Actually, the multiplication tensor for 2x4x4 has rank at most 2*4*4 = 32 (schoolbook).
    # Using 2x2x2 rank 7: if we can decompose 2x4x4 as a Kronecker product structure...
    # 2x4x4 = (2x2) x (4x4) where 4x4 = (2x2) x (2x2).
    # So 2x4x4 = (2x2x2) x (2x2x2) in terms of tensor structure? Not exactly.
    # 
    # Let's try a direct approach: search for a low-rank decomposition of 2x4x4 and 1x4x2.
    # 1x4x2: rank at most 8 (schoolbook). We can likely do rank 4 or less.
    # 2x4x4: rank at most 32. We can likely do better.
    
    # Known: mu(2,2,2) = 7
    # For 1x4x2: This is like multiplying a 1x4 matrix by 4x2. 
    # The tensor has entries T[i,j,k] where i in {0}, j in {0..3}, k in {0,1}.
    # T[0,j,k] = 1 if j == j and k == k... wait, the multiplication tensor for (1x4)*(4x2) -> (1x2)
    # T[i,j,k] = 1 if i is the row of output, j is the column of A / row of B, k is the column of output.
    # Actually, for matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p:
    # T[i,j,k] = 1 for all i in {0..n-1}, j in {0..m-1}, k in {0..p-1}
    # Wait no, the tensor is defined such that C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all valid (i,j,k).
    
    # For 1x4x2: n=1, m=4, p=2. T[0,j,k] = 1 for j in {0..3}, k in {0,1}.
    # This is just a sum over j. Rank is at most 4*2 = 8, but can be lower.
    # Actually, C[0,k] = sum_j A[0,j] * B[j,k]. This is a bilinear form.
    # The tensor is T[0,j,k] = 1. This is equivalent to the identity tensor for 1x4x2.
    # Rank of this tensor: it's the same as the rank of the matrix of size 4x2 if we fix i=0.
    # No, tensor rank is different. For 1xm x px1 type, it's just matrix rank.
    # But here it's 1x4x2, which is a 3-tensor.
    
    # Let's just search for both blocks.
    
    def search_block(n_b, m_b, p_b, target_rank, max_time=20.0, seed=42):
        """Search for a rank-target_rank decomposition of n_b x m_b x p_b."""
        t_start = time.time()
        import random
        random.seed(seed)
        
        def verify_block(U, V, W, rank):
            for i in range(n_b):
                for j in range(m_b):
                    for k in range(p_b):
                        expected = 1
                        actual = 0.0
                        for c in range(rank):
                            actual += U[i*m_b + j][c] * V[j*p_b + k][c] * W[k*n_b + i][c]
                        if abs(actual - expected) > 1e-9:
                            return False
            return True
        
        def count_errors_block(U, V, W, rank):
            errors = 0
            for i in range(n_b):
                for j in range(m_b):
                    for k in range(p_b):
                        actual = 0.0
                        for c in range(rank):
                            actual += U[i*m_b + j][c] * V[j*p_b + k][c] * W[k*n_b + i][c]
                        if abs(actual - 1.0) > 1e-9:
                            errors += 1
            return errors
        
        for restart in range(100):
            if time.time() - t_start > max_time:
                break
            
            U = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(n_b*m_b)]
            V = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(m_b*p_b)]
            W = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(p_b*n_b)]
            
            errors = count_errors_block(U, V, W, target_rank)
            
            for step in range(300):
                if time.time() - t_start > max_time:
                    break
                if errors == 0:
                    if verify_block(U, V, W, target_rank):
                        return U, V, W
                    break
                
                choice = random.choice(['U', 'V', 'W'])
                if choice == 'U':
                    row = random.randint(0, n_b*m_b - 1)
                    col_idx = random.randint(0, target_rank - 1)
                    old_val = U[row][col_idx]
                    new_val = old_val + random.choice([-1, 1])
                    U[row][col_idx] = new_val
                elif choice == 'V':
                    row = random.randint(0, m_b*p_b - 1)
                    col_idx = random.randint(0, target_rank - 1)
                    old_val = V[row][col_idx]
                    new_val = old_val + random.choice([-1, 1])
                    V[row][col_idx] = new_val
                else:
                    row = random.randint(0, p_b*n_b - 1)
                    col_idx = random.randint(0, target_rank - 1)
                    old_val = W[row][col_idx]
                    new_val = old_val + random.choice([-1, 1])
                    W[row][col_idx] = new_val
                    
                new_errors = count_errors_block(U, V, W, target_rank)
                if new_errors > errors:
                    if choice == 'U':
                        U[row][col_idx] = old_val
                    elif choice == 'V':
                        V[row][col_idx] = old_val
                    else:
                        W[row][col_idx] = old_val
                else:
                    errors = new_errors
        
        return None

    # Block 1: 2x4x4 (rows 0,1 of A, all 4 columns, first 4 of 6 columns of B)
    # Block 2: 1x4x2 (row 2 of A, all 4 columns, last 2 of 6 columns of B)
    # Total rank = rank_block1 + rank_block2
    
    # Try to find low rank for each block
    # 2x4x4 schoolbook rank = 32, 1x4x2 schoolbook rank = 8
    # Try ranks: for 2x4x4: 20, 22, 24, 26, 28, 30
    # For 1x4x2: 4, 5, 6, 7
    
    best_block1_rank = 32
    best_block2_rank = 8
    block1_result = None
    block2_result = None
    
    # Search for 1x4x2 first (smaller problem)
    for r2 in [4, 5, 6, 7]:
        if time.time() > deadline - 30.0:
            break
        result2 = search_block(1, 4, 2, r2, max_time=10.0, seed=100+r2)
        if result2 is not None:
            block2_result = result2
            best_block2_rank = r2
            break
    
    # Search for 2x4x4
    for r1 in [20, 22, 24, 26, 28, 30]:
        if time.time() > deadline - 15.0:
            break
        result1 = search_block(2, 4, 4, r1, max_time=15.0, seed=200+r1)
        if result1 is not None:
            block1_result = result1
            best_block1_rank = r1
            break
    
    # Assemble the full decomposition
    if block1_result is not None and block2_result is not None:
        total_rank = best_block1_rank + best_block2_rank
        U_full = [[0] * total_rank for _ in range(n*m)]
        V_full = [[0] * total_rank for _ in range(m*p)]
        W_full = [[0] * total_rank for _ in range(p*n)]
        
        U1, V1, W1 = block1_result
        U2, V2, W2 = block2_result
        
        # Block 1: rows 0,1 of A (i=0,1), all j (0..3), first 4 columns of B (k=0..3)
        for i in range(2):
            for j in range(4):
                for c in range(best_block1_rank):
                    U_full[i*m + j][c] = U1[i*4 + j][c]
        
        for j in range(4):
            for k in range(4):
                for c in range(best_block1_rank):
                    V_full[j*p + k][c] = V1[j*4 + k][c]
        
        for k in range(4):
            for i in range(2):
                for c in range(best_block1_rank):
                    W_full[k*n + i][c] = W1[k*2 + i][c]
        
        # Block 2: row 2 of A (i=2), all j (0..3), last 2 columns of B (k=4,5)
        offset = best_block1_rank
        for j in range(4):
            for c in range(best_block2_rank):
                U_full[2*m + j][offset + c] = U2[0*4 + j][c]
        
        for j in range(4):
            for k in range(2):
                for c in range(best_block2_rank):
                    V_full[j*p + 4 + k][offset + c] = V2[j*2 + k][c]
        
        for k in range(2):
            for c in range(best_block2_rank):
                W_full[(4 + k)*n + 2][offset + c] = W2[k*1 + 0][c]
        
        if verify_decomposition(U_full, V_full, W_full, total_rank):
            if total_rank < best_rank:
                best_rank = total_rank
                best_result = {"U": U_full, "V": V_full, "W": W_full}
    
    # If block approach didn't improve, try direct search
    if best_rank == rank_school and time.time() < deadline - 15.0:
        import random
        random.seed(123)
        
        for target_rank in [60, 55, 50]:
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
