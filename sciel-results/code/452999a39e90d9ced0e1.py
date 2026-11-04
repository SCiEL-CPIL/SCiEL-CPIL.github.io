import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    n, m, p = 3, 3, 8
    schoolbook_rank = n * m * p  # 72

    # Initialize with schoolbook decomposition
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = schoolbook_rank

    def check_full(U, V, W, rank):
        """Check if the decomposition exactly represents 3x3x8 matrix multiplication."""
        # U: (9, rank), V: (24, rank), W: (24, rank)
        # Tensor T[i, j, k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # We want T[i, j, k] = 1 for all i,j,k
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        # Efficient check using einsum or loops
        # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This is a bit tricky to vectorize directly without large intermediate tensors.
        # Loop over i, j, k is 3*3*8 = 72 checks.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U_np[i * m + j, r] * V_np[j * p + k, r] * W_np[k * n + i, r]
                    if abs(s - 1) > 1e-10:
                        return False
        return True

    def check_333(U, V, W, rank):
        """Check 3x3x3 decomposition."""
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = 0
                    for r in range(rank):
                        s += U_np[i * 3 + j, r] * V_np[j, r] * W_np[k, r]
                    if abs(s - 1) > 1e-10:
                        return False
        return True

    def check_332(U, V, W, rank):
        """Check 3x3x2 decomposition."""
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        for i in range(3):
            for j in range(3):
                for k in range(2):
                    s = 0
                    for r in range(rank):
                        s += U_np[i * 3 + j, r] * V_np[j * 2 + k, r] * W_np[k * 3 + i, r]
                    if abs(s - 1) > 1e-10:
                        return False
        return True

    def search_333(target_rank, time_limit=10, seed=42):
        """Search for a rank decomposition of 3x3x3 using structured random search."""
        start = time.time()
        random.seed(seed)
        vals = [-1, 0, 1]
        
        for attempt in range(50000):
            if time.time() > start + time_limit:
                break
            
            U = [[random.choice(vals) for _ in range(target_rank)] for _ in range(9)]
            V = [[random.choice(vals) for _ in range(target_rank)] for _ in range(3)]
            W = [[random.choice(vals) for _ in range(target_rank)] for _ in range(3)]
            
            if check_333(U, V, W, target_rank):
                return U, V, W, target_rank
        
        return None

    def search_332(target_rank, time_limit=5, seed=43):
        """Search for a rank decomposition of 3x3x2."""
        start = time.time()
        random.seed(seed)
        vals = [-1, 0, 1]
        
        for attempt in range(50000):
            if time.time() > start + time_limit:
                break
            
            U = [[random.choice(vals) for _ in range(target_rank)] for _ in range(9)]
            V = [[random.choice(vals) for _ in range(target_rank)] for _ in range(6)]
            W = [[random.choice(vals) for _ in range(target_rank)] for _ in range(6)]
            
            if check_332(U, V, W, target_rank):
                return U, V, W, target_rank
        
        return None

    # Strategy: Try to find better decompositions for blocks
    # We need two 3x3x3 blocks and one 3x3x2 block for 3x3x8
    # Or four 3x3x2 blocks
    
    found_333 = []
    found_332 = []
    
    # Try to find rank 23, 24, 25 for 3x3x3
    for tr in [23, 24, 25]:
        if time.time() > deadline - 30:
            break
        res = search_333(tr, time_limit=min(10, deadline - 30 - (time.time() - start_time)), seed=42 + tr)
        if res:
            found_333.append(res)
            print(f"Found 3x3x3 rank {tr}")
            break  # Use the best (lowest rank) found
    
    # Try to find rank 15, 16, 17, 18 for 3x3x2
    for tr in [15, 16, 17, 18]:
        if time.time() > deadline - 15:
            break
        res = search_332(tr, time_limit=min(5, deadline - 15 - (time.time() - start_time)), seed=43 + tr)
        if res:
            found_332.append(res)
            print(f"Found 3x3x2 rank {tr}")
            break

    # Construct full decomposition if we found components
    # Case 1: Found 333 and 332
    if found_333 and found_332:
        U1, V1, W1, r1 = found_333[0]
        U2, V2, W2, r2 = found_333[0] # Reuse same decomposition for second block
        U3, V3, W3, r3 = found_332[0]
        
        # We need two independent 3x3x3 decompositions?
        # No, we can reuse the same one for both blocks if the blocks are independent.
        # The blocks correspond to disjoint sets of k indices.
        # Block 1: k=0,1,2. Block 2: k=3,4,5. Block 3: k=6,7.
        # Since the k indices are disjoint, the tensors are direct sums.
        # So we can reuse the same U, V, W structure for the first two blocks.
        
        total_rank = r1 + r2 + r3
        
        U_full = [[0] * total_rank for _ in range(9)]
        V_full = [[0] * total_rank for _ in range(24)]
        W_full = [[0] * total_rank for _ in range(24)]
        
        # Block 1: k=0,1,2
        for i in range(9):
            for r in range(r1):
                U_full[i][r] = U1[i][r]
        for j in range(3):
            for k in range(3):
                for r in range(r1):
                    V_full[j * 8 + k][r] = V1[j][r]
        for k in range(3):
            for i in range(3):
                for r in range(r1):
                    W_full[k * 3 + i][r] = W1[k][r]
        
        # Block 2: k=3,4,5
        offset1 = r1
        for i in range(9):
            for r in range(r2):
                U_full[i][offset1 + r] = U2[i][r]
        for j in range(3):
            for k in range(3):
                for r in range(r2):
                    V_full[j * 8 + (3 + k)][offset1 + r] = V2[j][r]
        for k in range(3):
            for i in range(3):
                for r in range(r2):
                    W_full[(3 + k) * 3 + i][offset1 + r] = W2[k][r]
        
        # Block 3: k=6,7
        offset2 = r1 + r2
        for i in range(9):
            for r in range(r3):
                U_full[i][offset2 + r] = U3[i][r]
        for j in range(3):
            for k in range(2):
                for r in range(r3):
                    V_full[j * 8 + (6 + k)][offset2 + r] = V3[j * 2 + k][r]
        for k in range(2):
            for i in range(3):
                for r in range(r3):
                    W_full[(6 + k) * 3 + i][offset2 + r] = W3[k * 3 + i][r]
        
        if check_full(U_full, V_full, W_full, total_rank):
            best_result = {"U": U_full, "V": V_full, "W": W_full}
            best_rank = total_rank
            print(f"Found full decomposition with rank {total_rank}")
        else:
            print("Verification failed for combined blocks")
            
    return best_result
