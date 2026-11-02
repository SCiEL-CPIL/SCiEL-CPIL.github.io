"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook baseline
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(nm)]
    V = [[0] * rank_school for _ in range(mp)]
    W = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    best = {'U': U, 'V': V, 'W': W, 'rank': rank_school}
    
    # Verify function
    def verify(U, V, W, rank, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Try to find lower rank via structured approach
    # The tensor for matrix multiplication: T[i,j,k] = 1 for all i,j,k
    # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # Strategy: Use Kronecker product structure
    # For 4x4x6, we can think of it as blocks
    # But let's try a more direct approach with known results
    
    # Known: rank of 4x4x4 matrix multiplication is at most 49 (Strassen-like)
    # For 4x4x6, we can pad to 4x4x4 and use that, or find direct decomposition
    
    # Let's try to construct a rank-49 or lower decomposition
    # Using the fact that 4x4x4 can be done with rank 49 (or less with better algorithms)
    
    # For now, let's try random search with better structure
    deadline = time.time() + 115
    random.seed(42)
    
    # Try ranks from 49 downward (since 4x4x4 has rank <= 49, 4x4x6 might be similar or slightly higher)
    # Actually, 4x4x6 should have rank at least max(16, 24, 24) = 24
    # But realistically, it's going to be around 48-60
    
    # Let's try a structured random search
    for target_rank in range(48, 65):
        if time.time() > deadline - 10:
            break
        
        found = False
        attempts = 0
        max_attempts = 2000000
        
        while attempts < max_attempts and not found and time.time() < deadline - 10:
            attempts += 1
            if attempts % 50000 == 0 and time.time() > deadline - 10:
                break
            
            # Generate random matrices with small integer entries
            U_r = [[0]*target_rank for _ in range(nm)]
            V_r = [[0]*target_rank for _ in range(mp)]
            W_r = [[0]*target_rank for _ in range(pn)]
            
            # Use structured random generation
            for c in range(target_rank):
                # Each column should have a specific pattern
                # Try to make it so that the tensor product sums correctly
                
                # Random sparse with entries in {-2, -1, 1, 2}
                for idx in range(nm):
                    if random.random() < 0.3:
                        U_r[idx][c] = random.choice([-2, -1, 1, 2])
                
                for idx in range(mp):
                    if random.random() < 0.3:
                        V_r[idx][c] = random.choice([-2, -1, 1, 2])
                
                for idx in range(pn):
                    if random.random() < 0.3:
                        W_r[idx][c] = random.choice([-2, -1, 1, 2])
            
            # Quick check on a subset
            if random.random() < 0.05:
                ok = True
                for _ in range(30):
                    i = random.randint(0, n-1)
                    j = random.randint(0, m-1)
                    k = random.randint(0, p-1)
                    s = 0
                    for c in range(target_rank):
                        s += U_r[i*m+j][c] * V_r[j*p+k][c] * W_r[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        ok = False
                        break
                if ok:
                    if verify(U_r, V_r, W_r, target_rank, n, m, p):
                        found = True
                        best = {'U': U_r, 'V': V_r, 'W': W_r, 'rank': target_rank}
                        break
    
    return {'U': best['U'], 'V': best['V'], 'W': best['W']}
# EVOLVE-BLOCK-END
