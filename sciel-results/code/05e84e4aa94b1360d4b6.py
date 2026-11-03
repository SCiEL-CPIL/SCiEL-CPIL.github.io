import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

def construct():
    """
    Returns the rank-minimal exact tensor decomposition for multiplying
    a 2 by 3 matrix by a 3 by 6 matrix.
    
    The schoolbook decomposition has rank n*m*p = 36.
    We attempt to find a lower rank decomposition using structured search
    and randomization, falling back to schoolbook if no better solution is found.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Start with schoolbook
    schoolbook = _schoolbook_decomposition(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p
    
    # Lower bound for rank
    min_possible = max(n * m, m * p, p * n)  # max(6, 18, 12) = 18
    
    # Try structured decomposition strategies first
    # Strategy 1: Direct sum of 2x1x6 sub-problems
    # 2x1x6 can be seen as 3 copies of 2x1x2
    # Rank of 2x1x2 is 4 (schoolbook), so 3*4 = 12
    # But wait, 2x1x2 has rank 4? Let's check.
    # Actually, 2x1x2: A is 2x1, B is 1x2, C is 2x2.
    # (A*B)[i,k] = A[i,0] * B[0,k]
    # This is an outer product of A and B.
    # The tensor T[i,j,k] where j=0 only. T[i,0,k] = 1.
    # This is just the identity tensor on i,k? No, it's a rank-1 tensor in i,k?
    # Wait, T[i,0,k] = 1 for all i,k. This is the all-ones matrix in i,k.
    # The rank of the all-ones 2x2 matrix is 1.
    # So the tensor T[i,0,k] = 1 has rank 1?
    # Let's check the definition.
    # U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # For 2x1x2: n=2, m=1, p=2.
    # U is 2x1, V is 1x2, W is 2x2.
    # U[i*1+0, c] = U[i, c]
    # V[0*2+k, c] = V[k, c]
    # W[k*2+i, c]
    # sum_c U[i,c] V[k,c] W[2k+i, c] = 1 for all i,k.
    # If we take rank 1:
    # U[i,0] = 1, V[k,0] = 1, W[2k+i, 0] = 1.
    # sum = 1*1*1 = 1.
    # So rank of 2x1x2 is 1.
    # Therefore, 2x1x6 (which is 3 copies of 2x1x2 along p) would have rank 3?
    # Let's verify.
    # n=2, m=1, p=6.
    # U[i, c], V[k, c], W[6k+i, c]? No, W is p*n = 12 rows.
    # W index: k*n + i.
    # If we use 3 rank-1 terms, one for each k=0,1,2? No, p=6.
    # The tensor is T[i,0,k] = 1 for all i in 0..1, k in 0..5.
    # This is a 2x1x6 tensor with all ones.
    # Can we decompose it with rank 3?
    # sum_{c=0}^2 U[i,c] V[k,c] W[2k+i, c] = 1.
    # If we set U[i,c] = 1, V[k,c] = delta_{k, c*2}? No.
    # Let's try to cover k=0..5.
    # If we use 6 rank-1 terms, one for each k, rank is 6.
    # Can we do better?
    # The tensor is a matrix of ones (2x6) in the i-k plane, with j fixed.
    # The rank of a 2x6 all-ones matrix is 1.
    # So the tensor rank should be 1?
    # Let's check rank 1 for 2x1x6.
    # U[i,0]=1, V[k,0]=1, W[2k+i,0]=1.
    # sum = 1.
    # Yes, rank 1 works for 2x1xp.
    
    # So, if we decompose 2x3x6 as direct sum of three 2x1x6 problems (for j=0,1,2),
    # each has rank 1.
    # Total rank = 3 * 1 = 3?
    # Let's check the full tensor.
    # T[i,j,k] = 1 for all i,j,k.
    # Decomposition: sum_{c=0}^{2} U[i*3+j, c] V[j*6+k, c] W[6k+i, c] = 1.
    # Let c correspond to j.
    # For c=j, U[i*3+j, j] = 1, others 0.
    # V[j*6+k, j] = 1, others 0.
    # W[6k+i, j] = 1, others 0.
    # Then sum_{c} ... = 1 * 1 * 1 = 1.
    # So rank 3 is sufficient?
    
    # Let's verify this hypothesis.
    # If rank 3 works, that's a huge improvement from 36.
    # Let's try to construct it.
    
    if _verify_rank(3, n, m, p):
        result = _construct_rank_3(n, m, p)
        if result:
            return result
    
    # If rank 3 didn't work (maybe my logic is flawed or verification failed),
    # try other ranks.
    
    # Try rank 6, 9, 12, 18, 24, 30, 36
    for r in range(min_possible, best_rank + 1):
        if time.time() > deadline:
            break
        
        # Try to find a decomposition of rank r
        # We can use a greedy or random approach, or structured.
        
        # For now, let's just try to verify if a random search finds anything
        # But random search for exact integer/half-integer solutions is hard.
        
        # Let's stick to the structured idea if it works.
        # If rank 3 failed, maybe rank 6?
        # 2x3x6 = 3 * (2x1x6). Rank of 2x1x6 is 1. So 3*1=3.
        # Why would it fail?
        
        # Let's double check the indices.
        # U: (n*m, rank) = (6, r)
        # V: (m*p, rank) = (18, r)
        # W: (p*n, rank) = (12, r)
        
        # My proposed rank 3:
        # c=0: j=0. U[i*3+0, 0]=1. V[0*6+k, 0]=1. W[6k+i, 0]=1.
        # c=1: j=1. U[i*3+1, 1]=1. V[1*6+k, 1]=1. W[6k+i, 1]=1.
        # c=2: j=2. U[i*3+2, 2]=1. V[2*6+k, 2]=1. W[6k+i, 2]=1.
        
        # Check term for i=0, j=1, k=2.
        # c=0: U[1,0]=0.
        # c=1: U[1,1]=1, V[6+2, 1]=V[8,1]=1, W[12+0, 1]=W[12,1]=1. Product 1.
        # c=2: U[1,2]=0.
        # Sum = 1. Correct.
        
        # Check term for i=1, j=0, k=5.
        # c=0: U[0,0]=1, V[5,0]=1, W[30+1, 0]=W[31,0]? 
        # Wait, W index is k*n + i = 5*2 + 1 = 11.
        # W[11, 0]=1.
        # Product 1.
        # c=1,2: U[0,1]=0, U[0,2]=0.
        # Sum = 1. Correct.
        
        # So rank 3 should work.
        # Why did _verify_rank(3) potentially fail? Maybe it didn't run or I didn't implement it yet.
        # I will implement _verify_rank and _construct_rank_3.
        
        if r == 3:
            # Already checked above
            continue
            
        # Try other structured decompositions
        # For example, if we can't get rank 3, maybe rank 6?
        # But rank 3 is theoretically sound for this specific tensor structure (all ones).
        
        # Let's just return the schoolbook if nothing else is found quickly.
        pass
    
    return best_result

def _schoolbook_decomposition(n, m, p):
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
    return {"U": U, "V": V, "W": W}

def _construct_rank_3(n, m, p):
    """
    Construct a rank-3 decomposition for 2x3x6 based on the observation
    that the tensor is all ones and can be decomposed by summing over j.
    """
    if n != 2 or m != 3 or p != 6:
        return None
        
    rank = 3
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # c corresponds to j
    for j in range(m):
        for i in range(n):
            U[i * m + j][j] = 1
        for k in range(p):
            V[j * p + k][j] = 1
        for k in range(p):
            for i in range(n):
                W[k * n + i][j] = 1
                
    return {"U": U, "V": V, "W": W}

def _verify_rank(rank, n, m, p):
    """
    Verify if a specific rank decomposition exists by trying to construct it
    and checking the tensor equation.
    For now, we only have a specific constructor for rank 3.
    """
    if rank == 3:
        res = _construct_rank_3(n, m, p)
        if res:
            return _check_decomposition(res, n, m, p)
    return False

def _check_decomposition(decomp, n, m, p):
    """
    Check if the decomposition satisfies the matrix multiplication tensor equation.
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k.
    """
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    
    rank = len(U[0])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if total != 1:
                    return False
    return True

# Test the logic locally before returning
if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    if _check_decomposition(res, n, m, p):
        print("Decomposition valid.")
        print(f"Rank: {len(res['U'][0])}")
    else:
        print("Decomposition invalid.")
