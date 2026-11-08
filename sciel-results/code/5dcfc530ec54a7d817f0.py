"""Tensor decomposition for 3x4 x 4x6 matrix multiplication."""
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin: 120s limit, use 120s
    
    n, m, p = DIMENSIONS
    # T[i, j, k] = 1 if output position (i,j) depends on A[i,j]*B[j,k]
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Convention: W[k*n + i, r]
    
    # The tensor T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Schoolbook rank = n*m*p = 72
    schoolbook_rank = n * m * p
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    # Start with structured approaches, then random search
    
    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if total != 1:
                        return False
        return True
    
    def try_rank(rank):
        """Try to find a decomposition of given rank using random search."""
        nonlocal best_result, best_rank, deadline
        
        # Random initialization with small integers
        for attempt in range(200):
            if time.time() > deadline - 5:
                return False
            
            # Generate random matrices with small integer entries
            U = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(n*m)]
            V = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(m*p)]
            W = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(p*n)]
            
            # Verify
            if verify(U, V, W, rank):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                return True
        return False
    
    # First, ensure we have the schoolbook solution as fallback
    U_school = [[0] * schoolbook_rank for _ in range(n*m)]
    V_school = [[0] * schoolbook_rank for _ in range(m*p)]
    W_school = [[0] * schoolbook_rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = schoolbook_rank
    
    # Try structured lower-rank approaches
    # The minimum rank for matrix multiplication is known to be at least max(n*m, m*p, p*n)
    # For 3x4x6: max(12, 24, 18) = 24
    
    # Try ranks from lower bound up to schoolbook
    # Known lower bound: at least max(n*m, m*p, p*n) = 24
    # But we can try various ranks
    
    # Try some specific structured decompositions
    # For rectangular matrix multiplication, there are known results
    
    # Let's try a systematic approach with small ranks
    # The rank must satisfy certain constraints
    
    # Try rank values that might work
    candidate_ranks = []
    for r in range(max(n*m, m*p, p*n), min(schoolbook_rank, max(n*m, m*p, p*n) + 20)):
        candidate_ranks.append(r)
    
    # Also try some specific values
    for r in [24, 25, 26, 27, 28, 29, 30]:
        if r not in candidate_ranks and r < schoolbook_rank:
            candidate_ranks.append(r)
    
    candidate_ranks = sorted(set(candidate_ranks))
    
    for rank in candidate_ranks:
        if time.time() > deadline - 10:
            break
        if try_rank(rank):
            break  # Found a valid decomposition at this rank
    
    # If we found a better result, use it
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
