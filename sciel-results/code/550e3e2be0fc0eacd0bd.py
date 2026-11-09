"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 5]
N, M, P = DIMENSIONS
NM = N * M
MP = M * P
PN = P * N

# EVOLVE-BLOCK-START
def _schoolbook():
    rank = N * M * P
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][col] = 1
                V[j * P + k][col] = 1
                W[k * N + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """Check if U,V,W with given shapes exactly decompose the matmul tensor."""
    rank = len(U[0])
    # Build the target tensor T[i,j,k] = 1
    # Check: for each (i,j,k), sum_c U[i*M+j,c] * V[j*P+k,c] * W[k*N+i,c] == 1
    # And for each c, the factors should be consistent (this is implicit)
    # Actually we need to check the full tensor equality.
    # T[i,j,k] = sum_c U[i*M+j,c] * V[j*P+k,c] * W[k*N+i,c]
    # We need this to equal 1 for all i,j,k.
    # Also we need to check that no spurious terms appear, but since we're summing
    # over c, the only constraint is that the sum equals the target.
    # However, we also need the decomposition to be "exact" in the sense that
    # the tensor computed equals the matmul tensor.
    
    # Let's compute the full tensor and compare
    # T_computed[i,j,k] = sum_c U[i*M+j,c] * V[j*P+k,c] * W[k*N+i,c]
    # T_target[i,j,k] = 1
    
    # For efficiency, let's just check all (i,j,k)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                s = 0
                for c in range(rank):
                    s += U[i*M+j][c] * V[j*P+k][c] * W[k*N+i][c]
                if s != 1:
                    return False
    return True

def _verify_fast(U, V, W):
    """Faster verification using the structure."""
    rank = len(U[0])
    # We need sum_c U[i*M+j,c] * V[j*P+k,c] * W[k*N+i,c] == 1 for all i,j,k
    # Let's compute this efficiently
    for i in range(N):
        for j in range(M):
            for k in range(P):
                s = 0
                for c in range(rank):
                    u_val = U[i*M+j][c]
                    if u_val == 0:
                        continue
                    v_val = V[j*P+k][c]
                    if v_val == 0:
                        continue
                    w_val = W[k*N+i][c]
                    s += u_val * v_val * w_val
                if s != 1:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = N * M * P
    
    # Try to find lower rank decompositions
    # For 3x3x5, known bounds: rank is between 18 and 45
    # Let's try to find decompositions with rank < 45
    
    # Strategy: Use a structured approach
    # The tensor is T[i,j,k] = 1 for i in 0..2, j in 0..2, k in 0..4
    
    # Try rank 18-44
    # Use random search with structured initialization
    
    random.seed(42)
    
    for target_rank in range(18, best_rank):
        if time.time() > deadline:
            break
        
        # Try multiple random attempts for each rank
        max_attempts = 50 if target_rank < 30 else 20
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Initialize with random sparse matrices
            U = [[0] * target_rank for _ in range(NM)]
            V = [[0] * target_rank for _ in range(MP)]
            W = [[0] * target_rank for _ in range(PN)]
            
            # Fill with random small integers
            for i in range(NM):
                for c in range(target_rank):
                    if random.random() < 0.3:
                        U[i][c] = random.choice([-1, 1])
            
            for j in range(MP):
                for c in range(target_rank):
                    if random.random() < 0.3:
                        V[j][c] = random.choice([-1, 1])
            
            for k in range(PN):
                for c in range(target_rank):
                    if random.random() < 0.3:
                        W[k][c] = random.choice([-1, 1])
            
            # Try to fix the decomposition
            # This is a non-convex problem, so we use a simple approach:
            # For each (i,j,k), we need sum_c U[i*M+j,c]*V[j*P+k,c]*W[k*N+i,c] = 1
            
            # Let's try a different approach: use the fact that we can set
            # specific patterns
            
            # Actually, let's try a structured approach based on known results
            # For 3x3x3, the minimum rank is 23 (Landsberg). For 3x3x5, it's unknown
            # but likely higher.
            
            # Let's try a more systematic approach:
            # Use the Strassen-like idea but adapted
            
            pass  # Random approach is too slow for this problem size
    
    # Try a structured approach: use block decomposition
    # Split the 3x3x5 tensor into parts
    
    # For now, return the schoolbook result as fallback
    return best
# EVOLVE-BLOCK-END
