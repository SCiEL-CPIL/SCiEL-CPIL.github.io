"""Matrix multiplication 2,3,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import random

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit
    
    # Schoolbook rank = n*m*p = 30
    rank_schoolbook = n * m * p
    
    def make_schoolbook():
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = make_schoolbook()
    best_rank = rank_schoolbook
    
    def verify(U, V, W, rank):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for ip in range(n):
                        for jp in range(m):
                            for kp in range(p):
                                expected = 1 if (j == jp and ip == i and kp == k) else 0
                                actual = 0
                                for r in range(rank):
                                    actual += U[i*m+j][r] * V[jp*p+kp][r] * W[k*n+ip][r]
                                if actual != expected:
                                    return False
        return True
    
    # Verify schoolbook
    assert verify(best_U, best_V, best_W, best_rank)
    
    # Try to find a lower rank decomposition using a structured search.
    # We'll use a randomized approach with a few restarts, trying to find
    # a decomposition with rank < 30.
    
    # For small ranks, we can try to solve the system of equations directly.
    # The tensor T has entries T[i,j,k; i',j',k'] = 1 if (j==j', i'==i, k'==k) else 0.
    # We want to find U, V, W such that T = sum_r U_r ⊗ V_r ⊗ W_r
    
    # Let's try rank 29, 28, ... down to some minimum.
    # For each rank, we'll try a few random initializations.
    
    def try_random_rank(target_rank, num_trials=5):
        """Try to find a decomposition with the given rank using random initialization."""
        for trial in range(num_trials):
            # Random initialization with small values
            U = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Check if this happens to be a valid decomposition (unlikely for random)
            # For small ranks, we might be able to solve it directly.
            # But for now, let's just verify and return if valid.
            if verify(U, V, W, target_rank):
                return U, V, W
        return None
    
    # Try ranks from 29 down to 25
    for target_rank in range(rank_schoolbook - 1, 24, -1):
        if time.time() > deadline:
            break
        result = try_random_rank(target_rank, num_trials=3)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = target_rank
            break
    
    # Final verification
    if not verify(best_U, best_V, best_W, best_rank):
        # Fallback to schoolbook
        best_U, best_V, best_W = make_schoolbook()
        best_rank = rank_schoolbook
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
