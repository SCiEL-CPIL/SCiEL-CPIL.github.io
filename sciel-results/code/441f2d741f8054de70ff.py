"""Matrix multiplication tensor decomposition for 2x3 * 3x9.
Uses Kronecker product structure: decompose 2x3 x 3x3 sub-problem,
then tensor with 9-dim diagonal algebra. Also tries direct constructions.
"""
import time
import numpy as np
import itertools

DIMENSIONS = [2, 3, 9]
START_TIME = time.time()
TIME_LIMIT = 145.0  # safety margin before 150s kill


def _remaining_time():
    return TIME_LIMIT - (time.time() - START_TIME)


def _verify(U, V, W, n=2, m=3, p=9):
    """Verify tensor decomposition: T[i,k,j] = sum_t U[i*m+k,t]*V[k*p+j,t]*W[j*n+i,t] = 1"""
    rank = len(U[0])
    for i in range(n):
        for k in range(m):
            for j in range(p):
                s = 0.0
                for t in range(rank):
                    s += U[i * m + k][t] * V[k * p + j][t] * W[j * n + i][t]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _verify_small(U, V, W, n, m, p):
    """Verify for small dimensions."""
    rank = len(U[0])
    for i in range(n):
        for k in range(m):
            for j in range(p):
                s = 0.0
                for t in range(rank):
                    s += U[i * m + k][t] * V[k * p + j][t] * W[j * n + i][t]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _build_schoolbook(n=2, m=3, p=9):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][col] = 1.0
                V[k * p + j][col] = 1.0
                W[j * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}


def _try_rank27_direct(n=2, m=3, p=9):
    """Try the rank-27 construction: V=identity, U and W structured.
    
    U[i*m+k, k*p+j] = 1 for all i,k,j  (6x27)
    V = I_27
    W[j*n+i, k*p+j] = 1 for all i,j,k  (18x27)
    """
    rank = 27
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][k * p + j] = 1.0
    
    for t in range(rank):
        V[t][t] = 1.0
    
    for j in range(p):
        for i in range(n):
            for k in range(m):
                W[j * n + i][k * p + j] = 1.0
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    return None


def _search_2x3x3(rank, deadline):
    """Search for a rank-r decomposition of 2x3 x 3x3 matrix multiplication.
    
    T[i,k,j] = sum_t U[i*3+k, t] * V[k*3+j, t] * W[j*2+i, t] = 1
    U: 6 x rank, V: 9 x rank, W: 6 x rank
    """
    n, m, p = 2, 3, 3
    num_terms = n * m * p  # 18 terms to satisfy
    
    # Try structured approach: exploit symmetry
    # For 2x3 x 3x3, the tensor has some structure.
    # Let's try: can we do rank 9?
    # Lower bound is max(6, 9, 6) = 9.
    
    # Approach: try to find U, V, W with entries in {-1, 0, 1} or {-1, -0.5, 0, 0.5, 1}
    # Use random search with verification
    
    import random
    
    best = None
    attempts = 0
    while time.time() < deadline and attempts < 50000:
        attempts += 1
        if attempts % 10000 == 0 and time.time() > deadline:
            break
        
        # Random initialization with small integer/half-integer values
        vals = [-1, -0.5, 0, 0.5, 1]
        U = [[random.choice(vals) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.choice(vals) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.choice(vals) for _ in range(rank)] for _ in range(p * n)]
        
        # Quick check
        valid = True
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    s = 0.0
                    for t in range(rank):
                        s += U[i * m + k][t] * V[k * p + j][t] * W[j * n + i][t]
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            best = {"U": U, "V": V, "W": W}
            break
    
    return best


def _kronecker_decompose(sub_decomp, n=2, m=3, p_sub=3, p_full=9):
    """Tensor a 2x3 x 3x3 decomposition with the 9-dim diagonal algebra.
    
    If sub_decomp has rank r for [2,3,3], this gives rank r*9 for [2,3,9].
    
    The idea: B is 3x9. Split into 3 blocks of 3x3: B = [B1, B2, B3].
    C = A * B = A * [B1, B2, B3] = [A*B1, A*B2, A*B3].
    Each A*Bb is a 2x3 product computed via the sub-decomposition.
    
    For the tensor structure:
    - U_full[i*m+k, b*rank_sub + t] = U_sub[i*m+k, t] for block b
    - V_full[k*p_full+j, b*rank_sub + t] = V_sub[k*p_sub+j, t] if j < p_sub, else 0
      Actually, V_full[k*9+j, t_global] where t_global = b*rank_sub + t
      V_full[k*9+j, b*rank_sub+t] = V_sub[k*3+j, t] for j in 0..2
    - W_full[j*p_full... wait, W is (p*n) x rank = 18 x rank
    
    Let me think more carefully.
    
    Full problem: [2,3,9]
    U_full: 6 x (9*r)
    V_full: 27 x (9*r)  
    W_full: 18 x (9*r)
    
    Sub problem: [2,3,3]
    U_sub: 6 x r
    V_sub: 9 x r
    W_sub: 6 x r
    
    Construction:
    For each block b in {0,1,2} (corresponding to columns 3b..3b+2 of B):
      For each sub-term t in {0..r-1}:
        global_col = b*r + t
        U_full[i*3+k, global_col] = U_sub[i*3+k, t]
        V_full[k*9+j, global_col] = V_sub[k*3+j, t]  (for j in 0..2)
        W_full[j*2+i, global_col] = W_sub[j*2+i, t]  (for j in 0..2)
    
    But wait: W_full has 18 rows (j in 0..8, i in 0..1).
    For block b, the j index in the full problem is j_full = 3b + j_sub.
    W_full[j_full*2+i, global_col] = W_sub[j_sub*2+i, t]
    
    Let's verify:
    sum_t_full U_full[i*3+k, t_full] * V_full[k*9+j_full, t_full] * W_full[j_full*2+i, t_full]
    
    t_full = b*r + t_sub
    = sum_b sum_t_sub U_sub[i*3+k, t_sub] * V_sub[k*3+j_sub, t_sub] * W_sub[j_sub*2+i, t_sub]
    where j_sub = j_full - 3b
    
    But V_full[k*9+j_full, b*r+t_sub] = V_sub[k*3+j_full, t_sub] if j_full < 3, else 0?
    No, that's wrong. Let me reconsider.
    
    Actually, the issue is that V_full[k*9+j_full, global_col] should be nonzero only
    when j_full is in the correct block. Since each block b handles columns 3b..3b+2,
    V_full[k*9+j_full, b*r+t] should be V_sub[k*3+(j_full-3b), t] if 0 <= j_full-3b < 3, else 0.
    
    And W_full[j_full*2+i, b*r+t] should be W_sub[(j_full-3b)*2+i, t] if 0 <= j_full-3b < 3, else 0.
    
    Let's verify the sum:
    sum_{b,t} U_sub[i*3+k,t] * [V_sub[k*3+(j-3b),t] if 0<=j-3b<3 else 0] * [W_sub[(j-3b)*2+i,t] if 0<=j-3b<3 else 0]
    
    For a fixed j, only one b gives 0 <= j-3b < 3 (namely b = j//3, with j_sub = j%3).
    So the sum reduces to:
    sum_t U_sub[i*3+k,t] * V_sub[k*3+(j%3),t] * W_sub[(j%3)*2+i,t]
    
    But this only equals 1 if j%3 is the "right" index... no, it should equal 1 for ALL j.
    But the sub-decomposition gives sum_t U_sub[i*3+k,t]*V_sub[k*3+j_sub,t]*W_sub[j_sub*2+i,t] = 1
    for all i,k,j_sub in {0,1,2}.
    
    So for j = 3b + j_sub, the sum is:
    sum_t U_sub[i*3+k,t]*V_sub[k*3+j_sub,t]*W_sub[j_sub*2+i,t] = 1. ✓
    
    This works! The rank is 9*r where r is the sub-rank.
    """
    r = len(sub_decomp['U'][0])
    total_rank = 9 * r
    
    U_full = [[0.0] * total_rank for _ in range(n * m)]
    V_full = [[0.0] * total_rank for _ in range(m * p_full)]
    W_full = [[0.0] * total_rank for _ in range(p_full * n)]
    
    U_sub = sub_decomp['U']
    V_sub = sub_decomp['V']
    W_sub = sub_decomp['W']
    
    for b in range(3):
        for t in range(r):
            gc = b * r + t
            for i in range(n):
                for k in range(m):
                    U_full[i * m + k][gc] = U_sub[i * m + k][t]
            for k in range(m):
                for j_sub in range(p_sub):
                    j_full = b * p_sub + j_sub
                    V_full[k * p_full + j_full][gc] = V_sub[k * p_sub + j_sub][t]
            for j_sub in range(p_sub):
                for i in range(n):
                    j_full = b * p_sub + j_sub
                    W_full[j_full * n + i][gc] = W_sub[j_sub * n + i][t]
    
    return {"U": U_full, "V": V_full, "W": W_full}


def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x9 matrix multiplication.
    
    Returns:
        dict with keys 'U', 'V', 'W' - factor matrices with integer/half-integer entries
    """
    n, m, p = DIMENSIONS
    deadline = START_TIME + TIME_LIMIT - 5
    
    # Start with schoolbook as fallback
    best = _build_schoolbook(n, m, p)
    best_rank = n * m * p  # 54
    
    # Try rank-27 direct construction
    if time.time() < deadline:
        r27 = _try_rank27_direct(n, m, p)
        if r27 is not None:
            best = r27
            best_rank = 27
    
    # Try Kronecker approach: search for 2x3x3 decomposition
    # If we find rank r for [2,3,3], we get rank 9r for [2,3,9]
    # Lower bound for [2,3,3] is 9, so best Kronecker rank is 81 (worse than 54)
    # But let's try anyway in case there's a better structure
    
    # Actually, let's try a different Kronecker: use [2,3,1] x [1,1,9]
    # [2,3,1]: rank 6 (schoolbook), tensor with [1,1,9] rank 9 -> rank 54. No improvement.
    
    # Try: can we do better than 27?
    # The lower bound for [2,3,9] is max(6, 27, 18) = 27 (if non-degenerate).
    # So 27 should be optimal if the construction works.
    
    # Let's also try to verify the rank-27 construction more carefully
    # and try alternative constructions at rank 27
    
    if best_rank <= 27:
        # Try to find rank < 27 (should be impossible but verify)
        # Actually, let's try a different rank-27 construction that might be more robust
        pass
    
    # Try searching for lower rank via random methods (unlikely to succeed below 27)
    if time.time() < deadline - 10 and best_rank > 27:
        # Try rank 27 with random search
        import random
        for attempt in range(10000):
            if time.time() > deadline - 10:
                break
            rank = 27
            vals = [-1, 0, 1]
            U = [[random.choice(vals) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.choice(vals) for _ in range(rank)] for _ in range(m * p)]
            W = [[random.choice(vals) for _ in range(rank)] for _ in range(p * n)]
            
            valid = True
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        s = 0.0
                        for t in range(rank):
                            s += U[i * m + k][t] * V[k * p + j][t] * W[j * n + i][t]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best = {"U": U, "V": V, "W": W}
                best_rank = 27
                break
    
    return best


if __name__ == "__main__":
    r = construct()
    rank = len(r['U'][0])
    print(f"Found decomposition with rank {rank}")
    print(f"U shape: ({len(r['U'])}, {len(r['U'][0])})")
    print(f"V shape: ({len(r['V'])}, {len(r['V'][0])})")
    print(f"W shape: ({len(r['W'])}, {len(r['W'][0])})")
    print(f"Verified: {_verify(r['U'], r['V'], r['W'])}")
