import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 180
    
    # Precompute: for each a, dict of b -> c where 1/a = 1/b + 1/c, a<b, c integer in [1,N], c!=a,b
    a_to_bc = [dict() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            den = b - a
            num = a * b
            if num % den == 0:
                c = num // den
                if c != a and c != b and 1 <= c <= N:
                    a_to_bc[a][b] = c
    
    def is_valid(S):
        S_set = set(S)
        for a in S:
            if a in a_to_bc:
                for b, c in a_to_bc[a].items():
                    if b in S_set and c in S_set:
                        return False
        return True
    
    # Find largest valid suffix
    K = N
    suffix = set(range(K, N + 1))
    while K > 1:
        new_K = K - 1
        valid_add = True
        for b in range(new_K + 1, N + 1):
            if b in a_to_bc[new_K]:
                c = a_to_bc[new_K][b]
                if c in suffix:
                    valid_add = False
                    break
        if valid_add:
            K = new_K
            suffix.add(K)
        else:
            break
    
    best_set = set(suffix)
    
    # Greedily add elements from 1 to K-1
    for candidate in range(1, K):
        if time.time() > deadline:
            break
        if is_valid(best_set | {candidate}):
            best_set.add(candidate)
    
    # Local search with time limit
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding any missing element
        for candidate in range(1, N + 1):
            if time.time() > deadline:
                break
            if candidate not in best_set and is_valid(best_set | {candidate}):
                best_set.add(candidate)
                improved = True
                break
    
    return list(best_set)
# EVOLVE-BLOCK-END
