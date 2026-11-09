import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Precompute coprime matrix
    N = 512
    coprime = [[True] * N for _ in range(N)]
    for i in range(1, N):
        coprime[i][i] = False
        for j in range(i + 1, N):
            if gcd(i, j) == 1:
                coprime[i][j] = True
                coprime[j][i] = True
            else:
                coprime[i][j] = False
                coprime[j][i] = False

    def score(S):
        return sum(1.0 / (512.0 - a) for a in S)

    def is_coprime_to_all(a, S):
        for b in S:
            if not coprime[a][b]:
                return False
        return True

    def greedy_set(candidates, background):
        selected = []
        # candidates is already sorted descending
        for a in candidates:
            if is_coprime_to_all(a, background) and is_coprime_to_all(a, selected):
                selected.append(a)
        return selected

    # 1. Initial Greedy
    S = []
    for a in range(511, 0, -1):
        if is_coprime_to_all(a, S):
            S.append(a)
    
    current_score = score(S)
    improved = True
    S_set = set(S)
    
    # Pre-sorted candidates list
    all_candidates = list(range(511, 0, -1))
    
    # 2. Local Search
    while improved and time.time() < deadline:
        improved = False
        # Try removing each element
        for a in list(S_set):
            # Check if we can find a better set by removing a
            background = S_set - {a}
            
            # Filter candidates: must be coprime to background
            # We can just pass all candidates to greedy_set, it will check
            # But filtering first is faster
            valid_candidates = [c for c in all_candidates if c not in S_set and is_coprime_to_all(c, background)]
            
            B = greedy_set(valid_candidates, background)
            
            if B:
                new_score = current_score - (1.0 / (512.0 - a)) + score(B)
                if new_score > current_score + 1e-9:
                    S_set = background.union(B)
                    current_score = new_score
                    improved = True
                    break
    
    return list(S_set)
