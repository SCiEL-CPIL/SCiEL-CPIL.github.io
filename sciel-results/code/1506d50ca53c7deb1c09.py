import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit
    
    N = 512
    weights = [0] * N
    for a in range(1, N):
        weights[a] = 1.0 / (N - a)
        
    # Precompute gcds or factors? 
    # For N=512, O(N^2) is fine.
    
    # 1. Greedy Initialization
    chosen = []
    for a in range(N - 1, 0, -1):
        if all(gcd(a, b) == 1 for b in chosen):
            chosen.append(a)
            
    def score(S):
        return sum(weights[a] for a in S)
        
    current_score = score(chosen)
    chosen_set = set(chosen)
    
    # 2. Local Search
    # Precompute coprime matrix for speed? 
    # coprime[a][b] = 1 if gcd(a,b)==1 else 0
    # 512x512 bool matrix is ~256KB. Fast.
    coprime = [[True]*N for _ in range(N)]
    for i in range(1, N):
        for j in range(i, N):
            coprime[i][j] = coprime[j][i] = (gcd(i, j) == 1)
            
    improved = True
    while improved and (time.time() < deadline):
        improved = False
        # Try 1-for-1 swaps
        for a in list(chosen_set):
            # Find best x not in chosen_set, coprime to all in chosen_set \ {a}
            best_x = -1
            best_gain = 0
            # Optimization: only consider x that are coprime to all chosen except a
            # And x must be coprime to a? No, x is replacing a, so x doesn't need to be coprime to a.
            # But x must be coprime to ALL other chosen.
            for x in range(1, N):
                if x in chosen_set:
                    continue
                # Check if x is coprime to all b in chosen_set where b != a
                ok = True
                for b in chosen_set:
                    if b == a:
                        continue
                    if not coprime[x][b]:
                        ok = False
                        break
                if ok:
                    gain = weights[x] - weights[a]
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
            if best_x > 0:
                chosen_set.remove(a)
                chosen_set.add(best_x)
                current_score += best_gain
                improved = True
                
        # Try 1-for-2 swaps
        # Remove a, add x, y
        # x, y coprime to each other and to all chosen \ {a}
        # This is O(N^3) per a. 512^3 is too slow.
        # Limit search space: Only consider x, y that are "neighbors" of a?
        # Or just iterate over all pairs of unchosen numbers?
        # Number of unchosen is ~400. 400^2 = 160k pairs.
        # For each a (50), 160k checks. 8M checks per iteration.
        # Python is slow. 8M * 100 iterations = 800M. Too slow.
        
        # Optimized 1-for-2:
        # For each a, find list L of x not in chosen, coprime to chosen\{a}.
        # Then for each pair x, y in L, check if coprime(x, y).
        # Gain = w[x] + w[y] - w[a].
        
        for a in list(chosen_set):
            candidates = []
            for x in range(1, N):
                if x in chosen_set:
                    continue
                ok = True
                for b in chosen_set:
                    if b == a:
                        continue
                    if not coprime[x][b]:
                        ok = False
                        break
                if ok:
                    candidates.append(x)
                    
            # Sort candidates by weight desc to find best pairs quickly?
            # Or just iterate.
            # If candidates is large, this is slow.
            # But candidates are those blocked ONLY by a (and maybe others, but we removed a).
            # Actually, candidates are those coprime to chosen\{a}.
            
            best_pair_gain = 0
            best_x = -1
            best_y = -1
            n_cand = len(candidates)
            for i in range(n_cand):
                x = candidates[i]
                for j in range(i+1, n_cand):
                    y = candidates[j]
                    if coprime[x][y]:
                        gain = weights[x] + weights[y] - weights[a]
                        if gain > best_pair_gain:
                            best_pair_gain = gain
                            best_x = x
                            best_y = y
            if best_pair_gain > 0:
                chosen_set.remove(a)
                chosen_set.add(best_x)
                chosen_set.add(best_y)
                current_score += best_pair_gain
                improved = True
                
    return list(chosen_set)
