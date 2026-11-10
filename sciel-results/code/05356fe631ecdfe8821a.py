"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 55
    n = 192
    max_size = 48
    
    def compute_triple_sums(S):
        """Return set of all triple sums from S."""
        m = len(S)
        sums = set()
        for i in range(m):
            for j in range(i+1, m):
                for k in range(j+1, m):
                    sums.add(S[i] + S[j] + S[k])
        return sums
    
    def can_add(x, S, S_set, triple_sums):
        if x in S_set:
            return False
        # Check x divides no triple sum from S
        for s in triple_sums:
            if s % x == 0:
                return False
        # Check each a in S: a divides no (x+b+c) for b,c in S\{a}
        m = len(S)
        for idx_a in range(m):
            a = S[idx_a]
            for i in range(m):
                if i == idx_a:
                    continue
                b = S[i]
                for j in range(i+1, m):
                    if j == idx_a:
                        continue
                    if (x + b + S[j]) % a == 0:
                        return False
        return True
    
    # Greedy: descending
    S = []
    S_set = set()
    triple_sums = set()
    for x in range(n, 0, -1):
        if can_add(x, S, S_set, triple_sums):
            # Add x and update triple_sums
            m = len(S)
            for i in range(m):
                for j in range(i+1, m):
                    triple_sums.add(x + S[i] + S[j])
            S.append(x)
            S_set.add(x)
            if len(S) >= max_size:
                break
    best = S[:]
    best_set = S_set.copy()
    
    # Greedy: ascending
    S = []
    S_set = set()
    triple_sums = set()
    for x in range(1, n+1):
        if can_add(x, S, S_set, triple_sums):
            m = len(S)
            for i in range(m):
                for j in range(i+1, m):
                    triple_sums.add(x + S[i] + S[j])
            S.append(x)
            S_set.add(x)
            if len(S) >= max_size:
                break
    if len(S) > len(best):
        best = S[:]
        best_set = S_set.copy()
    
    # Local search
    def local_search(S, S_set, time_limit):
        S = S[:]
        S_set = S_set.copy()
        triple_sums = compute_triple_sums(S)
        while time.time() < time_limit:
            improved = False
            # Try adding
            for x in range(1, n+1):
                if x not in S_set and can_add(x, S, S_set, triple_sums):
                    m = len(S)
                    for i in range(m):
                        for j in range(i+1, m):
                            triple_sums.add(x + S[i] + S[j])
                    S.append(x)
                    S_set.add(x)
                    improved = True
                    if len(S) >= max_size:
                        return S, S_set
            # Try swaps
            if len(S) >= max_size:
                for i in range(len(S)):
                    a = S[i]
                    S_temp = S[:i] + S[i+1:]
                    S_temp_set = S_set - {a}
                    triple_sums_temp = compute_triple_sums(S_temp)
                    found = False
                    for x in range(1, n+1):
                        if x not in S_temp_set and can_add(x, S_temp, S_temp_set, triple_sums_temp):
                            # Compute new triple sums
                            m2 = len(S_temp)
                            for ii in range(m2):
                                for jj in range(ii+1, m2):
                                    triple_sums_temp.add(x + S_temp[ii] + S_temp[jj])
                            S = S_temp + [x]
                            S_set = S_temp_set | {x}
                            triple_sums = triple_sums_temp
                            improved = True
                            found = True
                            break
                    if found:
                        break
            if not improved:
                break
        return S, S_set
    
    time_limit = deadline - 3
    best, best_set = local_search(best, best_set, time_limit)
    
    # Perturbation restarts
    rng = Random(42)
    while time.time() < deadline - 3:
        S = best[:]
        S_set = best_set.copy()
        k = rng.randint(1, 3)
        for _ in range(k):
            if not S:
                break
            idx = rng.randint(0, len(S)-1)
            S.pop(idx)
        S_set = set(S)
        S, S_set = local_search(S, S_set, deadline - 3)
        if len(S) > len(best):
            best = S
            best_set = S_set
    
    return best
# EVOLVE-BLOCK-END
