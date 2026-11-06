# EVOLVE-BLOCK-START
import random
from itertools import combinations_with_replacement

def construct():
    n = 4096
    max_k = 40 # Theoretical upper bound
    
    def can_add(S, T, x):
        """
        Check if x can be added to S without creating duplicate triple sums.
        Fails fast on collision.
        """
        new_sums = set()
        
        # 1. Triple (x, x, x)
        v = 3 * x
        if v in T or v in new_sums:
            return False
        new_sums.add(v)
        
        # 2. Triples (x, x, s)
        for s in S:
            v = 2 * x + s
            if v in T or v in new_sums:
                return False
            new_sums.add(v)
            
        # 3. Triples (x, s1, s2) with s1 <= s2
        k = len(S)
        for i in range(k):
            s1 = S[i]
            for j in range(i, k):
                s2 = S[j]
                v = x + s1 + s2
                if v in T or v in new_sums:
                    return False
                new_sums.add(v)
                
        return True

    def build_set(skeleton, candidates):
        S = list(skeleton)
        # Compute initial sums
        T = set()
        for t in combinations_with_replacement(S, 3):
            T.add(sum(t))
            
        for x in candidates:
            if x in S:
                continue
            if can_add(S, T, x):
                S.append(x)
                # Update T with new sums
                T.add(3 * x)
                old_S = S[:-1]
                for s in old_S:
                    T.add(2 * x + s)
                k = len(old_S)
                for i in range(k):
                    s1 = old_S[i]
                    for j in range(i, k):
                        s2 = old_S[j]
                        T.add(x + s1 + s2)
                        
                if len(S) >= max_k:
                    break
                    
        return S

    def base4_popcount(x):
        count = 0
        while x:
            if x % 4 != 0:
                count += 1
            x //= 4
        return count

    # Skeleton: Powers of 4
    skeleton = [1, 4, 16, 64, 256, 1024, 4096]
    all_candidates = [x for x in range(1, 4097) if x not in skeleton]
    
    best_set = []
    
    # Strategy 1: Ascending
    s1 = build_set(skeleton, sorted(all_candidates))
    if len(s1) > len(best_set):
        best_set = s1
        
    # Strategy 2: Descending
    s2 = build_set(skeleton, sorted(all_candidates, reverse=True))
    if len(s2) > len(best_set):
        best_set = s2
        
    # Strategy 3: Sparsity (fewer base-4 digits first)
    cands_sparse = sorted(all_candidates, key=lambda x: (base4_popcount(x), x))
    s3 = build_set(skeleton, cands_sparse)
    if len(s3) > len(best_set):
        best_set = s3
        
    # Strategy 4: Randomized trials
    random.seed(42)
    for _ in range(20):
        cands_rand = all_candidates[:]
        random.shuffle(cands_rand)
        s_rand = build_set(skeleton, cands_rand)
        if len(s_rand) > len(best_set):
            best_set = s_rand
            if len(best_set) >= max_k:
                break
                
    # Strategy 5: Random small seed
    for _ in range(10):
        seed_set = random.sample(all_candidates, 5)
        cands_rem = [x for x in all_candidates if x not in seed_set]
        random.shuffle(cands_rem)
        s_seed = build_set(seed_set, cands_rem)
        if len(s_seed) > len(best_set):
            best_set = s_seed
            if len(best_set) >= max_k:
                break

    return best_set
# EVOLVE-BLOCK-END
