# EVOLVE-BLOCK-START
import random
from itertools import combinations_with_replacement

def construct():
    n = 4096
    target_size = 40 # Theoretical max
    
    def build_set(skeleton, candidates):
        S = list(skeleton)
        # Initialize T and P
        T = set()
        P = set()
        
        # Pre-compute sums for skeleton
        for i in range(len(S)):
            s1 = S[i]
            for j in range(i, len(S)):
                s2 = S[j]
                P.add(s1 + s2)
        
        for t in combinations_with_replacement(S, 3):
            T.add(sum(t))
            
        if not S:
            return []

        for x in candidates:
            if x in S:
                continue
            
            # Check internal collision conditions
            # 1. 2x in P?
            if 2 * x in P:
                continue
            # 2. x+s in P for any s in S?
            conflict = False
            for s in S:
                if x + s in P:
                    conflict = True
                    break
            if conflict:
                continue
                
            # Check collision with existing T
            # 3. 3x in T?
            if 3 * x in T:
                continue
            # 4. 2x+s in T for any s in S?
            conflict = False
            for s in S:
                if 2 * x + s in T:
                    conflict = True
                    break
            if conflict:
                continue
            # 5. x+p in T for any p in P?
            conflict = False
            for p in P:
                if x + p in T:
                    conflict = True
                    break
            if conflict:
                continue
                
            # All checks passed, add x
            S.append(x)
            
            # Update T
            T.add(3 * x)
            for s in S[:-1]: # s is from old S
                T.add(2 * x + s)
            for p in P: # p is from old P
                T.add(x + p)
                
            # Update P
            P.add(2 * x)
            for s in S[:-1]: # s is from old S
                P.add(x + s)
                
            if len(S) >= target_size:
                break
                
        return S

    # Skeleton: Powers of 4
    skeleton1 = [1, 4, 16, 64, 256, 1024, 4096]
    all_candidates = list(range(1, 4097))
    
    best_set = []
    
    # Strategy 1: Skeleton + Ascending
    cands_asc = [x for x in all_candidates if x not in skeleton1]
    s1 = build_set(skeleton1, cands_asc)
    if len(s1) > len(best_set):
        best_set = s1
        
    # Strategy 2: Skeleton + Descending
    cands_desc = [x for x in reversed(all_candidates) if x not in skeleton1]
    s2 = build_set(skeleton1, cands_desc)
    if len(s2) > len(best_set):
        best_set = s2

    # Strategy 3: No Skeleton + Random (Multiple trials)
    random.seed(42)
    for i in range(20):
        cands_rand = all_candidates[:]
        random.shuffle(cands_rand)
        s_rand = build_set([], cands_rand)
        if len(s_rand) > len(best_set):
            best_set = s_rand
            
    # Strategy 4: No Skeleton + Middle-out (Sorted by distance from n/2)
    # This might pack the middle better
    cands_mid = sorted(all_candidates, key=lambda x: abs(x - n/2))
    s_mid = build_set([], cands_mid)
    if len(s_mid) > len(best_set):
        best_set = s_mid
        
    # Strategy 5: Skeleton + Random
    for i in range(10):
        cands_rand = [x for x in all_candidates if x not in skeleton1]
        random.shuffle(cands_rand)
        s_rand = build_set(skeleton1, cands_rand)
        if len(s_rand) > len(best_set):
            best_set = s_rand
            
    return best_set
# EVOLVE-BLOCK-END
