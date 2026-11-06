# EVOLVE-BLOCK-START
import random
from itertools import combinations_with_replacement

def construct():
    n = 4096
    max_k = 40  # Theoretical upper bound: (k+2)(k+1)k/6 <= 3n => k <= 40

    def check_and_add(S, T, x, S_set):
        """
        Check if x can be added to S without creating duplicate triple sums.
        S: list of current elements.
        T: set of current triple sums.
        x: candidate element.
        S_set: set of current elements for O(1) lookup (though we check x not in S before calling).
        Returns True if valid, and updates S and T in place if valid.
        """
        # Generate new sums
        new_sums_list = []
        new_sums_list.append(3 * x)
        
        for s in S:
            new_sums_list.append(2 * x + s)
            
        k = len(S)
        for i in range(k):
            s1 = S[i]
            for j in range(i, k):
                s2 = S[j]
                new_sums_list.append(x + s1 + s2)
                
        # Check for internal collisions
        if len(new_sums_list) != len(set(new_sums_list)):
            return False
            
        new_sums_set = set(new_sums_list)
        
        # Check collision with existing sums
        if new_sums_set & T:
            return False
            
        # Valid: Update S and T
        S.append(x)
        T.update(new_sums_set)
        return True

    def build_set(skeleton, candidates_order):
        S = list(skeleton)
        S_set = set(S)
        
        # Compute initial sums
        T = set()
        if len(S) >= 3:
            for t in combinations_with_replacement(S, 3):
                T.add(sum(t))
        elif len(S) == 2:
            a, b = S
            T.add(3*a)
            T.add(2*a+b)
            T.add(a+2*b)
            T.add(3*b)
        elif len(S) == 1:
            a = S[0]
            T.add(3*a)
            
        for x in candidates_order:
            if x in S_set:
                continue
            if check_and_add(S, T, x, S_set):
                S_set.add(x)
                if len(S) >= max_k:
                    break
        return S

    # Define candidate skeletons
    # Skeleton 1: Powers of 4
    skel1 = [1, 4, 16, 64, 256, 1024, 4096]
    # Skeleton 2: Powers of 5
    skel2 = [1, 5, 25, 125, 625, 3125]
    # Skeleton 3: Random sparse set
    random.seed(42)
    skel3 = sorted(random.sample(range(1, 4097), 10))
    
    all_candidates = list(range(1, 4097))
    
    best_set = []
    
    # Strategy 1: Fixed Skeletons with Ascending/Descending/Random orders
    skeletons = [skel1, skel2, skel3]
    
    for skel in skeletons:
        # Ascending
        cands = [x for x in all_candidates if x not in skel]
        s = build_set(skel, cands)
        if len(s) > len(best_set):
            best_set = s
            
        # Descending
        cands_desc = sorted(cands, reverse=True)
        s = build_set(skel, cands_desc)
        if len(s) > len(best_set):
            best_set = s
            
        # Random orders (multiple trials)
        for _ in range(30):
            cands_rand = cands[:]
            random.shuffle(cands_rand)
            s = build_set(skel, cands_rand)
            if len(s) > len(best_set):
                best_set = s
                
        if len(best_set) >= max_k:
            break
            
    # If we haven't reached max_k, try pure random starts (no skeleton)
    if len(best_set) < max_k:
        for _ in range(50):
            cands_rand = all_candidates[:]
            random.shuffle(cands_rand)
            s = build_set([], cands_rand)
            if len(s) > len(best_set):
                best_set = s
            if len(best_set) >= max_k:
                break

    return best_set
# EVOLVE-BLOCK-END
