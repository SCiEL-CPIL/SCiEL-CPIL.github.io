import math

def construct():
    # Difference-sparse construction: subsets of arithmetic progressions
    # U = {0, d, 2d, ..., (n-1)d} with some elements removed
    # Optimize gap d and subset size n
    
    def calc_score(u):
        if not u or len(u) < 2:
            return 0
        max_u = max(u)
        if max_u == 0:
            return 0
        
        # Use sets for sums and diffs (simpler, works for moderate sizes)
        sums = set()
        diffs = set()
        for i in u:
            for j in u:
                sums.add(i + j)
                diffs.add(i - j)
        
        num_sums = len(sums)
        num_diffs = len(diffs)
        
        if num_sums == 0:
            return 0
            
        ratio = num_diffs / num_sums
        if ratio <= 1:
            return 0
            
        denom = math.log2(2 * max_u + 1)
        if denom == 0:
            return 0
            
        return 1 + math.log2(ratio) / denom

    best_score = 0
    best_set = [0, 1, 3, 7, 15]
    
    # Search over arithmetic progressions: base d, length L, subset size n
    # U is a subset of {0, d, 2d, ..., (L-1)d}
    
    for d in range(1, 50):  # gap size
        for L in range(3, 20):  # length of AP
            max_val = (L - 1) * d
            if max_val > 2000:
                break
            
            # Try all subsets of size n from the AP
            # AP elements: 0, d, 2d, ..., (L-1)d
            ap = list(range(0, L * d, d))
            
            # Try subset sizes from 2 to min(L, 8)
            max_n = min(L, 8)
            for n in range(2, max_n + 1):
                # Generate all subsets of size n from ap
                # C(L, n) should be manageable
                from itertools import combinations
                for comb in combinations(ap, n):
                    u = sorted(list(comb))
                    if 0 not in u:
                        continue
                    s = calc_score(u)
                    if s > best_score:
                        best_score = s
                        best_set = u
    
    return best_set
