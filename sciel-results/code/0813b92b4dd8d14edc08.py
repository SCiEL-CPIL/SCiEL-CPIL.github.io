import itertools
import math

def construct():
    best_score = 0
    best_set = [0, 1, 3]
    
    # Search for optimal A_small
    # K is max of A_small
    for K in range(1, 16):
        # Iterate all subsets of {1, ..., K}
        # There are 2^K subsets
        # For efficiency, we can use bit masks
        for mask in range(1 << K):
            S_small = []
            for i in range(K):
                if mask & (1 << i):
                    S_small.append(i + 1)
            A_small = [0] + S_small
            S = len(A_small)
            
            # Calculate P and D
            # Sumset
            sum_set = set()
            for a in A_small:
                for b in A_small:
                    sum_set.add(a + b)
            P = len(sum_set)
            
            # Diffset
            diff_set = set()
            for a in A_small:
                for b in A_small:
                    diff_set.add(a - b)
            D = len(diff_set)
            
            M = 2 * K + 1
            A_full = A_small + [M]
            
            # Check disjointness (should be true)
            # Max A_small sum = 2K
            # Min M + A_small = M
            # M > 2K is true
            
            # Total sizes
            total_P = P + S + 1
            total_D = D + 2 * S
            
            max_A = M
            denom = math.log(2 * max_A + 1)
            if denom == 0:
                continue
            num = math.log(total_D / total_P)
            score = 1 + num / denom
            
            if score > best_score:
                best_score = score
                best_set = A_full
                
    return best_set
