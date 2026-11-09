# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    # Precompute 5-smooth numbers up to 128
    smooths = []
    for x in range(1, 129):
        y = x
        while y % 2 == 0: y //= 2
        while y % 3 == 0: y //= 3
        while y % 5 == 0: y //= 5
        if y == 1:
            smooths.append(x)
    
    M = len(smooths)
    
    # Precompute best (min size) antichain for each sum up to 128
    # best[sum] = (size, list_of_terms)
    best = {}
    
    # Iterate over all subsets of smooths? 2^35 is too big.
    # But we only care about sums <= 128.
    # We can use DFS/Backtracking to generate antichains with sum <= 128.
    
    def backtrack(idx, current_sum, current_set):
        # Pruning: if current_sum > 128, stop
        if current_sum > 128:
            return
        
        # If current_set is not empty, it's a valid antichain candidate for its sum
        if current_set:
            s = current_sum
            if s not in best:
                best[s] = (len(current_set), list(current_set))
            else:
                # Update if smaller size
                if len(current_set) < best[s][0]:
                    best[s] = (len(current_set), list(current_set))
        
        # Try adding more elements
        for i in range(idx, M):
            val = smooths[i]
            # Check antichain condition: val must not be divisible by any in current_set
            # And no element in current_set should divide val (handled by above if we check all)
            # Actually, if we maintain sorted order, we just need to check if any existing element divides val.
            # Wait, if a in set divides b in set, it's invalid.
            # If we add val, we need:
            # 1. val % x != 0 for all x in current_set
            # 2. x % val != 0 for all x in current_set (impossible if x < val, which is true if we pick from sorted list and idx increases? No, smooths is sorted. So x <= val. If x divides val, then val % x == 0. If val divides x, then x % val == 0. Since x <= val, val divides x implies x==val. But distinct, so x<val. So only check x | val.)
            
            valid = True
            for x in current_set:
                if val % x == 0:
                    valid = False
                    break
                if x % val == 0:
                    # This would mean x is a multiple of val. Since x <= val, x==val.
                    # But elements are distinct. So this shouldn't happen unless logic error.
                    valid = False
                    break
            
            if valid:
                current_set.append(val)
                backtrack(i + 1, current_sum + val, current_set)
                current_set.pop()

    # Start backtracking
    backtrack(0, 0, [])
    
    # Now best[s] contains the minimal antichain for sum s.
    # We need to fill in any missing targets?
    # The problem states "All targets require valid representations".
    # If a target n is not in best, it means no antichain sums to n?
    # This is unlikely for 1..128.
    # Let's check if any are missing.
    
    answers = []
    for n in range(1, 129):
        if n in best:
            answers.append(best[n][1])
        else:
            # Fallback: find any valid representation
            # This should not happen.
            # But to be safe, we can try a simple greedy or just return []
            # Actually, if n is smooth, {n} is an antichain.
            # If n is not smooth, it must be decomposed.
            # My backtracking should have found it.
            # Let's add a fallback search just in case.
            candidates = [x for x in smooths if x <= n]
            found = None
            for k in range(1, len(candidates) + 1):
                for comb in combinations(candidates, k):
                    if sum(comb) == n:
                        # Check antichain
                        is_anti = True
                        for i in range(len(comb)):
                            for j in range(i+1, len(comb)):
                                if comb[j] % comb[i] == 0:
                                    is_anti = False
                                    break
                            if not is_anti: break
                        if is_anti:
                            found = list(comb)
                            break
                if found: break
            if found:
                answers.append(found)
            else:
                answers.append([]) # Should not happen

    return answers
# EVOLVE-BLOCK-END
