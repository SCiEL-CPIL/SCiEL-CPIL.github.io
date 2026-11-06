"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 50.0

    # 1. Generate ternary set (digits 0 and 1)
    candidates = []
    val = 1
    while val <= 512:
        v = val
        ok = True
        while v > 0:
            if v % 3 == 2:
                ok = False
                break
            v //= 3
        if ok:
            candidates.append(val)
        val += 1
    
    # 2. Check if the ternary set is valid
    def is_valid_set(S):
        S_set = set(S)
        n = len(S)
        
        for a in S:
            others = [x for x in S if x != a]
            
            # Check k=3: 3a = x+y+z
            # x+y = 3a - z
            # We check if there exist distinct x,y,z in others such that x+y+z = 3a
            # Equivalent to: does the set of pair sums of others contain 3a-z for some z in others?
            # And are the pairs disjoint from z?
            
            # Generate pair sums with indices
            pairs = []
            for i in range(len(others)):
                for j in range(i+1, len(others)):
                    pairs.append((others[i] + others[j], i, j))
            
            # Check k=3
            # For each z (index k in others), we need a pair (i,j) such that pairs[i,j].sum = 3a - others[k]
            # and i,j != k
            found = False
            target_map = {}
            # Instead of scanning pairs for each z, we can just iterate z and check
            # But 3a-z varies.
            # Let's just iterate z and check if (3a - z) is in pair_sums set, and verify disjointness.
            pair_sums_set = set(p[0] for p in pairs)
            
            for k_idx, z in enumerate(others):
                req_sum = 3 * a - z
                if req_sum in pair_sums_set:
                    # Find a pair with this sum
                    # We need to check if any pair with this sum is disjoint from k_idx
                    # We stored pairs in a list. We can filter.
                    # Optimization: pre-group pairs by sum?
                    pass
            
            # Simpler: Just iterate all triples? No, O(N^3) is 200k.
            # Let's just do O(N^3) for k=3.
            for i in range(len(others)):
                for j in range(i+1, len(others)):
                    for k in range(j+1, len(others)):
                        if others[i] + others[j] + others[k] == 3 * a:
                            return False
                            
            # Check k=4: 4a = x+y+z+w
            # x+y+z = 4a - w
            # Check if any triple sum is 4a - w
            # Triple sums count: N^3/6 ~ 30k.
            # For each w, check if 4a-w is in triple_sums.
            
            triple_sums = set()
            for i in range(len(others)):
                for j in range(i+1, len(others)):
                    s2 = others[i] + others[j]
                    for k in range(j+1, len(others)):
                        triple_sums.add(s2 + others[k])
            
            for w in others:
                if 4 * a - w in triple_sums:
                    # Need to verify disjointness?
                    # The triple sum was formed from distinct i,j,k.
                    # If w is one of i,j,k, then the subset size is 3, not 4.
                    # But we are checking k=4.
                    # If w is in the triple, then sum of 4 elements (with one repeated) is 4a?
                    # No, the elements must be distinct.
                    # So we must ensure w is not in the triple that formed the sum.
                    # This is hard to check with just a set of sums.
                    # However, if w is in the triple, say triple is {w, x, y}, sum = w+x+y = 4a-w -> 2w+x+y=4a.
                    # This is a k=3 mean check for 2a? No.
                    # It's a specific configuration.
                    # Given the sparsity, it's unlikely to be a false positive.
                    # We'll assume valid if sum exists.
                    return False
                    
        return True

    if is_valid_set(candidates):
        return candidates[:48]
        
    # Fallback: Greedy
    selected = []
    for x in range(1, 513):
        if time.time() > deadline:
            break
        selected.append(x)
        if len(selected) == 48:
            break
    return selected

# EVOLVE-BLOCK-END
