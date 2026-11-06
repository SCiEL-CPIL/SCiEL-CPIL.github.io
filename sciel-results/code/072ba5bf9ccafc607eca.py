import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    p = 101
    denominators = list(range(1, 11))
    invs = [pow(x, -1, p) for x in denominators]
    
    # Precompute all non-empty subsets: (sum_mod_p, size, denom_list)
    # There are only 2^10 - 1 = 1023 subsets
    all_subsets = []
    for mask in range(1, 1 << 10):
        denom_list = []
        s = 0
        size = 0
        for i in range(10):
            if mask & (1 << i):
                denom_list.append(denominators[i])
                s = (s + invs[i]) % p
                size += 1
        all_subsets.append((s, size, denom_list))
    
    # For each residue, find the minimum-size subset that produces it
    # Since denominators can recur across rows, each row is independent
    min_size_for_res = [11] * p  # 11 = infinity (max possible size is 10)
    best_subset_for_res = [None] * p
    
    for s, size, dl in all_subsets:
        if size < min_size_for_res[s]:
            min_size_for_res[s] = size
            best_subset_for_res[s] = dl
    
    # Verify all residues are covered
    for r in range(p):
        if best_subset_for_res[r] is None:
            # Should not happen, but fallback
            best_subset_for_res[r] = [1]
            min_size_for_res[r] = 1
    
    # The optimal max row length is the maximum over all residues of their min size
    optimal_max = max(min_size_for_res)
    
    # Construct the result: for each residue, use its minimum-size subset
    result = []
    for r in range(p):
        result.append(best_subset_for_res[r])
    
    # Verify correctness
    for r in range(p):
        s = sum(invs[x - 1] for x in result[r]) % p
        if s != r:
            # This should not happen; but if it does, we have a bug
            # Fallback: search for any valid subset
            for s2, size2, dl2 in all_subsets:
                if s2 == r:
                    result[r] = dl2
                    break
    
    return result
