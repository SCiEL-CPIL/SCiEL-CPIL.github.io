# EVOLVE-BLOCK-START
import time
import sys
import random
sys.setrecursionlimit(10000)

def construct():
    n = 256
    start_time = time.time()
    time_limit = 55.0  # Leave 5s buffer

    # Precompute APs of length >= 2
    # Length 1 APs have sum +/-1, so they are always satisfied if B >= 1.
    # We start with B=1, so we must handle length 2 APs?
    # Sum of 2 terms is -2, 0, 2. If B=1, sum must be 0.
    # So we MUST include length 2 APs.
    
    ap_list = []
    index_aps = [[] for _ in range(n)]
    
    for d in range(1, n):
        max_len = (n - 1) // d + 1
        for L in range(2, max_len + 1):  # Start from 2
            max_start = n - (L - 1) * d
            for a in range(max_start):
                indices = [a + k * d for k in range(L)]
                ap_id = len(ap_list)
                ap_list.append(indices)
                for idx in indices:
                    index_aps[idx].append(ap_id)

    ap_lengths = [len(ap) for ap in ap_list]
    
    # Static Variable Ordering: Assign indices involved in more APs first
    # This helps prune earlier.
    ap_counts = [len(aps) for aps in index_aps]
    # Sort indices by count descending. Tie-break by index for determinism.
    order = sorted(range(n), key=lambda i: (-ap_counts[i], i))
    
    best_s = None
    best_B = float('inf')
    
    B = 1
    while time.time() - start_time < time_limit:
        s = [0] * n  # 0 means unassigned
        ap_sums = [0] * len(ap_list)
        unassigned_counts = ap_lengths[:]
        
        t_start_B = time.time()
        
        def is_consistent(idx, sigma):
            # Check if assigning sigma to index idx violates any AP bound
            for ap_id in index_aps[idx]:
                # Optimization: If B >= 2, APs of length 2 are always satisfied
                # because max sum is 2.
                if B >= 2 and ap_lengths[ap_id] == 2:
                    continue
                new_sum = ap_sums[ap_id] + sigma
                remaining = unassigned_counts[ap_id] - 1
                # If the current sum already exceeds B, it's invalid.
                # Actually, we check if it's *possible* to stay within [-B, B].
                # Max possible sum = new_sum + remaining
                # Min possible sum = new_sum - remaining
                if new_sum + remaining < -B or new_sum - remaining > B:
                    return False
            return True

        def bt(k):
            nonlocal s, ap_sums, unassigned_counts
            if time.time() - t_start_B > 5.0:
                return False
            if k == n:
                return True
            
            idx = order[k]
            
            # Heuristic: Try the sign that minimizes the absolute sum for the first AP
            # This is a simple "Least Constrained Value" variant.
            # Pick the first AP containing idx to guide the choice.
            if index_aps[idx]:
                first_ap = index_aps[idx][0]
                s1 = abs(ap_sums[first_ap] + 1)
                s2 = abs(ap_sums[first_ap] - 1)
                if s1 < s2:
                    signs = [1, -1]
                else:
                    signs = [-1, 1]
            else:
                signs = [1, -1]
                
            for sigma in signs:
                if is_consistent(idx, sigma):
                    s[idx] = sigma
                    for ap_id in index_aps[idx]:
                        ap_sums[ap_id] += sigma
                        unassigned_counts[ap_id] -= 1
                    if bt(k + 1):
                        return True
                    for ap_id in index_aps[idx]:
                        ap_sums[ap_id] -= sigma
                        unassigned_counts[ap_id] += 1
                    s[idx] = 0
            return False

        if bt(0):
            best_s = s[:]
            best_B = B
            break
        else:
            B += 1
            if B > 10:
                break

    if best_s is None:
        # Fallback: Random assignment
        rng = random.Random(42)
        best_s = [rng.choice([-1, 1]) for _ in range(n)]
        
    return best_s

# EVOLVE-BLOCK-END
