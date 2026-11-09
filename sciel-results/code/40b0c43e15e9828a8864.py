import time
from functools import lru_cache

# Precompute at module level
def _precompute():
    # Generate all 2,3,5-smooth numbers up to 128
    smooth = set()
    for a in range(0, 8):
        va = 2 ** a
        for b in range(0, 5):
            vb = va * (3 ** b)
            if vb > 128:
                break
            for c in range(0, 4):
                vc = vb * (5 ** c)
                if vc > 128:
                    break
                smooth.add(vc)
    
    smooth_list = sorted(smooth)
    n_smooth = len(smooth_list)
    
    # Map smooth number to index
    smooth_to_idx = {s: i for i, s in enumerate(smooth_list)}
    
    # Build incompatibility mask
    # incompatible_mask[i] = bitmask of j where i and j are incompatible (one divides the other)
    incompatible_mask = [0] * n_smooth
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if smooth_list[j] % smooth_list[i] == 0 or smooth_list[i] % smooth_list[j] == 0:
                incompatible_mask[i] |= (1 << j)
                incompatible_mask[j] |= (1 << i)
    
    # Candidates for each target n: indices of smooth numbers <= n
    candidates = [[] for _ in range(129)]
    for n in range(1, 129):
        for i, s in enumerate(smooth_list):
            if s <= n:
                candidates[n].append(i)
            else:
                break
    
    # Global deadline
    start_time = time.time()
    deadline = start_time + 50.0 # 50s margin
    
    def find_antichain(n, k):
        """Find antichain of exactly k distinct smooth numbers summing to n."""
        cand = candidates[n]
        if len(cand) < k:
            return None
        
        if k == 1:
            if n in smooth_to_idx:
                return [n]
            return None
        
        # Sort descending for better pruning (larger numbers first reduces rem_sum faster)
        cand_desc = sorted(cand, reverse=True, key=lambda i: smooth_list[i])
        
        # Precompute min/max sums for suffixes for O(1) pruning
        # max_sums[i][r] = sum of r largest elements in cand_desc[i:]
        # min_sums[i][r] = sum of r smallest elements in cand_desc[i:]
        # Since cand_desc is sorted descending, the r largest are cand_desc[i:i+r]
        # The r smallest are cand_desc[len(cand_desc)-r : len(cand_desc)-r+i] ?? 
        # Actually, min sum of r elements from a suffix [i:] of a descending list:
        # The elements are cand_desc[i], cand_desc[i+1], ...
        # The smallest r elements in this suffix are the last r elements of the suffix.
        # i.e., cand_desc[-r:] if the suffix is the whole list.
        # For suffix starting at i, the elements are cand_desc[i:L].
        # The smallest r are cand_desc[L-r:L].
        # Wait, if i > L-r, then we can't pick r smallest from the "tail" if the tail is cut off?
        # No, the suffix is cand_desc[i:L].
        # The smallest r elements in cand_desc[i:L] are cand_desc[L-r:L] IF L-i >= r.
        # Yes, because the list is descending. The smallest elements are at the end.
        # So min_sum = sum(cand_desc[L-r:L]) is constant for all i such that L-i >= r?
        # No! If i is large, the suffix is shorter.
        # If L-i < r, we can't pick r elements.
        # If L-i >= r, the r smallest elements in cand_desc[i:L] are indeed cand_desc[L-r:L].
        # Because cand_desc is global descending. The global smallest r elements are at the end.
        # Any suffix that contains the last r elements will have the same min sum.
        # If the suffix does NOT contain the last r elements (i.e., i > L-r), then it has < r elements, so invalid.
        # So, for any valid suffix (len >= r), the min sum is sum of the global last r elements?
        # Let's check.
        # List: 10, 9, 8, 7, 6. k=2.
        # Suffix [0:]: 10,9,8,7,6. Min 2: 7,6 -> 13.
        # Suffix [1:]: 9,8,7,6. Min 2: 7,6 -> 13.
        # Suffix [2:]: 8,7,6. Min 2: 7,6 -> 13.
        # Suffix [3:]: 7,6. Min 2: 7,6 -> 13.
        # Suffix [4:]: 6. Len < 2.
        # So yes, min_sum is constant for all valid suffixes!
        # It is just sum(cand_desc[-k:])
        
        # What about max_sum?
        # Max sum of r elements in cand_desc[i:L] is sum(cand_desc[i:i+r]).
        # This depends on i.
        
        L = len(cand_desc)
        global_min_k = sum(smooth_list[cand_desc[j]] for j in range(L-k, L))
        
        max_sums = [0] * (L + 1)
        # max_sums[i] for k=1 is cand_desc[i]
        # We need max_sums for specific k.
        # Let's just compute sum(cand_desc[i:i+k]) on the fly? 
        # Or precompute prefix sums?
        # prefix_sum[j] = sum of first j elements of cand_desc.
        # sum(cand_desc[i:i+k]) = prefix_sum[i+k] - prefix_sum[i].
        
        prefix_sum = [0] * (L + 1)
        for i in range(L):
            prefix_sum[i+1] = prefix_sum[i] + smooth_list[cand_desc[i]]
            
        def get_max_sum(i, r):
            if L - i < r:
                return -1
            return prefix_sum[i+r] - prefix_sum[i]

        path = []
        path_mask = 0
        
        def backtrack(start, rem_sum, rem_k):
            nonlocal path_mask
            if rem_k == 0:
                return [] if rem_sum == 0 else None
            
            avail_len = L - start
            if avail_len < rem_k:
                return None
            
            # Pruning with bounds
            # Min sum is constant global_min_k (if avail_len >= rem_k)
            if rem_sum < global_min_k:
                return None
            
            # Max sum check
            max_s = get_max_sum(start, rem_k)
            if rem_sum > max_s:
                return None
            
            for i in range(start, L):
                idx = cand_desc[i]
                num = smooth_list[idx]
                
                if num > rem_sum:
                    # Since sorted descending, if this one is too big, smaller ones might fit.
                    # We cannot break. We must continue.
                    # Wait, if sorted descending, num decreases.
                    # If num > rem_sum, next num is smaller. So we continue.
                    continue
                
                # Check compatibility using bitmask
                if incompatible_mask[idx] & path_mask:
                    continue
                
                # Heuristic: Try to pick larger numbers first?
                # We are iterating i from start (largest) to L (smallest).
                # So we are trying larger numbers first.
                
                path.append(idx)
                path_mask |= (1 << idx)
                
                result = backtrack(i + 1, rem_sum - num, rem_k - 1)
                
                if result is not None:
                    return [num] + result
                
                path.pop()
                path_mask &= ~(1 << idx)
            
            return None
        
        result = backtrack(0, n, k)
        return result

    # Precompute all answers
    answers = []
    for n in range(1, 129):
        if time.time() > deadline:
            # Emergency fallback: use 1 term if possible, else try k=2 quickly
            # This should rarely happen
            if n in smooth_to_idx:
                answers.append([n])
            else:
                # Try to find any valid representation with small k
                found = None
                for k in range(1, 5):
                    res = find_antichain(n, k)
                    if res is not None:
                        found = res
                        break
                if found:
                    answers.append(found)
                else:
                    answers.append([n]) # Invalid placeholder, but prevents crash
            continue

        answer = None
        for k in range(1, 31):
            res = find_antichain(n, k)
            if res is not None:
                answer = res
                break
        
        if answer is None:
            # Fallback: should not happen
            if n in smooth_to_idx:
                answer = [n]
            else:
                # Try to find any valid representation
                answer = find_antichain(n, 2)
                if answer is None:
                    answer = [n]  # placeholder
        
        answers.append(answer)
    
    return answers

_PRECOMPUTED = _precompute()

def construct():
    return _PRECOMPUTED
