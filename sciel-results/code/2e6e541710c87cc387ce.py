import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 4096
    max_sum = 3 * n
    # bytearray for O(1) lookup of used sums. Index is the sum.
    # We need to store 0 or 1.
    
    best_set = []
    best_len = 0
    
    # Initial guess: Power of 3s is a valid B_3[1] set
    # 1, 3, 9, 27, 81, 243, 729, 2187
    # Next is 6561 which is > 4096.
    # So size 8 is a safe fallback.
    power3_set = [3**i for i in range(8) if 3**i <= n]
    best_set = list(power3_set)
    best_len = len(power3_set)
    
    def is_valid_set(S):
        """Check if set S has all unique unordered triple sums."""
        if not S:
            return True
        sums_used = bytearray(max_sum + 1)
        S = list(S)
        k = len(S)
        for i in range(k):
            si = S[i]
            for j in range(i, k):
                sj = S[j]
                for l in range(j, k):
                    sl = S[l]
                    s = si + sj + sl
                    if sums_used[s]:
                        return False
                    sums_used[s] = 1
        return True

    def greedy_build_mrv(ordering, sums_used):
        """
        Build a set using MRV heuristic: at each step, pick the candidate
        with the fewest conflicts (most constrained) that is still valid.
        Uses NumPy for fast conflict counting.
        """
        S = []
        remaining = list(ordering)
        k = 0
        
        while remaining and time.time() < deadline:
            # Find the best candidate: valid one with fewest new sum collisions
            # For efficiency, we check candidates in a smart order
            best_idx = -1
            best_conflicts = float('inf')
            
            # To speed up, we can sample or check all if small
            # For now, check all remaining but use vectorized conflict count
            if len(remaining) > 500:
                # Sample a subset for speed
                sample_size = min(200, len(remaining))
                sample_indices = random.sample(range(len(remaining)), sample_size)
            else:
                sample_indices = range(len(remaining))
            
            for idx in sample_indices:
                x = remaining[idx]
                
                # Quick check: does 3*x conflict?
                if sums_used[3 * x]:
                    continue
                    
                # Count conflicts with existing S
                # New sums: 3x, 2x+s_i for each s_i in S, x+s_i+s_j for pairs
                conflicts = 0
                valid = True
                
                # Check 2x + s_i
                for si in S:
                    if sums_used[2 * x + si]:
                        valid = False
                        break
                if not valid:
                    continue
                    
                # Check x + s_i + s_j for pairs
                for i in range(len(S)):
                    for j in range(i, len(S)):
                        if sums_used[x + S[i] + S[j]]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                # Count how many new sums are "close" to existing ones (heuristic)
                # For MRV, we want the most constrained valid candidate
                # Use a simple metric: number of sums that are near occupied sums
                # This is expensive, so use a simpler proxy: prefer smaller x
                # (smaller x tends to be more constrained in sum space)
                
                # Actually, for B_3[1], smaller numbers create more dense sum regions
                # so they're more likely to conflict. Prefer larger x among valid?
                # Let's use: valid candidates, pick the one with smallest x
                # (greedy from small to large tends to work well for these problems)
                
                if x < best_conflicts:
                    best_conflicts = x
                    best_idx = idx
            
            if best_idx == -1:
                # No valid candidate found in sample, break
                break
                
            x = remaining.pop(best_idx)
            S.append(x)
            
            # Update sums_used
            sums_used[3 * x] = 1
            for si in S[:-1]:  # S[:-1] is the set before adding x
                sums_used[2 * x + si] = 1
            for i in range(len(S) - 1):
                for j in range(i, len(S) - 1):
                    sums_used[x + S[i] + S[j]] = 1
            
            k = len(S)
            
            # Early termination if we hit a good size
            if k >= 16:
                break
                
        return S

    # Strategy 1: Randomized Greedy with MRV and multiple seeds
    num_restarts = 100
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for seed in range(num_restarts):
        if time.time() > deadline:
            break
        
        random.seed(seed)
        
        # Create a randomized ordering with bias towards smaller numbers
        # but not purely sorted (to escape local minima)
        candidates = list(range(1, n + 1))
        
        # Mix: 50% small numbers, 50% random
        small = list(range(1, min(1000, n + 1)))
        large = list(range(min(1000, n) + 1, n + 1))
        random.shuffle(small)
        random.shuffle(large)
        
        # Interleave
        candidates = []
        si, li = 0, 0
        while si < len(small) or li < len(large):
            if si < len(small) and (li >= len(large) or random.random() < 0.5):
                candidates.append(small[si])
                si += 1
            else:
                candidates.append(large[li])
                li += 1
        
        sums_used = bytearray(max_sum + 1)
        S = greedy_build_mrv(candidates, sums_used)
        
        if len(S) > best_len:
            best_set = S
            best_len = len(S)
            
            if best_len >= 16:
                pass

    # Strategy 2: Local Search - try to add more elements to best set
    if best_len < 16 and time.time() < deadline - 5:
        current_S = list(best_set)
        
        sums_used = bytearray(max_sum + 1)
        for i in range(len(current_S)):
            si = current_S[i]
            for j in range(i, len(current_S)):
                sj = current_S[j]
                for k in range(j, len(current_S)):
                    sk = current_S[k]
                    sums_used[si + sj + sk] = 1
        
        # Scan for addable elements
        for x in range(1, n + 1):
            if x in current_S:
                continue
            if sums_used[3 * x]:
                continue
            
            valid = True
            for si in current_S:
                if sums_used[2 * x + si]:
                    valid = False
                    break
            if not valid:
                continue
                
            for i in range(len(current_S)):
                for j in range(i, len(current_S)):
                    if sums_used[x + current_S[i] + current_S[j]]:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Valid, add it
            current_S.append(x)
            sums_used[3 * x] = 1
            for si in current_S[:-1]:
                sums_used[2 * x + si] = 1
            for i in range(len(current_S) - 1):
                for j in range(i, len(current_S) - 1):
                    sums_used[x + current_S[i] + current_S[j]] = 1
                    
            if len(current_S) > best_len:
                best_set = current_S
                best_len = len(current_S)
                
            if len(current_S) >= 16:
                break

    # Final validation
    if best_set and not is_valid_set(best_set):
        best_set = [3**i for i in range(8) if 3**i <= n]

    return best_set
