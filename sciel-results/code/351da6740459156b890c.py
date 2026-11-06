import time
from math import lcm, gcd

def construct():
    deadline = time.time() + 55
    
    # Numbers 2..256, index 0..254
    N = 255
    nums = list(range(2, 257))
    
    # Precompute incompatibility bitsets
    # compat_mask[i] = bitset of numbers incompatible with nums[i]
    compat_mask = [0] * N
    for i in range(N):
        for j in range(i+1, N):
            if lcm(nums[i], nums[j]) <= 256:
                compat_mask[i] |= (1 << j)
                compat_mask[j] |= (1 << i)
    
    # Sort by reciprocal descending: already in order 2,3,...,256
    # recip[i] = 1.0 / nums[i]
    recip = [1.0 / nums[i] for i in range(N)]
    
    # Greedy initial solution
    best_sum = 0.0
    best_set = set()
    remaining = (1 << N) - 1
    for i in range(N):
        if remaining & (1 << i):
            # Check if i is compatible with all in best_set
            ok = True
            for j in best_set:
                if compat_mask[i] & (1 << j):
                    ok = False
                    break
            if ok:
                best_set.add(i)
                best_sum += recip[i]
                # Remove incompatible
                remaining &= ~compat_mask[i]
                remaining &= ~(1 << i)
    
    # B&B
    # State: (pos, current_sum, current_set_bitset, remaining_bitset)
    # We process in order 0,1,2,...
    # Upper bound: current_sum + sum of recip[j] for j in remaining that are compatible with current_set
    
    import sys
    sys.setrecursionlimit(10000)
    
    best = [best_sum]
    best_solution = [set(best_set)]
    
    def upper_bound(current_set, remaining):
        # Sum of recip[j] for j in remaining where j is compatible with all in current_set
        ub = 0.0
        for j in range(N):
            if remaining & (1 << j):
                # Check compatibility with current_set
                ok = True
                for k in current_set:
                    if compat_mask[j] & (1 << k):
                        ok = False
                        break
                if ok:
                    ub += recip[j]
        return ub
    
    def branch_and_bound(pos, current_sum, current_set, remaining):
        if time.time() > deadline:
            return
        if pos >= N:
            if current_sum > best[0]:
                best[0] = current_sum
                best_solution[0] = set(current_set)
            return
        
        # Upper bound: current_sum + max possible from remaining
        ub = current_sum
        for j in range(pos, N):
            if remaining & (1 << j):
                ok = True
                for k in current_set:
                    if compat_mask[j] & (1 << k):
                        ok = False
                        break
                if ok:
                    ub += recip[j]
        
        if ub <= best[0]:
            return
        
        # Try including nums[pos] if possible
        if remaining & (1 << pos):
            ok = True
            for k in current_set:
                if compat_mask[pos] & (1 << k):
                    ok = False
                    break
            if ok:
                new_remaining = remaining & ~compat_mask[pos] & ~(1 << pos)
                branch_and_bound(pos+1, current_sum + recip[pos], current_set | {pos}, new_remaining)
        
        # Try excluding nums[pos]
        branch_and_bound(pos+1, current_sum, current_set, remaining & ~(1 << pos))
    
    branch_and_bound(0, 0.0, set(), (1 << N) - 1)
    
    result = [nums[i] for i in best_solution[0]]
    if not result:
        result = [2]
    return result
