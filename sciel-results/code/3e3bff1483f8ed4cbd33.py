import time
import random

_DEADLINE = 85.0

def _zero_sum_size_ge_2(zs):
    """
    Check if 0 can be formed as a sum of a subset of zs with size >= 2.
    zs are integers (can be negative, zero).
    """
    n = len(zs)
    if n < 2:
        return False
    
    # Separate positive and negative values for bounds
    pos = [z for z in zs if z > 0]
    neg = [-z for z in zs if z < 0]
    zeros = sum(1 for z in zs if z == 0)
    
    if zeros >= 2:
        return True
        
    total_pos = sum(pos)
    total_neg = sum(neg)
    
    # If total_pos != total_neg, it's still possible, but bounds help.
    # Range of sums: [-total_neg, total_pos]
    offset = total_neg
    max_bit = total_pos + total_neg
    
    # dp[k] is a bitset of achievable sums with size k
    # Bit i in dp[k] represents sum (i - offset)
    dp = [0] * (n + 1)
    dp[0] = 1 << offset # Sum 0 with size 0
    
    # Pruning arrays
    min_sum = [0] * (n + 1)
    max_sum = [0] * (n + 1)
    reachable = [False] * (n + 1)
    reachable[0] = True
    
    for z in zs:
        if z == 0:
            continue
        for k in range(n, 0, -1):
            if not reachable[k-1]:
                continue
            if z > 0:
                shifted = dp[k-1] << z
                dp[k] |= shifted
                new_min = min_sum[k-1] + z
                new_max = max_sum[k-1] + z
                if not reachable[k]:
                    min_sum[k] = new_min
                    max_sum[k] = new_max
                else:
                    if new_min < min_sum[k]: min_sum[k] = new_min
                    if new_max > max_sum[k]: max_sum[k] = new_max
                reachable[k] = True
            else:
                shifted = dp[k-1] >> (-z)
                dp[k] |= shifted
                new_min = min_sum[k-1] + z
                new_max = max_sum[k-1] + z
                if not reachable[k]:
                    min_sum[k] = new_min
                    max_sum[k] = new_max
                else:
                    if new_min < min_sum[k]: min_sum[k] = new_min
                    if new_max > max_sum[k]: max_sum[k] = new_max
                reachable[k] = True
    
    # Check if any k >= 2 has bit offset set (sum 0)
    # We can prune k if min_sum[k] > 0 or max_sum[k] < 0
    for k in range(2, n + 1):
        if reachable[k]:
            if min_sum[k] <= 0 <= max_sum[k]:
                if dp[k] & (1 << offset):
                    return True
    return False

def _can_form_sum(target, vals):
    """
    Check if target can be formed as a sum of a subset of vals with size >= 1.
    vals are integers (can be negative).
    Returns True if yes.
    """
    if not vals:
        return False
        
    n = len(vals)
    max_sum_abs = sum(abs(v) for v in vals)
    if abs(target) > max_sum_abs:
        return False
        
    offset = max_sum_abs
    dp = [0] * (n + 1)
    dp[0] = 1 << offset
    
    min_sum = [0] * (n + 1)
    max_sum = [0] * (n + 1)
    reachable = [False] * (n + 1)
    reachable[0] = True
    
    for v in vals:
        for k in range(n, 0, -1):
            if not reachable[k-1]:
                continue
            if v >= 0:
                shifted = dp[k-1] << v
                dp[k] |= shifted
                new_min = min_sum[k-1] + v
                new_max = max_sum[k-1] + v
                if not reachable[k]:
                    min_sum[k] = new_min
                    max_sum[k] = new_max
                else:
                    if new_min < min_sum[k]: min_sum[k] = new_min
                    if new_max > max_sum[k]: max_sum[k] = new_max
                reachable[k] = True
            else:
                shifted = dp[k-1] >> (-v)
                dp[k] |= shifted
                new_min = min_sum[k-1] + v
                new_max = max_sum[k-1] + v
                if not reachable[k]:
                    min_sum[k] = new_min
                    max_sum[k] = new_max
                else:
                    if new_min < min_sum[k]: min_sum[k] = new_min
                    if new_max > max_sum[k]: max_sum[k] = new_max
                reachable[k] = True
            
    bit_idx = offset + target
    for k in range(1, n + 1):
        if reachable[k]:
            if min_sum[k] <= target <= max_sum[k]:
                if dp[k] & (1 << bit_idx):
                    return True
    return False

def _is_valid_incremental(selected, x):
    """
    Check if x can be added to selected without violating constraints.
    """
    # Check 1: x is mean of subset of selected (size >= 2)
    if len(selected) >= 2:
        if _zero_sum_size_ge_2([y - x for y in selected]):
            return False

    # Check 2: any y in selected is mean of subset of selected U {x} containing x
    for y in selected:
        target = y - x
        if target == 0:
            continue
        vals = [z - y for z in selected if z != y]
        if not vals:
            continue
        if _can_form_sum(target, vals):
            return False
            
    return True

def construct():
    start_time = time.time()
    best_set = []
    
    candidates = list(range(1, 513))
    
    # Define several orders
    orders = []
    
    # 1. Spread (High-Low)
    spread = []
    lo, hi = 1, 512
    while lo <= hi:
        spread.append(hi)
        if lo != hi:
            spread.append(lo)
        lo += 1
        hi -= 1
    orders.append(spread)
    
    # 2. Spread (Low-High)
    orders.append(spread[::-1])
    
    # 3. Boundary focused
    boundary = []
    for i in range(256):
        boundary.append(1 + i)
        boundary.append(512 - i)
    orders.append(boundary)
    
    # 4. Randoms
    rng = random.Random(123)
    for _ in range(20):
        r = candidates[:]
        rng.shuffle(r)
        orders.append(r)
        
    for order in orders:
        if time.time() - start_time > _DEADLINE - 5:
            break
            
        selected = []
        sel_set = set()
        
        for x in order:
            if time.time() - start_time > _DEADLINE - 5:
                break
                
            if len(selected) >= 48:
                break
                
            if x in sel_set:
                continue
                
            # Fast pair check
            valid = True
            # 1. x mean of pair in selected
            for i in range(len(selected)):
                for j in range(i+1, len(selected)):
                    if selected[i] + selected[j] == 2 * x:
                        valid = False
                        break
                if not valid: break
            
            # 2. selected[i] mean of pair (x, selected[j])
            if valid:
                for a in selected:
                    y = 2 * a - x
                    if y in sel_set and y != a:
                        valid = False
                        break
                        
            if not valid:
                continue
                
            # Full incremental check
            if _is_valid_incremental(selected, x):
                selected.append(x)
                sel_set.add(x)
                
        if len(selected) > len(best_set):
            best_set = selected[:]
            
    # Final verification
    if best_set:
        C = best_set[:]
        for i in range(len(C)):
            a = C[i]
            others = C[:i] + C[i+1:]
            if len(others) >= 2:
                if _zero_sum_size_ge_2([y - a for y in others]):
                    return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        return best_set
    else:
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

if __name__ == "__main__":
    res = construct()
    print(len(res))
