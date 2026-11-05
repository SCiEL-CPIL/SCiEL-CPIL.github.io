import time
import random

def _valid_mean_free(selected):
    """
    Returns True if the list 'selected' satisfies the condition that no element
    is the exact arithmetic mean of any subset of at least two OTHER elements.
    """
    n = len(selected)
    if n < 3:
        return True
    
    selected = list(set(selected))
    n = len(selected)
    
    for i, a in enumerate(selected):
        others = [x for x in selected if x != a]
        m = len(others)
        if m < 2:
            continue
            
        # DP: dp[k] is a bitset where bit s is set if sum s is achievable 
        # with exactly k elements from 'others'
        dp = [0] * (m + 1)
        dp[0] = 1  # sum 0 with 0 elements
        
        for x in others:
            # Update in reverse order of k to avoid reusing x multiple times
            for k in range(m, 0, -1):
                dp[k] |= (dp[k-1] << x)
                
        # Check if any k in [2, m] has a target sum k*a present in dp[k]
        for k in range(2, m + 1):
            target = k * a
            if dp[k] & (1 << target):
                return False
    return True

def _generate_candidates():
    """
    Generate candidates with prime factors only in {2, 3, 5}.
    These numbers have a multiplicative structure that tends to avoid
    additive mean relations.
    """
    cands = set()
    for i in range(10):  # 2^0 .. 2^9 (512)
        val_i = 2 ** i
        if val_i > 512:
            break
        for j in range(7):  # 3^0 .. 3^6 (729)
            val_ij = val_i * (3 ** j)
            if val_ij > 512:
                break
            for k in range(4):  # 5^0 .. 5^3 (125)
                val_ijk = val_ij * (5 ** k)
                if val_ijk > 512:
                    break
                cands.add(val_ijk)
    return list(cands)

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # 1. Generate structured candidates
    candidates = _generate_candidates()
    
    # 2. Greedy Construction
    # Try multiple orders to find a large valid set
    best_set = []
    
    # Fixed orders
    orders = [
        sorted(candidates),
        sorted(candidates, reverse=True),
        sorted(candidates, key=lambda x: x % 10), # Heuristic
    ]
    
    # Random orders
    for _ in range(10):
        lst = candidates[:]
        random.shuffle(lst)
        orders.append(lst)
        
    for order in orders:
        if time.time() > deadline:
            break
        current = []
        for x in order:
            if time.time() > deadline:
                break
            candidate_set = current + [x]
            if _valid_mean_free(candidate_set):
                current = candidate_set
        if len(current) > len(best_set):
            best_set = current[:]
            
    # 3. Local Search / Expansion
    # Try to add any remaining candidate from the pool
    if best_set:
        best_set_set = set(best_set)
        remaining = [x for x in candidates if x not in best_set_set]
        
        # Try to add elements one by one, re-shuffling if stuck
        improved = True
        while improved and time.time() < deadline:
            improved = False
            random.shuffle(remaining)
            for x in remaining:
                if time.time() > deadline:
                    break
                candidate_set = best_set + [x]
                if _valid_mean_free(candidate_set):
                    best_set = candidate_set
                    best_set_set.add(x)
                    remaining.remove(x)
                    improved = True
                    # After adding, re-shuffle remaining to try different orders
                    break
                    
    # 4. Try to add random numbers from 1..512 if we haven't reached 48
    # The structured set might be limited. Random numbers might fill gaps.
    if len(best_set) < 48:
        best_set_set = set(best_set)
        all_nums = list(range(1, 513))
        random.shuffle(all_nums)
        
        for x in all_nums:
            if time.time() > deadline:
                break
            if x in best_set_set:
                continue
            candidate_set = best_set + [x]
            if _valid_mean_free(candidate_set):
                best_set = candidate_set
                best_set_set.add(x)
                
    return best_set

if __name__ == "__main__":
    import sys
    s = construct()
    print(f"Found set of size {len(s)}")
    print(s)
    assert _valid_mean_free(s), "Invalid set found!"
    assert len(set(s)) == len(s), "Duplicates found!"
    assert all(1 <= x <= 512 for x in s), "Out of range!"
