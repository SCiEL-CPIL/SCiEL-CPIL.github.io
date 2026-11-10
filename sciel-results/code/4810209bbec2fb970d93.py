import time
import random

def construct():
    start = time.time()
    deadline = start + 85  # Leave 5s margin before 90s kill
    
    def is_valid_candidate(x, selected, sums_bitset):
        """
        Check if x can be added to 'selected'.
        1. x is not the mean of any subset of 'selected' of size >= 2.
           Condition: For no k >= 2, does there exist a subset of 'selected' of size k with sum k*x?
           sums_bitset[k] is a bitset where bit S is 1 if sum S is achievable by k elements.
        2. No a in 'selected' is the mean of any subset of (selected U {x}) containing x.
           Condition: For no a in selected, and no k >= 2, does there exist a subset T of 'selected' 
           of size k-1 such that sum(T) + x = k*a?
           i.e., sum(T) = k*a - x.
           We check if bit (k*a - x) is set in sums_bitset[k-1].
        """
        n = len(selected)
        
        # Check 1: x is not the mean
        for k in range(2, n + 1):
            if k > len(sums_bitset) - 1:
                continue
            # Check if sum k*x is achievable by k elements
            target = k * x
            if target < 0:
                continue
            # Bitset index check
            if sums_bitset[k] >> target & 1:
                return False
        
        # Check 2: No a in selected becomes a mean
        for a in selected:
            for k in range(2, n + 2):
                # Subset size k contains x, so k-1 elements from selected
                # Sum of those k-1 elements must be k*a - x
                target = k * a - x
                if target < 0:
                    continue
                if k - 1 > len(sums_bitset) - 1:
                    continue
                if sums_bitset[k - 1] >> target & 1:
                    return False
        
        return True

    def update_sums(selected, sums_bitset, x):
        """
        Update sums_bitset after adding x.
        sums_bitset[k] represents sums of k elements.
        New sums_bitset[k] = old_sums_bitset[k] | (old_sums_bitset[k-1] << x)
        """
        n = len(selected)
        new_sums = [0] * (n + 1)
        new_sums[0] = 1  # Bit 0 set for 0 elements sum 0
        
        for k in range(1, n + 1):
            # Old sums of k elements
            if k < len(sums_bitset):
                new_sums[k] |= sums_bitset[k]
            # Sums of k-1 elements from old set, plus x
            if k - 1 < len(sums_bitset):
                new_sums[k] |= (sums_bitset[k - 1] << x)
        
        return new_sums

    def verify_set(S):
        """
        Full verification of set S.
        Returns True if valid.
        """
        n = len(S)
        for i, a in enumerate(S):
            others = [S[j] for j in range(n) if j != i]
            m = len(others)
            for k in range(2, m + 1):
                target = k * a
                # DP for subset sum of size k
                # dp[j] is bitset of sums
                dp = [0] * (k + 1)
                dp[0] = 1
                for val in others:
                    for j in range(k, 0, -1):
                        dp[j] |= (dp[j - 1] << val)
                if dp[k] >> target & 1:
                    return False
        return True

    best_set = None
    best_len = 0
    
    # Try multiple random seeds
    for seed in range(50):
        if time.time() > deadline - 2:
            break
        rng = random.Random(seed)
        
        candidates = list(range(1, 513))
        rng.shuffle(candidates)
        
        selected = []
        # sums_bitset[k] is an integer where bit s is 1 if sum s is achievable by k elements
        # Max sum is 512 * 48 = 24576. So we need bits up to ~25000.
        # Python ints handle this natively.
        sums_bitset = [0] * (50)  # Enough for max 48 elements
        sums_bitset[0] = 1  # Sum 0 with 0 elements
        
        for x in candidates:
            if time.time() > deadline - 2:
                break
            
            if is_valid_candidate(x, selected, sums_bitset):
                selected.append(x)
                sums_bitset = update_sums(selected, sums_bitset, x)
                
                if len(selected) > best_len:
                    best_len = len(selected)
                    best_set = list(selected)
                
                if best_len >= 48:
                    break
        
        if best_len >= 48:
            break
    
    # If we didn't find 48, try a few more random restarts with different shuffles
    if best_len < 48:
        for seed in range(50, 100):
            if time.time() > deadline - 2:
                break
            rng = random.Random(seed + 1000)
            candidates = list(range(1, 513))
            # Bias towards larger numbers? Or smaller? 
            # Let's try reversing or sorting by some heuristic
            # Actually, just shuffle again
            rng.shuffle(candidates)
            
            selected = []
            sums_bitset = [0] * (50)
            sums_bitset[0] = 1
            
            for x in candidates:
                if time.time() > deadline - 2:
                    break
                
                if is_valid_candidate(x, selected, sums_bitset):
                    selected.append(x)
                    sums_bitset = update_sums(selected, sums_bitset, x)
                    
                    if len(selected) > best_len:
                        best_len = len(selected)
                        best_set = list(selected)
                    
                    if best_len >= 48:
                        break
            
            if best_len >= 48:
                break

    if best_set is None:
        # Fallback: return a small valid set
        return [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    # Final verification
    if not verify_set(best_set):
        # If verification fails (shouldn't happen if logic is correct), 
        # try to repair by removing problematic elements
        # For now, just return the best we have, but let's try to clean it up
        # Simple repair: remove elements that are means of others
        S = list(best_set)
        changed = True
        while changed:
            changed = False
            for i in range(len(S)):
                if not verify_set(S):
                    # Find which element is problematic?
                    # Just remove the last one added? Or random?
                    # Let's remove the one that is a mean if possible
                    # For simplicity, if verify fails, remove random element and re-check
                    # This is expensive, so only do a few steps
                    if time.time() > deadline - 1:
                        break
                    idx = random.randint(0, len(S) - 1)
                    S.pop(idx)
                    changed = True
            if time.time() > deadline - 1:
                break
        best_set = S
        best_len = len(best_set)
    
    return best_set
