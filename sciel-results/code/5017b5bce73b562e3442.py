import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 85  # Safety margin before 90s kill
    
    best_set = [31, 32, 34]
    best_size = 3
    
    def get_sums(S):
        """Generate all non-empty subset sums of S."""
        sums = set()
        for x in S:
            new_sums = set(s + x for s in sums)
            sums = sums | new_sums
        return sums
    
    def is_valid(S):
        """Check if S is valid: distinct non-empty subset sums, no two divide each other."""
        if not S:
            return True
        if len(set(S)) != len(S):
            return False
        sums = get_sums(S)
        if not sums:
            return False
        # Check divisibility: no two distinct sums divide each other
        sum_list = list(sums)
        n = len(sum_list)
        for i in range(n):
            u = sum_list[i]
            for j in range(n):
                if i == j:
                    continue
                v = sum_list[j]
                if v > u and v % u == 0:
                    return False
        return True
    
    def is_valid_fast(S):
        """Faster validity check using sorted sums."""
        if not S:
            return True
        if len(set(S)) != len(S):
            return False
        sums = sorted(get_sums(S))
        n = len(sums)
        for i in range(n):
            u = sums[i]
            for j in range(i + 1, n):
                v = sums[j]
                if v % u == 0:
                    return False
        return True
    
    def try_add(S, x):
        """Check if adding x to valid set S keeps it valid."""
        if x in S:
            return False
        S_new = S + [x]
        return is_valid_fast(S_new)
    
    def local_search(S, max_iters=100):
        """Try to improve S by swapping elements."""
        S = list(S)
        improved = True
        iter_count = 0
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            if time.time() - start_time > time_limit - 5:
                break
            # Try removing each element and adding a different one
            for i in range(len(S)):
                if time.time() - start_time > time_limit - 5:
                    break
                S_without = S[:i] + S[i+1:]
                # Try adding each candidate not in S
                candidates = [x for x in range(1, 257) if x not in S_without]
                random.shuffle(candidates)
                for x in candidates:
                    if time.time() - start_time > time_limit - 5:
                        break
                    S_new = S_without + [x]
                    if is_valid_fast(S_new) and len(S_new) > len(S):
                        S = S_new
                        improved = True
                        break
                    # Even if same size, might lead to better future additions
                    # But for now, only accept improvements
            # Also try just adding
            candidates = [x for x in range(1, 257) if x not in S]
            random.shuffle(candidates)
            for x in candidates:
                if time.time() - start_time > time_limit - 5:
                    break
                S_new = S + [x]
                if is_valid_fast(S_new):
                    S = S_new
                    improved = True
                    break
        return S
    
    # Strategy 1: Superincreasing construction with randomization
    # Start with a random base, then build superincreasing
    
    # Precompute some good starting points
    # Numbers that are "good" to start with: large, prime-ish
    
    def build_superincreasing(base, target_size=15):
        """Build a superincreasing sequence starting from base."""
        S = [base]
        total = base
        for _ in range(target_size - 1):
            # Next element must be > total
            # Pick a random value > total, but <= 256
            if total >= 256:
                break
            # Choose from range (total, 256]
            # To ensure distinctness and avoid divisibility, pick randomly
            candidates = list(range(total + 1, 257))
            if not candidates:
                break
            # Prefer larger values to leave less room for divisibility issues
            # Actually, we want the sequence to grow fast
            # Pick a value that makes the next constraint tight
            # For now, random
            x = random.choice(candidates)
            S.append(x)
            total += x
        return S
    
    # Run multiple random restarts
    max_restarts = 500
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit - 5:
            break
        
        # Strategy A: Superincreasing with random start
        base = random.randint(1, 50)
        S = build_superincreasing(base)
        if len(S) > best_size:
            # Try to improve with local search
            S_improved = local_search(S, max_iters=20)
            if len(S_improved) > best_size:
                best_size = len(S_improved)
                best_set = S_improved[:]
        
        # Strategy B: Random greedy with bias toward larger numbers
        if time.time() - start_time > time_limit - 5:
            break
        S = []
        # Bias: prefer larger numbers first
        candidates = list(range(1, 257))
        # Weight toward larger numbers
        weights = [x ** 0.5 for x in candidates]
        # Random shuffle but try larger ones first sometimes
        random.shuffle(candidates)
        # Sort by a mix of random and size
        for x in candidates:
            if time.time() - start_time > time_limit - 5:
                break
            if try_add(S, x):
                S.append(x)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
        
        # Strategy C: Start from best_set and perturb
        if best_size > 3 and time.time() - start_time > time_limit - 10:
            S = best_set[:]
            random.shuffle(S)
            # Remove one random element, try to add a better one
            if len(S) > 1:
                idx = random.randint(0, len(S) - 1)
                removed = S[idx]
                S_without = S[:idx] + S[idx+1:]
                candidates = [x for x in range(1, 257) if x not in S_without and x != removed]
                random.shuffle(candidates)
                for x in candidates:
                    if time.time() - start_time > time_limit - 10:
                        break
                    S_new = S_without + [x]
                    if is_valid_fast(S_new):
                        S = S_new
                        break
                # Try to add more
                candidates = [x for x in range(1, 257) if x not in S]
                random.shuffle(candidates)
                for x in candidates:
                    if time.time() - start_time > time_limit - 10:
                        break
                    if try_add(S, x):
                        S.append(x)
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S[:]
    
    # Final verification
    if not is_valid(best_set):
        best_set = [31, 32, 34]
        best_size = 3
    
    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(res)
