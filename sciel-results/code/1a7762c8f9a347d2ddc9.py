import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin before 60s limit
    
    def check_valid_addition(S, x, sums_set):
        """Check if adding x to S preserves unique triple sums. Returns (valid, new_sums) or (False, None)."""
        n = len(S)
        new_sums = set()
        
        # (x,x,x)
        val = 3 * x
        if val in sums_set or val in new_sums:
            return False, None
        new_sums.add(val)
        
        # (x,x,s) for each s in S
        for s in S:
            val = 2 * x + s
            if val in sums_set or val in new_sums:
                return False, None
            new_sums.add(val)
        
        # (x,s,t) for s <= t in S
        for i in range(n):
            for j in range(i, n):
                val = x + S[i] + S[j]
                if val in sums_set or val in new_sums:
                    return False, None
                new_sums.add(val)
        
        return True, new_sums

    def is_valid_set(S):
        """Verify that all unordered triples in S have unique sums."""
        if not S:
            return True
        sums = set()
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    val = S[i] + S[j] + S[k]
                    if val in sums:
                        return False
                    sums.add(val)
        return True

    def greedy_forward(start_val=1, end_val=4096, step=1):
        S = []
        sums_set = set()
        for x in range(start_val, min(end_val + 1, 4097), step):
            valid, new_sums = check_valid_addition(S, x, sums_set)
            if valid:
                S.append(x)
                sums_set.update(new_sums)
        return S

    def randomized_greedy(seed_offset=0):
        """Randomized greedy."""
        S = []
        sums_set = set()
        # Bias towards smaller numbers initially to allow more room?
        # Or just shuffle.
        candidates = list(range(1, 4097))
        random.seed(42 + seed_offset)
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if x in S:
                continue
            valid, new_sums = check_valid_addition(S, x, sums_set)
            if valid:
                S.append(x)
                sums_set.update(new_sums)
        return S

    def local_search(S):
        """Simple local search: try to add any missing number."""
        S = list(S)
        sums_set = set()
        # Rebuild sums_set from scratch to ensure consistency
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    sums_set.add(S[i] + S[j] + S[k])
        
        progress = True
        while progress and time.time() < deadline:
            progress = False
            # Try to add any number from 1 to 4096
            for x in range(1, 4097):
                if time.time() > deadline:
                    break
                if x not in S:
                    valid, new_sums = check_valid_addition(S, x, sums_set)
                    if valid:
                        S.append(x)
                        sums_set.update(new_sums)
                        progress = True
                        break
        return S

    best = []
    
    # Phase 1: Systematic greedy passes
    # Small starts often yield larger sets because they pack tighter
    for start in range(1, 10):
        if time.time() > deadline - 5.0:
            break
        S = greedy_forward(start, 4096, 1)
        if len(S) > len(best):
            best = S[:]
    
    # Phase 2: Randomized greedy attempts
    attempts = 0
    while time.time() < deadline - 5.0:
        attempts += 1
        S = randomized_greedy(attempts)
        # Apply local search to try to fill gaps
        S = local_search(S)
        
        # Verify before accepting
        if is_valid_set(S) and len(S) > len(best):
            best = S[:]
    
    # Final check: if best is empty or small, try a known valid construction (powers of 2)
    # Powers of 2: 1, 2, 4, ..., 2048 (12 elements)
    if len(best) < 12:
        fallback = [1 << i for i in range(12)]
        if is_valid_set(fallback):
            best = fallback

    # Final verification of the best set
    if not is_valid_set(best):
        # Fallback to the smallest valid set if something went wrong
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        
    return best

# EVOLVE-BLOCK-END
