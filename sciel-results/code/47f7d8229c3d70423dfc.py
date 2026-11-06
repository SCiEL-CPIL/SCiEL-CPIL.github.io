import time
import random

N = 5000
DEADLINE_MARGIN = 20.0
T_START = time.time()

def _time_left():
    return (330.0 - DEADLINE_MARGIN) - (time.time() - T_START)

def _primes(limit):
    """Sieve of Eratosthenes: list of primes <= limit."""
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = bytearray(len(sieve[p * p::p]))
    return [i for i in range(2, limit + 1) if sieve[i]]

def construct():
    # 1. Setup
    primes = _primes(N + 1)
    # We only care about primes <= current size, which starts at N+1.
    # As size decreases, we ignore larger primes.
    
    # Initial set
    S = set(range(N + 1))
    
    # counts[p][r] = number of elements in S congruent to r mod p
    # To save memory and time, we only store counts for primes that are "active" (p <= |S|)
    # But initially all are active.
    # Structure: dict mapping prime p to list of counts
    counts = {}
    for p in primes:
        cnt = [0] * p
        # Calculate initial counts for [0, N]
        # This is fast enough for 600 primes * 5000 elements? No.
        # Use formula:
        base = N + 1
        q, r = divmod(base, p)
        for c in range(p):
            cnt[c] = q + (1 if c < r else 0)
        counts[p] = cnt
        
    # Precompute x % p for all x in S? 
    # That's 5000 * 600 = 3M ops. Acceptable.
    # But we update counts dynamically.
    
    # Helper to check if a prime p is "bad" (all classes non-empty)
    def is_bad(p):
        return min(counts[p]) > 0

    # Helper to get minimal classes for a prime p
    def get_min_classes(p):
        cnt = counts[p]
        mn = min(cnt)
        return [r for r in range(p) if cnt[r] == mn]

    # Main Loop
    while _time_left() > 5.0:
        M = len(S)
        if M == 0:
            break
            
        # Check admissibility
        bad_primes = []
        for p in primes:
            if p > M:
                break
            if is_bad(p):
                bad_primes.append(p)
                
        if not bad_primes:
            break # Admissible!
            
        # Identify candidate elements for removal
        # An element x is a candidate if it belongs to a minimal class of some bad prime.
        candidates = set()
        # Map from candidate x to list of (p, r) where x is in min class of p
        # We just need to score them.
        
        # To find candidates efficiently:
        # For each bad prime p, find min class r.
        # All x in S such that x % p == r are candidates.
        # We can iterate through S and check if x is in min class of ANY bad prime.
        # This is O(|S| * |bad_primes|) which is 5000 * 600 = 3M.
        # Doing this every step is too slow if we have many steps.
        
        # Optimization: 
        # We only need to consider elements that are "rare".
        # Let's pick the "most urgent" prime first.
        # Urgency: smallest min-count?
        # Let's just pick the largest bad prime. Large primes have min-count 1 or 2.
        # Elements in their min-classes are very few.
        
        p_star = max(bad_primes)
        min_classes = get_min_classes(p_star)
        
        # Candidates are elements in S that are in min_classes of p_star
        # AND ideally also in min classes of other bad primes.
        
        # Let's gather candidates from ALL bad primes, but prioritize those with low counts.
        # Actually, just taking the min class of the largest prime is a good heuristic.
        # Let's try to score elements in the min class of p_star.
        
        cand_list = []
        for r in min_classes:
            # Find all x in S such that x % p_star == r
            # We can iterate through S?
            # Or maintain a list of elements per residue?
            # Maintaining lists is expensive to update.
            # Iterating S is 5000.
            for x in S:
                if x % p_star == r:
                    cand_list.append(x)
                    
        if not cand_list:
            # Should not happen if is_bad is true
            break
            
        # Score each candidate
        best_x = None
        best_score = -1
        
        for x in cand_list:
            score = 0
            # Check how many other bad primes this x helps
            for p in bad_primes:
                if p == p_star:
                    continue
                r_x = x % p
                cnt_x = counts[p][r_x]
                min_cnt = min(counts[p])
                if cnt_x == min_cnt:
                    # x is in a minimal class of p
                    score += 1
                    # Bonus if removing x empties the class
                    if cnt_x == 1:
                        score += 10 
            if score > best_score:
                best_score = score
                best_x = x
                
        if best_x is None:
            # Fallback: remove random element from min class
            best_x = random.choice(cand_list)
            
        # Remove best_x
        S.remove(best_x)
        for p in primes:
            if p > len(S) + 1: # Slightly larger window to be safe
                break
            counts[p][best_x % p] -= 1

    # Final check and return
    # Ensure we return a list of distinct integers
    return list(S)
