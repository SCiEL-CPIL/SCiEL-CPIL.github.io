import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 85  # Safety margin before 90s kill

    MAX_SUM = 7680  # 30 * 256

    # Precompute divisibility: for each sum s, a bitset of all sums that conflict with s
    # conflict_bits[s] has bit j set if s and j are distinct and one divides the other
    # We'll use Python ints as bitsets
    # For speed, precompute for each s: multiples_bits[s] and divisors_bits[s]
    # Actually, let's precompute a single conflict bitset per s
    
    # For efficiency, we only need to check conflicts among sums that can actually appear
    # Let's precompute conflict relationships
    
    # conflict[s] = set of t != s such that s|t or t|s
    # But storing as sets is slow. Let's use a different approach:
    # For each s, precompute the list of multiples and divisors (as sorted lists)
    # Then during checking, we iterate and test membership in the current sum set
    
    # Actually, the fastest approach: use a bitset for the current sums,
    # and for each new sum u, check if any bit in conflict_bits[u] is set in the current bitset.
    # But conflict_bits[u] would need to be a huge integer with bits up to MAX_SUM.
    # That's 7680 bits = ~960 bytes per entry, times 7680 entries = ~7MB. Acceptable.
    
    # Let's build conflict_bits as a list of Python ints (bitsets)
    # conflict_bits[u] has bit v set iff v != u and (u % v == 0 or v % u == 0)
    
    # Actually, we need: for a new sum u, check if any existing sum v (v != 0, v != u) 
    # divides u or is divided by u.
    # So for each u, we want a bitset of all v where v|u or u|v, v != 0, v != u.
    
    # Let's build this:
    # For each u from 1 to MAX_SUM:
    #   - All divisors d of u (d != u, d >= 1): set bit d
    #   - All multiples m of u (m > u, m <= MAX_SUM): set bit m
    
    conflict_bits = [0] * (MAX_SUM + 1)
    for u in range(1, MAX_SUM + 1):
        bits = 0
        # Divisors of u (excluding u itself)
        for d in range(1, int(u**0.5) + 1):
            if u % d == 0:
                if d != u:
                    bits |= (1 << d)
                other = u // d
                if other != d and other != u:
                    bits |= (1 << other)
        # Multiples of u (excluding u itself)
        for m in range(u * 2, MAX_SUM + 1, u):
            bits |= (1 << m)
        conflict_bits[u] = bits
    
    # Now, the key operation: given a bitset of current sums `sums_bits` 
    # (bit s set means sum s is present), and a new sum u,
    # check if conflict_bits[u] & sums_bits != 0 (and also check u itself isn't already present)
    
    # For adding a number x to a set with subset sums represented by sums_bits:
    # 1. new_sums_bits = sums_bits | (sums_bits << x)  -- wait, this isn't right for arbitrary x
    #    Actually, subset sums: if we have sums S, adding x gives S ∪ {s+x for s in S}
    #    In bitset form: new_bits = bits | (bits << x)
    # 2. Check that no two sums conflict: for each new sum u in (bits << x),
    #    check conflict_bits[u] & (bits | (bits << x)) doesn't include any other sum
    #    Actually simpler: after computing all_bits = bits | (bits << x),
    #    check that for every u where all_bits has bit set, conflict_bits[u] & all_bits == 0
    #    But that's expensive. Better: only check the NEW sums against everything.
    #    new_sums = bits << x (these are the sums that were added)
    #    For each u in new_sums (iterate set bits), check conflict_bits[u] & all_bits
    #    But we also need to check that new_sums don't have duplicates with old sums
    #    (i.e., bits & (bits << x) should be 0 for distinctness)
    
    def try_add(x, sums_bits):
        """
        Try to add x to a set with subset sums represented by sums_bits.
        sums_bits is a Python int where bit s is set iff sum s is achievable.
        Returns (True, new_sums_bits) if valid, (False, 0) otherwise.
        """
        # Check distinctness: no existing sum s such that s+x is also an existing sum
        # i.e., sums_bits & (sums_bits << x) should be 0
        if sums_bits & (sums_bits << x):
            return (False, 0)
        
        new_sums_bits = sums_bits << x
        all_bits = sums_bits | new_sums_bits
        
        # Check divisibility for each new sum
        # Iterate over set bits in new_sums_bits
        temp = new_sums_bits
        while temp:
            # Get lowest set bit
            lsb = temp & (-temp)
            u = lsb.bit_length() - 1  # position of the bit
            # Check if any conflict exists
            if conflict_bits[u] & all_bits:
                return (False, 0)
            temp ^= lsb
        
        return (True, all_bits)
    
    def greedy_build(candidates, start_S=None, start_bits=None):
        """Greedy build: iterate through candidates, add if valid."""
        if start_S is None:
            S = []
            bits = 1  # bit 0 set (empty subset sum)
        else:
            S = list(start_S)
            bits = start_bits
        
        for x in candidates:
            if x in S:
                continue
            ok, new_bits = try_add(x, bits)
            if ok:
                S.append(x)
                bits = new_bits
        
        return S, bits
    
    def is_valid_full(S):
        """Verify a set is valid."""
        if not S:
            return True
        bits = 1
        for x in S:
            if bits & (bits << x):
                return False
            new_bits = bits << x
            all_bits = bits | new_bits
            temp = new_bits
            while temp:
                lsb = temp & (-temp)
                u = lsb.bit_length() - 1
                if conflict_bits[u] & all_bits:
                    return False
                temp ^= lsb
            bits = all_bits
        return True
    
    # Strategy: Compute a "conflict score" for each number 1..256
    # that estimates how many other numbers it would conflict with.
    # We'll use this to order candidates.
    
    def compute_conflict_scores():
        """
        For each number x in 1..256, estimate how many other numbers y in 1..256
        would create a conflict if both were in the set.
        A rough proxy: check if x and y have a common divisor relationship in their
        subset sums. For two single elements, the subset sums are {x}, {y}, {x+y}.
        Conflicts: x|y, y|x, x|(x+y) (always true if x|y... wait no), etc.
        
        Actually for two numbers x, y: sums are x, y, x+y.
        Conflicts: x|y, y|x, x|(x+y) [true iff x|y], y|(x+y) [true iff y|x], (x+y)|x [never], (x+y)|y [never]
        So conflict iff x|y or y|x.
        
        But for larger sets, it's more complex. Let's just use a heuristic:
        score[x] = number of y in 1..256 such that x|y or y|x
        """
        scores = {}
        for x in range(1, 257):
            count = 0
            for y in range(1, 257):
                if y == x:
                    continue
                if x % y == 0 or y % x == 0:
                    count += 1
            scores[x] = count
        return scores
    
    conflict_scores = compute_conflict_scores()
    
    # Strategy 1: Greedy with conflict-score ordering
    def greedy_by_conflict_score(order='asc'):
        """Greedy build ordering candidates by conflict score."""
        if order == 'asc':
            # Prefer numbers with fewer conflicts (more "free")
            candidates = sorted(range(1, 257), key=lambda x: conflict_scores[x])
        else:
            # Prefer numbers with more conflicts (more "selective")
            candidates = sorted(range(1, 257), key=lambda x: -conflict_scores[x])
        
        S, bits = greedy_build(candidates)
        return S, bits
    
    # Strategy 2: Randomized greedy with multiple restarts
    def randomized_greedy(n_restarts=5000):
        best_S = []
        best_bits = 1
        
        for r in range(n_restarts):
            if time.time() - start_time > time_limit - 10:
                break
            
            # Random perturbation of candidate order
            candidates = list(range(1, 257))
            
            # Mix different ordering strategies
            strategy = random.random()
            if strategy < 0.2:
                candidates.sort(key=lambda x: conflict_scores[x])
            elif strategy < 0.4:
                candidates.sort(key=lambda x: -conflict_scores[x])
            elif strategy < 0.6:
                candidates.sort(key=lambda x: x)  # ascending
            elif strategy < 0.8:
                candidates.sort(key=lambda x: -x)  # descending
            else:
                random.shuffle(candidates)
            
            # Add some randomness
            if random.random() < 0.3:
                # Swap a few random pairs
                for _ in range(random.randint(1, 5)):
                    i, j = random.sample(range(len(candidates)), 2)
                    candidates[i], candidates[j] = candidates[j], candidates[i]
            
            S, bits = greedy_build(candidates)
            
            if len(S) > len(best_S):
                best_S = S[:]
                best_bits = bits
        
        return best_S, best_bits
    
    # Strategy 3: Local search - try to swap out a bad element for a better one
    def local_search(start_S, start_bits, max_rounds=50):
        S = list(start_S)
        bits = start_bits
        
        for round_num in range(max_rounds):
            if time.time() - start_time > time_limit - 10:
                break
            
            improved = False
            for i in range(len(S)):
                if time.time() - start_time > time_limit - 10:
                    break
                
                removed = S[i]
                S_reduced = S[:i] + S[i+1:]
                
                # Recompute bits without removed
                reduced_bits = 1
                for x in S_reduced:
                    reduced_bits = reduced_bits | (reduced_bits << x)
                
                # Try to add each candidate not in S_reduced
                for x in range(1, 257):
                    if x in S_reduced:
                        continue
                    ok, new_bits = try_add(x, reduced_bits)
                    if ok:
                        # We can swap: remove S[i], add x
                        # But we need the size to increase, which it doesn't in a swap.
                        # For size increase, we'd need to add without removing.
                        # Let's just do add-only first.
                        pass
            
            # Try to add any element to the current set
            for x in range(1, 257):
                if x in S:
                    continue
                ok, new_bits = try_add(x, bits)
                if ok:
                    S.append(x)
                    bits = new_bits
                    improved = True
                    break
            
            if not improved:
                break
        
        return S, bits
    
    # Strategy 4: Multi-start with local search
    def multi_start_local_search(n_starts=100):
        best_S = []
        best_bits = 1
        
        for s in range(n_starts):
            if time.time() - start_time > time_limit - 15:
                break
            
            # Get a starting set from greedy
            candidates = list(range(1, 257))
            random.shuffle(candidates)
            S, bits = greedy_build(candidates)
            
            # Improve with local search
            S2, bits2 = local_search(S, bits, max_rounds=20)
            
            if len(S2) > len(best_S):
                best_S = S2[:]
                best_bits = bits2
        
        return best_S, best_bits
    
    # Run strategies
    best_S = [31, 32, 34]  # fallback
    best_bits = None
    
    # Quick greedy with different orderings
    for order in ['asc', 'desc']:
        S, bits = greedy_by_conflict_score(order)
        if len(S) > len(best_S):
            best_S = S[:]
            best_bits = bits
    
    # Randomized greedy
    S, bits = randomized_greedy(3000)
    if len(S) > len(best_S):
        best_S = S[:]
        best_bits = bits
    
    # Multi-start local search with remaining time
    if time.time() - start_time < time_limit - 20:
        S, bits = multi_start_local_search(50)
        if len(S) > len(best_S):
            best_S = S[:]
            best_bits = bits
    
    # Final local search on best
    if time.time() - start_time < time_limit - 5:
        S, bits = local_search(best_S, best_bits, max_rounds=100)
        if len(S) > len(best_S):
            best_S = S[:]
            best_bits = bits
    
    # Verify
    if len(best_S) > 0 and not is_valid_full(best_S):
        best_S = [31, 32, 34]
    
    return list(best_S)

if __name__ == "__main__":
    res = construct()
    print(res)
