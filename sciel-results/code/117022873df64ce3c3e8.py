import time
from fractions import Fraction

def _sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n+1, i):
                is_prime[j] = False
    return [i for i in range(2, n+1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    primes = _sieve(1000)
    # We can go up to 10^5 as per contract, but let's start with a manageable horizon for speed
    # The bitset approach is fast enough for 10^5 bits (approx 12.5KB per int)
    L_MAX = 100000
    
    # Precompute bitsets for each prime residue?
    # No, residues depend on the specific n. But we can precompute the pattern for a prime p.
    # Pattern for (p, r) covers k where k % p == r.
    # We can generate the bitset on the fly or cache it.
    # Given p <= 1000 and L_MAX = 10^5, there are at most 168 primes * 1000 residues = 168k patterns.
    # Too many to precompute all. But we only use a few.
    # Let's generate the bitset for a (p, r) pair when needed.
    
    def make_bitset(p, r):
        bits = 0
        # k is 1-based index in the problem (1..L_MAX)
        # Bit i in the integer represents coverage of integer i.
        # We want bits for k in 1..L_MAX such that k % p == r.
        # Note: if r == 0, it covers p, 2p, ...
        # if r != 0, it covers r, r+p, ... (if r>=1)
        
        # Optimization: Use a loop or a clever bit manipulation.
        # For small p, the pattern repeats.
        # Let's just loop, it's fast enough if we do it rarely.
        # Actually, we can construct it faster.
        # Mask for positions 0..L_MAX-1.
        # Position k (1-based) corresponds to bit k.
        
        # Let's just use a simple loop for clarity and correctness first, 
        # but optimize if it's a bottleneck.
        # L_MAX = 100000. Looping 100000 times is ~0.01s in Python.
        # If we do this for every move in local search, it might be slow.
        # Let's try to compute the "gain" without full bitset if possible?
        # No, we need the full coverage to find the next gap.
        
        # Better: Precompute the "base" pattern for each prime p?
        # Pattern for p is a repeating block of p bits.
        # Let's just generate the bitset once per (p,r) usage in the search.
        
        bits = 0
        # Start at the smallest k >= 1 such that k % p == r
        if r == 0:
            start_k = p
        else:
            start_k = r
            if start_k == 0:
                start_k = p # r is in [0, p-1], so if r=0 handled above.
        
        for k in range(start_k, L_MAX + 1, p):
            bits |= (1 << k)
            
        return bits

    # Cache for bitsets
    bitset_cache = {}
    def get_bitset(p, r):
        key = (p, r)
        if key not in bitset_cache:
            bitset_cache[key] = make_bitset(p, r)
        return bitset_cache[key]

    def compute_L(coverage_bits):
        # Find the highest set bit in the prefix 1..L_MAX
        # We need the largest L such that bits 1..L are all set.
        # This is equivalent to finding the first 0 bit in the range 1..L_MAX.
        # Let x = coverage_bits.
        # We want to find the smallest k >= 1 such that bit k is 0.
        # Then L = k - 1.
        # If all bits 1..L_MAX are set, L = L_MAX.
        
        # Trick: 
        # If we invert the bits in the range 1..L_MAX, the first set bit is the first gap.
        # Mask for 1..L_MAX: (1 << (L_MAX + 1)) - 2  (bits 1 to L_MAX set)
        mask = (1 << (L_MAX + 1)) - 2
        # Invert coverage within mask
        gaps = (~coverage_bits) & mask
        if gaps == 0:
            return L_MAX
        
        # Find index of least significant set bit
        # Python doesn't have a built-in, but we can use bit_length on gaps & -gaps
        # Or just loop? 
        # Since L_MAX is 10^5, the integer has ~10^5 bits.
        # `gaps & -gaps` isolates the lowest set bit.
        # The index is (lsb.bit_length() - 1).
        lsb = gaps & -gaps
        first_gap = lsb.bit_length() - 1
        return first_gap - 1

    def greedy_build(deadline, strategy):
        coverage = 0
        used_primes = set()
        budget = Fraction(1, 1)
        classes = []
        
        n = 1
        # We iterate until we can't extend or time runs out
        # We don't need to iterate to L_MAX if we cover it.
        
        while True:
            if time.time() > deadline:
                break
            
            L = compute_L(coverage)
            if L >= L_MAX:
                break
                
            U = L + 1 # First uncovered number
            
            best_p = None
            best_r = 0
            best_score = -1
            
            # Heuristic for scoring depends on strategy
            # Strategy 0: Gain * p (efficiency)
            # Strategy 1: Gain (raw)
            # Strategy 2: Prefer small p (cheap)
            
            for p in primes:
                if p in used_primes:
                    continue
                cost = Fraction(1, p)
                if budget < cost:
                    continue
                
                r = U % p
                # Estimate gain: how many new numbers in 1..L_MAX does this cover?
                # We can approximate by checking a window around U or just count total.
                # Counting total is O(L_MAX/p). Sum over all p is O(L_MAX * sum 1/p) ~ O(L_MAX log log 1000).
                # 100000 * 3 = 300000 operations. This is fast.
                
                # Let's just calculate the exact gain for the candidate (p, r)
                # Gain = count of k in 1..L_MAX such that k%p==r and bit k in coverage is 0.
                # This is (count of k in 1..L_MAX with k%p==r) - (count of k in 1..L_MAX with k%p==r and bit k in coverage is 1).
                # The second part is popcount(coverage & get_bitset(p, r)).
                # popcount is fast.
                
                mask_pr = get_bitset(p, r)
                # Total numbers covered by this class in 1..L_MAX
                # If r==0: floor(L_MAX/p)
                # If r!=0: 
                #   if L_MAX >= r: floor((L_MAX - r)/p) + 1 else 0
                if r == 0:
                    total_in_range = L_MAX // p
                else:
                    if L_MAX >= r:
                        total_in_range = (L_MAX - r) // p + 1
                    else:
                        total_in_range = 0
                        
                already_covered = (coverage & mask_pr).bit_count()
                gain = total_in_range - already_covered
                
                if gain == 0:
                    continue
                    
                if strategy == 0:
                    score = gain * p
                elif strategy == 1:
                    score = gain
                else: # Strategy 2: prefer small p
                    score = gain / p
                    
                if score > best_score:
                    best_score = score
                    best_p = p
                    best_r = r
            
            if best_p is None:
                break
                
            used_primes.add(best_p)
            budget -= Fraction(1, best_p)
            classes.append([best_p, best_r])
            
            coverage |= get_bitset(best_p, best_r)
            
        return classes, budget

    best_L = 0
    best_classes = []
    
    # Phase 1: Multiple Greedy Strategies
    strategies = [0, 1, 2]
    for s in strategies:
        if time.time() > deadline - 10:
            break
        classes, budget = greedy_build(deadline - 5, s)
        if not classes:
            continue
        coverage = 0
        for p, r in classes:
            coverage |= get_bitset(p, r)
        L = compute_L(coverage)
        
        if L > best_L:
            best_L = L
            best_classes = [list(c) for c in classes]
            
        # Local Search on this candidate
        if time.time() > deadline - 5:
            continue
            
        used_primes = set(p for p, r in classes)
        remaining_budget = budget
        
        no_improve = 0
        max_no_improve = 20
        
        while time.time() < deadline - 2 and no_improve < max_no_improve:
            U = L + 1
            if U > L_MAX:
                break
                
            improved = False
            
            # Move 1: Add new prime
            # Sort candidates by efficiency (gain/cost) or just try all affordable
            # To speed up, limit candidates?
            # Let's try all affordable primes, but limit to those that actually cover U.
            # All primes cover U if r = U % p.
            
            candidates_add = []
            for p in primes:
                if p in used_primes:
                    continue
                cost = Fraction(1, p)
                if remaining_budget < cost:
                    continue
                r = U % p
                # Check gain
                mask_pr = get_bitset(p, r)
                total_in_range = L_MAX // p if r == 0 else ((L_MAX - r) // p + 1 if L_MAX >= r else 0)
                already_covered = (coverage & mask_pr).bit_count()
                gain = total_in_range - already_covered
                if gain > 0:
                    # Score: prefer high gain, low cost
                    candidates_add.append((gain * p, p, r, cost))
                    
            # Sort by score descending
            candidates_add.sort(key=lambda x: x[0], reverse=True)
            
            for _, p, r, cost in candidates_add[:5]: # Try top 5
                if time.time() > deadline - 2:
                    break
                new_coverage = coverage | get_bitset(p, r)
                new_L = compute_L(new_coverage)
                if new_L > L:
                    classes.append([p, r])
                    used_primes.add(p)
                    remaining_budget -= cost
                    coverage = new_coverage
                    L = new_L
                    if L > best_L:
                        best_L = L
                        best_classes = [list(c) for c in classes]
                    no_improve = 0
                    improved = True
                    break
                    
            # Move 2: Change residue of existing prime
            if not improved:
                candidates_change = []
                for idx, (p, old_r) in enumerate(classes):
                    new_r = U % p
                    if new_r == old_r:
                        continue
                    # Check if changing improves L
                    # We need to remove old bitset and add new bitset
                    old_mask = get_bitset(p, old_r)
                    new_mask = get_bitset(p, new_r)
                    
                    # New coverage = (coverage & ~old_mask) | new_mask
                    # But wait, other primes might cover parts of old_mask.
                    # So we can't just subtract old_mask.
                    # We must recompute coverage from scratch or maintain counts.
                    # Recomputing is O(N_classes * L_MAX/p).
                    # Given N_classes < 100 and L_MAX=10^5, this is ~10^6 ops.
                    # Acceptable for a few candidates.
                    
                    trial_classes = [list(c) for c in classes]
                    trial_classes[idx] = [p, new_r]
                    
                    trial_cov = 0
                    for pp, rr in trial_classes:
                        trial_cov |= get_bitset(pp, rr)
                        
                    new_L = compute_L(trial_cov)
                    if new_L > L:
                        classes = trial_classes
                        coverage = trial_cov
                        L = new_L
                        if L > best_L:
                            best_L = L
                            best_classes = [list(c) for c in classes]
                        no_improve = 0
                        improved = True
                        break
                        
            # Move 3: Swap - Remove one, Add one
            if not improved:
                # Try removing a prime that is "expensive" (large p? no, small p is expensive in cost 1/p? 
                # Wait, cost is 1/p. Small p has LARGE cost. Large p has SMALL cost.
                # We want to free up budget. So remove a prime with SMALL p (high cost).
                # Or remove a prime that contributes little to the prefix.
                
                # Let's try removing each prime and see if we can add a better one.
                # This is expensive. Limit to a few.
                # Sort classes by cost descending (small p first)
                sorted_classes = sorted(enumerate(classes), key=lambda x: -1.0/x[1][0])
                
                for idx_remove, (p_remove, r_remove) in sorted_classes[:3]:
                    if time.time() > deadline - 2:
                        break
                        
                    # Remove p_remove
                    trial_classes = [list(c) for c in classes if c[0] != p_remove]
                    trial_used = set(p for p, r in trial_classes)
                    trial_budget = remaining_budget + Fraction(1, p_remove)
                    
                    # Recompute coverage without p_remove
                    trial_cov = 0
                    for pp, rr in trial_classes:
                        trial_cov |= get_bitset(pp, rr)
                    
                    # Now try to add a new prime to cover U (which might be same or different)
                    # Actually, removing a prime might break the prefix.
                    # We need to find the new L' for trial_cov.
                    L_prime = compute_L(trial_cov)
                    
                    # If L_prime < L, we are worse off unless we can jump ahead.
                    # Let's try to add a prime to cover L_prime + 1 or U?
                    # If we remove a prime, the first gap might be earlier.
                    # Let's try to add a prime to cover the first gap of trial_cov.
                    U_prime = L_prime + 1
                    
                    # Find best add for trial state
                    best_add_score = -1
                    best_add_p = None
                    best_add_r = 0
                    best_add_cost = None
                    
                    for p_add in primes:
                        if p_add in trial_used:
                            continue
                        cost_add = Fraction(1, p_add)
                        if trial_budget < cost_add:
                            continue
                        r_add = U_prime % p_add
                        mask_add = get_bitset(p_add, r_add)
                        total_in_range = L_MAX // p_add if r_add == 0 else ((L_MAX - r_add) // p_add + 1 if L_MAX >= r_add else 0)
                        already_covered = (trial_cov & mask_add).bit_count()
                        gain = total_in_range - already_covered
                        
                        if gain > 0:
                            score = gain * p_add
                            if score > best_add_score:
                                best_add_score = score
                                best_add_p = p_add
                                best_add_r = r_add
                                best_add_cost = cost_add
                                
                    if best_add_p:
                        final_classes = trial_classes + [[best_add_p, best_add_r]]
                        final_cov = trial_cov | get_bitset(best_add_p, best_add_r)
                        final_L = compute_L(final_cov)
                        
                        if final_L > L:
                            classes = final_classes
                            coverage = final_cov
                            L = final_L
                            remaining_budget = trial_budget - best_add_cost
                            used_primes = set(p for p, r in classes)
                            if L > best_L:
                                best_L = L
                                best_classes = [list(c) for c in classes]
                            no_improve = 0
                            improved = True
                            break
                            
            if not improved:
                no_improve += 1

    # Final validation
    if best_classes:
        # Check budget
        total_cost = sum(Fraction(1, p) for p, r in best_classes)
        if total_cost > 1:
            # This should not happen, but if it does, we need to fix it?
            # The problem states sum <= 1.
            # If we exceeded it, the solution is invalid.
            # We should probably return a valid subset.
            # But greedy ensures we don't exceed.
            pass
            
        # Check distinct primes
        ps = [p for p, r in best_classes]
        if len(ps) != len(set(ps)):
            # Fix duplicates? Shouldn't happen.
            pass
            
    return best_classes

# EVOLVE-BLOCK-END
