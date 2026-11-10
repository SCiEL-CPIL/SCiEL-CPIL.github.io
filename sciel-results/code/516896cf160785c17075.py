"""
Strategy: Deterministic Binary Search on Window Length with Bitmask DP for Feasibility.
"""
import time

# Fixed inputs
DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
N = len(DIVISORS)
MAX_VAL = 1000000

def _check_window(L):
    """
    Returns True if there exists a window of length L (span L-1) containing 
    distinct multiples for each divisor.
    Uses Bitmask DP to check if all divisors can be assigned distinct values 
    within [0, L-1] relative to a window start S.
    """
    # 1. Precompute valid residues for each divisor within [0, L-1]
    # valid_residues[i] is a bitmask of residues r in [0, L-1] such that r % d_i == 0
    valid_residues = []
    for d in DIVISORS:
        mask = 0
        # The smallest non-negative multiple of d is 0.
        # Multiples are 0, d, 2d, ...
        # We need r <= L-1
        for r in range(0, L, d):
            if r < L:
                mask |= (1 << r)
            else:
                break
        valid_residues.append(mask)
        
    # Optimization: If any divisor has no valid residue, return False
    for mask in valid_residues:
        if mask == 0:
            return False

    # 2. DP over subsets
    # dp[mask] = set of possible residues for the last divisor added (index = popcount(mask)-1)
    # Actually, we need to know which residues are used to ensure distinctness.
    # Standard subset DP: dp[mask] = set of residues r such that there exists a valid assignment
    # for the set of divisors in 'mask' where the last divisor (highest index in mask) takes residue r.
    
    # To speed up, we can iterate S (window start) and check if any S works.
    # However, checking all S is expensive.
    # Let's optimize the DP check for a single S first.
    
    def is_valid_start(S):
        # S is the minimum value in the window.
        # Values are S + r, where r in [0, L-1].
        # Condition: (S + r) % d_i == 0 => r % d_i == (-S) % d_i
        
        # Precompute required residues for each divisor for this specific S
        req_residues = []
        for i, d in enumerate(DIVISORS):
            target_r = (-S) % d
            # Find all r in [0, L-1] such that r % d == target_r
            mask = 0
            # Start from the smallest r >= 0 that satisfies r % d == target_r
            # r = target_r + k*d
            r = target_r
            while r < L:
                mask |= (1 << r)
                r += d
            req_residues.append(mask)
            if mask == 0:
                return False

        # DP: dp[mask] = set of residues used by the last divisor in 'mask'
        # We process divisors in order 0..N-1.
        # dp[0] = {0} (dummy, no residue)
        # Transition: for each mask, for each next divisor j not in mask, 
        # check if there is a residue r in req_residues[j] that is not used by others.
        
        # Efficient DP:
        # dp[mask] is a set of residues.
        # Actually, since we just need existence, we can store the set of available residues?
        # No, we need to know which residues are *taken*.
        
        # Let's use: dp[mask] = set of residues r such that there is a valid assignment for divisors in mask
        # where the divisor corresponding to the highest bit in mask takes residue r.
        # Wait, the divisors are fixed in order.
        # Let's iterate by adding one divisor at a time.
        
        # dp[mask] = set of residues for the last added divisor (index = popcount(mask)-1)
        # This works if we add divisors in a fixed order? No, the order of addition in the mask
        # doesn't have to match the divisor index, but we can enforce it or just use the highest bit.
        # Simpler: dp[mask] = set of residues used by the divisor at index (popcount(mask)-1)
        # assuming we build the set by adding divisors in increasing index order?
        # No, standard subset DP allows any order.
        # Let's stick to: dp[mask] = set of residues r such that there exists a valid assignment
        # for the set of divisors in 'mask' where the divisor with the largest index in 'mask' takes residue r.
        
        # Initialize
        dp = [set() for _ in range(1 << N)]
        dp[0] = {0} # dummy residue 0 for empty set
        
        for mask in range(1, 1 << N):
            # Find the highest set bit
            # This corresponds to the last divisor added in this specific construction?
            # No, this formulation is tricky. 
            # Let's use a different DP state:
            # dp[mask] = set of residues r such that there is a valid assignment for divisors in 'mask'
            # where the divisor at index i (where i is the highest set bit in mask) takes residue r.
            
            # To compute dp[mask], we look at all subsets prev_mask = mask without bit i.
            # And check if there is a residue r in req_residues[i] that is not in the set of residues used by prev_mask?
            # We don't store the *set* of used residues, only the last one. This is insufficient.
            
            # Correct Approach:
            # Since N=10, the number of masks is 1024.
            # We can store the *entire set of used residues*? No, that's 2^L states, too big.
            
            # Alternative: Backtracking with pruning is fast enough for N=10 if we order by constraints.
            # But we are inside a binary search loop.
            
            # Let's use a simpler Bitmask DP that tracks which residues are used.
            # State: (mask_divisors, mask_residues) -> too big.
            
            # Let's go back to Backtracking for the feasibility check of a specific S.
            # N=10. Depth 10.
            # Pruning: If a divisor has no available residue, fail.
            # This is likely fast enough for a single S.
            # But we need to check many S?
            
            # Optimization: We don't need to check all S.
            # We can just check S=0? No, the window can start anywhere.
            # However, if a solution exists for some S, does it exist for S=0? No.
            
            # Let's analyze the periodicity.
            # The condition (S+r) % d == 0 depends on S mod d.
            # The LCM of divisors is LCM(6,10,14,15,21,22,26,33,35,39).
            # This LCM is huge.
            
            # However, for a fixed L, if we find *any* valid S, we return True.
            # We can iterate S from 0 to LCM-1? No, LCM is too big.
            
            # Let's try a different angle.
            # Is it possible that the optimal window always aligns with 0?
            # No.
            
            # Let's use a randomized check for S if deterministic is too slow?
            # Or, notice that for small L, the number of valid residues per divisor is small.
            
            # Let's implement a fast backtracking for a given S.
            # And we will test a few candidate S values.
            # Which S values?
            # The residues r must be multiples of d (shifted by S).
            # For d=39, r must be in {0, 39, 78, ...} shifted.
            # The "tightest" constraint is the largest divisor.
            
            # Actually, we can just iterate S from 0 to max_divisor - 1?
            # No, S can be anything.
            
            # Let's try iterating S from 0 to 1000?
            # If L is small, maybe S doesn't need to be huge?
            # Wait, the values are <= 1,000,000.
            
            # Let's try a heuristic: Check S=0, S=1, ..., S=100.
            # If that fails, check random S.
            
            # But we need a GUARANTEED correct answer for the score.
            # The task asks to minimize the span.
            # If we miss the optimal S, we might return a larger L.
            
            # Let's look at the inspiration scores. 1.8333 is the best.
            # Seed is 1.0.
            # The span for seed (divisors themselves) is 39-6=33.
            # Score 1.0 = seed/value. If seed value is 33, score 1.0 means value=33?
            # Wait, "raw value of 0 is impossible".
            # If the seed returns the divisors [6,10,...,39], max=39, min=6, span=33.
            # If score = seed/value, and seed score is 1.0, then seed_value / seed_value = 1.
            # So the baseline value is 33.
            # An improvement means value < 33.
            # Score 1.8333 implies value = 33 / 1.8333 = 18.
            # So we need to find a span of 18 or less.
            
            # If we find span 18, score is 1.8333.
            # Can we go lower?
            # Min span >= 9 (since 10 distinct ints).
            # If span is 9, score = 33/9 = 3.66.
            
            # Let's aim for span < 18.
            
            # Back to algorithm:
            # For a given L, we want to know if there exists S such that
            # there are distinct r_i in [0, L-1] with (S+r_i) % d_i == 0.
            
            # We can iterate S from 0 to L-1?
            # No, S is the start of the window. The residues r are relative to S.
            # If we fix the set of values V = {v_1, ..., v_10}, then S = min(V).
            # The residues are r_i = v_i - S.
            # So S is determined by the values.
            
            # We can iterate over possible S values?
            # The values are bounded by 1,000,000.
            
            # Let's try a smarter search:
            # Iterate L from 9 upwards.
            # For each L, iterate S from 0 to LCM(DIVISORS) - 1? No.
            
            # Note that the condition (S+r) % d == 0 is periodic in S with period d.
            # The combined condition is periodic in S with period LCM(DIVISORS).
            # But we don't need to check all S.
            
            # Let's just check S = 0, 1, ..., 1000.
            # If a solution exists, is it likely to have a small S?
            # Not necessarily.
            
            # However, we can reformulate:
            # We need distinct v_i such that v_i % d_i == 0 and max(v)-min(v) <= L.
            # This is equivalent to: there exists an interval [X, X+L] such that
            # for each i, there is a multiple of d_i in [X, X+L], and we can pick distinct ones.
            
            # Let's iterate X from 0 to 1000000? Too slow.
            
            # Let's use the property that if L is small, the number of multiples in [X, X+L] is small.
            # For d=39, there is at most 1 multiple if L < 39.
            # If L < 39, each divisor has at most 1 multiple in the window (except small d).
            # If each divisor has exactly 1 multiple, then the values are FIXED for a given X.
            # Then we just check if they are distinct.
            
            # If L >= 39, some divisors might have 2 multiples.
            
            # Strategy:
            # 1. Iterate L from 9 to 50.
            # 2. For each L, iterate X from 0 to 10000 (heuristic bound).
            # 3. For each X, find all multiples of each d_i in [X, X+L].
            # 4. If any list is empty, skip X.
            # 5. If all lists have size 1, check if the 10 values are distinct.
            # 6. If some lists have size > 1, use backtracking to see if distinct assignment exists.
            
            # This is a structural change: Direct enumeration of windows X for small L.
            
            pass # Placeholder for logic below

    # Implementation of the heuristic search for L
    # We will search L from 9 upwards.
    # For each L, we will check X in a range.
    
    # To make it robust, we will check X from 0 to 10000.
    # If L is very small, the "period" of the pattern of available multiples is small.
    
    # Let's refine the check for a specific L and X.
    def check_X(X, L):
        candidates = []
        for d in DIVISORS:
            # Find multiples of d in [X, X+L]
            # First multiple >= X:
            k = (X + d - 1) // d
            m = k * d
            cands = []
            while m <= X + L:
                cands.append(m)
                m += d
            if not cands:
                return False, None
            candidates.append(cands)
        
        # Now we need to pick one from each list such that all are distinct.
        # Since L is small (<= 50), the total number of candidates is small.
        # Use backtracking.
        used = set()
        def backtrack(idx):
            if idx == N:
                return True
            for v in candidates[idx]:
                if v not in used:
                    used.add(v)
                    if backtrack(idx + 1):
                        return True
                    used.remove(v)
            return False
        
        if backtrack(0):
            return True, None
        return False, None

    # Search for minimal L
    # We will search L from 9 to 100.
    # For each L, we will check X from 0 to 10000.
    # To speed up, we can stop early if found.
    
    # However, this is inside _check_window(L) which is called by binary search?
    # No, we can just iterate L directly in construct().
    
    # Let's change the structure: construct() iterates L.
    
    return False # Fallback

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # 50s safety margin
    
    # We know seed span is 33.
    # We try to find smaller spans.
    
    # Precompute divisors
    divs = DIVISORS
    
    # Iterate L from 9 to 50
    for L in range(9, 51):
        if time.time() > deadline:
            break
            
        # Check X from 0 to 10000
        # Optimization: The pattern of multiples repeats every LCM(divs).
        # But we can just check a reasonable range.
        # If a solution exists, is it guaranteed to have X < 10000?
        # Not strictly, but highly likely for small L.
        # Let's check X from 0 to 20000 to be safer.
        
        found = False
        best_vals = None
        
        for X in range(0, 20001):
            if time.time() > deadline:
                break
            
            candidates = []
            valid = True
            for d in divs:
                k = (X + d - 1) // d
                m = k * d
                cands = []
                while m <= X + L:
                    cands.append(m)
                    m += d
                if not cands:
                    valid = False
                    break
                candidates.append(cands)
            
            if not valid:
                continue
            
            # Backtracking
            used = set()
            def backtrack(idx):
                if idx == 10:
                    return True
                for v in candidates[idx]:
                    if v not in used:
                        used.add(v)
                        if backtrack(idx + 1):
                            return True
                        used.remove(v)
                return False
            
            if backtrack(0):
                # Reconstruct values
                vals = [0] * 10
                used.clear()
                def reconstruct(idx):
                    if idx == 10:
                        return True
                    for v in candidates[idx]:
                        if v not in used:
                            used.add(v)
                            vals[idx] = v
                            if reconstruct(idx + 1):
                                return True
                            used.remove(v)
                    return False
                reconstruct(0)
                # Verify
                if len(set(vals)) == 10 and all(vals[i] % divs[i] == 0 for i in range(10)):
                    if max(vals) - min(vals) <= L:
                        return [int(v) for v in vals]
        
    # Fallback to seed if no better found
    return divs[:]

# EVOLVE-BLOCK-END
