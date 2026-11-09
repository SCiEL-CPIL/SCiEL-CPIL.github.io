"""Model-and-optimize approach to pack distinct multiples of fixed divisors."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from functools import reduce

P = {
    'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39],
    'maximum': 1000000
}

def construct():
    divisors = P['divisors']
    n = len(divisors)
    MAX_V = P['maximum']
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Precompute LCM of all divisors
    def lcm(a, b):
        return a * b // gcd(a, b)
    LCM = reduce(lcm, divisors)
    
    # Initial solution: use the divisors themselves
    naive = divisors[:]
    best_val = max(naive) - min(naive)
    best_assign = naive[:]
    
    # For each divisor d_i and each offset r in [0, LCM-1],
    # precompute which multiples of d_i fall in window [L, L+W] when L % LCM = r
    # Actually, we need: for a given W and L, which multiples of d_i are in [L, L+W]?
    # 
    # Key insight: The pattern of which multiples appear in a window of width W
    # depends only on L mod d_i for each divisor. Since d_i divides LCM, 
    # L mod d_i is determined by L mod LCM.
    #
    # For a window [L, L+W], the multiples of d in this window are:
    # k*d where ceil(L/d) <= k <= floor((L+W)/d)
    # The set of relative positions (k*d - L) depends on L mod d.
    
    # Precompute for each divisor index i and each residue r in [0, LCM-1]:
    # For window [L, L+W] with L % LCM = r, what are the candidate multiples?
    # But W varies, so let's think differently.
    #
    # Alternative: For a given W and L, the candidates for divisor d_i are the
    # multiples of d_i in [L, L+W]. Let's precompute for each d_i and each
    # possible "phase" (L mod d_i), the set of offsets where a multiple falls.
    #
    # For divisor d and phase p = L % d:
    # Multiples in [L, L+W] are at positions L + t where t is such that (L+t) % d == 0
    # i.e., t ≡ -p (mod d), so t = (d - p) % d, (d - p) % d + d, ...
    # up to W.
    # So the offsets are: ((d - p) % d) + j*d for j = 0, 1, ... while <= W
    #
    # This is fast to compute on the fly. The key optimization is to use
    # bitsets to represent which positions in the window are occupied by
    # which divisor's multiples.
    
    def check_window(W, L, divisors, deadline):
        """Check if window [L, L+W] has a valid distinct assignment using bitset propagation."""
        if L < 1 or L + W > MAX_V:
            return None
        
        # For each divisor, compute the set of offsets (relative to L) where multiples fall
        # Use Python ints as bitsets (arbitrary precision)
        # offset j means value L + j is a multiple of the divisor
        
        candidate_bitsets = []
        for d in divisors:
            p = L % d
            # First multiple offset: (d - p) % d
            first_offset = (d - p) % d
            if first_offset > W:
                return None  # No multiple in window
            # All offsets: first_offset, first_offset + d, first_offset + 2d, ...
            bits = 0
            for offset in range(first_offset, W + 1, d):
                bits |= (1 << offset)
            candidate_bitsets.append(bits)
        
        # Now we need to assign distinct positions to each divisor.
        # Position j can be assigned to divisor i iff bit j is set in candidate_bitsets[i].
        #
        # Use constraint propagation with bitsets:
        # For each position j, let possible_divisors[j] = bitmask of divisors that can use position j
        # For each divisor i, let available_positions[i] = candidate_bitsets[i]
        #
        # We need a perfect matching. Use a greedy + backtracking approach
        # with MRV heuristic, but accelerated with bitsets.
        
        # Build position -> possible divisors mapping
        # pos_possible[j] = bitmask of divisors that have bit j set
        pos_possible = [0] * (W + 1)
        for i in range(n):
            bits = candidate_bitsets[i]
            j = 0
            while bits:
                if bits & 1:
                    pos_possible[j] |= (1 << i)
                bits >>= 1
                j += 1
        
        # Check: each position must be assignable to at least one divisor
        # And each divisor must have at least one available position
        for i in range(n):
            if candidate_bitsets[i] == 0:
                return None
        
        # Use backtracking with MRV, but prune early using bitset operations
        assignment = [None] * n
        used_positions = 0  # bitset of used positions
        
        # Order divisors by number of available positions (MRV)
        order = sorted(range(n), key=lambda i: bin(candidate_bitsets[i] & ~used_positions).count('1'))
        
        def backtrack(idx):
            nonlocal used_positions
            if time.time() > deadline:
                return False
            if idx == n:
                return True
            
            i = order[idx]
            # Available positions for divisor i
            avail = candidate_bitsets[i] & ~used_positions
            
            # MRV: try positions in order of how many other divisors they conflict with
            # For each available position, count how many other unassigned divisors also want it
            positions = []
            bits = avail
            j = 0
            while bits:
                if bits & 1:
                    # Count conflicts: how many other unassigned divisors also have this position
                    conflicts = 0
                    for k in range(idx + 1, n):
                        if candidate_bitsets[k] & (1 << j):
                            conflicts += 1
                    positions.append((conflicts, j))
                bits >>= 1
                j += 1
            
            # Sort by conflicts (ascending) - prefer positions with fewer conflicts
            positions.sort()
            
            for _, j in positions:
                bit = 1 << j
                if avail & bit:
                    used_positions |= bit
                    assignment[i] = L + j
                    if backtrack(idx + 1):
                        return True
                    used_positions &= ~bit
                    assignment[i] = None
            
            return False
        
        if backtrack(0):
            return assignment[:]
        return None
    
    # Systematic search over W from theoretical minimum
    # Minimum possible span for 10 distinct integers is 9
    for W in range(9, 100):
        if time.time() > deadline:
            break
        
        # Due to LCM periodicity, L only needs to be tried modulo LCM
        # But LCM = 30030, so we try L in [1, min(LCM, MAX_V - W)]
        # Actually, we can try all L in [1, LCM] since the pattern repeats every LCM
        
        max_L = min(LCM, MAX_V - W)
        
        # Try all L values from 1 to max_L
        # This is at most 30030 values, each check is fast with bitsets
        found = False
        for L in range(1, max_L + 1):
            if time.time() > deadline:
                break
            
            result = check_window(W, L, divisors, deadline)
            if result is not None:
                span = max(result) - min(result)
                if span < best_val:
                    best_val = span
                    best_assign = result[:]
                    found = True
                    break  # Found solution with span W, move to next W
        
        if found and best_val == W:
            break
    
    # If we found a good solution, do local search to polish
    if best_assign is not None and best_val > 9:
        for iteration in range(200):
            if time.time() > deadline:
                break
            
            candidate = best_assign[:]
            improved = False
            
            idx = random.randint(0, n - 1)
            d = divisors[idx]
            current = candidate[idx]
            
            for delta in [-2, -1, 1, 2]:
                new_val = current + delta * d
                if new_val < 1 or new_val > MAX_V:
                    continue
                # Check distinctness
                if any(candidate[j] == new_val for j in range(n) if j != idx):
                    continue
                
                candidate[idx] = new_val
                span = max(candidate) - min(candidate)
                if span < best_val:
                    best_val = span
                    best_assign = candidate[:]
                    improved = True
                    break
                candidate[idx] = current
            
            if improved:
                iteration = 0
    
    # Final validation
    if best_assign is None:
        best_assign = divisors[:]
    
    result = []
    for i, d in enumerate(divisors):
        v = int(best_assign[i])
        result.append(v)
    
    # Verify all constraints
    for i, d in enumerate(divisors):
        assert isinstance(result[i], int) and not isinstance(result[i], bool)
        assert 1 <= result[i] <= MAX_V
        assert result[i] % d == 0
    assert len(set(result)) == n
    
    return result

# EVOLVE-BLOCK-END
