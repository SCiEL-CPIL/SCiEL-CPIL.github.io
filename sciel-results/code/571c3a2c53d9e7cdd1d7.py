"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s kill
    N = 128
    FULL_MASK = (1 << N) - 1
    
    # Precompute conflict masks:
    # For a triple (a,b,c) with a<b<c, the forbidden condition is:
    #   a+b, a+c, b+c all in the set
    # For a pair (a,b) in the set, adding c creates a violation iff
    #   (a+b) in set AND (a+c) in set AND (b+c) in set
    # But for incremental checking when adding element x to current set S:
    #   x creates a violation iff there exist a,b in S such that
    #   (a+b) in S and (a+x) in S and (b+x) in S
    #   Equivalently: there exist a,b in S\{x} with a<b<x or a<x<b or x<a<b
    #   such that a+b, a+x, b+x are all in S∪{x}
    
    # Better approach: precompute for each pair (a,b) the set of c values
    # such that {a,b,c} forms a forbidden triple (i.e., a+b, a+c, b+c could all be in set)
    # Actually, the forbidden condition depends on which elements are in the set.
    # A triple {a,b,c} is forbidden iff a+b, a+c, b+c are ALL in the set.
    # So for a given set S, a triple {a,b,c} ⊆ S is a violation iff
    #   (a+b) ∈ S AND (a+c) ∈ S AND (b+c) ∈ S
    
    # For incremental check: when adding x to S, new violations are triples
    # {x, a, b} where a,b ∈ S and (x+a) ∈ S∪{x}, (x+b) ∈ S∪{x}, (a+b) ∈ S
    # Note: x+a = x+a, so we need x+a ∈ S (since x+a ≠ x unless a=0, but a≥1)
    # Similarly x+b ∈ S, and a+b ∈ S.
    # Also need x+a ≠ x+b (i.e., a≠b, already true) and x+a, x+b ≠ a+b possible.
    
    # Precompute: for each x, which pairs (a,b) with a<b, a,b≠x, 
    # have a+b, x+a, x+b all ≤ 128?
    # conflict_pair[x] = bitmask of pairs... this is complex.
    
    # Alternative: precompute for each x, the set of (a,b) pairs such that
    # if a,b ∈ S and x is added, then {x,a,b} is a violation.
    # This requires a+b ∈ S, x+a ∈ S, x+b ∈ S.
    # So for fixed x, the "conflict partners" depend on S.
    
    # Let's use a different strategy: precompute all forbidden triples.
    # A triple {a,b,c} (a<b<c) is "potentially forbidden" if a+b ≤ 128, a+c ≤ 128, b+c ≤ 128.
    # For a given set S, it's actually forbidden iff a+b, a+c, b+c ∈ S.
    
    # For fast checking: given set S (as bitmask), count violations by iterating
    # over all triples (a,b,c) with a<b<c all in S, and checking if a+b, a+c, b+c in S.
    # But that's O(n^3) which is slow.
    
    # Better: for each pair (a,b) in S, the "third element" c that would complete
    # a forbidden triple must satisfy: a+c ∈ S, b+c ∈ S, and a+b ∈ S.
    # If a+b ∉ S, then no c can form a forbidden triple with a,b.
    # If a+b ∈ S, then c must be such that a+c ∈ S and b+c ∈ S.
    # c must be in S, c ≠ a, c ≠ b.
    
    # Precompute for each pair (a,b): the set of c values where a+c and b+c are both ≤ 128.
    # Then for a given S, violation count for pair (a,b) is:
    #   if a+b ∈ S: |{c ∈ S : c ≠ a, c ≠ b, a+c ∈ S, b+c ∈ S}|
    
    # Let's precompute: for each a < b, the list of c > b such that a+c ≤ 128 and b+c ≤ 128.
    # And also c < a, a < c < b cases.
    
    # Actually, let me just precompute a 3D-ish structure differently.
    
    # For each element x, precompute which other elements y can "pair" with x
    # to potentially form part of a forbidden triple.
    
    # Simplest fast approach: precompute for each pair (a,b) a bitmask of all c
    # such that a+b, a+c, b+c are all ≤ 128. Then for set S, a violation exists
    # for triple {a,b,c} iff a+b∈S, a+c∈S, b+c∈S, and a,b,c∈S.
    
    # Let me precompute: pair_mask[a][b] = bitmask of c values (c≠a,c≠b) such that
    # a+b ≤ 128, a+c ≤ 128, b+c ≤ 128.
    # Then for set S, the number of violations is:
    #   sum over all a<b<c in S of [a+b∈S] * [a+c∈S] * [b+c∈S]
    
    # Even faster: for each a < b in S with a+b ∈ S, count c > b in S such that
    # a+c ∈ S and b+c ∈ S. Also count c < a and a < c < b.
    
    # Let me precompute for each (a,b) with a<b:
    #   higher_c[a][b] = bitmask of c > b such that a+c ≤ 128 and b+c ≤ 128
    #   lower_c[a][b] = bitmask of c < a such that a+c ≤ 128 and b+c ≤ 128  
    #   mid_c[a][b] = bitmask of c where a < c < b such that a+c ≤ 128 and b+c ≤ 128
    
    # For a set S (bitmask), violations = 0
    # For each a in S:
    #   For each b in S with b > a:
    #     if (a+b) in S:
    #       c_mask = higher_c[a][b] & S  # c > b in S
    #       violations += popcount(c_mask)
    #       c_mask = mid_c[a][b] & S     # a < c < b in S
    #       violations += popcount(c_mask)
    #       c_mask = lower_c[a][b] & S   # c < a in S
    #       violations += popcount(c_mask)
    
    # Precompute these masks
    # Index: use 0-based, so element i corresponds to bit (i-1)
    
    # Precompute higher_c, mid_c, lower_c as lists of lists
    # This is 128*128 = 16384 entries, each a 128-bit int. Fine.
    
    higher_c = [[0] * (N + 1) for _ in range(N + 1)]
    mid_c = [[0] * (N + 1) for _ in range(N + 1)]
    lower_c = [[0] * (N + 1) for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            h = 0
            m = 0
            l = 0
            for c in range(1, N + 1):
                if c == a or c == b:
                    continue
                if a + c > N or b + c > N:
                    continue
                if c > b:
                    h |= (1 << (c - 1))
                elif c < a:
                    l |= (1 << (c - 1))
                else:  # a < c < b
                    m |= (1 << (c - 1))
            higher_c[a][b] = h
            mid_c[a][b] = m
            lower_c[a][b] = l
    
    def popcount(x):
        """Count set bits in integer x."""
        count = 0
        while x:
            x &= x - 1
            count += 1
        return count
    
    def get_bits(mask):
        """Return list of numbers present in mask."""
        result = []
        for i in range(1, N + 1):
            if mask & (1 << (i - 1)):
                result.append(i)
        return result
    
    def count_violations(mask):
        """Count the number of violating triples using precomputed masks."""
        violations = 0
        for a in range(1, N + 1):
            if not (mask & (1 << (a - 1))):
                continue
            for b in range(a + 1, N + 1):
                if not (mask & (1 << (b - 1))):
                    continue
                # Check if a+b is in mask
                if a + b > N:
                    continue
                if not (mask & (1 << (a + b - 1))):
                    continue
                # a+b is in S, so count c values
                c_mask = higher_c[a][b] & mask
                violations += popcount(c_mask)
                c_mask = mid_c[a][b] & mask
                violations += popcount(c_mask)
                c_mask = lower_c[a][b] & mask
                violations += popcount(c_mask)
        return violations
    
    def initial_solution():
        mask = 0
        # Start with high numbers
        for i in range(64, 129):
            mask |= (1 << (i - 1))
        # Add some low numbers greedily
        for i in range(1, 64):
            new_mask = mask | (1 << (i - 1))
            if count_violations(new_mask) == 0:
                mask = new_mask
        return mask
    
    def random_flip(mask, num_flips=1):
        """Randomly flip num_flips bits."""
        for _ in range(num_flips):
            pos = random.randint(0, N - 1)
            mask ^= (1 << pos)
        return mask
    
    # Run simulated annealing
    best_mask = initial_solution()
    best_violations = count_violations(best_mask)
    best_size = popcount(best_mask)
    
    current_mask = best_mask
    current_violations = best_violations
    current_size = best_size
    
    temperature = 100.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Propose a move: flip 1-3 random bits
        num_flips = random.choice([1, 1, 1, 2, 2, 3])
        new_mask = random_flip(current_mask, num_flips)
        
        new_violations = count_violations(new_mask)
        new_size = popcount(new_mask)
        
        # Accept if better or with probability based on temperature
        if new_violations < current_violations or (new_violations == current_violations and new_size > current_size):
            current_mask = new_mask
            current_violations = new_violations
            current_size = new_size
            
            if current_violations == 0 and current_size > best_size:
                best_mask = current_mask
                best_size = current_size
        else:
            # Accept worse with probability based on temperature
            delta = (new_violations - current_violations) - (new_size - current_size) * 0.1
            if delta < 0:
                current_mask = new_mask
                current_violations = new_violations
                current_size = new_size
            else:
                prob = min(1.0, max(0.0, 1.0 - delta / (temperature + 1e-10)))
                if random.random() < prob:
                    current_mask = new_mask
                    current_violations = new_violations
                    current_size = new_size
        
        temperature *= cooling
        if temperature < 0.01:
            temperature = 0.01
    
    # If we found a valid solution, use it
    if best_violations == 0:
        result = get_bits(best_mask)
    else:
        # Fallback: use the high set plus a greedy low set
        mask = 0
        for i in range(64, 129):
            mask |= (1 << (i - 1))
        for i in range(1, 64):
            new_mask = mask | (1 << (i - 1))
            if count_violations(new_mask) == 0:
                mask = new_mask
        result = get_bits(mask)
    
    return result

# EVOLVE-BLOCK-END
