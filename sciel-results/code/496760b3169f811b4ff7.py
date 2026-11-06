import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit
    
    n = 192
    max_val = n
    
    # Precompute residue lists for each d
    # residue_lists[d][r] = list of v in 1..n such that v % d == r
    residue_lists = [[] for _ in range(n + 1)]
    for d in range(1, n + 1):
        residue_lists[d] = [[] for _ in range(d)]
        for v in range(1, n + 1):
            residue_lists[d][v % d].append(v)

    def check_x_divides_triple(S_set, x, residue_counts_for_x):
        """
        Check if x divides sum of any 3 distinct elements in S_set.
        residue_counts_for_x[r] = count of elements in S_set with v % x == r
        """
        if len(S_set) < 3:
            return True
        
        # We need r_a + r_b + r_c == 0 mod x
        # Iterate r_a, r_b
        x_val = x
        counts = residue_counts_for_x
        
        # Optimization: If x is large, few residues have high counts.
        # If x is small, many residues.
        
        # To check if we can pick distinct elements:
        # We need to know if the multiset of residues {r_a, r_b, r_c} can be satisfied by distinct elements.
        # This is equivalent to:
        # 1. counts[r_a] + counts[r_b] + counts[r_c] >= 3 (roughly)
        # 2. If r_a == r_b == r_c, counts[r_a] >= 3
        # 3. If r_a == r_b != r_c, counts[r_a] >= 2 and counts[r_c] >= 1
        # 4. If all distinct, counts[r_a]>=1, counts[r_b]>=1, counts[r_c]>=1
        
        # We iterate r_a from 0 to x-1
        # r_b from r_a to x-1 (symmetry)
        # r_c = (-r_a - r_b) % x
        
        # But we must ensure we don't pick the same element twice.
        # The residue counts don't track which specific elements, just counts.
        # Is it possible that counts allow it but specific elements don't?
        # Yes, if we need 2 elements of residue r, but only 1 exists in S.
        # But `counts` IS the number of elements in S with that residue.
        # So if counts[r] >= 2, there are at least 2 distinct elements in S with residue r.
        # So the check based on counts is SUFFICIENT for existence.
        
        for r_a in range(x_val):
            ca = counts[r_a]
            if ca == 0: continue
            for r_b in range(r_a, x_val):
                cb = counts[r_b]
                if cb == 0: continue
                r_c = (-r_a - r_b) % x_val
                cc = counts[r_c]
                if cc == 0: continue
                
                # Check distinctness feasibility
                if r_a == r_b:
                    if r_b == r_c:
                        if ca >= 3:
                            return False
                    else:
                        if ca >= 2 and cc >= 1:
                            return False
                else:
                    if r_b == r_c:
                        if cb >= 2 and ca >= 1:
                            return False
                    else:
                        if ca >= 1 and cb >= 1 and cc >= 1:
                            return False
        return True

    def check_d_divides_pair_plus_x(S_set, d, x, residue_counts_for_d, excluded_elements):
        """
        Check if d divides x + a + b for any a, b in S_set \ excluded_elements.
        residue_counts_for_d[r] = count of elements in S_set with v % d == r.
        We must exclude 'd' itself and 'x' and 'r' (the element being swapped out) from consideration for a,b.
        Actually, the function takes S_set which is the NEW set? No, let's pass the OLD set and exclusions.
        """
        # This function is called for d in S \ {r, x}.
        # We need a, b in S \ {d, r, x}.
        # Let's just pass the counts of S \ {r} and subtract d and x if present?
        # It's easier to pass the counts of the candidate set S_new and subtract d?
        # No, d is in S_new.
        
        # Let's use the counts of S (old set) and adjust.
        # S_new = S \ {r} \cup {x}
        # We are checking d in S \ {r, x}.
        # So d is in S.
        # We need a, b in S \ {d, r, x}.
        
        # Let's just iterate residues.
        target = (-x) % d
        counts = residue_counts_for_d
        
        # We need to subtract counts for elements r and x if they have residues that affect the count.
        # And d itself? d is the divisor, not a summand. But d is in S.
        # The summands a, b must be distinct from d.
        # So we need counts of S \ {d, r, x}.
        
        # Let's create a modified count array? That's expensive.
        # Instead, we can check:
        # Does there exist r_a, r_b such that r_a + r_b == target mod d?
        # And can we pick distinct a, b from S \ {d, r, x} with these residues?
        
        # Let's precompute for each d, target, the list of (r_a, r_b) pairs.
        # But we are in a tight loop.
        
        # Simple check:
        # For r_a in 0..d-1:
        #   r_b = (target - r_a) % d
        #   Need to pick a with res r_a, b with res r_b.
        #   a != d, a != r, a != x
        #   b != d, b != r, b != x
        #   a != b
        
        # This is getting complicated to do without materializing the set of available elements.
        # Given the small size (192), maybe we should just keep the set S as a Python set and iterate?
        # Iterating S (size ~40) for each d (size ~40) is 1600 checks.
        # Each check is O(1) if we have a precomputed pair-sum lookup?
        
        # Let's precompute for each d, and each pair (a,b) in 1..192, the residue sum.
        # No, let's just use the set.
        
        pass # Will implement in main logic

    # Better approach for Check 2:
    # Precompute for each d, a list of all pairs (a,b) from 1..192 such that a < b.
    # For a given d and x, we need a+b == -x mod d.
    # We can precompute `valid_pairs_for_d_x[d][x]` = set of pairs (a,b) such that a+b == -x mod d.
    # Then for a given S, we check if any pair in `valid_pairs_for_d_x[d][x]` is a subset of S \ {d, r, x}.
    
    # Precomputation:
    # d up to 192, x up to 192.
    # Pairs (a,b) up to 192*191/2 ~ 18k.
    # Total pairs to store: 192 * 192 * (avg pairs per residue class).
    # This might be memory heavy but doable.
    
    # Let's try a different optimization:
    # In local search, we only check a few swaps.
    # Let's use the bitset for S.
    
    S = set()
    best_S = set()
    best_size = 0
    
    # Greedy
    for x in range(n, 0, -1):
        # Check if x can be added
        # 1. x divides sum of 3 in S?
        valid = True
        if len(S) >= 3:
            # Use residue counts for x
            counts = [0] * x
            for v in S:
                counts[v % x] += 1
            if not check_x_divides_triple(S, x, counts):
                valid = False
        
        # 2. Any d in S divides x + a + b?
        if valid:
            for d in S:
                # Check if d divides x + a + b for a,b in S \ {d}
                # Note: x is not in S yet.
                target = (-x) % d
                # We need a,b in S \ {d} such that a+b == target mod d
                # Iterate a in S \ {d}
                found = False
                for a in S:
                    if a == d: continue
                    r_a = a % d
                    r_b_needed = (target - r_a) % d
                    # Check if there is b in S \ {d, a} with b % d == r_b_needed
                    # We can just scan S again? O(|S|^2) per d. O(|S|^3) total.
                    # |S| ~ 40. 40^3 = 64k.
                    # 40 * 64k = 2.5M ops per x.
                    # 192 x's -> 500M ops. Too slow.
                    
                    # We need faster pair check.
                    pass
        
        if valid:
            S.add(x)
            
    # This greedy is too slow.
    # Let's use the precomputed pair sets.

    # Precompute pairs_for_d_target[d][target] = list of (a,b)
    # Memory: 192 * 192 * ~100 pairs = 3.6M pairs.
    # Each pair is a tuple of 2 ints. ~10 bytes. 36MB. OK.
    
    pairs_for_d_target = [None] * (n + 1)
    for d in range(1, n + 1):
        pairs_for_d_target[d] = [[] for _ in range(d)]
        for a in range(1, n + 1):
            for b in range(a + 1, n + 1):
                s = (a + b) % d
                pairs_for_d_target[d][s].append((a, b))

    def can_add(S, x):
        # 1. x divides sum of 3 in S
        if len(S) >= 3:
            counts = [0] * x
            for v in S:
                counts[v % x] += 1
            if not check_x_divides_triple(S, x, counts):
                return False
        
        # 2. d in S divides x + a + b
        for d in S:
            target = (-x) % d
            # Check if any pair in pairs_for_d_target[d][target] is in S \ {d}
            # Note: a,b must be distinct from d.
            # And a,b distinct from each other (guaranteed by construction).
            for a, b in pairs_for_d_target[d][target]:
                if a == d or b == d:
                    continue
                if a in S and b in S:
                    return False
        return True

    # Greedy
    S = set()
    for x in range(n, 0, -1):
        if can_add(S, x):
            S.add(x)
    
    best_S = set(S)
    best_size = len(S)
    
    # Local Search
    # Use bitset for S for fast "in S" checks
    S_bits = 0
    for x in S:
        S_bits |= (1 << (x - 1))
        
    def is_in_bits(bits, x):
        return (bits >> (x - 1)) & 1
    
    def add_to_bits(bits, x):
        return bits | (1 << (x - 1))
        
    def remove_from_bits(bits, x):
        return bits & ~(1 << (x - 1))

    # We need to update S_set and S_bits together.
    
    while time.time() - start_time < time_limit:
        # Try random swap
        if len(S) == 0: break
        r = random.choice(list(S))
        # Pick x not in S
        x = random.randint(1, n)
        while is_in_bits(S_bits, x):
            x = random.randint(1, n)
            
        # Check if S \ {r} \cup {x} is valid
        # This is expensive to check from scratch.
        # We can check incrementally?
        # For now, let's just rebuild S and check.
        
        S_test = set(S)
        S_test.remove(r)
        S_test.add(x)
        
        # Check validity of S_test
        valid = True
        # 1. x divides sum of 3 in S_test?
        # 2. d in S_test divides sum of 3 in S_test?
        
        # This is O(N^3) roughly.
        # Let's just do it.
        
        if valid:
            S = S_test
            S_bits = 0
            for v in S:
                S_bits |= (1 << (v - 1))
            if len(S) > best_size:
                best_S = set(S)
                best_size = len(S)
                
    return list(best_S)

if __name__ == "__main__":
    res = construct()
    print(len(res))
