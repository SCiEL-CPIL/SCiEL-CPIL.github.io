"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from itertools import combinations

P = {'n': 256}

def _is_prime(n):
    if n < 2:
        return False
    for d in range(2, isqrt(n) + 1):
        if n % d == 0:
            return False
    return True

def _valid_set(s):
    """Check if set s satisfies the constraint."""
    lst = list(s)
    for i in range(len(lst)):
        a = lst[i]
        for j in range(i+1, len(lst)):
            b = lst[j]
            for k in range(j+1, len(lst)):
                c = lst[k]
                if (b * c) % a == 0:
                    return False
    return True

def construct():
    deadline = time.time() + 85  # 5s safety margin
    n = P['n']
    candidates = list(range(2, n + 1))
    m = len(candidates)  # 255 numbers: 2..256
    
    # Map value to index
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    
    # Build conflict graph using bitsets
    # Edge between i and j means: there exists a third distinct member k in candidates
    # such that one of {i,j,k} divides the product of the other two.
    # More precisely: for any permutation (a,b,c) of three distinct members,
    # if a | b*c, then the set is invalid.
    # So vertices i and j are "conflicting" if there exists k != i,j such that:
    #   candidates[i] | candidates[j]*candidates[k]  OR
    #   candidates[j] | candidates[i]*candidates[k]  OR
    #   candidates[k] | candidates[i]*candidates[j]
    
    # For efficiency, precompute for each pair (i,j) whether they conflict.
    # Two numbers x,y conflict if:
    #   (1) There exists z in candidates, z!=x,y, such that x | y*z
    #   (2) There exists z in candidates, z!=x,y, such that y | x*z
    #   (3) There exists z in candidates, z!=x,y, such that z | x*y
    
    # Condition (3): z | x*y means z divides x*y. Since z is in 2..256,
    #   this is true if x*y has a divisor z in [2,256] with z != x, z != y.
    #   Actually we need z to be a member of our candidate set, which is all of 2..256.
    #   So condition (3) is: x*y has a divisor in 2..256 that is not x and not y.
    #   If x*y >= 2 and x*y has a proper divisor d where 2 <= d <= 256, d != x, d != y,
    #   then conflict. But also if x*y itself is in 2..256 and x*y != x and x*y != y,
    #   then z = x*y works. Wait, z must be distinct from both x and y.
    #   If x*y <= 256 and x*y != x and x*y != y, then z=x*y is a valid third member.
    #   Also any divisor of x*y in [2,256] distinct from x,y works.
    
    # Let's think differently. For the independent set, we need NO triple (a,b,c) 
    # distinct with a|b*c. This is equivalent to: for every pair (i,j) in our set,
    # there is NO k in our set (k distinct from i,j) such that any of the three
    # divisibility conditions hold.
    
    # So i and j are incompatible (edge in conflict graph) if there EXISTS k in 
    # candidates (k != i, k != j) such that:
    #   candidates[i] | candidates[j] * candidates[k]  OR
    #   candidates[j] | candidates[i] * candidates[k]  OR
    #   candidates[k] | candidates[i] * candidates[j]
    
    # Precompute conflict as bitsets
    # adj[i] is a bitset of all j such that i and j conflict
    
    adj = [0] * m  # will use Python ints as bitsets
    
    # For each pair (i, j), check if they conflict
    # To make this faster, let's precompute some helper structures
    
    # For condition (3): z | x*y where z is in 2..256, z != x, z != y
    # This is true if x*y has any divisor d in [2, 256] with d != x and d != y
    # Note: if x*y < 2, impossible. x,y >= 2 so x*y >= 4.
    # The smallest divisor of x*y is at most sqrt(x*y).
    
    # For conditions (1) and (2): x | y*z means z must be a multiple of x/gcd(x,y) 
    #   divided by... actually x | y*z iff (x/gcd(x,y)) | z.
    #   So we need z in candidates such that z is a multiple of (x/gcd(x,y)) and z != x, z != y.
    #   Let g = gcd(x,y), x' = x/g. Need z multiple of x' with z in [2,256], z != x, z != y.
    
    from math import gcd
    
    # Precompute for each x in 2..256, the set of multiples in 2..256
    # multiples_of[x] = set of z in 2..256 where z % x == 0
    # But we need this for x' = x/gcd(x,y) which can be any value 1..256
    # If x' == 1, then every z works (as long as z != x, z != y)
    
    # Let's precompute: for each d in 1..256, which candidates are multiples of d
    # mult_set[d] = frozenset of indices where candidates[idx] % d == 0
    # But d can be 1, and candidates are 2..256, so all are multiples of 1
    
    # Actually, let's just build the conflict graph directly. With 255 vertices,
    # that's ~32k pairs. For each pair, checking existence of k can be done quickly.
    
    # Precompute: for each value v in 2..256, list of indices where candidate is multiple of v
    # And for each value v, list of indices where candidate is a divisor of v (v % candidate == 0)
    
    # For condition (1): x | y*z. Let g = gcd(x,y), x' = x//g.
    #   Need z in candidates, z != x, z != y, such that z is a multiple of x'.
    #   If x' == 1: need any z in candidates not equal to x or y. Since |candidates| = 255 >= 3,
    #     this is always true (there's always a third element). So if gcd(x,y) == x, i.e., x|y,
    #     then x' = 1, and condition (1) is satisfied for any third z.
    #   Wait, x' = x/gcd(x,y). If x|y, gcd(x,y)=x, so x'=1. Then we need z multiple of 1,
    #     i.e., any z. Since there are 255 candidates and we exclude x and y, there are 253 others.
    #     So condition (1) holds.
    #   More generally, x' = x/gcd(x,y). We need a multiple of x' in candidates excluding x and y.
    #   Multiples of x' in 2..256: x', 2*x', 3*x', ... up to 256.
    #   We need at least one such multiple that is not x and not y.
    
    # For condition (2): symmetric to (1).
    
    # For condition (3): z | x*y. Need z in 2..256, z != x, z != y, z divides x*y.
    #   The divisors of x*y in [2,256] that are not x and not y.
    
    # Let's build the adjacency list
    # For speed, precompute for each possible x' (1..256) the set of candidate values
    # that are multiples of x'
    mult_vals = [set() for _ in range(257)]  # mult_vals[d] = set of values in 2..256 divisible by d
    for d in range(1, 257):
        for v in range(d, 257, d):
            if v >= 2:
                mult_vals[d].add(v)
    
    # For condition (3), precompute divisors in range for each product
    # divs_in_range[p] = set of divisors of p in [2, 256] for p up to 256*256
    # That's 65536 values, each with at most ~100 divisors. Feasible but let's be smart.
    
    # Actually, for condition (3), z | x*y with z in [2,256].
    # z is a divisor of x*y. The number of such z is at most the number of divisors of x*y
    # that are <= 256. For x,y <= 256, x*y <= 65536.
    
    # Let me precompute divisors for all numbers up to 65536
    # divisors[n] = list of divisors of n in [2, 256]
    max_prod = 256 * 256
    divs_in_range = [[] for _ in range(max_prod + 1)]
    for d in range(2, 257):
        for multiple in range(d, max_prod + 1, d):
            divs_in_range[multiple].append(d)
    
    # Now build conflict graph
    for i in range(m):
        xi = candidates[i]
        for j in range(i + 1, m):
            xj = candidates[j]
            g = gcd(xi, xj)
            conflict = False
            
            # Condition (1): xi | xj * zk for some zk != xi, xj
            xp = xi // g
            if xp == 1:
                # Need any zk in candidates, zk != xi, xj. Always exists (253 others)
                conflict = True
            else:
                # Need zk in mult_vals[xp] with zk != xi and zk != xj
                # mult_vals[xp] is a set of values
                # Check if there's an element in mult_vals[xp] not equal to xi or xj
                # Since |mult_vals[xp]| can be large, just check:
                # If len(mult_vals[xp]) > 2, there must be one not equal to xi,xj
                # (since at most 2 of them are xi and xj)
                if len(mult_vals[xp]) > 2:
                    conflict = True
                elif len(mult_vals[xp]) == 2:
                    # Check if both are xi and xj
                    if not (xi in mult_vals[xp] and xj in mult_vals[xp]):
                        conflict = True
                    # If both are in mult_vals[xp], and they are exactly xi and xj, no conflict from this
                elif len(mult_vals[xp]) == 1:
                    # The only multiple is some value v. If v != xi and v != xj, conflict
                    v = next(iter(mult_vals[xp]))
                    if v != xi and v != xj:
                        conflict = True
            
            # Condition (2): xj | xi * zk for some zk != xi, xj
            if not conflict:
                yp = xj // g
                if yp == 1:
                    conflict = True
                else:
                    if len(mult_vals[yp]) > 2:
                        conflict = True
                    elif len(mult_vals[yp]) == 2:
                        if not (xi in mult_vals[yp] and xj in mult_vals[yp]):
                            conflict = True
                    elif len(mult_vals[yp]) == 1:
                        v = next(iter(mult_vals[yp]))
                        if v != xi and v != xj:
                            conflict = True
            
            # Condition (3): zk | xi * xj for some zk != xi, xj
            if not conflict:
                prod = xi * xj
                divs = divs_in_range[prod]
                # Need a divisor zk in divs with zk != xi and zk != xj
                # Count how many divs are not xi and not xj
                cnt = 0
                for d in divs:
                    if d != xi and d != xj:
                        cnt += 1
                        if cnt >= 1:
                            conflict = True
                            break
            
            if conflict:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Now we have the conflict graph. We want the maximum independent set.
    # Use branch and bound with bitsets.
    
    # For maximum independent set, we can use the following approach:
    # MIS(G) = complement of minimum vertex cover, but that's NP-hard too.
    # Use recursive branch and bound.
    
    # Represent sets as Python ints (bitsets)
    all_mask = (1 << m) - 1
    
    best_size = 0
    best_set = 0  # bitset of the best independent set found
    
    def popcount(x):
        return bin(x).count('1')
    
    def branch_and_bound(candidates_mask, current_set, depth=0):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
        
        # Upper bound: current_size + number of candidates
        ub = popcount(current_set) + popcount(candidates_mask)
        if ub <= best_size:
            return
        
        if candidates_mask == 0:
            if popcount(current_set) > best_size:
                best_size = popcount(current_set)
                best_set = current_set
            return
        
        # Choose a vertex to branch on
        # Heuristic: pick vertex with highest degree in the remaining conflict subgraph
        # Or just pick the first one for simplicity
        # Let's pick the vertex with the most conflicts within candidates_mask
        
        # For efficiency, pick the lowest set bit
        lsb = candidates_mask & (-candidates_mask)
        v = lsb.bit_length() - 1
        
        # Branch 1: include v in the independent set
        # Remove v and all its neighbors from candidates
        neighbors_in_cand = adj[v] & candidates_mask
        new_candidates = candidates_mask & ~(lsb | neighbors_in_cand)
        new_current = current_set | lsb
        branch_and_bound(new_candidates, new_current, depth + 1)
        
        # Branch 2: exclude v
        new_candidates2 = candidates_mask & ~lsb
        branch_and_bound(new_candidates2, current_set, depth + 1)
    
    # Initial call
    branch_and_bound(all_mask, 0)
    
    # Convert best_set bitset to list of values
    if best_set == 0:
        # Fallback: return all primes (they should form an independent set)
        result = [v for v in candidates if _is_prime(v)]
    else:
        result = []
        for i in range(m):
            if best_set & (1 << i):
                result.append(candidates[i])
        result.sort()
    
    # Verify the result
    if not _valid_set(result):
        # Fallback to a known valid set: all primes
        result = [v for v in candidates if _is_prime(v)]
        if not _valid_set(result):
            # Even primes might have issues? Let's check
            # Actually primes should be fine since if p is prime and p|q*r with q,r != p,
            # then p|q or p|r, but q,r are distinct from p and if they're not multiples of p...
            # Wait, q and r are in the set, not necessarily prime.
            # Let's just return primes and verify
            pass
    
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
