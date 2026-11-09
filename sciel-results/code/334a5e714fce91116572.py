import time
import random
from math import gcd
from itertools import combinations

# Precompute LCM table for all pairs in [1,128]
_LCM_TABLE = [[0] * 129 for _ in range(129)]
for i in range(1, 129):
    for j in range(1, 129):
        _LCM_TABLE[i][j] = i * j // gcd(i, j)

# Precompute conflict graph: for each element x, which pairs (a,b) in [1,128]
# would form a bad triple with x. Store as bitmask for fast lookup.
# conflict_mask[x] is a 128-bit integer where bit y is set if x conflicts with y
# in the sense that there exists some z such that (x,y,z) is a bad triple.
# Actually, we need: can_add(x, S) checks if adding x to S creates any bad triple.
# A bad triple (x, a, b) exists if lcm(x,a) == lcm(x,b) == lcm(a,b).
# So for fixed x and S, we check all pairs (a,b) in S.

# Better approach: precompute for each x, the set of "conflicting pairs" (a,b)
# such that (x,a,b) is a bad triple. Then adding x to S is valid iff no pair
# in S is in conflict_pairs[x].

# For efficiency with bitmasks, let's use a different approach:
# For each x, compute conflict_mask[x] as a bitmask over [1,128] where
# bit a is set if there exists b such that (x,a,b) is bad AND a < b.
# This isn't quite right either.

# Let's think differently. The constraint is: no three distinct elements a,b,c
# with lcm(a,b) == lcm(a,c) == lcm(b,c).
# 
# For a given set S, x can be added iff for all pairs (a,b) in S with a != b,
# it's NOT the case that lcm(x,a) == lcm(x,b) == lcm(a,b).
#
# Precompute: for each x, and for each pair (a,b), whether (x,a,b) is bad.
# But that's 128 * C(128,2) = ~1M entries. Too many for a table.
#
# Alternative: for each x, compute a "conflict signature" that allows fast checking.
# For fixed x, define for each a: lcm_xa = lcm(x,a). Then (x,a,b) is bad iff
# lcm(x,b) == lcm(x,a) == lcm(a,b).
# So for fixed x, group elements by their lcm with x. For each group G_v = {a : lcm(x,a) = v},
# the pair (a,b) in G_v is problematic iff lcm(a,b) = v.
#
# Precompute for each x: a dictionary mapping v -> list of a with lcm(x,a)=v.
# Then to check if x can be added to S: for each v, check if S contains two elements
# a,b from G_v with lcm(a,b)=v.

# Even better: precompute for each x, a 128-bit mask conflict_mask[x] where
# bit a is set if there EXISTS b such that (x,a,b) is bad.
# Wait, that's not sufficient. We need: for all pairs (a,b) in S, (x,a,b) is NOT bad.
#
# Let me use a different strategy. Precompute bad_triples as a set of frozensets.
# Then for a candidate x and set S, check if there exists (a,b) in S such that
# {x,a,b} is in bad_triples.

# Precompute all bad triples
_BAD_TRIPLES = set()
for a in range(1, 129):
    for b in range(a + 1, 129):
        lab = _LCM_TABLE[a][b]
        for c in range(b + 1, 129):
            if _LCM_TABLE[a][c] == lab and _LCM_TABLE[b][c] == lab:
                _BAD_TRIPLES.add((a, b, c))

# For each x, precompute which pairs (a,b) would form a bad triple with x.
# Store as a list of pairs for each x.
_BAD_PAIRS_FOR_X = [[] for _ in range(129)]
for (a, b, c) in _BAD_TRIPLES:
    # (a,b,c) is bad, so for x=a, pair (b,c) is bad; for x=b, pair (a,c); for x=c, pair (a,b)
    _BAD_PAIRS_FOR_X[a].append((b, c))
    _BAD_PAIRS_FOR_X[b].append((a, c))
    _BAD_PAIRS_FOR_X[c].append((a, b))

# For faster checking, for each x, create a set of the pairs
_BAD_PAIR_SETS = [set(pairs) for pairs in _BAD_PAIRS_FOR_X]

def _can_add_bitmask(x, S_mask):
    """Check if element x (1-indexed) can be added to set represented by S_mask.
    S_mask is a Python int where bit (y-1) is set if y is in S.
    Returns True if x can be added without creating a bad triple."""
    # For each bad pair (a,b) for x, check if both a and b are in S
    for (a, b) in _BAD_PAIRS_FOR_X[x]:
        if (S_mask >> (a - 1)) & 1 and (S_mask >> (b - 1)) & 1:
            return False
    return True

def _greedy_bitmask(order, deadline=None):
    """Greedy construction using bitmask for fast conflict checking."""
    S_mask = 0
    for x in order:
        if (S_mask >> (x - 1)) & 1:
            continue
        if _can_add_bitmask(x, S_mask):
            S_mask |= (1 << (x - 1))
        if deadline and time.time() > deadline:
            break
    return S_mask

def _local_search_bitmask(S_mask, max_iters=10000, deadline=None):
    """Local search using bitmask representation."""
    best_size = bin(S_mask).count('1')
    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False
        # Try adding any element not in S
        for x in range(1, 129):
            if (S_mask >> (x - 1)) & 1:
                continue
            if _can_add_bitmask(x, S_mask):
                S_mask |= (1 << (x - 1))
                best_size = max(best_size, bin(S_mask).count('1'))
                improved = True
                break
        if improved:
            continue
        # Try swapping: remove one, add another
        for a in range(1, 129):
            if not ((S_mask >> (a - 1)) & 1):
                continue
            S_mask_without_a = S_mask & ~(1 << (a - 1))
            for x in range(1, 129):
                if (S_mask_without_a >> (x - 1)) & 1:
                    continue
                if _can_add_bitmask(x, S_mask_without_a):
                    new_mask = S_mask_without_a | (1 << (x - 1))
                    if bin(new_mask).count('1') > best_size:
                        S_mask = new_mask
                        best_size = bin(S_mask).count('1')
                        improved = True
                    break
            if not improved:
                S_mask |= (1 << (a - 1))  # restore
        if not improved:
            break
    return S_mask

def _mask_to_list(S_mask):
    """Convert bitmask to list of elements (1-indexed)."""
    result = []
    for i in range(1, 129):
        if (S_mask >> (i - 1)) & 1:
            result.append(i)
    return result

def _list_to_mask(lst):
    """Convert list of elements to bitmask."""
    mask = 0
    for x in lst:
        mask |= (1 << (x - 1))
    return mask

def _compute_edge_weights():
    """Compute edge weights based on frequency of bad triple participation.
    For each pair (a,b), weight = number of c such that (a,b,c) is a bad triple.
    This helps prioritize which conflicts to avoid first."""
    edge_weights = {}
    for (a, b, c) in _BAD_TRIPLES:
        edge_weights[(a, b)] = edge_weights.get((a, b), 0) + 1
        edge_weights[(b, a)] = edge_weights.get((b, a), 0) + 1
    return edge_weights

def _weighted_greedy(order, edge_weights, deadline=None):
    """Weighted greedy: prioritize elements that avoid high-weight conflicts."""
    S_mask = 0
    for x in order:
        if (S_mask >> (x - 1)) & 1:
            continue
        if _can_add_bitmask(x, S_mask):
            S_mask |= (1 << (x - 1))
        if deadline and time.time() > deadline:
            break
    return S_mask

def _spectral_ordering(edge_weights):
    """Generate element ordering based on spectral properties of the conflict graph.
    Elements with lower weighted degree come first (less constrained)."""
    # Compute weighted degree for each element
    weighted_degree = [0] * 129
    for (a, b), w in edge_weights.items():
        weighted_degree[a] += w
        weighted_degree[b] += w
    
    # Sort by weighted degree (ascending) - less constrained elements first
    order = list(range(1, 129))
    order.sort(key=lambda x: weighted_degree[x])
    return order

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s limit
    
    all_elements = list(range(1, 129))
    
    # Compute edge weights
    edge_weights = _compute_edge_weights()
    
    # Generate multiple orderings
    orderings = []
    
    # 1. Spectral ordering (by weighted degree)
    orderings.append(_spectral_ordering(edge_weights))
    
    # 2. Primes first
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    primes = [x for x in all_elements if is_prime(x)]
    non_primes = [x for x in all_elements if not is_prime(x)]
    orderings.append(primes + non_primes)
    
    # 3. By number of divisors (fewer first)
    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count
    
    orderings.append(sorted(all_elements, key=lambda x: num_divisors(x)))
    
    # 4. Reverse order
    orderings.append(list(reversed(all_elements)))
    
    # 5. By max prime factor
    def _max_prime_factor(n):
        if n < 2:
            return 1
        largest = 1
        d = 2
        while d * d <= n:
            while n % d == 0:
                largest = d
                n //= d
            d += 1
        if n > 1:
            largest = n
        return largest
    
    orderings.append(sorted(all_elements, key=lambda x: _max_prime_factor(x), reverse=True))
    
    # 6. Random orderings
    random.seed(42)
    for _ in range(12):
        orderings.append(random.sample(all_elements, 128))
    
    best_mask = 0
    best_size = 0
    
    # Run greedy with each ordering
    for order in orderings:
        if time.time() > deadline:
            break
        S_mask = _greedy_bitmask(order, deadline)
        size = bin(S_mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = S_mask
    
    # Local search on best
    if time.time() < deadline:
        improved_mask = _local_search_bitmask(best_mask, max_iters=20000, deadline=deadline)
        if bin(improved_mask).count('1') > best_size:
            best_mask = improved_mask
            best_size = bin(best_mask).count('1')
    
    # Additional local search passes
    while time.time() < deadline - 5:
        prev_size = best_size
        improved_mask = _local_search_bitmask(best_mask, max_iters=5000, deadline=deadline - 5)
        new_size = bin(improved_mask).count('1')
        if new_size > prev_size:
            best_mask = improved_mask
            best_size = new_size
        else:
            break
    
    # Convert to list
    best_set = _mask_to_list(best_mask)
    
    # Final validation
    S_list = sorted(best_set)
    for a, b, c in combinations(S_list, 3):
        if (a, b, c) in _BAD_TRIPLES:
            best_set.remove(c)
    
    if not best_set:
        best_set = [1]
    
    return best_set
