"""Cap set greedy builder with global combinatorial line-completion penalty."""
import time
import numpy as np

# EVOLVE-BLOCK-START
import math
_start_time = time.time()
_DEADLINE = 310  # seconds, safety margin

# Global state
_accepted_vectors = set()
_accepted_list = []
_n = None
_size = 3
# pair_sum_freq[s] = number of pairs (a, b) in accepted set such that (a+b) mod 3 = s
# We need to check if v forms a line with a, b: a + b + v = 0 mod 3 => a + b = -v mod 3
# So we need count of pairs summing to (-v) mod 3.
# However, the cap set condition is that a, b, c are distinct and a+b+c=0.
# If v is added, it forms a line with a, b if a+b = -v.
# We must ensure a != b and a != v and b != v.
# Since v is not in the set yet, a != v and b != v are guaranteed if a,b are in set.
# We must ensure a != b.
# The pair sum frequency should count ordered pairs (a, b) with a != b?
# Or just count pairs {a, b}.
# Let's maintain count of unordered pairs {a, b} summing to s.
# When adding a new vector x to the set:
# New pairs involving x: {x, a} for all a in set. Sum = x + a.
# So for each a in set, increment pair_sum_freq[(x+a) mod 3].
# This is O(|S|) per insertion, but O(1) per query.
# Total O(N^2) for building the whole set, which is fast.

_pair_sum_freq = np.zeros(3**30, dtype=np.int64) # Max n=30? No, n is small usually. Let's assume n <= 20 for safety, 3^20 is ~3.5e9, too big.
# Wait, n is the dimension. The task says vectors of {0,1,2}^n.
# Typical cap set problems are n=8, 9, 10, 11, 12, 13.
# 3^13 = 1.59e6. This is manageable.
# If n is larger, we can't allocate 3^n.
# Let's check n at runtime. If 3^n is too large, fall back to O(|S|) counting.

_pair_sum_freq = None
_n_current = None

def _init_state(n):
    global _pair_sum_freq, _n_current
    _n_current = n
    if 3**n <= 20_000_000: # ~80MB for int64, might be tight but ok. Let's be safer: 5M
        if 3**n <= 5_000_000:
            _pair_sum_freq = np.zeros(3**n, dtype=np.int64)
        else:
            _pair_sum_freq = None
    else:
        _pair_sum_freq = None

def _vec_to_idx(v):
    """Convert vector of {0,1,2} to base-3 integer index."""
    idx = 0
    for x in reversed(v):
        idx = idx * 3 + x
    return idx

def _idx_to_vec(idx, n):
    """Convert base-3 integer index to vector of {0,1,2}."""
    v = [0] * n
    for i in range(n - 1, -1, -1):
        v[i] = idx % 3
        idx //= 3
    return tuple(v)

def _add_to_pair_freq(v):
    """Update pair sum frequencies when v is added to the accepted set."""
    if _pair_sum_freq is None:
        return
    idx_v = _vec_to_idx(v)
    # For every existing vector a in the set, add 1 to freq[(v + a) mod 3]
    # This is O(|S|).
    # To speed this up, we could iterate over the whole space? No, |S| is small.
    for a in _accepted_list:
        # Calculate (v + a) mod 3 coordinate-wise
        # Since v and a are tuples, we can do this efficiently
        # But we need the index of (v+a) mod 3
        # (v+a) mod 3 is not simply idx_v + idx_a due to carries.
        # We must compute the vector sum mod 3 then convert to idx.
        # This is O(n) per pair. Total O(|S| * n).
        # For n=13, |S|~200, this is 2600 ops. Fast.
        
        # Optimization: Precompute sums?
        # Let's just do it directly.
        sum_vec = tuple((v[i] + a[i]) % 3 for i in range(_n_current))
        idx_sum = _vec_to_idx(sum_vec)
        _pair_sum_freq[idx_sum] += 1

def priority(vector, n):
    """Priority for greedy cap set construction.
    
    Uses a global combinatorial penalty: counts how many pairs (a, b) in the
    current set would form a line with the candidate vector v.
    This is computed in O(1) using a precomputed pair-sum frequency table.
    """
    global _n_current, _pair_sum_freq
    
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        return -1e18
    
    v = tuple(vector)
    
    # Initialize state if n changes
    if _n_current != n:
        _init_state(n)
        _accepted_vectors.clear()
        _accepted_list.clear()
        if _pair_sum_freq is not None:
            _pair_sum_freq[:] = 0

    # Calculate the "line completion" penalty
    # We want to know how many pairs (a, b) in the set satisfy a + b = -v mod 3.
    # This is exactly _pair_sum_freq[(-v) mod 3 index].
    
    penalty = 0
    if _pair_sum_freq is not None and _accepted_list:
        # Calculate index of (-v) mod 3
        neg_v = tuple((-x) % 3 for x in v)
        idx_neg_v = _vec_to_idx(neg_v)
        # This count includes pairs (a, a) if we didn't exclude them?
        # Our _add_to_pair_freq iterates over a in _accepted_list.
        # When we added 'a', we iterated over all 'b' in _accepted_list (which didn't include 'a' yet? No, 'a' was just added).
        # Wait, the logic in _add_to_pair_freq is:
        # When adding v, for each a in _accepted_list (old set), increment freq[v+a].
        # This counts each unordered pair {a, b} exactly once when the second element is added.
        # So freq[s] is the number of unordered pairs {a, b} with a+b=s.
        # Does this include a=b? No, because when adding v, we only look at a in the *previous* set.
        # So a != v is guaranteed. And since we only add pairs involving the new element, we never count (a, a).
        # So freq[s] is the number of distinct pairs {a, b} with a != b and a+b=s.
        
        # However, there is a subtlety:
        # If a+b = -v, then a, b, v form a line.
        # We must ensure that v is not equal to a or b.
        # Since v is a candidate and not in the set, v != a and v != b is true.
        # So the penalty is simply the count of such pairs.
        
        penalty = int(_pair_sum_freq[idx_neg_v])
    else:
        # Fallback for large n or if table not initialized
        # O(|S|) count
        if _accepted_list:
            count = 0
            for a in _accepted_list:
                for b in _accepted_list:
                    if a == b: continue
                    # Check if a + b + v = 0 mod 3
                    # This is O(n) per pair. O(|S|^2 * n). Too slow for large S.
                    # But we only use this if table is None, which implies n is large.
                    # If n is large, |S| is likely small? Not necessarily.
                    # Let's use a faster check:
                    # a + b = -v
                    # We can precompute -v
                    neg_v = tuple((-x) % 3 for x in v)
                    # Check if a + b == neg_v
                    # We can do this by checking if b == neg_v - a
                    # But we don't have a hash map for vectors to vectors.
                    # We have _accepted_vectors as a set of tuples.
                    # So for each a, check if (neg_v - a) mod 3 is in _accepted_vectors.
                    # This is O(|S| * n).
                    for a in _accepted_list:
                        # Calculate c = (-v - a) mod 3
                        c = tuple((neg_v[i] - a[i]) % 3 for i in range(n))
                        if c in _accepted_vectors and c != a:
                            count += 1
            # Each pair is counted twice (a,b) and (b,a)?
            # No, we iterate a in list, and check if c is in set.
            # If c is in set, it's a valid partner.
            # If a != c, we count it.
            # This counts each unordered pair twice?
            # If a and b form a line, then for a, c=b is found. For b, c=a is found.
            # So count is 2 * (number of pairs).
            penalty = count // 2
        else:
            penalty = 0

    # Heuristic: 
    # We want to MINIMIZE penalty.
    # So priority should be INVERSE of penalty, or negative.
    # But we also want to break ties.
    # Let's use a score that favors low penalty.
    # Since penalty can be 0, 1, 2, ..., we can use -penalty.
    # To break ties, use a hash or geometric property.
    
    # Geometric tie-breaker: Hamming weight balance
    hw = sum(1 for x in v if x != 0)
    hw_score = -abs(hw - n/3.0)
    
    # Deterministic tie-breaker
    h = 0
    for i, x in enumerate(v):
        h = (h * 31 + x * (i + 1)) & 0xFFFFFFFF
    tie_break = (h / 0xFFFFFFFF) * 0.001
    
    # Return negative penalty as primary, tie-breaker as secondary
    # To make higher priority better, we return -penalty + tie_break
    # But if penalty is 0 for many, tie_break matters.
    # If penalty is high, it dominates.
    
    return -penalty + hw_score * 0.1 + tie_break

def _accept_vector(v):
    """Record that vector v has been accepted into the cap set."""
    _accepted_vectors.add(v)
    _accepted_list.append(v)
    _add_to_pair_freq(v)

# EVOLVE-BLOCK-END
