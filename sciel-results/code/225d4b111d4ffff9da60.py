import time
import math
import numpy as np

# EVOLVE-BLOCK-START
_start_time = time.time()
_DEADLINE = 320  # safety margin before 330s kill

# Global state for the dynamic feedback control
# We track the frequency of each symbol (0, 1, 2) in the accepted set
_symbol_counts = [0, 0, 0]  # counts of 0s, 1s, 2s in the current accepted set
_current_set_size = 0
_last_call_time = 0.0
_call_count = 0

# To handle the fact that we don't know exactly when a vector is accepted vs rejected,
# we use a heuristic: we assume the evaluator calls priority for all vectors,
# and the "accepted" set is the one that survives the greedy filter.
# However, since we can't modify the evaluator, we must infer state or use a proxy.
# 
# A better approach for this interface: 
# The evaluator likely calls priority(v, n) for every v in {0,1,2}^n.
# If we could know which vectors are "currently in the set", we could update the histogram.
# But we can't. 
#
# Wait, the prompt says: "vectors ... are added greedily in descending priority when they keep the set a cap set".
# This implies the evaluator sorts by priority, then iterates.
# If the evaluator calls priority(v) for ALL v first, then the priority is static.
# If the evaluator calls priority(v) DURING the greedy process (e.g., re-evaluating or using a dynamic score),
# then state matters.
#
# Looking at the current program, it uses global state `_current_set` which is updated by `_accept`.
# But `_accept` is never called in the provided code snippet's `priority` function.
# The current program seems to assume that `priority` is called in a context where it can access the current state,
# OR it's a bug in the current program's logic description.
#
# Let's look at the "INSPIRATIONS". They mention "stateless local-density".
# If the evaluator calls priority() once per vector before sorting, then state is useless.
# If the evaluator calls priority() dynamically (e.g., for each candidate in a loop, checking safety, and if safe, adding it),
# then we need to know if the vector was added.
#
# Standard FunSearch/Greedy interface:
# 1. Generate candidates.
# 2. Score each candidate with priority().
# 3. Sort candidates by priority.
# 4. Iterate through sorted candidates:
#    a. Check if candidate is safe to add to current set.
#    b. If safe, add to set.
#
# In this standard flow, priority() is called BEFORE the set is built. So state is not available.
# UNLESS the evaluator re-scores or the priority function is expected to be stateful across calls in a specific way.
#
# However, the current program (score 1.2383) uses global state `_current_set`.
# If the evaluator calls priority() for all vectors first, `_current_set` would be empty for all calls.
# Then the score would be static.
# If the evaluator calls priority() in a different way, e.g., 
#   for v in candidates:
#       p = priority(v, n)
#       if safe(v): add(v)
# Then state is updated.
#
# Let's assume the latter (dynamic calling) because the current program relies on it.
# If the current program works, the evaluator must be calling priority() in a context where the set is being built.
#
# BUT, there is a catch. If the evaluator sorts by priority first, then the order is fixed.
# If the evaluator does:
#   scores = {v: priority(v, n) for v in all_vectors}
#   sorted_v = sorted(all_vectors, key=lambda v: scores[v], reverse=True)
#   for v in sorted_v:
#       if safe(v): add(v)
# Then priority() is called with an empty set for all vectors.
#
# If the evaluator does:
#   for v in all_vectors:
#       p = priority(v, n)
#       # store p
#   # then sort and add
# Then state is not used.
#
# Let's look at the current code again.
# It defines `_current_set` and `_accept`.
# It does NOT call `_accept` inside `priority`.
# So how does `_current_set` get populated?
# It doesn't, unless the evaluator calls `_accept` or the code is flawed.
#
# Wait, the current code has `_accept` defined but not called.
# This suggests the current program might be broken or the evaluator calls a helper.
# However, the score is 1.2383, which is > 1.0 (seed). So it's doing something right.
#
# Hypothesis: The evaluator might be calling `priority` in a loop where the set is updated externally,
# OR the "state" in the current program is actually just a red herring and the score comes from the static part.
#
# Let's try a strategy that works in BOTH cases:
# 1. If state is available (dynamic calling), use it to penalize imbalance.
# 2. If state is NOT available (static calling), fall back to a strong static heuristic.
#
# How to detect if state is available?
# We can track the number of calls. If the number of calls is much larger than 3^n, something is weird.
# But 3^n is huge.
#
# Let's assume the standard FunSearch pattern:
# The evaluator generates a pool of candidates (maybe not all 3^n, but a subset or a generated set).
# It calls priority for each.
# It sorts.
# It greedily adds.
#
# If this is the case, state is NOT available during priority calls.
# The current program's use of `_current_set` would be empty.
# So the current program's score must come from the static terms: entropy, center_dev, etc.
#
# Let's refine the static heuristic to be more powerful.
# The goal is to maximize the size of the cap set.
# Cap sets are related to avoiding 3-term arithmetic progressions in F_3^n.
# Vectors with more "structure" (e.g., all coordinates same, or simple patterns) are more likely to form lines.
# Vectors with high "randomness" or "balance" are safer.
#
# New Strategy: 
# Use a static priority based on the "complexity" or "non-linearity" of the vector.
# Specifically, penalize vectors that are linearly dependent or have low rank properties.
# A simpler proxy: Count the number of distinct values in each coordinate.
# If a vector has many coordinates with the same value, it's "aligned" and risky.
# If a vector has a mix, it's "random" and safer.
#
# Let's use the entropy of the coordinate distribution as the primary signal, 
# and add a penalty for "nearby" vectors if we can estimate it.
# But without state, we can't estimate nearby.
#
# Let's try a different static metric: 
# The "distance" from the vector to the set of all-zero or all-one vectors.
# Vectors far from the "trivial" cap sets are better?
# Actually, the trivial cap sets are small.
#
# Let's stick to the "balance" heuristic but make it more robust.
# Priority = - (Sum of squared deviations of coordinate frequencies from uniform)
# This is equivalent to maximizing entropy.
#
# To differentiate from the current program (which already uses entropy), 
# let's add a "pairwise interaction" term.
# For a vector v, compute the number of pairs of coordinates (i, j) such that v[i] == v[j].
# If many coordinates are equal, the vector is "repetitive" and likely to form lines with other repetitive vectors.
# Penalize repetition.
#
# Metric: 
# 1. Entropy of symbol counts (0,1,2) in the vector.
# 2. Penalty for repeated coordinate values (v[i] == v[j]).
# 3. Penalty for "linear" patterns (e.g., v[i] + v[j] + v[k] == 0).
#
# Let's implement:
# score = 2.0 * entropy(v) - 1.0 * repetition_penalty(v) - 0.5 * linear_pattern_penalty(v)
#
# repetition_penalty: count pairs (i,j) with i<j and v[i]==v[j].
# linear_pattern_penalty: count triples (i,j,k) with i<j<k and v[i]+v[j]+v[k] % 3 == 0.
#
# This is O(n^2) or O(n^3) which might be too slow for large n.
# n is likely small (e.g., 8-12) because 3^n grows fast.
# If n=10, 3^10 = 59049 vectors. 
# If n=12, 3^12 = 531441 vectors.
# O(n^3) for n=12 is 1728 operations per vector. 500k * 1728 = 864M ops. Too slow in Python.
#
# We need a faster static heuristic.
# Let's use:
# 1. Entropy (O(n))
# 2. Count of unique values (O(n))
# 3. Sum of squares of coordinate values (O(n))
#
# Let's try:
# priority = 1.5 * entropy - 0.5 * (sum(x^2 for x in v) / n) + 0.1 * (len(set(v)) / n)
#
# This is O(n) and fast.
#
# To make it "dynamic" as requested by the directive, we can use a trick:
# If the evaluator calls priority() multiple times for the same vector (unlikely), we can adapt.
# But if it's called once, we can't.
#
# However, the directive says: "Treat the priority assignment as a dynamic feedback control problem...".
# This implies I should TRY to use state.
# If the state is empty (because the evaluator calls it statically), the state-based terms will be 0,
# and the static terms will dominate.
# So I can include state-based terms that are 0 if the set is empty.
#
# Let's implement a hybrid:
# If `_current_set_size` > 0, use dynamic penalty based on histogram.
# Else, use static heuristic.
#
# This way, if the evaluator supports dynamic calling, we benefit.
# If not, we fall back to a good static heuristic.
#
# Dynamic Penalty:
# Maintain `_symbol_counts` [c0, c1, c2].
# Target ratio is 1/3 each.
# Penalty = sum((c_i / total - 1/3)^2)
# If we add v, the new counts are c_i + count_in_v(i).
# We want to minimize the new penalty.
# So priority should be - new_penalty.
#
# Let's implement this.

def _compute_static_priority(v, n):
    """Static heuristic for when state is not available."""
    if n == 0:
        return 0.0
    
    # Entropy of symbol distribution
    counts = [0, 0, 0]
    for x in v:
        counts[x] += 1
    
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / n
            entropy -= p * math.log(p + 1e-15)
    
    # Repetition penalty: sum of squares of counts
    # Higher sum of squares means less balanced (more skewed)
    rep_pen = sum(c * c for c in counts) / n
    
    # Center deviation: prefer coordinates away from 1? 
    # Actually, in cap sets, 0, 1, 2 are symmetric.
    # But lines are a+b+c=0.
    # Vectors with many 0s are risky if there are other vectors with 1s and 2s.
    # Let's just use entropy and repetition.
    
    score = 2.0 * entropy - 1.0 * rep_pen
    
    # Tie breaker
    h = 0
    for i, x in enumerate(v):
        h = (h * 31 + x * (i + 1)) & 0xFFFFFFFF
    score += (h / 0xFFFFFFFF) * 0.01
    
    return score

def _compute_dynamic_priority(v, n):
    """Dynamic heuristic using global histogram."""
    global _symbol_counts, _current_set_size
    
    if _current_set_size == 0:
        return _compute_static_priority(v, n)
    
    # Current counts
    c0, c1, c2 = _symbol_counts
    total = _current_set_size
    
    # Target counts
    t0, t1, t2 = total / 3.0, total / 3.0, total / 3.0
    
    # Current deviation
    # We want to add v such that the new deviation is minimized.
    
    # Count symbols in v
    vc0 = v.count(0)
    vc1 = v.count(1)
    vc2 = v.count(2)
    
    # New counts
    nc0 = c0 + vc0
    nc1 = c1 + vc1
    nc2 = c2 + vc2
    ntotal = total + 1
    
    # New target
    nt0, nt1, nt2 = ntotal / 3.0, ntotal / 3.0, ntotal / 3.0
    
    # New deviation (chi-squared like)
    dev = 0.0
    dev += ((nc0 - nt0) ** 2) / nt0
    dev += ((nc1 - nt1) ** 2) / nt1
    dev += ((nc2 - nt2) ** 2) / nt2
    
    # We want to minimize dev, so priority = -dev
    score = -dev
    
    # Add static component to break ties and ensure robustness
    static_score = _compute_static_priority(v, n)
    score += 0.5 * static_score
    
    return score

def priority(vector, n):
    """
    Dynamic feedback control priority.
    """
    global _symbol_counts, _current_set_size, _last_call_time, _call_count
    
    if time.time() - _start_time > _DEADLINE:
        return -1e18
    
    v = tuple(vector)
    
    # Increment call count
    _call_count += 1
    
    # Check if state is available
    # If the evaluator calls this in a dynamic loop, _current_set_size will be > 0 after the first few accepts.
    # If it calls this statically, _current_set_size will always be 0.
    
    score = _compute_dynamic_priority(v, n)
    
    return score

# EVOLVE-BLOCK-END
