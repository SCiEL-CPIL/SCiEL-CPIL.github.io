import time
import random

_START_TIME = time.time()
_DEADLINE = 330.0

# Stateful adaptive heuristic:
# We will pre-compute a static priority order based on a "conflict avoidance" strategy.
# The idea is that vectors which are "far" from each other in a structural sense are safer.
# However, since the greedy algorithm processes vectors in a fixed order determined by `priority`,
# we can't easily adapt the priority based on the *current* set state unless we make `priority`
# stateful (which is tricky if `priority` is called many times for the same vector or if the
# order needs to be stable for sorting).

# Wait, the interface is: `priority(vector: tuple, n: int) -> float`.
# The evaluator sorts ALL vectors by this priority *once* (presumably) or calls it repeatedly?
# "vectors of {0,1,2}^n are added greedily in descending priority"
# This implies a single sort of all 3^n vectors.
# Therefore, `priority` must be deterministic and independent of the current set state
# at the time of calling, because the set state is not an input.
#
# HOWEVER, looking at inspiration 1: "stateful, adaptive heuristic... relative to the current set size".
# This suggests that maybe `priority` is called *during* the greedy addition process?
# No, "added greedily in descending priority" usually means:
# 1. Generate all vectors.
# 2. Sort them by priority (desc).
# 3. Iterate through sorted list.
# 4. If vector is compatible with current cap set, add it.
#
# If `priority` is called *before* the greedy loop, it cannot know the current set.
# But what if `priority` is called *inside* the greedy loop for each candidate?
# "priority(vector, n) -> float"
# If the evaluator does:
# `for v in sorted(all_vectors, key=lambda v: priority(v, n), reverse=True):`
# Then `priority` is called 3^n times *before* any vector is added.
# In that case, a stateful heuristic based on "current set" is impossible because the set is empty.
#
# Let's re-read carefully: "vectors ... are added greedily in descending priority".
# This phrasing strongly implies a static sort key.
#
# BUT, Inspiration 1 mentions "stateful, adaptive heuristic". How is that possible?
# Maybe the evaluator calls `priority` for each vector *as it considers it*?
# No, that would be O(3^n * n^2) or worse if checking compatibility.
#
# Let's look at the current program: it's static.
# Let's look at the "improvement" in island 7: "Static conflict penalty minimization... avoiding early choices that create dense line structures."
# This implies the *order* matters. A static order that avoids "bad" early choices is better.
#
# So, the goal is to find a *static* ordering (defined by `priority`) that results in a larger cap set when greedily processed.
#
# Strategy: **Conflict-Aware Static Ordering**
# We want to order vectors such that "dangerous" vectors (those that form lines with many other vectors) are placed *later* in the order,
# OR such that "safe" vectors are placed *first*.
#
# A vector is "dangerous" if it forms a line with many other vectors.
# Every vector is part of many lines.
#
# Alternative Strategy: **Spectral/Graph-Based Ordering**
# We can view the problem as finding an independent set in a hypergraph (lines are hyperedges of size 3).
# We want to greedily select vertices.
# A good heuristic for greedy independent set is often to pick vertices with low degree *first*?
# No, if you pick low degree first, you might block high degree vertices that could have been picked later?
# Actually, in graph coloring, you pick high degree first. In independent set, it's complex.
#
# Let's try a different structural change: **Randomized Perturbation with Seeded Determinism**.
# The current program uses a hash.
# What if we use a *more complex* hash that correlates with "line density"?
#
# Let's compute a "local line density" proxy for each vector.
# For a vector v, how many other vectors u are there such that v, u, w form a line?
# This is fixed for all vectors (symmetry).
#
# However, *interactions* matter.
# Two vectors v1, v2 are "conflicting" if they form a line with some v3.
#
# Let's try **Lexicographic Order with a Twist**.
# Standard lex order might be bad.
#
# Let's try **Sum of Coordinates** as primary key?
# If we group by sum of coordinates, we might isolate "layers".
# Lines in {0,1,2}^n often connect different sum layers.
# A line a, a+d, a+2d.
# Sum(a) + Sum(d) + Sum(2d) = Sum(a) + Sum(a+d) + Sum(a+2d).
#
# If we prioritize vectors with "rare" sums, we might pick them first.
# But sums are distributed normally.
#
# Let's look at the "Conflict Penalty" idea from the island finding.
# "Static conflict penalty minimization".
# This suggests that for each vector v, we calculate a penalty based on how many *other high-priority* vectors conflict with it.
# But we don't know the other high-priority vectors until we define the order.
# This is a chicken-and-egg problem unless we use an iterative approximation or a fixed structural property.
#
# What if we define priority based on the **number of "nearby" vectors** that are *not* lines?
#
# Let's try a **Greedy Pre-computation** approach inside the module (not inside `priority` itself, but using the time available).
# Since `priority` is called for sorting, we can do heavy lifting *before* the first call?
# No, `priority` is a function. It might be called 3^n times.
# We can cache the result.
#
# We can compute a **global conflict graph** approximation.
# For each vector, count how many *random* vectors form a line with it? No, all do.
#
# Let's try **Priority = - (Number of vectors that form a line with this vector and have a higher "natural" priority)**.
# This is circular.
#
# Let's try a **Modular Hash** that spreads out vectors with similar sums.
#
# **New Idea: "Safe" Vectors First**
# Vectors that are "isolated" in the line structure.
# In {0,1,2}^n, every vector is in (3^n-1)/2 lines? No.
# Number of lines through a point: (3^n - 1) / 2.
#
# What if we prioritize vectors that are **linearly dependent** in a specific way?
#
# Let's try **Randomized Greedy Simulation** inside the module load?
# No, `priority` must be fast.
#
# Let's assume the evaluator sorts once.
# We can pre-compute a "score" for each vector in a global dict?
# 3^n is huge for n=20 (3^20 ~ 3.4 billion). We can't pre-compute for all.
# n is likely small (e.g., 10-15). 3^15 = 14 million. 3^10 = 59049.
# If n=10, we can pre-compute!
# If n=15, we can maybe pre-compute in a few seconds?
#
# Let's check the deadline. 330s.
# If n=15, 3^15 = 14,348,907.
# Computing a hash for each is fast.
# Computing a "conflict score" might be expensive.
#
# Let's assume n is small enough that we can afford some pre-computation in `priority` if we cache it.
# But `priority` is called for each vector.
#
# **Strategy: Adaptive Thresholding**
#
# Let's look at the inspiration again: "stateful, adaptive heuristic".
# If `priority` is called *during* the greedy process, we can access a global `current_set`.
# But the problem says "vectors ... are added greedily in descending priority".
# This usually means:
# `ordered = sorted(vectors, key=priority)`
# `for v in ordered: ...`
#
# If the evaluator uses `sorted`, `priority` is called for all vectors *before* the loop.
# So `current_set` is empty.
#
# UNLESS the evaluator does:
# `while True: pick best remaining...`
# That would be O(N^2) or worse, and unlikely for N=3^10+.
#
# So it must be a static sort.
#
# **Structural Change: Use a "Line-Breaking" Metric**
# A line is a, a+d, a+2d.
# If we prioritize vectors such that for any high-priority pair, their "midpoint" is low-priority, we might avoid forming lines.
#
# Let's define a score for a vector v based on the "cost" of its neighbors.
# Cost(v) = sum( priority(u) for u such that v, u, w is a line )?
# Again, circular.
#
# Let's try **Randomized Coordinate Permutation**.
# The symmetry of the cube suggests that any coordinate-wise function is symmetric.
# Breaking symmetry via a random permutation of coordinates might help?
# No, the set is symmetric under permutation.
#
# **Let's try a "Sparse" Priority.**
# Prioritize vectors with many 0s or many 2s, but *not* both.
# i.e., prefer "pure" vectors?
# (0,0,...,0) and (2,2,...,2) are far apart.
# (1,1,...,1) is in the middle.
#
# Current program: Sum of squares. Favors 2s.
# New program: **Favor Imbalance**.
# Maximize variance of coordinates.
# Variance = E[x^2] - (E[x])^2.
# High variance means coordinates are spread out (0s and 2s).
# Low variance means coordinates are close (1s).
#
# Hypothesis: High variance vectors are "spikier" and might form fewer lines with other high-variance vectors than low-variance vectors do with each other?
#
# Let's implement:
# 1. Calculate mean and variance of coordinates.
# 2. Priority = Variance + Hash.
#
# This is structurally different from Sum of Squares (which is related to variance but doesn't subtract the mean square).
# Sum of Squares favors 2s heavily. Variance favors 0s and 2s equally.
#
# Let's try this.

import time

_START_TIME = time.time()
_DEADLINE = 330.0
_CACHE = {}
_CACHE_LIMIT = 10_000_000

def _hash_vector(vec, n):
    # Fast deterministic hash
    # Use Python's built-in hash? No, not stable across runs.
    # Polynomial hash
    h = 0
    base = 31
    for v in vec:
        h = (h * base + v) % (10**9 + 7)
    return h / (10**9 + 7)

def priority(vector, n):
    if time.time() > _DEADLINE - 5.0:
        # Fallback: just sum of squares
        return -sum(v * v for v in vector)
        
    # Check cache
    # Tuples are hashable
    if vector in _CACHE:
        return _CACHE[vector]
        
    # Calculate Variance
    # mean = sum(v)/n
    # var = sum((v-mean)^2)/n
    # We want to MAXIMIZE variance.
    
    sum_v = 0
    sum_v_sq = 0
    for v in vector:
        sum_v += v
        sum_v_sq += v * v
        
    mean = sum_v / n
    var = (sum_v_sq / n) - (mean * mean)
    
    h = _hash_vector(vector, n)
    
    # Combine
    # Variance ranges from 0 to 1 (for {0,1,2}, max var is 1: e.g., 0 and 2)
    # Sum of squares ranges 0 to 4n.
    # We want variance to be the primary key.
    
    result = var + h * 0.001
    
    # Cache if not too many
    if len(_CACHE) < _CACHE_LIMIT:
        _CACHE[vector] = result
        
    return result
