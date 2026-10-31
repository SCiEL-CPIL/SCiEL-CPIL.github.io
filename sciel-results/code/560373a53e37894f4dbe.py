import time
import math
import zlib

_start_time = time.time()
_DEADLINE = _start_time + 280.0

# Global state for the greedy construction
# The evaluator calls priority() for vectors in descending order of priority.
# Since we don't know the order in advance, we cannot pre-sort.
# However, we can maintain a state that updates as vectors are "accepted".
# But wait: the interface is `priority(vector, n) -> float`.
# The evaluator sorts ALL vectors by this priority? 
# "vectors of {0,1,2}^n are added greedily in descending priority"
# This implies the evaluator might:
# 1. Generate all 3^n vectors.
# 2. Call priority(v, n) for each.
# 3. Sort them by the returned priority.
# 4. Iterate through the sorted list, adding v to the set if it doesn't create an AP.
#
# If this is the case, my priority function is called 3^n times BEFORE any vector is added.
# Therefore, I CANNOT use the state of the current set inside `priority` because the set is empty (or undefined) during the sorting phase.
# The "stateful" inspiration likely refers to a different interface or a misunderstanding of the static sorting nature.
#
# Let's re-read the contract: "vectors ... are added greedily in descending priority".
# This usually means:
# 1. Compute priority for all v.
# 2. Sort.
# 3. Greedy add.
#
# If the priority is static (doesn't depend on the set), then we are back to finding a good static ordering.
# The current program uses a static heuristic (FFT + Centrality).
#
# Is there a way to make the priority "stateful" in a static context?
# No, unless the evaluator calls priority *during* the greedy addition, passing the current set?
# The signature is `priority(vector, n)`. It does NOT pass the current set.
# So the priority MUST be static for a given vector v and dimension n.
#
# Therefore, the "stateful" inspiration might be a red herring or applies to a different variant where the API allows querying the set.
# OR, it implies that we should simulate the greedy process to estimate which vectors are "safe" early on.
#
# Let's assume the standard FunSearch contract:
# 1. The evaluator has a list of all 3^n vectors.
# 2. It asks for the priority of each.
# 3. It sorts them.
# 4. It greedily adds them.
#
# Since I cannot know the set, I must predict which vectors are "good" for a cap set.
# A "good" vector is one that is unlikely to form an AP with other "good" vectors.
#
# Let's rethink the "Structural Change".
# The current program uses FFT (frequency domain) and Centrality (count balance).
# A different design family: **Combinatorial Conflict Estimation**.
#
# Instead of analyzing the vector's internal signal properties, we analyze its **combinatorial potential** to form APs with *typical* vectors.
#
# Key Insight:
# In a greedy cap set construction, the "hard" vectors are those that are "central" in the Hamming cube or have high symmetry, because they are more likely to collide with many other vectors.
# Actually, vectors with low weight (sparse) are often safer?
# Or vectors with high weight?
#
# Let's look at the "2-weight" finding from the knowledge:
# "retaining the current program's symmetry-breaking 2-weight and variance danger terms"
#
# Let's try a different metric: **Local Density of Conflicts**.
# For a vector v, how many vectors u are there such that v and u are part of an AP?
# An AP is (a, b, c) where a+c=2b.
# If we pick v, it can be the middle term b, or an end term a or c.
# If v is the middle term: we need a and c such that a+c=2v.
# If v is an end term: say a=v, then c=2b-v.
#
# The number of potential APs involving v is roughly proportional to the number of "directions" d such that v+d and v-d are in the space.
# Since the space is the entire {0,1,2}^n, the number of potential partners is always 3^n - 1.
# This doesn't help distinguish vectors.
#
# However, the *greedy* process rejects vectors.
# Vectors that are "popular" (part of many potential APs with other popular vectors) should be penalized.
#
# Let's try a **Randomized Consensus** approach? No, must be deterministic.
#
# Let's try **Hamming Weight Distribution**.
# Cap sets in Z3^n are known to have specific weight distributions.
# The "Ellenberg-Gijswijt" bound suggests cap sets are sparse.
#
# Let's try a completely different heuristic: **Algebraic Independence Proxy**.
# Map the vector v to a polynomial P_v(x) = sum v_i x_i.
# Vectors that are "algebraically independent" in some sense are better.
#
# Actually, let's look at the **2-weight** mention.
# Maybe vectors with exactly 2 non-zero coordinates (or close to it) are good?
#
# Let's try a **Dynamic Programming / Lookup Table** approach for small n?
# n is likely small (<= 10 or 12) for 300s.
# If n is small, we can precompute the "quality" of each vector by simulating a small greedy process?
# No, 3^n is too big to simulate for each vector.
#
# Let's try **Coordinate-wise Uniqueness**.
# Vectors that have distinct coordinate values are "more random" and less likely to align?
#
# Let's go with a **Collision Probability Model**.
# Assume the greedy set S is a random subset of size M (where M is the expected cap set size).
# The probability that v conflicts with S is proportional to the number of APs containing v.
# But all v have the same number of APs in the full space.
#
# The difference comes from the *structure* of the APs.
# APs involving "regular" vectors (like 000...0) are more likely to be formed by other "regular" vectors which might be picked early.
#
# Let's try **Penalizing Symmetry**.
# A vector is symmetric if it is invariant under some non-trivial permutation of coordinates.
# Highly symmetric vectors (like 000, 111, 0101) are "bad" because they form APs with other symmetric vectors.
# Random vectors have trivial symmetry.
#
# Strategy: **Symmetry Breaking Score**.
# Calculate the number of non-trivial automorphisms of the vector's coordinate multiset?
# Or simpler: Count the number of distinct coordinates.
# If all coordinates are distinct, it's "maximally random".
# If many are repeated, it's "structured".
#
# Let's combine:
# 1. **Distinctness**: Reward vectors with many distinct coordinate values.
# 2. **Weight Balance**: Reward vectors with balanced counts of 0, 1, 2 (as before, but maybe weighted differently).
# 3. **No-AP Self-Check**: Does the vector itself contain a 3-term AP in its coordinates? (Indices 0..n-1 are not in Z3, so this doesn't apply directly to the vector's own coordinates as a set in Z3^n. Wait, the vector IS a point in Z3^n. The AP is between points.)
#
# Let's try **Distance to Origin**.
# Vectors far from the origin (high Hamming weight) might be safer?
#
# Let's implement a **Hybrid Heuristic**:
# 1. **Hamming Weight** (count of non-zeros).
# 2. **Value Variance** (how spread out the values 0,1,2 are).
# 3. **Pattern Regularity** (penalize periodic patterns).
#
# New Idea: **Fourier Magnitude of the Indicator of the Vector's Support**.
# No, let's stick to the "Different Design Family" directive.
#
# Design Family: **Combinatorial Game Theory / Local Search Proxy**.
# We approximate the "danger" of a vector v by counting how many "simple" APs it participates in.
# Simple APs are those where the difference vector d has small Hamming weight.
# If v is part of many APs with small-weight differences, it is "dangerous".
#
# For a fixed v, the number of APs with difference d is 1 (if v+d and v-d are in space).
# So the total number of APs is fixed.
# BUT, if the greedy algorithm picks low-weight vectors first, then high-weight vectors later, the "danger" of a high-weight vector depends on how many low-weight vectors it conflicts with.
#
# Let's assume the greedy algorithm picks **Low Complexity** vectors first.
# Complexity = Hamming Weight.
# So, vectors with LOW Hamming Weight are picked first.
# Therefore, we should prioritize Low Hamming Weight vectors?
# If we pick 000...0 first, then 000...1, etc.
# Does this lead to a large cap set?
# The set of all vectors with weight 0 and 1 is a cap set?
# 000, 001, 010...
# 000, 001, 002 is an AP. So weight 1 vectors conflict with 000.
# So if we pick 000, we cannot pick 001 and 002.
#
# So, **Penalize Low Weight**?
# If we avoid 000, we can pick more weight 1 vectors?
#
# Let's try **Prioritizing High Weight** vectors.
# Hypothesis: High weight vectors are "more random" and less likely to form APs with each other?
#
# Let's try a **Weight-Biased Randomness** score.
# Score = Hamming Weight + Random Tie-Breaker.
#
# Let's test this hypothesis mentally.
# If we sort by Hamming Weight descending:
# We pick 222...2, 222...1, etc.
# 222, 221, 220? No, 222 + 220 = 444 = 111. Not an AP.
# 222, 221, 220 is not an AP in Z3. 222+220=444=111. Midpoint would be (222+220)/2 = 111.
# So 222, 111, 220 is an AP.
# If we pick 222 and 220, we must reject 111.
#
# This is complex.
#
# Let's go with a **Simple, Robust Heuristic**:
# **Maximize Entropy of Coordinates**.
# Vectors with high entropy (balanced 0,1,2) are "central".
# But we know central vectors are dangerous.
# So **Minimize Entropy**?
# Vectors with low entropy (e.g., mostly 0s) are "sparse".
#
# Let's try **Sparse Vectors**.
# Score = -Hamming Weight.
#
# Let's try **Dense Vectors**.
# Score = Hamming Weight.
#
# Let's look at the current score 1.2930.
# The current program uses `-HFE + Centrality`.
# Centrality rewards balanced counts.
# So it rewards "Central" vectors.
#
# Let's try the **Opposite**: Reward **Unbalanced** vectors.
# Score = Variance of Counts.
#
# Let's implement a program that rewards **High Variance** (Unbalanced) and **High Hamming Weight**.
# This is a structural change: from "Smooth/Central" to "Rough/Edge".

import numpy as np

def priority(vector, n):
    if time.time() > _DEADLINE:
        return 0.0

    v = np.array(vector, dtype=np.int64)
    
    # Hamming Weight: count of non-zeros
    hw = np.sum(v != 0)
    
    # Counts of 0, 1, 2
    c0 = np.sum(v == 0)
    c1 = np.sum(v == 1)
    c2 = np.sum(v == 2)
    
    # Variance of counts (how unbalanced)
    ideal = n / 3.0
    var = (c0 - ideal)**2 + (c1 - ideal)**2 + (c2 - ideal)**2
    max_var = 2 * n * n / 3.0
    if max_var > 0:
        norm_var = var / max_var
    else:
        norm_var = 0.0
    
    # Score: Reward Unbalanced (High Var) and High Weight
    # We want to pick vectors that are "far" from the center of the cube.
    score = norm_var + (hw / n)
    
    # Tie-breaker
    h = 0
    for val in vector:
        h = (h * 37 + val) & 0xFFFFFFFF
    score += (h / 4294967296.0) * 0.1
    
    return float(score)
