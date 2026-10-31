import numpy as np
import time

# Global state for the greedy construction
_current_set = []
_start_time = time.time()
_DEADLINE = _start_time + 280  # Safety margin for 300s limit

def _forms_line(a, b, c):
    """Check if a, b, c form a line in F_3^n.
    A line exists if a + b + c = 0 (mod 3) for all coordinates.
    Equivalently, for each coordinate i:
    if a[i] == b[i] == c[i], sum is 3a[i] % 3 = 0.
    if {a[i], b[i], c[i]} is a permutation of {0, 1, 2}, sum is 3 % 3 = 0.
    Otherwise, sum is not 0.
    """
    # Vectorized check
    # Sum of coordinates
    s = (a + b + c) % 3
    # If any coordinate sum is not 0, it's not a line
    return np.any(s != 0) == False

def _conflict_score(vec, current_set):
    """
    Score how 'bad' it is to add vec to current_set.
    Higher score means more conflicts (worse).
    We want to minimize this.
    
    Conflict 1: Does it form a line with two existing points?
    Conflict 2: How many future points does it block?
    """
    if not current_set:
        return 0.0
    
    # Check if it forms a line with any pair in current_set
    # This is O(N^2) which is too slow.
    # Optimization: Precompute "dangerous pairs" or use a different metric.
    # For now, let's use a proxy: 
    # Count how many points in current_set are "close" to vec in a way that suggests lines.
    
    # Actually, the most direct check for "is it a cap set addition" is:
    # For every pair (p, q) in current_set, check if vec forms a line with p and q.
    # This is O(N^2) per candidate. If N is large, this is slow.
    
    # Alternative heuristic:
    # A vector v is "conflicting" if it completes a line with two existing points.
    # Let's approximate the number of lines it creates.
    
    # Efficient check:
    # For each point p in current_set, calculate the "mirror" point m = (2*p - v) % 3.
    # If m is in current_set, then v, p, m form a line.
    # We can store current_set in a set of tuples for O(1) lookup.
    
    # However, doing this for every candidate in priority() is expensive if priority is called many times.
    # The FunSearch interface calls priority(vector, n) for many vectors.
    # We need a fast heuristic.
    
    # Let's try a simpler structural heuristic that correlates with cap set properties:
    # 1. Balance (already in seed, but we can refine)
    # 2. "Distance" to existing points in terms of potential lines.
    
    # Let's use the number of "nearby" points that would form lines.
    # If we have a set S, adding v creates a line if there exist p, q in S such that v = (p+q)/2 (in F_3, this is 2p+2q? No.
    # Line: p, q, r. p+q+r=0 => r = -(p+q) = 2(p+q) mod 3.
    # So v conflicts if v = 2(p+q) mod 3 for some p, q in S.
    
    # Precomputation of all 2(p+q) for p,q in S is O(N^2).
    # If N is small, we can do it. If N is large, we need a faster proxy.
    
    # Let's assume the evaluator calls priority in a way that allows us to maintain state.
    # The contract says: "vectors of {0,1,2}^n are added greedily in descending priority".
    # This implies the evaluator sorts vectors by priority and adds them one by one.
    # It does NOT say we can update state between priority calls easily unless the evaluator
    # is designed to allow it (e.g., by calling a separate accept function or if the 
    # "greedy" process is implicit in the ordering).
    
    # Wait, the contract says: "vectors ... are added greedily in descending priority when they keep the set a cap set".
    # This usually means:
    # 1. Generate all vectors.
    # 2. Sort them by priority.
    # 3. Iterate through sorted list.
    # 4. If adding vector keeps set a cap set, add it.
    
    # If this is the case, the priority function is called ONCE per vector to determine the sort order.
    # The state of the set is NOT known during the priority calculation if the sorting happens before the adding.
    # UNLESS the priority function is called dynamically?
    # "priority(vector: tuple, n: int) -> float"
    # If it's a static sort, we cannot use the current set state in priority() because the set is built AFTER sorting.
    
    # Let's re-read carefully: "vectors of {0,1,2}^n are added greedily in descending priority when they keep the set a cap set"
    # This phrasing is ambiguous. Does it mean:
    # A) Sort all vectors by priority. Then iterate. (Static priority)
    # B) Repeatedly pick the highest priority vector that is valid, add it, remove from pool, repeat. (Dynamic priority)
    
    # FunSearch typically uses a static ordering or a simple greedy loop where the "priority" is fixed or updated.
    # If it's A, we can't use state.
    # If it's B, we can use state.
    
    # The seed program uses a static heuristic (balance + hash). This suggests A or that the state isn't accessible/usable in a way that the seed author thought was beneficial.
    # However, inspiration 3 mentions "Cap set greedy builder with geometric conflict minimization" and uses `_accept_vector`. This suggests a different interface or a more complex interaction.
    
    # But I must stick to `priority(vector, n)`.
    # If the evaluator sorts once, state is useless.
    # If the evaluator calls priority repeatedly to find the max, state is useful.
    
    # Let's look at the "FunSearch contract" hint. Usually, FunSearch generates a population, evaluates fitness (priority), selects best, adds to set, repeats.
    # If so, the priority function is called on a pool of candidates.
    # We can maintain a global set of accepted vectors.
    
    # Strategy:
    # 1. Maintain `_current_set` (list of tuples) and `_current_set_hash` (set of tuples) for fast lookup.
    # 2. In `priority()`, check if `vector` is already in `_current_set` (return -inf or low score).
    # 3. Calculate a score based on how many lines it would form with pairs in `_current_set`.
    #    Score = - (number of lines formed) + (balance score).
    # 4. We need to update `_current_set`. But `priority` is just a scoring function.
    #    How do we know when a vector is accepted?
    #    We don't, unless the evaluator calls a separate function.
    #    The contract only lists `priority`.
    
    # This is a critical constraint. If I can't update the set, I can't do stateful greedy.
    # UNLESS... the "greedy" part is handled by the evaluator calling `priority` on the remaining pool, and the evaluator itself handles the acceptance and removal.
    # But then how does `priority` know what's in the set? It doesn't.
    
    # Is it possible the evaluator calls `priority` in a loop like this:
    # while pool:
    #    best = max(pool, key=lambda v: priority(v, n))
    #    if is_cap_set(current_set + [best]):
    #        current_set.append(best)
    #    pool.remove(best)
    # If so, `priority` is called many times. But `priority` still doesn't know `current_set`.
    
    # Wait, look at Inspiration 3: `uses itertools, numpy, time; defines _init, _is_cap_set, _forms_line_with_new, _compute_conflict_score, priority, _accept_vector, priority`
    # It defines `_accept_vector`. This suggests the interface might be richer or the "priority" function is part of a larger class or the evaluator calls it differently.
    # BUT the task says: "Keep the function names and signatures of the seed program."
    # Seed program only has `priority`.
    
    # If I can only provide `priority`, and the evaluator uses it for static sorting, then stateful approaches are impossible.
    # If the evaluator uses it for dynamic selection, state is still invisible to `priority` unless I use global state that I can update... but I can't update it because I don't know when a vector is accepted.
    
    # Let's reconsider the "Static" assumption.
    # If it's static, the best I can do is a better heuristic for "likely to be in a large cap set".
    # Vectors that are "balanced" are good.
    # Vectors that are "far apart" from each other in some metric are good.
    # Since we can't see the other vectors, we can't optimize for distance to others.
    
    # Is there a different structural change?
    # Maybe the "priority" isn't just a score, but the evaluator expects the function to be part of a module that can have other functions?
    # "Keep the function names and signatures of the seed program."
    # I can add helper functions.
    
    # What if the evaluator is actually running a greedy loop *inside* the evaluation of the program, and the program is responsible for the entire construction?
    # No, "Interface: priority(vector: tuple, n: int) -> float".
    
    # Let's look at the score 1.2344. The seed is 1.0.
    # The seed uses balance + hash.
    # Inspiration 1 (1.2539) uses "distinct count + chi-squared balance + hash".
    # Inspiration 3 (1.2500) uses "geometric conflict minimization".
    
    # If Inspiration 3 works, it implies that either:
    # 1. The interface allows state (e.g., the evaluator calls priority, and if it's the best, it calls another function? No, signature is fixed).
    # 2. The "priority" function is called in a context where it can infer state? Unlikely.
    # 3. The "greedy" description is misleading and it's actually a static sort, but "geometric conflict" refers to a static property of the vector itself that correlates with being in a cap set?
    #    E.g., vectors with certain patterns are less likely to form lines with *other* vectors of similar patterns?
    
    # Let's assume the standard FunSearch static sort model:
    # 1. Generate all 3^n vectors.
    # 2. Score each with priority().
    # 3. Sort descending.
    # 4. Greedily add if valid.
    
    # In this model, `priority` must be a static function of the vector.
    # The "geometric conflict minimization" in Inspiration 3 might refer to minimizing the *potential* for conflict, which is a static property.
    # For example, vectors with many 0s might form lines with other vectors with many 0s?
    
    # Let's try a different static heuristic:
    # 1. Balance (count of 0,1,2).
    # 2. "Entropy" or "Spread" of the vector.
    # 3. Correlation with known cap set structures? (Too complex).
    
    # What if we use a random projection or hash that better correlates with cap set membership?
    # Or, what if we use the property that cap sets are independent sets in a specific hypergraph?
    # The hypergraph edges are lines.
    # We want to select a large independent set.
    # Greedy by "degree" (number of lines incident to the vertex) is a standard heuristic for independent sets.
    # The "degree" of a vector v is the number of lines containing v.
    # In F_3^n, how many lines contain a specific vector v?
    # A line is defined by two points.
    # Total lines in F_3^n is roughly 3^(n-1) * (3^n - 1) / 2?
    # Each line has 3 points.
    # Each point is on many lines.
    # Actually, the number of lines passing through a specific point is constant for all points in the affine space?
    # No, in F_3^n, the number of lines through a point is (3^n - 1) / 2.
    # This is the same for ALL vectors!
    # So "degree" is not a differentiator.
    
    # However, the *structure* of the lines differs.
    # Vectors that are "balanced" might be in "better" positions?
    
    # Let's look at the "chi-squared balance" from Inspiration 1.
    # Chi-squared statistic for the distribution of symbols.
    # Seed uses L1 deviation. Chi-squared is L2.
    # L2 might be a better measure of "balance" for this problem.
    
    # Let's try:
    # 1. Chi-squared balance score.
    # 2. A second-order statistic: e.g., sum of squared differences between coordinates?
    # 3. Or, "linear independence" of the vector with respect to a basis?
    
    # Let's try a structural change: **Use a hash-based clustering approach.**
    # Vectors that are "similar" (close in Hamming distance or L2 distance) are more likely to form lines with each other?
    # If we prioritize vectors that are "diverse" from a reference set, we might get a larger cap set.
    # But we don't have a reference set.
    
    # How about prioritizing vectors that are "extreme" in some sense?
    
    # Let's go back to the stateful idea. Is it possible the evaluator *does* allow state?
    # If the evaluator calls `priority` for each vector, and the order of calls is the order of the greedy selection?
    # i.e., the evaluator calls `priority` on the best candidate, gets a score, accepts it, then calls `priority` on the next best?
    # No, "added greedily in descending priority" implies sorting.
    
    # Let's assume the static model.
    # New Strategy: **Optimize for "Pairwise Distance" in a projected space.**
    # If we map vectors to a lower-dimensional space using a random projection, and prioritize vectors that are far from the "centroid" of the space, we might encourage diversity.
    # But we don't know the other vectors.
    
    # Let's try a simpler structural change: **Use a different balance metric.**
    # Instead of global balance, use "local balance" or "frequency of pairs".
    # Or, use the **number of distinct values** (always 3 for n>=3).
    
    # Let's look at Inspiration 1 again: "distinct count + chi-squared balance + hash".
    # Distinct count is trivial.
    # Chi-squared balance is L2.
    # Hash is tie-breaker.
    
    # What if we use **Fourier Transform** features?
    # The indicator function of a cap set has certain Fourier properties.
    # Vectors with low "Fourier energy" at non-zero frequencies might be better?
    # This is computationally expensive for large n.
    
    # Let's try a **Greedy Independent Set Heuristic** approximation:
    # Since all vertices have the same degree in the line graph, the standard greedy (pick any) is not helpful.
    # But we can use **Coloring** or **Partitioning**.
    
    # Alternative Idea: **Randomized Priority with Deterministic Seed.**
    # The seed uses a deterministic hash.
    # What if we use a hash that is specifically designed to separate vectors that are likely to form lines?
    # E.g., hash based on the sum of coordinates mod a prime?
    
    # Let's try a **Hybrid of Balance and "Line Potential"**.
    # A vector v has a "line potential" with another vector u if they are "close".
    # We can't compute this statically.
    
    # Let's try **Maximizing the minimum distance to a set of "anchor" vectors**.
    # Define a set of anchor vectors (e.g., all-zero, all-one, etc.).
    # Prioritize vectors that are far from all anchors.
    # This encourages diversity.
    
    # Implementation:
    # 1. Define a few anchor vectors (e.g., 0^n, 1^n, 2^n, and some random ones).
    # 2. For a given vector, calculate the minimum Hamming distance to any anchor.
    # 3. Score = - (min_distance) + Balance_Score.
    
    # This is a structural change: from "self-contained balance" to "diversity relative to anchors".
    
    # Let's implement this.
    
    # Anchors:
    # A0 = (0, 0, ..., 0)
    # A1 = (1, 1, ..., 1)
    # A2 = (2, 2, ..., 2)
    # A3 = (0, 1, 2, 0, 1, 2, ...)
    # A4 = (0, 0, 1, 1, 2, 2, ...)
    
    # Score:
    # balance = -chi_squared
    # min_dist = min(Hamming(v, a) for a in anchors)
    # score = balance * 100 + min_dist * 10 + hash
    
    # Let's refine the anchors to be more representative.
    # Use a small set of random vectors generated with a fixed seed.
    
    # Also, let's keep the deadline guard.

def _hamming_distance(v1, v2):
    return sum(1 for a, b in zip(v1, v2) if a != b)

# Precompute anchors
import random
random.seed(42)
# We don't know n yet, so we can't precompute anchors of length n.
# We must compute them on the fly or use a pattern.
# Pattern anchors are better.

def _get_anchors(n):
    anchors = []
    # Constant vectors
    anchors.append(tuple([0]*n))
    anchors.append(tuple([1]*n))
    anchors.append(tuple([2]*n))
    # Periodic patterns
    anchors.append(tuple([i % 3 for i in range(n)]))
    anchors.append(tuple([i % 2 for i in range(n)])) # 0,1,0,1...
    anchors.append(tuple([i % 2 + 1 for i in range(n)])) # 1,2,1,2...
    # Random vectors
    for _ in range(5):
        anchors.append(tuple(random.randint(0, 2) for _ in range(n)))
    return anchors

_anchor_cache = {}

def _get_anchors_cached(n):
    if n not in _anchor_cache:
        _anchor_cache[n] = _get_anchors(n)
    return _anchor_cache[n]

def priority(vector, n):
    global _start_time, _DEADLINE
    if time.time() > _DEADLINE:
        return 0.0
        
    # Check cache
    key = (vector, n)
    if key in _priority_cache:
        return _priority_cache[key]
    
    # Count occurrences
    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
            
    # Chi-squared balance
    ideal = n / 3.0
    chi_sq = (c0 - ideal)**2 / ideal + (c1 - ideal)**2 / ideal + (c2 - ideal)**2 / ideal
    balance_score = -chi_sq * 1000.0
    
    # Diversity score (distance to anchors)
    anchors = _get_anchors_cached(n)
    min_dist = min(_hamming_distance(vector, a) for a in anchors)
    diversity_score = min_dist * 100.0
    
    # Hash tie-breaker
    h = 0
    base = 31
    for val in vector:
        h = (h * base + val) % (10**9 + 7)
    hash_score = h / (10**9 + 7)
    
    result = float(balance_score + diversity_score + hash_score)
    _priority_cache[key] = result
    return result
