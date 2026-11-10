# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from collections import defaultdict


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that
    the minimum squared pairwise distance is >= the maximum squared norm.
    Aims to maximize the number of points k.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600s limit, keep margin
    
    d = 11
    
    # We want to find a set of integer vectors v_1, ..., v_k in Z^11 such that:
    # 1. ||v_i||^2 <= min_{i != j} ||v_i - v_j||^2
    # 2. v_i != 0 for all i
    # 3. v_i != v_j for all i != j (implied by condition 1 if norms are positive)
    
    # Strategy: 
    # The condition ||v_i||^2 <= ||v_i - v_j||^2 can be rewritten.
    # ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2 v_i . v_j
    # So we need: ||v_i||^2 <= ||v_i||^2 + ||v_j||^2 - 2 v_i . v_j
    # => 0 <= ||v_j||^2 - 2 v_i . v_j
    # => 2 v_i . v_j <= ||v_j||^2
    
    # This is a symmetric condition if we also require 2 v_j . v_i <= ||v_i||^2.
    # So for any pair (i, j), we must have:
    # 2 v_i . v_j <= min(||v_i||^2, ||v_j||^2)
    
    # Let's try to build a set of vectors with small norms first.
    # Vectors with norm^2 = 1: e_i, -e_i. There are 22 such vectors.
    # Let's check if we can include all of them.
    # For v_i = e_a and v_j = e_b (a != b):
    # v_i . v_j = 0. ||v_j||^2 = 1. 2*0 <= 1. OK.
    # For v_i = e_a and v_j = -e_a:
    # v_i . v_j = -1. ||v_j||^2 = 1. 2*(-1) = -2 <= 1. OK.
    # For v_i = e_a and v_j = -e_b (a != b):
    # v_i . v_j = 0. OK.
    # So the 22 unit vectors (and their negatives) are mutually compatible.
    
    # Can we add vectors with norm^2 = 2?
    # Example: v = e_1 + e_2. ||v||^2 = 2.
    # Check against e_1: v . e_1 = 1. ||e_1||^2 = 1. 2*1 = 2 <= 1? NO.
    # So we cannot have both e_1 and e_1+e_2.
    # This suggests we should pick a "code" of vectors.
    
    # Let's use a greedy approach to add vectors.
    # We will consider candidate vectors in order of increasing norm^2.
    # For each candidate, we check if it is compatible with all existing vectors.
    
    # Candidate generation:
    # We'll generate integer vectors with coordinates in {-2, -1, 0, 1, 2} initially,
    # and expand the range if we have time.
    
    # To be efficient, we can precompute dot products or use a data structure.
    # But for now, a simple loop might suffice if the set isn't too large.
    # The kissing number in 11D is known to be 558 (or close to it for integer lattices).
    # Actually, the integer kissing number is much smaller.
    # Let's see how many we can get.
    
    # Start with the 22 unit vectors.
    points = []
    
    # Add e_i and -e_i
    for i in range(d):
        v = np.zeros(d, dtype=np.int64)
        v[i] = 1
        points.append(v)
        v[i] = -1
        points.append(v)
    
    # Now try to add other vectors.
    # We'll iterate over possible vectors with small norms.
    # Let's define a range for coordinates.
    # Start with range 2, i.e., coords in {-2, -1, 0, 1, 2}.
    
    # To avoid duplicates and zero vector, we can iterate over all combinations.
    # But 5^11 is 48,828,125 which is too many to check all against all existing points
    # if the set grows large.
    
    # Better approach:
    # Only consider vectors that are "compatible" with the current set.
    # We can prune candidates early.
    
    # Let's try a different approach:
    # Use the fact that the condition is 2 v_i . v_j <= min(||v_i||^2, ||v_j||^2).
    # If we fix a set of vectors, we can check compatibility.
    
    # Let's try to add vectors of norm^2 = 2, 3, 4, ... in order.
    # For each norm^2, generate all integer vectors with that norm.
    # Check each against the current set.
    
    # Function to check if a new vector v is compatible with all vectors in points
    def is_compatible(v, points):
        v_norm_sq = np.dot(v, v)
        for p in points:
            p_norm_sq = np.dot(p, p)
            dot = np.dot(v, p)
            # Condition: 2 * dot <= min(v_norm_sq, p_norm_sq)
            if 2 * dot > min(v_norm_sq, p_norm_sq):
                return False
        return True
    
    # Function to generate all integer vectors with a given norm^2
    def generate_vectors_with_norm_sq(norm_sq, d, max_coord=3):
        # This is a combinatorial explosion. We need a smarter way.
        # Let's just iterate over a limited range of coordinates.
        # For now, let's just try a few specific patterns.
        pass
    
    # Let's try a simpler heuristic:
    # Add vectors of the form e_i + e_j, e_i - e_j, etc., but we saw e_i + e_j conflicts with e_i.
    # What about vectors with disjoint supports?
    # If v_i and v_j have disjoint supports, then v_i . v_j = 0, so the condition is always satisfied.
    # So we can pack as many disjoint-support vectors as we want, as long as they don't conflict with each other.
    # But disjoint-support vectors with norm^2 = 1 are just the e_i and -e_i.
    # If we use norm^2 = 2, e.g., e_1 + e_2, it has support {1, 2}. It conflicts with e_1 and e_2.
    # So we can't have both.
    
    # This is a packing problem.
    # Let's try to use a greedy algorithm that picks vectors in a smart order.
    
    # Let's try to add vectors with norm^2 = 1, 2, 3, ... 
    # But we need to generate them.
    
    # Alternative: Use a known construction.
    # The Leech lattice is in 24D. In 11D, maybe we can use a sub-lattice.
    # But we need integer coordinates.
    
    # Let's try to be more systematic.
    # We'll generate candidate vectors by increasing norm^2.
    # For each norm^2, we'll generate all integer vectors with that norm.
    # We'll check them against the current set.
    
    # To generate vectors with a given norm^2, we can use recursion or backtracking.
    # But for now, let's just try a limited set of candidates.
    
    # Let's try to add vectors of the form:
    # - e_i (already added)
    # - e_i + e_j (conflicts with e_i, e_j)
    # - e_i + 2e_j (conflicts with e_i, e_j)
    # - 2e_i (conflicts with e_i? 2*(2e_i).e_i = 4, min(4,1)=1. 4>1. Yes, conflicts.)
    
    # It seems hard to add anything to the 22 unit vectors.
    # Let's verify: if we have e_i, can we add any other vector v?
    # We need 2 v . e_i <= min(||v||^2, 1) = 1.
    # So v . e_i <= 0.5, which means v_i <= 0 (since v_i is integer).
    # Similarly, for -e_i, we need 2 v . (-e_i) <= 1 => -2 v_i <= 1 => v_i >= -0.5 => v_i >= 0.
    # So for all i, v_i <= 0 and v_i >= 0, which means v_i = 0 for all i.
    # So v = 0, which is not allowed.
    
    # Therefore, we cannot add any vector to the set of 22 unit vectors.
    # So the maximum k with this simple set is 22.
    
    # But we can do better by not including all 22 unit vectors.
    # For example, if we drop e_1, we might be able to add e_1 + e_2.
    
    # This is a maximum independent set problem in a graph where edges represent conflicts.
    # The graph is large, so we need a heuristic.
    
    # Let's try a greedy approach:
    # 1. Generate a large set of candidate vectors.
    # 2. Order them by norm^2 (or some other criterion).
    # 3. For each candidate, add it if it's compatible with all currently added vectors.
    
    # Candidate generation:
    # Let's generate vectors with coordinates in {-2, -1, 0, 1, 2} and norm^2 <= 10.
    
    candidates = []
    for v in itertools.product([-2, -1, 0, 1, 2], repeat=d):
        v = np.array(v, dtype=np.int64)
        if np.all(v == 0):
            continue
        norm_sq = np.dot(v, v)
        if norm_sq <= 10:
            candidates.append(v)
    
    # Sort by norm^2
    candidates.sort(key=lambda v: np.dot(v, v))
    
    # Greedy selection
    selected = []
    for v in candidates:
        if is_compatible(v, selected):
            selected.append(v)
    
    # Convert to numpy array
    if selected:
        points = np.array(selected, dtype=np.int64)
    else:
        points = np.array([], dtype=np.int64).reshape(0, d)
    
    return points


# EVOLVE-BLOCK-END
