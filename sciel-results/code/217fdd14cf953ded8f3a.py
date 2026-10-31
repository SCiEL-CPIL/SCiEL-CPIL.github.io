import numpy as np
import zlib
import time

# Global state
_sum_counts = None
_selected = None
_selected_indices = []
_n_global = None
_base_size = None
_start_time = time.time()
_DEADLINE = 300.0

def _init_state(n):
    global _sum_counts, _selected, _selected_indices, _n_global, _base_size
    _n_global = n
    _base_size = 3 ** n
    # Only use the exact method if the space is small enough
    # 3^12 = 531441, 3^13 = 1594323.
    # If n > 13, we fall back to a heuristic.
    if _base_size <= 2000000:
        _sum_counts = np.zeros(_base_size, dtype=np.int32)
        _selected = np.zeros(_base_size, dtype=np.bool_)
    else:
        _sum_counts = None
        _selected = None

def _encode(vector, n):
    idx = 0
    for i in range(n):
        idx = idx * 3 + vector[i]
    return idx

def _decode(idx, n):
    vec = [0] * n
    for i in range(n):
        vec[i] = idx % 3
        idx //= 3
    # Reverse because we encoded from left to right?
    # My encoding: idx = idx*3 + v[0]. Then idx = idx*3 + v[1].
    # So v[0] is the most significant digit.
    # Decoding:
    # vec[n-1] = idx % 3.
    # Let's re-verify.
    # v = [1, 2, 0]. n=3.
    # idx = 0
    # idx = 0*3 + 1 = 1
    # idx = 1*3 + 2 = 5
    # idx = 5*3 + 0 = 15.
    # 15 in base 3 is 120.
    # So v[0] is MSB.
    # Decoding:
    # 15 % 3 = 0 (v[2])
    # 5 % 3 = 2 (v[1])
    # 1 % 3 = 1 (v[0])
    # So the loop `vec[i] = idx % 3` fills vec from right to left if we don't reverse.
    # The above loop fills vec[0] with the last remainder (LSB).
    # So vec[0] corresponds to v[n-1].
    # We need to be consistent.
    # For sum operations, order doesn't matter as long as it's consistent.
    # But for `vector` input, we use the given order.
    pass

def _mod_neg(idx, size):
    # -idx mod size
    # Since size = 3^n, and we are in Z_3^n, -1 is 2, -2 is 1, -3 is 0.
    # idx is an integer representation.
    # We need to compute the integer representation of -v.
    # This is not simply (size - idx) % size because of the base-3 structure?
    # Actually, yes, it is.
    # In Z_{3^n} as a ring? No, it's Z_3^n.
    # The mapping from Z_3^n to Z_{3^n} is not a ring homomorphism for addition if we just treat it as base-3 integers?
    # Wait.
    # v = [1, 0]. w = [1, 0].
    # v+w = [2, 0].
    # idx(v) = 3. idx(w) = 3.
    # idx(v+w) = 6.
    # 3+3=6. It works for addition?
    # Let's check [2, 0] + [1, 0] = [0, 0] (mod 3).
    # idx([2,0]) = 6. idx([1,0]) = 3.
    # 6+3 = 9.
    # idx([0,0]) = 0.
    # 9 mod 9 is 0.
    # It seems that if we map v to sum(v[i] * 3^i), then addition mod 3^n corresponds to component-wise addition mod 3?
    # No.
    # v = [1, 1]. idx = 1*3 + 1 = 4.
    # w = [1, 1]. idx = 4.
    # v+w = [2, 2]. idx = 2*3 + 2 = 8.
    # 4+4=8.
    # v = [2, 1]. idx = 7.
    # w = [1, 1]. idx = 4.
    # v+w = [0, 2]. idx = 0*3 + 2 = 2.
    # 7+4 = 11. 11 mod 9 = 2.
    # It works! The mapping $v \mapsto \sum v_i 3^i$ is a group isomorphism from $(\mathbb{Z}_3)^n$ to $\mathbb{Z}_{3^n}$?
    # No, $(\mathbb{Z}_3)^n$ is not cyclic for $n>1$. $\mathbb{Z}_{3^n}$ is cyclic.
    # They are not isomorphic as groups.
    # However, the operation $a+b \pmod{3^n}$ on the indices DOES NOT correspond to component-wise addition mod 3.
    # Example: v=[1,0], w=[1,0]. v+w=[2,0].
    # idx(v)=1, idx(w)=1. Sum=2. idx([2,0])=2. OK.
    # Example: v=[1,1], w=[1,1]. v+w=[2,2].
    # idx(v)=4, idx(w)=4. Sum=8. idx([2,2])=8. OK.
    # Example: v=[2,0], w=[1,0]. v+w=[0,0].
    # idx(v)=2, idx(w)=1. Sum=3. idx([0,0])=0.
    # 3 mod 9 = 3 != 0.
    # So it DOES NOT work.
    # We cannot use simple integer addition on the indices.
    # We must perform component-wise addition.
    
    # This makes the "Sum-Set" approach expensive if $n$ is large.
    # We need to compute the index of $v+a$ for all $a \in S$.
    # This requires decoding, adding, and encoding.
    # Cost: $O(n)$ per pair.
    # Total cost: $O(|S|^2 n)$.
    # If $|S|=1000, n=10$, cost is $10^7$. Manageable.
    # If $|S|=5000, n=10$, cost is $2.5 \times 10^8$. Might be slow in Python.
    
    # Given the 300s limit, we should be careful.
    # We will implement the exact method only if $n$ is small enough.
    # Otherwise, we use a heuristic.
    pass

def priority(vector, n):
    global _sum_counts, _selected, _selected_indices, _n_global, _base_size, _start_time, _DEADLINE
    
    # Deadline check
    if time.time() - _start_time > _DEADLINE - 5:
        return 0.0

    if _n_global != n:
        _init_state(n)
        _selected_indices = []

    idx = _encode(vector, n)
    
    if _selected is not None:
        if _selected[idx]:
            return -1000.0
            
        # Check validity: -v in S+S
        # We need the index of -v.
        # Negating in Z_3^n:
        # neg_vec = [(-v[i]) % 3 for i in range(n)]
        # neg_idx = _encode(neg_vec, n)
        # If _sum_counts[neg_idx] > 0, it's invalid.
        neg_vec = [(-v) % 3 for v in vector]
        neg_idx = _encode(neg_vec, n)
        
        if _sum_counts[neg_idx] > 0:
            return -500.0 # Low priority, evaluator will likely skip

        # Calculate cost: sum of _sum_counts for all (v + a)
        # This is the number of "collisions" this vector would create in the sum set.
        # We want to minimize this.
        # cost = 0
        # for a_idx in _selected_indices:
        #     # Compute v + a_idx
        #     # This is expensive.
        #     pass
        
        # Approximation: 
        # If we can't do exact, use a heuristic.
        # Heuristic: 
        # 1. Hamming weight balance.
        # 2. Distance to "average" vector.
        
        # Let's try a fast heuristic if |S| is large or n is large.
        if len(_selected_indices) > 200 or n > 10:
            # Heuristic: Minimize sum of squares of coordinates?
            # Or just random?
            # Let's use a simple geometric heuristic:
            # Prefer vectors that are "orthogonal" to the current set's centroid?
            # Centroid C = sum(S) / |S|
            # Score = - <v, C>
            
            # But we need to be fast.
            # Let's just return a hash-based score to break ties, 
            # but prioritize vectors with weight close to n/3?
            weight = sum(vector)
            # Ideal weight for uniform distribution is n/3.
            # But cap sets often have specific weights.
            # Let's penalize deviation from n/3.
            deviation = abs(weight - n/3.0)
            return -deviation
            
        else:
            # Exact cost calculation
            cost = 0
            for a_idx in _selected_indices:
                # Decode a_idx
                a_vec = [0] * n
                temp = a_idx
                for i in range(n):
                    a_vec[i] = temp % 3
                    temp //= 3
                a_vec.reverse() # Because encoding was MSB first? 
                # Wait, my _encode does: idx = idx*3 + v[0].
                # So v[0] is MSB.
                # My decode loop: a_vec[0] = idx%3 (LSB).
                # So a_vec is reversed.
                # I need to align them.
                # Let's re-encode/decode carefully.
                
                # Actually, let's just do the addition in the encoded space?
                # No, we proved it doesn't work.
                
                # Let's optimize the addition.
                # We can pre-decode all selected vectors?
                # That uses memory.
                
                # Let's just do it.
                sum_vec = [0] * n
                for i in range(n):
                    sum_vec[i] = (vector[i] + a_vec[i]) % 3
                sum_idx = _encode(sum_vec, n)
                cost += _sum_counts[sum_idx]
                
            # We want to MINIMIZE cost.
            # So priority should be -cost.
            # Add a small hash for tie-breaking.
            h = zlib.crc32(bytes(vector))
            return -cost + (h / 100000.0)

    else:
        # Fallback for large n
        weight = sum(vector)
        deviation = abs(weight - n/3.0)
        h = zlib.crc32(bytes(vector))
        return -deviation + (h / 100000.0)

# We need to update the state when a vector is selected.
# But the interface only gives `priority`.
# How do we know which vector was selected?
# The prompt says: "vectors ... are added greedily ... when they keep the set a cap set".
# It does NOT provide a callback for "vector added".
# This is a HUGE problem.
# If we don't know what was added, we can't maintain `_selected` or `_sum_counts`.
# 
# Re-reading the prompt:
# "Interface: priority(vector: tuple, n: int) -> float; vectors of {0,1,2}^n are added greedily in descending priority when they keep the set a cap set"
# 
# This implies the evaluator:
# 1. Gets a list of all vectors (or generates them).
# 2. Calls `priority` on each.
# 3. Sorts by priority.
# 4. Iterates through the sorted list.
# 5. If adding `v` keeps the set a cap set, it adds it.
# 
# If this is the case, `priority` is called ONCE for each vector, BEFORE any selection is made.
# The priorities are STATIC.
# The greedy selection is done by the evaluator using the static priorities.
# 
# If priorities are static, my dynamic "sum-count" strategy CANNOT work because I don't know the set S when computing the priority.
# I only know the vector v and n.
# 
# This changes everything.
# The current program (1.0039) uses a "rarity" counter.
# `_type_counts` is updated every time `priority` is called.
# If `priority` is called for all vectors in a single batch, the order of calls matters.
# If the evaluator shuffles the vectors before calling `priority`, the "rarity" is random.
# If the evaluator calls `priority` in a fixed order, the "rarity" is deterministic.
# 
# If the priorities are static (computed once), then the strategy must be a STATIC scoring function that, when sorted, leads to a good cap set.
# 
# What static function leads to a large cap set?
# 1. Random: Score = hash(v). (Baseline)
# 2. Weight-based: Score = -|weight(v) - n/3|.
# 3. Spectral: Score = -|DFT(v)|.
# 
# Inspiration 1 (1.2930) used "global spectral representation".
# This suggests that a STATIC spectral score works well.
# 
# Let's implement a STATIC Spectral Score.
# Score(v) = - \sum_{\xi} |\hat{1}_{\{v\}}(\xi)|^2 ? No, that's constant.
# 
# The idea is that vectors with "smooth" Fourier transforms (low energy in high frequencies) are "better" for packing?
# Or vectors that are "far" from the origin in some sense?
# 
# Let's try: Score = - \sum_{i} v_i^2 ? (Prefers 0s? No, 0,1,2. 0^2=0, 1^2=1, 2^2=4. Prefers 0s. Bad.)
# 
# Let's try: Score = Hash. (Random)
# 
# Let's try: Score = - Variance of coordinates.
# 
# Actually, the "rarity" in the current program might be working because it acts like a randomized selection if the call order is random, or a structured selection if the order is fixed.
# 
# If the call order is fixed (e.g., lexicographic), then "rarity" picks the first few of each type, then the next few.
# This creates a balanced set.
# 
# A better static strategy: **Randomized Hash with Symmetry Breaking**.
# But we need to beat 1.0039.
# 
# Let's look at Inspiration 1 again: "scoring vectors based on the magnitude of their Discrete Fourier Transform".
# For a single vector v, the DFT is $\hat{v}(\xi) = \omega^{\langle \xi, v \rangle}$. The magnitude is always 1.
# So that doesn't make sense for a single vector.
# Unless it means the DFT of the *indicator* of the set of vectors with similar properties?
# 
# Maybe the inspiration meant: Compute the DFT of the *entire space*? No.
# 
# Let's assume the evaluator calls `priority` for each vector exactly once.
# We can use the `vector` itself to compute a score.
# 
# Strategy: **Minimize the "Energy" of the vector in the Fourier domain of the cube**.
# This is equivalent to minimizing the sum of squares of the Fourier coefficients?
# For a single vector, all coefficients have magnitude 1.
# 
# What if we score based on the **Hamming Weight**?
# Cap sets in $\mathbb{F}_3^n$ are often constructed using vectors of weight $n/3$ or similar.
# 
# Let's try a score that favors vectors with weight close to $n/3$.
# Score = - (weight - n/3)^2.
# 
# To break ties, use a hash.
# 
# This is a structural change from "rarity of type" to "geometric centrality".

def priority(vector, n):
    if time.time() - _start_time > _DEADLINE - 5:
        return 0.0
        
    weight = sum(vector)
    target = n / 3.0
    penalty = (weight - target) ** 2
    
    # Tie-breaking hash
    h = zlib.crc32(bytes(vector))
    
    # We want higher score for better priority.
    # So return -penalty + small_hash
    return -penalty + (h / 1000000.0)
