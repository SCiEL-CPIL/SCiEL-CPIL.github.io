import time
import itertools
import numpy as np

# EVOLVE-BLOCK-START

_DEADLINE = time.time() + 300  # Safety margin from 330s limit
_optimal_penalty = None
_priority_cache = {}
_vector_tuples = {}  # Cache for converting numpy arrays to tuples

def _to_tuple(v):
    """Convert numpy array or tuple to a consistent tuple key."""
    if isinstance(v, tuple):
        return v
    # Assume numpy array or list
    return tuple(int(x) for x in v)

def _greedy_with_penalty(n, penalty_weight, time_limit=None):
    """
    Run greedy cap set construction with a dynamic penalty.
    penalty_weight: how much to penalize vectors that form lines with existing set.
    Returns: size of the resulting cap set.
    """
    start_time = time.time()
    
    # Generate all vectors in {0,1,2}^n
    # For large n, this is 3^n which can be huge.
    # However, the task implies n is small enough for the evaluator to handle.
    # If n is large (e.g., > 10), 3^n is 59049, which is manageable.
    # If n is 15, 3^15 = 14M, which might be slow for pure Python but okay for numpy.
    # Let's assume n is small enough that we can iterate.
    
    # Optimization: Pre-generate all vectors as numpy arrays for speed?
    # Or just iterate.
    
    # We need to sort by priority. But priority depends on the current set.
    # This is a "greedy with lookahead" or "adaptive greedy".
    # The interface is priority(vector, n) -> float.
    # The evaluator sorts vectors by priority DESCENDING and adds them greedily.
    # Wait, the contract says: "vectors of {0,1,2}^n are added greedily in descending priority".
    # This implies the priority is STATIC for a given vector in a single evaluation run?
    # "vectors ... are added greedily in descending priority when they keep the set a cap set".
    # Usually, in these FunSearch tasks, the priority function is called for each vector once to determine the order.
    # If the priority function is called only once per vector, it CANNOT know the current set state during the greedy addition.
    # It can only know the vector and n.
    
    # RE-READING CONTRACT:
    # "Interface: priority(vector: tuple, n: int) -> float; vectors of {0,1,2}^n are added greedily in descending priority when they keep the set a cap set"
    
    # This phrasing is ambiguous. Does it mean:
    # 1. Compute priority for all vectors, sort, then greedily add? (Static Priority)
    # 2. At each step, compute priority for remaining vectors, pick best, add if valid? (Dynamic Priority)
    
    # If it's (2), the priority function can access global state (the current set).
    # If it's (1), the priority function is stateless (or stateless per evaluation).
    
    # Let's look at the current program. It uses a cache `_priority_cache`.
    # If the evaluator calls priority(v, n) for all v, sorts, and then adds, the cache is useful.
    # If the evaluator calls priority(v, n) inside a loop `while set not full: for v in remaining: p = priority(v, n) ...`, then the priority can depend on the current set.
    
    # Most FunSearch "greedy" contracts imply that the priority function determines the ORDER.
    # If the order is fixed, it's static.
    # If the order is re-evaluated, it's dynamic.
    
    # However, the current program's strategy is "Static priority (sort-first)".
    # The comment says: "Since we're doing static priority (sort-first), we can't use current set."
    # This suggests the evaluator sorts first.
    
    # BUT, if I can change the strategy to DYNAMIC, that would be a structural change.
    # Can I?
    # If the evaluator does:
    #   priorities = {v: priority(v, n) for v in all_vectors}
    #   sorted_v = sorted(all_vectors, key=lambda v: priorities[v], reverse=True)
    #   for v in sorted_v:
    #       if is_cap_set(set + v): set.add(v)
    # Then dynamic state is impossible.
    
    # If the evaluator does:
    #   remaining = set(all_vectors)
    #   while remaining:
    #       best_v = max(remaining, key=lambda v: priority(v, n))
    #       if is_cap_set(set + best_v): set.add(best_v)
    #       remaining.remove(best_v)
    # Then dynamic state is possible.
    
    # Given the ambiguity, and the fact that the current program assumes static, I should probably stick to static OR find a way to make a "better" static priority.
    
    # HOWEVER, there is a third option.
    # What if the "priority" function is called during the greedy process?
    # Let's assume the standard FunSearch contract for "greedy" often allows the priority to depend on the current state if the implementation supports it, BUT the signature `priority(vector, n)` doesn't take the current set.
    # So, unless the function uses global state that is updated by the evaluator (which it isn't, the evaluator is external), we cannot know the current set inside `priority`.
    
    # WAIT. If the evaluator is external, how would it update the global state of the submitted program? It wouldn't.
    # So `priority` MUST be a function of `vector` and `n` only.
    # Therefore, the priority is STATIC for a given `n`.
    
    # So my "dynamic" idea is invalid under the standard interpretation where the evaluator is a black box that calls `priority`.
    
    # So I must find a BETTER STATIC PRIORITY.
    
    # Current: Balance + Structural + Hash.
    # Inspiration 1: Offline search for best static ordering.
    # This is a strong candidate. "Offline search" means: during the first call (or using a warmup phase if allowed, but we only have `priority`), we can't really do offline search for the specific `n` if `n` varies.
    # But we CAN do a "local search" on the priority function parameters if we had access to the evaluation. We don't.
    
    # Alternative Structural Change:
    # Instead of heuristic scores (balance, etc.), use a **Spectral Graph Theory** approach or **Combinatorial Design** approach.
    # 
    # Idea: The cap set problem is equivalent to finding a large independent set in a graph where edges connect points that form a line.
    # Actually, it's a 3-uniform hypergraph independent set.
    # 
    # A known heuristic for independent sets in hypergraphs is the **Greedy Algorithm with Degree-Based Prioritization**.
    # But we can't see the degree in the current set.
    # 
    # However, we can estimate the "intrinsic" degree of a vector in the full space {0,1,2}^n.
    # The number of lines passing through a vector v is fixed for all v by symmetry?
    # No, the number of lines in {0,1,2}^n is fixed. Each line has 3 points.
    # Total lines = (3^n - 3) / (3 * 2)? No.
    # Number of lines in AG(n, 3) is (3^n - 1) / (3^1 - 1) * (3^n - 3) / (3^2 - 1) ... no.
    # Number of lines = (3^n choose 2) / 3 ? No.
    # Each line is defined by a point v and a direction d (non-zero).
    # Line: {v, v+d, v+2d}.
    # Number of directions = 3^n - 1.
    # Number of points = 3^n.
    # Total incidences (point, line) = 3^n * (3^n - 1) / 2 ? No.
    # Each point is on how many lines?
    # For a fixed point v, any other point w defines a line.
    # Number of w != v is 3^n - 1.
    # Each line through v contains 2 other points.
    # So number of lines through v = (3^n - 1) / 2.
    # This is constant for all v!
    # So "degree" is not a differentiator.
    
    # What differentiates vectors?
    # The structure of the lines they form.
    # Two vectors v and w form a line with u = -v-w.
    # If we pick v, we "block" all lines through v.
    # 
    # Is there a vector that is "more central" or "less central"?
    # In the affine geometry, all points are equivalent under the group of translations and linear transformations that preserve the structure?
    # The group of affine transformations of AG(n, 3) is transitive on points.
    # So, structurally, ALL vectors are equivalent!
    # 
    # If all vectors are structurally equivalent, then any static priority that distinguishes them based on structural properties (like balance) is essentially arbitrary with respect to the geometry's symmetry.
    # 
    # Wait, is the group transitive?
    # Yes, the affine group AGL(n, 3) acts transitively on the points of AG(n, 3).
    # This means there is an isomorphism mapping any point to any other point that preserves the line structure.
    # 
    # Implication:
    # If the greedy algorithm is symmetric, the size of the cap set should be independent of which vector we pick first, provided the rest of the ordering is symmetric.
    # 
    # BUT, the greedy algorithm breaks symmetry.
    # The specific ORDER matters.
    # 
    # If all points are equivalent, why would "balance" help?
    # Maybe it doesn't, and the current score 1.238 is just due to the specific hash tiebreaker interacting with the specific greedy implementation's quirks (e.g., Python's sorting stability, or specific collision patterns).
    # 
    # OR, maybe the "balance" heuristic is a proxy for something else.
    # 
    # Let's reconsider the "Offline Search" inspiration.
    # If we can't do dynamic, and static structure is symmetric, what can we do?
    # 
    # We can use a **Randomized** priority that is deterministic per run but effectively random.
    # The current program uses a polynomial hash.
    # 
    # What if we use a **Learning-based** approach?
    # We can't train a model.
    # 
    # What if we use a **Local Search** on the priority parameters?
    # We have 300 seconds.
    # We can spend 290 seconds simulating the greedy process with different "perturbations" of the priority function to find a better static ordering.
    # 
    # Strategy:
    # 1. On the first call to `priority(v, n)`, check if we have already run the "calibration".
    # 2. If not, run a calibration phase:
    #    - Generate all vectors for this `n`.
    #    - Define a parameterized priority function P(v, theta).
    #    - Try several values of theta (or random permutations).
    #    - For each, simulate the greedy cap set construction.
    #    - Record the best theta.
    #    - Store the best theta in a global variable.
    # 3. Subsequent calls to `priority(v, n)` use the best theta.
    # 
    # Constraint: The calibration must finish well before the deadline.
    # 
    # What is the parameterized function?
    # Since structural symmetry implies no inherent "good" vector, the "best" ordering is likely one that avoids "bad" interactions early on.
    # 
    # A common heuristic for independent sets is the **Luby's Algorithm** or **Greedy with Randomization**.
    # 
    # Let's try a **Simulated Annealing** or **Hill Climbing** on the priority function parameters.
    # 
    # Parameters:
    # We can define the priority as:
    # P(v) = Base(v) + Noise(v; seed)
    # Where Base(v) is 0 (since all are equivalent) or a simple hash.
    # 
    # Actually, if all vectors are equivalent, the "best" cap set size is the same regardless of the starting vector. The variation comes from the subsequent choices.
    # 
    # Let's implement a **Random Restart Greedy** with the best result cached.
    # 
    # Plan:
    # 1. Global `_best_ordering` = None.
    # 2. In `priority(v, n)`:
    #    - If `_best_ordering` is None:
    #        - Start a timer.
    #        - Run K random greedy trials.
    #        - Keep the trial that produced the largest cap set.
    #        - Store the priority values (or the permutation) from that trial.
    #        - Ensure we stop before deadline.
    #    - Return the priority for `v` from the stored best ordering.
    # 
    # How to store the ordering?
    # We can't store a list of 3^n floats if n is large (e.g., n=15, 14M floats = 112MB, okay).
    # But we need to map `vector` -> `priority`.
    # We can store a dictionary or just re-compute the hash with the "best seed".
    # 
    # If we use a hash-based priority:
    # P(v) = Hash(v, seed)
    # We can search for the best `seed`.
    # 
    # This is much more efficient than storing the whole ordering.
    # 
    # Algorithm:
    # 1. Define `priority_with_seed(v, n, seed)`.
    #    Use a high-quality hash (e.g., xxhash if available, else zlib.crc32 or a custom mix).
    #    Return hash / max_val.
    # 
    # 2. In `priority(v, n)`:
    #    - If `_best_seed` is None:
    #        - `start_time = time.time()`
    #        - `max_time = 250` (safety margin)
    #        - `best_size = 0`
    #        - `best_seed = 0`
    #        - `seed = 0`
    #        - While `time.time() - start_time < max_time`:
    #            - `size = simulate_greedy(n, seed)`
    #            - If `size > best_size`:
    #                - `best_size = size`
    #                - `best_seed = seed`
    #            - `seed += 1`
    #            - (Optional: If size improves, reset some counter or use adaptive step)
    #        - `_best_seed = best_seed`
    #    - Return `priority_with_seed(v, n, _best_seed)`
    # 
    # This is a **Meta-Heuristic**: We are optimizing the priority function itself via search.
    # 
    # Is this "structurally different"?
    # Yes. The current program uses a fixed analytic formula (balance + hash).
    # This program uses a **search-optimized hash**.
    # 
    # Why might this work?
    # Random orderings vary in quality. Some random permutations yield larger cap sets than others.
    # By searching for the "best" random seed, we find a permutation that is particularly "good" for the greedy algorithm.
    # 
    # Risk:
    # - Simulation cost. `simulate_greedy` needs to be fast.
    # - If `n` is large, `3^n` is large.
    #   - n=10: 59049 vectors. Simulation is fast.
    #   - n=15: 14M vectors. Simulation might take 1-2 seconds in Python if not optimized.
    #     We need to optimize `simulate_greedy`.
    # 
    # Optimization for `simulate_greedy`:
    # - Use NumPy for vector operations?
    # - Or use a fast Python set/dict.
    # - Checking "is cap set" incrementally:
    #   When adding `v` to `S`, we need to check if `v` forms a line with any two existing points `a, b in S`.
    #   Line condition: `a + b + v = 0 mod 3`.
    #   So for each `a in S`, `b` must be `-a - v`.
    #   So we just need to check if `-a - v` is in `S` for any `a in S`.
    #   This is O(|S|) per insertion.
    #   Total O(N * |S|) ~ O(N^2) worst case?
    #   Actually, we only insert if valid.
    #   If we iterate through all N vectors in sorted order, and for each, check O(|S|) existing points.
    #   If |S| grows to ~N/3, total ops ~ N * N/3 = N^2/3.
    #   For N=14M, N^2 is 200 trillion. Too slow.
    # 
    # We need a faster check.
    # 
    # Alternative Check:
    # Maintain a set `S`.
    # When adding `v`, we need to ensure no `a, b in S` such that `a+b+v=0`.
    # This is equivalent to: `v` is not in `-S - S` (sumset).
    # 
    # Can we precompute conflicts?
    # For a fixed `v`, the set of "conflicting pairs" is determined.
    # 
    # If we use the **Bitset** approach?
    # Represent `S` as a bitset of length N.
    # Then `a+b+v=0` means `b = -a-v`.
    # We can precompute for each `v`, a bitset `Conflict_v` where bit `i` is 1 if `v_i` (the i-th vector) conflicts with `v` given some `a`?
    # No, the conflict depends on `a` being in `S`.
    # 
    # Let's stick to the O(|S|) check but optimize it.
    # If `n` is small (<= 10), N=59049. N^2 = 3.5e9. Still too slow for Python in 250s.
    # 
    # We need a faster simulation.
    # 
    # Idea: **Bitset Convolution**?
    # Or simply: **If n is large, we can't do many trials.**
    # 
    # Let's check the likely `n` range.
    # Cap set problems are often studied for n=8, 9, 10.
    # n=8: 6561 vectors. N^2 = 43M. Fast.
    # n=9: 19683 vectors. N^2 = 387M. ~1-2 seconds in Python.
    # n=10: 59049 vectors. N^2 = 3.5B. ~10-20 seconds in Python.
    # 
    # If n=10, we can maybe do 10-20 trials in 250s.
    # If n=15, we can do 0 trials.
    # 
    # Fallback: If n is too large for search, fall back to the current heuristic or a simple random hash.
    # 
    # Let's implement the search with a fallback.
    
    # Implementation details:
    # 1. `_simulate_greedy(n, seed)`:
    #    - Generate all vectors.
    #    - Compute priorities using `hash(v, seed)`.
    #    - Sort vectors by priority.
    #    - Iterate and add to set if valid.
    #    - Return size.
    # 
    # 2. Optimization for `_simulate_greedy`:
    #    - Use `numpy` for sorting?
    #    - Use a set for `S`.
    #    - Check: `for a in S: if (-a - v) in S: return False`
    #    - This is O(|S|).
    # 
    # 3. Vector Representation:
    #    - Tuples of ints.
    #    - Hashing tuples is fast.
    
    # Let's refine the "Hash" function.
    # We want a hash that is sensitive to the seed.
    # `hash((v, seed))`? Python's hash is randomized per process.
    # We need a deterministic hash.
    # Use `zlib.crc32`?
    # Or a custom mix.
    
    def _fast_hash(v, seed):
        # v is a tuple of ints
        # Simple mixing
        h = seed
        for x in v:
            h = (h * 31 + x + 12345) & 0xFFFFFFFF
        return h

    def _simulate_greedy(n, seed):
        # Generate all vectors
        # For efficiency, generate once and cache?
        # But n might vary.
        
        # If n is large, generating 3^n vectors is expensive.
        # We'll assume n is small enough.
        
        vectors = list(itertools.product([0, 1, 2], repeat=n))
        
        # Compute priorities
        # Using a list of (priority, index) to sort
        # Or just sort vectors by key.
        
        # To speed up sorting, we can compute the hash for each.
        # 
        # Optimization:
        # If we just want the SIZE, we don't need to store the set of vectors,
        # but we do need to check conflicts.
        
        S = set()
        
        # Sort vectors by priority
        # Sorting 59049 items is fast.
        # 
        # Key function:
        # We need to avoid calling _fast_hash repeatedly if possible, but it's O(n) per call.
        # Total O(N * n) for hashing.
        
        sorted_vectors = sorted(vectors, key=lambda v: _fast_hash(v, seed), reverse=True)
        
        for v in sorted_vectors:
            # Check if v can be added
            valid = True
            for a in S:
                # b = -a - v mod 3
                # b_i = (-a_i - v_i) % 3
                # If b in S, then line (a, b, v) exists.
                # Note: a, b, v must be distinct.
                # If b == a, then 2a + v = 0 => v = -2a = a (mod 3).
                # If v == a, we are checking if a is in S (it is).
                # So we must ensure a, b, v are distinct.
                
                # Calculate b
                # b = tuple((-a[i] - v[i]) % 3 for i in range(n))
                
                # Optimization: Pre-compute "complement" vectors?
                # No, v changes.
                
                # Let's just compute b.
                b = tuple((-a[i] - v[i]) % 3 for i in range(n))
                
                if b in S:
                    # Check distinctness
                    # If b == v, then a + v + v = 0 => a = -2v = v. So a==v.
                    # If a == v, then v is in S, so we wouldn't be adding v.
                    # So if b in S and b != v and a != v, it's a conflict.
                    # Actually, if b == v, then a = v. Since v not in S, a not in S. Contradiction.
                    # So b cannot be v if a is in S and v is not in S?
                    # Wait. If b = v, then -a - v = v => -a = 2v => a = -2v = v.
                    # So if b == v, then a == v.
                    # Since v is not in S, a is not in S.
                    # So this case is impossible for a in S.
                    
                    # If b == a, then -a - v = a => -v = 2a => v = -2a = a.
                    # So if b == a, then v == a.
                    # Since v not in S, a not in S.
                    # Impossible.
                    
                    # So if b in S, and a in S, and v not in S, then a, b, v are automatically distinct?
                    # Yes.
                    
                    valid = False
                    break
            
            if valid:
                S.add(v)
        
        return len(S)

    # Now, the main logic.
    
    def priority(vector, n):
        global _best_seed, _priority_cache
        
        # Convert vector to tuple
        v_tuple = _to_tuple(vector)
        
        # Check cache
        key = (v_tuple, n)
        if key in _priority_cache:
            return _priority_cache[key]
        
        # If we haven't calibrated, do it.
        if _best_seed is None:
            start_time = time.time()
            max_time = 250  # Safety margin
            
            # Determine if we can afford to search
            # Estimate N = 3^n
            N = 3**n
            
            # Heuristic: If N is too large, skip search.
            # Let's say if N > 100000, search is risky.
            # But even for N=59049, one simulation might take 5-10s.
            # 250s / 10s = 25 trials.
            
            best_size = 0
            best_seed = 0
            seed = 0
            
            # Try a few seeds
            # We can also try "smart" seeds, but random is fine.
            
            while time.time() - start_time < max_time:
                size = _simulate_greedy(n, seed)
                if size > best_size:
                    best_size = size
                    best_seed = seed
                    # Optional: if we found a very good one, we could stop?
                    # No, keep searching.
                
                seed += 1
                
                # If n is very large, this loop might only run once or twice.
                # If it runs once, we just use seed 0.
            
            _best_seed = best_seed
        
        # Compute priority using best_seed
        p = _fast_hash(v_tuple, _best_seed)
        # Normalize to [0, 1) or just return the int?
        # The contract says "return float".
        # Higher is better.
        # We can return the hash value directly.
        
        _priority_cache[key] = p
        return p

# EVOLVE-BLOCK-END
