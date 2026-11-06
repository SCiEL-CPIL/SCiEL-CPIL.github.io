"""Strategy: Greedy Difference Selection + Vectorized Local Search.
Instead of random SA on S, we first greedily pick differences to maximize 
structural potential, then refine S using fast bitset-based scoring.
"""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _score_from_bitset(bitset):
    """
    Compute the number of distinct differences d > 0 such that 
    there exists a with a, a+d, a+2d in the set represented by bitset.
    bitset is a numpy array of bools, length 512.
    """
    # Indices of elements in the set
    idx = np.nonzero(bitset)[0]
    if len(idx) < 3:
        return 0
    
    # To count distinct d, we can iterate over pairs (a, b) where a < b
    # d = b - a
    # Check if a + 2d is in set.
    # Since n=32, n^2 is small. But we can be smarter.
    # Let's just do the O(n^2) check with numpy vectorization for the inner loop?
    # Actually, for n=32, pure python might be faster than numpy overhead for tiny arrays.
    # But we are in a loop potentially.
    # Let's stick to a very fast python loop for the score, as it's called many times.
    # Or better, pre-compute? No, set changes.
    
    # Fast python implementation for small n
    S_set = set(idx.tolist())
    diffs = set()
    # Optimization: iterate a, then b
    # We can break early if d is too large? No.
    for a in idx:
        for b in idx:
            if b > a:
                d = b - a
                # Check if d already in diffs? 
                # If we just want count, we can add to set.
                # To speed up, check if c exists
                c = a + 2 * d
                if c in S_set:
                    diffs.add(d)
    return len(diffs)

def _random_bitset():
    """Generate a random 32-element bitset."""
    bitset = np.zeros(512, dtype=bool)
    idx = np.random.choice(512, size=32, replace=False)
    bitset[idx] = True
    return bitset

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Phase 1: Greedy Construction of Differences
    # We want to pick a set of differences D such that we can find S realizing them.
    # This is hard. Instead, let's use a heuristic:
    # Start with a random S.
    # Identify "promising" differences: those that are "easy" to satisfy or 
    # that cover many potential triples.
    
    # Actually, a more direct approach for "many distinct differences":
    # Small differences are easier to pack. Large differences require large spans.
    # A set of consecutive small integers {1, ..., 32} has differences 1..31.
    # Score = 31.
    # Can we do better?
    # If we use {1, ..., 16, 100, ..., 115}, we get diffs from first block, second block, and cross.
    # But cross diffs are large and unique.
    
    # Let's try a hybrid:
    # 1. Generate several candidate sets using different heuristics.
    # 2. Run fast local search on the best one.
    
    candidates = []
    
    # Heuristic 1: Consecutive block
    s1 = np.zeros(512, dtype=bool)
    s1[:32] = True
    candidates.append(s1)
    
    # Heuristic 2: Two blocks
    s2 = np.zeros(512, dtype=bool)
    s2[:16] = True
    s2[100:116] = True
    candidates.append(s2)
    
    # Heuristic 3: Random
    for _ in range(5):
        candidates.append(_random_bitset())
    
    # Heuristic 4: Greedy "spread"
    # Try to maximize spread to get large diffs?
    # No, small diffs are denser.
    
    best_bitset = None
    best_score = -1
    
    for b in candidates:
        sc = _score_from_bitset(b)
        if sc > best_score:
            best_score = sc
            best_bitset = b.copy()
    
    # Phase 2: Vectorized Local Search (Simulated Annealing / Hill Climbing)
    # We use the fast scorer.
    # Move: Swap one element in, one out.
    # To make it fast, we pre-compute the "cost" of removing/adding?
    # No, score is global. But n=32 is small.
    
    # Let's optimize the score function further if possible.
    # The current _score_from_bitset is O(n^2) with set lookups.
    # For n=32, this is ~1000 ops. Very fast in Python.
    
    current_bitset = best_bitset
    current_score = best_score
    T = 5.0
    cooling = 0.999
    min_T = 0.1
    
    # Precompute list of indices for faster access?
    # We will convert to list for the loop, then back to bitset for scoring?
    # Or keep as list and use a fast set-based scorer?
    # Let's keep as list for the search, it's easier to swap.
    
    current_list = np.nonzero(current_bitset)[0].tolist()
    
    # Fast scorer for list
    def _score_list(L):
        S_set = set(L)
        diffs = set()
        # n=32
        for i in range(len(L)):
            a = L[i]
            for j in range(len(L)):
                if i == j: continue
                b = L[j]
                d = b - a
                if d > 0:
                    c = a + 2 * d
                    if c in S_set:
                        diffs.add(d)
        return len(diffs)
    
    # Verify initial score
    current_score = _score_list(current_list)
    best_list = current_list.copy()
    best_score = current_score
    
    # Local Search
    while time.time() < deadline and T > min_T:
        # Generate neighbor: Swap random element in current with random element from 1..512 not in current
        idx_in = random.randint(0, 31)
        elem_out = current_list[idx_in]
        
        # Pick random elem in
        elem_in = random.randint(1, 512)
        # Ensure it's not already in the list (approx 31/512 chance it is, retry if so)
        # To avoid O(n) check, we can just check set
        # But we don't have set in loop.
        # Let's just try. If it's in, score is same (if we swap out and in same? no, we remove one, add one).
        # If elem_in is in current_list, and elem_in != elem_out, then we are just moving it?
        # No, we remove elem_out. If elem_in is in current_list, we add it back?
        # Logic: new_list = current_list - {elem_out} + {elem_in}
        # If elem_in is in current_list and elem_in != elem_out, size stays 32?
        # No. Set semantics.
        # If elem_in in current_list:
        #   If elem_in == elem_out: no change.
        #   Else: we remove elem_out, add elem_in (which was already there). 
        #   Wait, if elem_in is in current_list, and we remove elem_out, elem_in is still there unless elem_in == elem_out.
        #   So new_list is same as current_list minus elem_out plus elem_in.
        #   If elem_in was already in list, and elem_out != elem_in, then we lost elem_out and gained nothing new?
        #   No, the set operation {x} - {y} + {z}.
        #   If z is in {x} and z != y, then z is still in the set. y is removed.
        #   So size becomes 31. We need size 32.
        #   So we MUST pick elem_in NOT in current_list.
        
        # Efficient pick:
        # Since 32/512 is small, random pick has 94% chance of being valid.
        while elem_in in current_list:
            elem_in = random.randint(1, 512)
            
        new_list = current_list.copy()
        new_list[idx_in] = elem_in
        # Note: this works because we ensured elem_in is not in the list, 
        # and we are replacing an existing element.
        # Wait, if elem_in is not in the list, and we replace elem_out with elem_in, 
        # the list remains unique.
        
        new_score = _score_list(new_list)
        
        delta = new_score - current_score
        
        if delta > 0 or random.random() < np.exp(delta / T):
            current_list = new_list
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_list = current_list.copy()
        
        T *= cooling
        
        # Periodic check
        if time.time() > deadline - 0.5:
            break
            
    return best_list
# EVOLVE-BLOCK-END
