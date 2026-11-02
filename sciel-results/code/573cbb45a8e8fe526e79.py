import itertools
import time
import random
import sys

# Global cache for neighbors to avoid recomputation if called multiple times (though construct is called once)
_NEIGHBORS_CACHE = {}

def _get_neighbors(q, n, radius):
    key = (q, n, radius)
    if key in _NEIGHBORS_CACHE:
        return _NEIGHBORS_CACHE[key]
    
    # Precompute neighbors for all words
    # Word is a tuple of ints
    # We map word to index for faster array access
    # Index = w[0]*q^(n-1) + ... + w[n-1]
    
    total_words = q ** n
    neighbors = [[] for _ in range(total_words)]
    
    # Generate all words
    all_words = []
    for idx in range(total_words):
        w = [0] * n
        rem = idx
        for i in range(n - 1, -1, -1):
            w[i] = rem % q
            rem //= q
        all_words.append(tuple(w))
        
    # For each word, find neighbors within radius
    for idx, w in enumerate(all_words):
        # Neighbors include the word itself and words at dist 1..radius
        # For radius 1, it's the word itself + 1-step changes
        # We only need to check dist 0 and 1 for covering radius 1
        # Actually, the problem asks for covering radius 1.
        # A codeword covers itself and its 1-neighbors.
        
        neigh = [idx] # self
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                # Create new word
                new_w = list(w)
                new_w[pos] = new_val
                new_idx = 0
                base = 1
                for i in range(n - 1, -1, -1):
                    new_idx += new_w[i] * base
                    base *= q
                neigh.append(new_idx)
        neighbors[idx] = neigh
        
    _NEIGHBORS_CACHE[key] = (neighbors, all_words, total_words)
    return _NEIGHBORS_CACHE[key]

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    neighbors, all_words, total_words = _get_neighbors(q, n, radius)
    
    # If the problem is small enough, we could do exact, but n=8 is large for exact.
    # We use Greedy + Pruning with Restart.
    
    best_code = None
    best_size = float('inf')
    
    # Initial greedy run with deterministic tie-breaking (lowest index)
    # Then random restarts
    
    def run_greedy(seed):
        random.seed(seed)
        uncovered = [1] * total_words
        code = []
        
        # To speed up max finding, we could use a heap, but N=6561 is small.
        # Simple scan is O(N) per step. Steps ~ K. Total O(K*N).
        # 486 * 6561 ~ 3.2M ops. Fast.
        
        remaining = total_words
        while remaining > 0:
            # Find word with max uncovered count
            max_count = -1
            best_idx = -1
            # Random tie breaking
            # We can iterate and pick randomly among maxes
            # Or just pick first max. To add diversity, we can shuffle the order of checks?
            # No, shuffling N items is O(N log N).
            # Let's just pick the first max. The randomness comes from the fact that
            # if we start with a slightly different state, the maxes change.
            # But here state is deterministic if tie-breaking is deterministic.
            # So we need random tie-breaking.
            
            # Efficient random tie-breaking:
            # Collect all indices with max_count? That's O(N).
            # We do O(N) scan anyway.
            
            candidates = []
            for i in range(total_words):
                if uncovered[i] > max_count:
                    max_count = uncovered[i]
                    candidates = [i]
                elif uncovered[i] == max_count:
                    candidates.append(i)
            
            if not candidates:
                break
                
            # Pick random candidate
            w_idx = random.choice(candidates)
            
            code.append(w_idx)
            
            # Update uncovered
            for nb in neighbors[w_idx]:
                if uncovered[nb] > 0:
                    uncovered[nb] -= 1
                    if uncovered[nb] == 0:
                        remaining -= 1
            
            # Check deadline
            if time.time() > deadline:
                return code, True # Timeout flag
        
        # Pruning Phase
        # Calculate coverage count for each word
        coverage = [0] * total_words
        for w_idx in code:
            for nb in neighbors[w_idx]:
                coverage[nb] += 1
        
        # Try to remove codewords
        changed = True
        while changed:
            changed = False
            new_code = []
            for w_idx in code:
                # Check if any word covered by w_idx has coverage 1
                removable = True
                for nb in neighbors[w_idx]:
                    if coverage[nb] == 1:
                        removable = False
                        break
                
                if removable:
                    # Remove w_idx
                    for nb in neighbors[w_idx]:
                        coverage[nb] -= 1
                    changed = True
                else:
                    new_code.append(w_idx)
            code = new_code
            
            if time.time() > deadline:
                break
                
        return code, False

    # Run 1 deterministic-ish (seed 0)
    code, timeout = run_greedy(0)
    if not timeout:
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
    
    # Run random restarts
    # Estimate time per run. 
    # If first run took T, we can do ~ (115 - T_elapsed) / T runs.
    elapsed = time.time() - start_time
    if elapsed > 0.1:
        runs_per_sec = 1.0 / elapsed
        max_restarts = int((deadline - time.time()) * runs_per_sec * 0.8)
    else:
        max_restarts = 1000 # Fallback
    
    for i in range(1, max_restarts + 1):
        if time.time() > deadline:
            break
        code, timeout = run_greedy(i)
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
    
    # Convert indices back to words
    if best_code is None:
        # Fallback: Tetracode direct sum
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(all_words[i]) for i in best_code]
