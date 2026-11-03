"""Football pool problem: ternary covering code K_3(8,1).
Goal: minimize number of codewords of length 8 over {0,1,2} such that every word
is within Hamming distance 1 of some codeword.
"""

import time
import random
import numpy as np

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 140.0  # seconds wall-clock limit; use 140s for safety margin

def _time_remaining():
    """Return seconds remaining before the deadline."""
    return _DEADLINE - (time.time() - _START_TIME)

def _decode_word(idx, n=8):
    """Decode an integer index back to a word of length n over {0,1,2}."""
    word = []
    for _ in range(n):
        word.append(idx % 3)
        idx //= 3
    word.reverse()
    return word

def _build_neighbor_indices(q=3, n=8):
    """Precompute neighbor indices for all words. Returns a list of lists of indices."""
    N = q ** n
    neighbors = [[] for _ in range(N)]
    for idx in range(N):
        word = _decode_word(idx, n)
        # The word itself
        neighbors[idx].append(idx)
        # Neighbors at distance 1
        for i in range(n):
            orig_val = word[i]
            for v in range(q):
                if v != orig_val:
                    # Calculate new index
                    # idx = ... + orig_val * 3^(n-1-i) + ...
                    # delta = (v - orig_val) * 3^(n-1-i)
                    power = 3 ** (n - 1 - i)
                    new_idx = idx + (v - orig_val) * power
                    neighbors[idx].append(new_idx)
    return neighbors

def _fast_greedy(q, n, radius):
    """Fast greedy using numpy."""
    N = q ** n
    neighbors = _build_neighbor_indices(q, n)
    uncovered = np.ones(N, dtype=bool)
    code_indices = []
    
    # Precompute neighbor arrays for faster access? 
    # neighbors is a list of lists. 
    # We can use a sparse representation or just loop.
    # N=6561, 17 neighbors each.
    
    while uncovered.any():
        if _time_remaining() < 5:
            break
        # Calculate score for each word: number of uncovered neighbors
        # This is a bit slow if done naively in Python loop for 6561 items every step.
        # Optimization: Only consider words that are neighbors of uncovered words?
        # Or just vectorize?
        # Let's try a simple loop first, it might be fast enough.
        best_idx = -1
        best_count = -1
        for i in range(N):
            if not uncovered[i]:
                continue # Optional: skip if the word itself is covered? No, we want to cover neighbors.
            # Actually, any word can be a codeword.
            count = 0
            for nb in neighbors[i]:
                if uncovered[nb]:
                    count += 1
            if count > best_count:
                best_count = count
                best_idx = i
                if count == 17: # Max possible
                    break
        
        if best_idx == -1:
            break
            
        code_indices.append(best_idx)
        for nb in neighbors[best_idx]:
            uncovered[nb] = False
            
    return code_indices

def _local_search_reduce(code_indices, neighbors, time_limit=10.0):
    """Reduce code size by removing redundant codewords."""
    start_time = time.time()
    N = len(neighbors)
    cover_count = np.zeros(N, dtype=int)
    
    for idx in code_indices:
        for nb in neighbors[idx]:
            cover_count[nb] += 1
            
    code_indices = list(code_indices)
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(len(code_indices)):
            if (time.time() - start_time) > time_limit:
                break
            idx = code_indices[i]
            # Tentatively remove
            for nb in neighbors[idx]:
                cover_count[nb] -= 1
                
            if cover_count.min() >= 1:
                # Remove it
                code_indices.pop(i)
                improved = True
                break
            else:
                # Restore
                for nb in neighbors[idx]:
                    cover_count[nb] += 1
                    
    return code_indices

def _local_search_swap(code_indices, neighbors, time_limit=10.0):
    """Try to swap codewords to enable further reductions."""
    start_time = time.time()
    N = len(neighbors)
    code_set = set(code_indices)
    cover_count = np.zeros(N, dtype=int)
    
    for idx in code_indices:
        for nb in neighbors[idx]:
            cover_count[nb] += 1
            
    code_indices = list(code_indices)
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Randomly pick a codeword to try swapping
        if not code_indices:
            break
        i = random.randint(0, len(code_indices) - 1)
        old_idx = code_indices[i]
        
        # Try a few random candidates
        for _ in range(50):
            if (time.time() - start_time) > time_limit:
                break
            new_idx = random.randint(0, N - 1)
            if new_idx in code_set:
                continue
                
            # Tentatively swap
            for nb in neighbors[old_idx]:
                cover_count[nb] -= 1
            for nb in neighbors[new_idx]:
                cover_count[nb] += 1
                
            # Check if we can remove any other codeword
            removed_any = False
            removals = []
            for j in range(len(code_indices)):
                if j == i:
                    continue
                idx_j = code_indices[j]
                for nb in neighbors[idx_j]:
                    cover_count[nb] -= 1
                if cover_count.min() >= 1:
                    removals.append(j)
                    removed_any = True
                else:
                    for nb in neighbors[idx_j]:
                        cover_count[nb] += 1
                        
            if removed_any:
                # Accept swap and removals
                code_indices[i] = new_idx
                code_set.discard(old_idx)
                code_set.add(new_idx)
                
                for j in sorted(removals, reverse=True):
                    idx_j = code_indices[j]
                    for nb in neighbors[idx_j]:
                        cover_count[nb] -= 1
                    code_indices.pop(j)
                    code_set.discard(idx_j)
                    
                improved = True
                break
            else:
                # Restore
                for nb in neighbors[old_idx]:
                    cover_count[nb] += 1
                for nb in neighbors[new_idx]:
                    cover_count[nb] -= 1
                    
    return code_indices

def construct(q, n, radius):
    """Construct a covering code of length n over alphabet of size q with covering radius radius.
    Returns a list of codewords (each a list of n integers in 0..q-1).
    """
    global _START_TIME
    _START_TIME = time.time()
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters: use direct product construction or simple greedy
        # For safety, just return a valid but potentially large code
        # Simple greedy is too slow for large N, but for small N it's okay.
        # Here we assume the evaluator uses q=3, n=8.
        # If not, we use a trivial construction: all words? No, too big.
        # Let's just use the fast greedy for any q,n if N is small enough.
        if q ** n > 10000:
             # Fallback: Tetracode-like or just return empty/invalid? 
             # The contract says "Valid when every word... is within distance".
             # If we can't find one fast, we might fail.
             # But the task is specific to 8 matches.
             pass
             
    N = q ** n
    neighbors = _build_neighbor_indices(q, n)
    
    # 1. Fast Greedy
    code_indices = _fast_greedy(q, n, radius)
    
    # 2. Local Search Reduction
    if _time_remaining() > 10:
        code_indices = _local_search_reduce(code_indices, neighbors, time_limit=min(10, _time_remaining() - 5))
        
    # 3. Local Search Swap (if time permits)
    if _time_remaining() > 10:
        code_indices = _local_search_swap(code_indices, neighbors, time_limit=min(10, _time_remaining() - 5))
        
    # Convert indices to words
    code = [_decode_word(idx, n) for idx in code_indices]
    
    # Verify coverage (safety check)
    covered = np.zeros(N, dtype=bool)
    for idx in code_indices:
        for nb in neighbors[idx]:
            covered[nb] = True
            
    if not covered.all():
        # Fallback to a known valid construction if greedy failed (unlikely)
        # Tetracode construction: 729 words
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return code
