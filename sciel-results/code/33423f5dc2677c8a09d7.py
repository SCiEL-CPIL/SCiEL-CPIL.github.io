import random
import time
import numpy as np

# Global cache for priority sets
_PRIORITY_CACHE = {}
_DEADLINE = time.time() + 320.0  # Safety margin before 330s kill

def _generate_vectors(n):
    """Generate all 3^n vectors as numpy array of shape (3^n, n)."""
    total = 3 ** n
    # Using base-3 representation for efficiency
    vectors = np.zeros((total, n), dtype=np.int8)
    for i in range(n):
        base = 3 ** (n - 1 - i)
        # Create a pattern for this digit
        # The digit changes every 3^i steps, cycling 0, 1, 2
        # Pattern length is 3^(n-i)
        # Within that, 0 for 3^i, 1 for 3^i, 2 for 3^i
        block = 3 ** i
        pattern = np.tile([0, 1, 2], block)
        repeat = 3 ** (n - i - 1)
        col = np.tile(pattern, repeat)
        vectors[:, i] = col
    return vectors

def _greedy_cap_set_numpy(vectors, n, time_limit):
    """
    Greedily build a cap set using numpy for speed.
    vectors: numpy array of shape (3^n, n)
    Returns a boolean mask of selected vectors.
    """
    start_time = time.time()
    m = len(vectors)
    
    # Random permutation of indices
    perm = np.random.permutation(m)
    
    selected = np.zeros(m, dtype=bool)
    forbidden = np.zeros(m, dtype=bool)
    selected_count = 0
    
    # Precompute some things? 
    # The check for a new vector v against existing selected vectors S is:
    # For each a in S, the third point w = (2*v - a) % 3 must not be in S.
    # Also v must not be in forbidden (which is derived from previous additions).
    
    # Actually, maintaining a 'forbidden' set is efficient.
    # When we add v, we update forbidden.
    # For each a in S (old S, before adding v), the new forbidden points are:
    # w1 = (2*v - a) % 3
    # w2 = (2*a - v) % 3
    # Note: v itself is added to S, so it becomes "selected".
    # The pair (v, v) gives w = v, which is selected, not forbidden.
    # The pair (v, a) gives w1 and w2.
    
    # To update forbidden efficiently:
    # We need to find the indices of w1 and w2 in the original vector list.
    # This requires a mapping from vector tuple to index.
    # Since vectors are unique, we can use a dictionary mapping tuple -> index.
    # Or, since we have the numpy array, we can flatten and use a lookup table?
    # For n=10, 3^10 = 59049. A dictionary is fine.
    
    # Let's build the index map once outside if possible, or inside.
    # It's better to pass it in or build it once.
    
    return selected, selected_count

def _build_index_map(vectors):
    """Build a mapping from tuple to index."""
    index_map = {}
    for i, v in enumerate(vectors):
        index_map[tuple(v)] = i
    return index_map

def _greedy_cap_set_fast(vectors, index_map, n, time_limit):
    """
    Optimized greedy cap set construction.
    vectors: numpy array (m, n)
    index_map: dict tuple -> int
    """
    start_time = time.time()
    m = len(vectors)
    
    perm = np.random.permutation(m)
    
    selected = np.zeros(m, dtype=bool)
    forbidden = np.zeros(m, dtype=bool)
    selected_indices = [] # List of indices of selected vectors
    
    # Precompute vectors as list of tuples for fast access?
    # Or just use the numpy array and index_map.
    
    for idx in perm:
        if time.time() - start_time > time_limit:
            break
        
        if selected[idx] or forbidden[idx]:
            continue
        
        # Check if adding idx creates a cap set violation.
        # Actually, the 'forbidden' set logic handles this.
        # If idx is not forbidden, we can add it.
        # Then we must update forbidden.
        
        selected[idx] = True
        selected_indices.append(idx)
        
        # Update forbidden based on new vector v and all existing selected a.
        v = vectors[idx]
        
        # We need to iterate over existing selected vectors (excluding v itself, but v was just added so it's in the list)
        # To avoid checking v against itself, we can iterate over selected_indices[:-1]
        
        for a_idx in selected_indices[:-1]:
            a = vectors[a_idx]
            
            # w1 = 2*v - a
            w1 = (2 * v - a) % 3
            # w2 = 2*a - v
            w2 = (2 * a - v) % 3
            
            # Map back to indices
            i1 = index_map.get(tuple(w1))
            i2 = index_map.get(tuple(w2))
            
            if i1 is not None:
                forbidden[i1] = True
            if i2 is not None:
                forbidden[i2] = True
                
    return selected

def priority(vector, n):
    if n not in _PRIORITY_CACHE:
        # Check time
        if time.time() > _DEADLINE - 10:
             # Not enough time, return random or default
             _PRIORITY_CACHE[n] = set()
             return 0.0

        vectors = _generate_vectors(n)
        index_map = _build_index_map(vectors)
        
        best_mask = None
        best_size = 0
        
        # Run multiple restarts
        # Estimate time per restart
        # For n=10, m=59049. Greedy loop is O(m * K) where K is set size.
        # K ~ 100-200. m*K ~ 10^7. With numpy/dict overhead, maybe 0.5-1s per restart?
        # Let's try to fit as many as possible in 200s.
        
        start_search = time.time()
        max_search_time = 250.0 # Leave 60s for safety
        
        random.seed(42)
        np.random.seed(42)
        
        restart_count = 0
        while time.time() - start_search < max_search_time:
            # Calculate remaining time for this restart
            remaining = max_search_time - (time.time() - start_search)
            if remaining < 1.0:
                break
                
            # Run greedy with a sub-limit to allow multiple restarts
            # Or just run one long greedy?
            # Randomized greedy with full time is often better than many short ones?
            # Actually, for cap sets, the "greedy" is very sensitive to order.
            # Randomizing the order is key.
            # Let's run a greedy that uses the full remaining time? No, that's 1 restart.
            # Better to run multiple restarts with a fixed time limit each, or just loop.
            
            # Let's try running a greedy that stops when time is up for that specific run,
            # but we want multiple runs.
            # So we set a time_limit for this specific run.
            # If we have 250s, and a run takes 5s, we get 50 runs.
            
            # How long does one run take?
            # We don't know yet. Let's assume ~5s for n=10.
            # So we can do many.
            
            # Let's just run one greedy with the *remaining* time?
            # No, if the first run takes 250s, we get 1 run.
            # If the first run takes 1s, we get 250 runs.
            # The greedy loop breaks on time limit.
            # So we can pass the *total remaining* time as the limit for this run?
            # No, that would just make it run until the end.
            
            # Strategy: Run greedy with a fixed time slice?
            # Or just let it run until the global deadline?
            # If we let it run until the global deadline, we only get 1 restart.
            # Is 1 restart better than many?
            # Usually, for greedy, diversity (many restarts) is better.
            
            # Let's estimate time per restart by running a short one first?
            # Or just use a heuristic: 5 seconds per restart.
            
            time_limit_for_restart = 5.0
            if time.time() + time_limit_for_restart > _DEADLINE - 10:
                 time_limit_for_restart = _DEADLINE - 10 - time.time()
                 if time_limit_for_restart < 0.1:
                     break
            
            mask = _greedy_cap_set_fast(vectors, index_map, n, time_limit_for_restart)
            size = np.sum(mask)
            
            if size > best_size:
                best_size = size
                best_mask = mask.copy()
                
            restart_count += 1
            
            # If the run finished early (didn't hit time limit), we can do more.
            # _greedy_cap_set_fast doesn't report if it finished early.
            # We can check if it processed all vectors?
            # For now, assume it hits the limit or finishes quickly.
            
        # Convert best_mask to a set of tuples for fast lookup
        if best_mask is not None:
            best_set = set()
            for i in range(len(vectors)):
                if best_mask[i]:
                    best_set.add(tuple(vectors[i]))
            _PRIORITY_CACHE[n] = best_set
        else:
            _PRIORITY_CACHE[n] = set()

    return 1.0 if tuple(vector) in _PRIORITY_CACHE[n] else 0.0
