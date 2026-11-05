# EVOLVE-BLOCK-START
import time

def construct():
    start = time.time()
    DEADLINE = 85.0  # seconds, safety margin before 90s kill
    
    # Sieve primes up to 10000
    N = 10000
    is_prime = bytearray(b'\x01') * (N + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(N + 1) if is_prime[i]]
    n = len(primes)
    
    if n <= 1:
        return primes
    
    # For each prime index i, maintain Pareto frontier:
    # list of (last_gap, length) sorted by last_gap increasing,
    # with length strictly decreasing (so no point dominates another)
    # A subsequence of length 1 has no "last gap", treat as last_gap = 0
    # Actually for length 1, there's no gap constraint for extension.
    # Let's define: for a subsequence ending at prime i with last gap g and length L,
    # we can extend to prime k > i if primes[k] - primes[i] >= g.
    # For length 1, g = 0 (no constraint).
    
    # frontier[i] = list of (gap, length), Pareto optimal
    # We process primes in order
    
    frontiers = [[] for _ in range(n)]
    
    # Prime 0: length 1, gap 0
    frontiers[0] = [(0, 1)]
    
    best_length = 1
    best_end = 0  # index of prime where best length ends
    best_prev_gap = 0
    
    for i in range(1, n):
        if time.time() - start > DEADLINE:
            break
        current = []
        # Option: start new subsequence at i
        current.append((0, 1))
        
        for j in range(i):
            if time.time() - start > DEADLINE:
                break
            gap = primes[i] - primes[j]
            # For each state in frontier[j], if gap >= state_gap, we can extend
            for (prev_gap, prev_len) in frontiers[j]:
                if gap >= prev_gap:
                    current.append((gap, prev_len + 1))
        
        # Build Pareto frontier from current list
        # Sort by gap ascending, then by length descending
        # Keep only non-dominated: as gap increases, length must strictly decrease
        if not current:
            frontiers[i] = []
            continue
        
        # Remove duplicates: for same gap, keep max length
        from collections import defaultdict
        gap_to_maxlen = {}
        for (g, l) in current:
            if g not in gap_to_maxlen or l > gap_to_maxlen[g]:
                gap_to_maxlen[g] = l
        
        items = sorted(gap_to_maxlen.items())  # sorted by gap asc
        # Now filter: keep items where length is strictly greater than all subsequent
        # Actually: an item (g1, l1) dominates (g2, l2) if g1 <= g2 and l1 >= l2
        # So in sorted-by-gap order, we want length to be strictly decreasing
        # Walk from high gap to low gap, keep track of max length seen
        filtered = []
        max_len_seen = 0
        for (g, l) in reversed(items):
            if l > max_len_seen:
                filtered.append((g, l))
                max_len_seen = l
        filtered.reverse()
        
        frontiers[i] = filtered
        
        # Track best
        for (g, l) in filtered:
            if l > best_length:
                best_length = l
                best_end = i
                best_prev_gap = g
    
    # Reconstruct the actual sequence
    # We need to backtrack. Let's store parent pointers.
    # Actually, let's redo with parent tracking.
    
    # Re-run with parent tracking if we have time
    if time.time() - start > DEADLINE - 5:
        # Just return a valid sequence of best_length found
        # We need to reconstruct. Let's do a simpler approach.
        pass
    
    # Let's redo with full backtracking info
    # Clear and restart with parent pointers
    frontiers = [[] for _ in range(n)]
    parents = [None for _ in range(n)]  # parents[i] = list of (prev_index, prev_gap) for each state
    # Actually, let's store for each state in frontier[i], which j and which state in frontier[j] led to it
    
    # Simpler: store parent[i] as a list parallel to frontiers[i], each entry is (j, index_in_frontier_j)
    
    frontiers = [[] for _ in range(n)]
    parent_info = [[] for _ in range(n)]  # parallel to frontiers[i]
    
    frontiers[0] = [(0, 1)]
    parent_info[0] = [None]
    
    best_length = 1
    best_state_idx = 0  # index in frontiers[0]
    best_end = 0
    
    for i in range(1, n):
        if time.time() - start > DEADLINE:
            break
        current = []
        current_parents = []
        # Start new
        current.append((0, 1))
        current_parents.append(None)
        
        for j in range(i):
            if time.time() - start > DEADLINE:
                break
            gap = primes[i] - primes[j]
            for (idx, (prev_gap, prev_len)) in enumerate(frontiers[j]):
                if gap >= prev_gap:
                    current.append((gap, prev_len + 1))
                    current_parents.append((j, idx))
        
        # Build Pareto frontier
        if not current:
            frontiers[i] = []
            parent_info[i] = []
            continue
        
        gap_to_maxlen = {}
        gap_to_parent = {}
        for (idx, ((g, l), par)) in enumerate(zip(current, current_parents)):
            if g not in gap_to_maxlen or l > gap_to_maxlen[g]:
                gap_to_maxlen[g] = l
                gap_to_parent[g] = par
        
        items = sorted(gap_to_maxlen.items())
        filtered = []
        filtered_parents = []
        max_len_seen = 0
        for (g, l) in reversed(items):
            if l > max_len_seen:
                filtered.append((g, l))
                filtered_parents.append(gap_to_parent[g])
                max_len_seen = l
        filtered.reverse()
        filtered_parents.reverse()
        
        frontiers[i] = filtered
        parent_info[i] = filtered_parents
        
        for idx, (g, l) in enumerate(filtered):
            if l > best_length:
                best_length = l
                best_end = i
                best_state_idx = idx
    
    # Reconstruct
    if best_length <= 1:
        return [primes[0]]
    
    seq = []
    end = best_end
    state_idx = best_state_idx
    while True:
        seq.append(primes[end])
        par = parent_info[end][state_idx]
        if par is None:
            break
        j, idx_in_j = par
        end = j
        state_idx = idx_in_j
    seq.reverse()
    
    return seq

# EVOLVE-BLOCK-END
