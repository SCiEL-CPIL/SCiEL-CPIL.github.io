"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque
import itertools

N = 12

# EVOLVE-BLOCK-START
def _bfs_reset_length(a, b):
    """
    Computes the exact shortest reset word length for a 2-letter automaton using bitmasks.
    Returns -1 if not synchronizing.
    Optimized to compute transitions on-the-fly to avoid O(2^N * N) precomputation overhead.
    """
    start_mask = (1 << N) - 1
    if start_mask == 0:
        return 0
    
    # BFS using bitmask states
    visited = {start_mask: 0}
    queue = deque([start_mask])
    
    # Helper to compute next mask for a given letter
    # We precompute the destination state for each input state for each letter
    # dest_a[i] is the state reached from i by 'a'
    # This is just a, b from the argument, but let's use local vars for speed
    dest_a = a
    dest_b = b
    
    while queue:
        s = queue.popleft()
        dist = visited[s]
        
        # Calculate next mask for 'a'
        next_s_a = 0
        m = s
        while m:
            # Get lowest set bit
            lsb = m & -m
            # Index is log2(lsb). Since N=12, we can use bit_length - 1
            idx = lsb.bit_length() - 1
            next_s_a |= (1 << dest_a[idx])
            m ^= lsb
            
        if next_s_a not in visited:
            if next_s_a & (next_s_a - 1) == 0:  # single bit
                return dist + 1
            visited[next_s_a] = dist + 1
            queue.append(next_s_a)
            
        # Calculate next mask for 'b'
        next_s_b = 0
        m = s
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            next_s_b |= (1 << dest_b[idx])
            m ^= lsb
            
        if next_s_b not in visited:
            if next_s_b & (next_s_b - 1) == 0:  # single bit
                return dist + 1
            visited[next_s_b] = dist + 1
            queue.append(next_s_b)
            
    return -1

def _pinwheel():
    a = [(i + 1) % N for i in range(N)]
    b = [0] * N
    for i in range(2, N):
        b[i] = i - 1
    return [a, b]

def _rank2_cycle(k):
    """
    Generate a rank-2 automaton with a cycle of length k.
    One letter is a permutation with a cycle of length k and fixed points.
    The other letter maps most states to 0, with some exceptions.
    """
    a = list(range(N))
    b = [0] * N
    
    # Create a cycle of length k among states 0..k-1
    for i in range(k):
        a[i] = (i + 1) % k
    
    # Remaining states are fixed by a
    for i in range(k, N):
        a[i] = i
    
    # b maps all states to 0 except maybe some
    b[0] = 1
    for i in range(1, N):
        b[i] = 0
    
    return [a, b]

def _systematic_search():
    """
    Systematically enumerate known structural families to find high-score automata.
    """
    candidates = []
    
    # 1. Pinwheel
    candidates.append(_pinwheel())
    
    # 2. Rank-2 cycles of various lengths
    for k in range(2, N + 1):
        candidates.append(_rank2_cycle(k))
    
    # 3. Modified pinwheels with different b mappings
    a_base = [(i + 1) % N for i in range(N)]
    for b_pattern in [
        [0] * N,
        [0] * N,  # standard pinwheel
        [i if i < N - 1 else 0 for i in range(N)],
        [i + 1 if i < N - 1 else 0 for i in range(N)],
        [N - 1 - i for i in range(N)],
    ]:
        candidates.append([a_base[:], b_pattern[:]])
    
    # 4. Try some structured permutations for a
    for perm in [
        [(i + 2) % N for i in range(N)],  # step 2
        [(i + 5) % N for i in range(N)],  # step 5
        [(i * 5) % N for i in range(N)],  # multiply by 5
    ]:
        for b_pattern in [
            [0] * N,
            [i - 1 if i > 0 else 0 for i in range(N)],
        ]:
            candidates.append([perm[:], b_pattern[:]])
    
    # Evaluate all candidates
    best_a, best_b = candidates[0]
    best_score = _bfs_reset_length(best_a, best_b)
    
    for a, b in candidates[1:]:
        score = _bfs_reset_length(a, b)
        if score > best_score:
            best_a, best_b = a[:], b[:]
            best_score = score
    
    return best_a, best_b, best_score

def _local_search(a, b, deadline):
    """
    Local search from a starting point using bitmask-based evaluation.
    """
    current_a, current_b = a[:], b[:]
    current_score = _bfs_reset_length(current_a, current_b)
    best_a, best_b = current_a[:], current_b[:]
    best_score = current_score
    
    random.seed(123)
    
    # Precompute transition tables for current automaton
    def compute_trans(a_curr, b_curr):
        trans_a = [0] * (1 << N)
        trans_b = [0] * (1 << N)
        for mask in range(1 << N):
            ma = 0
            mb = 0
            for i in range(N):
                if mask & (1 << i):
                    ma |= (1 << a_curr[i])
                    mb |= (1 << b_curr[i])
            trans_a[mask] = ma
            trans_b[mask] = mb
        return trans_a, trans_b
    
    while time.time() < deadline:
        # Generate a neighbor
        new_a = current_a[:]
        new_b = current_b[:]
        
        # Choose mutation type
        mut_type = random.random()
        
        if mut_type < 0.3:
            # Single state change in a
            idx = random.randint(0, N - 1)
            new_a[idx] = random.randint(0, N - 1)
        elif mut_type < 0.6:
            # Single state change in b
            idx = random.randint(0, N - 1)
            new_b[idx] = random.randint(0, N - 1)
        elif mut_type < 0.8:
            # Swap in a
            i1, i2 = random.sample(range(N), 2)
            new_a[i1], new_a[i2] = new_a[i2], new_a[i1]
        else:
            # Swap in b
            i1, i2 = random.sample(range(N), 2)
            new_b[i1], new_b[i2] = new_b[i2], new_b[i1]
        
        score = _bfs_reset_length(new_a, new_b)
        
        if score > current_score:
            current_a, current_b = new_a, new_b
            current_score = score
            if current_score > best_score:
                best_a, best_b = current_a[:], current_b[:]
                best_score = current_score
    
    return best_a, best_b, best_score

def construct():
    # First, do systematic search over known families
    # This is fast and provides a good starting point
    best_a, best_b, best_score = _systematic_search()
    
    # Time limit: 60s wall-clock limit for evaluator.
    # We must finish well before 60s. Let's aim for 50s to be safe.
    start_time = time.time()
    deadline = start_time + 50
    
    # If we have time, do local search from the best systematic result
    # The local search uses the optimized _bfs_reset_length, so it should be fast enough.
    if time.time() < deadline - 2:
        best_a, best_b, best_score = _local_search(best_a, best_b, deadline)
    
    # Final safety check: if we are dangerously close to the limit, return immediately
    # The local search loop already checks time, but this ensures we don't do anything else.
    return [best_a, best_b]
# EVOLVE-BLOCK-END
