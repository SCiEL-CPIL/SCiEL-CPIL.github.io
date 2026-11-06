"""Snake-in-the-box in Q_9: Simulated Annealing with specialized hypercube moves."""
import time
import random

N = 9
MASK = (1 << N) - 1

# Precompute neighbors for all 512 vertices
NEIGHBORS = [[] for _ in range(1 << N)]
for v in range(1 << N):
    for b in range(N):
        NEIGHBORS[v].append(v ^ (1 << b))

def _greedy_extend(path, rng):
    """Extend path greedily from the end. Returns extended path."""
    used = set(path)
    while True:
        cur = path[-1]
        candidates = []
        for nb in NEIGHBORS[cur]:
            if nb in used:
                continue
            # Check if adding nb violates induced path condition
            valid = True
            for p in path[:-1]:
                if (nb ^ p).bit_count() == 1:
                    valid = False
                    break
            if valid:
                candidates.append(nb)
        if not candidates:
            break
        nxt = rng.choice(candidates)
        path.append(nxt)
        used.add(nxt)
    return path

def _greedy_extend_start(path, rng):
    """Extend path greedily from the start. Returns extended path."""
    used = set(path)
    while True:
        cur = path[0]
        candidates = []
        for nb in NEIGHBORS[cur]:
            if nb in used:
                continue
            # Check if adding nb violates induced path condition
            valid = True
            for p in path[1:]:
                if (nb ^ p).bit_count() == 1:
                    valid = False
                    break
            if valid:
                candidates.append(nb)
        if not candidates:
            break
        nxt = rng.choice(candidates)
        path.insert(0, nxt)
        used.add(nxt)
    return path

def _sa_search(path, rng, deadline):
    """
    Simulated Annealing search.
    Moves:
    1. Extend end (always accept if possible)
    2. Extend start (always accept if possible)
    3. Swap internal vertex (accept based on SA criterion or if it enables extension)
    """
    current_path = list(path)
    current_len = len(current_path)
    best_path = list(current_path)
    best_len = current_len
    
    used_set = set(current_path)
    
    # SA parameters
    T = 1.0
    T_min = 0.01
    alpha = 0.9999
    
    while time.time() < deadline:
        # 1. Try to extend ends
        extended = False
        # Extend end
        cur = current_path[-1]
        for nb in NEIGHBORS[cur]:
            if nb in used_set:
                continue
            valid = True
            for p in current_path[:-1]:
                if (nb ^ p).bit_count() == 1:
                    valid = False
                    break
            if valid:
                current_path.append(nb)
                used_set.add(nb)
                current_len += 1
                extended = True
                if current_len > best_len:
                    best_path = list(current_path)
                    best_len = current_len
                break
        
        # Extend start
        if not extended:
            cur = current_path[0]
            for nb in NEIGHBORS[cur]:
                if nb in used_set:
                    continue
                valid = True
                for p in current_path[1:]:
                    if (nb ^ p).bit_count() == 1:
                        valid = False
                        break
                if valid:
                    current_path.insert(0, nb)
                    used_set.add(nb)
                    current_len += 1
                    extended = True
                    if current_len > best_len:
                        best_path = list(current_path)
                        best_len = current_len
                    break
        
        if extended:
            # If we extended, reset temperature slightly or keep going
            T = max(T * 0.99, T_min)
            continue
            
        # 2. No extension possible. Try a swap.
        if current_len < 3:
            break
            
        # Pick a random internal index
        i = rng.randint(1, current_len - 2)
        u = current_path[i-1]
        v = current_path[i]
        w = current_path[i+1]
        
        # Find the other common neighbor of u and w
        # v is one common neighbor.
        # u ^ w has 2 bits set.
        diff = u ^ w
        # The bits are b1, b2.
        # v = u ^ (1<<b1) and w = u ^ (1<<b2) is one case?
        # Actually, v is adjacent to u and w.
        # The other common neighbor v' can be found by:
        # v' = u ^ (diff & ~(v ^ u)) ?
        # Let's just iterate neighbors of u and check if they are neighbors of w.
        v_prime = None
        for nb in NEIGHBORS[u]:
            if nb == v:
                continue
            if (nb ^ w).bit_count() == 1:
                v_prime = nb
                break
        
        if v_prime is None:
            # Should not happen in hypercube for valid path
            T *= alpha
            continue
            
        if v_prime in used_set:
            # Can't swap to a vertex already in path
            T *= alpha
            continue
            
        # Check if v_prime is valid with the rest of the path
        valid = True
        for idx, p in enumerate(current_path):
            if idx == i - 1 or idx == i + 1:
                continue
            if (v_prime ^ p).bit_count() == 1:
                valid = False
                break
                
        if not valid:
            T *= alpha
            continue
            
        # Perform swap
        # Remove v
        used_set.remove(v)
        # Add v_prime
        used_set.add(v_prime)
        current_path[i] = v_prime
        
        # Acceptance criterion:
        # Since length doesn't change, we use SA to accept "neutral" moves
        # that might lead to extensions.
        # Always accept if T > T_min to explore.
        # We can be more aggressive: always accept swaps to avoid local minima of "stuck ends".
        
        T *= alpha
        
    return best_path

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    rng = random.Random(42)
    
    best_path = None
    best_len = 0
    
    # Initial greedy snake
    path = [0]
    path = _greedy_extend(path, rng)
    path = _greedy_extend_start(path, rng)
    best_path = list(path)
    best_len = len(path)
    
    # Run SA search
    improved = _sa_search(best_path, rng, deadline)
    if len(improved) > best_len:
        best_path = improved
        best_len = len(improved)
        
    # If we have time left, try a few restarts
    while time.time() < deadline - 5:
        # Try a fresh greedy start
        start_v = rng.randint(0, (1 << N) - 1)
        fresh = [start_v]
        fresh = _greedy_extend(fresh, rng)
        fresh = _greedy_extend_start(fresh, rng)
        
        if len(fresh) > best_len:
            best_path = list(fresh)
            best_len = len(fresh)
            
        # Run a short SA search on this fresh start
        # Use remaining time / 2 for this search
        sub_deadline = min(deadline - 2, time.time() + 10)
        improved_fresh = _sa_search(fresh, rng, sub_deadline)
        if len(improved_fresh) > best_len:
            best_path = list(improved_fresh)
            best_len = len(improved_fresh)
            
    return best_path
