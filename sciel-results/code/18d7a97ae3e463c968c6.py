import time
import random
import math

# Pre-compute neighbor offsets for 3D
_NEIGHBOR_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

def _energy(sequence, chain):
    """Calculate total energy (negative of contact count)."""
    if not chain:
        return 0
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                q = (p[0] + dx, p[1] + dy, p[2] + dz)
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _grow_greedy(sequence, dimension, rng, seed_pos=(0,0,0)):
    """
    Greedy growth with backtracking-lite (random restarts on dead end).
    Returns a valid self-avoiding chain or None if it fails (unlikely with randomness).
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [seed_pos]
    
    chain = [seed_pos]
    occupied = {seed_pos: 0}
    
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        
        # Get valid neighbors
        valid_neighbors = []
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            q = (last[0] + dx, last[1] + dy, last[2] + dz)
            if q not in occupied:
                valid_neighbors.append(q)
        
        if not valid_neighbors:
            return None # Dead end
        
        # Score candidates
        best_score = -1e9
        best_site = None
        
        for q in valid_neighbors:
            score = 0.0
            
            # 1. Immediate H-H contact gain
            if sequence[i] == 'H':
                for dx, dy, dz in _NEIGHBOR_OFFSETS:
                    r = (q[0] + dx, q[1] + dy, q[2] + dz)
                    if r in occupied:
                        idx = occupied[r]
                        # Only count if not the previous residue (chain adjacency is not a contact)
                        if idx != i - 1:
                            if sequence[idx] == 'H':
                                score += 1.5 # Strong weight
            
            # 2. Future potential: Free space
            free = 0
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                r = (q[0] + dx, q[1] + dy, q[2] + dz)
                if r not in occupied:
                    free += 1
            score += free * 0.1
            
            # 3. Randomness
            score += rng.random() * 0.5
            
            if score > best_score:
                best_score = score
                best_site = q
        
        if best_site is None:
            return None
            
        chain.append(best_site)
        occupied[best_site] = i
        
    return chain

def _csp_search(sequence, n, start_time, max_time, best_chain_holder):
    """
    Depth-First Search with Branch and Bound and Forward Checking.
    best_chain_holder: list [chain, energy]
    """
    is_h = [1 if s == 'H' else 0 for s in sequence]
    occupied = {}
    chain = [None] * n
    chain[0] = (0, 0, 0)
    occupied[(0, 0, 0)] = 0
    current_contacts = 0
    node_count = 0
    max_nodes = 500000 # Hard limit to prevent infinite loops in Python
    
    def dfs(i):
        nonlocal current_contacts
        nonlocal node_count
        
        if time.time() - start_time > max_time:
            return
        if node_count > max_nodes:
            return
            
        node_count += 1
        
        if i == n:
            # Complete chain found
            energy = -current_contacts
            if best_chain_holder[1] is None or energy < best_chain_holder[1]:
                best_chain_holder[0] = chain.copy()
                best_chain_holder[1] = energy
            return
        
        last = chain[i-1]
        
        # Generate valid neighbors for i
        valid_neighbors = []
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            q = (last[0] + dx, last[1] + dy, last[2] + dz)
            if q not in occupied:
                valid_neighbors.append(q)
        
        if not valid_neighbors:
            return
        
        # Value Ordering: Sort by immediate contact gain, then free space
        def score_pos(q):
            gain = 0
            if is_h[i]:
                for dx, dy, dz in _NEIGHBOR_OFFSETS:
                    r = (q[0] + dx, q[1] + dy, q[2] + dz)
                    if r in occupied:
                        idx = occupied[r]
                        if idx != i - 1:
                            if is_h[idx]:
                                gain += 1
            # Tie-breaker: Free space (to avoid dead ends)
            free = 0
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                r = (q[0] + dx, q[1] + dy, q[2] + dz)
                if r not in occupied:
                    free += 1
            return (gain, free)
            
        valid_neighbors.sort(key=score_pos, reverse=True)
        
        for q in valid_neighbors:
            # Forward Checking: Ensure next residue (i+1) has at least one move
            if i + 1 < n:
                next_valid = False
                for dx, dy, dz in _NEIGHBOR_OFFSETS:
                    r = (q[0] + dx, q[1] + dy, q[2] + dz)
                    if r not in occupied and r != last:
                        next_valid = True
                        break
                if not next_valid:
                    continue
            
            # Place i
            chain[i] = q
            occupied[q] = i
            
            # Calculate delta contacts
            delta = 0
            if is_h[i]:
                for dx, dy, dz in _NEIGHBOR_OFFSETS:
                    r = (q[0] + dx, q[1] + dy, q[2] + dz)
                    if r in occupied:
                        idx = occupied[r]
                        if idx != i - 1:
                            if is_h[idx]:
                                delta += 1
            
            current_contacts += delta
            
            # Branch and Bound
            if best_chain_holder[1] is not None:
                best_contacts = -best_chain_holder[1]
                # Max possible contacts from remaining H residues
                # Each H can form at most 6 contacts, but many are with already placed residues.
                # Loose bound: 6 * (number of H in suffix)
                remaining_h = sum(is_h[i+1:])
                max_future = 6 * remaining_h
                if current_contacts + max_future <= best_contacts:
                    # Prune
                    current_contacts -= delta
                    del occupied[q]
                    continue
            
            dfs(i + 1)
            
            # Backtrack
            current_contacts -= delta
            del occupied[q]
            
            # Check time after each node to be safe
            if time.time() - start_time > max_time:
                return

    dfs(0)

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    # Global time budget
    # 26 sequences, total 300s. ~11.5s per sequence.
    # We use a safety margin of 10s per sequence.
    start_time = time.time()
    max_time = 9.0 # Safety margin
    
    rng = random.Random(42 + len(sequence) + hash(sequence) % 1000)
    
    is_h = [1 if s == 'H' else 0 for s in sequence]
    
    best_chain = None
    best_energy = 0 # Energy is negative. 0 is worse than -1. So we want to minimize energy.
    
    # 1. Get a valid baseline using Greedy
    # This ensures we always have a valid result to return.
    init_chain = _grow_greedy(sequence, dimension, rng)
    
    if init_chain is not None:
        best_chain = init_chain.copy()
        best_energy = _energy(sequence, init_chain)
    else:
        # Fallback: Straight line (always valid)
        best_chain = [(i, 0, 0) for i in range(n)]
        best_energy = 0 # No contacts in a straight line
    
    # 2. Try to improve with CSP Search
    # Only if we have time left
    if time.time() - start_time < max_time * 0.5:
        # CSP holder: [chain, energy]
        csp_best = [best_chain, best_energy]
        
        _csp_search(sequence, n, start_time, max_time, csp_best)
        
        if csp_best[0] is not None and csp_best[1] < best_energy:
            best_chain = csp_best[0]
            best_energy = csp_best[1]
            
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]
