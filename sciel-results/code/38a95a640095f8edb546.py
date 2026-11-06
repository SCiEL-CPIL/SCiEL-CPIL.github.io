"""HP lattice protein folding on the cubic lattice with local search."""
# EVOLVE-BLOCK-START
import random
import time
import sys

def _neighbours(point):
    """Yield all 6 neighbors of a point on the cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Compute energy: negative number of H-H contacts (non-bonded neighbors)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _grow_random(sequence, dimension, rng):
    """Randomized greedy growth with some randomness."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            # Score: prefer sites that create H-H contacts, then free space
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == "H":
                            gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 2.0 + free * 0.1 + rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None
        
        # Weighted random selection
        total = sum(s for s, _ in options)
        if total <= 0:
            site = options[rng.randint(0, len(options) - 1)][1]
        else:
            r = rng.random() * total
            cum = 0
            site = options[-1][1]
            for s, s_site in options:
                cum += s
                if r <= cum:
                    site = s_site
                    break
        
        occupied[site] = i
        chain.append(site)
    
    return chain

def _local_search(sequence, chain, dimension, rng, max_iters=5000, deadline=None):
    """Local search using Monte Carlo moves to improve energy."""
    if chain is None:
        return None, 1
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    n = len(sequence)
    if n < 4:
        return best_chain, best_energy
    
    # Precompute residue types
    is_h = [c == 'H' for c in sequence]
    
    def _try_move(chain, i, j, new_positions):
        """Try to move segment i..j to new_positions. Returns (success, new_chain, delta_energy)."""
        # Check self-avoidance for new positions
        occupied = {}
        for k, p in enumerate(chain):
            if i <= k <= j:
                continue
            if p in occupied:
                return False, None, 0
            occupied[p] = k
        
        for k in range(i, j + 1):
            p = new_positions[k - i]
            if p in occupied:
                return False, None, 0
            occupied[p] = k
        
        # Compute energy change
        old_occupied = {p: k for k, p in enumerate(chain)}
        
        # Remove old contacts involving moved segment
        old_contacts = 0
        for k in range(i, j + 1):
            p = chain[k]
            if is_h[k]:
                for q in _neighbours(p):
                    if q in old_occupied:
                        l = old_occupied[q]
                        if l > k + 1 and is_h[l]:
                            old_contacts += 1
        
        # Add new contacts
        new_contacts = 0
        for k in range(i, j + 1):
            p = new_positions[k - i]
            if is_h[k]:
                for q in _neighbours(p):
                    if q in occupied:
                        l = occupied[q]
                        if l > k + 1 and is_h[l]:
                            new_contacts += 1
        
        delta = -(new_contacts - old_contacts)
        
        new_chain = list(chain)
        for k in range(i, j + 1):
            new_chain[k] = new_positions[k - i]
        
        return True, new_chain, delta
    
    def _pivot_move(chain, i):
        """Rotate the chain at position i. Returns list of possible new chains."""
        if i <= 0 or i >= len(chain) - 1:
            return []
        
        # Get the segment from i to end
        segment = chain[i:]
        n_seg = len(segment)
        
        # Try different rotations by moving the tail
        results = []
        
        # Simple move: try to move the last few residues
        for tail_len in [2, 3, 4]:
            if i + tail_len > len(chain):
                continue
            tail_start = len(chain) - tail_len
            if tail_start <= i:
                continue
            
            # Get current tail positions
            tail = chain[tail_start:]
            prefix_end = chain[tail_start - 1]
            
            # Try to place tail at different positions
            for axis in range(dimension):
                for step in (-1, 1):
                    # Try shifting the entire tail
                    new_tail = []
                    valid = True
                    for k, p in enumerate(tail):
                        new_p = list(p)
                        new_p[axis] += step
                        new_p = tuple(new_p)
                        new_tail.append(new_p)
                    
                    # Check if valid (no collision with prefix)
                    prefix_set = set(chain[:tail_start])
                    for p in new_tail:
                        if p in prefix_set:
                            valid = False
                            break
                        if k > 0 and abs(new_p[0] - new_tail[k-1][0]) + abs(new_p[1] - new_tail[k-1][1]) + abs(new_p[2] - new_tail[k-1][2]) != 1:
                            valid = False
                            break
                    
                    if valid:
                        new_chain = chain[:tail_start] + new_tail
                        results.append(new_chain)
        
        return results
    
    def _crank_move(chain, i):
        """Crank move: rotate a 3-residue segment."""
        if i <= 0 or i >= len(chain) - 2:
            return []
        
        # Get three consecutive residues
        p0 = chain[i]
        p1 = chain[i + 1]
        p2 = chain[i + 2]
        
        # The vector from p0 to p1
        v = [p1[a] - p0[a] for a in range(dimension)]
        
        # Try rotating p2 around the p0-p1 axis
        results = []
        
        # Find all possible positions for new p2 that are:
        # 1. Adjacent to p1
        # 2. Not adjacent to p0 (to maintain unit step)
        # 3. Not occupied by other chain parts
        
        occupied = {p: k for k, p in enumerate(chain)}
        # Remove the segment we're moving
        for k in range(i, i + 3):
            if chain[k] in occupied:
                del occupied[chain[k]]
        
        for new_p2 in _neighbours(p1):
            # Check it's not where p2 was
            if new_p2 == p2:
                continue
            # Check it's not adjacent to p0 (would create a bond)
            if new_p2 in occupied:
                continue
            # Check unit step from p1
            if sum(abs(new_p2[a] - p1[a]) for a in range(dimension)) != 1:
                continue
            # Check it doesn't create a bond with p0
            if sum(abs(new_p2[a] - p0[a]) for a in range(dimension)) == 1:
                continue
            
            new_chain = list(chain)
            new_chain[i + 2] = new_p2
            results.append(new_chain)
        
        return results
    
    temperature = 1.0
    cooling = 0.999
    min_temp = 0.01
    
    iter_count = 0
    while iter_count < max_iters:
        if deadline and time.time() > deadline:
            break
        
        # Choose a random move type
        move_type = rng.random()
        
        if move_type < 0.4:
            # Crank move
            i = rng.randint(1, n - 3)
            candidates = _crank_move(current_chain, i)
            if not candidates:
                iter_count += 1
                continue
            new_chain = candidates[rng.randint(0, len(candidates) - 1)]
            new_energy = _energy(sequence, new_chain)
            delta = new_energy - current_energy
            
            if delta < 0 or rng.random() < min(1.0, exp(-delta / temperature)):
                current_chain = new_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
        elif move_type < 0.7:
            # Pivot-like move: move a short tail
            tail_len = rng.choice([2, 3, 4])
            if n - tail_len <= 0:
                iter_count += 1
                continue
            tail_start = n - tail_len
            candidates = _pivot_move(current_chain, tail_start)
            if not candidates:
                iter_count += 1
                continue
            new_chain = candidates[rng.randint(0, len(candidates) - 1)]
            new_energy = _energy(sequence, new_chain)
            delta = new_energy - current_energy
            
            if delta < 0 or rng.random() < min(1.0, exp(-delta / temperature)):
                current_chain = new_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
        else:
            # Random restart of a small segment
            seg_len = rng.choice([3, 4, 5])
            if n - seg_len <= 0:
                iter_count += 1
                continue
            start_idx = rng.randint(1, n - seg_len)
            end_idx = start_idx + seg_len - 1
            
            # Try to regrow this segment
            prefix = current_chain[:start_idx]
            suffix = current_chain[end_idx + 1:]
            
            # Regrow from prefix
            occupied = {p: k for k, p in enumerate(prefix)}
            for k, p in enumerate(suffix):
                occupied[p] = start_idx + k + seg_len - 1  # approximate
            
            new_segment = []
            last = prefix[-1]
            for k in range(seg_len):
                options = []
                for site in _neighbours(last):
                    if site in occupied:
                        continue
                    options.append(site)
                
                if not options:
                    break
                
                # Prefer sites with H-H contacts
                if is_h[start_idx + k]:
                    scored = []
                    for site in options:
                        gain = 0
                        for q in _neighbours(site):
                            if q in occupied:
                                j = occupied[q]
                                if j < start_idx and is_h[j]:
                                    gain += 1
                        scored.append((gain + rng.random() * 0.5, site))
                    scored.sort(key=lambda x: -x[0])
                    last = scored[0][1]
                else:
                    last = options[rng.randint(0, len(options) - 1)]
                
                new_segment.append(last)
            
            if len(new_segment) == seg_len:
                new_chain = prefix + new_segment + suffix
                # Verify self-avoidance
                if len(set(new_chain)) == len(new_chain):
                    new_energy = _energy(sequence, new_chain)
                    delta = new_energy - current_energy
                    
                    if delta < 0 or rng.random() < min(1.0, exp(-delta / temperature)):
                        current_chain = new_chain
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(current_chain)
        
        iter_count += 1
        temperature = max(min_temp, temperature * cooling)
    
    return best_chain, best_energy

def exp(x):
    """Safe exponential."""
    if x > 50:
        return 1.0
    if x < -50:
        return 0.0
    import math
    return math.exp(x)

def fold(sequence, dimension):
    """Find best conformation using multiple strategies."""
    rng = random.Random(12345 + len(sequence))
    
    start_time = time.time()
    deadline = start_time + 250  # Leave margin before 300s limit
    
    n = len(sequence)
    
    # Phase 1: Generate diverse initial conformations
    initial_chains = []
    for i in range(30):
        if time.time() > deadline - 100:
            break
        chain = _grow_random(sequence, dimension, rng)
        if chain is not None:
            initial_chains.append(chain)
    
    if not initial_chains:
        # Fallback: straight line
        return [list([i] + [0] * (dimension - 1)) for i in range(n)]
    
    # Phase 2: Local search on each initial conformation
    best_chain = None
    best_energy = 1
    
    for i, chain in enumerate(initial_chains):
        if time.time() > deadline - 50:
            break
        
        # Allocate more time to better initial conformations
        energy = _energy(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
        
        # Do local search
        remaining_time = deadline - 60 - time.time()
        if remaining_time < 1:
            break
        
        max_iters = int(remaining_time * 500)  # ~500 iterations per second
        max_iters = min(max_iters, 10000)
        
        searched_chain, searched_energy = _local_search(
            sequence, chain, dimension, rng, 
            max_iters=max_iters, 
            deadline=deadline - 60
        )
        
        if searched_energy < best_energy:
            best_energy = searched_energy
            best_chain = searched_chain
    
    # Phase 3: If time remains, do more search on the best
    if best_chain is not None and time.time() < deadline - 10:
        remaining_time = deadline - 10 - time.time()
        if remaining_time > 5:
            max_iters = int(remaining_time * 1000)
            searched_chain, searched_energy = _local_search(
                sequence, best_chain, dimension, rng,
                max_iters=max_iters,
                deadline=deadline - 5
            )
            if searched_energy < best_energy:
                best_energy = searched_energy
                best_chain = searched_chain
    
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
