"""HP lattice protein folding: beam search with delta-energy local search."""
# EVOLVE-BLOCK-START
import time
import random
import heapq

_T0 = time.time()
_DEADLINE = 200.0  # 210s limit, 10s safety margin


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _neighbours(p):
    x, y = p
    return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]


def _compute_energy(seq, chain):
    """Compute HP energy: -number of H-H non-bonded contacts."""
    n = len(chain)
    if n < 3:
        return 0
    pos_to_idx = {}
    for i, p in enumerate(chain):
        pos_to_idx[p] = i
    h_indices = set(i for i, c in enumerate(seq) if c == 'H')
    contacts = 0
    for i in h_indices:
        x, y = chain[i]
        for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
            j = pos_to_idx.get((x+dx, y+dy))
            if j is not None and j > i + 1 and seq[j] == 'H':
                contacts += 1
    return -contacts


def _count_contacts_fast(seq, chain):
    """Fast contact count using position lookup."""
    n = len(chain)
    if n < 3:
        return 0
    pos_to_idx = {}
    for i, p in enumerate(chain):
        pos_to_idx[p] = i
    h_indices = set(i for i, c in enumerate(seq) if c == 'H')
    contacts = 0
    for i in h_indices:
        x, y = chain[i]
        for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
            j = pos_to_idx.get((x+dx, y+dy))
            if j is not None and j > i + 1 and seq[j] == 'H':
                contacts += 1
    return contacts


def _beam_search(seq, beam_width=20, max_time_frac=0.4):
    """Beam search over partial self-avoiding walks.
    
    Returns list of (energy, chain) for top beam_width partial conformations
    that are complete (full length).
    """
    n = len(seq)
    if n <= 2:
        return [(-0, [[i, 0] for i in range(n)])]
    
    # Priority queue: (negative_score, counter, chain, occupied, current_contacts)
    # We maximize contacts, so use -contacts as key (min-heap)
    counter = 0
    start_chain = [(0, 0)]
    start_occupied = {(0, 0): 0}
    
    # Initial: place first residue at origin
    heap = []
    heapq.heappush(heap, (0, counter, start_chain, start_occupied, 0))
    counter += 1
    
    results = []
    max_steps = int(max_time_frac * _remaining())
    step = 0
    
    while _remaining() > 30.0 and step < 500000:
        if not heap:
            break
        
        neg_score, _, chain, occupied, contacts = heapq.heappop(heap)
        step += 1
        
        if len(chain) == n:
            # Complete conformation
            results.append((-contacts, list(chain)))
            if len(results) >= beam_width:
                break
            continue
        
        # Extend chain
        last = chain[-1]
        i = len(chain)  # next index
        
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            # Count new contacts if this site is H
            new_contacts = contacts
            if seq[i] == 'H':
                x, y = site
                for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
                    j = occupied.get((x+dx, y+dy))
                    if j is not None and j <= i - 2 and seq[j] == 'H':
                        new_contacts += 1
            
            new_chain = chain + [site]
            new_occupied = dict(occupied)
            new_occupied[site] = i
            
            # Score: prioritize high contact count
            heapq.heappush(heap, (-new_contacts, counter, new_chain, new_occupied, new_contacts))
            counter += 1
            
            # Limit heap size
            if len(heap) > beam_width * 10:
                heapq.heappop(heap)
    
    # Sort results by energy (most negative first)
    results.sort(key=lambda x: x[0])
    return results[:beam_width]


def _local_search_delta(seq, chain, max_steps=2000, rng=None):
    """Pivot-based local search with delta-energy computation.
    
    Returns (best_chain, best_energy).
    """
    if rng is None:
        rng = random.Random()
    
    n = len(chain)
    if n < 3:
        return list(chain), 0
    
    current_energy = _compute_energy(seq, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    occupied = {p: i for i, p in enumerate(chain)}
    
    for step in range(max_steps):
        if _remaining() < 10.0:
            break
        
        # Choose random pivot position
        i = rng.randint(1, n - 2)
        
        pivot = chain[i-1]
        seg = chain[i:]
        
        # Try all 4 rotations
        for angle in range(4):
            new_seg = []
            new_occupied = {}
            valid = True
            
            # Build new occupied set (remove old segment)
            orig_occupied = dict(occupied)
            for p in seg:
                if p in orig_occupied:
                    del orig_occupied[p]
            
            # Rotate each point
            for k, p in enumerate(seg):
                dx = p[0] - pivot[0]
                dy = p[1] - pivot[1]
                
                if angle == 0:
                    nx, ny = dx, dy
                elif angle == 1:
                    nx, ny = -dy, dx
                elif angle == 2:
                    nx, ny = -dx, -dy
                else:
                    nx, ny = dy, -dx
                
                new_p = (pivot[0] + nx, pivot[1] + ny)
                
                if new_p in new_occupied:
                    valid = False
                    break
                if new_p in orig_occupied:
                    valid = False
                    break
                
                new_seg.append(new_p)
                new_occupied[new_p] = i + k
            
            if not valid:
                continue
            
            # Build new chain
            new_chain = chain[:i] + new_seg
            
            # Verify self-avoidance
            if len(set(new_chain)) != n:
                continue
            
            # Compute delta energy
            new_energy = _compute_energy(seq, new_chain)
            delta = new_energy - current_energy
            
            # Accept if better or with small probability
            if delta < 0 or (delta == 0 and rng.random() < 0.3) or rng.random() < 0.05:
                chain = new_chain
                occupied = new_occupied
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_chain = list(chain)
                    best_energy = current_energy
                    break  # Restart pivot selection after improvement
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Fold HP sequence using beam search with local refinement."""
    n = len(sequence)
    if n <= 2:
        return [[i, 0] for i in range(n)]
    
    best_chain = None
    best_energy = 0  # Energy is negative, so 0 is "worse" than any valid fold with contacts
    
    rng = random.Random(42 + n)
    
    # Phase 1: Beam search to get good initial conformations
    beam_results = _beam_search(sequence, beam_width=10, max_time_frac=0.3)
    
    for energy, chain in beam_results:
        if energy < best_energy or best_chain is None:
            best_chain = chain
            best_energy = energy
    
    if best_chain is None:
        # Fallback: straight line
        return [[i, 0] for i in range(n)]
    
    # Phase 2: Local refinement on best solution
    if _remaining() > 30.0:
        refined_chain, refined_energy = _local_search_delta(
            sequence, best_chain, max_steps=5000, rng=rng
        )
        if refined_energy < best_energy:
            best_chain = refined_chain
            best_energy = refined_energy
    
    # Phase 3: Try local search on other beam results
    if _remaining() > 20.0:
        for energy, chain in beam_results[1:]:
            if _remaining() < 15.0:
                break
            refined_chain, refined_energy = _local_search_delta(
                sequence, chain, max_steps=2000, rng=rng
            )
            if refined_energy < best_energy:
                best_chain = refined_chain
                best_energy = refined_energy
    
    # Phase 4: More beam search restarts if time allows
    if _remaining() > 40.0:
        for restart in range(5):
            if _remaining() < 20.0:
                break
            beam_results2 = _beam_search(sequence, beam_width=5, max_time_frac=0.1)
            for energy, chain in beam_results2:
                if energy < best_energy:
                    best_chain = chain
                    best_energy = energy
                    # Refine immediately
                    if _remaining() > 15.0:
                        refined_chain, refined_energy = _local_search_delta(
                            sequence, best_chain, max_steps=3000, rng=rng
                        )
                        if refined_energy < best_energy:
                            best_chain = refined_chain
                            best_energy = refined_energy
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
