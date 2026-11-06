"""HP lattice protein folding with multi-strategy search and intensive local search."""
# EVOLVE-BLOCK-START
import random
import time

_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    _init_time()
    return 195.0 - (time.time() - _START_TIME)

def _neighbours(point):
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

def _count_contacts(sequence, chain):
    n = len(chain)
    if n == 0:
        return 0
    occupied = {p: i for i, p in enumerate(chain)}
    contacts = 0
    is_h = [c == 'H' for c in sequence]
    for i in range(n):
        if not is_h[i]:
            continue
        x, y = chain[i]
        for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
            j = occupied.get((nx, ny))
            if j is not None and j > i + 1 and is_h[j]:
                contacts += 1
    return contacts

def _energy(sequence, chain):
    return -_count_contacts(sequence, chain)

def _grow_greedy(sequence, rng, bias=0.7):
    """Greedy growth with randomization."""
    n = len(sequence)
    if n == 0:
        return None
    is_h = [c == 'H' for c in sequence]
    
    # Random start
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    for step in range(1, n):
        current = chain[-1]
        candidates = []
        for site in _neighbours(current):
            if site in occupied:
                continue
            gain = 0
            if is_h[step]:
                for nb in _neighbours(site):
                    j = occupied.get(nb)
                    if j is not None and j < step - 1 and is_h[j]:
                        gain += 1
            free = sum(1 for nb in _neighbours(site) if nb not in occupied)
            score = gain * 10.0 + free * 0.1 + rng.random() * bias
            candidates.append((score, site))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        # Pick from top few with probability
        top_k = min(3, len(candidates))
        if rng.random() < 0.8:
            chosen = candidates[0][1]
        else:
            chosen = candidates[rng.randint(0, top_k - 1)][1]
        
        chain.append(chosen)
        occupied[chosen] = step
    
    return chain

def _beam_search(sequence, rng, beam_width=40):
    """Beam search with diversity."""
    n = len(sequence)
    if n == 0:
        return None
    is_h = [c == 'H' for c in sequence]
    
    beams = [([(0, 0)], {(0, 0): 0}, 0)]
    
    for step in range(1, n):
        if _time_remaining() < 3.0:
            return None
        new_beams = []
        for chain, occupied, contacts in beams:
            current = chain[-1]
            for site in _neighbours(current):
                if site in occupied:
                    continue
                gain = 0
                if is_h[step]:
                    for nb in _neighbours(site):
                        j = occupied.get(nb)
                        if j is not None and j < step - 1 and is_h[j]:
                            gain += 1
                free = sum(1 for nb in _neighbours(site) if nb not in occupied)
                score = gain * 100.0 + free * 1.0 + rng.random() * 0.5
                new_chain = chain + [site]
                new_occ = dict(occupied)
                new_occ[site] = step
                new_beams.append((score, new_chain, new_occ, contacts + gain))
        
        if not new_beams:
            return None
        new_beams.sort(key=lambda x: x[0], reverse=True)
        seen = set()
        diverse = []
        for score, chain, occ, contacts in new_beams:
            key = tuple(chain[-5:])
            if key not in seen:
                seen.add(key)
                diverse.append((chain, occ, contacts))
            if len(diverse) >= beam_width:
                break
        beams = diverse
    
    if not beams:
        return None
    best = max(beams, key=lambda x: x[2])
    return best[0]

def _local_search(sequence, chain, rng, max_steps=500):
    """Intensive local search with kink and segment moves."""
    n = len(chain)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    is_h = [c == 'H' for c in sequence]
    
    start_temp = 2.0
    end_temp = 0.05
    
    for step in range(max_steps):
        if _time_remaining() < 2.0:
            break
        
        progress = step / max(1, max_steps)
        temp = start_temp * (end_temp / start_temp) ** progress
        
        move_type = rng.random()
        
        if move_type < 0.55:
            # Kink move
            if n < 3:
                continue
            i = rng.randint(1, n - 2)
            prev = chain[i-1]
            curr = chain[i]
            next_ = chain[i+1]
            
            candidates = set()
            for p in _neighbours(prev):
                if p == curr:
                    continue
                if abs(p[0] - next_[0]) + abs(p[1] - next_[1]) == 1:
                    candidates.add(p)
            
            if not candidates:
                continue
            
            new_pos = rng.choice(list(candidates))
            if new_pos in occupied:
                continue
            
            del occupied[curr]
            occupied[new_pos] = i
            chain[i] = new_pos
            
            new_energy = _energy(sequence, chain)
            delta = new_energy - current_energy
            
            if delta < 0 or (temp > 0.01 and rng.random() < pow(2.718281828, -delta / temp)):
                current_energy = new_energy
            else:
                chain[i] = curr
                del occupied[new_pos]
                occupied[curr] = i
                
        elif move_type < 0.80:
            # Segment move length 2-3
            seg_len = rng.choice([2, 3])
            if n < seg_len + 2:
                continue
            start_idx = rng.randint(1, n - seg_len - 1)
            end_idx = start_idx + seg_len - 1
            seg_indices = list(range(start_idx, end_idx + 1))
            old_seg = [chain[i] for i in seg_indices]
            
            anchor_start = chain[start_idx - 1]
            anchor_end = chain[end_idx + 1]
            
            for _ in range(15):
                pos = anchor_start
                path = [pos]
                for _ in range(seg_len):
                    pos = rng.choice(_neighbours(pos))
                    path.append(pos)
                
                if path[-1] != anchor_end:
                    continue
                
                new_seg = path[1:]
                
                collision = False
                for p in new_seg:
                    if p in occupied and p not in old_seg:
                        collision = True
                        break
                
                if collision or len(set(new_seg)) != seg_len:
                    continue
                
                for i in seg_indices:
                    del occupied[chain[i]]
                
                for k, idx in enumerate(seg_indices):
                    occupied[new_seg[k]] = idx
                    chain[idx] = new_seg[k]
                
                new_energy = _energy(sequence, chain)
                delta = new_energy - current_energy
                
                if delta < 0 or (temp > 0.01 and rng.random() < pow(2.718281828, -delta / temp)):
                    current_energy = new_energy
                else:
                    for k, idx in enumerate(seg_indices):
                        del occupied[chain[idx]]
                        occupied[old_seg[k]] = idx
                        chain[idx] = old_seg[k]
                
                break
                
        else:
            # Segment move length 4-6
            seg_len = rng.randint(4, min(6, n - 2))
            if n < seg_len + 2:
                continue
            start_idx = rng.randint(1, n - seg_len - 1)
            end_idx = start_idx + seg_len - 1
            seg_indices = list(range(start_idx, end_idx + 1))
            old_seg = [chain[i] for i in seg_indices]
            
            anchor_start = chain[start_idx - 1]
            anchor_end = chain[end_idx + 1]
            
            for _ in range(10):
                pos = anchor_start
                path = [pos]
                for _ in range(seg_len):
                    pos = rng.choice(_neighbours(pos))
                    path.append(pos)
                
                if path[-1] != anchor_end:
                    continue
                
                new_seg = path[1:]
                
                collision = False
                for p in new_seg:
                    if p in occupied and p not in old_seg:
                        collision = True
                        break
                
                if collision or len(set(new_seg)) != seg_len:
                    continue
                
                for i in seg_indices:
                    del occupied[chain[i]]
                
                for k, idx in enumerate(seg_indices):
                    occupied[new_seg[k]] = idx
                    chain[idx] = new_seg[k]
                
                new_energy = _energy(sequence, chain)
                delta = new_energy - current_energy
                
                if delta < 0 or (temp > 0.01 and rng.random() < pow(2.718281828, -delta / temp)):
                    current_energy = new_energy
                else:
                    for k, idx in enumerate(seg_indices):
                        del occupied[chain[idx]]
                        occupied[old_seg[k]] = idx
                        chain[idx] = old_seg[k]
                
                break
    
    return chain, current_energy

def _hill_climb(sequence, chain, rng, max_steps=200):
    """Fast hill climbing for quick refinement."""
    n = len(chain)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    
    for step in range(max_steps):
        if _time_remaining() < 1.0:
            break
        
        improved = False
        for i in range(1, n - 1):
            if _time_remaining() < 1.0:
                break
            prev = chain[i-1]
            curr = chain[i]
            next_ = chain[i+1]
            
            for new_pos in _neighbours(prev):
                if new_pos == curr:
                    continue
                if abs(new_pos[0] - next_[0]) + abs(new_pos[1] - next_[1]) != 1:
                    continue
                if new_pos in occupied:
                    continue
                
                del occupied[curr]
                occupied[new_pos] = i
                chain[i] = new_pos
                
                new_energy = _energy(sequence, chain)
                if new_energy < current_energy:
                    current_energy = new_energy
                    improved = True
                else:
                    chain[i] = curr
                    del occupied[new_pos]
                    occupied[curr] = i
        
        if not improved:
            break
    
    return chain, current_energy

def fold(sequence, dimension):
    """
    Fold an HP sequence on the square lattice using multi-strategy search.
    """
    _init_time()
    
    n = len(sequence)
    if n == 0:
        return []
    
    if _time_remaining() < 1.0:
        return [[i, 0] for i in range(n)]
    
    rng = random.Random(42 + len(sequence))
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Generate diverse initial conformations
    initial_chains = []
    
    # Greedy with different biases
    for bias in [0.3, 0.7, 1.5]:
        chain = _grow_greedy(sequence, rng, bias=bias)
        if chain is not None:
            initial_chains.append(chain)
    
    # Beam search
    for bw in [30, 50]:
        chain = _beam_search(sequence, rng, beam_width=bw)
        if chain is not None:
            initial_chains.append(chain)
    
    # Quick hill climb on all initial chains
    refined_chains = []
    for chain in initial_chains:
        if _time_remaining() < 5.0:
            break
        chain, energy = _hill_climb(sequence, chain, rng, max_steps=100)
        refined_chains.append((chain, energy))
    
    # Phase 2: Intensive local search on best candidates
    refined_chains.sort(key=lambda x: x[1])
    
    for chain, energy in refined_chains[:5]:
        if _time_remaining() < 10.0:
            break
        chain, energy = _local_search(sequence, chain, rng, max_steps=400)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    # Phase 3: More restarts if time permits
    restart_count = 0
    while _time_remaining() > 15.0 and restart_count < 20:
        restart_count += 1
        chain = _grow_greedy(sequence, rng, bias=rng.uniform(0.2, 2.0))
        if chain is None:
            continue
        chain, energy = _hill_climb(sequence, chain, rng, max_steps=80)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
            # Further refine
            if _time_remaining() > 20.0:
                chain, energy = _local_search(sequence, chain, rng, max_steps=200)
                if energy < best_energy:
                    best_energy = energy
                    best_chain = chain
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
