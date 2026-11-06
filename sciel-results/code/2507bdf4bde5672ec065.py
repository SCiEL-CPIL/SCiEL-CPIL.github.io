"""
HP lattice protein folding using Beam Search over partial self-avoiding walks.
This strategy maintains a population of the K lowest-energy partial conformations
at each step, pruning inferior paths to manage the combinatorial explosion.
Enhanced with upper-bound pruning and local search polish.
"""
# EVOLVE-BLOCK-START
import time
import random
import heapq

_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    _init_time()
    # Safety margin: 210s limit, leave 15s buffer -> 195s usable
    return 195.0 - (time.time() - _START_TIME)

def _neighbours(point):
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _compute_energy(sequence, chain):
    """Compute the energy of a complete conformation."""
    n = len(chain)
    is_h = [c == 'H' for c in sequence]
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    energy = 0
    for i, p in enumerate(chain):
        if is_h[i]:
            for q in _neighbours(p):
                j = pos_to_idx.get(q)
                if j is not None and j > i + 1 and is_h[j]:
                    energy -= 1
    return energy

def _is_valid_conformation(chain):
    """Check if the chain is self-avoiding and has unit steps."""
    n = len(chain)
    if n == 0:
        return True
    if len(set(chain)) != n:
        return False
    for i in range(1, n):
        dx = abs(chain[i][0] - chain[i-1][0])
        dy = abs(chain[i][1] - chain[i-1][1])
        if dx + dy != 1:
            return False
    return True

def _local_search_polish(sequence, chain, max_iters=2000, time_limit=5.0, rng=None):
    """
    Local search polish: try single-residue moves and small segment reversals
    to improve energy. Returns the best chain found.
    """
    n = len(chain)
    is_h = [c == 'H' for c in sequence]
    best_chain = list(chain)
    best_energy = _compute_energy(sequence, best_chain)
    
    start_time = time.time()
    rng = rng or random.Random()
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Randomly choose move type: single residue (70%), segment reversal (30%)
        if rng.random() < 0.7:
            # Single residue move
            i = rng.randint(1, n - 2)  # Avoid endpoints
            if i == 0 or i == n - 1:
                continue
            old_pos = best_chain[i]
            prev_pos = best_chain[i - 1]
            next_pos = best_chain[i + 1]
            
            # Try each neighbor of old_pos
            candidates = []
            for nb in _neighbours(old_pos):
                # Must not be the same as current neighbors (would be a backtrack or stay)
                if nb == prev_pos or nb == next_pos:
                    continue
                
                # Check if nb is occupied by another residue (excluding i)
                # We can optimize this by maintaining a set, but for n<=100, set creation is okay
                # However, doing this every iteration is O(N^2). Let's just check the specific indices
                # Actually, a simple set of all positions except i is safer.
                occupied_set = set(best_chain)
                occupied_set.discard(old_pos)
                if nb in occupied_set:
                    continue
                
                # Check unit steps with new neighbors
                # prev_pos to nb
                if abs(nb[0] - prev_pos[0]) + abs(nb[1] - prev_pos[1]) != 1:
                    continue
                # nb to next_pos
                if abs(next_pos[0] - nb[0]) + abs(next_pos[1] - nb[1]) != 1:
                    continue
                
                candidates.append(nb)
            
            if not candidates:
                continue
            
            new_pos = rng.choice(candidates)
            new_chain = best_chain.copy()
            new_chain[i] = new_pos
            
            # Verify self-avoidance and unit steps (double safety)
            if not _is_valid_conformation(new_chain):
                continue
            
            # Compute new energy
            new_energy = _compute_energy(sequence, new_chain)
            
            if new_energy <= best_energy:
                best_chain = new_chain
                best_energy = new_energy
        else:
            # Segment reversal (flip a short segment)
            seg_len = rng.randint(2, min(5, n - 2))
            i = rng.randint(1, n - 1 - seg_len)
            # Reverse segment from i to i+seg_len-1
            new_chain = best_chain.copy()
            new_chain[i:i+seg_len] = list(reversed(new_chain[i:i+seg_len]))
            
            # Verify self-avoidance and unit steps
            if not _is_valid_conformation(new_chain):
                continue
            
            new_energy = _compute_energy(sequence, new_chain)
            
            if new_energy <= best_energy:
                best_chain = new_chain
                best_energy = new_energy
    
    return best_chain

def _fold_beam_search(sequence, dimension, beam_width=50, time_limit=10.0, rng=None, upper_bound=None):
    """
    Perform Beam Search for the given sequence.
    Returns the best full conformation found.
    upper_bound: if provided, prune states that can't beat this energy.
    """
    n = len(sequence)
    is_h = [c == 'H' for c in sequence]
    
    # Precompute: for each position i, the max additional H-H contacts possible
    # from positions i..n-1 (upper bound on further energy decrease)
    # A simple bound: at most (n-1-i) * (number of H's remaining) / 2, but that's loose.
    # Better: for each H at position j > i, it can contact at most 4 other H's,
    # but we need a tighter bound. Let's use: remaining H count * 4 / 2 = 2 * remaining_H
    # Actually each H can form at most 4 contacts, so max additional energy decrease
    # from positions i..n-1 is bounded by sum over j=i..n-1 of (4 if is_h[j] else 0) / 2
    # But that's still loose. Let's just use 2 * (count of H in positions i..n-1)
    # as a rough upper bound on additional negative energy.
    
    remaining_h = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        remaining_h[i] = remaining_h[i + 1] + (1 if is_h[i] else 0)
    
    # Initial state
    start_pos = (0, 0)
    initial_chain = [start_pos]
    initial_occupied = {start_pos}
    initial_h = {start_pos} if is_h[0] else set()
    initial_energy = 0
    
    beam = [(initial_energy, initial_chain, initial_occupied, initial_h)]
    
    start_time = time.time()
    
    for i in range(1, n):
        new_beam = []
        is_current_h = is_h[i]
        
        for state in beam:
            energy, chain, occupied, h_pos = state
            last_pos = chain[-1]
            pred_pos = chain[-2] if len(chain) > 1 else None
            
            # Upper bound check: can this state possibly beat upper_bound?
            if upper_bound is not None:
                # Max additional energy decrease from positions i..n-1
                # Each remaining H can form at most 4 contacts, but each contact involves 2 H's
                # So max additional negative energy ~ 2 * remaining_h[i]
                # But we're at position i, so remaining positions are i..n-1
                max_additional = 2 * remaining_h[i]
                if energy + max_additional > upper_bound:
                    continue  # Can't beat upper_bound, skip this state
            
            for nb in _neighbours(last_pos):
                if nb in occupied:
                    continue
                
                new_energy = energy
                new_h_pos = h_pos
                new_occupied = occupied | {nb}
                
                if is_current_h:
                    new_h_pos = h_pos | {nb}
                    for q in _neighbours(nb):
                        if q in h_pos:
                            if q != pred_pos:
                                new_energy -= 1
                
                # Upper bound check for new state
                if upper_bound is not None:
                    max_additional_new = 2 * remaining_h[i + 1]
                    if new_energy + max_additional_new > upper_bound:
                        continue  # Can't beat upper_bound
                
                new_chain = chain + [nb]
                new_beam.append((new_energy, new_chain, new_occupied, new_h_pos))
                
        if not new_beam:
            break
            
        # Prune with randomization for ties
        # Sort by energy
        new_beam.sort(key=lambda x: x[0])
        
        if len(new_beam) > beam_width:
            cutoff_energy = new_beam[beam_width - 1][0]
            
            # Separate states into 'definitely keep' (energy < cutoff) and 'maybe keep' (energy == cutoff)
            keep = []
            maybe = []
            for state in new_beam:
                if state[0] < cutoff_energy:
                    keep.append(state)
                elif state[0] == cutoff_energy:
                    maybe.append(state)
                else:
                    break # Since sorted, rest are worse
            
            # Need to fill remaining slots from 'maybe'
            slots_needed = beam_width - len(keep)
            if slots_needed > 0:
                if rng:
                    rng.shuffle(maybe)
                else:
                    random.shuffle(maybe)
                keep.extend(maybe[:slots_needed])
            
            beam = keep
        else:
            beam = new_beam
            
        if time.time() - start_time > time_limit:
            break
            
    if not beam:
        return None
        
    # Pick the best state from the final beam
    best_state = min(beam, key=lambda x: x[0])
    chain = best_state[1]
    occupied = best_state[2]
    h_pos = best_state[3]
    
    # If we broke early, extend greedily
    if len(chain) < n:
        for j in range(len(chain), n):
            is_j_h = is_h[j]
            last_pos = chain[-1]
            pred_pos = chain[-2] if len(chain) > 1 else None
            best_nb = None
            best_e = float('inf')
            
            candidates = []
            for nb in _neighbours(last_pos):
                if nb in occupied:
                    continue
                e = 0
                if is_j_h:
                    for q in _neighbours(nb):
                        if q in h_pos and q != pred_pos:
                            e -= 1
                candidates.append((e, nb))
                
            if candidates:
                candidates.sort(key=lambda x: x[0])
                # Pick best, or random among best if tie
                best_e_val = candidates[0][0]
                best_opts = [nb for e, nb in candidates if e == best_e_val]
                if rng:
                    best_nb = rng.choice(best_opts)
                else:
                    best_nb = random.choice(best_opts)
                    
                chain.append(best_nb)
                occupied.add(best_nb)
                if is_j_h:
                    h_pos.add(best_nb)
            else:
                # Dead end, return None to signal failure
                return None
                
    return chain

def fold(sequence, dimension):
    """
    Fold the sequence using Beam Search with local search polish.
    """
    _init_time()
    n = len(sequence)
    
    # Adjust beam width based on time remaining and sequence length
    # Longer sequences need more time, but also explode faster.
    # Short sequences can use wider beams.
    if n <= 20:
        base_beam_width = 300
        base_time_limit = 5.0
        polish_time = 3.0
    elif n <= 50:
        base_beam_width = 150
        base_time_limit = 10.0
        polish_time = 5.0
    else:
        base_beam_width = 80
        base_time_limit = 15.0
        polish_time = 8.0
        
    best_chain = None
    best_energy = 0 # 0 is neutral, we want negative
    
    # Strategy: Run multiple beam searches with different randomizations to explore different topologies.
    # We also adaptively increase beam width if time allows.
    # Use upper bound pruning after first successful run.
    
    num_runs = 0
    max_runs = 15
    
    for run in range(max_runs):
        if _time_remaining() < 15:
            break
            
        seed = 12345 + run * 7
        rng = random.Random(seed)
        
        current_bw = base_beam_width
        current_tl = base_time_limit
        
        # Use upper bound pruning for subsequent runs
        ub = best_energy if run > 0 and best_energy < 0 else None
        
        chain = _fold_beam_search(sequence, dimension, current_bw, current_tl, rng, upper_bound=ub)
        
        if chain is not None and len(chain) == n:
            # Validate and compute energy
            valid = True
            occupied = set()
            for i, p in enumerate(chain):
                if p in occupied:
                    valid = False
                    break
                occupied.add(p)
                if i > 0:
                    prev = chain[i-1]
                    if abs(p[0]-prev[0]) + abs(p[1]-prev[1]) != 1:
                        valid = False
                        break
            
            if valid:
                # Calculate energy
                energy = _compute_energy(sequence, chain)
                
                if best_chain is None or energy < best_energy:
                    best_chain = chain
                    best_energy = energy
                num_runs += 1
            else:
                # Invalid chain, ignore
                pass
        else:
            # Failed to complete, ignore
            pass
    
    # Store the best valid chain from beam search before polishing
    best_valid_chain = best_chain
    best_valid_energy = best_energy

    # Local search polish on the best chain found
    if best_chain is not None and _time_remaining() > 10:
        polish_rng = random.Random(99999)
        polished_chain = _local_search_polish(sequence, best_chain, 
                                           max_iters=5000, 
                                           time_limit=polish_time,
                                           rng=polish_rng)
        
        # Verify the polished chain is valid
        if _is_valid_conformation(polished_chain):
            polished_energy = _compute_energy(sequence, polished_chain)
            if polished_energy < best_valid_energy:
                best_valid_chain = polished_chain
                best_valid_energy = polished_energy
        # If invalid, we keep the best_valid_chain from beam search
    
    if best_valid_chain is None:
        # Fallback: Straight line
        best_valid_chain = [list(p) for p in [(i, 0) for i in range(n)]]
        
    return best_valid_chain
# EVOLVE-BLOCK-END
