"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import sys

# Explicit deadline: evaluator kills after 210s wall-clock; keep 30s safety margin.
_DEADLINE = time.time() + 180.0


def _compute_energy(sequence, chain):
    """Compute the HP contact energy: -1 for each H-H non-bonded contact."""
    n = len(chain)
    if n < 3:
        return 0
    pos_to_idx = {}
    for i, p in enumerate(chain):
        pos_to_idx[p] = i

    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y = chain[i]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j_pos = (x + dx, y + dy)
            j = pos_to_idx.get(j_pos)
            if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts // 2  # Each contact counted twice


def _energy_delta_kink(sequence, chain, i):
    """
    Compute the energy change if residue i is moved to a valid new position
    that maintains connectivity. Returns (delta, new_pos) or (None, None).
    This is for a single kink move.
    """
    n = len(chain)
    if i < 1 or i > n - 2:
        return None, None
    
    old_pos = chain[i]
    prev_pos = chain[i-1]
    next_pos = chain[i+1]
    
    # Find valid new positions
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        new_pos = (old_pos[0] + dx, old_pos[1] + dy)
        if new_pos == prev_pos or new_pos == next_pos:
            continue
        
        # Check if new_pos is adjacent to prev and next
        if abs(new_pos[0] - prev_pos[0]) + abs(new_pos[1] - prev_pos[1]) != 1:
            continue
        if abs(new_pos[0] - next_pos[0]) + abs(new_pos[1] - next_pos[1]) != 1:
            continue
        
        # Check if new_pos is already occupied (excluding i itself)
        # We need to check if any other residue is at new_pos
        # This requires a position map, so this function is better used
        # in a context where we have the map.
        pass
    
    return None, None


def _fast_grow(sequence, rng, temperature=0.1):
    """
    Greedy growth with temperature-based noise.
    Returns a valid self-avoiding chain or None.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        candidates = []
        last = chain[-1]
        for site in ((last[0] + 1, last[1]), (last[0] - 1, last[1]), 
                     (last[0], last[1] + 1), (last[0], last[1] - 1)):
            if site not in occupied:
                # Calculate immediate gain (new H-H contacts)
                gain = 0
                if sequence[i] == 'H':
                    for nb in ((site[0] + 1, site[1]), (site[0] - 1, site[1]), 
                               (site[0], site[1] + 1), (site[0], site[1] - 1)):
                        idx = occupied.get(nb)
                        if idx is not None and idx < i - 1 and sequence[idx] == 'H':
                            gain += 1
                # Calculate free space
                free = sum(1 for nb in ((site[0] + 1, site[1]), (site[0] - 1, site[1]), 
                                        (site[0], site[1] + 1), (site[0], site[1] - 1)) 
                          if nb not in occupied)
                
                score = gain * 10.0 + free * 0.5 + rng.random() * temperature
                candidates.append((score, site))
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: -x[0])
        
        # With some probability, pick a random top-3 to allow exploration
        if len(candidates) > 1 and rng.random() < 0.15:
            k = min(3, len(candidates))
            choice_idx = rng.randint(0, k - 1)
        else:
            choice_idx = 0
            
        best_site = candidates[choice_idx][1]
        chain.append(best_site)
        occupied[best_site] = i
        
    return chain


def _local_improve_fast(sequence, chain, rng, time_budget=2.0):
    """
    Fast local improvement using kink moves with incremental energy calculation.
    """
    if not chain or len(chain) < 3:
        return chain
    
    n = len(chain)
    current_energy = _compute_energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    start_time = time.time()
    
    # Build position map
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    
    # Precompute H positions
    h_positions = [i for i in range(n) if sequence[i] == 'H']
    
    t = 1.0
    t_min = 0.05
    cooling = 0.9995
    
    while time.time() - start_time < time_budget:
        t = max(t * cooling, t_min)
        
        # Pick a random interior residue
        i = rng.randint(1, n - 2)
        old_pos = chain[i]
        prev_pos = chain[i-1]
        next_pos = chain[i+1]
        
        # Try each of the 4 neighbours
        moved = False
        for new_pos in ((old_pos[0] + 1, old_pos[1]), (old_pos[0] - 1, old_pos[1]), 
                        (old_pos[0], old_pos[1] + 1), (old_pos[0], old_pos[1] - 1)):
            if new_pos == prev_pos or new_pos == next_pos:
                continue
            
            # Check connectivity
            if abs(new_pos[0] - prev_pos[0]) + abs(new_pos[1] - prev_pos[1]) != 1:
                continue
            if abs(new_pos[0] - next_pos[0]) + abs(new_pos[1] - next_pos[1]) != 1:
                continue
            
            # Check if new_pos is occupied by another residue
            if new_pos in pos_to_idx and pos_to_idx[new_pos] != i:
                continue
            
            # Compute energy delta
            delta = 0
            if sequence[i] == 'H':
                # Remove old contacts
                for nb in ((old_pos[0] + 1, old_pos[1]), (old_pos[0] - 1, old_pos[1]), 
                           (old_pos[0], old_pos[1] + 1), (old_pos[0], old_pos[1] - 1)):
                    j = pos_to_idx.get(nb)
                    if j is not None and j != i and abs(i - j) > 1 and sequence[j] == 'H':
                        delta -= 1
                # Add new contacts
                for nb in ((new_pos[0] + 1, new_pos[1]), (new_pos[0] - 1, new_pos[1]), 
                           (new_pos[0], new_pos[1] + 1), (new_pos[0], new_pos[1] - 1)):
                    j = pos_to_idx.get(nb)
                    if j is not None and j != i and abs(i - j) > 1 and sequence[j] == 'H':
                        delta += 1
            
            # Accept if better or with probability exp(-delta/t)
            if delta <= 0 or rng.random() < pow(2.718281828, -delta / t):
                # Update
                if new_pos in pos_to_idx and pos_to_idx[new_pos] != i:
                    pass  # Already checked
                del pos_to_idx[old_pos]
                pos_to_idx[new_pos] = i
                chain[i] = new_pos
                current_energy += delta
                
                if current_energy < best_energy:
                    best_chain = chain[:]
                    best_energy = current_energy
                moved = True
                break
        
        if not moved:
            # Try a different residue or break if stuck
            pass
    
    return best_chain


def _regrow_sa_fast(sequence, chain, rng, time_budget=3.0):
    """
    End-to-Tail regrowth with simulated annealing.
    """
    if not chain or len(chain) < 5:
        return chain
    
    n = len(chain)
    current_energy = _compute_energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    start_time = time.time()
    
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    
    t = 1.0
    t_min = 0.1
    cooling = 0.999
    
    while time.time() - start_time < time_budget:
        t = max(t * cooling, t_min)
        
        # Pick a random cut point in the second half
        i = rng.randint(n // 2, n - 2)
        if i >= n:
            continue
        
        anchor = chain[i-1]
        tail = chain[i:]
        
        # Remove tail from occupied
        for pos in tail:
            del pos_to_idx[pos]
        
        new_occupied = set(pos_to_idx.keys())
        new_occupied.add(anchor)
        
        # Regrow tail
        new_tail = [anchor]
        success = True
        
        for k in range(i, n):
            candidates = []
            last = new_tail[-1]
            for site in ((last[0] + 1, last[1]), (last[0] - 1, last[1]), 
                         (last[0], last[1] + 1), (last[0], last[1] - 1)):
                if site in new_occupied:
                    continue
                gain = 0
                if sequence[k] == 'H':
                    for nb in ((site[0] + 1, site[1]), (site[0] - 1, site[1]), 
                               (site[0], site[1] + 1), (site[0], site[1] - 1)):
                        idx = pos_to_idx.get(nb)
                        if idx is not None and idx < k - 1 and sequence[idx] == 'H':
                            gain += 1
                free = sum(1 for nb in ((site[0] + 1, site[1]), (site[0] - 1, site[1]), 
                                        (site[0], site[1] + 1), (site[0], site[1] - 1)) 
                          if nb not in new_occupied)
                
                score = gain * 10.0 + free * 0.5 + rng.random() * t
                candidates.append((score, site))
            
            if not candidates:
                success = False
                break
            
            candidates.sort(key=lambda x: -x[0])
            if len(candidates) > 1 and rng.random() < 0.2:
                k_choice = min(3, len(candidates))
                choice_idx = rng.randint(0, k_choice - 1)
            else:
                choice_idx = 0
                
            best_site = candidates[choice_idx][1]
            new_tail.append(best_site)
            new_occupied.add(best_site)
        
        if success:
            new_chain = chain[:i] + new_tail[1:]
            new_energy = _compute_energy(sequence, new_chain)
            delta = new_energy - current_energy
            
            if delta <= 0 or rng.random() < pow(2.718281828, -delta / t):
                chain = new_chain
                current_energy = new_energy
                pos_to_idx = {p: idx for idx, p in enumerate(chain)}
                
                if current_energy < best_energy:
                    best_chain = chain[:]
                    best_energy = current_energy
            else:
                # Restore original tail
                for idx, pos in enumerate(tail):
                    pos_to_idx[pos] = i + idx
        else:
            for idx, pos in enumerate(tail):
                pos_to_idx[pos] = i + idx
    
    return best_chain


def _min_conflicts_fold(sequence, dimension, rng, time_budget=5.0):
    """
    Min-conflicts style local search: place chain randomly, then fix violations.
    This is a different approach: start with a random self-avoiding walk,
    then iteratively move residues that are involved in conflicts or low-value contacts.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    # Generate a random self-avoiding walk
    chain = _fast_grow(sequence, rng, temperature=0.5)
    if chain is None:
        return None
    
    current_energy = _compute_energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    start_time = time.time()
    
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    
    # Identify residues with low "quality" (H residues with few contacts, or residues in tight spaces)
    # For simplicity, we'll just do random local moves but biased towards H residues
    
    t = 1.0
    cooling = 0.9995
    
    while time.time() - start_time < time_budget:
        t = max(t * 0.999, 0.05)
        
        # Pick a residue, biased towards H
        if rng.random() < 0.7:
            # Pick an H residue
            h_indices = [i for i in range(n) if sequence[i] == 'H' and 0 < i < n - 1]
            if not h_indices:
                i = rng.randint(1, n - 2)
            else:
                i = rng.choice(h_indices)
        else:
            i = rng.randint(1, n - 2)
        
        old_pos = chain[i]
        prev_pos = chain[i-1]
        next_pos = chain[i+1]
        
        moved = False
        for new_pos in ((old_pos[0] + 1, old_pos[1]), (old_pos[0] - 1, old_pos[1]), 
                        (old_pos[0], old_pos[1] + 1), (old_pos[0], old_pos[1] - 1)):
            if new_pos == prev_pos or new_pos == next_pos:
                continue
            if abs(new_pos[0] - prev_pos[0]) + abs(new_pos[1] - prev_pos[1]) != 1:
                continue
            if abs(new_pos[0] - next_pos[0]) + abs(new_pos[1] - next_pos[1]) != 1:
                continue
            if new_pos in pos_to_idx and pos_to_idx[new_pos] != i:
                continue
            
            delta = 0
            if sequence[i] == 'H':
                for nb in ((old_pos[0] + 1, old_pos[1]), (old_pos[0] - 1, old_pos[1]), 
                           (old_pos[0], old_pos[1] + 1), (old_pos[0], old_pos[1] - 1)):
                    j = pos_to_idx.get(nb)
                    if j is not None and j != i and abs(i - j) > 1 and sequence[j] == 'H':
                        delta -= 1
                for nb in ((new_pos[0] + 1, new_pos[1]), (new_pos[0] - 1, new_pos[1]), 
                           (new_pos[0], new_pos[1] + 1), (new_pos[0], new_pos[1] - 1)):
                    j = pos_to_idx.get(nb)
                    if j is not None and j != i and abs(i - j) > 1 and sequence[j] == 'H':
                        delta += 1
            
            if delta <= 0 or rng.random() < pow(2.718281828, -delta / t):
                del pos_to_idx[old_pos]
                pos_to_idx[new_pos] = i
                chain[i] = new_pos
                current_energy += delta
                
                if current_energy < best_energy:
                    best_chain = chain[:]
                    best_energy = current_energy
                moved = True
                break
    
    return best_chain


def fold(sequence, dimension):
    """
    Find the best conformation for the given HP sequence.
    Returns a list of [x, y] coordinates.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    # Fallback: straight line
    straight = [[i, 0] for i in range(n)]
    
    best_chain = None
    best_energy = 0  # Energy is negative, so we want the most negative (lowest)
    
    rng = random.Random(42)
    
    # Multi-start with increasing time budgets
    # First pass: quick greedy + local improve
    # Second pass: longer refinement with E2T
    # Third pass: min-conflicts style search
    
    # Phase 1: Quick greedy restarts
    for r in range(50):
        if time.time() > _DEADLINE - 30:
            break
        
        temp = 0.05 + rng.random() * 0.1
        chain = _fast_grow(sequence, rng, temperature=temp)
        
        if chain is None:
            continue
        
        # Quick local improvement
        chain = _local_improve_fast(sequence, chain, rng, time_budget=0.5)
        
        energy = _compute_energy(sequence, chain)
        
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
    
    if best_chain is None:
        return straight
    
    # Phase 2: Deeper refinement with E2T
    if time.time() < _DEADLINE - 20:
        best_chain = _regrow_sa_fast(sequence, best_chain, rng, time_budget=5.0)
        best_energy = _compute_energy(sequence, best_chain)
    
    # Phase 3: Min-conflicts style search on the best chain
    if time.time() < _DEADLINE - 10:
        mc_chain = _min_conflicts_fold(sequence, dimension, rng, time_budget=5.0)
        if mc_chain is not None:
            mc_energy = _compute_energy(sequence, mc_chain)
            if mc_energy < best_energy:
                best_chain = mc_chain
                best_energy = mc_energy
    
    # Phase 4: Final polish
    if time.time() < _DEADLINE - 5:
        best_chain = _local_improve_fast(sequence, best_chain, rng, time_budget=3.0)
    
    # Convert tuples to lists
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
