"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time

# Explicit deadline: evaluator kills after 210s wall-clock; keep 30s safety margin.
_DEADLINE = time.time() + 180.0


def _neighbours(point):
    """Yield the 4 unit-step neighbours of a 2D point (tuple)."""
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


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


def _greedy_fold(sequence, dimension, rng, temperature=0.05):
    """
    Simulated annealing style chain growth.
    Returns a valid self-avoiding chain or None if stuck.
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
        for site in _neighbours(last):
            if site not in occupied:
                # Calculate immediate gain (new H-H contacts)
                gain = 0
                if sequence[i] == 'H':
                    for nb in _neighbours(site):
                        idx = occupied.get(nb)
                        if idx is not None and idx < i - 1 and sequence[idx] == 'H':
                            gain += 1
                # Calculate free space (heuristic for avoiding dead ends)
                free = sum(1 for nb in _neighbours(site) if nb not in occupied)
                
                # Score: gain is primary, free space secondary, random noise
                score = gain * 10.0 + free * 0.5 + rng.random() * temperature
                candidates.append((score, site))
        
        if not candidates:
            return None
        
        # Pick best candidate (with some randomness if close scores)
        # Sort by score descending
        candidates.sort(key=lambda x: -x[0])
        
        # With some probability, pick a random top-3 to allow exploration
        if len(candidates) > 1 and rng.random() < 0.2:
            k = min(3, len(candidates))
            choice_idx = rng.randint(0, k - 1)
        else:
            choice_idx = 0
            
        best_site = candidates[choice_idx][1]
        chain.append(best_site)
        occupied[best_site] = i
        
    return chain


def _monte_carlo_refine(sequence, chain, rng, time_budget=2.0):
    """
    Refine the chain using Monte Carlo with Local and End-to-Tail moves.
    End-to-Tail moves allow topological reconfiguration of the tail,
    which is crucial for escaping local minima in rugged landscapes.
    """
    if not chain or len(chain) < 3:
        return chain
    
    n = len(chain)
    current_energy = _compute_energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    start_time = time.time()
    
    # Simulated annealing parameters
    t_start = 1.0
    t_end = 0.01
    # Estimate iterations based on time budget, assume ~10000 it/s for simple moves
    # E2T moves are more expensive (O(n) regrowth), so we limit them
    max_iterations = 5000 
    cooling = (t_start - t_end) / max(max_iterations, 1)
    t = t_start
    
    # Build position-to-index map for O(1) lookup in local moves
    pos_to_idx = {}
    for i, p in enumerate(chain):
        pos_to_idx[p] = i
        
    # Precompute H positions for faster energy delta calculation if needed,
    # but for simplicity we recompute or use incremental where easy.
    # For E2T, we must recompute energy of the whole chain or use a fast delta.
    # Given n<=100, full recompute is O(n) ~ 100 ops, which is fast.
    
    for it in range(max_iterations):
        t = max(t - cooling, t_end)
        
        if time.time() - start_time > time_budget:
            break
        
        # Randomly choose between Local Move and End-to-Tail Move
        # Local moves are cheap, E2T are expensive but powerful
        if rng.random() < 0.7:
            # --- LOCAL MOVE ---
            i = rng.randint(1, n - 2)
            old_pos = chain[i]
            
            # Try each of the 4 neighbours
            moved = False
            for new_pos in _neighbours(old_pos):
                # Check if new position is valid (not occupied by other residues)
                if new_pos in pos_to_idx:
                    continue
                
                # Check if the move maintains connectivity
                prev_pos = chain[i-1]
                next_pos = chain[i+1]
                
                # Check adjacency to previous
                if abs(new_pos[0] - prev_pos[0]) + abs(new_pos[1] - prev_pos[1]) != 1:
                    continue
                    
                # Check adjacency to next
                if abs(new_pos[0] - next_pos[0]) + abs(new_pos[1] - next_pos[1]) != 1:
                    continue
                
                # Compute new energy
                # Temporarily update chain
                chain[i] = new_pos
                new_energy = _compute_energy(sequence, chain)
                delta_e = new_energy - current_energy
                
                # Accept if better, or with probability exp(-delta_e / t) if worse
                if delta_e <= 0 or rng.random() < pow(2.718281828, -delta_e / t):
                    # Update pos_to_idx
                    del pos_to_idx[old_pos]
                    pos_to_idx[new_pos] = i
                    current_energy = new_energy
                    
                    if current_energy < best_energy:
                        best_chain = chain[:]
                        best_energy = current_energy
                    moved = True
                else:
                    # Revert
                    chain[i] = old_pos
                if moved:
                    break
        else:
            # --- END-TO-TAIL (E2T) MOVE ---
            # Pick a random cut point i. The tail i..n-1 will be regrown.
            # To avoid trivial moves, pick i in the second half
            i = rng.randint(n // 2, n - 2)
            if i >= n:
                continue
            
            anchor = chain[i-1]
            # Remove tail from occupied set
            tail = chain[i:]
            for pos in tail:
                del pos_to_idx[pos]
            
            # Regrow the tail from anchor
            new_tail = [anchor]
            new_occupied = set(pos_to_idx.keys())
            new_occupied.add(anchor)
            
            success = True
            # We need to place residues i..n-1
            # This is a constrained growth: we must follow the sequence
            # and avoid occupied sites.
            # We use a randomized greedy strategy with temperature t
            for k in range(i, n):
                candidates = []
                last = new_tail[-1]
                for site in _neighbours(last):
                    if site in new_occupied:
                        continue
                    gain = 0
                    if sequence[k] == 'H':
                        for nb in _neighbours(site):
                            idx = pos_to_idx.get(nb)
                            if idx is not None and idx < k - 1 and sequence[idx] == 'H':
                                gain += 1
                            # Also check against the new tail itself?
                            # pos_to_idx only has the prefix.
                            # We need to check against new_tail positions too.
                            # Since new_tail is being built, we can check manually
                            # or add to a temp set.
                            # For speed, we only check against prefix in gain calc,
                            # which is an approximation but acceptable for move proposal.
                    free = sum(1 for nb in _neighbours(site) if nb not in new_occupied)
                    
                    score = gain * 10.0 + free * 0.5 + rng.random() * t
                    candidates.append((score, site))
                
                if not candidates:
                    success = False
                    break
                
                candidates.sort(key=lambda x: -x[0])
                # Pick best or random top-k
                if len(candidates) > 1 and rng.random() < 0.3:
                    k_choice = min(3, len(candidates))
                    choice_idx = rng.randint(0, k_choice - 1)
                else:
                    choice_idx = 0
                    
                best_site = candidates[choice_idx][1]
                new_tail.append(best_site)
                new_occupied.add(best_site)
            
            if success:
                # Construct new chain
                new_chain = chain[:i] + new_tail[1:]
                new_energy = _compute_energy(sequence, new_chain)
                delta_e = new_energy - current_energy
                
                if delta_e <= 0 or rng.random() < pow(2.718281828, -delta_e / t):
                    # Accept
                    chain = new_chain
                    current_energy = new_energy
                    
                    # Rebuild pos_to_idx
                    pos_to_idx = {p: idx for idx, p in enumerate(chain)}
                    
                    if current_energy < best_energy:
                        best_chain = chain[:]
                        best_energy = current_energy
                else:
                    # Revert: we already removed tail from pos_to_idx, so we must put it back
                    # Actually, we modified pos_to_idx by deleting the tail.
                    # If we reject, we must restore the original tail positions.
                    for idx, pos in enumerate(tail):
                        pos_to_idx[pos] = i + idx
            else:
                # Restore original tail to pos_to_idx
                for idx, pos in enumerate(tail):
                    pos_to_idx[pos] = i + idx

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
    
    # Number of restarts based on time budget
    # We have 180s total.
    # Greedy is fast. Refine is slower.
    # Let's do 100 restarts with aggressive refinement.
    num_restarts = 100
    
    for r in range(num_restarts):
        if time.time() > _DEADLINE - 15:
            break
            
        # Vary temperature slightly for diversity
        temp = 0.01 + rng.random() * 0.1
        chain = _greedy_fold(sequence, dimension, rng, temperature=temp)
        
        if chain is None:
            continue
        
        # Apply Monte Carlo refinement with E2T moves
        # Allocate more time for refinement as it's the structural improvement
        time_per_refine = 1.5
        chain = _monte_carlo_refine(sequence, chain, rng, time_budget=time_per_refine)
        
        energy = _compute_energy(sequence, chain)
        
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
            
    if best_chain is None:
        return straight
    
    # Final high-precision polish pass: run more iterations with tighter tolerance
    if time.time() < _DEADLINE - 5:
        best_chain = _monte_carlo_refine(sequence, best_chain, rng, time_budget=5.0)
    
    # Convert tuples to lists
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
