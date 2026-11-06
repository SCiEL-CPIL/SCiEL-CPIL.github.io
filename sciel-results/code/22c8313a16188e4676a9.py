import time
import random

_NEIGHBOR_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

def _calc_delta_energy(sequence, occupied, new_pos, new_idx):
    """Calculate energy delta when placing residue new_idx at new_pos."""
    if sequence[new_idx] != 'H':
        return 0
    
    delta = 0
    x, y, z = new_pos
    for dx, dy, dz in _NEIGHBOR_OFFSETS:
        q = (x + dx, y + dy, z + dz)
        if q in occupied:
            j = occupied[q]
            # Contact if j is not adjacent in chain (j != new_idx - 1) and j is H
            if j != new_idx - 1 and sequence[j] == 'H':
                delta += 1
    return delta

def _greedy_complete(sequence, chain, occupied, start_idx, rng, time_limit):
    """Complete the chain greedily with some randomness."""
    n = len(sequence)
    if start_idx >= n:
        return chain, occupied, True
    
    for i in range(start_idx, n):
        last_pos = chain[-1]
        valid_moves = []
        
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            q = (last_pos[0] + dx, last_pos[1] + dy, last_pos[2] + dz)
            if q not in occupied:
                # Score this move
                score = 0.0
                if sequence[i] == 'H':
                    for ndx, ndy, ndz in _NEIGHBOR_OFFSETS:
                        r = (q[0] + ndx, q[1] + ndy, q[2] + ndz)
                        if r in occupied:
                            idx = occupied[r]
                            if idx != i - 1 and sequence[idx] == 'H':
                                score += 10.0  # Strong preference for H-H contacts
                
                # Add free space bonus (avoid trapping)
                free_count = 0
                for ndx, ndy, ndz in _NEIGHBOR_OFFSETS:
                    r = (q[0] + ndx, q[1] + ndy, q[2] + ndz)
                    if r not in occupied:
                        free_count += 1
                score += free_count * 1.0
                
                # Add small random component for diversity
                score += rng.random() * 0.5
                
                valid_moves.append((score, q))
        
        if not valid_moves:
            return chain, occupied, False
        
        # Sort by score descending and pick from top few
        valid_moves.sort(key=lambda x: x[0], reverse=True)
        
        # Pick from top 3 with weighted probability
        top_k = min(3, len(valid_moves))
        weights = [1.0 / (k + 1) for k in range(top_k)]
        total_w = sum(weights)
        r = rng.random() * total_w
        cum = 0
        chosen = valid_moves[0][1]
        for k in range(top_k):
            cum += weights[k]
            if r <= cum:
                chosen = valid_moves[k][1]
                break
        
        chain.append(chosen)
        occupied[chosen] = i
        
        # Time check
        if time.time() > time_limit:
            return chain, occupied, (i == n - 1)
    
    return chain, occupied, True

def _beam_search(sequence, beam_width, rng, time_limit):
    """Run beam search and return best complete chain."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    
    # Initialize beam
    beam = [(0, [(0, 0, 0)], {(0, 0, 0): 0})]
    
    for i in range(1, n):
        new_beam = []
        
        for energy, chain, occupied in beam:
            last_pos = chain[-1]
            
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                q = (last_pos[0] + dx, last_pos[1] + dy, last_pos[2] + dz)
                if q not in occupied:
                    delta = _calc_delta_energy(sequence, occupied, q, i)
                    new_energy = energy - delta  # More contacts = lower energy
                    
                    new_chain = chain + [q]
                    new_occupied = dict(occupied)
                    new_occupied[q] = i
                    
                    new_beam.append((new_energy, new_chain, new_occupied))
        
        if not new_beam:
            # Dead end - return what we have
            break
        
        # Sort by energy (ascending)
        new_beam.sort(key=lambda x: x[0])
        
        # Keep top beam_width with some diversity
        # Take top 80% strictly by energy, rest random from next 40%
        strict_count = int(beam_width * 0.8)
        diversity_count = beam_width - strict_count
        
        beam = new_beam[:strict_count]
        
        if diversity_count > 0 and len(new_beam) > strict_count:
            # Randomly select from the next portion
            pool = new_beam[strict_count:strict_count + int(beam_width * 0.6)]
            if pool:
                selected = rng.sample(pool, min(diversity_count, len(pool)))
                beam.extend(selected)
        
        # Time check
        if time.time() > time_limit:
            # Complete the best chain greedily
            best_energy, best_chain, best_occupied = beam[0]
            _, _, complete = _greedy_complete(sequence, best_chain, best_occupied, i, rng, time_limit)
            if complete and len(best_chain) == n:
                return [list(p) for p in best_chain]
            else:
                # Fallback: return straight line
                return [[i, 0, 0] for i in range(n)]
    
    # If we completed all residues
    if len(beam[0][1]) == n:
        return [list(p) for p in beam[0][1]]
    else:
        # Complete the best chain
        best_energy, best_chain, best_occupied = beam[0]
        _, _, complete = _greedy_complete(sequence, best_chain, best_occupied, len(best_chain), rng, time_limit)
        if complete and len(best_chain) == n:
            return [list(p) for p in best_chain]
        else:
            return [[i, 0, 0] for i in range(n)]

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    global _global_start_time
    if not hasattr(fold, '_global_start_time'):
        fold._global_start_time = time.time()
    
    # Total time budget: 320 seconds (leave 10s margin for 330s limit)
    total_budget = 320.0
    elapsed = time.time() - fold._global_start_time
    remaining = total_budget - elapsed
    
    if remaining <= 0:
        # Out of time, return straight line
        return [[i, 0, 0] for i in range(n)]
    
    # Allocate time for this sequence based on length
    # Longer sequences get more time
    base_time = 5.0
    extra_per_residue = 0.3
    seq_time = min(base_time + extra_per_residue * n, remaining * 0.8)
    
    if seq_time < 1.0:
        seq_time = 1.0
    
    time_limit = time.time() + seq_time
    
    # Multiple restarts with different beam widths and seeds
    best_chain = None
    best_energy = float('inf')
    
    # Strategy: Try multiple configurations
    configs = [
        (100, 42),   # Wide beam, seed 42
        (50, 123),   # Medium beam, seed 123
        (200, 777),  # Very wide beam, seed 777
        (30, 999),   # Narrow beam, seed 999
    ]
    
    # Adjust configs based on available time
    if seq_time > 8.0:
        configs.append((150, 555))
    if seq_time > 12.0:
        configs.append((250, 1111))
    
    for beam_width, seed in configs:
        if time.time() > time_limit:
            break
        
        rng = random.Random(seed)
        try:
            chain = _beam_search(sequence, beam_width, rng, time_limit)
            
            if chain and len(chain) == n:
                # Validate self-avoidance
                occupied = set()
                valid = True
                for i, pos in enumerate(chain):
                    if pos in occupied:
                        valid = False
                        break
                    occupied.add(tuple(pos))
                
                if valid:
                    # Calculate energy
                    energy = 0
                    pos_to_idx = {tuple(pos): i for i, pos in enumerate(chain)}
                    for i, pos in enumerate(chain):
                        if sequence[i] == 'H':
                            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                                q = (pos[0] + dx, pos[1] + dy, pos[2] + dz)
                                j = pos_to_idx.get(q)
                                if j is not None and j != i - 1 and j != i + 1 and sequence[j] == 'H':
                                    energy -= 1
                    
                    if energy < best_energy:
                        best_energy = energy
                        best_chain = chain
                        
        except Exception:
            continue
    
    if best_chain is not None:
        return best_chain
    else:
        # Fallback: straight line
        return [[i, 0, 0] for i in range(n)]
