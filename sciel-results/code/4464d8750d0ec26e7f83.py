"""HP lattice protein folding on the cubic lattice using beam search + simulated annealing."""
import random
import time
import math

_DEADLINE = 315.0
_START_TIME = None


def _get_start():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME


def _time_left():
    return _DEADLINE - (time.time() - _get_start())


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute energy as minus the number of H-H contacts (non-adjacent along chain)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _beam_search(sequence, dimension, beam_width=50, max_beam=200):
    """
    Beam search growth: maintain a set of partial chains, expanding each step.
    Score each partial chain by energy so far plus a heuristic for potential.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    # Each beam entry: (chain, occupied_set, energy_so_far)
    # Start with all 6 possible first steps (or just one for small chains)
    start = (0, 0, 0)
    
    # Initialize beam with just the starting point
    beam = [([start], {start: 0}, 0)]
    
    for i in range(1, n):
        new_beam = []
        for chain, occupied, energy in beam:
            last = chain[-1]
            for site in _neighbours(last):
                if site in occupied:
                    continue
                # Check that the new position doesn't create a bond violation
                # (it's always valid since we're growing step by step)
                
                new_occupied = occupied | {site}
                new_chain = chain + [site]
                
                # Compute incremental energy: only need to check if this new H
                # forms contacts with previous H's
                new_energy = energy
                if sequence[i] == 'H':
                    for q in _neighbours(site):
                        if q in occupied:
                            j = occupied[q]
                            if j < i - 1 and sequence[j] == 'H':
                                new_energy -= 1
                
                new_beam.append((new_chain, new_occupied, new_energy))
        
        if not new_beam:
            break
        
        # Sort by energy (most negative first) and keep top beam_width
        new_beam.sort(key=lambda x: x[2])
        
        # Deduplicate: keep only unique chains (by their set of positions)
        # Use a hash of the last few positions to break symmetry
        seen = set()
        deduped = []
        for entry in new_beam:
            # Use the last 3 positions as a symmetry-breaking key
            key = tuple(entry[0][-3:]) if len(entry[0]) >= 3 else tuple(entry[0])
            if key not in seen:
                seen.add(key)
                deduped.append(entry)
            if len(deduped) >= max_beam:
                break
        
        beam = deduped
        
        if _time_left() < 5.0:
            break
    
    if not beam:
        return None
    
    # Return the best chain from the beam
    beam.sort(key=lambda x: x[2])
    return beam[0][0]


def _simulated_annealing(sequence, chain, dimension, rng, max_iters=50000):
    """
    Simulated annealing with crank moves and pivot moves.
    """
    n = len(chain)
    if n < 5:
        return chain
    
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    # Temperature schedule
    T_start = 2.0
    T_end = 0.01
    cooling = (T_end / T_start) ** (1.0 / max(1, max_iters))
    
    no_improve = 0
    best_no_improve = 0
    
    for it in range(max_iters):
        if _time_left() < 3.0:
            break
        
        T = T_start * (cooling ** it)
        
        # Choose move type
        move_type = rng.random()
        
        if move_type < 0.6:
            # Crank move: translate a subchain
            start_idx = rng.randint(1, n - 3)
            end_idx = rng.randint(start_idx + 1, n - 2)
            subchain = chain[start_idx:end_idx + 1]
            
            ax = rng.randint(0, dimension - 1)
            s = rng.choice((-1, 1))
            
            new_positions = []
            for p in subchain:
                np = list(p)
                np[ax] += s
                new_positions.append(tuple(np))
            
            # Remove old positions
            for p in subchain:
                del occupied[p]
            
            # Check validity
            valid = True
            seen = set()
            for np in new_positions:
                if np in occupied or np in seen:
                    valid = False
                    break
                seen.add(np)
            
            if valid and start_idx > 0:
                prev = chain[start_idx - 1]
                dist = sum(abs(a - b) for a, b in zip(prev, new_positions[0]))
                if dist != 1:
                    valid = False
            
            if valid and end_idx < n - 1:
                next_p = chain[end_idx + 1]
                dist = sum(abs(a - b) for a, b in zip(new_positions[-1], next_p))
                if dist != 1:
                    valid = False
            
            if not valid:
                for i, p in enumerate(subchain):
                    occupied[p] = start_idx + i
                no_improve += 1
                continue
            
            # Apply move
            for i, np in enumerate(new_positions):
                occupied[np] = start_idx + i
            
            new_chain = chain[:start_idx] + new_positions + chain[end_idx + 1:]
            new_energy = _energy(sequence, new_chain)
            
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                chain = new_chain
                current_energy = new_energy
                no_improve = 0
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
                    best_no_improve = 0
                else:
                    best_no_improve += 1
            else:
                # Revert
                for np in new_positions:
                    del occupied[np]
                for i, p in enumerate(subchain):
                    occupied[p] = start_idx + i
                no_improve += 1
                best_no_improve += 1
        
        else:
            # Pivot move: rotate a subchain around a pivot point
            if n < 6:
                no_improve += 1
                continue
                
            pivot_idx = rng.randint(1, n - 3)
            length = rng.randint(2, min(5, n - pivot_idx - 1))
            
            # Get the subchain to rotate
            subchain = chain[pivot_idx:pivot_idx + length]
            
            # Try all 24 rotations in 3D (simplified: try 6 basic rotations)
            rotations = []
            for ax in range(dimension):
                for s in (-1, 1):
                    rot = (ax, s)
                    rotations.append(rot)
            
            valid_rotation = False
            for ax, s in rotations:
                new_positions = []
                for p in subchain:
                    np = list(p)
                    np[ax] += s
                    new_positions.append(tuple(np))
                
                # Check if this rotation is valid
                # Remove old positions from pivot onwards
                for p in subchain:
                    del occupied[p]
                
                # Check for collisions
                collides = False
                seen = set()
                for np in new_positions:
                    if np in occupied or np in seen:
                        collides = True
                        break
                    seen.add(np)
                
                if not collides:
                    # Check bond connectivity at pivot
                    if pivot_idx > 0:
                        prev = chain[pivot_idx - 1]
                        dist = sum(abs(a - b) for a, b in zip(prev, new_positions[0]))
                        if dist != 1:
                            collides = True
                    else:
                        # First residue, check it's at origin or adjacent
                        pass
                    
                    if not collides and pivot_idx + length < n:
                        next_p = chain[pivot_idx + length]
                        dist = sum(abs(a - b) for a, b in zip(new_positions[-1], next_p))
                        if dist != 1:
                            collides = True
                
                if not collides:
                    # Apply rotation
                    for i, np in enumerate(new_positions):
                        occupied[np] = pivot_idx + i
                    
                    new_chain = chain[:pivot_idx] + new_positions + chain[pivot_idx + length:]
                    new_energy = _energy(sequence, new_chain)
                    
                    delta_e = new_energy - current_energy
                    
                    if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                        chain = new_chain
                        current_energy = new_energy
                        no_improve = 0
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(chain)
                            best_no_improve = 0
                        else:
                            best_no_improve += 1
                        valid_rotation = True
                        break
                    else:
                        # Revert
                        for np in new_positions:
                            del occupied[np]
                        for i, p in enumerate(subchain):
                            occupied[p] = pivot_idx + i
                
                else:
                    # Revert
                    for i, p in enumerate(subchain):
                        occupied[p] = pivot_idx + i
            
            if not valid_rotation:
                no_improve += 1
                best_no_improve += 1
        
        if no_improve > 2000:
            break
    
    return best_chain


def _local_search_polish(sequence, chain, dimension, rng, max_iters=10000):
    """
    Fast local search polish: only accept improvements, use crank moves.
    """
    n = len(chain)
    if n < 5:
        return chain
    
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    no_improve = 0
    
    for _ in range(max_iters):
        if _time_left() < 2.0:
            break
        
        start_idx = rng.randint(1, n - 3)
        end_idx = rng.randint(start_idx + 1, n - 2)
        subchain = chain[start_idx:end_idx + 1]
        
        ax = rng.randint(0, dimension - 1)
        s = rng.choice((-1, 1))
        
        new_positions = []
        for p in subchain:
            np = list(p)
            np[ax] += s
            new_positions.append(tuple(np))
        
        for p in subchain:
            del occupied[p]
        
        valid = True
        seen = set()
        for np in new_positions:
            if np in occupied or np in seen:
                valid = False
                break
            seen.add(np)
        
        if valid and start_idx > 0:
            prev = chain[start_idx - 1]
            dist = sum(abs(a - b) for a, b in zip(prev, new_positions[0]))
            if dist != 1:
                valid = False
        
        if valid and end_idx < n - 1:
            next_p = chain[end_idx + 1]
            dist = sum(abs(a - b) for a, b in zip(new_positions[-1], next_p))
            if dist != 1:
                valid = False
        
        if not valid:
            for i, p in enumerate(subchain):
                occupied[p] = start_idx + i
            no_improve += 1
            if no_improve > 500:
                break
            continue
        
        for i, np in enumerate(new_positions):
            occupied[np] = start_idx + i
        
        new_chain = chain[:start_idx] + new_positions + chain[end_idx + 1:]
        new_energy = _energy(sequence, new_chain)
        
        if new_energy < current_energy:
            chain = new_chain
            current_energy = new_energy
            no_improve = 0
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        else:
            for np in new_positions:
                del occupied[np]
            for i, p in enumerate(subchain):
                occupied[p] = start_idx + i
            no_improve += 1
            if no_improve > 500:
                break
    
    return best_chain


def fold(sequence, dimension):
    """
    Fold an HP sequence on the cubic lattice using beam search + simulated annealing.
    """
    rng = random.Random(42 + len(sequence) * 7 + sum(ord(c) for c in sequence[:5]))
    
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Beam search to get a good initial conformation
    if _time_left() > 10.0:
        beam_result = _beam_search(sequence, dimension, beam_width=80, max_beam=150)
        if beam_result is not None:
            best_chain = beam_result
            best_energy = _energy(sequence, best_chain)
    
    # Phase 2: Simulated annealing to refine
    if best_chain is not None and _time_left() > 15.0:
        refined = _simulated_annealing(sequence, best_chain, dimension, rng, max_iters=30000)
        if refined is not None:
            refined_energy = _energy(sequence, refined)
            if refined_energy < best_energy:
                best_chain = refined
                best_energy = refined_energy
    
    # Phase 3: Local search polish
    if best_chain is not None and _time_left() > 5.0:
        polished = _local_search_polish(sequence, best_chain, dimension, rng, max_iters=5000)
        if polished is not None:
            polished_energy = _energy(sequence, polished)
            if polished_energy < best_energy:
                best_chain = polished
                best_energy = polished_energy
    
    # Fallback: straight line
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
