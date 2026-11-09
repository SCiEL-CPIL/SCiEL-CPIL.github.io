"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

_DEADLINE = 320.0
_START_TIME = None


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _precompute(sequence):
    h_set = set(i for i, c in enumerate(sequence) if c == 'H')
    return h_set


def _grow_fast(sequence, dimension, rng, h_set):
    """Fast greedy growth."""
    n = len(sequence)
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    for i in range(1, n):
        best_score = -999
        best_site = None
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if i in h_set:
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and j in h_set:
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 10 + free + rng.random()
            if score > best_score:
                best_score = score
                best_site = site
        if best_site is None:
            return None
        chain.append(best_site)
        occupied[best_site] = i
    return chain


def _compute_energy(sequence, chain, h_set):
    """Compute energy: negative count of H-H non-bonded contacts."""
    n = len(chain)
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i in range(n):
        if i not in h_set:
            continue
        p = chain[i]
        for q in _neighbours(p):
            j = pos_to_idx.get(q)
            if j is not None and j > i + 1 and j in h_set:
                contacts += 1
    return -contacts


def _anneal(sequence, dimension, rng, h_set, time_limit, start_chain):
    """Simulated annealing with efficient energy updates."""
    n = len(sequence)
    chain = list(start_chain)
    current_energy = _compute_energy(sequence, chain, h_set)
    best_chain = list(chain)
    best_energy = current_energy
    
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    
    T0 = max(0.5, abs(current_energy) * 0.05)
    T_end = 0.005
    max_iters = max(50, int(time_limit / 0.0005))
    alpha = (T_end / T0) ** (1.0 / max_iters)
    T = T0
    
    start_t = time.time()
    
    for it in range(max_iters):
        if time.time() - start_t > time_limit:
            break
        
        move_type = rng.random()
        accepted = False
        
        if move_type < 0.5:
            # Single residue move
            idx = rng.randint(1, n - 2)
            old_pos = chain[idx]
            new_pos = None
            for _ in range(15):
                candidate = tuple(old_pos[a] + rng.choice([-1, 0, 1]) for a in range(dimension))
                if candidate == old_pos or candidate in pos_to_idx:
                    continue
                prev = chain[idx - 1]
                nxt = chain[idx + 1]
                if (abs(candidate[0] - prev[0]) + abs(candidate[1] - prev[1]) + abs(candidate[2] - prev[2]) == 1 or
                    abs(candidate[0] - nxt[0]) + abs(candidate[1] - nxt[1]) + abs(candidate[2] - nxt[2]) == 1):
                    new_pos = candidate
                    break
            
            if new_pos is None:
                T *= alpha
                continue
            
            # Apply tentatively
            chain[idx] = new_pos
            del pos_to_idx[old_pos]
            pos_to_idx[new_pos] = idx
            
            new_energy = _compute_energy(sequence, chain, h_set)
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                current_energy = new_energy
                accepted = True
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
            else:
                chain[idx] = old_pos
                del pos_to_idx[new_pos]
                pos_to_idx[old_pos] = idx
        
        elif move_type < 0.8:
            # Segment flip
            seg_len = rng.randint(2, min(4, n - 2))
            start_idx = rng.randint(1, n - seg_len - 1)
            end_idx = start_idx + seg_len - 1
            
            segment = chain[start_idx:end_idx + 1]
            reversed_segment = list(reversed(segment))
            
            # Check validity
            temp_pos = {p: i for i, p in enumerate(chain)}
            for i in range(start_idx, end_idx + 1):
                del temp_pos[chain[i]]
            
            valid = True
            for offset, new_p in enumerate(reversed_segment):
                if new_p in temp_pos:
                    valid = False
                    break
            
            if valid:
                if start_idx > 0:
                    prev = chain[start_idx - 1]
                    first_new = reversed_segment[0]
                    if abs(prev[0] - first_new[0]) + abs(prev[1] - first_new[1]) + abs(prev[2] - first_new[2]) != 1:
                        valid = False
                if valid and end_idx < n - 1:
                    nxt = chain[end_idx + 1]
                    last_new = reversed_segment[-1]
                    if abs(nxt[0] - last_new[0]) + abs(nxt[1] - last_new[1]) + abs(nxt[2] - last_new[2]) != 1:
                        valid = False
            
            if not valid:
                T *= alpha
                continue
            
            chain[start_idx:end_idx + 1] = reversed_segment
            new_energy = _compute_energy(sequence, chain, h_set)
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                current_energy = new_energy
                accepted = True
                pos_to_idx = {p: i for i, p in enumerate(chain)}
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
            else:
                chain[start_idx:end_idx + 1] = segment
                pos_to_idx = {p: i for i, p in enumerate(chain)}
        
        else:
            # Local 2-residue move
            idx = rng.randint(1, n - 3)
            old1, old2 = chain[idx], chain[idx + 1]
            
            # Try moving both residues
            new1 = None
            new2 = None
            for _ in range(10):
                c1 = tuple(old1[a] + rng.choice([-1, 0, 1]) for a in range(dimension))
                c2 = tuple(old2[a] + rng.choice([-1, 0, 1]) for a in range(dimension))
                if c1 == old1 or c2 == old2 or c1 == c2:
                    continue
                if c1 in pos_to_idx or c2 in pos_to_idx:
                    continue
                # Check connectivity
                prev = chain[idx - 1]
                nxt = chain[idx + 2]
                if abs(c1[0] - prev[0]) + abs(c1[1] - prev[1]) + abs(c1[2] - prev[2]) != 1:
                    continue
                if abs(c2[0] - nxt[0]) + abs(c2[1] - nxt[1]) + abs(c2[2] - nxt[2]) != 1:
                    continue
                if abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]) + abs(c1[2] - c2[2]) != 1:
                    continue
                new1, new2 = c1, c2
                break
            
            if new1 is None:
                T *= alpha
                continue
            
            chain[idx] = new1
            chain[idx + 1] = new2
            del pos_to_idx[old1]
            del pos_to_idx[old2]
            pos_to_idx[new1] = idx
            pos_to_idx[new2] = idx + 1
            
            new_energy = _compute_energy(sequence, chain, h_set)
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                current_energy = new_energy
                accepted = True
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
            else:
                chain[idx] = old1
                chain[idx + 1] = old2
                del pos_to_idx[new1]
                del pos_to_idx[new2]
                pos_to_idx[old1] = idx
                pos_to_idx[old2] = idx + 1
        
        T *= alpha
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using greedy growth + simulated annealing."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    h_set = _precompute(sequence)
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    elapsed = time.time() - _START_TIME
    remaining = _DEADLINE - elapsed
    if remaining < 5:
        chain = _grow_fast(sequence, dimension, rng, h_set)
        if chain is None:
            return [list([i] + [0] * (dimension - 1)) for i in range(n)]
        return [list(p) for p in chain]
    
    best_chain = None
    best_energy = 1
    
    # Allocate time for annealing restarts
    anneal_time = remaining * 0.75
    num_restarts = max(1, int(anneal_time / 1.5))
    per_restart_time = anneal_time / num_restarts
    
    for r in range(num_restarts):
        if time.time() - _START_TIME > _DEADLINE - 5:
            break
        
        init_chain = _grow_fast(sequence, dimension, rng, h_set)
        if init_chain is None:
            continue
        
        chain, energy = _anneal(sequence, dimension, rng, h_set, per_restart_time, init_chain)
        
        if chain is not None and energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    # Final polish
    if best_chain is not None:
        remaining_polish = _DEADLINE - 3 - (time.time() - _START_TIME)
        if remaining_polish > 1:
            chain, energy = _anneal(sequence, dimension, rng, h_set, remaining_polish, best_chain)
            if chain is not None and energy < best_energy:
                best_energy = energy
                best_chain = chain
    
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
