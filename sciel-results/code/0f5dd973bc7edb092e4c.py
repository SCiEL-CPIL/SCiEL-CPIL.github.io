# EVOLVE-BLOCK-START
import time
import random
import math

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _beam_search_init(sequence, dimension, beam_width, rng):
    n = len(sequence)
    if n == 0:
        return [([], 0)]
    
    start = (0,) * dimension
    start_h = {start} if sequence[0] == "H" else set()
    current_beam = [((start,), {start}, start_h, 0)]
    
    for i in range(1, n):
        next_candidates = []
        for chain, occ, h_pos, energy in current_beam:
            last = chain[-1]
            for site in _neighbours(last):
                if site in occ:
                    continue
                
                new_chain = chain + (site,)
                new_occ = set(occ)
                new_occ.add(site)
                
                delta_e = 0
                new_h_pos = set(h_pos)
                
                if sequence[i] == "H":
                    new_h_pos.add(site)
                    for q in _neighbours(site):
                        if q in h_pos:
                            if q != last:
                                delta_e -= 1
                # If sequence[i] is P, no new H-H contacts
                
                new_energy = energy + delta_e
                next_candidates.append((new_chain, new_occ, new_h_pos, new_energy))
        
        if not next_candidates:
            return []
            
        next_candidates.sort(key=lambda x: x[3])
        current_beam = next_candidates[:beam_width]
        
    results = []
    for chain, occ, h_pos, energy in current_beam:
        results.append((list(chain), energy))
    return results

def _local_search(sequence, chain, dimension, rng, time_limit):
    n = len(chain)
    best_chain = chain[:]
    best_energy = _energy(sequence, best_chain)
    
    start_time = time.time()
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        if no_improve_count > 500:
            break
            
        if rng.random() < 0.8:
            if n < 2: break
            k = rng.randint(1, n - 1)
            pivot_point = best_chain[k-1]
            segment = best_chain[k:]
            
            moved = False
            for axis in range(3):
                for angle in [1, 2, 3]:
                    new_segment = []
                    valid = True
                    occupied_by_head = set(best_chain[:k])
                    
                    rel_positions = []
                    for p in segment:
                        rel = [p[d] - pivot_point[d] for d in range(3)]
                        rel_positions.append(rel)
                        
                    rotated_rels = []
                    for rel in rel_positions:
                        x, y, z = rel
                        if axis == 0:
                            if angle == 1: nr = (x, -z, y)
                            elif angle == 2: nr = (x, -y, -z)
                            elif angle == 3: nr = (x, z, -y)
                        elif axis == 1:
                            if angle == 1: nr = (z, y, -x)
                            elif angle == 2: nr = (-x, y, -z)
                            elif angle == 3: nr = (-z, y, x)
                        else:
                            if angle == 1: nr = (-y, x, z)
                            elif angle == 2: nr = (-x, -y, z)
                            elif angle == 3: nr = (y, -x, z)
                        rotated_rels.append(nr)
                        
                    for nr in rotated_rels:
                        np = [pivot_point[d] + nr[d] for d in range(3)]
                        np = tuple(np)
                        if np in occupied_by_head:
                            valid = False
                            break
                        new_segment.append(np)
                    
                    if not valid:
                        continue
                        
                    if len(set(new_segment)) != len(new_segment):
                        continue
                        
                    candidate_chain = best_chain[:k] + new_segment
                    cand_energy = _energy(sequence, candidate_chain)
                    
                    if cand_energy < best_energy:
                        best_chain = candidate_chain
                        best_energy = cand_energy
                        no_improve_count = 0
                        moved = True
                        break
                    else:
                        no_improve_count += 1
                        
            if moved:
                continue
            else:
                no_improve_count += 1
        else:
            if n < 3: break
            k = rng.randint(1, n - 2)
            old_pos = best_chain[k]
            neighbors = list(_neighbours(old_pos))
            rng.shuffle(neighbors)
            moved = False
            for new_pos in neighbors:
                if new_pos in set(best_chain):
                    continue
                candidate_chain = best_chain[:]
                candidate_chain[k] = new_pos
                cand_energy = _energy(sequence, candidate_chain)
                if cand_energy < best_energy:
                    best_chain = candidate_chain
                    best_energy = cand_energy
                    no_improve_count = 0
                    moved = True
                    break
                else:
                    no_improve_count += 1
            if not moved:
                no_improve_count += 1
                
    return best_chain, best_energy

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0]*dimension]
        
    time_per_seq = 10.0
    start_time = time.time()
    rng = random.Random(12345 + len(sequence))
    
    beam_width = 50
    inits = _beam_search_init(sequence, dimension, beam_width, rng)
    
    if not inits:
        return [[i] + [0]*(dimension-1) for i in range(n)]
        
    best_chain = None
    best_energy = 0
    
    time_left = time_per_seq - (time.time() - start_time)
    if time_left <= 0.1:
        best_chain, best_energy = min(inits, key=lambda x: x[1])
        return [list(p) for p in best_chain]
        
    inits.sort(key=lambda x: x[1])
    top_inits = inits[:5]
    
    time_per_init = time_left / len(top_inits)
    
    for init_chain, init_energy in top_inits:
        if time.time() - start_time > time_per_seq - 0.5:
            break
        local_time = min(time_per_init, time_per_seq - (time.time() - start_time) - 0.5)
        if local_time < 0.1:
            break
            
        new_chain, new_energy = _local_search(sequence, init_chain, dimension, rng, local_time)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_chain = new_chain
            
    if best_chain is None:
        best_chain = inits[0][0]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
