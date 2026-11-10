"""HP lattice protein folding on the square lattice using E2T Monte Carlo moves."""
# EVOLVE-BLOCK-START
import random
import time
import sys

_DEADLINE = time.time() + 175.0


def _compute_energy(sequence, chain):
    """Compute the negative number of non-adjacent H-H contacts."""
    n = len(chain)
    if n < 2:
        return 0
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i in range(n):
        if sequence[i] == 'H':
            x, y = chain[i]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                j = where.get((x + dx, y + dy))
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2


def _fast_grow(sequence, rng, temperature=0.0):
    """
    Fast randomized greedy chain growth.
    Prefer sites that create new H-H contacts, then sites with more free space.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    res_type = [1 if c == 'H' else 0 for c in sequence]
    
    for i in range(1, n):
        last = chain[-1]
        x, y = last
        options = []
        
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            site = (x + dx, y + dy)
            if site in occupied:
                continue
            
            gain = 0
            if res_type[i]:
                sx, sy = site
                for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -4)):
                    j = occupied.get((sx + ddx, sy + ddy))
                    if j is not None and j < i - 1 and res_type[j]:
                        gain += 1
            
            free = 0
            for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (site[0] + ddx, site[1] + ddy) not in occupied:
                    free += 1
            
            score = gain * 10.0 + free * 0.1 + rng.random() * temperature
            options.append((score, site))
        
        if not options:
            return None
        
        best_score = -1e18
        best_site = None
        for s, st in options:
            if s > best_score:
                best_score = s
                best_site = st
        
        occupied[best_site] = i
        chain.append(best_site)

    return chain


def _e2t_move(sequence, chain, rng, max_attempts=50):
    """
    Monte Carlo End-to-Tail (E2T) move: pick a random cut point k,
    then regrow the tail from residue k onwards while keeping the head fixed.
    Returns the new chain if successful, else None.
    """
    n = len(sequence)
    if n < 4:
        return None
    
    # Pick a random cut point (at least 3 residues from the end to allow some growth)
    k = rng.randint(2, n - 3)
    
    head = chain[:k]
    head_set = set(head)
    
    # Start regrowing from the last position of the head
    current_pos = head[-1]
    new_tail = [current_pos]
    occupied = dict(head_set)
    # Update occupied dict to map position -> index
    for i, p in enumerate(head):
        occupied[p] = i
    
    res_type = [1 if c == 'H' else 0 for c in sequence]
    
    success = False
    for attempt in range(max_attempts):
        # Try to grow the tail
        trial_tail = [current_pos]
        trial_occupied = {p: i for i, p in enumerate(head)}
        
        valid = True
        for i in range(k, n):
            x, y = trial_tail[-1]
            options = []
            
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                site = (x + dx, y + dy)
                if site in trial_occupied:
                    continue
                
                gain = 0
                if res_type[i]:
                    sx, sy = site
                    for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        j = trial_occupied.get((sx + ddx, sy + ddy))
                        if j is not None and j < i - 1 and res_type[j]:
                            gain += 1
                
                free = 0
                for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    if (site[0] + ddx, site[1] + ddy) not in trial_occupied:
                        free += 1
                
                score = gain * 10.0 + free * 0.1 + rng.random()
                options.append((score, site))
            
            if not options:
                valid = False
                break
            
            best_score = -1e18
            best_site = None
            for s, st in options:
                if s > best_score:
                    best_score = s
                    best_site = st
            
            trial_occupied[best_site] = i
            trial_tail.append(best_site)
        
        if valid:
            success = True
            break
    
    if not success:
        return None
    
    new_chain = head + trial_tail[1:]
    return new_chain


def _local_improve(sequence, chain, rng, max_moves=1000):
    """
    Improve conformation via E2T moves and local kink flips.
    """
    n = len(chain)
    if n < 4:
        return chain
    
    best_chain = chain
    best_energy = _compute_energy(sequence, chain)
    
    res_type = [1 if c == 'H' else 0 for c in sequence]
    
    moves_made = 0
    for _ in range(max_moves):
        if time.time() > _DEADLINE:
            break
        
        # Try an E2T move
        new_chain = _e2t_move(sequence, best_chain, rng, max_attempts=20)
        if new_chain is not None:
            new_energy = _compute_energy(sequence, new_chain)
            if new_energy < best_energy:
                best_chain = new_chain
                best_energy = new_energy
                moves_made += 1
            continue
        
        # Try a simple kink flip: pick a random internal residue and try to move it
        # to an adjacent empty site
        if n > 3:
            i = rng.randint(1, n - 2)
            x, y = best_chain[i]
            old_pos = best_chain[i]
            
            # Try all 4 neighbors
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                new_pos = (x + dx, y + dy)
                
                # Check if new_pos is occupied by another residue
                occupied_set = set(best_chain)
                if new_pos in occupied_set:
                    # It's occupied, so we can only move if it's the same residue (no-op)
                    continue
                
                # Check if the move is valid: new_pos must not be occupied by any other residue
                # and the chain must remain self-avoiding
                # Also, new_pos must not be adjacent to the same residue's neighbors in a way that causes overlap
                # Actually, we just need to ensure new_pos is not in the set of all other positions
                
                # Check if new_pos is adjacent to chain[i-1] or chain[i+1] (it will be, by construction of a kink)
                # But we need to ensure no other residue is at new_pos
                if new_pos in occupied_set:
                    continue
                
                # Perform the move tentatively
                best_chain[i] = new_pos
                
                # Check self-avoidance: new_pos must not be in the chain at any other index
                # Since we already checked it's not in occupied_set (which includes all positions), this is fine
                # But wait, occupied_set includes new_pos now? No, we haven't updated it yet.
                # Let's be more careful.
                
                # Actually, let's just compute the energy and check self-avoidance explicitly
                # for simplicity, since n is small
                
                # Check self-avoidance
                pos_set = set()
                valid = True
                for j, p in enumerate(best_chain):
                    if p in pos_set:
                        valid = False
                        break
                    pos_set.add(p)
                
                if not valid:
                    best_chain[i] = old_pos
                    continue
                
                new_energy = _compute_energy(sequence, best_chain)
                if new_energy < best_energy:
                    best_energy = new_energy
                    moves_made += 1
                    break
                else:
                    best_chain[i] = old_pos
                    continue
            
            if best_chain[i] != old_pos:
                continue
        
        # If no improvement in this iteration, we could try a random perturbation
        # but for now, just continue
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation using randomized greedy growth followed by E2T local search."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 1
    
    # Try multiple initial growths
    num_growths = min(50, max(10, n // 2))
    
    for _ in range(num_growths):
        if time.time() > _DEADLINE:
            break
        
        chain = _fast_grow(sequence, rng, temperature=0.5)
        if chain is None:
            continue
        
        # Local improvement via E2T
        chain = _local_improve(sequence, chain, rng, max_moves=500)
        
        value = _compute_energy(sequence, chain)
        if value < best_energy:
            best_chain = chain
            best_energy = value
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
