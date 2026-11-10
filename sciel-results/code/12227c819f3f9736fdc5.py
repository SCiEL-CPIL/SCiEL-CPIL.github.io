"""HP lattice protein folding on the square lattice."""
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


def _energy_delta_kink(sequence, chain, i):
    """
    Compute the energy change if residue i is moved to a new position.
    Only considers contacts that change: those involving residue i.
    Returns the delta in energy (negative is improvement).
    """
    n = len(chain)
    if i < 1 or i > n - 2:
        return 0
    
    # Current position
    old_pos = chain[i]
    old_x, old_y = old_pos
    
    # Neighbors of residue i in the chain (i-1 and i+1)
    # These are adjacent along the chain, so they don't count as contacts
    
    # Find all H residues that are near old_pos (within 1 unit)
    # and are not i-1 or i+1
    old_contacts = 0
    if sequence[i] == 'H':
        for dx, dy in ((1, 0), (-1, 1), (0, 1), (0, -1), (-1, 0)):
            pass  # This is wrong, let me think more carefully
    
    # Actually, let's just compute the energy difference directly
    # by checking which H-H pairs are affected
    
    # Save old chain
    old_chain = list(chain)
    
    # This is getting complex. Let me use a simpler approach:
    # Just compute the full energy before and after.
    # But that's O(n) which is what we're trying to avoid.
    
    # Let me reconsider. The energy only changes for contacts involving residue i.
    # A contact is between two H residues at positions p and q where p and q are adjacent
    # on the lattice but not adjacent in the chain.
    
    # So for residue i, we need to check:
    # 1. All H residues j (j != i-1, j != i+1) that are adjacent to old_pos
    # 2. All H residues j (j != i-1, j != i+1) that are adjacent to new_pos
    # 3. The difference in the number of such contacts
    
    # But we don't have new_pos here. This function should take new_pos as argument.
    
    # Let me redefine this function.
    pass


def _energy_delta_for_move(sequence, chain, i, new_pos):
    """
    Compute the energy delta when moving residue i from chain[i] to new_pos.
    Only considers contacts involving residue i.
    """
    n = len(chain)
    if i < 1 or i > n - 2:
        return 0
    
    old_pos = chain[i]
    old_x, old_y = old_pos
    new_x, new_y = new_pos
    
    # Build a set of occupied positions for quick lookup
    # Actually, we need to know which positions are occupied by H residues
    # and their indices
    
    # Let's build a map from position to index
    pos_to_idx = {}
    for idx, pos in enumerate(chain):
        pos_to_idx[pos] = idx
    
    # Count old contacts involving residue i
    old_contacts = 0
    if sequence[i] == 'H':
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor_pos = (old_x + dx, old_y + dy)
            j = pos_to_idx.get(neighbor_pos)
            if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                old_contacts += 1
    
    # Count new contacts involving residue i
    # But we need to account for the fact that new_pos might be occupied by another residue
    # In a valid move, new_pos is not occupied by any other residue
    # So we just check the neighbors of new_pos
    
    new_contacts = 0
    if sequence[i] == 'H':
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor_pos = (new_x + dx, new_y + dy)
            j = pos_to_idx.get(neighbor_pos)
            if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                new_contacts += 1
    
    # The energy delta is (new_contacts - old_contacts) / 2
    # because each contact is counted twice (once for each residue)
    # Wait, no. In _compute_energy, we count each contact once by iterating
    # over all H residues and checking their neighbors. So each contact is counted
    # exactly once (when we process the first residue in the pair).
    
    # Actually, let me re-examine _compute_energy:
    # for i in range(n):
    #     if sequence[i] == 'H':
    #         for dx, dy in ...:
    #             j = where.get(...)
    #             if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
    #                 contacts += 1
    # return -contacts // 2
    
    # So contacts is the number of (i, j) pairs where i and j are both H,
    # adjacent on the lattice, and not adjacent in the chain.
    # Each physical contact is counted twice: once when i is the first residue
    # and j is the second, and once when j is the first and i is the second.
    # So we divide by 2.
    
    # Therefore, old_contacts is the number of such pairs involving residue i,
    # and new_contacts is the number of such pairs involving residue i at new_pos.
    # The delta in the total contacts count is (new_contacts - old_contacts).
    # The delta in energy is -(new_contacts - old_contacts) // 2
    # But since we're dealing with integers, we need to be careful.
    
    # Actually, the energy is -contacts // 2, so:
    # new_energy - old_energy = -(new_contacts_total) // 2 - (-(old_contacts_total) // 2)
    # = -(new_contacts_total - old_contacts_total) // 2
    # = -((new_contacts_total - old_contacts_total)) // 2
    
    # And new_contacts_total - old_contacts_total = new_contacts - old_contacts
    # (because only the contacts involving residue i change)
    
    delta_contacts = new_contacts - old_contacts
    delta_energy = -delta_contacts // 2
    
    # But we need to be careful with the integer division.
    # Let's just return the delta in contacts, and let the caller handle the energy.
    
    return delta_contacts


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
                for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
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


def _local_improve_fast(sequence, chain, rng, max_moves=2000):
    """
    Improve conformation via kink flips with incremental energy computation.
    """
    n = len(chain)
    if n < 3:
        return chain, _compute_energy(sequence, chain)

    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    no_improve_count = 0
    
    res_type = [1 if c == 'H' else 0 for c in sequence]
    
    for _ in range(max_moves):
        if time.time() > _DEADLINE - 5:
            break

        # Kink flip: rotate residue i around residue i-1
        i = rng.randint(1, n - 2)
        p = chain[i]
        prev = chain[i - 1]
        nxt = chain[i + 1]
        px, py = prev
        
        candidates = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            np = (px + dx, py + dy)
            if np == p:
                continue
            if abs(np[0] - nxt[0]) + abs(np[1] - nxt[1]) != 1:
                continue
            if np in occupied and occupied[np] != i:
                continue
            candidates.append(np)

        if candidates:
            np = candidates[rng.randint(0, len(candidates) - 1)]
            
            # Compute energy delta incrementally
            old_x, old_y = p
            new_x, new_y = np
            delta_contacts = 0
            if res_type[i]:
                # Remove old contacts
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    j = occupied.get((old_x + dx, old_y + dy))
                    if j is not None and j != i and abs(j - i) > 1 and res_type[j]:
                        delta_contacts -= 1
                # Add new contacts
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    j = occupied.get((new_x + dx, new_y + dy))
                    if j is not None and j != i and abs(j - i) > 1 and res_type[j]:
                        delta_contacts += 1
            
            new_energy = current_energy - delta_contacts // 2

            if new_energy < current_energy:
                del occupied[p]
                chain[i] = np
                occupied[np] = i
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(chain)
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > 300:
                    break

    return best_chain, best_energy


def _regrow_sa_fast(sequence, chain, rng, max_iters=3000, start_temp=5.0, cool_rate=0.997):
    """
    Simulated annealing using regrow moves with E2T bias: pick a split point, regrow the suffix.
    Biased toward early split points (first 20-30%) for large-scale topological reconfiguration.
    """
    n = len(sequence)
    if n < 4:
        return chain, _compute_energy(sequence, chain)

    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    temp = start_temp
    no_improve = 0
    stuck_count = 0

    res_type = [1 if c == 'H' else 0 for c in sequence]

    for it in range(max_iters):
        if time.time() > _DEADLINE - 10:
            break
        if temp < 0.05:
            break

        min_split = 3
        max_split = max(min_split, n - 3)
        if max_split <= min_split:
            break
        
        # E2T bias: 60% chance to pick early split (first 25% of chain)
        if rng.random() < 0.6:
            early_max = min(max_split, max(min_split, int(n * 0.25)))
            split = rng.randint(min_split, early_max)
        elif rng.random() < 0.5:
            mid = (min_split + max_split) // 2
            offset = int(rng.gauss(0, (max_split - min_split) / 4))
            split = min(max(min_split, mid + offset), max_split)
        else:
            split = rng.randint(min_split, max_split)

        prefix = chain[:split]
        
        new_suffix = []
        last = prefix[-1]
        occupied = {p: i for i, p in enumerate(prefix)}
        success = True
        
        for i in range(split, n):
            x, y = last
            options = []
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                site = (x + dx, y + dy)
                if site in occupied:
                    continue
                gain = 0
                if res_type[i]:
                    sx, sy = site
                    for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        j = occupied.get((sx + ddx, sy + ddy))
                        if j is not None and j < i - 1 and res_type[j]:
                            gain += 1
                free = 0
                for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    if (site[0] + ddx, site[1] + ddy) not in occupied:
                        free += 1
                score = gain * 10.0 + free * 0.1 + rng.random() * temp
                options.append((score, site))
            
            if not options:
                success = False
                break
            
            best_score = -1e18
            best_site = None
            for s, st in options:
                if s > best_score:
                    best_score = s
                    best_site = st
            
            occupied[best_site] = i
            last = best_site
            new_suffix.append(best_site)
        
        if not success:
            temp *= cool_rate
            no_improve += 1
            stuck_count += 1
            continue

        new_chain = prefix + new_suffix
        new_energy = _compute_energy(sequence, new_chain)

        delta = new_energy - current_energy
        if delta < 0 or (temp > 0 and rng.random() < (10 ** (-delta / max(temp, 0.1)))):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                no_improve = 0
                stuck_count = 0
            else:
                no_improve += 1
                stuck_count += 1
        else:
            no_improve += 1
            stuck_count += 1

        if stuck_count > 100:
            temp *= cool_rate ** 1.5
            stuck_count = 0
        else:
            temp *= cool_rate

        if no_improve > 600:
            break

    return best_chain, best_energy


def _min_conflicts_fold(sequence, rng, max_iters=5000, grid_size=20):
    """
    Min-conflicts style search on a grid representation.
    Represent the conformation as a self-avoiding walk on a grid.
    Use random restarts and local search to minimize energy.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    best_chain = None
    best_energy = float('inf')
    
    for restart in range(10):
        if time.time() > _DEADLINE - 15:
            break
        
        # Initialize with a random self-avoiding walk
        # Start from the center of the grid
        cx, cy = grid_size // 2, grid_size // 2
        chain = [(cx, cy)]
        occupied = {(cx, cy): 0}
        
        success = True
        for i in range(1, n):
            last = chain[-1]
            x, y = last
            options = []
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                site = (x + dx, y + dy)
                if 0 <= site[0] < grid_size and 0 <= site[1] < grid_size:
                    if site not in occupied:
                        options.append(site)
            
            if not options:
                success = False
                break
            
            site = options[rng.randint(0, len(options) - 1)]
            occupied[site] = i
            chain.append(site)
        
        if not success:
            continue
        
        energy = _compute_energy(sequence, chain)
        
        # Local search: kink flips
        for _ in range(2000):
            if time.time() > _DEADLINE - 15:
                break
            
            i = rng.randint(1, n - 2)
            p = chain[i]
            prev = chain[i - 1]
            nxt = chain[i + 1]
            px, py = prev
            
            candidates = []
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                np = (px + dx, py + dy)
                if np == p:
                    continue
                if abs(np[0] - nxt[0]) + abs(np[1] - nxt[1]) != 1:
                    continue
                if np in occupied and occupied[np] != i:
                    continue
                if not (0 <= np[0] < grid_size and 0 <= np[1] < grid_size):
                    continue
                candidates.append(np)
            
            if candidates:
                np = candidates[rng.randint(0, len(candidates) - 1)]
                del occupied[p]
                chain[i] = np
                occupied[np] = i
                
                new_energy = _compute_energy(sequence, chain)
                
                if new_energy < energy:
                    energy = new_energy
                else:
                    del occupied[np]
                    chain[i] = p
                    occupied[p] = i
        
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
    
    if best_chain is None:
        # Fallback
        return _fast_grow(sequence, rng, temperature=0.0) or [[i, 0] for i in range(n)]
    
    return best_chain


def fold(sequence, dimension):
    """
    Fold a sequence into a low-energy conformation.
    """
    if dimension != 2:
        raise ValueError("Only dimension 2 is supported")

    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]

    rng = random.Random(12345 + len(sequence))

    best_chain = None
    best_energy = float('inf')

    # Phase 1: Generate diverse initial solutions with fast growth
    initial_chains = []
    
    temps = [0.0, 0.0, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    for temp in temps:
        for _ in range(12):
            if time.time() > _DEADLINE - 35:
                break
            chain = _fast_grow(sequence, rng, temperature=temp)
            if chain is not None:
                initial_chains.append(chain)
        
        if time.time() > _DEADLINE - 35:
            break

    # Evaluate all initial chains and keep the top K
    evaluated_chains = []
    for chain in initial_chains:
        energy = _compute_energy(sequence, chain)
        evaluated_chains.append((energy, chain))
    
    evaluated_chains.sort(key=lambda x: x[0])
    
    if not evaluated_chains:
        best_chain = [[i, 0] for i in range(n)]
        best_energy = _compute_energy(sequence, best_chain)
    else:
        top_k = 6
        for energy, chain in evaluated_chains[:top_k]:
            if time.time() > _DEADLINE - 30:
                break
            
            if time.time() < _DEADLINE - 25:
                improved_chain, improved_energy = _local_improve_fast(sequence, chain, rng, max_moves=4000)
                if improved_energy < energy:
                    energy = improved_energy
                    chain = improved_chain
            
            if time.time() < _DEADLINE - 20:
                sa_chain, sa_energy = _regrow_sa_fast(sequence, chain, rng, max_iters=4000, start_temp=6.0, cool_rate=0.996)
                if sa_energy < energy:
                    energy = sa_energy
                    chain = sa_chain
            
            if time.time() < _DEADLINE - 15:
                improved_chain2, improved_energy2 = _local_improve_fast(sequence, chain, rng, max_moves=2000)
                if improved_energy2 < energy:
                    energy = improved_energy2
                    chain = improved_chain2
            
            if energy < best_energy:
                best_energy = energy
                best_chain = chain

    # Phase 2: Extended SA with E2T bias on the best chain
    if time.time() < _DEADLINE - 15:
        sa_chain, sa_energy = _regrow_sa_fast(sequence, best_chain, rng, max_iters=6000, start_temp=8.0, cool_rate=0.995)
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_chain = sa_chain

    # Phase 3: Final high-precision polish with more kink iterations
    if time.time() < _DEADLINE - 5:
        for _ in range(5):
            if time.time() > _DEADLINE - 5:
                break
            final_chain, final_energy = _local_improve_fast(sequence, best_chain, rng, max_moves=5000)
            if final_energy < best_energy:
                best_energy = final_energy
                best_chain = final_chain
            else:
                break

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
