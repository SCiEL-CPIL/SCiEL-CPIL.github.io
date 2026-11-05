"""HP lattice protein folding with efficient Simulated Annealing and incremental energy updates."""
# EVOLVE-BLOCK-START
import random
import time
import math

_DEADLINE = 325.0
_START_TIME = None

def _get_deadline():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME + _DEADLINE

def _time_left():
    return _get_deadline() - time.time()

def _neighbours(point):
    x, y, z = point
    yield (x+1, y, z)
    yield (x-1, y, z)
    yield (x, y+1, z)
    yield (x, y-1, z)
    yield (x, y, z+1)
    yield (x, y, z-1)

def _lattice_adjacent(p1, p2):
    return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) + abs(p1[2]-p2[2]) == 1

class _FoldingState:
    """
    Represents the state of the folding process with incremental energy tracking.
    chain: list of tuples (x, y, z)
    occupied: dict mapping position tuple to index
    is_h: list of booleans
    contacts: set of tuples (i, j) where i < j and both are H and adjacent
    energy: -len(contacts)
    """
    def __init__(self, sequence, chain):
        self.seq = sequence
        self.n = len(chain)
        self.chain = list(chain)
        self.occupied = {p: i for i, p in enumerate(chain)}
        self.is_h = [c == 'H' for c in sequence]
        self.contacts = set()
        self._update_contacts_initial()
        self.energy = -len(self.contacts)
        self.best_energy = self.energy
        self.best_chain = list(self.chain)
        self.best_contacts = set(self.contacts)
        self.stagnation = 0

    def _update_contacts_initial(self):
        """Calculate initial contacts from scratch."""
        self.contacts.clear()
        for i in range(self.n):
            if not self.is_h[i]:
                continue
            pos = self.chain[i]
            for q in _neighbours(pos):
                j = self.occupied.get(q)
                if j is not None and j > i + 1 and self.is_h[j]:
                    self.contacts.add((i, j))

    def _snapshot(self):
        return {
            'chain': list(self.chain),
            'occupied': dict(self.occupied),
            'contacts': set(self.contacts),
            'energy': self.energy
        }

    def _restore(self, snap):
        self.chain = snap['chain']
        self.occupied = snap['occupied']
        self.contacts = snap['contacts']
        self.energy = snap['energy']

    def _accept_if_better(self):
        if self.energy < self.best_energy:
            self.best_energy = self.energy
            self.best_chain = list(self.chain)
            self.best_contacts = set(self.contacts)
            self.stagnation = 0
        else:
            self.stagnation += 1

    def _remove_contacts_involving(self, indices):
        """Remove all contacts involving any index in the set."""
        indices_set = set(indices)
        for i in indices_set:
            partners = [j for (a, b) in self.contacts if a == i or b == i for j in (a, b) if a != b]
            for j in partners:
                pair = (i, j) if i < j else (j, i)
                if pair in self.contacts:
                    self.contacts.discard(pair)
                    self.energy += 1

    def _add_new_contacts(self, indices):
        """Add new contacts for indices that are H and adjacent to other H residues."""
        indices_set = set(indices)
        for i in indices_set:
            if not self.is_h[i]:
                continue
            pos = self.chain[i]
            for q in _neighbours(pos):
                j = self.occupied.get(q)
                if j is None or j == i:
                    continue
                if abs(j - i) == 1:
                    continue
                if self.is_h[j]:
                    pair = (i, j) if i < j else (j, i)
                    if pair not in self.contacts:
                        self.contacts.add(pair)
                        self.energy -= 1

    def _update_energy_for_move(self, moved_indices):
        self._remove_contacts_involving(moved_indices)
        self._add_new_contacts(moved_indices)

    def _pivot_move(self, rng):
        """Rotate a sub-chain around a pivot point."""
        n = self.n
        if n < 4:
            return False
        # Pivot index is the point around which we rotate. 
        # We rotate the segment from pivot+1 to end.
        # Or we can rotate a segment [start, end] around a point.
        # Standard pivot: pick an index i (1 to n-2), rotate the segment i+1..n around i.
        # Actually, it's better to pick a pivot index i and rotate the segment i..n-1? No, that breaks the bond to i-1.
        # Let's rotate the segment [pivot_idx+1, n-1] around pivot_idx.
        
        pivot_idx = rng.randint(1, n - 3)
        
        # Get the vector from pivot to the next residue
        v = tuple(self.chain[pivot_idx + 1][k] - self.chain[pivot_idx][k] for k in range(3))
        
        # Find all valid rotation vectors for v in 3D lattice
        # Rotations of a unit vector in 3D:
        # If v is (1,0,0), rotations can be (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)
        # Basically, any unit vector perpendicular to v? No, any unit vector that is not v and not -v?
        # Actually, we can rotate to any of the 5 other unit directions?
        # The 6 unit directions are: (1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)
        # If v is (1,0,0), we can rotate to (0,1,0), (0,-1,0), (0,0,1), (0,0,-1).
        # We cannot rotate to (-1,0,0) because that would overlap with the previous residue (if it exists) or just be a 180 flip which might be valid if no collision.
        # Wait, a 180 flip is a valid rotation if the segment doesn't collide.
        
        possible_dirs = []
        for ax in range(3):
            for s in (-1, 1):
                d = [0, 0, 0]
                d[ax] = s
                d = tuple(d)
                if d != v:
                    possible_dirs.append(d)
        
        if not possible_dirs:
            return False
            
        new_v = possible_dirs[rng.randint(0, len(possible_dirs) - 1)]
        
        # Generate the new sub-chain
        subchain_len = n - pivot_idx - 1
        old_subchain = self.chain[pivot_idx + 1:]
        
        # We need to "unroll" the old subchain relative to the pivot, apply rotation, then "roll" it back.
        # This is complex. Easier: 
        # 1. Store the relative steps of the subchain.
        # 2. Apply a rotation matrix to the first step? No, the whole subchain rotates.
        # 
        # Alternative simpler pivot: 
        # Pick a segment [start, end]. 
        # Calculate the "shape" of the segment relative to the start.
        # Rotate the shape.
        # Check for collisions.
        
        # Let's implement a simpler "Crash" move or just use the translation/crank moves which are easier to implement correctly.
        # Pivot is hard to implement correctly with incremental updates if we don't rotate the whole coordinate system.
        
        # Let's try a simpler move: "Translate Subchain"
        return False # Placeholder, will implement Crankshaft instead

    def _crankshaft_move(self, rng):
        """
        Crankshaft move: Rotate a sub-chain around an axis defined by two non-adjacent bonds.
        This is a very powerful move for SAWs.
        """
        n = self.n
        if n < 5:
            return False
        
        # Pick a segment [start, end]
        start = rng.randint(1, n - 3)
        end = rng.randint(start + 1, n - 2)
        
        # The axis is defined by the vector from chain[start-1] to chain[start] and chain[end] to chain[end+1]?
        # No, the standard crankshaft rotates the segment [start, end] around the axis connecting the "hinges".
        # The hinges are the bonds (start-1, start) and (end, end+1).
        # The axis is the line connecting the midpoints of these bonds? Or just the vector between the fixed points?
        
        # Simplified Crankshaft:
        # 1. Identify the fixed points: p_fix1 = chain[start-1], p_fix2 = chain[end+1]
        # 2. The segment to move is chain[start:end+1]
        # 3. We want to rotate this segment such that chain[start] stays adjacent to p_fix1 and chain[end] stays adjacent to p_fix2.
        # This is geometrically complex.
        
        # Let's use a simpler "Translation" move first, then maybe "Single Residue".
        # Actually, "Pivot" is easier if we just rotate the entire tail.
        
        return False

    def _translate_move(self, rng):
        """Translate a sub-chain by one unit in a random direction."""
        n = self.n
        if n < 5:
            return False
        
        start = rng.randint(1, n - 3)
        end = rng.randint(start + 1, n - 2)
        
        # Directions
        dirs = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
        d = dirs[rng.randint(0, 5)]
        
        subchain = self.chain[start:end+1]
        
        # Check if the translated subchain is valid
        new_subchain = []
        for p in subchain:
            np = (p[0] + d[0], p[1] + d[1], p[2] + d[2])
            new_subchain.append(np)
        
        # Check for self-intersection within the new subchain
        if len(set(new_subchain)) != len(new_subchain):
            return False
            
        # Check for collision with the rest of the chain
        # The rest of the chain is chain[:start] and chain[end+1:]
        # We need to check if any p in new_subchain is in occupied, 
        # BUT we must ignore the original positions of the subchain itself.
        
        # Create a set of occupied positions, but remove the original subchain positions
        occupied_set = set(self.occupied.keys())
        for p in subchain:
            occupied_set.discard(p)
            
        for np in new_subchain:
            if np in occupied_set:
                return False
                
        # Check connectivity
        # chain[start-1] must be adjacent to new_subchain[0]
        if not _lattice_adjacent(self.chain[start-1], new_subchain[0]):
            return False
        # new_subchain[-1] must be adjacent to chain[end+1]
        if not _lattice_adjacent(new_subchain[-1], self.chain[end+1]):
            return False
            
        # Perform the move
        moved_indices = list(range(start, end + 1))
        
        # Remove old positions
        for p in subchain:
            del self.occupied[p]
            
        # Add new positions
        for i, np in enumerate(new_subchain):
            idx = start + i
            self.chain[idx] = np
            self.occupied[np] = idx
            
        # Update energy
        self._update_energy_for_move(moved_indices)
        
        return True

    def _single_residue_move(self, rng):
        """Move a single internal residue to an adjacent free site."""
        n = self.n
        if n < 3:
            return False
            
        idx = rng.randint(1, n - 2)
        pos = self.chain[idx]
        
        for new_pos in _neighbours(pos):
            if new_pos in self.occupied:
                continue
            # Check bonds
            if not _lattice_adjacent(self.chain[idx-1], new_pos):
                continue
            if not _lattice_adjacent(new_pos, self.chain[idx+1]):
                continue
            
            # Perform move
            del self.occupied[pos]
            self.chain[idx] = new_pos
            self.occupied[new_pos] = idx
            
            self._update_energy_for_move([idx])
            return True
            
        return False

def _grow_random(sequence, dimension, rng):
    """Generate a random self-avoiding walk."""
    n = len(sequence)
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site not in occupied:
                options.append(site)
        if not options:
            return None
        site = options[rng.randint(0, len(options) - 1)]
        chain.append(site)
        occupied[site] = i
        
    return chain

def _grow_greedy(sequence, dimension, rng):
    """Greedy growth maximizing H-H contacts."""
    n = len(sequence)
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    is_h = [c == 'H' for c in sequence]
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if is_h[i]:
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and is_h[j]:
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 10.0 + free * 0.1 + rng.random() * 0.5
            options.append((score, site))
            
        if not options:
            return None
            
        site = max(options)[1]
        chain.append(site)
        occupied[site] = i
        
    return chain

def _beam_grow(sequence, dimension, rng, beam_width=5):
    """Beam search growth."""
    n = len(sequence)
    if n < 4:
        return _grow_greedy(sequence, dimension, rng)
        
    is_h = [c == 'H' for c in sequence]
    start = (0, 0, 0)
    beams = [[0.0, [start], {start: 0}]]
    
    for i in range(1, n):
        if _time_left() < 2.0:
            break
        new_beams = []
        seen = set()
        
        for score, chain, occupied in beams:
            last = chain[-1]
            for site in _neighbours(last):
                if site in occupied:
                    continue
                key = tuple(chain[-3:] + [site]) if len(chain) >= 3 else tuple(chain + [site])
                if key in seen:
                    continue
                seen.add(key)
                
                gain = 0
                if is_h[i]:
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and is_h[j]:
                            gain += 1
                            
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                new_score = score + gain * 10.0 + free * 0.1 + rng.random() * 0.3
                new_chain = chain + [site]
                new_occ = dict(occupied)
                new_occ[site] = i
                new_beams.append([new_score, new_chain, new_occ])
                
        if not new_beams:
            break
            
        new_beams.sort(key=lambda x: -x[0])
        beams = new_beams[:beam_width]
        
    if not beams:
        return None
        
    best = max(beams, key=lambda x: x[0])
    return best[1]

def _energy(sequence, chain):
    """Calculate energy from scratch."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    is_h = [c == 'H' for c in sequence]
    for i, p in enumerate(chain):
        if is_h[i]:
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and is_h[j]:
                    contacts += 1
    return -contacts

def _local_search_greedy(sequence, chain, dimension, rng, max_iters=500):
    """Fast greedy local search (only accept improvements)."""
    if chain is None:
        return None
    n = len(chain)
    if n < 4:
        return chain
        
    state = _FoldingState(sequence, chain)
    
    for it in range(max_iters):
        if _time_left() < 0.5:
            break
            
        r = rng.random()
        if r < 0.5:
            state._translate_move(rng)
        else:
            state._single_residue_move(rng)
            
        state._accept_if_better()
        
    return state.best_chain

def _simulated_annealing(sequence, chain, dimension, rng, max_iters=1000):
    """Simulated Annealing."""
    if chain is None:
        return None
    n = len(chain)
    if n < 4:
        return chain
        
    state = _FoldingState(sequence, chain)
    T_start = 5.0
    T_end = 0.01
    cooling_rate = (T_end / T_start) ** (1.0 / max_iters)
    T = T_start
    
    for it in range(max_iters):
        if _time_left() < 1.0:
            break
        T *= cooling_rate
        
        old_energy = state.energy
        snap = state._snapshot()
        
        r = rng.random()
        if r < 0.6:
            state._translate_move(rng)
        else:
            state._single_residue_move(rng)
            
        delta = state.energy - old_energy
        if delta < 0 or (T > 0.01 and rng.random() < math.exp(-delta / T)):
            state._accept_if_better()
        else:
            state._restore(snap)
            
    return state.best_chain

def fold(sequence, dimension):
    """Main folding function."""
    n = len(sequence)
    rng = random.Random(12345 + n)
    best_chain = None
    best_energy = 0
    
    # Time allocation
    # We will run multiple restarts.
    # Each restart: Init + SA/LS
    # We estimate time per restart based on n.
    
    if n <= 50:
        time_per_restart = 0.05
    elif n <= 100:
        time_per_restart = 0.15
    else:
        time_per_restart = 0.3
        
    max_restarts = int(_time_left() / time_per_restart)
    max_restarts = max(10, min(max_restarts, 200))
    
    for r in range(max_restarts):
        if _time_left() < 5.0:
            break
            
        # Strategy selection
        strategy = r % 3
        
        if strategy == 0:
            # Beam search + SA
            chain = _beam_grow(sequence, dimension, rng, beam_width=5)
            if chain is None:
                chain = _grow_greedy(sequence, dimension, rng)
            if chain is None:
                continue
            max_sa = 500 if n <= 50 else (200 if n <= 100 else 100)
            chain = _simulated_annealing(sequence, chain, dimension, rng, max_iters=max_sa)
            
        elif strategy == 1:
            # Greedy + SA
            chain = _grow_greedy(sequence, dimension, rng)
            if chain is None:
                continue
            max_sa = 400 if n <= 50 else (150 if n <= 100 else 80)
            chain = _simulated_annealing(sequence, chain, dimension, rng, max_iters=max_sa)
            
        else:
            # Random + SA (more diversity)
            chain = _grow_random(sequence, dimension, rng)
            if chain is None:
                continue
            max_sa = 300 if n <= 50 else (100 if n <= 100 else 50)
            chain = _simulated_annealing(sequence, chain, dimension, rng, max_iters=max_sa)
            
        if chain is None:
            continue
            
        value = _energy(sequence, chain)
        if value < best_energy:
            best_energy = value
            best_chain = chain
            
        # Polishing step for the current best
        if _time_left() > 1.0:
            polished = _local_search_greedy(sequence, best_chain, dimension, rng, max_iters=100)
            if polished:
                p_val = _energy(sequence, polished)
                if p_val < best_energy:
                    best_energy = p_val
                    best_chain = polished

    if best_chain is None:
        # Fallback: linear chain
        best_chain = [(i, 0, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
