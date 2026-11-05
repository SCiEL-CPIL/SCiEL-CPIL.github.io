"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

_DEADLINE = 310.0
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
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _compute_contacts(sequence, chain):
    """Compute set of H-H contact pairs (i, j) with i < j."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = set()
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts.add((i, j))
    return contacts

def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _rotation_matrices_3d():
    """Generate all 24 rotation matrices for 3D lattice (proper rotations)."""
    import itertools
    matrices = []
    # All 24 proper rotation matrices for cubic lattice
    # These are orthogonal matrices with determinant +1 and integer entries
    # Generated from all permutations of axes with sign flips that give det=+1
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            for row in range(3):
                M[row][perm[row]] = signs[row]
            # Check determinant
            det = (M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1]) -
                   M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0]) +
                   M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]))
            if det == 1:
                matrices.append(tuple(tuple(row) for row in M))
    return matrices

_ROT_MATRICES = _rotation_matrices_3d()

def _apply_rotation(point, matrix):
    """Apply 3x3 rotation matrix to a 3D point."""
    x, y, z = point
    return (matrix[0][0]*x + matrix[0][1]*y + matrix[0][2]*z,
            matrix[1][0]*x + matrix[1][1]*y + matrix[1][2]*z,
            matrix[2][0]*x + matrix[2][1]*y + matrix[2][2]*z)

class _FoldingState:
    """State for incremental energy tracking during local search."""
    def __init__(self, sequence, chain, dimension):
        self.sequence = sequence
        self.chain = list(chain)
        self.dimension = dimension
        self.occupied = {p: i for i, p in enumerate(chain)}
        self.contacts = _compute_contacts(sequence, chain)
        self.energy = -len(self.contacts)
        self.n = len(chain)
    
    def _is_h(self, i):
        return self.sequence[i] == "H"
    
    def _remove_contact_if_exists(self, i, j):
        if i > j:
            i, j = j, i
        if (i, j) in self.contacts:
            self.contacts.remove((i, j))
            self.energy += 1
    
    def _add_contact_if_valid(self, i, j):
        if i > j:
            i, j = j, i
        if i + 1 < j and self._is_h(i) and self._is_h(j):
            self.contacts.add((i, j))
            self.energy -= 1
    
    def _get_contacts_for_residue(self, i):
        """Get all residues j that are in contact with residue i."""
        p = self.chain[i]
        result = []
        for q in _neighbours(p):
            j = self.occupied.get(q)
            if j is not None and j != i and abs(j - i) > 1:
                result.append(j)
        return result
    
    def _remove_residue_from_contacts(self, i):
        """Remove all contacts involving residue i from contact set."""
        p = self.chain[i]
        for q in _neighbours(p):
            j = self.occupied.get(q)
            if j is not None and j != i and abs(j - i) > 1:
                self._remove_contact_if_exists(i, j)
    
    def _add_residue_to_contacts(self, i):
        """Add all valid contacts for residue i to contact set."""
        p = self.chain[i]
        for q in _neighbours(p):
            j = self.occupied.get(q)
            if j is not None and j != i and abs(j - i) > 1:
                self._add_contact_if_valid(i, j)
    
    def _pivot_rotate(self, pivot_idx, length, rotation_matrix):
        """
        Rotate the subchain from pivot_idx to pivot_idx+length-1 around the bond
        at pivot_idx. Returns (valid, energy_delta).
        """
        if self.n < 5:
            return False, 0
        
        # The pivot bond is between chain[pivot_idx-1] and chain[pivot_idx]
        # We rotate chain[pivot_idx ... pivot_idx+length-1] around that bond axis
        
        if pivot_idx == 0 or pivot_idx + length > self.n:
            return False, 0
        
        # Get the bond direction vector
        if pivot_idx > 0:
            bond_vec = tuple(self.chain[pivot_idx][a] - self.chain[pivot_idx-1][a] for a in range(self.dimension))
        else:
            bond_vec = tuple(self.chain[pivot_idx+1][a] - self.chain[pivot_idx][a] for a in range(self.dimension))
        
        # For a proper pivot in 3D, we rotate around the bond axis
        # The subchain to rotate is chain[pivot_idx : pivot_idx+length]
        # We need to express each point relative to the pivot, rotate, and reattach
        
        pivot_point = self.chain[pivot_idx]
        subchain_indices = list(range(pivot_idx, pivot_idx + length))
        
        # Save old positions
        old_positions = {}
        for i in subchain_indices:
            old_positions[i] = self.chain[i]
        
        # Remove old positions from occupied
        for i in subchain_indices:
            p = self.chain[i]
            if p in self.occupied:
                del self.occupied[p]
        
        # Remove contacts involving subchain residues
        for i in subchain_indices:
            self._remove_residue_from_contacts(i)
        
        # Compute new positions by rotating relative to pivot
        new_positions = {}
        valid = True
        
        for i in subchain_indices:
            # Vector from pivot to this point
            rel = tuple(self.chain[i][a] - pivot_point[a] for a in range(self.dimension))
            # Apply rotation
            rotated = _apply_rotation(rel, rotation_matrix)
            # New absolute position
            new_pos = tuple(pivot_point[a] + rotated[a] for a in range(self.dimension))
            
            # Check if this new position is valid
            if new_pos in self.occupied:
                valid = False
                break
            # Check if this new position collides with any other new position
            if new_pos in new_positions:
                valid = False
                break
            
            new_positions[i] = new_pos
        
        if not valid:
            # Restore
            for i in subchain_indices:
                self.chain[i] = old_positions[i]
                self.occupied[self.chain[i]] = i
            for i in subchain_indices:
                self._add_residue_to_contacts(i)
            return False, 0
        
        # Check bond connectivity at the pivot
        if pivot_idx > 0:
            prev = self.chain[pivot_idx - 1]
            dist = sum(abs(a - b) for a, b in zip(prev, new_positions[pivot_idx]))
            if dist != 1:
                # Restore
                for i in subchain_indices:
                    self.chain[i] = old_positions[i]
                    self.occupied[self.chain[i]] = i
                for i in subchain_indices:
                    self._add_residue_to_contacts(i)
                return False, 0
        
        # Check bond connectivity at the end of subchain
        if pivot_idx + length < self.n:
            next_p = self.chain[pivot_idx + length]
            last_idx = pivot_idx + length - 1
            dist = sum(abs(a - b) for a, b in zip(new_positions[last_idx], next_p))
            if dist != 1:
                # Restore
                for i in subchain_indices:
                    self.chain[i] = old_positions[i]
                    self.occupied[self.chain[i]] = i
                for i in subchain_indices:
                    self._add_residue_to_contacts(i)
                return False, 0
        
        # Apply the move
        for i in subchain_indices:
            self.chain[i] = new_positions[i]
            self.occupied[new_positions[i]] = i
        
        # Add new contacts
        for i in subchain_indices:
            self._add_residue_to_contacts(i)
        
        return True, 0

def _simulated_annealing(sequence, chain, dimension, rng, max_iters=5000):
    """Simulated annealing with true 3D pivot rotations and incremental energy tracking."""
    state = _FoldingState(sequence, chain, dimension)
    n = state.n
    
    if n < 5:
        return state.chain
    
    best_chain = list(state.chain)
    best_energy = state.energy
    
    T = 5.0
    T_min = 0.1
    cooling_rate = 0.9995
    no_improve = 0
    
    for iteration in range(max_iters):
        if _time_left() < 3.0:
            break
        
        # Choose pivot and length
        pivot_idx = rng.randint(1, n - 2)
        max_len = n - pivot_idx - 1
        if max_len < 1:
            continue
        length = rng.randint(1, max_len)
        
        # Choose a random rotation matrix
        rot_matrix = _ROT_MATRICES[rng.randint(0, len(_ROT_MATRICES) - 1)]
        
        # Try the pivot rotation
        valid, _ = state._pivot_rotate(pivot_idx, length, rot_matrix)
        
        if not valid:
            no_improve += 1
            if no_improve > 2000:
                # Restart temperature
                T = max(T_min, T * 0.5)
                no_improve = 0
            continue
        
        new_energy = state.energy
        delta = new_energy - state.energy
        
        # Accept or reject
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            # Accept
            state.energy = new_energy
            no_improve = 0
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(state.chain)
        else:
            # Reject - need to undo the move
            # We need to reverse the rotation
            # Find the inverse rotation
            inv_matrix = None
            for rm in _ROT_MATRICES:
                # Check if rm is inverse of rot_matrix
                is_inv = True
                for a in range(3):
                    for b in range(3):
                        s = sum(rot_matrix[a][k] * rm[k][b] for k in range(3))
                        if s != (1 if a == b else 0):
                            is_inv = False
                            break
                    if not is_inv:
                        break
                if is_inv:
                    inv_matrix = rm
                    break
            
            if inv_matrix is not None:
                # Apply inverse rotation
                state._pivot_rotate(pivot_idx, length, inv_matrix)
            else:
                # Should not happen for proper rotations
                pass
        
        # Cool down
        T = max(T_min, T * cooling_rate)
    
    return best_chain

def fold(sequence, dimension):
    """Best conformation using greedy growth with simulated annealing refinement."""
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 0
    
    n = len(sequence)
    
    # Phase 1: Greedy growth with multiple restarts
    n_restarts = min(50, max(5, int(60.0 / max(1, n * 0.1))))
    for r in range(n_restarts):
        if _time_left() < 10.0:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if best is None or value < best_energy:
            best, best_energy = chain, value
    
    # Phase 2: Simulated annealing on the best greedy result
    if best is not None and _time_left() > 10.0:
        time_budget = _time_left() - 5.0
        # Estimate iterations based on chain length
        est_iters_per_sec = max(100, 50000 // n)
        max_sa_iters = int(time_budget * est_iters_per_sec)
        max_sa_iters = max(1000, min(max_sa_iters, 50000))
        
        refined = _simulated_annealing(sequence, best, dimension, rng, max_iters=max_sa_iters)
        if refined is not None:
            refined_energy = _energy(sequence, refined)
            if refined_energy < best_energy:
                best, best_energy = refined, refined_energy
    
    # Fallback: straight line is always self-avoiding
    if best is None:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
