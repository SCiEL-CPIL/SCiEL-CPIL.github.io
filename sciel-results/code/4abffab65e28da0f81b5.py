import random
import time

_DEADLINE = None
_BUDGET_FRACTION = 0.90

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 330 * _BUDGET_FRACTION
    return _DEADLINE

def _is_deadline_exceeded():
    return time.time() > _get_deadline()

_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

def _greedy_saw(n):
    """Generate a random self-avoiding walk of length n."""
    if n == 0:
        return []
    chain = [(0, 0, 0)]
    occ = {chain[0]}
    
    def backtrack(i):
        if i == n:
            return True
        last = chain[-1]
        moves = []
        for dx, dy, dz in _OFFSETS:
            np_ = (last[0] + dx, last[1] + dy, last[2] + dz)
            if np_ not in occ:
                moves.append(np_)
        random.shuffle(moves)
        for np_ in moves:
            chain.append(np_)
            occ.add(np_)
            if backtrack(i + 1):
                return True
            chain.pop()
            occ.remove(np_)
        return False
        
    if not backtrack(1):
        return None
    return chain

def _calc_delta_energy_pivot(chain, sequence, i, axis, new_seg, old_seg):
    """
    Calculate delta energy for a pivot move at index i.
    old_seg: list of points for indices i+1..n-1 (old positions)
    new_seg: list of points for indices i+1..n-1 (new positions)
    Fixed part: indices 0..i
    """
    n = len(chain)
    fixed_indices = list(range(0, i + 1))
    moved_indices = list(range(i + 1, n))
    
    # Build pos_to_idx for fixed part for quick lookup?
    # Or just use set of fixed positions.
    fixed_pos_set = set(chain[j] for j in fixed_indices)
    
    # We need to count H-H contacts between moved and fixed.
    # Contact exists if dist(p_moved, p_fixed) == 1 and both are H.
    # And they are not adjacent in chain.
    # Adjacent pairs are (i, i+1).
    # So when checking moved index k=i+1, we must exclude fixed index j=i.
    
    def count_contacts(pos_list, idx_list, other_pos_set, other_idx_list, exclude_pair=None):
        count = 0
        for k, p in zip(idx_list, pos_list):
            if sequence[k] != 'H':
                continue
            for dx, dy, dz in _OFFSETS:
                np_ = (p[0] + dx, p[1] + dy, p[2] + dz)
                if np_ in other_pos_set:
                    # Find index of np_ in other_idx_list
                    # This is O(n) if done naively.
                    # We need a map pos->idx for the other set.
                    pass
        return count
        
    # To make this fast, we need pos->idx maps.
    # But we are calling this inside the loop.
    # Let's precompute maps in the fold function and pass them?
    # No, the chain changes.
    
    # Alternative: Just calculate full energy for new chain.
    # Given n<=136, O(n^2) is ~20k ops.
    # If we do 1000 steps, 20M ops. ~2s.
    # If we do 10000 steps, 200M ops. ~20s.
    # This is acceptable for 26 sequences if we are careful.
    
    # I will stick to full energy recalculation for simplicity and correctness,
    # but optimize the energy function.
    
    pass

def _calc_energy(chain, sequence):
    """Calculate energy of a chain."""
    if not chain:
        return 0
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    energy = 0
    for i, p in enumerate(chain):
        if sequence[i] != 'H':
            continue
        for dx, dy, dz in _OFFSETS:
            neighbor = (p[0] + dx, p[1] + dy, p[2] + dz)
            if neighbor in pos_to_idx:
                j = pos_to_idx[neighbor]
                if abs(i - j) > 1:
                    if sequence[j] == 'H':
                        energy -= 1
    return energy // 2

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    
    chain = _greedy_saw(n)
    if chain is None:
        return [list(p) for p in [(0,0,0)]*n]
        
    current_energy = _calc_energy(chain, sequence)
    best_chain = list(chain)
    best_energy = current_energy
    
    # SA Parameters
    T_start = 1.0
    T_end = 0.01
    
    start_time = time.time()
    step = 0
    
    while not _is_deadline_exceeded():
        # Propose pivot move
        if n < 3:
            break
            
        i = random.randint(0, n - 2)
        
        p0 = chain[i]
        p1 = chain[i+1]
        
        # Find perpendicular axis
        v = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
        axes = []
        for dx, dy, dz in _OFFSETS:
            if v[0]*dx + v[1]*dy + v[2]*dz == 0:
                axes.append((dx, dy, dz))
                
        if not axes:
            continue
            
        axis = random.choice(axes)
        
        def rotate90(p, origin, axis):
            x, y, z = p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]
            ax, ay, az = axis
            if ax == 1: nx, ny, nz = x, -z, y
            elif ax == -1: nx, ny, nz = x, z, -y
            elif ay == 1: nx, ny, nz = -z, y, x
            elif ay == -1: nx, ny, nz = z, y, -x
            elif az == 1: nx, ny, nz = y, -x, z
            elif az == -1: nx, ny, nz = -y, x, z
            else: return None
            return (origin[0] + nx, origin[1] + ny, origin[2] + nz)
            
        new_seg = []
        valid = True
        new_occ = set(chain[:i+1]) # Fixed part + pivot
        
        for k in range(i+1, n):
            qk = rotate90(chain[k], p0, axis)
            if qk is None or qk in new_occ:
                valid = False
                break
            new_occ.add(qk)
            new_seg.append(qk)
            
        if not valid:
            continue
            
        new_chain = chain[:i+1] + new_seg
        
        new_energy = _calc_energy(new_chain, sequence)
        delta_e = new_energy - current_energy
        
        # Cooling
        # Linear cooling over estimated steps
        # Let's say 5000 steps
        step += 1
        T = T_start * (T_end / T_start) ** (step / 5000.0)
        
        if delta_e < 0 or random.random() < pow(2.718, -delta_e / T):
            chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                
        # Small chance to restart if stuck?
        # No, keep it simple.
        
    return [list(p) for p in best_chain]
