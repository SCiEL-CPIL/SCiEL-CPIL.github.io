"""HP lattice protein folding on the cubic lattice - pivot Monte Carlo with incremental energy."""
import time
import random
import math

# EVOLVE-BLOCK-START
import time as _time

# Precompute pivot rotation matrices for 3D
# A pivot at position k rotates the subchain k+1..n-1 around the bond (k-1, k)
# We use quaternion-free approach: for each pivot, we rotate the tail by
# applying a rotation matrix to all tail coordinates relative to pivot point.

# 3D rotation matrices for 90-degree rotations around each axis
# Rotation around x-axis by 90 deg: (x,y,z) -> (x, -z, y)
# Rotation around x-axis by -90 deg: (x,y,z) -> (x, z, -y)
# Rotation around y-axis by 90 deg: (x,y,z) -> (z, y, -x)
# Rotation around y-axis by -90 deg: (x,y,z) -> (-z, y, x)
# Rotation around z-axis by 90 deg: (x,y,z) -> (-y, x, z)
# Rotation around z-axis by -90 deg: (x,y,z) -> (y, -x, z)

_ROTATIONS_3D = [
    lambda p: (p[0], -p[2], p[1]),   # Rx+90
    lambda p: (p[0], p[2], -p[1]),    # Rx-90
    lambda p: (p[2], p[1], -p[0]),    # Ry+90
    lambda p: (-p[2], p[1], p[0]),    # Ry-90
    lambda p: (-p[1], p[0], p[2]),    # Rz+90
    lambda p: (p[1], -p[0], p[2]),    # Rz-90
]

_NEIGHBOR_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
]

def _init_chain(sequence, rng):
    """Initialize a random self-avoiding walk (greedy with randomization)."""
    n = len(sequence)
    if n <= 1:
        return [(0, 0, 0)]
    
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0)}
    rng.shuffle(_NEIGHBOR_OFFSETS)
    
    for i in range(1, n):
        last = chain[-1]
        placed = False
        # Try random neighbors, prefer ones that don't trap us
        attempts = list(range(len(_NEIGHBOR_OFFSETS)))
        rng.shuffle(attempts)
        for idx in attempts:
            d = _NEIGHBOR_OFFSETS[idx]
            new_pt = (last[0] + d[0], last[1] + d[1], last[2] + d[2])
            if new_pt not in occupied:
                # Check we won't immediately trap ourselves (simple heuristic)
                chain.append(new_pt)
                occupied.add(new_pt)
                placed = True
                break
        if not placed:
            # Backtrack: try to restart from a random earlier point
            # For simplicity, just return what we have (will be fixed by MC)
            return chain
    return chain


def _compute_contacts(sequence, chain):
    """Compute the set of H-H contact pairs (i, j) with i < j, |i-j| > 1."""
    n = len(chain)
    pos_to_idx = {chain[i]: i for i in range(n)}
    contacts = set()
    for i in range(n):
        if sequence[i] != 'H':
            continue
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            neighbor = (chain[i][0] + dx, chain[i][1] + dy, chain[i][2] + dz)
            j = pos_to_idx.get(neighbor)
            if j is not None and j > i + 1:
                contacts.add((i, j))
    return contacts


def _energy_from_contacts(contacts):
    return -len(contacts)


def _pivot_move(chain, k, rot_fn, sequence):
    """
    Apply a pivot rotation at position k (rotating chain[k+1:] around chain[k]).
    Returns (new_chain, added_contacts, removed_contacts) or None if self-intersection.
    k is the pivot bond index: rotate subchain starting at index k+1.
    Valid k: 1 <= k <= n-2 (so that k-1, k, k+1 exist)
    """
    n = len(chain)
    if k < 1 or k >= n - 1:
        return None
    
    pivot_point = chain[k]
    # Relative coordinates of tail
    tail_rel = [(chain[i][0] - pivot_point[0], chain[i][1] - pivot_point[1], chain[i][2] - pivot_point[2]) 
                for i in range(k + 1, n)]
    
    # Apply rotation
    new_tail = []
    for rx, ry, rz in tail_rel:
        rot_pt = rot_fn((rx, ry, rz))
        new_tail.append((pivot_point[0] + rot_pt[0], pivot_point[1] + rot_pt[1], pivot_point[2] + rot_pt[2]))
    
    # Check self-avoidance: new tail points must not collide with head (chain[0..k])
    # and must not collide with each other
    head_set = set(chain[:k + 1])
    new_tail_set = set()
    for pt in new_tail:
        if pt in head_set:
            return None
        if pt in new_tail_set:
            return None
        new_tail_set.add(pt)
    
    new_chain = chain[:k + 1] + new_tail
    return new_chain


def _delta_energy(sequence, old_chain, new_chain, old_contacts, k):
    """
    Compute energy delta after a pivot at position k.
    Only contacts involving residues in the moved tail (indices > k) can change.
    Head residues (<= k) keep their positions.
    """
    n = len(old_chain)
    tail_indices = set(range(k + 1, n))
    
    # Contacts that are removed: those in old_contacts where at least one index is in tail
    # and the pair no longer exists in new_chain
    # Contacts that are added: new H-H contacts where at least one index is in tail
    
    # Build position-to-index for new chain
    new_pos_to_idx = {new_chain[i]: i for i in range(n)}
    
    removed = set()
    added = set()
    
    # Check which old contacts are broken
    for (i, j) in old_contacts:
        if i in tail_indices or j in tail_indices:
            # Check if still a contact in new chain
            pi = new_chain[i]
            pj = new_chain[j]
            dx, dy, dz = abs(pi[0]-pj[0]), abs(pi[1]-pj[1]), abs(pi[2]-pj[2])
            if dx + dy + dz == 1:
                # Still adjacent
                pass
            else:
                removed.add((i, j))
    
    # Find new contacts involving tail residues
    for i in tail_indices:
        if sequence[i] != 'H':
            continue
        pi = new_chain[i]
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            neighbor = (pi[0] + dx, pi[1] + dy, pi[2] + dz)
            j = new_pos_to_idx.get(neighbor)
            if j is not None and j > i + 1:
                pair = (i, j)
                if pair not in removed:
                    added.add(pair)
            elif j is not None and j < i - 1:
                pair = (j, i)
                if pair not in removed:
                    added.add(pair)
    
    delta = len(added) - len(removed)
    return delta, added, removed


def _pivot_mc_fold(sequence, n_restarts, time_budget, rng):
    """
    Pivot Monte Carlo with simulated annealing for a single sequence.
    Returns the best chain found.
    """
    n = len(sequence)
    if n <= 2:
        return [(i, 0, 0) for i in range(n)]
    
    best_chain = None
    best_energy = float('inf')
    
    start_time = _time.time()
    total_deadline = start_time + time_budget
    
    for restart in range(n_restarts):
        if _time.time() > total_deadline - 5:
            break
        
        # Initialize
        chain = _init_chain(sequence, rng)
        if len(chain) < n:
            # Pad with a straight line extension if init failed
            # Actually, let's just use a simple spiral initialization
            chain = [(i, 0, 0) for i in range(n)]
        
        contacts = _compute_contacts(sequence, chain)
        current_energy = _energy_from_contacts(contacts)
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_chain = list(chain)
        
        # Simulated annealing
        T_start = max(1.0, abs(current_energy) * 0.5)
        T_end = 0.01
        # Number of MC steps per restart - adapt based on chain length
        n_steps = max(500, min(5000, n * 50))
        
        for step in range(n_steps):
            if _time.time() > total_deadline - 10:
                break
            
            # Temperature schedule (geometric cooling)
            if n_steps > 1:
                T = T_start * (T_end / T_start) ** (step / n_steps)
            else:
                T = T_start
            
            # Choose pivot position (biased toward middle)
            k = rng.randint(1, n - 2)
            
            # Choose rotation
            rot_fn = _ROTATIONS_3D[rng.randint(0, 5)]
            
            result = _pivot_move(chain, k, rot_fn, sequence)
            if result is None:
                continue
            
            new_chain = result
            delta, added, removed = _delta_energy(sequence, chain, new_chain, contacts, k)
            
            if delta <= 0:
                # Accept (improvement or equal)
                chain = new_chain
                current_energy += delta
                # Update contacts set
                contacts -= removed
                contacts |= added
            else:
                # Accept with probability exp(-delta/T)
                if T > 0 and rng.random() < math.exp(-delta / T):
                    chain = new_chain
                    current_energy += delta
                    contacts -= removed
                    contacts |= added
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        
        # Local improvement: greedy hill climbing at end of each restart
        improved = True
        while improved and _time.time() < total_deadline - 10:
            improved = False
            for k in range(1, n - 1):
                for rot_fn in _ROTATIONS_3D:
                    result = _pivot_move(chain, k, rot_fn, sequence)
                    if result is None:
                        continue
                    new_chain = result
                    delta, added, removed = _delta_energy(sequence, chain, new_chain, contacts, k)
                    if delta < 0:
                        chain = new_chain
                        current_energy += delta
                        contacts -= removed
                        contacts |= added
                        improved = True
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(chain)
    
    return best_chain if best_chain else [(i, 0, 0) for i in range(n)]


def _simple_spiral(sequence):
    """Generate a simple spiral conformation as fallback."""
    n = len(sequence)
    chain = [(0, 0, 0)]
    # Simple L-shaped then spiral pattern
    directions = [(1,0,0), (0,1,0), (-1,0,0), (0,-1,0)]
    dir_idx = 0
    for i in range(1, n):
        d = directions[dir_idx % 4]
        new_pt = (chain[-1][0] + d[0], chain[-1][1] + d[1], chain[-1][2] + d[2])
        # Avoid immediate backtracking
        if len(chain) >= 2:
            prev_dir = (chain[-1][0] - chain[-2][0], chain[-1][1] - chain[-2][1], chain[-1][2] - chain[-2][2])
            if (new_pt[0] - chain[-1][0], new_pt[1] - chain[-1][1], new_pt[2] - chain[-1][2]) == (-prev_dir[0], -prev_dir[1], -prev_dir[2]):
                # Would backtrack, turn instead
                dir_idx += 1
                d = directions[dir_idx % 4]
                new_pt = (chain[-1][0] + d[0], chain[-1][1] + d[1], chain[-1][2] + d[2])
        if len(chain) >= 3 and new_pt == chain[-2]:
            dir_idx += 1
            d = directions[dir_idx % 4]
            new_pt = (chain[-1][0] + d[0], chain[-1][1] + d[1], chain[-1][2] + d[2])
        chain.append(new_pt)
    # Check self-avoidance, if violated, fall back to straight line
    if len(set(chain)) != n:
        chain = [(i, 0, 0) for i in range(n)]
    return chain


def fold(sequence, dimension):
    """
    Fold an HP sequence on the simple cubic lattice.
    Returns a list of len(sequence) lattice points (lists of length dimension).
    """
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    # Deadline: use ~280 seconds total across all calls (safety margin from 330s)
    # Since fold is called once per sequence, we allocate time per call
    # But we don't know how many calls total, so use a per-call budget
    # The task says 240s CPU for all sequences, 300s wall clock per evaluation
    # We'll use a global deadline approach
    
    global _FOLD_START_TIME
    if not hasattr(fold, '_start_time'):
        fold._start_time = _time.time()
    
    # Per-call time budget: assume up to 26 sequences, so ~10s each
    # But be more generous for shorter chains
    elapsed = _time.time() - fold._start_time
    # Allocate remaining time / estimated remaining sequences
    # We'll just use a fixed per-call budget with a global cap
    per_call_budget = min(25.0, max(5.0, 280.0 - elapsed))
    
    # Number of restarts based on chain length and time budget
    # Shorter chains: more restarts; longer chains: fewer but more steps each
    if n <= 48:
        n_restarts = max(3, int(per_call_budget / 3.0))
    elif n <= 80:
        n_restarts = max(2, int(per_call_budget / 5.0))
    else:
        n_restarts = max(1, int(per_call_budget / 8.0))
    
    rng = random.Random(42 + n * 7)
    
    best_chain = _pivot_mc_fold(sequence, n_restarts, per_call_budget, rng)
    
    # Ensure self-avoidance
    if len(set(best_chain)) != n:
        best_chain = _simple_spiral(sequence)
    
    # Ensure unit steps between consecutive residues
    for i in range(1, n):
        d = abs(best_chain[i][0] - best_chain[i-1][0]) + \
            abs(best_chain[i][1] - best_chain[i-1][1]) + \
            abs(best_chain[i][2] - best_chain[i-1][2])
        if d != 1:
            best_chain = _simple_spiral(sequence)
            break
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
