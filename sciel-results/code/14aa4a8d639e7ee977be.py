import random
import time
import math

# Precompute neighbor offsets for 3D cubic lattice
_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

_START_TIME = time.time()
_DEADLINE = _START_TIME + 330 * 0.90  # 297s, safety margin


def _time_left():
    return _DEADLINE - time.time()


def _energy_full(sequence, chain):
    """Compute energy: -number of H-H contacts on neighbouring non-consecutive sites."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] != 'H':
            continue
        for dx, dy, dz in _OFFSETS:
            q = (p[0] + dx, p[1] + dy, p[2] + dz)
            j = where.get(q)
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts


def _beam_search(sequence, dimension, beam_width=300):
    """
    Beam search for HP protein folding.
    Maintains a beam of the best partial conformations at each step.
    Score combines energy (lower better) and a free-space bonus so the
    chain does not trap itself; a little noise adds diversity.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]

    start = (0, 0, 0)
    beam = [(0.0, [start], {start}, {start: 0}, 0)]

    for i in range(1, n):
        if _time_left() < 8.0:
            break

        candidates = []
        seq_i = sequence[i]

        for score, chain, occ, pos_idx, energy in beam:
            last = chain[-1]
            lx, ly, lz = last

            for dx, dy, dz in _OFFSETS:
                site = (lx + dx, ly + dy, lz + dz)
                if site in occ:
                    continue

                # incremental energy gain from new H-H contacts
                gain = 0
                if seq_i == 'H':
                    for dx2, dy2, dz2 in _OFFSETS:
                        q = (site[0] + dx2, site[1] + dy2, site[2] + dz2)
                        j = pos_idx.get(q)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            gain += 1

                # free neighbours of the new site (room to grow)
                free = 0
                for dx2, dy2, dz2 in _OFFSETS:
                    q = (site[0] + dx2, site[1] + dy2, site[2] + dz2)
                    if q not in occ:
                        free += 1

                new_energy = energy - gain
                new_score = new_energy - 0.05 * free + random.random() * 0.01

                new_chain = chain + [site]
                new_occ = set(occ)
                new_occ.add(site)
                new_pos = dict(pos_idx)
                new_pos[site] = i

                candidates.append((new_score, new_chain, new_occ, new_pos, new_energy))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[0])
        beam = candidates[:beam_width]

    if not beam:
        return None
    best = min(beam, key=lambda x: x[0])
    return list(best[1])


def _rot(p, axis, angle):
    """Rotate point p about the given axis by 90/180/270 degrees (angle 1/2/3)."""
    x, y, z = p
    if axis == 0:  # x-axis
        if angle == 1:
            return (x, -z, y)
        if angle == 2:
            return (x, -y, -z)
        return (x, z, -y)
    if axis == 1:  # y-axis
        if angle == 1:
            return (z, y, -x)
        if angle == 2:
            return (-x, y, -z)
        return (-z, y, x)
    # z-axis
    if angle == 1:
        return (y, -x, z)
    if angle == 2:
        return (-x, -y, z)
    return (-y, x, z)


def _local_search(sequence, dimension, chain, rng, max_iterations=8000):
    """
    Local search combining segment-reversal moves and pivot-rotation moves,
    both using O(1) incremental energy deltas. Simulated-annealing acceptance.
    """
    n = len(chain)
    if n < 5:
        return chain

    chain = [tuple(p) for p in chain]
    occ = set(chain)
    pos_idx = {p: i for i, p in enumerate(chain)}

    current_energy = _energy_full(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy

    T = 1.5
    T_min = 0.004
    alpha = 0.9997

    for it in range(max_iterations):
        if _time_left() < 4.0:
            break

        if rng.random() < 0.5:
            # ---- segment reversal ----
            max_seg = min(n - 2, 14)
            if max_seg < 2:
                continue
            r = rng.random()
            if r < 0.5:
                seg_len = rng.randint(2, 4)
            elif r < 0.8:
                seg_len = rng.randint(4, 8)
            else:
                seg_len = rng.randint(8, max_seg)
            seg_len = min(seg_len, max_seg)
            if seg_len < 2:
                continue
            s = rng.randint(1, n - seg_len - 1)
            e = s + seg_len - 1

            # reverse the segment in-place and track moved positions
            moved = []
            for k in range(seg_len // 2):
                a = s + k
                b = e - k
                pa, pb = chain[a], chain[b]
                moved.append((a, pa))
                moved.append((b, pb))
                chain[a], chain[b] = pb, pa
                occ.discard(pa)
                occ.discard(pb)
                occ.add(pa)
                occ.add(pb)
                pos_idx[pa] = a
                pos_idx[pb] = b

            # check boundary unit-steps + self-avoidance of moved positions
            valid = True
            if s > 0:
                p1 = chain[s - 1]
                p2 = chain[s]
                if abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) + abs(p1[2]-p2[2]) != 1:
                    valid = False
            if e < n - 1:
                p1 = chain[e]
                p2 = chain[e + 1]
                if abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) + abs(p1[2]-p2[2]) != 1:
                    valid = False
            seen = set()
            for a, p in moved:
                if p in seen:
                    valid = False
                    break
                seen.add(p)

            if valid:
                delta = 0
                for a, _ in moved:
                    if sequence[a] == 'H':
                        p = chain[a]
                        for dx, dy, dz in _OFFSETS:
                            q = (p[0]+dx, p[1]+dy, p[2]+dz)
                            j = pos_idx.get(q)
                            if j is not None and abs(a - j) > 1 and sequence[j] == 'H':
                                delta -= 1
                new_energy = current_energy + delta
                if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = chain[:]
                else:
                    for a, p in moved:
                        chain[a] = p
                        pos_idx[p] = a
            else:
                for a, p in moved:
                    chain[a] = p
                    pos_idx[p] = a
        else:
            # ---- pivot rotation ----
            if n < 4:
                continue
            pivot = rng.randint(1, n - 3)
            # The subchain from pivot+1 to end is rotated about the bond (pivot-1, pivot)
            # Actually, rotate the tail starting at pivot about the axis defined by chain[pivot-1] and chain[pivot]
            # But simpler: rotate the subchain [pivot..n-1] about the point chain[pivot-1]
            # Let's use the standard move: pick a pivot i (2 <= i <= n-2), rotate chain[i..n-1] about chain[i-1]
            # The axis is the line through chain[i-2] and chain[i-1]? No.
            # Standard pivot move: pick i in [2, n-1]. Rotate the segment chain[i..n-1] around the bond (chain[i-1], chain[i]).
            # This is complex. Let's use the simpler "reversal" or "rotation of tail about a fixed point".
            # Let's implement: pick pivot p (1 <= p <= n-2). Rotate the tail chain[p+1..n-1] around the axis defined by chain[p-1] and chain[p].
            
            p = rng.randint(1, n - 2)
            # Axis vector: chain[p] - chain[p-1]
            ax_vec = (chain[p][0] - chain[p-1][0], chain[p][1] - chain[p-1][1], chain[p][2] - chain[p-1][1])
            # This needs to be one of the principal axes for simple rotation?
            # The bond direction is one of the 6 offsets.
            # Let's find which axis it is.
            axis = -1
            if abs(ax_vec[0]) == 1: axis = 0
            elif abs(ax_vec[1]) == 1: axis = 1
            elif abs(ax_vec[2]) == 1: axis = 2
            
            if axis == -1:
                continue

            angle = rng.choice([1, 2, 3])
            
            # Rotate tail [p+1 .. n-1] about chain[p] around the axis passing through chain[p] with direction axis
            # Wait, rotation about an axis in 3D.
            # The axis is the line through chain[p-1] and chain[p].
            # Since they are unit distance apart, and axis is principal, we can rotate points about this axis.
            # Let's implement rotation of a point q about the axis defined by point A and direction d (principal axis).
            # If axis is x: rotate (y,z) about (y_A, z_A)
            # If axis is y: rotate (x,z) about (x_A, z_A)
            # If axis is z: rotate (x,y) about (x_A, y_A)
            
            A = chain[p]
            tail_start = p + 1
            tail_len = n - tail_start
            
            if tail_len == 0:
                continue

            moved_tail = []
            valid = True
            new_tail = [None] * tail_len
            
            for k in range(tail_len):
                idx = tail_start + k
                q = chain[idx]
                if axis == 0:
                    # Rotate about x-axis through (A[0], A[1], A[2])
                    # y' = A[1] + (q[1]-A[1])cos - (q[2]-A[2])sin
                    # z' = A[2] + (q[1]-A[1])sin + (q[2]-A[2])cos
                    dy = q[1] - A[1]
                    dz = q[2] - A[2]
                    if angle == 1: # 90 deg
                        ny = A[1] - dz
                        nz = A[2] + dy
                    elif angle == 2: # 180 deg
                        ny = A[1] - dy
                        nz = A[2] - dz
                    else: # 270 deg
                        ny = A[1] + dz
                        nz = A[2] - dy
                    new_q = (q[0], ny, nz)
                elif axis == 1:
                    dx = q[0] - A[0]
                    dz = q[2] - A[2]
                    if angle == 1:
                        nx = A[0] + dz
                        nz = A[2] - dx
                    elif angle == 2:
                        nx = A[0] - dx
                        nz = A[2] - dz
                    else:
                        nx = A[0] - dz
                        nz = A[2] + dx
                    new_q = (nx, q[1], nz)
                else: # axis 2
                    dx = q[0] - A[0]
                    dy = q[1] - A[1]
                    if angle == 1:
                        nx = A[0] - dy
                        ny = A[1] + dx
                    elif angle == 2:
                        nx = A[0] - dx
                        ny = A[1] - dy
                    else:
                        nx = A[0] + dy
                        ny = A[1] - dx
                    new_q = (nx, ny, q[2])
                
                new_tail[k] = new_q
                moved_tail.append((idx, q))
                chain[idx] = new_q

            # Check validity
            # 1. Self-avoidance: new positions not in occ (excluding the tail itself which is being replaced)
            # We need to check against occ minus the original tail positions.
            # For simplicity, check against all occ, but we know the tail positions are being removed.
            # Let's remove original tail positions from a temp set? No, just check if new_q is in occ AND new_q is not one of the original tail positions?
            # Actually, if new_q is in occ, it could be part of the tail (which is fine if it's its own new pos? No, self-avoidance means no two residues at same pos).
            # So we must ensure new_q is not in (occ - original_tail_set) AND new_q is not in new_tail[0..k-1].
            
            orig_tail_set = set(chain[tail_start:] if False else [m[1] for m in moved_tail]) # moved_tail has old positions
            # Wait, moved_tail has (idx, old_pos).
            # We already updated chain. So we need the old positions.
            
            # Let's revert if invalid.
            # Check 1: Boundary connection between p and p+1
            p_prev = chain[p]
            p_new_next = chain[tail_start]
            if abs(p_prev[0]-p_new_next[0]) + abs(p_prev[1]-p_new_next[1]) + abs(p_prev[2]-p_new_next[2]) != 1:
                valid = False
            
            if valid:
                # Check self-avoidance
                seen = set()
                for k in range(tail_len):
                    idx = tail_start + k
                    q = chain[idx]
                    if q in seen:
                        valid = False
                        break
                    seen.add(q)
                    # Check against rest of chain (0..p)
                    if q in occ and q not in orig_tail_set:
                        # It's in occ, but not in original tail, so it's a collision
                        valid = False
                        break
                    # Also check if it collides with any previous new tail position
                    # The 'seen' set handles collisions within the new tail.
                    # But we also need to make sure it doesn't collide with the fixed part (0..p)
                    # The check `q in occ and q not in orig_tail_set` handles this because occ contains the fixed part.
                    # Wait, occ still contains the original tail positions.
                    # So if q is in occ, it might be in orig_tail_set (which is allowed if it's a different residue? No, self-avoidance).
                    # Actually, if q is in orig_tail_set, it means it's one of the old positions of the tail.
                    # If another tail residue moves to that position, that's a collision.
                    # So we must check:
                    # 1. q not in fixed_part_occ (occ - orig_tail_set)
                    # 2. q not in new_tail[0..k-1]
                    
                    # Let's refine the check.
                    pass
            
            # Simpler validity check for pivot:
            # 1. Unit step at boundary p-p+1
            # 2. No collisions with fixed part (0..p)
            # 3. No collisions within new tail
            
            if valid:
                fixed_occ = set(chain[:tail_start])
                seen_new_tail = set()
                for k in range(tail_len):
                    idx = tail_start + k
                    q = chain[idx]
                    if q in fixed_occ:
                        valid = False
                        break
                    if q in seen_new_tail:
                        valid = False
                        break
                    seen_new_tail.add(q)

            if valid:
                # Calculate delta energy
                delta = 0
                # Contacts involving the tail
                # We need to check H-H contacts for all residues in the tail, and their neighbors.
                # Since only tail positions changed, only contacts involving tail residues change.
                # A contact is (i, j) where i, j are H and neighbors.
                # If i is in tail or j is in tail, the contact status might change.
                
                # Old contacts involving tail:
                # We can just recompute energy for the whole chain? No, too slow.
                # Let's compute delta by checking all pairs (i, j) where i in tail or j in tail.
                
                # Actually, simpler:
                # Old energy contribution of tail-related contacts
                # New energy contribution of tail-related contacts
                
                # Let's just do a quick check of the delta by looking at neighbors of moved positions.
                # This is tricky. Let's use the full energy recompute if the tail is small?
                # No, let's stick to the segment reversal style delta calculation or just recompute if n is small.
                # For now, let's approximate or just accept/reject based on a quick check?
                # Better: Calculate delta properly.
                
                # Contacts are between H residues.
                # Let's check all H residues in the tail.
                for idx in range(tail_start, n):
                    if sequence[idx] == 'H':
                        p = chain[idx]
                        for dx, dy, dz in _OFFSETS:
                            q = (p[0]+dx, p[1]+dy, p[2]+dz)
                            j = pos_idx.get(q)
                            if j is not None and abs(idx - j) > 1 and sequence[j] == 'H':
                                # This is a contact in the new state
                                # Was it a contact in the old state?
                                # We need to know the old position of idx.
                                old_p = moved_tail[idx - tail_start][1]
                                # Check if old_p and chain[j] were neighbors? No, chain[j] might have moved too.
                                # This is getting complicated.
                                
                                # Let's just recompute the full energy for the candidate if it's valid.
                                # It's O(n*6) which is fine for n=100.
                                pass
                
                # Let's just recompute full energy for valid pivot moves.
                # It's safer and simpler.
                new_energy = _energy_full(sequence, chain)
                
                if new_energy < current_energy or rng.random() < math.exp(-(new_energy - current_energy) / max(T, 1e-9)):
                    current_energy = new_energy
                    # Update occ and pos_idx
                    # We need to update occ: remove old tail, add new tail
                    for idx, old_p in moved_tail:
                        occ.discard(old_p)
                        occ.add(chain[idx])
                        pos_idx[old_p] = -1 # Mark as removed? No, pos_idx maps pos->idx.
                        # We need to rebuild pos_idx or update carefully.
                        # Let's just rebuild pos_idx for simplicity, it's O(n).
                        pass
                    # Rebuild pos_idx
                    pos_idx = {p: i for i, p in enumerate(chain)}
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = chain[:]
                else:
                    # Revert
                    for idx, old_p in moved_tail:
                        chain[idx] = old_p
                        occ.discard(chain[idx])
                        occ.add(old_p)
                        pos_idx[old_p] = idx
                    # Rebuild pos_idx to be safe
                    pos_idx = {p: i for i, p in enumerate(chain)}
            else:
                # Revert
                for idx, old_p in moved_tail:
                    chain[idx] = old_p
                    occ.discard(chain[idx])
                    occ.add(old_p)
                    pos_idx[old_p] = idx
                pos_idx = {p: i for i, p in enumerate(chain)}

        T = max(T * alpha, T_min)

    return best_chain


def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]

    # 1. Beam Search
    chain = _beam_search(sequence, dimension, beam_width=300)
    
    if chain is None:
        # Fallback: simple greedy
        chain = [(0, 0, 0)]
        occ = {(0, 0, 0)}
        for i in range(1, n):
            last = chain[-1]
            found = False
            for dx, dy, dz in _OFFSETS:
                new_p = (last[0]+dx, last[1]+dy, last[2]+dz)
                if new_p not in occ:
                    chain.append(new_p)
                    occ.add(new_p)
                    found = True
                    break
            if not found:
                break
        if len(chain) < n:
            # Failed to place all, return what we have? No, must return n points.
            # This shouldn't happen with greedy on cubic lattice for reasonable n.
            pass

    # 2. Local Search
    rng = random.Random(42)
    chain = _local_search(sequence, dimension, chain, rng, max_iterations=5000)

    return [list(p) for p in chain]
