"""
4-AP-free set construction using pair-swap local search with bit-parallel conflict tracking.
Structural change: uses a precomputed 4-AP list and tracks violations per-AP for O(1) delta evaluation.
"""
import time
import random
import math

N = 512

def _build_4aps():
    """Build list of all 4-APs as tuples (a, b, c, d)."""
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            aps.append((a, b, c, e))
    return aps

_ALL_APS = _build_4aps()
_NUM_APS = len(_ALL_APS)

# For each element x, store list of (ap_index, position) where x appears in that AP at that position.
# This allows O(degree) delta computation.
def _build_element_ap_map():
    elem_ap = [[] for _ in range(N + 1)]
    for idx, (a, b, c, e) in enumerate(_ALL_APS):
        elem_ap[a].append((idx, 0))
        elem_ap[b].append((idx, 1))
        elem_ap[c].append((idx, 2))
        elem_ap[e].append((idx, 3))
    return elem_ap

_ELEM_AP = _build_element_ap_map()

# Precompute for each element x, the list of other elements grouped by AP.
# _ELEM_OTHERS[x] = list of tuples (o1, o2, o3) where x, o1, o2, o3 form a 4-AP
def _build_element_others():
    elem_others = [[] for _ in range(N + 1)]
    for (a, b, c, e) in _ALL_APS:
        elem_others[a].append((b, c, e))
        elem_others[b].append((a, c, e))
        elem_others[c].append((a, b, e))
        elem_others[e].append((a, b, c))
    return elem_others

_ELEM_OTHERS = _build_element_others()

def _get_initial_cantor():
    """Cantor set: numbers with no digit 2 in base-3 representation."""
    bits = 0
    for x in range(1, N + 1):
        t = x
        valid = True
        while t > 0:
            if t % 3 == 2:
                valid = False
                break
            t //= 3
        if valid:
            bits |= (1 << x)
    return bits

def _greedy_forward():
    """Greedy forward: add elements 1..N if they don't create a 4-AP."""
    bits = 0
    for x in range(1, N + 1):
        ok = True
        for (ap_idx, pos) in _ELEM_AP[x]:
            a, b, c, e = _ALL_APS[ap_idx]
            # Check if the other 3 elements are in the set
            others = [a, b, c, e]
            others.remove(x)
            if all((bits >> o) & 1 for o in others):
                ok = False
                break
        if ok:
            bits |= (1 << x)
    return bits

def _greedy_reverse():
    """Greedy reverse: add elements N..1 if they don't create a 4-AP."""
    bits = 0
    for x in range(N, 0, -1):
        ok = True
        for (ap_idx, pos) in _ELEM_AP[x]:
            a, b, c, e = _ALL_APS[ap_idx]
            others = [a, b, c, e]
            others.remove(x)
            if all((bits >> o) & 1 for o in others):
                ok = False
                break
        if ok:
            bits |= (1 << x)
    return bits

def _greedy_random(rng):
    """Greedy with random ordering."""
    bits = 0
    order = list(range(1, N + 1))
    rng.shuffle(order)
    for x in order:
        ok = True
        for (ap_idx, pos) in _ELEM_AP[x]:
            a, b, c, e = _ALL_APS[ap_idx]
            others = [a, b, c, e]
            others.remove(x)
            if all((bits >> o) & 1 for o in others):
                ok = False
                break
        if ok:
            bits |= (1 << x)
    return bits

def _greedy_weighted(rng):
    """Greedy with weighted random ordering: prefer elements with fewer conflicts."""
    bits = 0
    remaining = list(range(1, N + 1))
    while remaining:
        # Score each remaining element by how many potential 4-APs it's involved in with current set
        best_x = None
        best_score = -1
        # Sample a few candidates for speed
        candidates = rng.sample(remaining, min(20, len(remaining)))
        for x in candidates:
            # Count how many 4-APs would be violated if we added x
            conflict_count = 0
            for (ap_idx, pos) in _ELEM_AP[x]:
                a, b, c, e = _ALL_APS[ap_idx]
                others = [a, b, c, e]
                others.remove(x)
                cnt = sum(1 for o in others if (bits >> o) & 1)
                if cnt == 3:
                    conflict_count += 1
                elif cnt == 2:
                    # Partial conflict - will become a problem if we add one more
                    conflict_count += 0.5
            score = -conflict_count + rng.random() * 0.1
            if score > best_score:
                best_score = score
                best_x = x
        if best_x is not None:
            # Check if best_x is actually safe
            ok = True
            for (ap_idx, pos) in _ELEM_AP[best_x]:
                a, b, c, e = _ALL_APS[ap_idx]
                others = [a, b, c, e]
                others.remove(best_x)
                if all((bits >> o) & 1 for o in others):
                    ok = False
                    break
            if ok:
                bits |= (1 << best_x)
            remaining.remove(best_x)
        else:
            break
    return bits

def _count_violations(bits):
    """Count number of complete 4-APs in the set."""
    count = 0
    for (a, b, c, e) in _ALL_APS:
        if (bits >> a) & 1 and (bits >> b) & 1 and (bits >> c) & 1 and (bits >> e) & 1:
            count += 1
    return count

def _delta_violations_add(bits, x):
    """Return number of new 4-AP violations created by adding x to the set."""
    count = 0
    for (o1, o2, o3) in _ELEM_OTHERS[x]:
        if ((bits >> o1) & 1) and ((bits >> o2) & 1) and ((bits >> o3) & 1):
            count += 1
    return count

def _delta_violations_remove(bits, x):
    """Return number of 4-AP violations removed by removing x from the set."""
    count = 0
    for (o1, o2, o3) in _ELEM_OTHERS[x]:
        if ((bits >> o1) & 1) and ((bits >> o2) & 1) and ((bits >> o3) & 1):
            count += 1
    return count

def _pair_swap_local_search(bits, deadline, rng):
    """
    Local search using pair swaps: remove one element, add another.
    Uses delta violation computation for O(degree) evaluation instead of O(num_APS).
    """
    current_bits = bits
    current_size = bin(current_bits).count('1')
    current_violations = _count_violations(current_bits)
    
    best_bits = current_bits
    best_size = current_size
    
    if current_violations == 0 and current_size > best_size:
        best_bits = current_bits
        best_size = current_size
    
    start_time = time.time()
    no_improve_count = 0
    
    while time.time() < deadline:
        # Try a pair swap: remove x (in set), add y (not in set)
        in_set = [i for i in range(1, N + 1) if (current_bits >> i) & 1]
        not_in_set = [i for i in range(1, N + 1) if not (current_bits >> i) & 1]
        
        if not in_set or not not_in_set:
            break
        
        x = rng.choice(in_set)
        y = rng.choice(not_in_set)
        
        if x == y:
            continue
        
        # Compute delta violations for swap: remove x, add y
        # Violations removed by removing x
        removed_violations = _delta_violations_remove(current_bits, x)
        # Violations added by adding y to (current_bits without x)
        bits_without_x = current_bits & ~(1 << x)
        added_violations = _delta_violations_add(bits_without_x, y)
        
        new_violations = current_violations - removed_violations + added_violations
        
        if new_violations < current_violations:
            current_bits = (current_bits & ~(1 << x)) | (1 << y)
            current_violations = new_violations
            no_improve_count = 0
        elif new_violations == current_violations:
            if rng.random() < 0.3:
                current_bits = (current_bits & ~(1 << x)) | (1 << y)
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        if current_violations == 0 and current_size > best_size:
            best_bits = current_bits
            best_size = current_size
        
        # Also try single adds if violations are 0
        if current_violations == 0:
            if not_in_set:
                y_add = rng.choice(not_in_set)
                add_violations = _delta_violations_add(current_bits, y_add)
                if add_violations == 0:
                    current_bits |= (1 << y_add)
                    current_size += 1
                    no_improve_count = 0
                    if current_size > best_size:
                        best_bits = current_bits
                        best_size = current_size
                    # Update not_in_set
                    not_in_set = [i for i in range(1, N + 1) if not (current_bits >> i) & 1]
        
        # Periodically try greedy fill
        if no_improve_count > 200:
            for z in range(1, N + 1):
                if not (current_bits >> z) & 1:
                    if _delta_violations_add(current_bits, z) == 0:
                        current_bits |= (1 << z)
                        current_size += 1
                        if current_size > best_size:
                            best_bits = current_bits
                            best_size = current_size
            no_improve_count = 0
        
        if no_improve_count > 5000:
            break
    
    return best_bits, best_size

def _hill_climbing(bits, deadline, rng):
    """
    Simple hill climbing: try single adds and removals.
    Uses delta violation computation for efficiency.
    """
    current_bits = bits
    current_size = bin(current_bits).count('1')
    current_violations = _count_violations(current_bits)
    
    best_bits = current_bits
    best_size = current_size
    
    if current_violations == 0 and current_size > best_size:
        best_bits = current_bits
        best_size = current_size
    
    start_time = time.time()
    
    while time.time() < deadline:
        improved = False
        
        # Try adding elements
        for x in range(1, N + 1):
            if (current_bits >> x) & 1:
                continue
            if _delta_violations_add(current_bits, x) == 0:
                current_bits |= (1 << x)
                current_size += 1
                improved = True
                if current_size > best_size:
                    best_bits = current_bits
                    best_size = current_size
        
        # Try removing and re-adding different elements (pair swap)
        if not improved and current_violations == 0:
            in_set = [i for i in range(1, N + 1) if (current_bits >> i) & 1]
            not_in_set = [i for i in range(1, N + 1) if not (current_bits >> i) & 1]
            if in_set and not_in_set:
                # Try a few random pair swaps
                for _ in range(50):
                    if time.time() > deadline:
                        break
                    x = rng.choice(in_set)
                    y = rng.choice(not_in_set)
                    if x == y:
                        continue
                    bits_without_x = current_bits & ~(1 << x)
                    removed_violations = _delta_violations_remove(current_bits, x)
                    added_violations = _delta_violations_add(bits_without_x, y)
                    if removed_violations - added_violations >= 0:
                        # Valid or better swap
                        current_bits = bits_without_x | (1 << y)
                        # Greedy fill
                        for z in range(1, N + 1):
                            if not (current_bits >> z) & 1:
                                if _delta_violations_add(current_bits, z) == 0:
                                    current_bits |= (1 << z)
                        current_size = bin(current_bits).count('1')
                        if current_size > best_size:
                            best_bits = current_bits
                            best_size = current_size
                        improved = True
                        break
        
        if not improved:
            break
    
    return best_bits, best_size

def _greedy_fill(bits, deadline):
    """Greedily add any element that doesn't create a 4-AP."""
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            if not ((bits >> x) & 1):
                if _delta_violations_add(bits, x) == 0:
                    bits |= (1 << x)
                    changed = True
    return bits

def _simulated_annealing(bits, deadline, rng):
    """
    Simulated annealing for 4-AP-free set maximization.
    State: set of elements. Move: add/remove/swap element.
    Energy: -size + penalty * violations. Accept worse moves with probability exp(-delta_E / T).
    """
    current_bits = bits
    current_size = bin(current_bits).count('1')
    current_violations = _count_violations(current_bits)
    
    # Objective: maximize size, minimize violations
    # Energy = violations * 100 - size (lower is better)
    current_energy = current_violations * 100 - current_size
    
    best_bits = current_bits
    best_size = current_size
    best_energy = current_energy
    
    start_time = time.time()
    total_time = max(deadline - start_time, 1)
    T_start = 10.0
    T_end = 0.01
    alpha = (T_end / T_start) ** (1.0 / max(1, int(total_time * 1000)))
    
    T = T_start
    no_improve = 0
    
    while time.time() < deadline and T > T_end:
        # Choose a move type
        move_type = rng.random()
        
        in_set = [i for i in range(1, N + 1) if (current_bits >> i) & 1]
        not_in_set = [i for i in range(1, N + 1) if not (current_bits >> i) & 1]
        
        if not in_set or not not_in_set:
            break
        
        if move_type < 0.4:
            # Single add
            y = rng.choice(not_in_set)
            new_bits = current_bits | (1 << y)
            new_violations = current_violations + _delta_violations_add(current_bits, y)
            new_size = current_size + 1
            new_energy = new_violations * 100 - new_size
            delta_e = new_energy - current_energy
            
            if delta_e <= 0 or rng.random() < math.exp(-delta_e / T):
                current_bits = new_bits
                current_violations = new_violations
                current_size = new_size
                current_energy = new_energy
                no_improve = 0
            else:
                no_improve += 1
                
        elif move_type < 0.7:
            # Single remove
            x = rng.choice(in_set)
            new_bits = current_bits & ~(1 << x)
            new_violations = current_violations - _delta_violations_remove(current_bits, x)
            new_size = current_size - 1
            new_energy = new_violations * 100 - new_size
            delta_e = new_energy - current_energy
            
            if delta_e <= 0 or rng.random() < math.exp(-delta_e / T):
                current_bits = new_bits
                current_violations = new_violations
                current_size = new_size
                current_energy = new_energy
                no_improve = 0
            else:
                no_improve += 1
                
        else:
            # Pair swap
            x = rng.choice(in_set)
            y = rng.choice(not_in_set)
            if x == y:
                T *= alpha
                continue
            bits_without_x = current_bits & ~(1 << x)
            removed_violations = _delta_violations_remove(current_bits, x)
            added_violations = _delta_violations_add(bits_without_x, y)
            new_bits = bits_without_x | (1 << y)
            new_violations = current_violations - removed_violations + added_violations
            new_size = current_size  # Same size
            new_energy = new_violations * 100 - new_size
            delta_e = new_energy - current_energy
            
            if delta_e <= 0 or rng.random() < math.exp(-delta_e / T):
                current_bits = new_bits
                current_violations = new_violations
                current_energy = new_energy
                no_improve = 0
            else:
                no_improve += 1
        
        # Track best valid solution
        if current_violations == 0 and current_size > best_size:
            best_bits = current_bits
            best_size = current_size
            best_energy = current_energy
        
        # Periodically try greedy fill on best solution
        if no_improve > 100:
            filled = _greedy_fill(best_bits, deadline)
            filled_size = bin(filled).count('1')
            if _count_violations(filled) == 0 and filled_size > best_size:
                best_bits = filled
                best_size = filled_size
                best_energy = -filled_size
                no_improve = 0
        
        # Cool down
        T *= alpha
        
        if no_improve > 5000:
            break
    
    return best_bits, best_size

def construct():
    deadline = time.time() + 55  # Safety margin
    
    # Generate diverse initial seeds
    seeds = []
    
    # Cantor
    cantor = _get_initial_cantor()
    seeds.append(cantor)
    
    # Forward greedy
    fwd = _greedy_forward()
    seeds.append(fwd)
    
    # Reverse greedy
    rev = _greedy_reverse()
    seeds.append(rev)
    
    # Random greedy (multiple)
    rng = random.Random(42)
    for _ in range(5):
        g = _greedy_random(rng)
        seeds.append(g)
    
    # Weighted greedy (multiple)
    for _ in range(3):
        g = _greedy_weighted(rng)
        seeds.append(g)
    
    best_bits = 0
    best_size = 0
    
    for seed in seeds:
        if time.time() > deadline - 10:
            break
        
        # Greedy fill
        filled = _greedy_fill(seed, deadline - 10)
        size = bin(filled).count('1')
        
        if size > best_size:
            best_bits = filled
            best_size = size
    
    # Phase 2: Hill climbing on best seed
    if time.time() < deadline - 5:
        hc_bits, hc_size = _hill_climbing(best_bits, deadline - 5, random.Random(123))
        if hc_size > best_size:
            best_bits = hc_bits
            best_size = hc_size
    
    # Phase 3: Pair swap local search
    if time.time() < deadline - 2:
        ps_bits, ps_size = _pair_swap_local_search(best_bits, deadline - 2, random.Random(456))
        if ps_size > best_size:
            best_bits = ps_bits
            best_size = ps_size
    
    # Phase 4: Simulated annealing for additional exploration
    if time.time() < deadline - 1:
        sa_bits, sa_size = _simulated_annealing(best_bits, deadline - 1, random.Random(789))
        if sa_size > best_size:
            best_bits = sa_bits
            best_size = sa_size
    
    # Final greedy fill
    if time.time() < deadline - 1:
        best_bits = _greedy_fill(best_bits, deadline - 1)
    
    # Convert to set
    best_set = {i for i in range(1, N + 1) if (best_bits >> i) & 1}
    
    # Verification
    if _count_violations(best_bits) != 0:
        # Fallback to Cantor
        best_set = {i for i in range(1, N + 1) if (1 << i) & _get_initial_cantor()}
    
    return sorted(best_set)
