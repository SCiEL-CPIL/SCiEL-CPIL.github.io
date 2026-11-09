import time
import random
import numpy as np

_N = 128
_LO, _HI = 128, 256

def _get_reps(A_set):
    """Return array of representation counts for sums [0, 2*N]."""
    vec = np.zeros(_N + 1, dtype=np.int32)
    for a in A_set:
        vec[a] = 1
    # Convolution gives ordered sum counts
    conv = np.convolve(vec, vec, mode='full')
    # conv[k] is count of (a,b) in AxA such that a+b=k
    # We need unordered counts for k in [_LO, _HI]
    reps = np.zeros(_HI - _LO + 1, dtype=np.int32)
    for i, m in enumerate(range(_LO, _HI + 1)):
        c = conv[m]
        # Check if m/2 is in A
        if m % 2 == 0:
            half = m // 2
            if 0 <= half <= _N and vec[half] == 1:
                c += 1
        reps[i] = c // 2
    return reps

def _is_valid(reps):
    return np.all(reps > 0)

def _compute_peak(reps):
    return int(np.max(reps))

def construct():
    start_time = time.time()
    DEADLINE = start_time + 55  # Safety margin

    best_A = None
    best_peak = float('inf')
    best_reps = None

    # 1. Initial Construction
    # Try to find a valid set with low peak.
    # Start with a random set of size ~30-40 and prune/optimize.
    
    # Heuristic start: 
    # A set of roughly equal spaced numbers might work.
    # Or just random.
    
    rng = random.Random(42)
    
    # Try multiple random initializations
    for attempt in range(20):
        if time.time() > DEADLINE: break
        
        # Random size between 25 and 40
        k = rng.randint(25, 40)
        A = set(rng.sample(range(_N + 1), k))
        
        # Ensure validity? Random sets often have gaps.
        # We can add elements to fill gaps.
        reps = _get_reps(A)
        while not _is_valid(reps):
            # Find a missing sum
            missing = np.where(reps == 0)[0]
            if len(missing) == 0: break
            m = missing[0] + _LO
            # Try to add an element that helps cover m
            # Ideally, add x such that m-x in A or we can add both?
            # Just add a random element? Or specific?
            # To cover m, we need a in A, m-a in A.
            # If we add x, we create new sums x+y for y in A.
            # Let's just add a random element not in A.
            candidates = [x for x in range(_N+1) if x not in A]
            if not candidates: break
            new_elem = rng.choice(candidates)
            A.add(new_elem)
            reps = _get_reps(A)
            # If set gets too large, stop trying to fill randomly and restart
            if len(A) > 60:
                break
        
        if _is_valid(reps):
            peak = _compute_peak(reps)
            if peak < best_peak:
                best_peak = peak
                best_A = A.copy()
                best_reps = reps.copy()

    if best_A is None:
        # Fallback: Interval
        A = set(range(64, 129))
        best_A = A
        best_peak = _compute_peak(_get_reps(A))
        best_reps = _get_reps(A)

    # 2. Local Search / Simulated Annealing
    A = best_A
    current_peak = best_peak
    current_reps = best_reps
    
    # Precompute list of elements
    in_A = list(A)
    not_in_A = [x for x in range(_N + 1) if x not in A]
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    no_improve = 0
    
    while time.time() < DEADLINE:
        if T < T_min:
            T = 1.0 # Reset temperature or restart?
            # Better: do a "kick"
            # Randomly swap 5 elements
            for _ in range(5):
                if in_A and not_in_A:
                    rem = rng.choice(in_A)
                    add = rng.choice(not_in_A)
                    A.discard(rem)
                    A.add(add)
                    in_A.remove(rem)
                    not_in_A.remove(add)
            current_reps = _get_reps(A)
            if not _is_valid(current_reps):
                # Revert? Or just accept invalid and try to fix?
                # Easier to revert.
                # Actually, let's just restart the local search from best_A
                A = best_A.copy()
                in_A = list(A)
                not_in_A = [x for x in range(_N + 1) if x not in A]
                current_reps = best_reps.copy()
                current_peak = best_peak
                T = 1.0
                continue
            
            current_peak = _compute_peak(current_reps)
            if current_peak < best_peak:
                best_peak = current_peak
                best_A = A.copy()
                best_reps = current_reps.copy()
                no_improve = 0
            else:
                # If kick made it worse, we might want to go back?
                # Or just continue from here?
                # Let's go back to best_A to avoid drifting too far
                A = best_A.copy()
                in_A = list(A)
                not_in_A = [x for x in range(_N + 1) if x not in A]
                current_reps = best_reps.copy()
                current_peak = best_peak
                T = 1.0
                continue

        # Generate a move
        move_type = rng.random()
        if move_type < 0.6: # Swap
            if not in_A or not not_in_A:
                T *= cooling
                continue
            rem = rng.choice(in_A)
            add = rng.choice(not_in_A)
            A.discard(rem)
            A.add(add)
            in_A.remove(rem)
            not_in_A.remove(add)
            move_desc = ('swap', rem, add)
        elif move_type < 0.8: # Remove
            if not in_A:
                T *= cooling
                continue
            rem = rng.choice(in_A)
            A.discard(rem)
            in_A.remove(rem)
            not_in_A.append(rem)
            move_desc = ('remove', rem, None)
        else: # Add
            if not not_in_A:
                T *= cooling
                continue
            add = rng.choice(not_in_A)
            A.add(add)
            in_A.append(add)
            not_in_A.remove(add)
            move_desc = ('add', None, add)
            
        new_reps = _get_reps(A)
        if not _is_valid(new_reps):
            # Revert
            if move_desc[0] == 'swap':
                A.discard(move_desc[2])
                A.add(move_desc[1])
                in_A.remove(move_desc[2])
                in_A.append(move_desc[1])
                not_in_A.append(move_desc[2])
                not_in_A.remove(move_desc[1])
            elif move_desc[0] == 'remove':
                A.add(move_desc[1])
                in_A.append(move_desc[1])
                not_in_A.remove(move_desc[1])
            elif move_desc[0] == 'add':
                A.discard(move_desc[2])
                in_A.remove(move_desc[2])
                not_in_A.append(move_desc[2])
            T *= cooling
            continue
            
        new_peak = _compute_peak(new_reps)
        
        # Acceptance
        if new_peak < current_peak:
            # Always accept
            current_peak = new_peak
            current_reps = new_reps
            no_improve = 0
            if new_peak < best_peak:
                best_peak = new_peak
                best_A = A.copy()
                best_reps = new_reps.copy()
        elif new_peak == current_peak:
            # Accept with prob 1 (to explore plateau) or based on variance?
            # Let's accept to allow further improvements
            current_reps = new_reps
            # no_improve += 1 # Don't count as no improve if we moved?
            # Actually, if peak is same, we might get stuck.
            # Let's accept.
        else:
            # Worse peak. Accept with prob exp(-delta/T)
            delta = new_peak - current_peak
            if delta > 0:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current_peak = new_peak
                    current_reps = new_reps
                else:
                    # Revert
                    if move_desc[0] == 'swap':
                        A.discard(move_desc[2])
                        A.add(move_desc[1])
                        in_A.remove(move_desc[2])
                        in_A.append(move_desc[1])
                        not_in_A.append(move_desc[2])
                        not_in_A.remove(move_desc[1])
                    elif move_desc[0] == 'remove':
                        A.add(move_desc[1])
                        in_A.append(move_desc[1])
                        not_in_A.remove(move_desc[1])
                    elif move_desc[0] == 'add':
                        A.discard(move_desc[2])
                        in_A.remove(move_desc[2])
                        not_in_A.append(move_desc[2])
            else:
                # Should not happen if new_peak > current_peak
                pass
                
        T *= cooling
        
        # Check for no improvement in best
        # We can track steps since best improvement
        # For now, rely on T reset/kick logic above
        
    return sorted(best_A)
