import time
import random

N = 18
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 85.0  # Safety margin from 90s limit

# Pre-compute masks at module load time to avoid doing it in construct()
_INITIAL_MASKS = [0] * N
for _w in range(N):
    block = 1 << _w
    pattern = (1 << block) - 1
    pattern <<= block
    num_periods = (1 << N) // (2 * block)
    m = 0
    shift = 0
    for _ in range(num_periods):
        m |= (pattern << shift)
        shift += 2 * block
    _INITIAL_MASKS[_w] = m

def _odd_even_transposition():
    """Return the odd-even transposition sorting network for N wires."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _verify(comparators):
    """
    Verify if the comparator list sorts all 2**N boolean inputs.
    Uses bitwise operations on Python integers representing the state of each wire.
    """
    state = _INITIAL_MASKS[:]
    
    for i, j in comparators:
        a = state[i]
        b = state[j]
        state[i] = a & b
        state[j] = a | b
        
    for i in range(N - 1):
        if (state[i] & ~state[i+1]) != 0:
            return False
            
    return True

def _greedy_reduce(network, deadline):
    """
    Greedily remove redundant comparators until no more can be removed.
    Returns the reduced network.
    """
    current = network[:]
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(current) - 1, -1, -1):
            if time.time() > deadline:
                return current
            neighbor = current[:i] + current[i+1:]
            if _verify(neighbor):
                current = neighbor
                changed = True
                break
    return current

def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE_SECONDS
    
    # Build initial network
    current = _odd_even_transposition()
    best = current[:]
    best_len = len(best)
    
    # Phase 0: Quick greedy reduction to get a good starting point
    current = _greedy_reduce(current, deadline)
    best = current[:]
    best_len = len(best)
    
    # Phase 1: Simulated Annealing with insert-delete moves
    # This is the main optimization phase. Insert-delete moves allow escaping
    # local minima by temporarily restructuring the network.
    
    T = 10.0
    T_min = 0.05
    cooling_rate = 0.995
    
    rng = random.Random(42)
    
    all_comparators = [(i, j) for i in range(N) for j in range(i + 1, N)]
    comp_set = set(current)
    
    fail_count = 0
    success_count = 0
    no_improve_count = 0
    
    while time.time() < deadline and T > T_min:
        # Weighted move selection
        move_type = rng.choices(
            ['remove', 'insert_delete', 'swap', 'move', 'double_insert_delete'],
            weights=[3, 4, 1, 2, 2]
        )[0]
        
        neighbor = current[:]
        neighbor_set = comp_set.copy()
        
        valid_move = True
        
        if move_type == 'remove' and len(neighbor) > 0:
            idx = rng.randrange(len(neighbor))
            removed = neighbor.pop(idx)
            neighbor_set.discard(removed)
            
        elif move_type == 'insert_delete':
            if len(neighbor) > 0:
                idx = rng.randrange(len(neighbor))
                removed = neighbor.pop(idx)
                neighbor_set.discard(removed)
                # Choose a new comparator
                available = [c for c in all_comparators if c not in neighbor_set]
                if available:
                    new_comp = rng.choice(available)
                    pos = rng.randrange(len(neighbor) + 1)
                    neighbor.insert(pos, new_comp)
                    neighbor_set.add(new_comp)
                else:
                    neighbor.insert(idx, removed)
                    neighbor_set.add(removed)
                    valid_move = False
                    
        elif move_type == 'double_insert_delete':
            # Remove two comparators and insert two new ones
            if len(neighbor) >= 2:
                idx1 = rng.randrange(len(neighbor))
                removed1 = neighbor.pop(idx1)
                neighbor_set.discard(removed1)
                
                idx2 = rng.randrange(len(neighbor))
                removed2 = neighbor.pop(idx2)
                neighbor_set.discard(removed2)
                
                available = [c for c in all_comparators if c not in neighbor_set]
                if len(available) >= 2:
                    new_comp1 = rng.choice(available)
                    available.remove(new_comp1)
                    new_comp2 = rng.choice(available)
                    
                    pos1 = rng.randrange(len(neighbor) + 1)
                    neighbor.insert(pos1, new_comp1)
                    pos2 = rng.randrange(len(neighbor) + 1)
                    neighbor.insert(pos2, new_comp2)
                    
                    neighbor_set.add(new_comp1)
                    neighbor_set.add(new_comp2)
                else:
                    # Put them back
                    neighbor.insert(idx1, removed1)
                    neighbor.insert(idx2, removed2)
                    neighbor_set.add(removed1)
                    neighbor_set.add(removed2)
                    valid_move = False
                    
        elif move_type == 'swap' and len(neighbor) >= 2:
            i = rng.randrange(len(neighbor))
            j = rng.randrange(len(neighbor))
            if i != j:
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            else:
                valid_move = False
                
        elif move_type == 'move' and len(neighbor) >= 2:
            i = rng.randrange(len(neighbor))
            j = rng.randrange(len(neighbor))
            if i != j:
                comp = neighbor.pop(i)
                if i < j:
                    j -= 1
                neighbor.insert(j, comp)
            else:
                valid_move = False

        if not valid_move:
            T *= cooling_rate
            continue

        if _verify(neighbor):
            delta = len(neighbor) - len(current)
            # Accept if better or with probability exp(-delta/T)
            if delta <= 0 or rng.random() < 2.718281828459045 ** (-delta / T):
                current = neighbor
                comp_set = neighbor_set
                if len(current) < best_len:
                    best = current[:]
                    best_len = len(current)
                    no_improve_count = 0
                    success_count += 1
                    fail_count = 0
                else:
                    no_improve_count += 1
            else:
                fail_count += 1
                no_improve_count += 1
        else:
            fail_count += 1
            no_improve_count += 1
        
        # Adaptive cooling
        if success_count > 10:
            cooling_rate = 0.998
            success_count = 0
        elif fail_count > 30:
            cooling_rate = 0.99
            fail_count = 0
        
        T *= cooling_rate
        
        # Periodic restart: if stuck for too long, try a fresh greedy pass from best
        if no_improve_count > 200 and time.time() < deadline - 5:
            candidate = best[:]
            candidate = _greedy_reduce(candidate, deadline - 2)
            if len(candidate) < len(current):
                current = candidate
                comp_set = set(current)
            no_improve_count = 0
            T = max(T, 5.0)  # Reheat

    # Final greedy reduction on the best solution
    if time.time() < deadline - 2:
        final_candidate = _greedy_reduce(best, deadline - 1)
        if len(final_candidate) < len(best):
            best = final_candidate
    
    # Final verification
    if not _verify(best):
        return _odd_even_transposition()
        
    return best
