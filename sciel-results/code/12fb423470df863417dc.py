"""Constructive synthesis strategy for sorting networks."""
import time

N = 17
MASK = (1 << N) - 1
DEADLINE = time.time() + 80.0  # Safety margin before 90s kill

# EVOLVE-BLOCK-START
def _seed_network():
    # Odd-even transposition sorting network: N phases, N-1 comparators each
    comps = []
    for phase in range(N):
        if phase % 2 == 0:
            comps.extend([(i, i + 1) for i in range(0, N - 1, 2)])
        else:
            comps.extend([(i, i + 1) for i in range(1, N - 1, 2)])
    return comps

def _run_network(comps, input_int):
    """Simulate the network on a single integer input using bitwise operations."""
    for i, j in comps:
        # Extract bits
        bi = (input_int >> i) & 1
        bj = (input_int >> j) & 1
        if bi > bj:
            # Swap: set i to 0, j to 1
            input_int &= ~(1 << i)
            input_int |= (1 << j)
    return input_int

def _verify(comps):
    """Check if the network sorts all 2^N Boolean inputs. Returns True if valid."""
    for inp in range(1 << N):
        if _run_network(comps, inp) != MASK:
            return False
    return True

def _count_unsorted(comps):
    """Count the number of Boolean inputs that are NOT sorted by the network."""
    bad_count = 0
    for inp in range(1 << N):
        if _run_network(comps, inp) != MASK:
            bad_count += 1
    return bad_count

def construct():
    # Start with the seed network
    current = _seed_network()
    best = current
    best_bad = _count_unsorted(current)
    
    # Constructive synthesis: grow the network from scratch
    # This explores a different part of the search space than removal
    cur = []
    cur_bad = 1 << N  # All inputs unsorted initially
    
    # We will try to build up to a size slightly smaller than the seed
    # to ensure we have a chance to beat it, or just keep building until time runs out
    # and return the best of (seed, constructed).
    
    while len(cur) < len(best) and time.time() < DEADLINE:
        best_gain = 0
        best_comp = None
        # Try all possible comparators
        for i in range(N):
            for j in range(i + 1, N):
                # Check if adding this comparator helps
                new_comp = (i, j)
                # Optimization: if we already have this comparator in cur, skip?
                # Actually, duplicate comparators are redundant but harmless. 
                # However, for efficiency in search, we can skip if already present.
                if new_comp in cur:
                    continue
                
                # Calculate new bad count
                # To speed this up, we could use a heuristic, but exact count is safer.
                # _count_unsorted is O(2^N * |cur|). 
                # With |cur| ~ 50 and 2^17 ~ 131k, this is ~6.5M ops.
                # Python might be too slow for full scan of 136 pairs every step.
                # Let's limit the scan or use a faster check if possible.
                
                # Alternative: Only check a subset? No, we need accuracy.
                # Let's just run it. If it's too slow, we'll just do fewer steps.
                
                # We can optimize by early exit in _count_unsorted? No, we need the count for comparison.
                # But we can break early if count > cur_bad + 1? No, we want min.
                
                # Let's just compute it.
                new_bad = _count_unsorted(cur + [new_comp])
                
                if cur_bad - new_bad > best_gain:
                    best_gain = cur_bad - new_bad
                    best_comp = new_comp
        
        if best_comp is None or best_gain == 0:
            # No single addition helps? This shouldn't happen if we haven't reached a valid state
            # unless we are in a local minimum where no single addition helps (impossible for sorting? 
            # Actually, if we are not valid, some comparator MUST help. 
            # Wait, if cur is empty, adding any comparator helps.
            # If cur is partial, adding a comparator might not help if it's "wrong" placement?
            # No, adding a comparator to a non-sorting network always reduces or keeps the number of unsorted inputs?
            # Let's verify that assumption. 
            # If a network is not sorting, there exists an input that is not sorted.
            # Adding a comparator can only change the output of that input.
            # It might make it worse for some inputs and better for others.
            # So greedy might get stuck.
            # In that case, we break the constructive loop and rely on the seed or removal.
            break
        
        cur.append(best_comp)
        cur_bad -= best_gain
        
        if cur_bad == 0:
            # We found a valid network!
            if len(cur) < len(best):
                best = cur
                best_bad = 0
            # Can we remove any?
            # Try removal from the constructed network
            for k in range(len(cur) - 1, -1, -1):
                if time.time() > DEADLINE: break
                candidate = cur[:k] + cur[k+1:]
                cand_bad = _count_unsorted(candidate)
                if cand_bad == 0:
                    cur = candidate
                    cur_bad = 0
                    best = cur
            break

    # Also try removal from the seed network (standard approach) to complement
    # But we already have the seed as `best`. 
    # Let's do a quick removal pass on the seed if we have time, just in case 
    # the constructive approach got stuck in a local optimum larger than a removable seed.
    if time.time() < DEADLINE - 5:
        seed = _seed_network()
        seed_bad = _count_unsorted(seed)
        improved_seed = seed
        improved_seed_bad = seed_bad
        for _ in range(1000): # Fixed small number of removal steps
            if time.time() > DEADLINE - 5: break
            found = False
            for k in range(len(improved_seed) - 1, -1, -1):
                candidate = improved_seed[:k] + improved_seed[k+1:]
                cand_bad = _count_unsorted(candidate)
                if cand_bad == improved_seed_bad:
                    improved_seed = candidate
                    found = True
                    break
            if not found:
                break
        
        if len(improved_seed) < len(best):
            best = improved_seed

    return best
# EVOLVE-BLOCK-END
