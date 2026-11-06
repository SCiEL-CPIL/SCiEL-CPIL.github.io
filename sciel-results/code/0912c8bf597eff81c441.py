"""Combined program: greedy coverage initialization + targeted swap local search."""
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    # Bitset representation: x in 1..512 maps to bit (x-1)
    def make_mask(s):
        m = 0
        for x in s:
            m |= (1 << (x - 1))
        return m

    def score_mask(m):
        """Count distinct d > 0 such that a, a+d, a+2d are in set for some a."""
        count = 0
        for d in range(1, 256):
            if (m & (m >> d) & (m >> (2 * d))) != 0:
                count += 1
        return count

    def missing_diffs(m):
        """Return list of d > 0 for which no 3-AP exists."""
        missing = []
        for d in range(1, 256):
            if not (m & (m >> d) & (m >> (2 * d))):
                missing.append(d)
        return missing

    # Greedy initialization using maximum coverage heuristic
    def greedy_init(rng):
        mask = 0
        elems = []
        for _ in range(32):
            best_elem = -1
            best_gain = -1
            # Evaluate all candidates
            for x in range(1, 513):
                if mask & (1 << (x - 1)):
                    continue
                new_mask = mask | (1 << (x - 1))
                sc = score_mask(new_mask)
                # Compute gain in covered differences
                old_covered = score_mask(mask)
                gain = sc - old_covered
                if gain > best_gain:
                    best_gain = gain
                    best_elem = x
            # Add some randomness: pick from top candidates
            # Re-evaluate to get top candidates
            candidates = []
            for x in range(1, 513):
                if mask & (1 << (x - 1)):
                    continue
                new_mask = mask | (1 << (x - 1))
                sc = score_mask(new_mask)
                old_covered = score_mask(mask)
                gain = sc - old_covered
                candidates.append((gain, x))
            candidates.sort(reverse=True, key=lambda t: t[0])
            top_k = min(10, len(candidates))
            if top_k > 0:
                _, x = rng.choice(candidates[:top_k])
                mask |= (1 << (x - 1))
                elems.append(x)
        return mask, elems

    # Targeted swap local search that prioritizes missing differences
    def targeted_local_search(mask, elems, rng, time_limit):
        current_score = score_mask(mask)
        best_score = current_score
        best_mask = mask
        best_elems = list(elems)
        
        T = 2.0
        cooling = 0.9995
        min_temp = 0.01
        stagnation = 0
        max_stagnation = 5000
        
        while time.time() < time_limit:
            if T < min_temp:
                break
            if stagnation > max_stagnation:
                break
            
            # Get missing differences to guide search
            missing = missing_diffs(mask)
            
            # Choose move: targeted swap
            # Pick an element to replace
            i = rng.randrange(len(elems))
            old_elem = elems[i]
            
            # Generate candidate new elements
            # Bias towards elements that might cover missing differences
            candidates = []
            for _ in range(20):
                if missing and rng.random() < 0.5:
                    # Try to create a 3-AP for a missing d
                    d = rng.choice(missing)
                    # Need a, a+d, a+2d all in set
                    # We're replacing old_elem, so try to find a position
                    # that completes a progression
                    # Try various a values
                    for a in range(1, 513 - 2 * d + 1):
                        a1, a2, a3 = a, a + d, a + 2 * d
                        # Check how many are already in set
                        in_set = 0
                        positions = []
                        for val in (a1, a2, a3):
                            if val == old_elem:
                                positions.append(val)
                            elif mask & (1 << (val - 1)):
                                in_set += 1
                            else:
                                positions.append(val)
                        # If we can complete by adding one element
                        if in_set == 2 and len(positions) == 1:
                            candidates.append(positions[0])
                            break
                else:
                    # Random candidate
                    x = rng.randint(1, 512)
                    if not (mask & (1 << (x - 1))):
                        candidates.append(x)
            
            if not candidates:
                # Fallback: random swap
                while True:
                    new_elem = rng.randint(1, 512)
                    if not (mask & (1 << (new_elem - 1))):
                        candidates = [new_elem]
                        break
            
            new_elem = rng.choice(candidates)
            
            new_mask = mask ^ (1 << (old_elem - 1)) ^ (1 << (new_elem - 1))
            new_score = score_mask(new_mask)
            
            delta = new_score - current_score
            
            # Acceptance criterion
            if delta > 0 or (T > 1e-6 and rng.random() < math.exp(delta / T)):
                # Accept
                mask = new_mask
                elems[i] = new_elem
                current_score = new_score
                stagnation = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_mask = mask
                    best_elems = list(elems)
                
                if best_score == 255:
                    break
            else:
                stagnation += 1
            
            T *= cooling
        
        return best_mask, best_elems, best_score

    # Main search loop with multiple restarts
    rng = random.Random(42)
    best_mask = 0
    best_score = 0
    best_elems = []
    
    # First, run greedy initialization
    greedy_mask, greedy_elems = greedy_init(rng)
    greedy_score = score_mask(greedy_mask)
    
    if greedy_score > best_score:
        best_score = greedy_score
        best_mask = greedy_mask
        best_elems = list(greedy_elems)
    
    # Run targeted local search from greedy start
    if time.time() < deadline:
        search_time = min(30.0, deadline - time.time())
        lm, le, ls = targeted_local_search(greedy_mask, greedy_elems, rng, start_time + search_time)
        if ls > best_score:
            best_score = ls
            best_mask = lm
            best_elems = list(le)
    
    # Additional restarts with random initialization
    while time.time() < deadline - 5.0:
        # Random initialization
        current_elems = random.sample(range(1, 513), 32)
        current_mask = make_mask(current_elems)
        current_score = score_mask(current_mask)
        
        if current_score > best_score:
            best_score = current_score
            best_mask = current_mask
            best_elems = list(current_elems)
        
        # Run targeted local search
        remaining = deadline - time.time() - 2.0
        if remaining < 5.0:
            break
        search_time = min(remaining, 25.0)
        lm, le, ls = targeted_local_search(current_mask, current_elems, rng, start_time + search_time)
        if ls > best_score:
            best_score = ls
            best_mask = lm
            best_elems = list(le)
    
    # Final polish: run one more targeted search from best found
    if time.time() < deadline - 3.0 and best_elems:
        remaining = deadline - time.time() - 1.0
        if remaining > 5.0:
            lm, le, ls = targeted_local_search(best_mask, best_elems, rng, start_time + remaining)
            if ls > best_score:
                best_score = ls
                best_mask = lm
                best_elems = list(le)
    
    # Convert best_mask to list
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(idx + 1)  # Convert from 0-indexed to 1-indexed
        m ^= lsb
    
    # Ensure we have exactly 32 elements
    if len(result) != 32:
        # Fallback: use best_elems if valid
        if len(best_elems) == 32 and len(set(best_elems)) == 32:
            return sorted(best_elems)
        # Otherwise return a valid random set
        return random.sample(range(1, 513), 32)
    
    return sorted(result)
