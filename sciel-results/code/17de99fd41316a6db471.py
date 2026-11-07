"""Combined program with enhanced local search and optimized checks."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _subset_sums(S):
    """Return set of all nonempty subset sums of S."""
    sums = set()
    for x in S:
        new_sums = set()
        for s in sums:
            ns = s + x
            new_sums.add(ns)
        sums.update(new_sums)
        sums.add(x)
    return sums

def _is_valid(S):
    """Full validity check."""
    if not S:
        return True
    sums = _subset_sums(S)
    total = sum(S)
    sums_list = sorted(sums)
    sums_set = sums
    for s in sums_list:
        if s == 0:
            continue
        multiple = 2 * s
        while multiple <= total:
            if multiple in sums_set:
                return False
            multiple += s
    return True

def _check_divisibility(sums_set, total):
    """Check if any sum divides another in sums_set."""
    sums_list = sorted(sums_set)
    for s in sums_list:
        if s == 0:
            continue
        multiple = 2 * s
        while multiple <= total:
            if multiple in sums_set:
                return False
            multiple += s
    return True

def _greedy_build(pool, deadline, rng=None, order=None):
    """Build a set greedily."""
    if rng is None:
        rng = random
    current = []
    current_sums = set()
    total = 0
    used = set()
    
    if order is None:
        remaining = list(pool)
        rng.shuffle(remaining)
    else:
        remaining = order[:]
        # Add slight randomness to fixed orders to avoid getting stuck in the same local optimum
        if rng is not None:
            # Only shuffle the tail or small chunks? No, full shuffle of the list is fine if we want diversity,
            # but if the order is strategic (e.g., large first), we might want to keep the strategy.
            # Let's just use the order as is, but the caller can provide different orders.
            pass
            
    for x in remaining:
        if time.time() > deadline:
            break
        if x in used:
            continue
        # Try adding x
        new_sums = set(current_sums)
        for s in list(current_sums):
            new_sums.add(s + x)
        new_sums.add(x)
        new_total = total + x
        if _check_divisibility(new_sums, new_total):
            current.append(x)
            current_sums = new_sums
            total = new_total
            used.add(x)
            if len(current) >= 30:
                break
    
    return current, current_sums, total

def _local_search(S, sums, total, pool, deadline, rng=None):
    """Enhanced local search: 1-swap, 2-swap, and additions."""
    if rng is None:
        rng = random
    current = list(S)
    cur_sums = set(sums)
    cur_total = total
    used = set(current)
    available = [x for x in pool if x not in used]
    
    improved = True
    no_improve_count = 0
    max_no_improve = 5
    
    while improved and time.time() < deadline:
        improved = False
        no_improve_count += 1
        if no_improve_count > max_no_improve:
            break
            
        # 1. Try adding elements
        # Sort available by a heuristic: prefer numbers that don't divide existing sums
        # We just shuffle for now to keep it simple and fast
        rng.shuffle(available)
        for x in available:
            if time.time() > deadline:
                break
            new_sums = set(cur_sums)
            for s in list(cur_sums):
                new_sums.add(s + x)
            new_sums.add(x)
            new_total = cur_total + x
            if _check_divisibility(new_sums, new_total):
                current.append(x)
                cur_sums = new_sums
                cur_total = new_total
                used.add(x)
                available.remove(x)
                improved = True
                no_improve_count = 0
                if len(current) >= 30:
                    return current, cur_sums, cur_total

        # 2. Try 1-swap (remove one, add one)
        if not improved and len(current) > 0:
            rng.shuffle(available)
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                removed = current[i]
                # Temporarily remove
                temp_current = current[:i] + current[i+1:]
                temp_sums = _subset_sums(temp_current)
                temp_total = sum(temp_current)
                
                for x in available[:50]: # Limit candidates for speed
                    if time.time() > deadline:
                        break
                    if x == removed:
                        continue
                    # Try adding x to temp_current
                    new_sums = set(temp_sums)
                    for s in list(temp_sums):
                        new_sums.add(s + x)
                    new_sums.add(x)
                    new_total = temp_total + x
                    if _check_divisibility(new_sums, new_total):
                        # Success
                        current = temp_current + [x]
                        cur_sums = new_sums
                        cur_total = new_total
                        used.discard(removed)
                        used.add(x)
                        available.remove(x)
                        available.append(removed)
                        improved = True
                        no_improve_count = 0
                        break
                if improved:
                    break

        # 3. Try 2-swap (remove two, add two) - more expensive, try less often
        if not improved and len(current) > 1 and rng.random() < 0.5:
            rng.shuffle(available)
            # Try a few random pairs
            for _ in range(10):
                if time.time() > deadline:
                    break
                if len(current) < 2:
                    break
                # Pick two indices to remove
                idx1, idx2 = rng.sample(range(len(current)), 2)
                rem1 = current[idx1]
                rem2 = current[idx2]
                
                # Create temp current without rem1, rem2
                temp_current = []
                for idx, val in enumerate(current):
                    if idx != idx1 and idx != idx2:
                        temp_current.append(val)
                
                temp_sums = _subset_sums(temp_current)
                temp_total = sum(temp_current)
                
                # Try to add two from available
                # This is O(N^2) in available, so limit it
                found = False
                for x1 in available[:20]:
                    if found: break
                    for x2 in available[:20]:
                        if x1 == x2: continue
                        if x1 == rem1 or x1 == rem2 or x2 == rem1 or x2 == rem2:
                            continue
                        # Try adding x1
                        new_sums1 = set(temp_sums)
                        for s in list(temp_sums):
                            new_sums1.add(s + x1)
                        new_sums1.add(x1)
                        if not _check_divisibility(new_sums1, temp_total + x1):
                            continue
                        # Try adding x2 to new_sums1
                        new_sums2 = set(new_sums1)
                        for s in list(new_sums1):
                            new_sums2.add(s + x2)
                        new_sums2.add(x2)
                        new_total = temp_total + x1 + x2
                        if _check_divisibility(new_sums2, new_total):
                            # Success
                            current = temp_current + [x1, x2]
                            cur_sums = new_sums2
                            cur_total = new_total
                            used.discard(rem1)
                            used.discard(rem2)
                            used.add(x1)
                            used.add(x2)
                            available.remove(x1)
                            available.remove(x2)
                            available.append(rem1)
                            available.append(rem2)
                            improved = True
                            no_improve_count = 0
                            found = True
                            break
                if improved:
                    break

    return current, cur_sums, cur_total

def construct():
    start_time = time.time()
    deadline = start_time + 88  # Safety margin
    
    pool = list(range(1, 257))
    
    best_set = []
    best_size = 0
    
    # Precompute some orders
    primes = [x for x in range(2, 257) if all(x % i != 0 for i in range(2, int(math.isqrt(x)) + 1))]
    non_primes = [x for x in pool if x not in set(primes)]
    
    def divisor_count(x):
        if x < 2:
            return 1
        count = 2
        for i in range(2, int(math.isqrt(x)) + 1):
            if x % i == 0:
                count += 2
        return count
        
    by_divisors = sorted(pool, key=lambda x: divisor_count(x))
    
    strategies = [
        ("random", None),
        ("large", sorted(pool, reverse=True)),
        ("small", sorted(pool)),
        ("primes", primes + non_primes),
        ("few_divisors", by_divisors),
        ("large_prime", [x for x in primes if x > 100] + [x for x in pool if x not in primes]),
    ]
    
    num_restarts = 60 # Increased restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        strat_name, order = strategies[restart % len(strategies)]
        rng = random.Random(restart * 7919 + 42)
        
        if order is None:
            candidates = pool[:]
            rng.shuffle(candidates)
        else:
            candidates = order[:]
            # Introduce some noise
            if restart % 3 == 0:
                rng.shuffle(candidates)
            elif restart % 3 == 1:
                # Shuffle last 50%
                mid = len(candidates) // 2
                tail = candidates[mid:]
                rng.shuffle(tail)
                candidates[mid:] = tail
            # else: keep order strict
        
        # Build greedily
        current, cur_sums, cur_total = _greedy_build(pool, deadline, rng, candidates)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
        
        # Local search
        if time.time() < deadline - 2:
            extended, ext_sums, ext_total = _local_search(
                current, cur_sums, cur_total, pool, deadline, rng
            )
            if len(extended) > best_size:
                best_size = len(extended)
                best_set = extended[:]
                
    # Final validation
    if best_set and not _is_valid(best_set):
        # Fallback: rebuild greedily with large first
        current, _, _ = _greedy_build(pool, deadline, random.Random(0), sorted(pool, reverse=True))
        best_set = current
    
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
