"""Optimized doubly-covered prefix: targeted greedy + bottleneck-driven genetic algorithm."""
import time
import random

# EVOLVE-BLOCK-START

def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i in range(n + 1) if sieve[i]]

_PRIMES = _primes_up_to(1000)
_NPRIMES = len(_PRIMES)

def _build_count(residues, L_MAX):
    """Build count array for x in [1, L_MAX]."""
    count = [0] * (L_MAX + 1)
    for i, p in enumerate(_PRIMES):
        r = residues[i]
        start = p if r == 0 else r
        for x in range(start, L_MAX + 1, p):
            count[x] += 1
    return count

def _find_prefix_L(count, L_MAX):
    """Find largest L such that count[x] >= 2 for all x in [1..L]."""
    L = 0
    for x in range(1, L_MAX + 1):
        if count[x] >= 2:
            L = x
        else:
            break
    return L

def _score(residues):
    """Compute the actual score (largest L with full double coverage)."""
    L_MAX = 100000
    count = _build_count(residues, L_MAX)
    return _find_prefix_L(count, L_MAX)

def _identify_bottleneck_primes(residues, count, gap, L_MAX):
    """
    Identify primes that cover the gap position and nearby under-covered positions.
    Returns a list of prime indices that are candidates for crossover.
    """
    candidates = set()
    
    # Primes that currently cover the gap
    for i, p in enumerate(_PRIMES):
        r = residues[i]
        if gap % p == r:
            candidates.add(i)
    
    # Primes that cover nearby under-covered positions
    for dx in range(-5, 6):
        x = gap + dx
        if 1 <= x <= L_MAX and count[x] < 2:
            for i, p in enumerate(_PRIMES):
                r = residues[i]
                if x % p == r:
                    candidates.add(i)
    
    # Also include small primes (they have high leverage)
    for i, p in enumerate(_PRIMES):
        if p <= 100:
            candidates.add(i)
    
    return sorted(candidates)

def _crossover(parent1, parent2, bottleneck_indices):
    """
    Crossover: take residues from parent1 for non-bottleneck primes,
    and randomly choose between parent1 and parent2 for bottleneck primes.
    """
    child = list(parent1)
    for i in bottleneck_indices:
        if random.random() < 0.5:
            child[i] = parent2[i]
    return child

def _greedy_construct(deadline):
    """
    Greedy construction: fix small primes for dense coverage, then repair gaps.
    Uses multi-candidate residue selection for speed.
    """
    residues = [0] * _NPRIMES
    L_MAX = 3000
    count = [0] * (L_MAX + 1)
    
    # Phase A: Assign small primes (p <= 100) to maximize coverage in [1, 2000]
    small_prime_indices = [i for i in range(_NPRIMES) if _PRIMES[i] <= 100]
    
    for idx in small_prime_indices:
        if time.time() > deadline:
            break
        p = _PRIMES[idx]
        best_r = 0
        best_score = -1
        
        # Try all residues for small primes (p <= 100 is manageable)
        for r in range(p):
            start = p if r == 0 else r
            gain = 0
            for x in range(start, 2001, p):
                if count[x] <= 1:
                    gain += 1
                elif count[x] == 2:
                    gain -= 0.1
            if gain > best_score:
                best_score = gain
                best_r = r
        
        residues[idx] = best_r
        start = p if best_r == 0 else best_r
        for x in range(start, L_MAX + 1, p):
            count[x] += 1
    
    # Phase B: For larger primes, use coverage deficit heuristic
    large_prime_indices = [i for i in range(_NPRIMES) if _PRIMES[i] > 100]
    
    current_L = 0
    while current_L + 1 <= L_MAX and count[current_L + 1] >= 2:
        current_L += 1
    
    for idx in large_prime_indices:
        if time.time() > deadline:
            break
        p = _PRIMES[idx]
        gap = current_L + 1
        
        # Generate candidate residues: focus on covering the gap and nearby positions
        candidates = set()
        candidates.add(gap % p)
        for dx in range(-3, 4):
            x = gap + dx
            if x >= 1:
                candidates.add(x % p)
        candidates.add(0)
        candidates.add(1)
        
        # Also try to maximize coverage of under-covered positions
        best_r = residues[idx]
        best_gain = -1
        
        for r in candidates:
            if r >= p:
                continue
            start = p if r == 0 else r
            gain = 0
            for x in range(start, L_MAX + 1, p):
                if count[x] <= 1:
                    gain += 1
            if gap % p == r:
                gain += 50  # Strong preference for covering the gap
            
            if gain > best_gain:
                best_gain = gain
                best_r = r
        
        residues[idx] = best_r
        start = p if best_r == 0 else best_r
        for x in range(start, L_MAX + 1, p):
            count[x] += 1
        
        # Extend current_L
        while current_L + 1 <= L_MAX and count[current_L + 1] >= 2:
            current_L += 1
    
    return residues, current_L

def _local_search_targeted(residues, deadline, L_MAX=5000):
    """
    Targeted local search: only consider flips that affect the bottleneck.
    Two-tier: first small primes, then larger primes if no improvement.
    """
    residues = list(residues)
    count = _build_count(residues, L_MAX)
    current_L = _find_prefix_L(count, L_MAX)
    
    improved = True
    stall_count = 0
    max_stall = 30
    tier = 0  # 0 = small primes, 1 = all primes
    
    while improved and time.time() < deadline:
        improved = False
        stall_count += 1
        if stall_count > max_stall:
            break
        
        gap = current_L + 1
        if gap > L_MAX:
            break
        
        if count[gap] >= 2:
            while gap <= L_MAX and count[gap] >= 2:
                current_L = gap
                gap += 1
            improved = True
            stall_count = 0
            tier = 0
            continue
        
        # Determine which primes to try based on tier
        if tier == 0:
            prime_range = [i for i in range(_NPRIMES) if _PRIMES[i] <= 200]
        else:
            prime_range = list(range(_NPRIMES))
        
        best_move = None
        best_gain = 0
        
        for i in prime_range:
            p = _PRIMES[i]
            current_r = residues[i]
            desired_r = gap % p
            
            if current_r == desired_r:
                continue
            
            # Check safety: all positions in [1, current_L] losing coverage must have count >= 3
            old_start = p if current_r == 0 else current_r
            safe = True
            for x in range(old_start, current_L + 1, p):
                if count[x] < 3:
                    safe = False
                    break
            
            if not safe:
                continue
            
            # Simulate the flip
            new_start = p if desired_r == 0 else desired_r
            
            for x in range(old_start, L_MAX + 1, p):
                count[x] -= 1
            for x in range(new_start, L_MAX + 1, p):
                count[x] += 1
            
            # Find new L
            new_L = current_L
            while new_L + 1 <= L_MAX and count[new_L + 1] >= 2:
                new_L += 1
            
            # Undo
            for x in range(new_start, L_MAX + 1, p):
                count[x] -= 1
            for x in range(old_start, L_MAX + 1, p):
                count[x] += 1
            
            gain = new_L - current_L
            if gain > best_gain:
                best_gain = gain
                best_move = (i, desired_r, new_L)
        
        # If no move found in current tier, switch to next tier
        if best_move is None and tier == 0:
            tier = 1
            stall_count = 0
            continue
        
        if best_move is None:
            # Try 2-flips among small primes
            small_idx = [i for i in range(_NPRIMES) if _PRIMES[i] <= 150]
            for i1 in range(len(small_idx)):
                if time.time() > deadline:
                    break
                i = small_idx[i1]
                p = _PRIMES[i]
                current_r1 = residues[i]
                desired_r1 = gap % p
                
                if current_r1 == desired_r1:
                    continue
                
                old_start1 = p if current_r1 == 0 else current_r1
                new_start1 = p if desired_r1 == 0 else desired_r1
                
                # Check safety for first flip
                safe1 = True
                for x in range(old_start1, current_L + 1, p):
                    if count[x] < 3:
                        safe1 = False
                        break
                if not safe1:
                    continue
                
                # Apply first flip
                for x in range(old_start1, L_MAX + 1, p):
                    count[x] -= 1
                for x in range(new_start1, L_MAX + 1, p):
                    count[x] += 1
                
                for i2 in range(i1 + 1, len(small_idx)):
                    j = small_idx[i2]
                    q = _PRIMES[j]
                    current_r2 = residues[j]
                    desired_r2 = gap % q
                    
                    if current_r2 == desired_r2:
                        continue
                    
                    old_start2 = q if current_r2 == 0 else current_r2
                    new_start2 = q if desired_r2 == 0 else desired_r2
                    
                    # Check safety for second flip (after first applied)
                    safe2 = True
                    for x in range(old_start2, current_L + 1, q):
                        if count[x] < 3:
                            safe2 = False
                            break
                    if not safe2:
                        continue
                    
                    # Apply second flip
                    for x in range(old_start2, L_MAX + 1, q):
                        count[x] -= 1
                    for x in range(new_start2, L_MAX + 1, q):
                        count[x] += 1
                    
                    new_L = current_L
                    while new_L + 1 <= L_MAX and count[new_L + 1] >= 2:
                        new_L += 1
                    
                    # Undo second flip
                    for x in range(new_start2, L_MAX + 1, q):
                        count[x] -= 1
                    for x in range(old_start2, L_MAX + 1, q):
                        count[x] += 1
                    
                    gain = new_L - current_L
                    if gain > best_gain:
                        best_gain = gain
                        best_move = (i, desired_r1, j, desired_r2, new_L)
                
                # Undo first flip
                for x in range(new_start1, L_MAX + 1, p):
                    count[x] -= 1
                for x in range(old_start1, L_MAX + 1, p):
                    count[x] += 1
        
        if best_move is not None:
            if len(best_move) == 3:
                i, new_r, new_L = best_move
                p = _PRIMES[i]
                old_r = residues[i]
                old_start = p if old_r == 0 else old_r
                new_start = p if new_r == 0 else new_r
                
                for x in range(old_start, L_MAX + 1, p):
                    count[x] -= 1
                for x in range(new_start, L_MAX + 1, p):
                    count[x] += 1
                
                residues[i] = new_r
                current_L = new_L
            else:
                i, r1, j, r2, new_L = best_move
                p = _PRIMES[i]
                q = _PRIMES[j]
                old_r1 = residues[i]
                old_r2 = residues[j]
                old_start1 = p if old_r1 == 0 else old_r1
                new_start1 = p if r1 == 0 else r1
                old_start2 = q if old_r2 == 0 else old_r2
                new_start2 = q if r2 == 0 else r2
                
                for x in range(old_start1, L_MAX + 1, p):
                    count[x] -= 1
                for x in range(new_start1, L_MAX + 1, p):
                    count[x] += 1
                for x in range(old_start2, L_MAX + 1, q):
                    count[x] -= 1
                for x in range(new_start2, L_MAX + 1, q):
                    count[x] += 1
                
                residues[i] = r1
                residues[j] = r2
                current_L = new_L
            
            improved = True
            stall_count = 0
            tier = 0  # Reset to small primes after improvement
    
    return residues, current_L

def _polish_pass(residues, deadline, L_MAX=10000):
    """
    Final polish pass with larger L_MAX to extend the prefix further.
    Tries all primes, not just small ones.
    """
    residues = list(residues)
    count = _build_count(residues, L_MAX)
    current_L = _find_prefix_L(count, L_MAX)
    
    improved = True
    stall_count = 0
    max_stall = 15
    
    while improved and time.time() < deadline:
        improved = False
        stall_count += 1
        if stall_count > max_stall:
            break
        
        gap = current_L + 1
        if gap > L_MAX:
            break
        
        if count[gap] >= 2:
            while gap <= L_MAX and count[gap] >= 2:
                current_L = gap
                gap += 1
            improved = True
            stall_count = 0
            continue
        
        # Try all primes, prioritizing small ones
        best_move = None
        best_gain = 0
        
        for i in range(_NPRIMES):
            p = _PRIMES[i]
            
            # Skip very large primes if time is tight
            if time.time() > deadline - 5 and p > 300:
                continue
            
            current_r = residues[i]
            desired_r = gap % p
            
            if current_r == desired_r:
                continue
            
            old_start = p if current_r == 0 else current_r
            safe = True
            for x in range(old_start, current_L + 1, p):
                if count[x] < 3:
                    safe = False
                    break
            
            if not safe:
                continue
            
            new_start = p if desired_r == 0 else desired_r
            
            for x in range(old_start, L_MAX + 1, p):
                count[x] -= 1
            for x in range(new_start, L_MAX + 1, p):
                count[x] += 1
            
            new_L = current_L
            while new_L + 1 <= L_MAX and count[new_L + 1] >= 2:
                new_L += 1
            
            for x in range(new_start, L_MAX + 1, p):
                count[x] -= 1
            for x in range(old_start, L_MAX + 1, p):
                count[x] += 1
            
            gain = new_L - current_L
            if gain > best_gain:
                best_gain = gain
                best_move = (i, desired_r, new_L)
        
        if best_move is not None:
            i, new_r, new_L = best_move
            p = _PRIMES[i]
            old_r = residues[i]
            old_start = p if old_r == 0 else old_r
            new_start = p if new_r == 0 else new_r
            
            for x in range(old_start, L_MAX + 1, p):
                count[x] -= 1
            for x in range(new_start, L_MAX + 1, p):
                count[x] += 1
            
            residues[i] = new_r
            current_L = new_L
            improved = True
            stall_count = 0
    
    return residues, current_L

def _genetic_search(population, deadline, L_MAX=5000, pop_size=20, generations=50):
    """
    Bottleneck-driven genetic algorithm.
    Selects parents based on performance at the current failure point,
    and applies crossover only to the subset of primes that cover the first
    integer with count < 2.
    """
    # Evaluate initial population
    scored_pop = []
    for res in population:
        count = _build_count(res, L_MAX)
        L = _find_prefix_L(count, L_MAX)
        scored_pop.append((L, res, count))
    
    scored_pop.sort(key=lambda x: -x[0])
    best_L = scored_pop[0][0]
    best_res = list(scored_pop[0][1])
    
    for gen in range(generations):
        if time.time() > deadline:
            break
        
        # Get the current best solution's count array
        best_count = scored_pop[0][2]
        gap = best_L + 1
        if gap > L_MAX:
            break
        
        # Identify bottleneck primes
        bottleneck_indices = _identify_bottleneck_primes(scored_pop[0][1], best_count, gap, L_MAX)
        
        # Create new generation
        new_pop = []
        
        # Elitism: keep top solutions
        for L, res, count in scored_pop[:pop_size//4]:
            new_pop.append((L, list(res), count))
        
        # Crossover + mutation
        while len(new_pop) < pop_size:
            if time.time() > deadline:
                break
            
            # Tournament selection: pick two parents biased toward top performers
            # Weight by performance at the gap
            weights = []
            for L, res, count in scored_pop:
                # Higher weight if this solution covers the gap
                if count[gap] >= 2:
                    w = 10
                else:
                    w = 1
                # Bonus for being in the top tier
                idx = scored_pop.index((L, res, count))
                if idx < pop_size // 2:
                    w *= 2
                weights.append(w)
            
            total_w = sum(weights)
            r1 = random.random() * total_w
            r2 = random.random() * total_w
            cum = 0
            p1_idx = 0
            for i, w in enumerate(weights):
                cum += w
                if cum >= r1:
                    p1_idx = i
                    break
            cum = 0
            p2_idx = 0
            for i, w in enumerate(weights):
                cum += w
                if cum >= r2:
                    p2_idx = i
                    break
            
            parent1 = scored_pop[p1_idx][1]
            parent2 = scored_pop[p2_idx][1]
            
            # Crossover on bottleneck primes
            child = _crossover(parent1, parent2, bottleneck_indices)
            
            # Mutation: flip a few random residues
            for _ in range(3):
                if random.random() < 0.3:
                    i = random.randint(0, _NPRIMES - 1)
                    p = _PRIMES[i]
                    child[i] = random.randint(0, p - 1)
            
            # Evaluate child
            count = _build_count(child, L_MAX)
            L = _find_prefix_L(count, L_MAX)
            new_pop.append((L, child, count))
        
        # Sort new population
        new_pop.sort(key=lambda x: -x[0])
        
        # Update best
        if new_pop[0][0] > best_L:
            best_L = new_pop[0][0]
            best_res = list(new_pop[0][1])
        
        scored_pop = new_pop
    
    return best_res, best_L

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Phase 1: Greedy construction
    residues, L = _greedy_construct(deadline - 80)
    best_residues = list(residues)
    best_L = L
    
    # Phase 2: Targeted local search
    if time.time() < deadline - 40:
        residues, L = _local_search_targeted(residues, deadline - 40, L_MAX=5000)
        if L > best_L:
            best_L = L
            best_residues = list(residues)
    
    # Phase 3: Genetic search with multiple initial solutions
    if time.time() < deadline - 10:
        # Create diverse initial population
        population = [list(best_residues)]
        
        # Add variations
        for _ in range(19):
            variant = list(best_residues)
            # Randomly perturb some residues
            for i in range(random.randint(5, 15)):
                idx = random.randint(0, _NPRIMES - 1)
                p = _PRIMES[idx]
                variant[idx] = random.randint(0, p - 1)
            population.append(variant)
        
        residues, L = _genetic_search(population, deadline - 10, L_MAX=5000, pop_size=20, generations=50)
        if L > best_L:
            best_L = L
            best_residues = list(residues)
    
    # Phase 4: Polish pass with larger range
    if time.time() < deadline - 5:
        residues, L = _polish_pass(best_residues, deadline - 5, L_MAX=10000)
        if L > best_L:
            best_L = L
            best_residues = list(residues)
    
    # Final verification
    final_L = _score(best_residues)
    if final_L > best_L:
        best_L = final_L
    
    return best_residues

# EVOLVE-BLOCK-END
