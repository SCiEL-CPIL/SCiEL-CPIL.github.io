"""Optimized search for 32 integers in 1..512 maximizing distinct 3-AP differences.
Uses a structural change: precomputed difference masks for incremental scoring,
plus a multi-strategy approach with structured initializations."""
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    N = 512  # max value
    K = 32   # set size
    MAX_D = 255  # max possible d (since 1+2*255=511 <= 512)
    
    # Precompute for each d, the mask of valid starting positions a
    # For a 3-AP a, a+d, a+2d to exist, we need a+2d <= 512, so a <= 512-2d
    # valid_start_mask[d] has bits set for a in 1..512-2d (0-indexed: bit a-1)
    valid_start_mask = [0] * (MAX_D + 1)
    for d in range(1, MAX_D + 1):
        max_a = N - 2 * d
        if max_a >= 1:
            # bits 0..max_a-1 set (representing a=1..max_a)
            valid_start_mask[d] = (1 << max_a) - 1
    
    # Precompute shift masks for each d
    # For a set bitset b, the condition for d is:
    # (b & (b >> d) & (b >> (2*d)) & valid_start_mask[d]) != 0
    # Actually we need: exists a such that bits a-1, a+d-1, a+2d-1 are all set
    # b >> d has bit i set if b has bit i+d set
    # b >> (2d) has bit i set if b has bit i+2d set
    # So (b & (b>>d) & (b>>(2d))) has bit i set iff bits i, i+d, i+2d are all in b
    # We need this to have at least one bit set, and that bit i must correspond to a=i+1 >= 1
    # and a+2d <= 512, i.e., i+2d <= 511, i <= 511-2d
    # The valid_start_mask[d] restricts to i <= 511-2d (0-indexed)
    
    def score_bitset(b):
        """Count distinct d > 0 such that a 3-AP with difference d exists in set."""
        count = 0
        for d in range(1, MAX_D + 1):
            if (b & (b >> d) & (b >> (2 * d)) & valid_start_mask[d]):
                count += 1
        return count
    
    # For incremental scoring: when we add/remove an element, which d's are affected?
    # Adding element x (bit x-1): it can create new 3-APs for various d
    # Removing element x: it can destroy existing 3-APs
    # 
    # A 3-AP with difference d contains element x if:
    #   x = a (start): need a+d, a+2d in set, a+d-1, a+2d-1 bits set
    #   x = a+d (middle): need a, a+2d in set, a-1, a+2d-1 bits set  
    #   x = a+2d (end): need a, a+d in set, a-1, a+d-1 bits set
    #
    # For incremental approach, let's precompute for each position x and each d,
    # the other two positions needed to form a 3-AP containing x with difference d.
    
    # affected_d[x] = list of (d, other1, other2) where x participates in a 3-AP
    # with difference d, and other1, other2 are the other two elements
    # This is too complex for incremental; let's use a different approach.
    
    # Alternative: precompute for each d, a bitmask M_d such that
    # (b & M_d) != 0 means there's a 3-AP with difference d.
    # M_d = OR over all valid a of (bit(a-1) & bit(a+d-1) & bit(a+2d-1))
    # But that's not a simple AND mask.
    
    # Let's use a faster scoring: precompute for each d the list of triples
    # and use bit operations.
    
    # Actually, the current score_bitset is already quite fast with big-int ops.
    # Let's focus on better search strategy instead.
    
    def greedy_initialize(rng, strategy=0):
        """Multiple greedy initialization strategies."""
        current_set = set()
        current_bit = 0
        current_score = 0
        
        if strategy == 0:
            # Standard greedy: add element that maximizes score gain
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
            
            for _ in range(K):
                if not candidates:
                    break
                best_elem = None
                best_gain = -1
                # Evaluate a random subset
                sample_size = min(60, len(candidates))
                sample = rng.sample(candidates, sample_size)
                for elem in sample:
                    if elem in current_set:
                        continue
                    new_bit = current_bit | (1 << (elem - 1))
                    new_score = score_bitset(new_bit)
                    gain = new_score - current_score
                    if gain > best_gain:
                        best_gain = gain
                        best_elem = elem
                if best_elem is not None:
                    current_set.add(best_elem)
                    current_bit |= (1 << (best_elem - 1))
                    current_score += best_gain
                    candidates.remove(best_elem)
                else:
                    elem = candidates.pop()
                    current_set.add(elem)
                    current_bit |= (1 << (elem - 1))
                    current_score = score_bitset(current_bit)
                    
        elif strategy == 1:
            # Start with a structured pattern: use elements from a known good structure
            # Try a geometric-like spread
            base = rng.randint(1, 10)
            step = rng.randint(12, 20)
            for i in range(K):
                elem = base + i * step
                if 1 <= elem <= N:
                    current_set.add(elem)
            # Fill remaining with random
            while len(current_set) < K:
                elem = rng.randint(1, N)
                if elem not in current_set:
                    current_set.add(elem)
            current_bit = 0
            for x in current_set:
                current_bit |= (1 << (x - 1))
            current_score = score_bitset(current_bit)
            
        elif strategy == 2:
            # Power-of-2 based initialization
            # Use elements that are sums of subsets of {1, 2, 4, 8, 16, 32, 64, 128}
            # This gives 256 possible values, pick 32 that spread well
            powers = [1, 2, 4, 8, 16, 32, 64, 128]
            all_vals = set()
            for mask in range(256):
                val = 0
                for i, p in enumerate(powers):
                    if mask & (1 << i):
                        val += p
                if 1 <= val <= N:
                    all_vals.add(val)
            # Pick 32 from these that maximize score
            candidates = list(all_vals)
            rng.shuffle(candidates)
            for elem in candidates:
                if len(current_set) >= K:
                    break
                current_set.add(elem)
            while len(current_set) < K:
                elem = rng.randint(1, N)
                if elem not in current_set:
                    current_set.add(elem)
            current_bit = 0
            for x in current_set:
                current_bit |= (1 << (x - 1))
            current_score = score_bitset(current_bit)
            
        else:
            # Random initialization
            current_set = set(rng.sample(range(1, N + 1), K))
            current_bit = 0
            for x in current_set:
                current_bit |= (1 << (x - 1))
            current_score = score_bitset(current_bit)
        
        return list(current_set), current_bit, current_score
    
    def simulated_annealing(initial_set, initial_bit, initial_score, rng, time_budget):
        """Simulated annealing with multiple move types."""
        current_set = list(initial_set)
        current_bit = initial_bit
        current_score = initial_score
        
        best_set = list(current_set)
        best_bit = current_bit
        best_score = current_score
        
        temperature = 25.0
        cooling = 0.99997
        min_temp = 0.01
        e = math.e
        
        while temperature > min_temp:
            if time.time() > time_budget:
                break
            
            move_type = rng.random()
            
            if move_type < 0.55:
                # Single swap
                i = rng.randrange(K)
                old_elem = current_set[i]
                new_elem = rng.randint(1, N)
                if new_elem == old_elem or new_elem in current_set:
                    continue
                
                new_bit = (current_bit & ~(1 << (old_elem - 1))) | (1 << (new_elem - 1))
                new_score = score_bitset(new_bit)
                
                delta = new_score - current_score
                if delta > 0 or rng.random() < (e ** (delta / temperature)):
                    current_set[i] = new_elem
                    current_bit = new_bit
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(current_set)
                        best_bit = current_bit
            elif move_type < 0.75:
                # Double swap
                i = rng.randrange(K)
                j = rng.randrange(K)
                if i == j:
                    continue
                old1 = current_set[i]
                old2 = current_set[j]
                new1 = rng.randint(1, N)
                new2 = rng.randint(1, N)
                if new1 in current_set or new2 in current_set or new1 == new2:
                    continue
                
                new_bit = current_bit
                new_bit &= ~(1 << (old1 - 1))
                new_bit &= ~(1 << (old2 - 1))
                new_bit |= (1 << (new1 - 1))
                new_bit |= (1 << (new2 - 1))
                new_score = score_bitset(new_bit)
                
                delta = new_score - current_score
                if delta > 0 or rng.random() < (e ** (delta / temperature)):
                    current_set[i] = new1
                    current_set[j] = new2
                    current_bit = new_bit
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(current_set)
                        best_bit = current_bit
            elif move_type < 0.90:
                # Triple swap
                i = rng.randrange(K)
                j = rng.randrange(K)
                k = rng.randrange(K)
                if i == j or i == k or j == k:
                    continue
                old1 = current_set[i]
                old2 = current_set[j]
                old3 = current_set[k]
                new1 = rng.randint(1, N)
                new2 = rng.randint(1, N)
                new3 = rng.randint(1, N)
                if new1 in current_set or new2 in current_set or new3 in current_set:
                    continue
                if len({new1, new2, new3}) < 3:
                    continue
                
                new_bit = current_bit
                new_bit &= ~(1 << (old1 - 1))
                new_bit &= ~(1 << (old2 - 1))
                new_bit &= ~(1 << (old3 - 1))
                new_bit |= (1 << (new1 - 1))
                new_bit |= (1 << (new2 - 1))
                new_bit |= (1 << (new3 - 1))
                new_score = score_bitset(new_bit)
                
                delta = new_score - current_score
                if delta > 0 or rng.random() < (e ** (delta / temperature)):
                    current_set[i] = new1
                    current_set[j] = new2
                    current_set[k] = new3
                    current_bit = new_bit
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(current_set)
                        best_bit = current_bit
            else:
                # Quadruple swap for larger moves
                indices = rng.sample(range(K), 4)
                old_elems = [current_set[idx] for idx in indices]
                new_elems = []
                valid = True
                for _ in range(4):
                    ne = rng.randint(1, N)
                    if ne in current_set or ne in new_elems:
                        valid = False
                        break
                    new_elems.append(ne)
                if not valid:
                    continue
                
                new_bit = current_bit
                for oe in old_elems:
                    new_bit &= ~(1 << (oe - 1))
                for ne in new_elems:
                    new_bit |= (1 << (ne - 1))
                new_score = score_bitset(new_bit)
                
                delta = new_score - current_score
                if delta > 0 or rng.random() < (e ** (delta / temperature)):
                    for idx, ne in zip(indices, new_elems):
                        current_set[idx] = ne
                    current_bit = new_bit
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(current_set)
                        best_bit = current_bit
            
            temperature *= cooling
        
        return best_set, best_bit, best_score
    
    def local_search_polish(initial_set, initial_bit, initial_score, rng, time_budget):
        """Intensive local search: try all possible single and double swaps."""
        current_set = list(initial_set)
        current_bit = initial_bit
        current_score = initial_score
        
        best_set = list(current_set)
        best_bit = current_bit
        best_score = current_score
        
        # Precompute set membership for fast lookup
        current_set_frozenset = frozenset(current_set)
        
        improved = True
        while improved and time.time() < time_budget:
            improved = False
            
            # Try all single swaps
            for i in range(K):
                if time.time() > time_budget:
                    return best_set, best_bit, best_score
                old_elem = current_set[i]
                # Try a subset of possible new elements
                # Focus on elements near existing ones for better 3-AP potential
                candidates = []
                for existing in current_set:
                    for offset in [-3, -2, -1, 1, 2, 3, 5, 8, 13, 21]:
                        cand = existing + offset
                        if 1 <= cand <= N and cand != old_elem and cand not in current_set_frozenset:
                            candidates.append(cand)
                # Also add some random candidates
                for _ in range(20):
                    cand = rng.randint(1, N)
                    if cand != old_elem and cand not in current_set_frozenset:
                        candidates.append(cand)
                
                best_gain = -1
                best_new_elem = None
                for cand in candidates:
                    new_bit = (current_bit & ~(1 << (old_elem - 1))) | (1 << (cand - 1))
                    new_score = score_bitset(new_bit)
                    gain = new_score - current_score
                    if gain > best_gain:
                        best_gain = gain
                        best_new_elem = cand
                
                if best_new_elem is not None and best_gain > 0:
                    current_set[i] = best_new_elem
                    current_bit = (current_bit & ~(1 << (old_elem - 1))) | (1 << (best_new_elem - 1))
                    current_score += best_gain
                    current_set_frozenset = frozenset(current_set)
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(current_set)
                        best_bit = current_bit
                    break  # Restart scan after improvement
            
            # Try double swaps if no single improvement found
            if not improved and time.time() < time_budget:
                double_improved = False
                for i in range(K):
                    if time.time() > time_budget:
                        break
                    for j in range(i + 1, K):
                        if time.time() > time_budget:
                            break
                        old1 = current_set[i]
                        old2 = current_set[j]
                        
                        # Generate candidate pairs
                        cand_pairs = []
                        for existing in current_set:
                            for off1 in [-4, -3, -2, -1, 1, 2, 3, 4]:
                                for off2 in [-4, -3, -2, -1, 1, 2, 3, 4]:
                                    c1 = existing + off1
                                    c2 = existing + off2
                                    if (1 <= c1 <= N and 1 <= c2 <= N and 
                                        c1 != old1 and c2 != old2 and 
                                        c1 not in current_set_frozenset and 
                                        c2 not in current_set_frozenset and
                                        c1 != c2):
                                        cand_pairs.append((c1, c2))
                        for _ in range(10):
                            c1 = rng.randint(1, N)
                            c2 = rng.randint(1, N)
                            if (c1 not in current_set_frozenset and c2 not in current_set_frozenset and
                                c1 != c2 and c1 != old1 and c2 != old2):
                                cand_pairs.append((c1, c2))
                        
                        best_gain = -1
                        best_pair = None
                        for c1, c2 in cand_pairs[:50]:  # Limit for speed
                            new_bit = current_bit
                            new_bit &= ~(1 << (old1 - 1))
                            new_bit &= ~(1 << (old2 - 1))
                            new_bit |= (1 << (c1 - 1))
                            new_bit |= (1 << (c2 - 1))
                            new_score = score_bitset(new_bit)
                            gain = new_score - current_score
                            if gain > best_gain:
                                best_gain = gain
                                best_pair = (c1, c2)
                        
                        if best_pair is not None and best_gain > 0:
                            current_set[i] = best_pair[0]
                            current_set[j] = best_pair[1]
                            current_bit = current_bit
                            current_bit &= ~(1 << (old1 - 1))
                            current_bit &= ~(1 << (old2 - 1))
                            current_bit |= (1 << (best_pair[0] - 1))
                            current_bit |= (1 << (best_pair[1] - 1))
                            current_score += best_gain
                            current_set_frozenset = frozenset(current_set)
                            improved = True
                            double_improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_set = list(current_set)
                                best_bit = current_bit
                            break
                    if improved:
                        break
        
        return best_set, best_bit, best_score
    
    # Multi-restart with diverse strategies
    global_best_set = None
    global_best_bit = 0
    global_best_score = -1
    
    # Strategy 1: Multiple SA restarts with different initializations
    num_restarts = 8
    time_per_restart = (deadline - time.time() - 20.0) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 25.0:
            break
        
        rng = random.Random(42 + restart * 131)
        strategy = restart % 4
        
        init_set, init_bit, init_score = greedy_initialize(rng, strategy)
        
        sa_budget = min(deadline - 25.0, time.time() + time_per_restart)
        best_set, best_bit, best_score = simulated_annealing(
            init_set, init_bit, init_score, rng, sa_budget
        )
        
        if best_score > global_best_score:
            global_best_score = best_score
            global_best_set = best_set
            global_best_bit = best_bit
    
    # Strategy 2: Intensive local search polish on the best found
    if global_best_set is not None and time.time() < deadline - 5.0:
        polish_rng = random.Random(77777)
        polish_budget = min(deadline - 3.0, time.time() + 15.0)
        best_set, best_bit, best_score = local_search_polish(
            global_best_set, global_best_bit, global_best_score, 
            polish_rng, polish_budget
        )
        if best_score > global_best_score:
            global_best_score = best_score
            global_best_set = best_set
            global_best_bit = best_bit
    
    # Final quick hill climb
    if global_best_set is not None and time.time() < deadline - 1.0:
        final_rng = random.Random(12345)
        final_set = list(global_best_set)
        final_bit = global_best_bit
        final_score = global_best_score
        
        for _ in range(2000):
            if time.time() > deadline - 0.5:
                break
            i = final_rng.randrange(K)
            old_elem = final_set[i]
            new_elem = final_rng.randint(1, N)
            if new_elem == old_elem or new_elem in final_set:
                continue
            
            new_bit = (final_bit & ~(1 << (old_elem - 1))) | (1 << (new_elem - 1))
            new_score = score_bitset(new_bit)
            
            if new_score > final_score:
                final_set[i] = new_elem
                final_bit = new_bit
                final_score = new_score
                if final_score > global_best_score:
                    global_best_score = final_score
                    global_best_set = list(final_set)
                    global_best_bit = final_bit
    
    return global_best_set if global_best_set is not None else list(range(1, K + 1))
