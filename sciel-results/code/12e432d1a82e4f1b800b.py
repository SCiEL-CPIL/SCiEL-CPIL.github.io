import time
import random
import math

_start_time = time.time()

def construct():
    deadline = _start_time + 85
    n = 12
    m = 40
    total_masks = 1 << n  # 4096

    # Precompute comparability as bitsets
    # comp_bits[a] is a Python int with bit b set iff a and b are comparable (one is subset of other)
    comp_bits = [0] * total_masks
    for a in range(total_masks):
        for b in range(a + 1, total_masks):
            if (a | b) == a or (a | b) == b:
                comp_bits[a] |= (1 << b)
                comp_bits[b] |= (1 << a)
    
    # Precompute global degree (number of comparable sets in the full universe)
    global_deg = [bin(comp_bits[a]).count('1') for a in range(total_masks)]

    def count_comparables(family):
        """Count comparable pairs in a list of masks."""
        cnt = 0
        for i in range(len(family)):
            for j in range(i + 1, len(family)):
                if (comp_bits[family[i]] >> family[j]) & 1:
                    cnt += 1
        return cnt

    def greedy_initialization():
        """
        Build an initial family by greedily adding the set that is comparable 
        to the most existing sets in the family.
        """
        family = []
        used = set()
        
        # Start with a maximal chain to ensure some structure
        # Chain: 0, 1, 3, 7, ..., (1<<12)-1
        chain = []
        mask = 0
        for i in range(n + 1):
            chain.append(mask)
            if i < n:
                mask |= (1 << i)
        
        # Randomize the order of adding the chain slightly or just add them
        # Actually, just adding the chain is a good start.
        for mask in chain:
            family.append(mask)
            used.add(mask)
            
        # Greedily add remaining sets
        # We want to pick a set that is comparable to many in 'family'
        # To speed up, we can just iterate all masks and pick the best
        # Since m=40 is small, and total_masks=4096, this is 40 * 4096 * 40 ops ~ 6M ops, which is fine.
        
        for _ in range(m - len(family)):
            best_mask = -1
            best_score = -1
            # Shuffle to break ties randomly
            candidates = [mask for mask in range(total_masks) if mask not in used]
            random.shuffle(candidates)
            
            for c in candidates:
                score = 0
                # Count comparables with current family
                for s in family:
                    if (comp_bits[c] >> s) & 1:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_mask = c
            
            if best_mask != -1:
                family.append(best_mask)
                used.add(best_mask)
            else:
                # Fallback: pick any unused
                for mask in range(total_masks):
                    if mask not in used:
                        family.append(mask)
                        used.add(mask)
                        break
                        
        return family

    def simulate_annealing(family, time_budget):
        """
        Simulated Annealing to maximize comparable pairs.
        """
        current_family = list(family)
        current_score = count_comparables(current_family)
        best_family = list(current_family)
        best_score = current_score
        
        # Precompute degrees within the family for faster delta calculation?
        # deg[i] = number of j != i such that family[i] ~ family[j]
        # We can recompute this when needed or maintain it.
        # For simplicity and speed in Python, let's just calculate delta on the fly.
        # Delta = deg_new_in_family - deg_old_in_family
        
        # Temperature schedule
        T = 10.0
        cooling = 0.999
        min_T = 0.1
        
        used = set(current_family)
        
        while time.time() < time_budget:
            if T < min_T:
                break
                
            # Pick a random index to replace
            i = random.randint(0, m - 1)
            old_mask = current_family[i]
            
            # Pick a random new mask not in used
            # To avoid infinite loop if we pick a used one, we can just try a few times
            # Or maintain a list of unused masks? 
            # Total masks 4096, used 40. Probability of picking unused is high.
            new_mask = -1
            while new_mask == -1 or new_mask in used:
                new_mask = random.randint(0, total_masks - 1)
                
            # Calculate delta
            # deg_old: number of comparables of old_mask with other m-1 elements
            deg_old = 0
            for j in range(m):
                if j != i and (comp_bits[old_mask] >> current_family[j]) & 1:
                    deg_old += 1
                    
            # deg_new: number of comparables of new_mask with other m-1 elements
            deg_new = 0
            for j in range(m):
                if j != i and (comp_bits[new_mask] >> current_family[j]) & 1:
                    deg_new += 1
                    
            delta = deg_new - deg_old
            
            if delta > 0 or random.random() < math.exp(delta / T):
                # Accept
                current_family[i] = new_mask
                used.remove(old_mask)
                used.add(new_mask)
                current_score += delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_family = list(current_family)
            
            T *= cooling
            
        return best_family, best_score

    # Phase 1: Greedy Initialization
    family = greedy_initialization()
    score = count_comparables(family)
    best_family = list(family)
    best_score = score

    # Phase 2: Simulated Annealing
    # Run multiple restarts or one long SA?
    # Given the time limit, one long SA with good cooling is usually better.
    # Or a few short SAs from different starts.
    # Let's do a few restarts.
    
    remaining_time = deadline - 10
    num_restarts = 5
    time_per_restart = remaining_time / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Perturb the best family found so far to create a new start
        # Or just restart from greedy?
        # Let's try perturbing the best known solution.
        start_family = list(best_family)
        
        # Perturb: swap 10% of elements randomly
        for _ in range(m // 10):
            i = random.randint(0, m - 1)
            j = random.randint(0, m - 1)
            if i != j:
                start_family[i], start_family[j] = start_family[j], start_family[i]
                
        # Ensure no duplicates in start_family (should be fine since we swap)
        
        t_start = time.time()
        t_end = min(deadline - 5, t_start + time_per_restart)
        
        res_family, res_score = simulate_annealing(start_family, t_end)
        
        if res_score > best_score:
            best_score = res_score
            best_family = res_family

    # Final Polish: Local Search on the best family found
    # Try 1-swaps to see if we can improve further
    final_family = list(best_family)
    final_score = best_score
    used = set(final_family)
    
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for i in range(m):
            if time.time() > deadline - 2:
                break
            old_mask = final_family[i]
            best_new = -1
            best_gain = 0
            
            # Calculate current degree of old_mask
            deg_old = 0
            for j in range(m):
                if j != i and (comp_bits[old_mask] >> final_family[j]) & 1:
                    deg_old += 1
            
            # Find best replacement
            # We can iterate over all unused masks, but that's 4000 checks.
            # 40 * 4000 = 160k checks per pass. Fast enough.
            for new_mask in range(total_masks):
                if new_mask == old_mask or new_mask in used:
                    continue
                
                deg_new = 0
                for j in range(m):
                    if j != i and (comp_bits[new_mask] >> final_family[j]) & 1:
                        deg_new += 1
                
                gain = deg_new - deg_old
                if gain > best_gain:
                    best_gain = gain
                    best_new = new_mask
                    
            if best_new != -1 and best_gain > 0:
                final_family[i] = best_new
                used.remove(old_mask)
                used.add(best_new)
                final_score += best_gain
                improved = True
                break # Restart scan

    best_family = final_family
    best_score = final_score

    # Convert to lists of ints
    result = []
    for mask in best_family:
        lst = [int(i) for i in range(n) if (mask >> i) & 1]
        result.append(lst)

    return result

if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    assert len(res) == 40
    seen = set()
    for r in res:
        fr = frozenset(r)
        assert fr not in seen
        seen.add(fr)
        for v in r:
            assert isinstance(v, int)
            assert 0 <= v <= 11
    masks = []
    for r in res:
        m = 0
        for v in r:
            m |= (1 << v)
        masks.append(m)
    cnt = 0
    for i in range(40):
        for j in range(i+1, 40):
            a, b = masks[i], masks[j]
            if a | b == a or a | b == b:
                cnt += 1
    print(f"Time: {t1-t0:.2f}s, Comparables: {cnt}")
