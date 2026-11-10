import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 16
    points = list(range(n))
    
    # All possible triples
    all_triples = list(combinations(points, 3))
    # Index each triple by its 3 points
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    # For each point, which triples contain it
    point_triples = {p: [] for p in points}
    for t in all_triples:
        for p in t:
            point_triples[p].append(t)
    # For each pair, which triple contains it (unique if linear)
    pair_triple = {}
    for t in all_triples:
        for i in range(3):
            for j in range(i+1, 3):
                p1, p2 = t[i], t[j]
                key = (p1, p2) if p1 < p2 else (p2, p1)
                pair_triple[key] = t
    
    def get_alpha(triples):
        """Exact independence number via bitmask DP over subsets."""
        if not triples:
            return n
        # Build bitmask for each triple
        t_masks = []
        for t in triples:
            m = 0
            for p in t:
                m |= (1 << p)
            t_masks.append(m)
        
        # DP: dp[mask] = True if mask is independent (contains no triple)
        # We want max popcount(mask) where dp[mask] = True
        # dp[mask] = dp[mask without lowest set bit] AND (mask contains no triple)
        # Actually simpler: dp[mask] is True if for all triples, (mask & t_mask) != t_mask
        # We can compute incrementally.
        
        # Alternative: iterate all masks, check independence.
        # 2^16 = 65536 masks. For each, check against len(triples) masks.
        # Fast enough.
        
        max_ind = 0
        for mask in range(1 << n):
            independent = True
            for tm in t_masks:
                if (mask & tm) == tm:
                    independent = False
                    break
            if independent:
                cnt = bin(mask).count('1')
                if cnt > max_ind:
                    max_ind = cnt
                    if max_ind == n:
                        return n
        return max_ind
    
    def is_linear(triples):
        """Check if any pair is repeated."""
        seen_pairs = set()
        for t in triples:
            for i in range(3):
                for j in range(i+1, 3):
                    p1, p2 = t[i], t[j]
                    key = (p1, p2) if p1 < p2 else (p2, p1)
                    if key in seen_pairs:
                        return False
                    seen_pairs.add(key)
        return True
    
    def local_search(triples, deadline):
        """Try to improve alpha by swapping triples."""
        if not triples:
            return triples, get_alpha(triples)
        
        current_alpha = get_alpha(triples)
        best_alpha = current_alpha
        best_triples = triples[:]
        
        # Precompute used pairs
        used_pairs = set()
        for t in triples:
            for i in range(3):
                for j in range(i+1, 3):
                    p1, p2 = t[i], t[j]
                    key = (p1, p2) if p1 < p2 else (p2, p1)
                    used_pairs.add(key)
        
        # Candidate triples to swap in
        candidates = []
        for t in all_triples:
            if t in triples:
                continue
            # Check if t can be added (no pair conflict)
            can_add = True
            for i in range(3):
                for j in range(i+1, 3):
                    p1, p2 = t[i], t[j]
                    key = (p1, p2) if p1 < p2 else (p2, p1)
                    if key in used_pairs:
                        can_add = False
                        break
                if not can_add:
                    break
            if can_add:
                candidates.append(t)
        
        if not candidates:
            return best_triples, best_alpha
        
        # Try swaps: remove one triple, add one candidate
        # Simulated annealing style
        temperature = 0.5
        cooling = 0.995
        current = triples[:]
        cur_alpha = current_alpha
        
        max_steps = 5000
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            if step % 100 == 0:
                temperature *= cooling
            
            # Pick a random triple to remove
            idx = random.randint(0, len(current) - 1)
            removed = current[idx]
            
            # Pick a random candidate to add
            # Filter candidates that are compatible with current minus removed
            # Recompute used_pairs after removing
            new_used_pairs = set()
            for t in current:
                if t == removed:
                    continue
                for i in range(3):
                    for j in range(i+1, 3):
                        p1, p2 = t[i], t[j]
                        key = (p1, p2) if p1 < p2 else (p2, p1)
                        new_used_pairs.add(key)
            
            # Find candidates that can be added
            valid_cands = []
            for t in candidates:
                if t == removed:
                    continue
                can_add = True
                for i in range(3):
                    for j in range(i+1, 3):
                        p1, p2 = t[i], t[j]
                        key = (p1, p2) if p1 < p2 else (p2, p1)
                        if key in new_used_pairs:
                            can_add = False
                            break
                    if not can_add:
                        break
                if can_add:
                    valid_cands.append(t)
            
            if not valid_cands:
                continue
            
            added = random.choice(valid_cands)
            new_triples = current[:idx] + current[idx+1:] + [added]
            new_alpha = get_alpha(new_triples)
            
            # Accept if better, or with probability exp(-delta/temperature)
            delta = new_alpha - cur_alpha
            if delta < 0:
                current = new_triples
                cur_alpha = new_alpha
                if cur_alpha < best_alpha:
                    best_alpha = cur_alpha
                    best_triples = current[:]
            else:
                if random.random() < pow(2.718281828, -delta / max(temperature, 0.01)):
                    current = new_triples
                    cur_alpha = new_alpha
        
        return best_triples, best_alpha
    
    # Main: multiple random restarts with local search
    best_triples = []
    best_alpha = n
    
    # First, do a greedy construction to get a decent starting point
    def greedy_construct():
        triples = []
        used_pairs = set()
        shuffled = all_triples[:]
        random.shuffle(shuffled)
        for t in shuffled:
            valid = True
            for i in range(3):
                for j in range(i+1, 3):
                    p1, p2 = t[i], t[j]
                    key = (p1, p2) if p1 < p2 else (p2, p1)
                    if key in used_pairs:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                triples.append(t)
                for i in range(3):
                    for j in range(i+1, 3):
                        p1, p2 = t[i], t[j]
                        key = (p1, p2) if p1 < p2 else (p2, p1)
                        used_pairs.add(key)
        return triples
    
    # Run greedy + local search multiple times
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Greedy start
        triples = greedy_construct()
        if not triples:
            continue
        
        # Local search
        triples, alpha = local_search(triples, deadline)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples[:]
        
        if best_alpha <= 4:
            break
    
    # If we still have time, try more restarts with different strategies
    while time.time() < deadline - 5:
        triples = greedy_construct()
        if not triples:
            continue
        triples, alpha = local_search(triples, deadline)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples[:]
    
    # Convert to list of lists
    return [list(t) for t in best_triples]

if __name__ == "__main__":
    triples = construct()
    print(f"Found {len(triples)} triples")
    # Verify linearity
    used_pairs = set()
    for t in triples:
        for i in range(3):
            for j in range(i+1, 3):
                p1, p2 = t[i], t[j]
                key = (p1, p2) if p1 < p2 else (p2, p1)
                if key in used_pairs:
                    print("INVALID: Shared pair", t)
                    break
                used_pairs.add(key)
    else:
        print("Valid linear system")
