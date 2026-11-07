import time
import random

_start_time = time.time()

def construct():
    deadline = _start_time + 85
    n = 12
    m = 40
    total_masks = 1 << n  # 4096
    
    # Precompute comparability: for each mask, the list of masks it's comparable to
    # comp[a] is a set of b such that a|b==a or a|b==b
    # For efficiency, use a 2D bytearray matrix
    comp = [bytearray(total_masks) for _ in range(total_masks)]
    for a in range(total_masks):
        for b in range(a + 1, total_masks):
            if (a | b) == a or (a | b) == b:
                comp[a][b] = 1
                comp[b][a] = 1
    
    def is_comparable(a, b):
        return comp[a][b]
    
    def count_comparables(family):
        cnt = 0
        for i in range(len(family)):
            for j in range(i + 1, len(family)):
                if is_comparable(family[i], family[j]):
                    cnt += 1
        return cnt
    
    def get_degrees(family):
        degrees = [0] * len(family)
        for i in range(len(family)):
            for j in range(i + 1, len(family)):
                if is_comparable(family[i], family[j]):
                    degrees[i] += 1
                    degrees[j] += 1
        return degrees
    
    # Group masks by size
    masks_by_size = [[] for _ in range(n + 1)]
    for mask in range(total_masks):
        masks_by_size[bin(mask).count('1')].append(mask)
    
    # Generate all masks list
    all_masks = list(range(total_masks))
    
    def make_initial_family():
        """Create a good initial family using chain-based construction."""
        family = []
        seen = set()
        
        # Add empty set and full set
        family.append(0)
        seen.add(0)
        family.append(total_masks - 1)
        seen.add(total_masks - 1)
        
        # Add all singletons and co-singletons
        for mask in masks_by_size[1]:
            family.append(mask)
            seen.add(mask)
        for mask in masks_by_size[11]:
            family.append(mask)
            seen.add(mask)
        
        # Current: 2 + 12 + 12 = 26. Need 14 more.
        # Greedily add sets that maximize comparability with existing family
        candidates = []
        for sz in [2, 3, 9, 10]:
            candidates.extend(masks_by_size[sz])
        
        cand_scores = []
        for c in candidates:
            if c in seen:
                continue
            deg = 0
            for s in family:
                if is_comparable(c, s):
                    deg += 1
            cand_scores.append((deg, c))
        
        cand_scores.sort(key=lambda x: (-x[0], x[1]))
        
        for deg, c in cand_scores[:14]:
            family.append(c)
            seen.add(c)
        
        # Ensure exactly 40
        if len(family) > 40:
            family = family[:40]
            seen = set(family)
        while len(family) < 40:
            for mask in all_masks:
                if mask not in seen:
                    family.append(mask)
                    seen.add(mask)
                    break
        
        return family, seen
    
    def local_search(family, seen, score, max_time):
        """Perform local search to improve the family."""
        best_family = list(family)
        best_score = score
        best_seen = set(family)
        
        while time.time() < max_time:
            improved = False
            degrees = get_degrees(family)
            
            # Try swapping each element with a better candidate
            for i in range(len(family)):
                if time.time() > max_time:
                    break
                old_set = family[i]
                old_deg = degrees[i]
                
                best_new = None
                best_gain = 0
                
                # Generate candidates: neighbors in Hamming space + random
                candidates = set()
                
                # Hamming neighbors
                for bit in range(n):
                    new_m = old_set ^ (1 << bit)
                    if new_m not in seen:
                        candidates.add(new_m)
                
                # Neighbors of family members
                for s in family:
                    if s == old_set:
                        continue
                    for bit in range(n):
                        if (s >> bit) & 1:
                            new_m = s & ~(1 << bit)
                            if new_m != old_set and new_m not in seen:
                                candidates.add(new_m)
                        else:
                            new_m = s | (1 << bit)
                            if new_m != old_set and new_m not in seen:
                                candidates.add(new_m)
                
                # Random samples
                for _ in range(30):
                    candidates.add(random.choice(all_masks))
                
                # Sets of similar size
                old_size = bin(old_set).count('1')
                for sz in range(max(0, old_size - 2), min(n + 1, old_size + 3)):
                    if sz != old_size:
                        sample_size = min(10, len(masks_by_size[sz]))
                        if sample_size > 0:
                            for m in random.sample(masks_by_size[sz], sample_size):
                                if m not in seen and m != old_set:
                                    candidates.add(m)
                
                for new_set in candidates:
                    if new_set == old_set or new_set in seen:
                        continue
                    
                    new_deg = 0
                    for j, s in enumerate(family):
                        if j == i:
                            continue
                        if is_comparable(new_set, s):
                            new_deg += 1
                    
                    gain = new_deg - old_deg
                    if gain > best_gain:
                        best_gain = gain
                        best_new = new_set
                
                if best_new is not None:
                    family[i] = best_new
                    seen.remove(old_set)
                    seen.add(best_new)
                    score += best_gain
                    improved = True
                    
                    if score > best_score:
                        best_score = score
                        best_family = list(family)
                        best_seen = set(family)
                    
                    break  # Restart after one improvement
        
        return best_family, best_seen, best_score
    
    # Multi-restart local search
    best_overall_family = None
    best_overall_score = -1
    
    # First run with deterministic initial family
    family, seen = make_initial_family()
    score = count_comparables(family)
    
    if time.time() < deadline - 10:
        family, seen, score = local_search(family, seen, score, deadline - 5)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_family = list(family)
    
    # Additional restarts with randomized initial families
    remaining_time = deadline - 5 - time.time()
    if remaining_time > 5:
        num_restarts = max(1, int(remaining_time / 10))
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Create a randomized initial family
            family = []
            seen = set()
            
            # Randomly select 40 masks with preference for small/large sizes
            random.shuffle(all_masks)
            # Bias towards extreme sizes
            weighted_masks = []
            for mask in all_masks:
                sz = bin(mask).count('1')
                # Weight: higher for size 0,1,11,12; lower for middle sizes
                if sz <= 2 or sz >= 10:
                    weighted_masks.extend([mask] * 3)
                elif sz <= 4 or sz >= 8:
                    weighted_masks.extend([mask] * 2)
                else:
                    weighted_masks.append(mask)
            
            random.shuffle(weighted_masks)
            for mask in weighted_masks:
                if len(family) >= 40:
                    break
                if mask not in seen:
                    family.append(mask)
                    seen.add(mask)
            
            # Fill if needed
            while len(family) < 40:
                for mask in all_masks:
                    if mask not in seen:
                        family.append(mask)
                        seen.add(mask)
                        break
            
            score = count_comparables(family)
            
            # Local search with time limit
            time_limit = min(deadline - 5 - time.time(), remaining_time / (num_restarts - restart))
            if time_limit > 2:
                family, seen, score = local_search(family, seen, score, time.time() + time_limit)
            
            if score > best_overall_score:
                best_overall_score = score
                best_overall_family = list(family)
    
    # Final polish: try all possible single-element replacements
    if time.time() < deadline - 3 and best_overall_family:
        polish_family = list(best_overall_family)
        polish_seen = set(polish_family)
        polish_score = best_overall_score
        
        degrees = get_degrees(polish_family)
        for i in range(len(polish_family)):
            if time.time() > deadline - 2:
                break
            old_set = polish_family[i]
            old_deg = degrees[i]
            
            best_new = None
            best_gain = 0
            
            # Try all masks of same and adjacent sizes
            old_size = bin(old_set).count('1')
            for sz in range(max(0, old_size - 1), min(n + 1, old_size + 2)):
                for new_set in masks_by_size[sz]:
                    if new_set == old_set or new_set in polish_seen:
                        continue
                    
                    new_deg = 0
                    for j, s in enumerate(polish_family):
                        if j == i:
                            continue
                        if is_comparable(new_set, s):
                            new_deg += 1
                    
                    gain = new_deg - old_deg
                    if gain > best_gain:
                        best_gain = gain
                        best_new = new_set
            
            if best_new is not None:
                polish_family[i] = best_new
                polish_seen.remove(old_set)
                polish_seen.add(best_new)
                polish_score += best_gain
                
                if polish_score > best_overall_score:
                    best_overall_score = polish_score
                    best_overall_family = list(polish_family)
    
    # Convert to lists of ints
    result = []
    for mask in best_overall_family:
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
        for j in range(i + 1, 40):
            a, b = masks[i], masks[j]
            if a | b == a or a | b == b:
                cnt += 1
    print(f"Time: {t1 - t0:.2f}s, Comparables: {cnt}")
