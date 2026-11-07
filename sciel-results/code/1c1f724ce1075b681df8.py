import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _count_comparable(masks):
    """Count unordered pairs where one is a strict subset of the other."""
    n = len(masks)
    cnt = 0
    for i in range(n):
        mi = masks[i]
        for j in range(i + 1, n):
            mj = masks[j]
            # strict subset: mi < mj means mi & mj == mi and mi != mj
            if (mi & mj) == mi and mi != mj:
                cnt += 1
            elif (mi & mj) == mj and mi != mj:
                cnt += 1
    return cnt

def _local_search(masks, deadline):
    """Hill-climb by replacing individual sets to maximize comparable pairs."""
    current = list(masks)
    best_score = _count_comparable(current)
    n_sets = len(current)
    n_vertices = 12
    N = 1 << n_vertices
    
    # Precompute for speed: for each mask, which other masks it's comparable with
    # We'll just recompute score on each move since n_sets=40, pairs=780
    
    while time.time() < deadline:
        improved = False
        # Shuffle order of positions to avoid getting stuck in a particular order
        order = list(range(n_sets))
        random.shuffle(order)
        
        for pos in order:
            if time.time() >= deadline:
                break
            old_mask = current[pos]
            best_new_mask = old_mask
            best_new_score = best_score
            
            # Try random candidate masks
            # Focus on masks that are subsets or supersets of many existing sets
            # Try all possible masks? 4096 * 780 * 40 = too many
            # Try random masks and also masks derived from neighbors
            
            # Generate candidates:
            # 1. Random masks
            # 2. Masks that are subsets/supersets of current set
            # 3. Random masks of similar size
            
            candidates = set()
            
            # Add some random masks
            for _ in range(50):
                candidates.add(random.randint(0, N - 1))
            
            # Add subsets and supersets of old_mask
            # Subsets of old_mask
            sub = old_mask
            while True:
                candidates.add(sub)
                if sub == 0:
                    break
                sub = (sub - 1) & old_mask
            
            # Supersets of old_mask
            # Supersets: old_mask | x for any x
            # There are 2^(12-popcount(old_mask)) supersets
            popcount = bin(old_mask).count('1')
            if popcount <= 10:  # Limit to avoid too many
                # Iterate over all supersets
                remaining = (~old_mask) & (N - 1)
                sub = remaining
                while True:
                    candidates.add(old_mask | sub)
                    if sub == 0:
                        break
                    sub = (sub - 1) & remaining
            
            # Also try masks that differ by one bit from old_mask
            for bit in range(n_vertices):
                candidates.add(old_mask ^ (1 << bit))
            
            # Evaluate candidates
            for cand in candidates:
                if cand == old_mask:
                    continue
                # Quick check: compute score incrementally? 
                # For simplicity, just recompute full score
                # But that's 780 comparisons per candidate, and we have ~100+ candidates
                # Let's be smarter: only compute the change in score
                
                # Score change = (new pairs involving pos) - (old pairs involving pos)
                # Old pairs involving pos:
                old_pairs = 0
                for j in range(n_sets):
                    if j == pos:
                        continue
                    mj = current[j]
                    if (old_mask & mj) == old_mask and old_mask != mj:
                        old_pairs += 1
                    elif (old_mask & mj) == mj and old_mask != mj:
                        old_pairs += 1
                
                new_pairs = 0
                for j in range(n_sets):
                    if j == pos:
                        continue
                    mj = current[j]
                    if (cand & mj) == cand and cand != mj:
                        new_pairs += 1
                    elif (cand & mj) == mj and cand != mj:
                        new_pairs += 1
                
                delta = new_pairs - old_pairs
                if delta > 0:
                    if delta > best_new_score - best_score:  # Wait, best_new_score is best_score + delta
                        # Actually best_new_score should track the best score found
                        pass
                # Let me redo this logic
                
                new_total = best_score - old_pairs + new_pairs
                if new_total > best_new_score:
                    best_new_score = new_total
                    best_new_mask = cand
            
            if best_new_mask != old_mask:
                current[pos] = best_new_mask
                best_score = best_new_score
                improved = True
        
        if not improved:
            # Try a few random restarts within the local search
            # Or just break
            # Let's try one more pass with different random candidates
            any_improved = False
            for _ in range(5):
                if time.time() >= deadline:
                    break
                for pos in range(n_sets):
                    old_mask = current[pos]
                    best_new_mask = old_mask
                    best_new_score = best_score
                    old_pairs = 0
                    for j in range(n_sets):
                        if j == pos:
                            continue
                        mj = current[j]
                        if (old_mask & mj) == old_mask and old_mask != mj:
                            old_pairs += 1
                        elif (old_mask & mj) == mj and old_mask != mj:
                            old_pairs += 1
                    
                    for _ in range(20):
                        cand = random.randint(0, N - 1)
                        if cand == old_mask:
                            continue
                        new_pairs = 0
                        for j in range(n_sets):
                            if j == pos:
                                continue
                            mj = current[j]
                            if (cand & mj) == cand and cand != mj:
                                new_pairs += 1
                            elif (cand & mj) == mj and cand != mj:
                                new_pairs += 1
                        new_total = best_score - old_pairs + new_pairs
                        if new_total > best_new_score:
                            best_new_score = new_total
                            best_new_mask = cand
                    
                    if best_new_mask != old_mask:
                        current[pos] = best_new_mask
                        best_score = best_new_score
                        any_improved = True
            
            if not any_improved:
                break
    
    return current, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin
    
    n_vertices = 12
    m = 40
    N = 1 << n_vertices
    
    # Initial strategy: pick sets that form a chain or near-chain
    # A chain of 40 sets would give C(40,2) = 780 comparable pairs
    # But max chain length in 12 elements is 13 (sizes 0 to 12)
    # So we can't have a full chain of 40.
    
    # Strategy: use multiple chains or a "layered" approach
    # Pick sets of various sizes, ensuring many subset relations
    
    # Initial: random sets
    random.seed(42)
    masks = [random.randint(0, N - 1) for _ in range(m)]
    # Ensure distinct
    while len(set(masks)) < m:
        masks[random.randrange(m)] = random.randint(0, N - 1)
    
    best_masks = list(masks)
    best_score = _count_comparable(best_masks)
    
    # Try a few different initializations
    # 1. Chain-based: pick sets from different levels
    # Levels: size 0 to 12
    # We need 40 sets. Let's pick multiple sets from each level.
    # For maximum comparability, we want sets that are subsets of each other.
    # A good heuristic: pick sets that are "nested" as much as possible.
    
    # Alternative initialization: greedy construction
    # Start with empty set, then add sets that are supersets of existing ones
    greedy_masks = [0]  # start with empty set
    while len(greedy_masks) < m and time.time() < deadline:
        # Find a mask that is a superset of many existing sets
        best_mask = -1
        best_count = -1
        for _ in range(100):  # random samples
            cand = random.randint(0, N - 1)
            if cand in greedy_masks:
                continue
            count = 0
            for existing in greedy_masks:
                if (cand & existing) == existing and cand != existing:
                    count += 1
            if count > best_count:
                best_count = count
                best_mask = cand
        if best_mask >= 0:
            greedy_masks.append(best_mask)
        else:
            # Just add a random distinct mask
            while True:
                cand = random.randint(0, N - 1)
                if cand not in greedy_masks:
                    greedy_masks.append(cand)
                    break
    
    score_greedy = _count_comparable(greedy_masks)
    if score_greedy > best_score:
        best_score = score_greedy
        best_masks = list(greedy_masks)
    
    # Run local search from best initialization
    if time.time() < deadline:
        best_masks, best_score = _local_search(best_masks, deadline)
    
    # Convert masks to lists of integers
    result = []
    for mask in best_masks:
        s = [b for b in range(n_vertices) if (mask >> b) & 1]
        result.append(s)
    
    return result

# EVOLVE-BLOCK-END
