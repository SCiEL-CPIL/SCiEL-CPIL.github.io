import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # Precompute LCMs or use math.lcm
    # We will evaluate sets of the form {k} U {x in [129, 256] | lcm(k, x) > 256}
    # And potentially extend them.
    
    best_set = []
    best_score = 0.0
    
    # Candidate anchors
    for k in range(2, 257):
        if time.time() > deadline:
            break
            
        # Find compatible x in [129, 256]
        compatible = []
        for x in range(129, 257):
            if x == k:
                continue
            if math.lcm(k, x) > 256:
                compatible.append(x)
        
        # Check if compatible set is a clique
        # We assume [129, 256] is mostly a clique.
        # Let's verify pairwise for the subset.
        is_clique = True
        n = len(compatible)
        for i in range(n):
            for j in range(i+1, n):
                if math.lcm(compatible[i], compatible[j]) <= 256:
                    is_clique = False
                    break
            if not is_clique:
                break
        
        if not is_clique:
            # If not a clique, we could solve a smaller optimization, 
            # but for now, skip or use greedy.
            # Given the structure, it's likely a clique or close to it.
            # For k=2, it's odds. Odds > 128 are clique.
            # For k=3, non-multiples of 3.
            # Let's just take the clique part greedily if needed.
            # But for this first pass, we assume it's valid or we filter it.
            # Actually, if not clique, we can just pick a subset.
            # Let's stick to the full set if it's a clique.
            # If not, we can try to fix it.
            # For now, let's just calculate score if it were valid, 
            # but we must return a valid set.
            # Let's use a greedy clique finder on 'compatible' + [k].
            
            # Greedy: sort by 1/x desc, add if compatible with all selected.
            candidates = [k] + compatible
            candidates.sort(key=lambda x: -1.0/x)
            current_set = []
            for x in candidates:
                if all(math.lcm(x, y) > 256 for y in current_set):
                    current_set.append(x)
            
            score = sum(1.0/x for x in current_set)
            if score > best_score:
                best_score = score
                best_set = current_set
        else:
            current_set = [k] + compatible
            score = sum(1.0/x for x in current_set)
            if score > best_score:
                best_score = score
                best_set = current_set

    # Local search improvement on best_set
    # Try to add/remove/swap elements
    # Since we are close to the limit, do a simple improvement pass.
    
    # Sort current best set
    if not best_set:
        return [2]
        
    # Try adding any number not in best_set
    for x in range(2, 257):
        if time.time() > deadline:
            break
        if x in best_set:
            continue
        if all(math.lcm(x, y) > 256 for y in best_set):
            best_set.append(x)
            
    # Try swapping: remove one, add one
    # This might be too slow if done naively.
    # Just return the current best.
    
    return best_set
