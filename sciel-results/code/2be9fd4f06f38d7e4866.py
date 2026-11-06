"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _is_4ap_free(x, selected, n=512):
    """Check if adding x to selected would create a nonconstant 4-term AP."""
    # x can be the 1st term: x, x+d, x+2d, x+3d
    for d in range(1, (n - x) // 3 + 1):
        if (x + d) in selected and (x + 2*d) in selected and (x + 3*d) in selected:
            return False
    # x can be the 2nd term: x-d, x, x+d, x+2d
    for d in range(1, min(x - 1, (n - x) // 2) + 1):
        if (x - d) in selected and (x + d) in selected and (x + 2*d) in selected:
            return False
    # x can be the 3rd term: x-2d, x-d, x, x+d
    for d in range(1, min((x - 1) // 2, n - x) + 1):
        if (x - 2*d) in selected and (x - d) in selected and (x + d) in selected:
            return False
    # x can be the 4th term: x-3d, x-2d, x-d, x
    for d in range(1, (x - 1) // 3 + 1):
        if (x - 3*d) in selected and (x - 2*d) in selected and (x - d) in selected:
            return False
    return True

def _greedy_seed():
    """Build a greedy 4-AP-free set as starting point."""
    selected = set()
    for x in range(1, 513):
        if _is_4ap_free(x, selected):
            selected.add(x)
    return selected

def construct():
    deadline = time.time() + 85  # 85s safety margin before 90s kill
    
    # Start with greedy seed
    best = _greedy_seed()
    best_size = len(best)
    
    # Local search: try to add elements from the complement
    rng = random.Random(42)
    excluded = list(set(range(1, 513)) - best)
    
    while time.time() < deadline:
        # Shuffle excluded list periodically
        rng.shuffle(excluded)
        
        made_progress = False
        for x in excluded:
            if time.time() >= deadline:
                break
            if _is_4ap_free(x, best):
                best.add(x)
                best_size += 1
                made_progress = True
                excluded.remove(x)
        
        if not made_progress:
            # No single addition works; try a different random order
            rng.shuffle(excluded)
            # Also try removing one element and then adding two (swap-like move)
            if best_size > 10:
                # Try a simple swap: remove one, try to add two
                removed_candidates = list(best)
                rng.shuffle(removed_candidates)
                for rem in removed_candidates:
                    if time.time() >= deadline:
                        break
                    trial = set(best)
                    trial.discard(rem)
                    added = 0
                    for x in excluded:
                        if time.time() >= deadline:
                            break
                        if _is_4ap_free(x, trial):
                            trial.add(x)
                            added += 1
                    if added >= 2:
                        best = trial
                        best_size = len(best)
                        excluded = list(set(range(1, 513)) - best)
                        break
    
    return sorted(best)
# EVOLVE-BLOCK-END
