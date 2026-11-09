"""
Combined approach:
1. Fast verification using LCM-based integer arithmetic and numpy for subset-sum uniqueness.
2. Structural search: 
   - Phase 1: Greedy construction with multiple orderings (ascending, descending, random) to quickly find a large valid set.
   - Phase 2: Local search (add/swap) to extend the best set found.
   - Phase 3: If time permits, a more exhaustive DFS for small remaining gaps or random restarts with hill climbing.
"""
import time
import random
import numpy as np

# LCM of 1..22
_LCM = 232792560

def _verify_fast(subset):
    """
    Returns True if all reciprocal subset sums are distinct.
    Uses numpy for efficient subset sum generation and uniqueness check.
    """
    k = len(subset)
    if k == 0:
        return True
    if k > 20:
        # For very large k, this might be slow, but we expect max ~12-15
        # 2^20 is ~1M, which is fine for numpy
        pass
        
    # Convert to integer weights
    weights = np.array([_LCM // i for i in subset], dtype=np.int64)
    
    # Generate all subset sums
    # Start with [0]
    sums = np.array([0], dtype=np.int64)
    
    for w in weights:
        # New sums are existing sums + w
        # We can use broadcasting or simple addition
        sums = np.concatenate([sums, sums + w])
        
    # Check uniqueness
    # Using np.unique is efficient for large arrays
    unique_sums = np.unique(sums)
    return len(unique_sums) == len(sums)

def _verify_python(subset):
    """
    Fallback verification using pure Python sets, potentially faster for small k due to overhead.
    """
    k = len(subset)
    if k == 0:
        return True
        
    values = [_LCM // i for i in subset]
    
    sums = {0}
    for v in values:
        # Create new set of sums
        new_sums = {s + v for s in sums}
        # If there's any overlap, it's not necessarily a failure yet, 
        # but we need the final count to be 2^k.
        # Actually, if new_sums intersects sums, it means sum(A) = sum(B) + v for some A, B.
        # This implies sum(A) - sum(B) = v.
        # This is a dependency.
        # However, the most direct check is just building the full set and checking size.
        sums = sums.union(new_sums)
        
    return len(sums) == (1 << k)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    candidates = list(range(1, 23))
    best_set = []
    best_size = 0
    
    # Phase 1: Greedy Construction with multiple orderings
    # Try different orders to escape local optima
    orders = [
        sorted(candidates),
        sorted(candidates, reverse=True),
        # Order by weight (LCM//i) ascending (small i first)
        sorted(candidates, key=lambda x: _LCM // x),
        # Order by weight descending (large i first)
        sorted(candidates, key=lambda x: _LCM // x, reverse=True),
    ]
    
    # Add some random orders
    random.seed(42)
    for _ in range(5):
        r = candidates[:]
        random.shuffle(r)
        orders.append(r)
    
    for order in orders:
        if time.time() > deadline:
            break
            
        current_set = []
        for i in order:
            if i in current_set:
                continue
            new_set = current_set + [i]
            if _verify_python(new_set):
                current_set = new_set
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
        # If we found a very large set, we might want to stop early
        if best_size >= 14:
            break

    # Phase 2: Local Search (Add and Swap)
    # Try to add any missing element
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding
        for i in candidates:
            if i not in best_set:
                new_set = best_set + [i]
                if _verify_python(new_set):
                    best_set = new_set
                    best_size = len(best_set)
                    improved = True
                    break
        
        # Try swapping (replace one element with two? or one for one?)
        # Simple swap: remove one, try adding another
        # Since we just tried adding, maybe we need to remove one to add two?
        # Let's try: for each element in best_set, remove it, then try to add any missing element.
        # This is more expensive.
        if not improved and best_size < 22:
            # Try a single swap
            for i in list(best_set):
                if time.time() > deadline:
                    break
                temp_set = [x for x in best_set if x != i]
                # Try to add any missing element to temp_set
                for j in candidates:
                    if j not in temp_set:
                        new_set = temp_set + [j]
                        if _verify_python(new_set):
                            best_set = new_set
                            best_size = len(best_set)
                            improved = True
                            break
                if improved:
                    break

    # Phase 3: Randomized Hill Climbing / Restart
    # If we are stuck, try random perturbations
    if time.time() < deadline - 5:
        for _ in range(50):
            if time.time() > deadline:
                break
            
            # Start from best_set, remove 1-2 elements, then greedily add back
            perturbed = best_set[:]
            if len(perturbed) > 2:
                # Remove 1 or 2 random elements
                n_remove = random.randint(1, 2)
                for _ in range(n_remove):
                    if perturbed:
                        idx = random.randint(0, len(perturbed)-1)
                        perturbed.pop(idx)
            
            # Greedily add back
            current_set = perturbed[:]
            # Sort candidates by some heuristic? Or just random order?
            # Let's try to add missing elements in random order
            missing = [x for x in candidates if x not in current_set]
            random.shuffle(missing)
            
            for i in missing:
                new_set = current_set + [i]
                if _verify_python(new_set):
                    current_set = new_set
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]

    return best_set
