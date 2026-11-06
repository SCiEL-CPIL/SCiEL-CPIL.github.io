"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def _greedy_subset(elements, target=200, deadline=None):
    """Greedily select elements in given order, including if target not newly reachable."""
    selected = []
    reachable = 1  # bit 0 set, sum 0 reachable
    mask = (1 << (target + 1)) - 1
    for x in elements:
        new_reachable = reachable | ((reachable << x) & mask)
        if (new_reachable >> target) & 1 == 0:
            selected.append(x)
            reachable = new_reachable
        if deadline is not None and time.time() > deadline:
            break
    return selected

def _greedy_with_removal(all_elements, target=200, deadline=None):
    """Start with all elements, remove ones that cause target to be reachable."""
    elements = list(all_elements)
    reachable = 1
    mask = (1 << (target + 1)) - 1
    # First, compute which elements are "safe" to include together
    # Use a different strategy: sort by some heuristic and greedily include
    # but also try: start with all, and remove elements that are critical
    # Actually, let's try: sort descending, include greedily
    # But also try: sort by (x % target) to group complementary pairs
    pass

def construct():
    deadline = time.time() + 50  # safety margin before 60s limit
    all_elements = list(range(1, 101))
    target = 200
    
    best = []
    
    # Try descending order
    cand = _greedy_subset(sorted(all_elements, reverse=True), deadline=deadline)
    if len(cand) > len(best):
        best = cand
    
    # Try ascending order
    cand = _greedy_subset(sorted(all_elements), deadline=deadline)
    if len(cand) > len(best):
        best = cand
    
    # Try sorting by x % target (groups complementary sums)
    cand = _greedy_subset(sorted(all_elements, key=lambda x: x % target), deadline=deadline)
    if len(cand) > len(best):
        best = cand
    
    # Try sorting by target - x (prefer elements far from complement)
    cand = _greedy_subset(sorted(all_elements, key=lambda x: target - x), deadline=deadline)
    if len(cand) > len(best):
        best = cand
    
    # Try interleaving small and large
    small = list(range(1, 51))
    large = list(range(51, 101))
    interleaved = []
    for i in range(50):
        interleaved.append(small[i])
        interleaved.append(large[i])
    cand = _greedy_subset(interleaved, deadline=deadline)
    if len(cand) > len(best):
        best = cand
    
    # Try many random orderings with different seeds
    rng = random.Random(42)
    for _ in range(200):
        if time.time() > deadline:
            break
        shuffled = all_elements[:]
        rng.shuffle(shuffled)
        cand = _greedy_subset(shuffled, deadline=deadline)
        if len(cand) > len(best):
            best = cand
    
    # Try local improvement: from best, try swapping elements in/out
    if best:
        best_set = set(best)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding an element not in best
            for x in all_elements:
                if x in best_set:
                    continue
                test_list = list(best) + [x]
                # Check if any subset sums to target
                reachable = 1
                mask = (1 << (target + 1)) - 1
                valid = True
                for y in test_list:
                    new_reachable = reachable | ((reachable << y) & mask)
                    if (new_reachable >> target) & 1:
                        valid = False
                        break
                    reachable = new_reachable
                if valid:
                    best = test_list
                    best_set = set(best)
                    improved = True
                    break
            # Try removing an element and adding two others (net gain)
            # This is expensive, skip for now
    
    # Ensure nonempty
    if not best:
        best = [100]
    
    return best
# EVOLVE-BLOCK-END
