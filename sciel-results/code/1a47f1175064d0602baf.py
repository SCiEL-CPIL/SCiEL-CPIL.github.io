import time
import random
import numpy as np
from collections import defaultdict

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit

    # Precompute all 4-term arithmetic progressions
    all_aps = []
    max_d = (N - 1) // 3
    for d in range(1, max_d + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            all_aps.append((a, a + d, a + 2 * d, a + 3 * d))

    num_aps = len(all_aps)

    # Map each number to the indices of APs it belongs to
    ap_indices_by_num = {i: [] for i in range(1, N + 1)}
    for idx, ap in enumerate(all_aps):
        for x in ap:
            ap_indices_by_num[x].append(idx)

    # For each number, precompute the list of (other_three) tuples for each AP it's in
    ap_other_elements = {i: [] for i in range(1, N + 1)}
    for idx, ap in enumerate(all_aps):
        for pos, x in enumerate(ap):
            others = [ap[j] for j in range(4) if j != pos]
            ap_other_elements[x].append(others)

    # Precompute conflict graph: for each pair of elements, whether they conflict
    # Two elements conflict if there exist two other elements that form a 4-AP with both
    # Actually, we need: can_add(x, S) checks if any AP containing x has all other 3 in S

    def can_add_element(x, S):
        """Check if x can be added to set S without creating a 4-AP"""
        for others in ap_other_elements[x]:
            if all(o in S for o in others):
                return False
        return True

    def is_valid_set(S):
        """Check if set S contains any 4-AP"""
        for ap in all_aps:
            if all(x in S for x in ap):
                return False
        return True

    # Base-4 {0,1} construction
    base4_set = set()
    for i in range(1, N + 1):
        n = i
        valid = True
        while n > 0:
            if n % 4 > 1:
                valid = False
                break
            n //= 4
        if valid:
            base4_set.add(i)

    # Greedy construction with different orderings
    def greedy_with_order(order, S_init):
        S = set(S_init)
        for x in order:
            if x not in S and can_add_element(x, S):
                S.add(x)
        return S

    # Initial best from various heuristics
    candidates_for_best = []
    candidates_for_best.append(base4_set)
    candidates_for_best.append(greedy_with_order(range(1, N + 1), set()))
    candidates_for_best.append(greedy_with_order(range(N, 0, -1), set()))
    candidates_for_best.append(greedy_with_order(range(1, N + 1), set(base4_set)))
    candidates_for_best.append(greedy_with_order(range(N, 0, -1), set(base4_set)))
    
    # Try greedy with different random seeds
    for seed in range(100):
        if time.time() > deadline - 30:
            break
        rng = random.Random(seed)
        order = list(range(1, N + 1))
        rng.shuffle(order)
        candidates_for_best.append(greedy_with_order(order, set()))
    
    # Also try greedy starting from base4 with random order
    for seed in range(50):
        if time.time() > deadline - 25:
            break
        rng = random.Random(seed + 1000)
        order = list(range(1, N + 1))
        rng.shuffle(order)
        candidates_for_best.append(greedy_with_order(order, set(base4_set)))

    best_set = max(candidates_for_best, key=len)
    best_size = len(best_set)

    # Use bitset for fast membership testing in local search
    N_bits = N + 1
    S_bitset = np.zeros(N_bits, dtype=np.uint8)
    for x in best_set:
        S_bitset[x] = 1

    def can_add_fast(x, S_bits):
        """Fast check using bitset"""
        for others in ap_other_elements[x]:
            if S_bits[others[0]] and S_bits[others[1]] and S_bits[others[2]]:
                return False
        return True

    # Phase 1: Iterative improvement with single element removal and refill
    # Use bitset for speed
    for iteration in range(80000):
        if time.time() > deadline - 10:
            break
        
        current_set = set(best_set)
        elements = list(current_set)
        random.shuffle(elements)
        
        if not elements:
            break
            
        remove_elem = elements[0]
        current_set.remove(remove_elem)
        
        # Try to add candidates in random order, prioritizing those with fewer conflicts
        candidates = [j for j in range(1, N + 1) if j not in current_set]
        random.shuffle(candidates)
        
        added = []
        for c in candidates:
            if can_add_element(c, current_set):
                current_set.add(c)
                added.append(c)
        
        if len(added) > 1:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set
                # Update bitset
                S_bitset[:] = 0
                for x in best_set:
                    S_bitset[x] = 1
        else:
            # Revert
            current_set.add(remove_elem)
            for a in added:
                if a in current_set:
                    current_set.remove(a)

    # Phase 2: More aggressive perturbations - remove multiple elements
    for restart in range(50000):
        if time.time() > deadline - 8:
            break
        
        current_set = set(best_set)
        elements = list(current_set)
        random.shuffle(elements)
        
        num_remove = random.randint(1, 6)
        for _ in range(num_remove):
            if elements:
                rem = elements.pop()
                current_set.remove(rem)
        
        candidates = [j for j in range(1, N + 1) if j not in current_set]
        random.shuffle(candidates)
        
        for c in candidates:
            if can_add_element(c, current_set):
                current_set.add(c)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            S_bitset[:] = 0
            for x in best_set:
                S_bitset[x] = 1

    # Phase 3: Simulated annealing style - accept worse solutions with probability
    current_set = set(best_set)
    current_size = len(current_set)
    
    for step in range(200000):
        if time.time() > deadline - 5:
            break
        
        temp = 5.0 * (1.0 - step / 200000.0) + 0.5
        
        # Random move: remove one element, add one
        elements = list(current_set)
        if not elements:
            break
            
        remove_elem = random.choice(elements)
        candidates = [j for j in range(1, N + 1) if j not in current_set]
        if not candidates:
            break
            
        add_elem = random.choice(candidates)
        
        # Check if we can add add_elem after removing remove_elem
        test_set = current_set - {remove_elem}
        if can_add_element(add_elem, test_set):
            new_set = test_set | {add_elem}
            new_size = len(new_set)
            
            delta = new_size - current_size
            if delta > 0 or random.random() < np.exp(delta / max(temp, 0.1)):
                current_set = new_set
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_set = set(current_set)
                    S_bitset[:] = 0
                    for x in best_set:
                        S_bitset[x] = 1

    # Phase 4: Final polish - try to add elements in a smart order
    # Sort candidates by number of conflicts (fewer conflicts first)
    def count_conflicts(x, S):
        count = 0
        for others in ap_other_elements[x]:
            if all(o in S for o in others):
                count += 1
        return count

    current_set = set(best_set)
    candidates = [j for j in range(1, N + 1) if j not in current_set]
    
    # Sort by number of conflicts (ascending) - try low-conflict elements first
    candidates.sort(key=lambda x: count_conflicts(x, current_set))
    
    for c in candidates:
        if can_add_element(c, current_set):
            current_set.add(c)
    
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = current_set

    # Final verification
    if not is_valid_set(best_set):
        # Fallback to a known valid set
        best_set = base4_set

    return sorted(best_set)
