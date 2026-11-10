import time
import random
import numpy as np
from collections import defaultdict

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
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

    def is_valid_set(S):
        """Check if set S contains any 4-AP"""
        S_set = set(S) if not isinstance(S, set) else S
        for ap in all_aps:
            if all(x in S_set for x in ap):
                return False
        return True

    def can_add_element(x, S):
        """Check if x can be added to set S without creating a 4-AP"""
        for ap_idx in ap_indices_by_num[x]:
            ap = all_aps[ap_idx]
            count_in_set = 0
            for elem in ap:
                if elem == x:
                    continue
                if elem in S:
                    count_in_set += 1
            if count_in_set == 3:
                return False
        return True

    def greedy_with_order(order, S_init):
        """Greedy add elements in the given order to S_init"""
        S = set(S_init)
        for x in order:
            if x not in S and can_add_element(x, S):
                S.add(x)
        return S

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
    
    if not is_valid_set(base4_set):
        base4_set = set()
    
    # Initial best from various heuristics
    candidates_for_best = []
    candidates_for_best.append(base4_set)
    candidates_for_best.append(greedy_with_order(range(1, N + 1), set()))
    candidates_for_best.append(greedy_with_order(range(N, 0, -1), set()))
    candidates_for_best.append(greedy_with_order(range(1, N + 1), set(base4_set)))
    
    for seed in range(30):
        if time.time() > deadline - 10:
            break
        rng = random.Random(seed)
        order = list(range(1, N + 1))
        rng.shuffle(order)
        candidates_for_best.append(greedy_with_order(order, set()))
    
    best_set = max(candidates_for_best, key=len)
    best_size = len(best_set)
    
    # Build conflict graph: two numbers conflict if they appear together in some 4-AP
    # Actually, we need a different formulation. The constraint is that no 4-AP can be fully contained.
    # This is equivalent to: for each 4-AP (a,b,c,d), we cannot have all four in the set.
    # This is a 4-uniform hypergraph independent set problem.
    
    # For MIS formulation, we can think of it as: we want to select vertices (numbers) such that
    # no hyperedge (4-AP) is fully selected. This is a maximum independent set in a hypergraph.
    
    # Alternative: Use branch and bound on the hypergraph directly.
    
    # Let's use a different approach: iterative improvement with smarter local search
    # using bitset-based AP counting for speed.
    
    # Precompute for each number, which other numbers it conflicts with in terms of 4-APs
    # Two numbers x, y conflict if there exist z, w such that {x,y,z,w} forms a 4-AP
    # But we need all 4 to be present, so the conflict is 4-way.
    
    # Better approach: Use the AP count method from knowledge
    # For each number x, maintain count of how many 4-APs containing x are "complete" 
    # (i.e., all other 3 elements are in the set)
    
    # Let's implement a faster local search using precomputed AP structures
    
    # For each number, precompute the list of (other_three) tuples for each AP it's in
    ap_other_elements = {i: [] for i in range(1, N + 1)}
    for idx, ap in enumerate(all_aps):
        for pos, x in enumerate(ap):
            others = [ap[j] for j in range(4) if j != pos]
            ap_other_elements[x].append(others)
    
    def count_completed_aps_for(x, S):
        """Count how many 4-APs containing x have all other 3 elements in S"""
        count = 0
        S_set = S if isinstance(S, set) else set(S)
        for others in ap_other_elements[x]:
            if all(o in S_set for o in others):
                count += 1
        return count
    
    # Local Search with bitset-like optimization
    for restart in range(8000):
        if time.time() > deadline - 5:
            break
            
        current_set = set(best_set)
        elements = list(current_set)
        random.shuffle(elements)
        
        for i in range(len(elements)):
            if time.time() > deadline - 5:
                break
            
            remove_elem = elements[i]
            current_set.remove(remove_elem)
            
            # Try to add candidates in random order
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
                break
            else:
                current_set.add(remove_elem)
                for a in added:
                    if a in current_set:
                        current_set.remove(a)

    # Phase 2: More aggressive perturbations
    for restart in range(5000):
        if time.time() > deadline - 2:
            break
        
        current_set = set(best_set)
        elements = list(current_set)
        random.shuffle(elements)
        
        num_remove = random.randint(1, 4)
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

    # Phase 3: Branch and bound on subsets of the problem
    # Focus on improving specific regions
    
    # Identify "bottleneck" numbers that appear in many 4-APs
    ap_count_per_num = {i: len(ap_other_elements[i]) for i in range(1, N + 1)}
    
    # Try to optimize by considering high-degree vertices first
    sorted_by_degree = sorted(range(1, N + 1), key=lambda x: -ap_count_per_num[x])
    
    for restart in range(2000):
        if time.time() > deadline - 1:
            break
        
        # Start from best_set
        current_set = set(best_set)
        
        # Randomly decide to keep or remove some high-degree elements
        for x in sorted_by_degree[:50]:
            if random.random() < 0.3:
                if x in current_set:
                    current_set.remove(x)
        
        # Refill
        candidates = [j for j in range(1, N + 1) if j not in current_set]
        random.shuffle(candidates)
        
        for c in candidates:
            if can_add_element(c, current_set):
                current_set.add(c)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set

    # Final polish
    if time.time() - start_time < deadline:
        for _ in range(200):
            if time.time() > deadline:
                break
            current_set = set(best_set)
            elements = list(current_set)
            random.shuffle(elements)
            if not elements:
                break
            remove_elem = elements[0]
            current_set.remove(remove_elem)
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
                break
            else:
                current_set.add(remove_elem)
                for a in added:
                    if a in current_set:
                        current_set.remove(a)

    return sorted(best_set)
