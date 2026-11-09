import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s limit
    
    n = 16
    k = 4
    max_size = 250
    
    # Generate all possible 4-element subsets of [1..16]
    all_sets = [frozenset(c) for c in combinations(range(1, n + 1), k)]
    total = len(all_sets)  # 1820
    
    # Precompute all pairwise intersections
    # For speed, we'll use a different approach:
    # A sunflower triple (A, B, C) exists iff A∩B = A∩C = B∩C
    # This means the three sets all share the same pairwise intersection
    
    # Key insight: For 4-element sets, the intersection can be of size 0, 1, 2, or 3
    # If intersection size is 3, then two sets differ in exactly one element
    # If intersection size is 2, they share exactly 2 elements
    # If intersection size is 1, they share exactly 1 element  
    # If intersection size is 0, they are disjoint
    
    # Strategy: Use a conflict-based approach
    # Two sets "conflict" if there exists a third set that would form a sunflower
    # But this is transitive in a complex way, so let's use direct checking
    
    # Precompute for each pair of sets, what intersection they have
    # Then for a new set S, it conflicts with existing pair (A,B) if:
    # A∩B = A∩S = B∩S
    
    # For efficiency, group sets by their "type" - specifically by which elements they contain
    # Actually, let's just do a fast greedy with smart ordering
    
    def can_add(new_set, family, family_indices):
        """Check if new_set can be added without creating a sunflower."""
        # For each pair in family, check if adding new_set creates a sunflower
        for i in range(len(family)):
            a = family[i]
            inter_an = a & new_set
            for j in range(i + 1, len(family)):
                b = family[j]
                # Check if a∩b = inter_an = b∩new_set
                if (a & b) == inter_an and (b & new_set) == inter_an:
                    return False
        return True
    
    def greedy_with_order(order, family=None, family_indices=None):
        """Greedy build following a specific order."""
        if family is None:
            family = []
            family_indices = []
        
        for idx in order:
            if len(family) >= max_size:
                break
            s = all_sets[idx]
            if can_add(s, family, family_indices):
                family.append(s)
                family_indices.append(idx)
        
        return family, family_indices
    
    def local_search(family, family_indices, iterations=5000):
        """Improve via 1-for-1 and 2-for-1 swaps."""
        current = family[:]
        current_indices = family_indices[:]
        current_set = set(current)
        
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            
            # Try removing one set and adding two
            for _ in range(100):
                if time.time() > deadline - 10.0:
                    break
                
                if not current:
                    break
                
                # Pick a random set to potentially remove
                rem_pos = random.randint(0, len(current) - 1)
                removed = current[rem_pos]
                
                # Build family without this set
                temp_family = [x for x in current if x is not removed]
                temp_indices = [x for i, x in enumerate(current_indices) if i != rem_pos]
                
                # Try to add two sets
                added_count = 0
                for _add in range(2):
                    # Try random candidates
                    for _ in range(300):
                        idx = random.randint(0, total - 1)
                        s = all_sets[idx]
                        if s in set(temp_family):
                            continue
                        if can_add(s, temp_family, temp_indices):
                            temp_family.append(s)
                            temp_indices.append(idx)
                            added_count += 1
                            break
                    if len(temp_family) >= max_size:
                        break
                
                if len(temp_family) > len(current):
                    current = temp_family
                    current_indices = temp_indices
                    improved = True
            
            # Try 1-for-1 swaps
            if not improved:
                for _ in range(200):
                    if time.time() > deadline - 10.0:
                        break
                    
                    if not current:
                        break
                    
                    rem_pos = random.randint(0, len(current) - 1)
                    removed = current[rem_pos]
                    
                    temp_family = [x for x in current if x is not removed]
                    temp_indices = [x for i, x in enumerate(current_indices) if i != rem_pos]
                    
                    # Try to add one set
                    for _ in range(500):
                        idx = random.randint(0, total - 1)
                        s = all_sets[idx]
                        if s in set(temp_family):
                            continue
                        if can_add(s, temp_family, temp_indices):
                            temp_family.append(s)
                            temp_indices.append(idx)
                            break
                    
                    if len(temp_family) > len(current):
                        current = temp_family
                        current_indices = temp_indices
                        improved = True
        
        return current, current_indices
    
    # Try multiple random orderings
    best_family = []
    best_indices = []
    best_size = 0
    
    num_attempts = 0
    while time.time() < deadline - 30.0:
        order = list(range(total))
        random.shuffle(order)
        
        # Occasionally use a biased ordering (prefer sets with certain properties)
        if random.random() < 0.3:
            # Sort by size of intersection with a random reference set
            ref = random.randint(0, total - 1)
            ref_set = all_sets[ref]
            order.sort(key=lambda idx: len(all_sets[idx] & ref_set))
        
        family, indices = greedy_with_order(order)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
            best_indices = indices[:]
        
        num_attempts += 1
    
    # Local search on the best solution
    if best_family and time.time() < deadline - 10.0:
        best_family, best_indices = local_search(best_family, best_indices)
    
    # Final verification: ensure no sunflowers
    verified = []
    for s in best_family:
        if can_add(s, verified, []):
            verified.append(s)
    
    return [sorted(s) for s in verified]
