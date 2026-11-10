# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from collections import defaultdict

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    deadline = time.time() + 600  # 600 second safety margin
    
    # Compute forbidden differences: nonzero squares mod 205
    forbidden = set()
    for x in range(1, m):
        sq = (x * x) % m
        if sq != 0:
            forbidden.add(sq)
    
    forbidden_list = list(forbidden)
    forbidden_set = set(forbidden_list)
    n_forbidden = len(forbidden_list)
    
    # Precompute conflict matrix: conflict[a][b] = True if (a-b)%m or (b-a)%m is in forbidden
    # conflict[a][b] means a and b cannot both be in the set
    conflict = np.zeros((m, m), dtype=bool)
    for a in range(m):
        for b in range(a + 1, m):
            d = (a - b) % m
            if d in forbidden_set:
                conflict[a, b] = True
                conflict[b, a] = True
    
    def is_valid_set(selected_set):
        """Check if a set is valid (no forbidden differences)."""
        arr = sorted(selected_set)
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                if conflict[arr[i], arr[j]]:
                    return False
        return True
    
    def count_conflicts(selected_set, candidate):
        """Count how many elements in selected_set conflict with candidate."""
        count = 0
        for s in selected_set:
            if conflict[candidate, s]:
                count += 1
        return count
    
    # Greedy construction with different orderings
    def greedy(order):
        selected = []
        selected_set = set()
        for candidate in order:
            if count_conflicts(selected_set, candidate) == 0:
                selected.append(candidate)
                selected_set.add(candidate)
        return selected
    
    # Generate multiple greedy solutions with different orderings
    best_set = None
    best_size = 0
    
    # Natural order
    selected = greedy(list(range(m)))
    if len(selected) > best_size:
        best_size = len(selected)
        best_set = set(selected)
    
    # Random orderings
    random.seed(42)
    for trial in range(50):
        order = list(range(m))
        random.shuffle(order)
        selected = greedy(order)
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = set(selected)
    
    # Local search with simulated annealing
    def local_search(initial_set, time_limit=5.0):
        """Local search with 1-flip moves and simulated annealing."""
        current_set = set(initial_set)
        best_set_local = set(current_set)
        best_size_local = len(current_set)
        
        # Precompute for efficiency: for each element, which other elements it conflicts with
        # conflict_neighbors[x] = set of elements that conflict with x
        conflict_neighbors = [set() for _ in range(m)]
        for a in range(m):
            for b in range(a + 1, m):
                if conflict[a, b]:
                    conflict_neighbors[a].add(b)
                    conflict_neighbors[b].add(a)
        
        start_time = time.time()
        temperature = 10.0
        cooling_rate = 0.999
        min_temperature = 0.1
        
        while time.time() - start_time < time_limit:
            # Generate a random 1-flip move
            move_type = random.random()
            
            if move_type < 0.4:
                # Add a random element not in current_set
                candidates = [x for x in range(m) if x not in current_set]
                if not candidates:
                    continue
                candidate = random.choice(candidates)
                new_set = current_set | {candidate}
                # Check validity: candidate must not conflict with any element in current_set
                valid = all(not conflict[candidate, s] for s in current_set)
                if not valid:
                    continue
                delta = 1  # size increases by 1
            elif move_type < 0.7:
                # Remove a random element from current_set
                if not current_set:
                    continue
                candidate = random.choice(list(current_set))
                new_set = current_set - {candidate}
                delta = -1  # size decreases by 1
            else:
                # Swap: remove one, add another
                if not current_set:
                    continue
                remove_candidate = random.choice(list(current_set))
                candidates = [x for x in range(m) if x not in current_set]
                if not candidates:
                    continue
                add_candidate = random.choice(candidates)
                new_set = (current_set - {remove_candidate}) | {add_candidate}
                # Check validity: add_candidate must not conflict with any element in new_set except remove_candidate
                valid = all(not conflict[add_candidate, s] for s in new_set if s != add_candidate)
                if not valid:
                    continue
                delta = 0  # size stays the same
            
            # Accept based on simulated annealing
            if delta >= 0 or random.random() < np.exp(delta / temperature):
                current_set = new_set
                if len(current_set) > best_size_local:
                    best_size_local = len(current_set)
                    best_set_local = set(current_set)
            
            temperature *= cooling_rate
            if temperature < min_temperature:
                temperature = min_temperature
        
        return best_set_local
    
    # Run local search from best greedy solution
    if best_set is not None:
        improved = local_search(best_set, time_limit=30.0)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved
    
    # Try local search from multiple random valid starting points
    for trial in range(20):
        if time.time() > deadline - 60:
            break
        
        # Generate a random valid set
        random_set = set()
        order = list(range(m))
        random.shuffle(order)
        for candidate in order:
            if count_conflicts(random_set, candidate) == 0:
                random_set.add(candidate)
        
        if len(random_set) >= best_size:
            improved = local_search(random_set, time_limit=10.0)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved
    
    # Final polish: try to add any element that doesn't conflict
    if best_set is not None:
        for candidate in range(m):
            if candidate not in best_set:
                if all(not conflict[candidate, s] for s in best_set):
                    best_set.add(candidate)
    
    # Convert to sorted numpy array
    result = np.array(sorted(best_set), dtype=np.int64)
    
    # Final validation
    assert len(result) == len(set(result)), "Duplicates found"
    assert all(0 <= x <= 204 for x in result), "Values out of range"
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            d = (result[i] - result[j]) % m
            assert d not in forbidden_set, f"Forbidden difference {d} between {result[i]} and {result[j]}"
    
    return result

# EVOLVE-BLOCK-END
