import time
import random
import itertools

def _count_comparable(sets):
    """Count unordered pairs where one is a strict subset of the other."""
    n = len(sets)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if len(sets[i]) < len(sets[j]):
                if all(x in sets[j] for x in sets[i]):
                    count += 1
            elif len(sets[j]) < len(sets[i]):
                if all(x in sets[i] for x in sets[j]):
                    count += 1
    return count

def _construct_initial():
    """Construct a strong initial solution using multiple chains."""
    # Start with the main chain of size 13
    sets = [list(range(i)) for i in range(13)]
    
    # Add sets that extend the chain in different directions
    # For each position in the chain, add sets that are supersets
    for i in range(13):
        base = list(range(i))
        # Add one more element to create a superset
        for j in range(i, 12):
            if len(sets) >= 40:
                break
            candidate = base + [j]
            if candidate not in sets:
                sets.append(candidate)
        if len(sets) >= 40:
            break
    
    # Fill remaining with random sets that are likely comparable
    while len(sets) < 40:
        size = random.randint(0, 12)
        s = random.sample(range(12), size)
        s = sorted(s)
        if s not in sets:
            sets.append(s)
    
    return sets[:40]

def _local_search(sets, max_iterations=5000):
    """Improve the solution using local search."""
    sets = [list(s) for s in sets]
    current_score = _count_comparable(sets)
    
    for _ in range(max_iterations):
        # Pick a random set to modify
        idx = random.randint(0, 39)
        old_set = list(sets[idx])
        
        # Try modifying the set
        new_set = list(old_set)
        if new_set:
            # Remove a random element
            if random.random() < 0.5:
                new_set.pop(random.randint(0, len(new_set) - 1))
            else:
                # Add a random element
                available = [i for i in range(12) if i not in new_set]
                if available:
                    new_set.append(random.choice(available))
        else:
            # Add a random element
            new_set.append(random.randint(0, 11))
        
        new_set = sorted(new_set)
        
        # Check if it's already in the set
        if new_set in sets and new_set != old_set:
            continue
        
        # Swap and evaluate
        sets[idx] = new_set
        new_score = _count_comparable(sets)
        
        if new_score >= current_score:
            current_score = new_score
        else:
            # Revert
            sets[idx] = old_set
    
    return sets, current_score

def construct():
    """Main entry point."""
    deadline = time.time() + 85  # Safety margin before 90s limit
    
    # Start with a good initial solution
    best_sets = _construct_initial()
    best_score = _count_comparable(best_sets)
    
    # Run local search with multiple restarts
    while time.time() < deadline:
        # Run local search from current best
        improved_sets, improved_score = _local_search(best_sets, max_iterations=2000)
        
        if improved_score > best_score:
            best_sets = improved_sets
            best_score = improved_score
        
        # Check time
        if time.time() > deadline - 5:
            break
        
        # Occasional restart with different initial solution
        if random.random() < 0.3:
            candidate = _construct_initial()
            candidate_score = _count_comparable(candidate)
            if candidate_score > best_score:
                best_sets = candidate
                best_score = candidate_score
    
    # Ensure we return exactly 40 distinct sets
    return [list(s) for s in best_sets[:40]]
