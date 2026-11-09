import time
import random
from itertools import combinations

def _check_validity(selected):
    """Check if the entire set is valid."""
    n = len(selected)
    # For each a in selected
    for i in range(n):
        a = selected[i]
        others = [selected[j] for j in range(n) if j != i]
        # Check all combinations of 3 distinct others
        for u in range(len(others)):
            for v in range(u+1, len(others)):
                for w in range(v+1, len(others)):
                    s = others[u] + others[v] + others[w]
                    if s % a == 0:
                        return False
    return True

def _count_violations_for_element(a, selected):
    """Count violations involving a as the divisor."""
    count = 0
    others = [x for x in selected if x != a]
    for triplet in combinations(others, 3):
        if sum(triplet) % a == 0:
            count += 1
    return count

def _find_violation_triplets(selected):
    """Find all (a, triplet) violations."""
    violations = []
    n = len(selected)
    for i in range(n):
        a = selected[i]
        others = [selected[j] for j in range(n) if j != i]
        for triplet in combinations(others, 3):
            if sum(triplet) % a == 0:
                violations.append((a, triplet))
    return violations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Primary pool: 144..192 (49 elements)
    # We want to keep as many as possible, up to 48.
    base_pool = list(range(144, 193))
    
    best_set = []
    
    # Try multiple random removal orders or initial perturbations
    # Strategy: Start with full pool, remove elements causing most conflicts.
    
    for attempt in range(20):
        if time.time() > deadline:
            break
            
        current_set = base_pool[:]
        
        # If we already have a valid set of size 48, we can stop early in subsequent attempts?
        # No, we want to find if 49 is valid, or 48 is easily reachable.
        
        # Removal Loop
        while True:
            if time.time() > deadline:
                break
            
            # Check if valid
            violations = _find_violation_triplets(current_set)
            
            if not violations:
                # Valid set found
                if len(current_set) > len(best_set):
                    best_set = current_set[:]
                
                # If we have more than 48, trim to 48
                if len(best_set) > 48:
                    # Randomly remove extras to satisfy "at most 48"
                    # But we want to keep the "best" 48? Any 48 is fine if valid.
                    # Since removing any element from a valid set keeps it valid (relaxation),
                    # we can just truncate.
                    if len(best_set) > 48:
                        # Shuffle and take first 48 to be deterministic-ish or random
                        random.shuffle(best_set)
                        best_set = best_set[:48]
                        # Re-check validity? Removing elements cannot create new violations.
                        # So it is still valid.
                        pass
                
                if len(best_set) == 48:
                    # We hit the max. Can we do better? No, max is 48.
                    # But maybe we can find a set of 48 that allows adding more from lower range?
                    # The task says "at most 48 elements". So 48 is the global max.
                    # We can break out of attempts if we found 48.
                    return best_set
                
                break
            
            # Find the element involved in the most violations
            # Count how many times each 'a' appears as the divisor in violations
            # Also consider if removing an element from the triplet helps?
            # Removing 'a' fixes all violations where 'a' is the divisor.
            # Removing an element from the triplet fixes only that specific violation (and others involving that element in triplets).
            # Generally, removing the divisor 'a' is more effective as it clears all its conflicts.
            
            violation_counts = {}
            for a, triplet in violations:
                violation_counts[a] = violation_counts.get(a, 0) + 1
                # Also credit the triplet members?
                # If we remove b from triplet, we fix this violation.
                for x in triplet:
                    violation_counts[x] = violation_counts.get(x, 0) + 1
            
            # Pick the element with max count
            if not violation_counts:
                break # Should not happen if violations exist
                
            # Random tie-breaking
            max_count = max(violation_counts.values())
            candidates_to_remove = [k for k, v in violation_counts.items() if v == max_count]
            
            # Sometimes randomizing the choice helps escape local minima
            if random.random() < 0.3:
                # Pick random from all involved elements
                all_involved = set()
                for a, triplet in violations:
                    all_involved.add(a)
                    all_involved.update(triplet)
                elem_to_remove = random.choice(list(all_involved))
            else:
                elem_to_remove = random.choice(candidates_to_remove)
            
            if elem_to_remove in current_set:
                current_set.remove(elem_to_remove)
            else:
                break # Safety
        
        # End of while loop for this attempt
        # If we didn't break due to 48 found, we just saved best_set if it was better
        # But we only update best_set inside the valid check.
        
        # If we exhausted time or got stuck, we continue to next attempt
        # We can also try expanding the pool if we didn't reach 48
        if len(best_set) < 48:
             # Try adding lower numbers?
             # For now, stick to the pool 144-192. If we can't get 48 from 49,
             # maybe we need to swap some high numbers for lower ones?
             # But lower numbers have more divisors.
             # Let's try a local search: swap an element in best_set with one from 100-143?
             # This is complex. Let's rely on the removal heuristic on the dense pool.
             pass

    return best_set
