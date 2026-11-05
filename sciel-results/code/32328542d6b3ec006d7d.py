"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _precompute_triples():
    """Precompute all sets of 3 distinct elements {a,b,c} in [1,180] 
    such that 1/a = 1/b + 1/c for some permutation (i.e., one element is the 'sum' side).
    
    The condition 1/a = 1/b + 1/c means a = bc/(b+c).
    We need a, b, c all distinct and in [1,180].
    """
    # For each pair (b, c) with b < c, check if a = bc/(b+c) is an integer in [1,180] and distinct from b,c
    # Also consider that a could be larger or smaller than b,c.
    # Actually, 1/a = 1/b + 1/c implies a = bc/(b+c). Since b,c > 0, a < min(b,c) is NOT necessarily true.
    # Wait: 1/a = 1/b + 1/c > 1/c (assuming b>0), so a < c. Similarly a < b.
    # So a is always the smallest of the three? Let's check:
    # If a = bc/(b+c), then 1/a = 1/b + 1/c. Since 1/b > 0 and 1/c > 0, 1/a > 1/b => a < b.
    # And 1/a > 1/c => a < c.
    # So yes, a is always the smallest element.
    
    triples = []
    for b in range(1, 181):
        for c in range(b + 1, 181):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a <= 180 and a != b and a != c:
                    # a is the smallest, so a < b < c
                    # Store as a frozenset or sorted tuple for easy lookup
                    triples.append((a, b, c))
    return triples

def _is_valid_subset(subset_set, triples):
    """Check if subset_set contains no triple."""
    for (a, b, c) in triples:
        if a in subset_set and b in subset_set and c in subset_set:
            return False
    return True

def _count_violations(subset_set, triples):
    """Count how many triples are fully contained in subset_set."""
    count = 0
    for (a, b, c) in triples:
        if a in subset_set and b in subset_set and c in subset_set:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin out of 60s
    
    # Precompute all forbidden triples
    triples = _precompute_triples()
    
    # Strategy: Greedy + Local Search
    # 1. Start with a greedy construction that tries to include as many elements as possible
    # 2. Use local search (swap moves) to improve
    
    # Greedy initialization: 
    # Try to build a large set by adding elements in some order while maintaining validity
    # A good heuristic: start with large numbers (they're less likely to form triples as the 'a' part)
    # Actually, since a is always the smallest, large numbers can only be b or c in a triple.
    # Small numbers are more restrictive.
    
    # Let's try multiple random orderings and greedy constructions
    
    best_set = set()
    best_size = 0
    
    # First, try a deterministic greedy: add numbers from 180 down to 1
    # if they don't create a violation
    current_set = set()
    for x in range(180, 0, -1):
        test_set = current_set | {x}
        if _is_valid_subset(test_set, triples):
            current_set = test_set
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = set(current_set)
    
    # Also try from 1 to 180
    current_set = set()
    for x in range(1, 181):
        test_set = current_set | {x}
        if _is_valid_subset(test_set, triples):
            current_set = test_set
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = set(current_set)
    
    # Local search with swap moves
    # Try to improve by swapping elements: remove one element from the set and add another not in the set
    # Focus on removing elements that are in many triples and adding elements that are in few triples
    
    # Precompute for each element how many triples it participates in
    triple_count = {}
    for x in range(1, 181):
        triple_count[x] = 0
    for (a, b, c) in triples:
        triple_count[a] += 1
        triple_count[b] += 1
        triple_count[c] += 1
    
    # Local search: try swap moves
    # For each element in the set, try removing it and adding a different element
    # Accept if the new set is valid and has the same or larger size (we want to maximize size)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_set = set(best_set)
        
        # Try all possible swaps: remove x from set, add y not in set
        # To be efficient, only consider swaps that might help
        # We want to keep the size the same or increase it
        # Since we start with a maximal-ish set, swaps that keep size are most useful
        
        elements_in = sorted(current_set)
        elements_out = [x for x in range(1, 181) if x not in current_set]
        
        # Sort elements_out by triple_count (ascending) - prefer adding elements in few triples
        elements_out.sort(key=lambda x: triple_count[x])
        
        # Try removing each element in the set (sorted by triple_count descending - prefer removing those in many triples)
        elements_in.sort(key=lambda x: -triple_count[x])
        
        for x_remove in elements_in:
            if time.time() >= deadline:
                break
            # Try adding each candidate y
            for y_add in elements_out:
                if y_add == x_remove:
                    continue
                # Try the swap
                new_set = current_set - {x_remove} | {y_add}
                if _is_valid_subset(new_set, triples):
                    # Accept this swap
                    best_set = new_set
                    best_size = len(new_set)
                    improved = True
                    # Update current_set for subsequent swaps in this iteration
                    current_set = new_set
                    # Update elements lists
                    elements_in = sorted(current_set)
                    elements_in.sort(key=lambda x: -triple_count[x])
                    elements_out = [x for x in range(1, 181) if x not in current_set]
                    elements_out.sort(key=lambda x: triple_count[x])
                    break  # Move to next x_remove
        
        # Also try adding elements if the set is not maximal
        # Check if we can add any element without removing another
        for y_add in elements_out:
            if time.time() >= deadline:
                break
            new_set = current_set | {y_add}
            if _is_valid_subset(new_set, triples):
                best_set = new_set
                best_size = len(new_set)
                improved = True
                current_set = new_set
                elements_out = [x for x in range(1, 181) if x not in current_set]
                elements_out.sort(key=lambda x: triple_count[x])
    
    # Try a few random restarts with different greedy orderings
    for restart in range(50):
        if time.time() >= deadline:
            break
        
        # Random ordering
        order = list(range(1, 181))
        random.shuffle(order)
        
        current_set = set()
        for x in order:
            test_set = current_set | {x}
            if _is_valid_subset(test_set, triples):
                current_set = test_set
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
    
    # Final local search on the best set found
    if time.time() < deadline - 5:
        current_set = set(best_set)
        triple_count = {}
        for x in range(1, 181):
            triple_count[x] = 0
        for (a, b, c) in triples:
            triple_count[a] += 1
            triple_count[b] += 1
            triple_count[c] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            elements_in = sorted(current_set, key=lambda x: -triple_count[x])
            elements_out = [x for x in range(1, 181) if x not in current_set]
            elements_out.sort(key=lambda x: triple_count[x])
            
            for x_remove in elements_in:
                if time.time() >= deadline:
                    break
                for y_add in elements_out:
                    if y_add == x_remove:
                        continue
                    new_set = current_set - {x_remove} | {y_add}
                    if _is_valid_subset(new_set, triples):
                        best_set = new_set
                        best_size = len(new_set)
                        improved = True
                        current_set = new_set
                        elements_in = sorted(current_set, key=lambda x: -triple_count[x])
                        elements_out = [x for x in range(1, 181) if x not in current_set]
                        elements_out.sort(key=lambda x: triple_count[x])
                        break
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
