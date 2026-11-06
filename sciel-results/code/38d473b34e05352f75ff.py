"""Improved program for the triple systems task."""
import random
import time
import sys
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    n = 16
    # Precompute all C(16,3) = 560 triples as tuples
    all_triples = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                all_triples.append((a, b, c))
    
    num_triples = len(all_triples)  # 560
    
    # Precompute for each triple index, the list of (a', b', c') combinations
    # that would form a K(2,2,2) with this triple
    # For triple (a,b,c), a' != a, b' != b, c' != c, all distinct
    # The 7 other triples are determined
    
    # For fast lookup, we'll use a bitmask representation
    # Each triple gets an index 0..559
    triple_to_index = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute: for each triple index, which other 7 triple indices
    # complete a K(2,2,2) with it, for each valid (a',b',c') choice
    # This is expensive but done once
    
    # Actually, let's use a different approach for speed:
    # For each triple T=(a,b,c), precompute all possible "completions"
    # A completion is a set of 7 other triples such that T + those 7 = K(2,2,2)
    
    # For T=(a,b,c), the completions are indexed by (a',b',c') where
    # a' in {0..15}\{a}, b' in {0..15}\{b,a'}, c' in {0..15}\{c,a',b'}
    # The 7 other triples are:
    # (a',b,c), (a,b',c), (a,b,c'), (a',b',c), (a',b,c'), (a,b',c'), (a',b',c')
    
    # Precompute completion lists
    completions = [[] for _ in range(num_triples)]
    for idx, (a, b, c) in enumerate(all_triples):
        for a_prime in range(n):
            if a_prime == a:
                continue
            for b_prime in range(n):
                if b_prime == b or b_prime == a_prime:
                    continue
                for c_prime in range(n):
                    if c_prime == c or c_prime == a_prime or c_prime == b_prime:
                        continue
                    other_triples = [
                        (a_prime, b, c),
                        (a, b_prime, c),
                        (a, b, c_prime),
                        (a_prime, b_prime, c),
                        (a_prime, b, c_prime),
                        (a, b_prime, c_prime),
                        (a_prime, b_prime, c_prime),
                    ]
                    # Check all are valid triples (they should be since points are distinct)
                    # Convert to indices
                    other_indices = []
                    valid = True
                    for ot in other_triples:
                        if ot not in triple_to_index:
                            valid = False
                            break
                        other_indices.append(triple_to_index[ot])
                    if valid:
                        completions[idx].append(tuple(other_indices))
    
    # Greedy with randomized restarts
    best_triples = []
    best_size = 0
    
    # Use a faster check: maintain a set of present triple indices
    # When adding triple idx, check if any completion of idx has all 7 others present
    
    def can_add(idx, present_set):
        """Check if adding triple idx would complete a K(2,2,2)."""
        for comp in completions[idx]:
            all_present = True
            for oi in comp:
                if oi not in present_set:
                    all_present = False
                    break
            if all_present:
                return False
        return True
    
    # Multiple random restarts with greedy + local search
    for trial in range(200):
        if time.time() > deadline - 5:
            break
            
        random.shuffle(all_triples)
        present_set = set()
        current_triples = []
        
        # Greedy phase
        for triple in all_triples:
            if time.time() > deadline - 15:
                break
            idx = triple_to_index[triple]
            if can_add(idx, present_set):
                present_set.add(idx)
                current_triples.append(list(triple))
        
        if len(current_triples) > best_size:
            best_size = len(current_triples)
            best_triples = [t[:] for t in current_triples]
        
        # Local search: try to add more triples that were skipped
        if time.time() > deadline - 10:
            break
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for triple in all_triples:
                idx = triple_to_index[triple]
                if idx in present_set:
                    continue
                if can_add(idx, present_set):
                    present_set.add(idx)
                    current_triples.append(list(triple))
                    improved = True
        
        if len(current_triples) > best_size:
            best_size = len(current_triples)
            best_triples = [t[:] for t in current_triples]
        
        # Kick: remove a random triple and try to refill
        if time.time() > deadline - 8:
            break
        
        if len(current_triples) > 5:
            # Remove a random triple
            rem_idx = random.randint(0, len(current_triples) - 1)
            rem_triple = current_triples.pop(rem_idx)
            rem_idx_val = triple_to_index[tuple(rem_triple)]
            present_set.discard(rem_idx_val)
            
            # Try to add triples back, including the removed one
            kick_improved = True
            while kick_improved and time.time() < deadline - 8:
                kick_improved = False
                for triple in all_triples:
                    idx = triple_to_index[triple]
                    if idx in present_set:
                        continue
                    if can_add(idx, present_set):
                        present_set.add(idx)
                        current_triples.append(list(triple))
                        kick_improved = True
            
            if len(current_triples) > best_size:
                best_size = len(current_triples)
                best_triples = [t[:] for t in current_triples]
            
            # If kick didn't help, we could revert but let's keep going
    
    return best_triples
# EVOLVE-BLOCK-END
