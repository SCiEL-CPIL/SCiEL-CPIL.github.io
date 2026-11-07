"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

def construct():
    n = 256
    deadline = time.time() + 50  # safety margin
    
    # Greedy approach: select largest integers first, maintaining pairwise coprimality
    selected = set()
    selected.add(1)
    
    # Iterate from largest to smallest
    for a in range(n, 1, -1):
        if time.time() > deadline:
            break
        # Check if a is coprime with all already selected
        if all(math.gcd(a, b) == 1 for b in selected):
            selected.add(a)
    
    # Local search: try to swap out smaller numbers for larger ones
    # to increase total sum while maintaining pairwise coprimality
    def is_coprime_to_set(x, s):
        return all(math.gcd(x, y) == 1 for y in s)
    
    best_sum = sum(selected)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each selected element and adding a larger unselected one
        for rem in sorted(selected, reverse=True):
            if time.time() > deadline:
                break
            if rem == 1:
                continue  # never remove 1
            candidate_set = selected - {rem}
            # Find the largest number not in candidate_set that is coprime to all
            for add in range(n, 1, -1):
                if add in candidate_set:
                    continue
                if is_coprime_to_set(add, candidate_set):
                    new_sum = sum(candidate_set) + add
                    if new_sum > best_sum:
                        selected = candidate_set
                        selected.add(add)
                        best_sum = new_sum
                        improved = True
                        break
    
    return [int(x) for x in selected]
# EVOLVE-BLOCK-END
