"""Linear triple systems on 16 points with small independence number."""
import random
import time
import sys
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0  # safety margin before 120s limit
    N = 16
    rng = random.Random(12345)
    
    # Precompute all C(16,3) = 560 triples as bitmasks and point tuples
    all_triples = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                mask = (1 << a) | (1 << b) | (1 << c)
                all_triples.append((mask, (a, b, c)))
    num_triples = len(all_triples)  # 560
    
    # Precompute popcount for all 2^16 masks
    popcount = [0] * (1 << N)
    for i in range(1, 1 << N):
        popcount[i] = popcount[i >> 1] + (i & 1)

    def independence_number_fast(triples_masks):
        """
        Fast independence number calculation for a specific set of triples.
        triples_masks: list of int masks
        """
        size = 1 << N
        # is_ind[mask] is True if mask does not contain any triple from triples_masks
        is_ind = [True] * size
        
        # For each triple in the current system, mark all its supersets as non-independent
        for t in triples_masks:
            comp = ((1 << N) - 1) ^ t
            sub = comp
            while True:
                is_ind[t | sub] = False
                if sub == 0:
                    break
                sub = (sub - 1) & comp
            
        # Find max popcount among independent sets
        max_size = 0
        for mask in range(size):
            if is_ind[mask]:
                pc = popcount[mask]
                if pc > max_size:
                    max_size = pc
        return max_size

    def is_linear(triples):
        """Check that any two triples share at most one point."""
        n = len(triples)
        for i in range(n):
            for j in range(i + 1, n):
                overlap = triples[i][0] & triples[j][0]
                if bin(overlap).count('1') > 1:
                    return False
        return True

    def greedy_construct(rng):
        """Build a linear triple system via random greedy."""
        points = list(range(N))
        rng.shuffle(points)
        triples = []
        for p in points:
            # Try to add a triple containing p that maintains linearity
            candidates = [t for t in all_triples if (t[0] >> p) & 1]
            rng.shuffle(candidates)
            for tmask, tpoints in candidates:
                # Check linearity with existing triples
                ok = True
                for emask, _ in triples:
                    if bin(tmask & emask).count('1') > 1:
                        ok = False
                        break
                if ok:
                    triples.append((tmask, tpoints))
                    break
        return triples

    # Phase 1: Generate diverse starting systems
    best_triples = None
    best_alpha = N + 1
    
    # Run multiple greedy restarts
    for i in range(100):
        if time.time() > deadline:
            break
        triples = greedy_construct(rng)
        if not triples:
            continue
        if not is_linear(triples):
            continue
        alpha = independence_number_fast([t[0] for t in triples])
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples
        if best_alpha <= 5:
            break

    if best_triples is None:
        # Fallback
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]

    # Phase 2: Local Search (Hill Climbing with Random Restarts)
    current_triples = best_triples
    current_alpha = best_alpha
    temperature = 1.0
    cooling = 0.99
    
    # We perform swaps: remove one triple, add another valid one
    while time.time() < deadline:
        # Pick a random triple to remove
        idx = rng.randrange(len(current_triples))
        old_mask, old_points = current_triples[idx]
        remaining = [t for j, t in enumerate(current_triples) if j != idx]
        
        # Try to find a replacement triple that maintains linearity
        # and ideally improves or maintains alpha
        found_replacement = False
        # Sample random candidates
        for _ in range(50):
            new_mask, new_points = all_triples[rng.randrange(num_triples)]
            if new_mask == old_mask:
                continue
            
            # Check linearity with remaining triples
            ok = True
            for emask, _ in remaining:
                if bin(new_mask & emask).count('1') > 1:
                    ok = False
                    break
            
            if ok:
                # Candidate is valid. Evaluate new alpha
                candidate_triples = remaining + [(new_mask, new_points)]
                new_alpha = independence_number_fast([t[0] for t in candidate_triples])
                
                # Acceptance criterion
                if new_alpha < current_alpha:
                    current_triples = candidate_triples
                    current_alpha = new_alpha
                    found_replacement = True
                    break
                elif new_alpha == current_alpha:
                    # Accept with some probability to escape local minima
                    if rng.random() < 0.5:
                        current_triples = candidate_triples
                        found_replacement = True
                        break
                else:
                    # Accept worse solution with small probability (simulated annealing style)
                    # delta = new_alpha - current_alpha (positive is bad)
                    if rng.random() < temperature * 0.1:
                        current_triples = candidate_triples
                        current_alpha = new_alpha
                        found_replacement = True
                        break
        
        if not found_replacement:
            # Try to add a new triple if possible (increases system size, might lower alpha)
            # But we must maintain linearity.
            # This is harder. Let's stick to swaps for now to keep system size constant.
            # Or try to remove a triple if it doesn't increase alpha? No, we want to minimize alpha.
            # Removing a triple might INCREASE alpha (bad).
            # So we only want to add or swap.
            # Since we can't easily add without breaking linearity, we rely on swaps.
            
            # If we are stuck, try a random restart with a slightly perturbed greedy
            if rng.random() < 0.05:
                triples = greedy_construct(rng)
                if triples and is_linear(triples):
                    alpha = independence_number_fast([t[0] for t in triples])
                    if alpha < current_alpha:
                        current_triples = triples
                        current_alpha = alpha

        temperature *= cooling
        if temperature < 0.01:
            temperature = 0.01

    return [list(tpoints) for tmask, tpoints in current_triples]
# EVOLVE-BLOCK-END
