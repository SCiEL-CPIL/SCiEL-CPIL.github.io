"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    base = 51975
    # Precompute all odd divisors > 1 of base
    divisors = []
    for d in range(3, base + 1, 2):
        if base % d == 0:
            divisors.append(d)

    moduli = divisors
    num_moduli = len(moduli)

    best_score = base
    best_classes = None

    # Precompute index arrays for each modulus to speed up slicing
    # For each modulus m, we precompute the list of indices r, r+m, r+2m, ...
    # Actually, numpy slicing is fast enough. Let's stick with that.

    # Strategy:
    # 1. Greedy initialization with increasing moduli (small moduli first, as they cover more).
    # 2. Local search (simulated annealing / hill climbing) to refine the residue assignment.
    #    We maintain a count array to quickly evaluate the number of uncovered residues.

    def greedy_init(order):
        """
        Initialize residues using greedy method in the given order of moduli.
        Returns (residues, counts, score).
        """
        counts = np.zeros(base, dtype=np.int8)
        residues = [0] * num_moduli
        # Map modulus value to its index in the moduli list for easy lookup?
        # Or just iterate through the order list which contains indices.
        
        for idx in order:
            m = moduli[idx]
            best_r = 0
            best_gain = -1
            # Find residue that maximizes coverage of currently uncovered spots
            # To speed this up, we can sample or use numpy
            # Since m can be up to base, iterating all r in range(m) might be slow for large m.
            # But small m are processed first.
            # Let's just iterate.
            for r in range(m):
                # Count how many spots in r::m have count 0
                # Using numpy slicing
                gain = np.count_nonzero(counts[r::m] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            residues[idx] = best_r
            counts[best_r::m] += 1
        
        score = np.count_nonzero(counts == 0)
        return residues, counts, score

    # Initial greedy pass with increasing moduli (small to large)
    # This is the seed strategy, but we'll use the counts array for tracking.
    order_increasing = sorted(range(num_moduli), key=lambda i: moduli[i])
    residues, counts, score = greedy_init(order_increasing)
    
    if score < best_score:
        best_score = score
        best_classes = [[moduli[i], residues[i]] for i in range(num_moduli)]

    # Local Search / Simulated Annealing
    # We will try to improve by changing the residue of a single modulus at a time.
    # Objective: minimize score (number of uncovered residues).
    
    rng = random.Random(42)
    
    # Temperature schedule
    temperature = 5.0
    cooling = 0.995
    min_temp = 0.1
    
    # To make local search faster, we can precompute the "neighbors" or just random pick.
    # Changing residue r_old to r_new for modulus m:
    # 1. Decrement counts[r_old::m]
    # 2. Increment counts[r_new::m]
    # 3. Calculate new score.
    
    # Optimization: Instead of calculating full score (count of zeros in counts array) every time,
    # we can track the score incrementally?
    # Let U be the set of uncovered residues.
    # When we change r_old to r_new for m:
    # - Residues in r_old::m that were covered ONLY by this modulus become uncovered.
    # - Residues in r_new::m that were uncovered become covered.
    # - Residues in intersection are tricky.
    
    # Given base=51975, np.count_nonzero(counts == 0) is very fast (< 1ms).
    # Let's rely on that.
    
    while time.time() < deadline:
        if temperature < min_temp:
            break
            
        # Pick a random modulus index
        i = rng.randint(0, num_moduli - 1)
        m = moduli[i]
        old_r = residues[i]
        
        # Pick a new random residue
        # We can bias towards residues that cover more uncovered spots?
        # For now, uniform random.
        new_r = rng.randint(0, m - 1)
        
        if new_r == old_r:
            continue
        
        # Apply move
        counts[old_r::m] -= 1
        counts[new_r::m] += 1
        
        new_score = np.count_nonzero(counts == 0)
        delta = new_score - score
        
        if delta < 0:
            # Improvement
            residues[i] = new_r
            score = new_score
            if score < best_score:
                best_score = score
                best_classes = [[moduli[i], residues[i]] for i in range(num_moduli)]
        else:
            # Accept with probability exp(-delta / T)
            if rng.random() < np.exp(-delta / temperature):
                residues[i] = new_r
                score = new_score
            else:
                # Revert
                counts[old_r::m] += 1
                counts[new_r::m] -= 1
        
        temperature *= cooling
        
        # If we get stuck, maybe restart with a different greedy order?
        # Or just keep going.
        
        # Safety check: if time is running out, break early to return best
        if time.time() > deadline - 5.0:
            break

    # If we still have time, we can try a few more greedy restarts with different orders
    # e.g., decreasing moduli, or random orders.
    if time.time() < deadline - 10.0:
        # Try decreasing order
        order_decreasing = sorted(range(num_moduli), key=lambda i: -moduli[i])
        res2, cnt2, score2 = greedy_init(order_decreasing)
        if score2 < best_score:
            best_score = score2
            best_classes = [[moduli[i], res2[i]] for i in range(num_moduli)]
            
            # Do a short local search from this new start
            residues, counts, score = res2, cnt2, score2
            temperature = 2.0
            cooling = 0.99
            min_temp = 0.1
            while time.time() < deadline - 5.0 and temperature > min_temp:
                i = rng.randint(0, num_moduli - 1)
                m = moduli[i]
                old_r = residues[i]
                new_r = rng.randint(0, m - 1)
                if new_r == old_r:
                    continue
                counts[old_r::m] -= 1
                counts[new_r::m] += 1
                new_score = np.count_nonzero(counts == 0)
                delta = new_score - score
                if delta < 0 or rng.random() < np.exp(-delta / temperature):
                    residues[i] = new_r
                    score = new_score
                    if score < best_score:
                        best_score = score
                        best_classes = [[moduli[i], residues[i]] for i in range(num_moduli)]
                else:
                    counts[old_r::m] += 1
                    counts[new_r::m] -= 1
                temperature *= cooling

    # If we still have time, try random orders
    if time.time() < deadline - 10.0:
        for _ in range(3):
            order_random = list(range(num_moduli))
            rng.shuffle(order_random)
            res3, cnt3, score3 = greedy_init(order_random)
            if score3 < best_score:
                best_score = score3
                best_classes = [[moduli[i], res3[i]] for i in range(num_moduli)]
                
                residues, counts, score = res3, cnt3, score3
                temperature = 1.0
                cooling = 0.99
                min_temp = 0.1
                while time.time() < deadline - 5.0 and temperature > min_temp:
                    i = rng.randint(0, num_moduli - 1)
                    m = moduli[i]
                    old_r = residues[i]
                    new_r = rng.randint(0, m - 1)
                    if new_r == old_r:
                        continue
                    counts[old_r::m] -= 1
                    counts[new_r::m] += 1
                    new_score = np.count_nonzero(counts == 0)
                    delta = new_score - score
                    if delta < 0 or rng.random() < np.exp(-delta / temperature):
                        residues[i] = new_r
                        score = new_score
                        if score < best_score:
                            best_score = score
                            best_classes = [[moduli[i], residues[i]] for i in range(num_moduli)]
                    else:
                        counts[old_r::m] += 1
                        counts[new_r::m] -= 1
                    temperature *= cooling

    return best_classes

# EVOLVE-BLOCK-END
