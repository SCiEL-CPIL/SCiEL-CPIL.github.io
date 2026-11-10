import time
import random
import math
import numpy as np
from itertools import product

N = 5001 # Range [0, 5000]
DEADLINE = 290.0

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

# We need primes up to 5003 to guarantee validity for any subset of [0, 5000]
PRIMES = _primes(5003)
P_COUNT = len(PRIMES)

def _compute_score_and_cover(residues):
    """
    Computes cover_count and score for a given residue assignment.
    """
    cover_count = np.zeros(N, dtype=np.int32)
    for i, p in enumerate(PRIMES):
        r = residues[i]
        cover_count[r::p] += 1
    score = N - np.sum(cover_count > 0)
    return cover_count, score

def _greedy_init():
    """
    Greedy initialization: pick residue for each prime that removes fewest available numbers.
    """
    available = np.ones(N, dtype=bool)
    residues = [0] * P_COUNT
    for i, p in enumerate(PRIMES):
        min_count = N
        best_r = 0
        for r in range(p):
            count = available[r::p].sum()
            if count < min_count:
                min_count = count
                best_r = r
        residues[i] = best_r
        available[best_r::p] = False
    return residues

def _hill_climb(residues, cover_count, current_score, time_limit):
    """
    Greedy Hill Climbing: find best single move, apply, repeat until local optimum.
    Returns updated residues, cover_count, current_score.
    """
    improved = True
    stall_count = 0
    while improved and time.time() < time_limit:
        improved = False
        best_delta = 0
        best_move = None
        stall_count += 1
        
        # For larger stall counts, be more aggressive in exploring residues
        if stall_count <= 3:
            small_prime_threshold = 300
            large_prime_sample = 80
        elif stall_count <= 10:
            small_prime_threshold = 500
            large_prime_sample = 150
        else:
            small_prime_threshold = 800
            large_prime_sample = 300
        
        for i, p in enumerate(PRIMES):
            old_r = residues[i]
            # Calculate gain from removing old_r
            # Elements with count 1 in old_r::p will become 0 (good)
            gain_old = np.sum(cover_count[old_r::p] == 1)
            
            # Optimization: If gain_old is 0, no move for this prime can improve score.
            if gain_old == 0:
                continue
            
            min_loss = N
            best_new_r = -1
            
            if p <= small_prime_threshold:
                # Direct loop for small primes - check all residues
                for r in range(p):
                    if r == old_r:
                        continue
                    loss = np.sum(cover_count[r::p] == 0)
                    if loss < min_loss:
                        min_loss = loss
                        best_new_r = r
            else:
                # For large primes, use adaptive sampling
                # Sample more densely as we stall
                step = max(1, p // large_prime_sample)
                candidates = list(range(0, p, step))
                # Also include residues near old_r (local search)
                for dr in range(-5, 6):
                    nr = (old_r + dr) % p
                    if nr not in candidates:
                        candidates.append(nr)
                
                for r in candidates:
                    if r == old_r:
                        continue
                    loss = np.sum(cover_count[r::p] == 0)
                    if loss < min_loss:
                        min_loss = loss
                        best_new_r = r
            
            if best_new_r != -1:
                delta = gain_old - min_loss
                if delta > best_delta:
                    best_delta = delta
                    best_move = (i, best_new_r)
                    
        if best_move is not None:
            i, new_r = best_move
            p = PRIMES[i]
            old_r = residues[i]
            
            # Apply move
            residues[i] = new_r
            
            # Update cover_count
            cover_count[old_r::p] -= 1
            cover_count[new_r::p] += 1
            
            current_score += best_delta
            improved = True
            stall_count = 0  # Reset stall counter on improvement
            
    return residues, cover_count, current_score

def _branch_first_primes():
    """
    Branch over all combinations of residues for the first 5 primes (2,3,5,7,11).
    This explores 2*3*5*7*11 = 2310 distinct initial configurations.
    For each, do a quick greedy fill + hill climb, keep the best.
    """
    start_time = time.time()
    first_5_primes = PRIMES[:5]
    first_5_ranges = [range(p) for p in first_5_primes]
    
    best_score = -1
    best_residues = None
    best_cover = None
    
    # We'll iterate through all 2310 combinations
    for combo in product(*first_5_ranges):
        if time.time() > start_time + 120:  # 120 second budget for branching
            break
        
        # Initialize residues with the combo for first 5 primes
        residues = list(combo) + [0] * (P_COUNT - 5)
        
        # Complete with greedy for remaining primes
        available = np.ones(N, dtype=bool)
        for i, p in enumerate(PRIMES[:5]):
            r = residues[i]
            available[r::p] = False
        
        for i in range(5, P_COUNT):
            p = PRIMES[i]
            min_count = N
            best_r = 0
            for r in range(p):
                count = available[r::p].sum()
                if count < min_count:
                    min_count = count
                    best_r = r
            residues[i] = best_r
            available[best_r::p] = False
        
        # Compute score
        cover_count, current_score = _compute_score_and_cover(residues)
        
        if current_score > best_score:
            best_score = current_score
            best_residues = residues.copy()
            best_cover = cover_count.copy()
    
    return best_residues, best_cover, best_score

def construct():
    start_time = time.time()
    
    # Phase 1: Branch over first 5 primes to find a good starting point
    best_residues, best_cover, best_score = _branch_first_primes()
    
    if best_residues is None:
        # Fallback to greedy
        residues = _greedy_init()
        best_cover, best_score = _compute_score_and_cover(residues)
        best_residues = residues.copy()
    
    # Phase 2: Hill climb from the best branched start
    residues = best_residues.copy()
    cover_count = best_cover.copy()
    current_score = best_score
    
    residues, cover_count, current_score = _hill_climb(residues, cover_count, current_score, DEADLINE - 40)
    
    if current_score > best_score:
        best_score = current_score
        best_residues = residues.copy()
    
    # Phase 3: Restarts with Perturbation
    rng = np.random.default_rng()
    phase3_deadline = DEADLINE - 20
    
    # Track how many restarts we've done to adapt perturbation size
    restart_count = 0
    while time.time() < phase3_deadline:
        restart_count += 1
        
        # Adapt perturbation size: start smaller, increase if stuck
        if restart_count <= 5:
            num_perturb = rng.integers(1, 4)  # Small perturbations first
        elif restart_count <= 15:
            num_perturb = rng.integers(2, 6)  # Medium perturbations
        else:
            num_perturb = rng.integers(3, 10)  # Larger perturbations to escape deep local optima
            
        perturbed_residues = residues.copy()
        indices = rng.choice(P_COUNT, size=num_perturb, replace=False)
        for i in indices:
            p = PRIMES[i]
            # Bias perturbation toward residues that currently have low coverage
            # This helps escape local optima by trying "unexplored" regions
            if rng.random() < 0.3:
                # Try a random residue
                perturbed_residues[i] = rng.integers(0, p)
            else:
                # Try residues near the current one (local perturbation)
                dr = rng.integers(-3, 4)
                perturbed_residues[i] = (residues[i] + dr) % p
            
        # Recalculate state for perturbed
        perturbed_cover, perturbed_score = _compute_score_and_cover(perturbed_residues)
        
        # Hill Climb from perturbed with remaining time
        remaining = phase3_deadline - time.time()
        if remaining < 5:
            break
        p_res, p_cover, p_score = _hill_climb(perturbed_residues, perturbed_cover, perturbed_score, phase3_deadline - 5)
        
        if p_score > best_score:
            best_score = p_score
            best_residues = p_res.copy()
            
        # Update current state to the result of this climb
        residues = p_res
        cover_count = p_cover
        current_score = p_score
    
    # Phase 4: Final polish - try very small perturbations (1-2 changes)
    polish_deadline = DEADLINE - 5
    polish_count = 0
    while time.time() < polish_deadline:
        polish_count += 1
        perturbed_residues = best_residues.copy()
        # Use 1 change for most, occasionally 2
        if polish_count % 3 == 0:
            num_perturb = 2
        else:
            num_perturb = 1
        indices = rng.choice(P_COUNT, size=num_perturb, replace=False)
        for i in indices:
            p = PRIMES[i]
            # Local perturbation: try nearby residues
            if rng.random() < 0.5:
                dr = rng.integers(-2, 3)
                perturbed_residues[i] = (best_residues[i] + dr) % p
            else:
                perturbed_residues[i] = rng.integers(0, p)
            
        perturbed_cover, perturbed_score = _compute_score_and_cover(perturbed_residues)
        
        remaining = polish_deadline - time.time()
        if remaining < 3:
            break
        p_res, p_cover, p_score = _hill_climb(perturbed_residues, perturbed_cover, perturbed_score, polish_deadline - 2)
        
        if p_score > best_score:
            best_score = p_score
            best_residues = p_res.copy()
    
    # Reconstruct the set from best_residues
    mask = np.ones(N, dtype=bool)
    for i, p in enumerate(PRIMES):
        r = best_residues[i]
        mask[r::p] = False
        
    result = [x for x in range(N) if mask[x]]
    
    return result
