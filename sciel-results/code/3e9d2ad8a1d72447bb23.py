import math
import time
import random
import numpy as np

N = 5000
RANGE = N + 1

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(N)

# Precompute residue masks
# residue_masks[p][r] is a bitmask of integers in [0, N] congruent to r mod p
RESIDUE_MASKS = {}
for p in PRIMES:
    masks = []
    for r in range(p):
        mask = 0
        # Step p, starting at r
        for k in range(r, RANGE, p):
            mask |= (1 << k)
        masks.append(mask)
    RESIDUE_MASKS[p] = masks

FULL_MASK = (1 << RANGE) - 1

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin for 300s limit
    
    best_size = 0
    best_mask = 0
    
    # Run multiple randomized greedy attempts
    while time.time() < deadline:
        current_mask = FULL_MASK
        current_size = RANGE
        
        # Iterate through primes in increasing order
        for p in PRIMES:
            if p > current_size:
                break
            
            # Count active elements in each residue class mod p
            counts = []
            masks_p = RESIDUE_MASKS[p]
            for r in range(p):
                # Intersect current active mask with the mask for residue r
                # Count bits
                cnt = (current_mask & masks_p[r]).bit_count()
                counts.append(cnt)
            
            min_count = min(counts)
            
            # Find all residue classes with minimum count
            candidates = []
            for r in range(p):
                if counts[r] == min_count:
                    candidates.append(r)
            
            # Randomly pick one of the minimum count classes
            r = random.choice(candidates)
            
            # Remove elements in this class
            current_mask &= ~masks_p[r]
            current_size -= min_count
            
            if current_size == 0:
                break
        
        if current_size > best_size:
            best_size = current_size
            best_mask = current_mask
            
        # Optional: Early exit if we reach a very high size?
        # The theoretical max is not known precisely, but around 400-500.
        # If we hit 500, we might keep searching for 501?
        # No hard cap known, but diminishing returns.
        # However, if best_size is very high, we might want to continue to ensure we didn't miss a slightly better one.
        # Given the speed, we can just run until deadline.

    # Convert best_mask to list of integers
    points = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        points.append(idx)
        m ^= lsb
        
    return points
