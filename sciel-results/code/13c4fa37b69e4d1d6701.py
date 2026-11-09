"""Construct the largest admissible tuple in [0, 5000]."""
import time
import random
import math

N = 5000
MAX_TIME = 290.0  # Safety margin before 330s kill

# Precompute primes up to N (and beyond for checking)
def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(N)

def _is_admissible(S, primes=None):
    """
    Check if S is admissible: for every prime p <= len(S), some residue class mod p is missed.
    Returns True if admissible, False otherwise.
    """
    if primes is None:
        primes = PRIMES
    len_S = len(S)
    for p in primes:
        if p > len_S:
            break
        residues = set(x % p for x in S)
        if len(residues) == p:
            return False
    return True

def _greedy_construct():
    """
    Start with a large set and greedily remove elements to make it admissible.
    Strategy: start with all integers in [0, N], then for each prime p from smallest to largest,
    if all residues mod p are present, remove the least-populated residue class.
    """
    points = list(range(N + 1))
    
    # For each prime p <= len(points), check if all residues are covered
    # If so, remove the smallest residue class
    changed = True
    while changed:
        changed = False
        for p in PRIMES:
            if p > len(points):
                break
            # Count residues
            classes = {}
            for v in points:
                r = v % p
                if r not in classes:
                    classes[r] = []
                classes[r].append(v)
            
            if len(classes) == p:
                # Find the smallest class
                smallest_class = min(classes.values(), key=len)
                # Remove those elements
                remove_set = set(smallest_class)
                points = [v for v in points if v not in remove_set]
                changed = True
                break
    
    return points

def _randomized_greedy(seed=0):
    """
    Randomized variant: start with a random subset and greedily repair.
    """
    random.seed(seed)
    # Start with a random subset of size ~ N * 0.7
    target_size = int(N * 0.7)
    points = random.sample(range(N + 1), target_size)
    
    changed = True
    while changed:
        changed = False
        for p in PRIMES:
            if p > len(points):
                break
            classes = {}
            for v in points:
                r = v % p
                if r not in classes:
                    classes[r] = []
                classes[r].append(v)
            
            if len(classes) == p:
                # Remove the smallest class (with some randomness in tie-breaking)
                # Sort by size, then randomly pick among smallest
                min_size = min(len(c) for c in classes.values())
                smallest_classes = [c for c in classes.values() if len(c) == min_size]
                remove_list = random.choice(smallest_classes)
                remove_set = set(remove_list)
                points = [v for v in points if v not in remove_set]
                changed = True
                break
    
    return points

def construct():
    """
    Construct the largest admissible tuple in [0, 5000].
    Returns a list of distinct integers.
    """
    start_time = time.time()
    
    # First, try the deterministic greedy approach
    best = _greedy_construct()
    best_len = len(best)
    
    elapsed = time.time() - start_time
    if elapsed > MAX_TIME - 30:
        return best
    
    # Try several randomized variants
    best_found = best
    best_len = len(best)
    
    for seed in range(1000):
        elapsed = time.time() - start_time
        if elapsed > MAX_TIME - 5:
            break
        
        candidate = _randomized_greedy(seed=seed)
        if len(candidate) > best_len:
            best_len = len(candidate)
            best_found = candidate
    
    return best_found

# EVOLVE-BLOCK-END
