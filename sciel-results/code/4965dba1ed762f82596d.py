import time
import random
import math
import heapq
import numpy as np
from fractions import Fraction

def construct():
    t0 = time.time()
    deadline = t0 + 118  # Safety margin

    # Generate primes up to 1000
    def _primes_upto(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        return [i for i in range(2, n+1) if sieve[i]]

    all_primes = _primes_upto(1000)
    
    # Select primes: keep small primes (high density), drop largest to stay within budget
    selected_primes = all_primes[:]
    cost = sum(Fraction(1, p) for p in selected_primes)
    while cost > Fraction(1, 1) and len(selected_primes) > 1:
        # Remove the largest prime (last element)
        removed = selected_primes.pop()
        cost -= Fraction(1, removed)

    N = len(selected_primes)
    MAX_L = 100000  # Cap at 10^5 as specified

    # Initialize residues: use a structured initial assignment
    # Strategy: assign residues to maximize initial contiguous coverage
    residues = [0] * N
    
    # Initialize counts using NumPy
    counts = np.zeros(MAX_L + 1, dtype=np.int16)
    
    # Better initialization: greedy assignment to cover prefix
    # For each prime in order of size, pick residue that covers most new positions in 1..L
    # Start with a reasonable initial state
    for i in range(N):
        p = selected_primes[i]
        # Try to pick residue that aligns with early coverage
        residues[i] = i % p
        counts[residues[i]::p] += 1

    def get_L():
        """Find the largest L such that all 1..L are covered."""
        # Find first index >= 1 where count is 0
        zero_mask = counts[1:] == 0
        if not np.any(zero_mask):
            return MAX_L
        first_zero = np.argmax(zero_mask)
        return first_zero  # This is the index in counts[1:], so position = first_zero + 1... wait
        # Actually: counts[1:] has indices 0..MAX_L-1 corresponding to positions 1..MAX_L
        # If first zero is at index k in counts[1:], then position k+1 is uncovered
        # So L = k (all positions 1..k are covered)
    
    # Fix get_L
    def get_L():
        zero_mask = counts[1:] == 0
        if not np.any(zero_mask):
            return MAX_L
        first_zero_idx = np.argmax(zero_mask)
        # Position (first_zero_idx + 1) is the first uncovered
        # So all positions 1..first_zero_idx are covered
        return first_zero_idx

    L = get_L()
    best_L = L
    best_residues = residues[:]
    
    # Build a priority queue of "gap positions" weighted by how hard they are to cover
    # A position n is covered if any prime p has n % p == residues[i]
    # For the greedy phase, we focus on the first gap
    
    # Phase 1: Greedy gap-filling with targeted residue reassignment
    # For each gap, find the prime that minimizes "damage" when reassigned to cover the gap
    phase1_deadline = t0 + 40  # 40 seconds for greedy phase
    
    while time.time() < phase1_deadline:
        if L >= MAX_L:
            break
        
        first_gap = L + 1
        
        # Find all primes that could cover first_gap
        # For prime i, we need new_r = first_gap % p_i
        # Score each candidate by how many currently-covered positions would become uncovered
        best_i = -1
        best_score = float('inf')
        best_new_r = 0
        
        for i in range(N):
            p = selected_primes[i]
            new_r = first_gap % p
            old_r = residues[i]
            
            if new_r == old_r:
                # Already covers first_gap (shouldn't happen if L is correct, but safe)
                continue
            
            # Compute damage: positions in old_r::p that are NOT in new_r::p and have count == 1
            # These are positions that would become uncovered
            # Efficient: iterate over positions in old_r::p up to L
            # and check if they're in new_r::p
            
            # Positions in old_r::p: old_r, old_r+p, old_r+2p, ...
            # A position x is in new_r::p iff x % p == new_r
            # x = old_r + k*p, so x % p = old_r % p = old_r (since old_r < p)
            # So x is in new_r::p iff old_r == new_r, which is false
            # Wait, that's not right. Let me reconsider.
            # x = old_r + k*p for k >= 0
            # x % p = (old_r + k*p) % p = old_r % p = old_r (since 0 <= old_r < p)
            # So all positions in old_r::p have remainder old_r mod p
            # They are in new_r::p iff old_r == new_r
            # Since old_r != new_r, NO position in old_r::p is in new_r::p
            # So ALL positions in old_r::p that had count 1 will become uncovered!
            
            # Count positions in old_r::p from 1 to L with count == 1
            # These are the "critical" positions
            pos = old_r
            if pos < 1:
                pos += ((1 - pos + p - 1) // p) * p
            # Now pos >= 1 and pos % p == old_r
            damage = 0
            for pos in range(pos, L + 1, p):
                if counts[pos] == 1:
                    damage += 1
            
            # Also consider: how many new positions beyond L get covered?
            # Positions in new_r::p beyond L: new_r, new_r+p, ...
            # This helps extend coverage
            bonus = 0
            pos = new_r
            if pos < 1:
                pos += ((1 - pos + p - 1) // p) * p
            # Count positions in new_r::p from L+1 to some horizon
            for pos in range(pos, min(L + 200, MAX_L + 1), p):
                if counts[pos] == 0:
                    bonus += 1
            
            # Score: damage - bonus (lower is better)
            score = damage - bonus
            
            if score < best_score:
                best_score = score
                best_i = i
                best_new_r = new_r
        
        if best_i == -1:
            # No prime can cover first_gap without being already assigned
            # This shouldn't happen, but break to avoid infinite loop
            break
        
        # Apply the best change
        i = best_i
        p = selected_primes[i]
        old_r = residues[i]
        new_r = best_new_r
        
        counts[old_r::p] -= 1
        counts[new_r::p] += 1
        residues[i] = new_r
        
        new_L = get_L()
        
        if new_L > best_L:
            best_L = new_L
            best_residues = residues[:]
        
        L = new_L
    
    # Phase 2: Local search with targeted perturbations
    # Alternate between:
    #   (a) Targeted: pick a random gap position, try to cover it
    #   (b) Global: random perturbation with acceptance criterion
    
    phase2_deadline = deadline
    
    # Track current state
    current_L = L
    current_residues = residues[:]
    
    # Use a hill-climbing approach with occasional restarts
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    no_improve_count = 0
    
    while time.time() < phase2_deadline:
        # Check for stagnation
        if no_improve_count > 200:
            # Reset temperature and try a different strategy
            T = max(T_min, T * 2)
            no_improve_count = 0
        
        # Choose strategy: 70% targeted, 30% global
        if random.random() < 0.7:
            # Targeted: find a gap position and try to cover it
            # Pick a random gap position in [current_L + 1, current_L + 50]
            gap_start = current_L + 1
            gap_end = min(current_L + 50, MAX_L)
            if gap_start > gap_end:
                break
            
            # Find actual gaps in this range
            gap_positions = []
            for pos in range(gap_start, gap_end + 1):
                if counts[pos] == 0:
                    gap_positions.append(pos)
            
            if not gap_positions:
                # No gaps in range, update L
                current_L = get_L()
                if current_L > best_L:
                    best_L = current_L
                    best_residues = current_residues[:]
                continue
            
            target_pos = random.choice(gap_positions)
            
            # Find best prime to reassign to cover target_pos
            best_i = -1
            best_score = float('inf')
            best_new_r = 0
            
            for i in range(N):
                p = selected_primes[i]
                new_r = target_pos % p
                old_r = current_residues[i]
                
                if new_r == old_r:
                    continue
                
                # Compute damage
                pos = old_r
                if pos < 1:
                    pos += ((1 - pos + p - 1) // p) * p
                damage = 0
                for pos in range(pos, current_L + 1, p):
                    if counts[pos] == 1:
                        damage += 1
                
                score = damage
                
                if score < best_score:
                    best_score = score
                    best_i = i
                    best_new_r = new_r
            
            if best_i == -1:
                continue
            
            i = best_i
            p = selected_primes[i]
            old_r = current_residues[i]
            new_r = best_new_r
            
            counts[old_r::p] -= 1
            counts[new_r::p] += 1
            current_residues[i] = new_r
            
            new_L = get_L()
            
            if new_L >= current_L:
                current_L = new_L
                if new_L > best_L:
                    best_L = new_L
                    best_residues = current_residues[:]
                no_improve_count = 0
            else:
                # Revert
                counts[new_r::p] -= 1
                counts[old_r::p] += 1
                current_residues[i] = old_r
                no_improve_count += 1
        else:
            # Global: random perturbation
            i = random.randint(0, N - 1)
            p = selected_primes[i]
            old_r = current_residues[i]
            new_r = random.randint(0, p - 1)
            
            if old_r == new_r:
                continue
            
            counts[old_r::p] -= 1
            counts[new_r::p] += 1
            
            new_L = get_L()
            
            if new_L > current_L:
                current_L = new_L
                current_residues[i] = new_r
                if new_L > best_L:
                    best_L = new_L
                    best_residues = current_residues[:]
                no_improve_count = 0
            else:
                delta = current_L - new_L
                if delta == 0:
                    if random.random() < 0.3:
                        current_residues[i] = new_r
                    else:
                        counts[new_r::p] -= 1
                        counts[old_r::p] += 1
                else:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        current_residues[i] = new_r
                        current_L = new_L
                    else:
                        counts[new_r::p] -= 1
                        counts[old_r::p] += 1
                        no_improve_count += 1
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
    
    return [[selected_primes[i], best_residues[i]] for i in range(N)]
