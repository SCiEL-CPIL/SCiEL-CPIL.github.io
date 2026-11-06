"""Rewritten program: beam search over coprime sets to maximize sum 1/(512-a)."""
import time
from math import gcd
from fractions import Fraction
import heapq
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    N = 512
    # Candidate values: we want large 1/(512-a), so a close to 511 is best.
    # But we need pairwise coprime.
    # Greedy seed: pick from 511 down to 1, keeping coprime.
    
    def score_set(chosen):
        s = Fraction(0, 1)
        for a in chosen:
            s += Fraction(1, N - a)
        return s
    
    def greedy_seed():
        chosen = []
        for a in range(511, 0, -1):
            if all(gcd(a, b) == 1 for b in chosen):
                chosen.append(a)
        return chosen
    
    best_set = greedy_seed()
    best_score = score_set(best_set)
    
    # Beam search
    # State: (score_neg, chosen_tuple, next_candidate_index)
    # We want to explore combinations. Since 512 is small, we can use beam search.
    # Candidates are 1..511. We process in order of decreasing weight (increasing a).
    # Actually, let's process candidates in order of decreasing a (511, 510, ...)
    # At each step, we either include the candidate (if coprime with all in set) or skip it.
    # Beam search keeps top K partial solutions.
    
    # Precompute weights
    weights = [Fraction(1, N - a) for a in range(1, 512)]
    
    # Beam search
    beam_width = 500
    # Each beam entry: (score, chosen_list)
    # We process candidates from 511 down to 1.
    # At each candidate a, for each beam entry, we can:
    #   1. Include a (if coprime with all in chosen)
    #   2. Skip a
    # We keep the top beam_width entries by score.
    
    # To make this efficient, we represent chosen as a frozenset or tuple.
    # Since max size is ~100 (number of primes below 512 is 97, plus some composites),
    # tuples are fine.
    
    # Initial beam: one empty set
    beam = [(Fraction(0, 1), ())]
    
    # Process candidates in order of decreasing a (511, 510, ..., 1)
    # For each candidate, update the beam.
    
    for a in range(511, 0, -1):
        if time.time() > deadline:
            break
        
        new_beam = []
        w = weights[a - 1]  # 1/(512-a)
        
        for score, chosen in beam:
            # Option 1: Skip a
            new_beam.append((score, chosen))
            
            # Option 2: Include a (if coprime with all in chosen)
            if all(gcd(a, b) == 1 for b in chosen):
                new_score = score + w
                new_chosen = chosen + (a,)
                new_beam.append((new_score, new_chosen))
        
        # Keep top beam_width entries by score
        # Sort by score descending, take top beam_width
        if len(new_beam) > beam_width:
            # Use heapq.nlargest for efficiency
            new_beam = heapq.nlargest(beam_width, new_beam, key=lambda x: x[0])
        
        beam = new_beam
    
    # Find the best complete solution in the final beam
    # Note: all entries in the final beam are complete (all candidates processed)
    for score, chosen in beam:
        if score > best_score:
            best_score = score
            best_set = list(chosen)
    
    # Also check the greedy seed
    if score_set(best_set) < best_score:
        # This shouldn't happen, but just in case
        pass
    
    # Convert to list of integers
    return list(best_set)

# Let's test with a quick run
if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s, Size: {len(result)}")
    from fractions import Fraction
    s = sum(Fraction(1, 512-a) for a in result)
    print(f"Score: {float(s)}")
