"""
Strategy: Combinatorial search for periodic binary structures.
The problem of minimizing the maximum overlap of h with shifted 1-h is closely
related to finding sequences with low autocorrelation or specific combinatorial
properties (like Sidon sets). The constant solution h=0.5 has a max overlap of 
N/4 (normalized to 0.5 in some metrics, but here we look at raw sums).
Wait, let's check the score definition.
Score is "upper bound on C5".
The current program returns N=100 heights.
Sum is N/2 = 50.
If h is all 0.5, overlap O(s) = sum h[i](1-h[i+s]) = sum 0.5*0.5 = 0.25 * (N-s).
Max O(s) is at s=0: 0.25 * 100 = 25.
Or if circular: sum 0.25 = 25.
Is the score the raw value of the max overlap?
"score is the upper bound on C5" ... "evaluator measures the raw value (lower is better)".
"search score is seed/value". Seed value is likely the score of the seed program.
Seed program returns constant 0.5.
If seed score is 1.0, and it returns constant 0.5, then the "value" (raw overlap) must be such that seed/value = 1.0 => value = seed_value.
Let's assume the raw value for constant 0.5 is the baseline.

If we use a binary pattern (0s and 1s) with exactly 50 ones and 50 zeros:
Overlap O(s) = sum h[i] * (1-h[i+s]).
If h[i]=1, 1-h[i]=0. If h[i]=0, 1-h[i]=1.
O(s) = sum_{i: h[i]=1} (1-h[i+s]) + sum_{i: h[i]=0} (0)
O(s) = count of i where h[i]=1 and h[i+s]=0.
This is the number of 1->0 transitions at shift s (in a circular sense, effectively the cross-correlation of the binary sequence with its complement).
For a random binary sequence, this is roughly 25.
For a structured sequence (like a Barker sequence or similar low-correlation set), can we get lower?
Actually, the minimum possible max overlap for a binary sequence with 50 ones is related to the difference set properties.
If we can find a sequence where the number of 1s followed by 0s at any shift is minimized, we win.

Let's try to construct a sequence based on a known low-correlation structure or simply search for a binary periodic pattern.
Since N=100, we can try periods P that divide 100 (1, 2, 4, 5, 10, 20, 25, 50, 100).
We will use a Genetic Algorithm to optimize the bits of a period P.
The fitness function will be the maximum circular overlap of the extended sequence.
"""

import numpy as np
import time

N = 100

def compute_max_overlap_circular(h):
    """
    Compute max circular overlap of h with 1-h.
    O(s) = sum_{i=0}^{N-1} h[i] * (1 - h[(i+s) % N])
    """
    n = len(h)
    one_minus_h = 1.0 - h
    # Circular cross-correlation
    H = np.fft.fft(h)
    O = np.fft.fft(one_minus_h)
    corr = np.fft.ifft(H * np.conj(O)).real
    # corr[k] corresponds to shift k
    # We want max over all shifts
    return np.max(corr)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Baseline: Constant 0.5
    best_h = np.full(N, 0.5)
    best_score = compute_max_overlap_circular(best_h)
    
    # Strategy: Search for binary periodic structures
    # We will test periods P that divide N.
    # For each period, we represent the pattern as a binary vector of length P.
    # We need the total sum of the extended sequence to be N/2 = 50.
    # If the period has k ones, the extended sequence has k * (N/P) ones.
    # We need k * (N/P) = 50 => k = 50 * P / N.
    # So P must be such that 50 * P / N is an integer.
    # N=100, so k = P/2. P must be even.
    
    candidate_periods = [2, 4, 10, 20, 50, 100]
    
    # Genetic Algorithm setup
    def fitness(pattern, period):
        # pattern is a binary array of length 'period'
        # Extend to length N
        reps = N // period
        h = np.tile(pattern, reps)
        
        # Check sum constraint
        # It's guaranteed by construction if k=period/2, but let's be safe
        if np.sum(h) != N / 2:
            return float('inf')
            
        # Compute max overlap
        return compute_max_overlap_circular(h)

    def mutate(pattern, rate=0.1):
        p = pattern.copy()
        mask = np.random.random(len(p)) < rate
        # Flip bits
        p[mask] = 1 - p[mask]
        # Ensure exactly half are 1s? 
        # If we flip, the count changes.
        # We should use a swap mutation or repair.
        # Let's repair: if count > period/2, flip random 1s to 0s.
        # If count < period/2, flip random 0s to 1s.
        target_ones = period // 2
        current_ones = np.sum(p)
        if current_ones > target_ones:
            ones_idx = np.where(p == 1)[0]
            to_flip = np.random.choice(ones_idx, size=int(current_ones - target_ones), replace=False)
            p[to_flip] = 0
        elif current_ones < target_ones:
            zeros_idx = np.where(p == 0)[0]
            to_flip = np.random.choice(zeros_idx, size=int(target_ones - current_ones), replace=False)
            p[to_flip] = 1
        return p

    def crossover(p1, p2):
        # Uniform crossover
        mask = np.random.random(len(p1)) < 0.5
        child = np.where(mask, p1, p2)
        # Repair to ensure correct count
        target_ones = len(p1) // 2
        current_ones = np.sum(child)
        if current_ones > target_ones:
            ones_idx = np.where(child == 1)[0]
            to_flip = np.random.choice(ones_idx, size=int(current_ones - target_ones), replace=False)
            child[to_flip] = 0
        elif current_ones < target_ones:
            zeros_idx = np.where(child == 0)[0]
            to_flip = np.random.choice(zeros_idx, size=int(target_ones - current_ones), replace=False)
            child[to_flip] = 1
        return child

    np.random.seed(42)
    
    for period in candidate_periods:
        if time.time() > deadline - 10:
            break
            
        # Initialize population
        pop_size = 50
        # Initial population: random valid patterns
        population = []
        for _ in range(pop_size):
            # Create random permutation of 1s and 0s
            pat = np.zeros(period)
            ones_idx = np.random.choice(period, size=period//2, replace=False)
            pat[ones_idx] = 1
            population.append(pat)
            
        # Evaluate initial population
        scores = [fitness(p, period) for p in population]
        
        # GA Loop
        gen = 0
        max_gens = 200
        while gen < max_gens:
            if time.time() > deadline - 5:
                break
                
            # Sort by fitness (lower is better)
            indices = np.argsort(scores)
            population = [population[i] for i in indices]
            scores = [scores[i] for i in indices]
            
            # Elitism: keep top 10
            new_population = population[:10]
            
            # Generate offspring
            while len(new_population) < pop_size:
                # Selection: Tournament
                def select():
                    idxs = np.random.randint(0, pop_size, size=3)
                    best_idx = min(idxs, key=lambda x: scores[x])
                    return population[best_idx]
                
                p1 = select()
                p2 = select()
                
                # Crossover
                child = crossover(p1, p2)
                
                # Mutation
                child = mutate(child, rate=0.2)
                
                new_population.append(child)
                
            population = new_population
            scores = [fitness(p, period) for p in population]
            gen += 1
            
            # Check best in this period
            best_period_score = min(scores)
            if best_period_score < best_score:
                best_score = best_period_score
                # Reconstruct best_h
                best_pattern = population[np.argmin(scores)]
                best_h = np.tile(best_pattern, N // period)
                
    # Final check
    # It's possible a non-binary, non-periodic solution is better, 
    # but binary periodic is a strong candidate for "structured" low overlap.
    # Let's also try a simple local search on the best found binary pattern 
    # allowing non-binary values? No, stick to the structural change.
    
    return best_h

# EVOLVE-BLOCK-END
