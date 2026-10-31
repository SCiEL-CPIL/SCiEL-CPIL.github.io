import time
import random
import math
import numpy as np

_START = time.time()
_DEADLINE = _START + 140.0  # Safety margin vs 150s kill

def _fft_score(u):
    """
    Fast scoring using FFT to count unique sums and differences.
    O(M log M) where M is the span.
    """
    if not u or len(u) < 2:
        return 0.0
    m = max(u)
    if m < 1:
        return 0.0
    
    u_arr = np.array(u, dtype=np.int64)
    
    # Size of FFT needs to be at least 2*max(U) + 1
    size = 2 * m + 1
    n_fft = 1
    while n_fft < size:
        n_fft *= 2
        
    # Indicator vector
    f = np.zeros(n_fft, dtype=np.float64)
    f[u_arr] = 1.0
    
    # FFT
    F = np.fft.fft(f)
    
    # Sums: Convolution of f with f.
    conv = np.fft.ifft(F * F).real
    unique_sums = np.sum(conv[:size] > 0.5)
    
    # Diffs: Autocorrelation.
    autocorr = np.fft.ifft(F * np.conj(F)).real
    
    # Unique diffs are in [-m, m]. In circular convolution, 
    # positive diffs are in [0, m] and negative diffs are wrapped to [n_fft-m, n_fft-1].
    mask = np.zeros(n_fft, dtype=bool)
    mask[0] = True
    mask[1:m+1] = True
    mask[n_fft-m:n_fft] = True
    
    unique_diffs = np.sum(autocorr[mask] > 0.5)
    
    if unique_sums == 0:
        return 0.0
        
    ratio = unique_diffs / float(unique_sums)
    if ratio <= 0:
        return 0.0
        
    return 1.0 + math.log(ratio) / math.log(2 * m + 1)

def _ga_evaluate(population, max_span):
    """
    Evaluate a population of sets.
    population: list of lists of ints
    Returns: list of scores
    """
    scores = []
    for u in population:
        scores.append(_fft_score(u))
    return scores

def _ga_crossover(u1, u2, max_span):
    """
    Crossover: Uniform crossover on the binary representation of the sets.
    """
    # Create a set for fast lookup
    set1 = set(u1)
    set2 = set(u2)
    
    # Combine all elements from both
    all_elements = set1.union(set2)
    
    new_u = [0] # Always keep 0
    
    # For each element (except 0), randomly pick from parent 1 or 2
    # This effectively performs a "mask" crossover
    for elem in all_elements:
        if elem == 0:
            continue
        # Probability of taking from parent 1 is 0.5
        if random.random() < 0.5:
            if elem in set1:
                new_u.append(elem)
        else:
            if elem in set2:
                new_u.append(elem)
                
    # Ensure uniqueness and sort
    new_u = sorted(list(set(new_u)))
    return new_u

def _ga_mutate(u, max_span, mutation_rate=0.1):
    """
    Mutation: Randomly add, remove, or shift elements.
    """
    u = u[:] # Copy
    n = len(u)
    
    # Number of mutations
    num_mutations = max(1, int(n * mutation_rate))
    
    for _ in range(num_mutations):
        op = random.choice(['add', 'remove', 'shift'])
        
        if op == 'add':
            # Add a random element
            if len(u) < 200: # Limit size
                v = random.randint(1, max_span)
                if v not in u:
                    u.append(v)
                    u.sort()
                    
        elif op == 'remove':
            # Remove a random non-zero element
            if len(u) > 2:
                idx = random.randint(1, len(u)-1)
                u.pop(idx)
                
        elif op == 'shift':
            # Shift a random non-zero element
            if len(u) > 2:
                idx = random.randint(1, len(u)-1)
                old_val = u[idx]
                # Small shift
                delta = random.randint(-max(1, max_span//100), max(1, max_span//100))
                new_val = old_val + delta
                if 1 <= new_val <= max_span and new_val not in u:
                    u[idx] = new_val
                    u.sort()
                    
    return u

def _ga_select(population, scores, tournament_size=5):
    """
    Tournament selection.
    """
    idxs = random.sample(range(len(population)), tournament_size)
    best_idx = max(idxs, key=lambda i: scores[i])
    return population[best_idx]

def construct():
    global _DEADLINE
    
    random.seed(42)
    np.random.seed(42)
    
    best_u = [0, 1]
    best_score = _fft_score(best_u)
    
    # GA Parameters
    pop_size = 50
    max_span = 20000 # Initial span, can grow
    generations = 0
    
    # Initialize population with diverse random sets
    population = []
    for i in range(pop_size):
        # Random size between 5 and 50
        size = random.randint(5, 50)
        # Random elements
        u = [0]
        # Ensure some structure: mix of small and large numbers
        for _ in range(size - 1):
            if random.random() < 0.5:
                # Small numbers
                u.append(random.randint(1, 100))
            else:
                # Larger numbers
                u.append(random.randint(100, max_span))
        u = sorted(list(set(u)))
        population.append(u)
        
    # Evaluate initial population
    scores = _ga_evaluate(population, max_span)
    
    # Update best
    max_score_idx = np.argmax(scores)
    if scores[max_score_idx] > best_score:
        best_score = scores[max_score_idx]
        best_u = population[max_score_idx][:]
        
    # Main GA Loop
    while time.time() < _DEADLINE - 5.0:
        new_population = []
        
        # Elitism: Keep the best 5
        sorted_indices = np.argsort(scores)[::-1]
        for i in range(min(5, len(population))):
            new_population.append(population[sorted_indices[i]][:])
            
        # Fill the rest with offspring
        while len(new_population) < pop_size:
            if time.time() > _DEADLINE - 2.0:
                break
                
            p1 = _ga_select(population, scores)
            p2 = _ga_select(population, scores)
            
            child = _ga_crossover(p1, p2, max_span)
            child = _ga_mutate(child, max_span)
            
            new_population.append(child)
            
        population = new_population
        scores = _ga_evaluate(population, max_span)
        
        # Update best
        max_score_idx = np.argmax(scores)
        if scores[max_score_idx] > best_score:
            best_score = scores[max_score_idx]
            best_u = population[max_score_idx][:]
            
        # Adaptive Span: If we're making progress, maybe allow larger spans?
        # Or if stagnating, try different spans?
        # For now, keep max_span fixed to keep FFT size manageable.
        # 20000 span -> FFT size ~ 65536 or 131072. 
        # 50 sets * 131072 FFT is fast.
        
        generations += 1
        
        # If we've been running for a while and not improving, maybe restart with higher span?
        # But 150s is tight. Let's stick to one good span.
        
    # Final local search polish on the best individual
    if time.time() < _DEADLINE - 2.0:
        u_final = best_u[:]
        s_final = best_score
        # Simple local search
        for _ in range(500): # Limited steps
            if time.time() > _DEADLINE - 1.0:
                break
            idx = random.randint(1, len(u_final)-1) if len(u_final) > 1 else 0
            if idx == 0:
                idx = 1
                if len(u_final) < 2:
                    u_final.append(1)
                    s_final = _fft_score(u_final)
                    if s_final > best_score:
                        best_score = s_final
                        best_u = u_final[:]
                    continue
            
            old_val = u_final[idx]
            # Try shifting
            delta = random.randint(-50, 50)
            new_val = old_val + delta
            if 1 <= new_val <= max_span and new_val not in u_final:
                u_final[idx] = new_val
                u_final.sort()
                s_test = _fft_score(u_final)
                if s_test > s_final:
                    s_final = s_test
                    if s_test > best_score:
                        best_score = s_test
                        best_u = u_final[:]
                else:
                    # Revert
                    u_final[idx] = old_val
                    u_final.sort()
                    
    # Sanity check
    best_u = sorted(set(int(x) for x in best_u))
    if 0 not in best_u:
        best_u.append(0)
    if len(best_u) < 2:
        best_u = [0, 1]
        
    return best_u

# EVOLVE-BLOCK-END
