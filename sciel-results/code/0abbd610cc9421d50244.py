import time
from math import isqrt

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Sieve of Eratosthenes
    N = 10000
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i*i, N+1, i):
                is_prime[j] = False
                
    primes = [i for i, v in enumerate(is_prime) if v]
    num_primes = len(primes)
    
    best_sequence = []
    best_length = 0
    
    def run_beam_search(start_indices, beam_width=100):
        nonlocal best_sequence, best_length
        
        # Initial beam: states starting at each of the start_indices
        # State: (idx, gap, seq)
        # For the first step, gap is 0 (or -1 to allow any positive gap)
        beam = []
        for idx in start_indices:
            beam.append((idx, 0, [primes[idx]]))
            
        while beam:
            if time.time() > deadline:
                return
            
            next_beam = []
            for idx, gap, seq in beam:
                # Expand this state
                # Find valid next primes
                # We need to look at ALL valid primes, not just the first few
                # But we need to be smart about which ones to keep
                
                candidates = []
                for j in range(idx + 1, num_primes):
                    new_gap = primes[j] - primes[idx]
                    if new_gap >= gap:
                        candidates.append((j, new_gap))
                        # We can't break early because a larger gap might lead to a longer sequence
                        # But we need to limit memory - let's collect all and then prune
                            
                for j, new_gap in candidates:
                    new_seq = seq + [primes[j]]
                    if len(new_seq) > best_length:
                        best_length = len(new_seq)
                        best_sequence = new_seq
                        
                    next_beam.append((j, new_gap, new_seq))
                    
            # Prune next_beam to keep only the best 'beam_width' states
            # Ranking strategy: prefer states with smaller gaps and smaller primes
            # This allows more room for future extensions
            next_beam.sort(key=lambda x: (x[1], x[0]))
            beam = next_beam[:beam_width]
            
    # Try starting from the first 10 primes with a wider beam
    run_beam_search(range(10), beam_width=200)
    
    # If we have time, try even wider beam from fewer starts
    if time.time() < deadline - 10:
        run_beam_search(range(5), beam_width=500)
    
    # If we still have time, try greedy from each start as a baseline
    if time.time() < deadline - 5:
        for start_idx in range(min(20, num_primes)):
            if time.time() > deadline - 2:
                break
            seq = [primes[start_idx]]
            last_gap = 0
            last_idx = start_idx
            while True:
                if time.time() > deadline - 1:
                    break
                found = False
                for j in range(last_idx + 1, num_primes):
                    new_gap = primes[j] - primes[last_idx]
                    if new_gap >= last_gap:
                        seq.append(primes[j])
                        last_gap = new_gap
                        last_idx = j
                        if len(seq) > best_length:
                            best_length = len(seq)
                            best_sequence = seq[:]
                        found = True
                        break
                if not found:
                    break
    
    return best_sequence
