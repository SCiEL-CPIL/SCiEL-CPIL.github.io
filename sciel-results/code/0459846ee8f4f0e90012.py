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
    
    # Beam Search
    # State: (current_prime_index, last_gap, sequence_list)
    # We initialize the beam with all possible starting primes
    # But starting with 2 is likely best. Let's try a few starts.
    
    # To be safe and fast, we'll run a beam search from the first few primes
    # and also a pure greedy from the first prime as a baseline.
    
    def run_beam_search(start_indices, beam_width=20):
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
                # To keep beam small, we only look at the first few valid primes?
                # No, we should look at a range.
                # But if we branch too much, it explodes.
                # Let's limit the number of candidates per state.
                
                candidates = []
                for j in range(idx + 1, num_primes):
                    new_gap = primes[j] - primes[idx]
                    if new_gap >= gap:
                        candidates.append((j, new_gap))
                        # Limit candidates to keep beam manageable
                        if len(candidates) >= beam_width:
                            break
                            
                for j, new_gap in candidates:
                    new_seq = seq + [primes[j]]
                    if len(new_seq) > best_length:
                        best_length = len(new_seq)
                        best_sequence = new_seq
                        
                    next_beam.append((j, new_gap, new_seq))
                    
            # Prune next_beam to keep only the best 'beam_width' states
            # How to rank? Smaller gap is generally better. Smaller prime is better.
            # Let's sort by (gap, prime) and keep the first beam_width
            next_beam.sort(key=lambda x: (x[1], x[0]))
            beam = next_beam[:beam_width]
            
    # Try starting from the first 5 primes
    run_beam_search(range(5), beam_width=50)
    
    # If we have time, try a pure greedy as a fallback (though beam should cover it)
    # The seed program is essentially a greedy with beam_width=1 from start=0.
    # Our beam search with width 50 from starts 0-4 should find better solutions if they exist.
    
    return best_sequence
