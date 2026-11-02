"""Improved program for the third autocorrelation inequality task."""
import numpy as np
import time

N = 400
DEADLINE = 140  # safety margin before 150s kill

def _score(s):
    """Upper bound on C3; lower is better."""
    n_fft = 1
    while n_fft < 2 * N:
        n_fft *= 2
    s_f = s.astype(np.float64)
    F = np.fft.rfft(s_f, n=n_fft)
    acf = np.fft.irfft(F * np.conj(F), n=n_fft)[:N]
    acf = acf / N
    return float(np.max(np.abs(acf[1:])))

def _cellular_automaton_sequence(seed, rule, steps=20):
    """
    Generate a sign sequence using cellular automaton evolution.
    
    Args:
        seed: initial 1D binary array of length N
        rule: integer 0-255 for elementary CA rule
        steps: number of evolution steps
    
    Returns:
        Sign sequence of length N
    """
    state = seed.copy()
    # Convert rule to lookup table
    rule_bits = [(rule >> i) & 1 for i in range(8)]
    
    for _ in range(steps):
        new_state = np.empty_like(state)
        # Wrap around for toroidal boundary
        left = np.roll(state, 1)
        center = state
        right = np.roll(state, -1)
        
        # Compute pattern index (0-7)
        pattern = (left * 4 + center * 2 + right)
        new_state = np.array([rule_bits[p] for p in pattern])
        state = new_state
    
    # Map binary state to signs: 0 -> -1, 1 -> 1
    return (2 * state - 1).astype(np.int8)

def _greedy_polish(s, max_time=30.0):
    """
    Greedy coordinate descent: flip individual bits if it improves the score.
    """
    s = s.copy()
    cur_score = _score(s)
    t_start = time.time()
    
    while time.time() - t_start < max_time:
        improved = False
        # Try flipping each position
        for i in range(N):
            if time.time() - t_start > max_time:
                break
            
            s[i] *= -1
            new_score = _score(s)
            
            if new_score < cur_score:
                cur_score = new_score
                improved = True
            else:
                s[i] *= -1  # revert
        
        if not improved:
            break
    
    return s, cur_score

def construct():
    t_start = time.time()
    rng = np.random.default_rng(42)
    
    # Generate diverse initial candidates using cellular automata
    candidates = []
    
    # Try different CA rules and seeds
    ca_rules = [30, 45, 54, 72, 86, 90, 105, 110, 126, 150, 184, 225]
    num_seeds = 3
    
    for rule in ca_rules:
        for seed_idx in range(num_seeds):
            if time.time() - t_start > DEADLINE - 60:  # Leave time for polishing
                break
            
            # Generate random seed
            seed = rng.integers(0, 2, size=N, dtype=np.uint8)
            
            # Evolve CA for different numbers of steps
            for steps in [5, 10, 15, 20, 30]:
                seq = _cellular_automaton_sequence(seed, rule, steps)
                candidates.append(seq)
        
        if time.time() - t_start > DEADLINE - 60:
            break
    
    # Also add some random sequences for diversity
    for _ in range(5):
        candidates.append(rng.choice([-1, 1], size=N).astype(np.int8))
    
    # Evaluate all candidates and pick the best few
    scores = [_score(c) for c in candidates]
    
    # Sort by score and take top candidates
    sorted_indices = np.argsort(scores)
    top_k = min(10, len(candidates))
    
    best_score = scores[sorted_indices[0]]
    best_seq = candidates[sorted_indices[0]].copy()
    
    # Polish the top candidates with greedy coordinate descent
    remaining_time = DEADLINE - (time.time() - t_start)
    time_per_candidate = max(1.0, remaining_time / top_k)
    
    for idx in sorted_indices[:top_k]:
        if time.time() - t_start > DEADLINE - 5:  # Leave safety margin
            break
        
        seq = candidates[idx].copy()
        polished_seq, polished_score = _greedy_polish(seq, max_time=time_per_candidate)
        
        if polished_score < best_score:
            best_score = polished_score
            best_seq = polished_seq.copy()
    
    return best_seq
