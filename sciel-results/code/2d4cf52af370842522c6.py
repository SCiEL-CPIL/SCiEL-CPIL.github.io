"""
Strategy: Binary Cellular Automaton (CA) Generation with Rule Optimization.

This strategy generates candidate sequences not by random bit flips or continuous
optimization projected to signs, but by evolving a 1D binary cellular automaton
(CA). The CA provides a structured, deterministic source of pseudo-randomness with
inherent low-correlation properties (depending on the rule).

1. We define a parameterized family of CA rules (generalized linear CA over GF(2)
   with variable neighborhood sizes and biases).
2. We optimize the CA parameters (rule taps, initial state seed, evolution steps)
   to maximize the "quality" of the resulting sequence, defined by the ratio
   of the squared integral to the peak absolute autoconvolution sidelobe.
3. This avoids the local minima traps of direct bit-flip search by exploring
   distinct, high-entropy basins of attraction defined by the CA dynamics.
4. A final greedy polish is applied to the best CA-generated candidate.
"""

import numpy as np
import time
import scipy.optimize

N = 400

# EVOLVE-BLOCK-START

def _autoconv_full(x):
    """Full linear autoconvolution of x (length 2N-1)."""
    n = len(x)
    size = 1 << (2 * n - 1).bit_length()
    F = np.fft.rfft(x, n=size)
    conv = np.fft.irfft(F * F, n=size)
    return conv[:2 * n - 1]

def _score_offdiag(x):
    """
    Compute the metric: max(|autoconv|) for k != 0 / (sum(x))^2
    We want to minimize this.
    """
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return 1e12
    conv = _autoconv_full(x)
    if len(conv) <= 1:
        return 0.0
    max_abs_offdiag = np.max(np.abs(conv[1:]))
    return max_abs_offdiag / (integral * integral)

def _generate_ca_sequence(rule_params, seed, n_steps, length):
    """
    Generate a sequence of length 'length' from a binary CA.
    rule_params: list of tap indices (e.g., [0, 1, 2]) for XOR-based rule.
    seed: initial state (list of 0s and 1s).
    n_steps: number of evolution steps.
    
    We use a standard 1D CA where the new state of cell i is XOR of neighbors at taps.
    We output the state of the center cell (or a specific cell) at each step.
    """
    # Ensure seed is long enough to cover the neighborhood
    # Let's use a circular buffer of size L = max(tap) + 1
    max_tap = max(rule_params) if rule_params else 0
    L = max_tap + 1
    
    # Initialize state
    state = np.zeros(L, dtype=np.uint8)
    # Map seed to initial state
    seed_arr = np.array(seed, dtype=np.uint8)
    if len(seed_arr) < L:
        # Pad with zeros
        state = np.zeros(L, dtype=np.uint8)
        state[:len(seed_arr)] = seed_arr
    else:
        state = seed_arr[:L].copy()
    
    # Center index is 0
    center = 0
    sequence = np.zeros(n_steps, dtype=np.float64)
    
    for t in range(n_steps):
        # Compute new state for center
        val = 0
        for tap in rule_params:
            # Circular index
            idx = (center + tap) % L
            val ^= state[idx]
        
        # Store output (map 0->-1, 1->1)
        sequence[t] = 1.0 if val == 1 else -1.0
        
        # Shift state (left shift, new value enters at end)
        # Or we can just update the whole state? 
        # For efficiency, let's just shift.
        # Actually, standard CA updates all cells simultaneously.
        # But we only care about the center. 
        # If we shift left, the "center" moves relative to the state?
        # Let's implement a proper sliding window update.
        
        # New state at position i depends on old state at i+taps.
        # If we want the sequence of the center cell over time, we can just
        # keep a buffer and update it.
        
        # Let's use a simpler approach: keep a full state array of size L.
        # Update all positions simultaneously.
        new_state = np.zeros(L, dtype=np.uint8)
        for i in range(L):
            v = 0
            for tap in rule_params:
                idx = (i + tap) % L
                v ^= state[idx]
            new_state[i] = v
        
        state = new_state

    return sequence[:length]

def _ca_objective(params):
    """
    Objective function for CA parameter optimization.
    params: 
    - First K integers: rule taps (encoded as integers)
    - Next 1: seed offset
    - Next 1: number of steps (scaled)
    
    We minimize the negative log of the quality metric (or just the metric).
    To make it differentiable for scipy, we treat taps as continuous and round.
    However, CA rules are discrete. We will use a random search over the discrete
    space of rules, or a hybrid.
    
    Given the complexity of differentiating a CA, we will use a grid/random search
    over rule parameters and initial seeds, and then polish the best one.
    """
    pass # Not used directly in this implementation, we use explicit search.

def _evaluate_ca_candidate(rule_params, seed, n_steps, length=N):
    """
    Evaluate a specific CA configuration.
    Returns the sequence and its score.
    """
    seq = _generate_ca_sequence(rule_params, seed, n_steps, length)
    # Ensure length is exactly N
    if len(seq) < N:
        # Pad with zeros or repeat? 
        # If n_steps < N, we can't fill. 
        # We should ensure n_steps >= N.
        pass
    seq = seq[:N]
    
    # Check if valid (non-zero sum)
    s = np.sum(seq)
    if abs(s) < 1e-6:
        return seq, 1e12
        
    score = _score_offdiag(seq)
    return seq, score

def _greedy_polish(x, deadline):
    """
    Greedy local search: flip bit i if it reduces the exact score.
    """
    n = len(x)
    current = x.copy()
    current_score = _score_offdiag(current)
    best_score = current_score
    best_x = current.copy()
    
    size = 1 << (2 * n - 1).bit_length()
    
    no_improve = 0
    max_no_improve = 2000
    indices = np.arange(n)
    
    while time.time() < deadline and no_improve < max_no_improve:
        if no_improve > 100:
            np.random.shuffle(indices)
            
        improved = False
        for i in indices:
            if time.time() > deadline - 1.0:
                break
                
            old_val = current[i]
            current[i] = -old_val
            
            s = np.sum(current)
            if abs(s) < 1e-12:
                new_score = np.inf
            else:
                H = np.fft.rfft(current, size)
                ac = np.fft.irfft(H * H, size)[:2 * n - 1]
                max_abs = np.max(np.abs(ac[1:]))
                new_score = max_abs / (s * s)
                
            if new_score < current_score:
                current_score = new_score
                no_improve = 0
                improved = True
                if new_score < best_score:
                    best_score = new_score
                    best_x = current.copy()
            else:
                current[i] = old_val
                no_improve += 1
                
        if not improved:
            if no_improve > 1000:
                break

    return best_x, best_score

def construct():
    t0 = time.time()
    deadline = t0 + 145.0
    
    rng = np.random.default_rng(42)
    
    best_seq = None
    best_score = 1e12
    
    # Define a set of promising CA rules.
    # Rule 30, 90, 110, 150, etc. are known for complexity.
    # We define rules by their taps in a 3-cell neighborhood (0, 1, 2).
    # Generalized: taps can be any subset of {0, 1, 2, ..., M}
    
    # Let's try rules with taps in range [0, 4]
    # We will sample random rules and random seeds.
    
    # Pre-define some known good-looking rules (binary representation of CA rule number)
    # Rule 30: 11110000 -> Taps: 7,6,5,4 (if 8 cells) -> This is for 3-neighbor.
    # Let's just use XOR of neighbors at specific offsets.
    
    candidate_rules = []
    # Add some structured rules
    candidate_rules.append([1, 2]) # Rule 90-like (XOR of left and right)
    candidate_rules.append([0, 1, 2]) # Rule 150-like? (XOR of all 3)
    candidate_rules.append([1, 3])
    candidate_rules.append([2, 4])
    candidate_rules.append([0, 2, 4])
    
    # Generate random rules
    for _ in range(50):
        num_taps = rng.integers(2, 5)
        taps = rng.choice(np.arange(0, 6), size=num_taps, replace=False)
        candidate_rules.append(list(taps))
    
    # For each rule, try a few seeds
    # We need to generate N=400 steps.
    
    for rule in candidate_rules:
        if time.time() > deadline - 10.0:
            break
            
        for seed_idx in range(20):
            if time.time() > deadline - 10.0:
                break
                
            # Random seed of length 10
            seed = rng.integers(0, 2, size=10)
            
            # Generate sequence
            # We need at least N steps. Let's do N + 50 to allow for trimming?
            # Or just N.
            seq, score = _evaluate_ca_candidate(rule, seed, N, N)
            
            if score < best_score:
                best_score = score
                best_seq = seq.copy()
                
                # Early exit if we find a very good one?
                # Threshold for "very good" is subjective.
                # Let's just keep searching.
                
    # If we didn't find a good CA sequence, fall back to random
    if best_seq is None:
        best_seq = rng.choice([-1.0, 1.0], size=N)
        best_score = _score_offdiag(best_seq)
        
    # Polish the best CA candidate
    if time.time() < deadline - 5.0:
        best_seq, best_score = _greedy_polish(best_seq, deadline - 5.0)
        
    # Try a few other diverse candidates if time permits
    # 1. Random
    rand_seq = rng.choice([-1.0, 1.0], size=N)
    if time.time() < deadline - 10.0:
        rand_polished, rand_score = _greedy_polish(rand_seq, deadline - 10.0)
        if rand_score < best_score:
            best_score = rand_score
            best_seq = rand_polished
            
    # 2. M-sequence (if time permits)
    def _m_sequence(n, k):
        polys = {
            6: [5, 3, 0],
            7: [6, 5, 3, 0],
            8: [7, 5, 3, 2, 0],
        }
        if k not in polys:
            return np.sign(np.sin(np.arange(n) * 2 * np.pi / n * 3))[:n]
        
        taps = polys[k]
        state = [0] * k
        state[0] = 1
        seq = []
        for _ in range(n):
            fb = 0
            for t in taps:
                fb ^= state[t]
            for i in range(k - 1, 0, -1):
                state[i] = state[i - 1]
            state[0] = fb
            seq.append(1.0 if state[0] else -1.0)
        return np.array(seq, dtype=np.float64)
        
    if time.time() < deadline - 10.0:
        m_seq = _m_sequence(N, 8)
        m_polished, m_score = _greedy_polish(m_seq, deadline - 10.0)
        if m_score < best_score:
            best_score = m_score
            best_seq = m_polished

    # Final polish on the absolute best
    if time.time() < deadline - 1.0:
        best_seq, best_score = _greedy_polish(best_seq, deadline - 1.0)
        
    return best_seq

# EVOLVE-BLOCK-END
