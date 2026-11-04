"""
Cellular Automaton Trajectory Strategy for Signed Autocorrelation Minimization.

Strategy:
1. Represent the signal x (length N=400) as the time-evolution of a 1D Binary Cellular Automaton (CA).
2. The CA is defined by an initial state (length L) and a rule (neighborhood size k, rule index).
3. We run a "Rule Search" over a diverse set of CA rules and initial states to find a trajectory 
   that, when truncated to length N and mapped to signed values {-1, 1}, minimizes the metric:
   max|autoconv(x)[1:]| / (sum(x))^2.
4. To handle the "signed step heights" interface and the specific metric, we map binary 0/1 to -1/+1.
5. We use a fast evaluation loop. Since N=400 is small, FFT-based autoconvolution is fast.
6. We do not use gradient descent on the signal itself, but rather search the space of (Rule, Initial State) pairs.
7. We include a "Polish" phase where we allow local flips of bits in the generated trajectory if it reduces the metric, 
   acting as a local repair mechanism (addressing the 'error_repair' query contextually by testing if local repair helps).
"""

import numpy as np
import time

N = 400

def _autoconv(x):
    """Compute full linear autoconvolution of x (length N) -> length 2N-1."""
    n = 2 * N
    X = np.fft.fft(x, n=n)
    C = np.fft.ifft(X * X).real[:2*N-1]
    return C

def _eval_ratio(x):
    """Compute the raw value: max|c_k| / (sum x)^2 for k != 0."""
    c = _autoconv(x)
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    if len(c) > 1:
        max_sidelobe = np.max(np.abs(c[1:]))
    else:
        max_sidelobe = 0.0
    return max_sidelobe / (s * s)

def _run_ca(rule, init_state, length, k=3):
    """
    Run a 1D binary CA for 'length' steps.
    rule: integer index for the rule (0 to 2^(2k+1)-1).
    init_state: array of 0s and 1s of length L.
    k: neighborhood radius (total neighborhood size = 2k+1).
    Returns a binary array of length 'length' representing the center cell of each step.
    """
    L = len(init_state)
    trajectory = np.zeros(length, dtype=np.int8)
    current = init_state.copy()
    
    # Precompute rule lookup for speed if possible, but simple loop is fine for N=400
    # Rule lookup: index = sum(bit_i * 2^i) for neighborhood bits
    # For k=3, neighborhood size 7, 128 rules.
    
    for t in range(length):
        # Extract center cell (index L//2)
        trajectory[t] = current[L//2]
        
        # Compute next state
        # We need to handle boundaries. We'll use periodic boundary conditions.
        next_state = np.zeros(L, dtype=np.int8)
        
        # Vectorized neighborhood extraction is tricky with boundaries in pure numpy without padding.
        # For L=100, a python loop is ~100 ops. 400 steps * 100 ops = 40k ops. Very fast.
        for i in range(L):
            # Get neighborhood: i-k ... i+k (mod L)
            idxs = [(i + j) % L for j in range(-k, k+1)]
            # Calculate index
            idx = 0
            for j, id_idx in enumerate(idxs):
                # Standard CA convention: leftmost is MSB? Or LSB?
                # Let's stick to a consistent bit order.
                # Usually, the cell value is the MSB or LSB depending on implementation.
                # Let's use: bit 0 corresponds to leftmost (i-k), bit 2k to rightmost (i+k).
                if current[id_idx]:
                    idx |= (1 << j)
            
            next_state[i] = (rule >> idx) & 1
        
        current = next_state
        
    return trajectory

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # We will search for a good CA configuration.
    # Parameters:
    # - L: Initial state length. Must be large enough to support neighborhood k.
    # - k: Neighborhood radius. k=1 (3-neighbor) or k=2 (5-neighbor) or k=3 (7-neighbor).
    # - Rule: Integer.
    # - Init State: Random binary vector.
    
    # To ensure we find something good, we'll test a variety of known "chaotic" or "complex" rules.
    # Wolfram Code 30, 45, 110, 184, etc. are famous.
    # For k=1 (3-neighbor), rules are 0-7.
    # For k=2 (5-neighbor), rules are 0-31.
    # For k=3 (7-neighbor), rules are 0-127.
    
    # We'll focus on k=1 and k=2 for speed, as k=3 rules space is larger and L must be larger.
    # Actually, for N=400, L needs to be at least 2*k+1. But to avoid boundary effects dominating, 
    # L should be somewhat larger, e.g., L=100 or L=200.
    
    # Metric optimization:
    # We map binary trajectory b (0/1) to signed x (-1/1).
    # x = 2*b - 1.
    
    best_val = np.inf
    best_x = np.ones(N, dtype=np.float64) # Fallback
    
    # Candidate rules to test (Wolfram codes for k=1 and k=2)
    # k=1 rules: 0-7. Known complex: 30 (0b011110 -> wait, Wolfram uses specific bit ordering).
    # Let's just iterate over all rules for small k.
    
    # Time budget allocation:
    # 1. Random search over (k, rule, init_state)
    # 2. Local repair (flipping bits) on the best candidate found.
    
    rng = np.random.default_rng(42)
    
    # Search Parameters
    L = 100  # State length
    max_k = 2
    
    # Pre-generate some initial states
    init_states = [rng.integers(0, 2, size=L) for _ in range(10)]
    
    # List of (k, rule) pairs to test
    # For k=1, 8 rules. For k=2, 32 rules. Total 40 rules.
    # For each rule, we test a few initial states.
    
    candidates = []
    for k in [1, 2]:
        num_rules = 2 ** (2 * k + 1)
        for rule in range(num_rules):
            # Test with a few random initial states
            for init_state in init_states[:3]:
                candidates.append((k, rule, init_state))
    
    # Shuffle candidates to mix up the search
    rng.shuffle(candidates)
    
    # Evaluate candidates
    for k, rule, init_state in candidates:
        if time.time() > deadline - 10:
            break
            
        # Generate trajectory
        try:
            traj = _run_ca(rule, init_state, N, k=k)
        except Exception:
            continue
            
        # Map to signed
        x = 2.0 * traj.astype(np.float64) - 1.0
        
        # Evaluate
        val = _eval_ratio(x)
        
        if val < best_val:
            best_val = val
            best_x = x.copy()
            
    # Local Repair / Polish Phase
    # If we have time, try to improve the best_x by flipping individual bits.
    # This is a greedy local search.
    if time.time() < deadline - 5:
        x_curr = best_x.copy()
        val_curr = best_val
        
        # Number of polish iterations
        max_polish_iters = 500
        improved = True
        
        while improved and time.time() < deadline - 5:
            improved = False
            # Randomly select indices to try flipping
            # Instead of trying all N, try a subset per iteration to save time?
            # Or just one random flip per loop?
            # N=400 is small. Trying all N flips is 400 evaluations.
            # Each eval is FFT(800) ~ fast.
            # Let's try all flips in one pass if time permits.
            
            best_flip_idx = -1
            best_flip_val = val_curr
            
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                
                # Flip bit i
                x_test = x_curr.copy()
                x_test[i] = -x_test[i]
                
                val_test = _eval_ratio(x_test)
                
                if val_test < best_flip_val:
                    best_flip_val = val_test
                    best_flip_idx = i
            
            if best_flip_idx != -1:
                x_curr[best_flip_idx] = -x_curr[best_flip_idx]
                val_curr = best_flip_val
                improved = True
                if val_curr < best_val:
                    best_val = val_curr
                    best_x = x_curr.copy()
        
        # If we improved, update best
        if val_curr < best_val:
            best_val = val_curr
            best_x = x_curr.copy()

    return best_x
