import numpy as np
import time

N = 400

def _calc_score(h):
    """
    Calculate the score (upper bound on C3) for a given step function h.
    Score = max_{k != 0} |(h*h)_k| / ||h||_2^2
    For binary +/- 1, ||h||_2^2 = N.
    """
    corr = np.correlate(h, h, mode='full')
    mask = np.ones(len(corr), dtype=bool)
    mask[N-1] = False
    peak = np.max(np.abs(corr[mask]))
    norm = np.sum(h**2)
    if norm == 0:
        return 0.0
    return peak / norm

def _van_der_corput(n, base=2):
    """Generate van der Corput sequence (low-discrepancy)."""
    seq = []
    for i in range(1, n + 1):
        f = 0.0
        d = 1.0
        while i > 0:
            i, rem = divmod(i, base)
            d /= base
            f += rem * d
        seq.append(f)
    return np.array(seq)

def _logistic_map(n, r, x0):
    """Generate logistic map trajectory."""
    x = np.zeros(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = r * x[i-1] * (1.0 - x[i-1])
    return x

def _generate_deterministic_sequences(n_sequences):
    """Generate diverse deterministic sequences with low autocorrelation potential."""
    sequences = []
    
    # Van der Corput based sequences
    vdc = _van_der_corput(N)
    # Threshold at median to create +/-1
    med = np.median(vdc)
    seq1 = np.where(vdc > med, 1, -1).astype(float)
    sequences.append(seq1)
    
    # Logistic map with different parameters
    for r in [3.5, 3.7, 3.9, 3.99]:
        for x0 in [0.1, 0.3, 0.5, 0.7]:
            lm = _logistic_map(N, r, x0)
            med = np.median(lm)
            seq = np.where(lm > med, 1, -1).astype(float)
            sequences.append(seq)
    
    # Chaotic tent map
    for x0 in [0.1, 0.3, 0.5, 0.7]:
        x = np.zeros(N)
        x[0] = x0
        for i in range(1, N):
            x[i] = 2 * x[i-1] if x[i-1] < 0.5 else 2 * (1 - x[i-1])
        med = np.median(x)
        seq = np.where(x > med, 1, -1).astype(float)
        sequences.append(seq)
    
    return sequences[:n_sequences]

def _simulated_annealing(h, start_temp, end_temp, iterations, deadline):
    """Apply simulated annealing to escape local minima."""
    h = h.copy()
    current_score = _calc_score(h)
    best_h = h.copy()
    best_score = current_score
    
    for i in range(iterations):
        if time.time() > deadline:
            break
        
        # Cool temperature
        t = start_temp * (end_temp / start_temp) ** (i / max(iterations - 1, 1))
        
        # Pick random position to flip
        j = np.random.randint(N)
        old_val = h[j]
        h[j] = -h[j]
        
        new_score = _calc_score(h)
        delta = new_score - current_score
        
        # Accept if better or with probability exp(-delta/t)
        if delta < 0 or np.random.random() < np.exp(-delta / max(t, 1e-10)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
        else:
            h[j] = old_val
    
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_h = None
    best_score = float('inf')
    
    # Phase 1: Evaluate diverse deterministic sequences
    deterministic_seqs = _generate_deterministic_sequences(20)
    for seq in deterministic_seqs:
        if time.time() > deadline - 30:
            break
        score = _calc_score(seq)
        if score < best_score:
            best_score = score
            best_h = seq.copy()
    
    # Phase 2: Add some random sequences for diversity
    for _ in range(10):
        if time.time() > deadline - 30:
            break
        h = np.random.choice([-1, 1], size=N).astype(float)
        score = _calc_score(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Phase 3: Simulated annealing from best candidate
    h = best_h.copy()
    start_temp = best_score * 0.1
    end_temp = best_score * 0.001
    iterations = 5000
    best_h, best_score = _simulated_annealing(h, start_temp, end_temp, iterations, deadline - 10)
    
    # Phase 4: Hill climbing refinement
    h = best_h.copy()
    current_score = best_score
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for j in range(N):
            if time.time() > deadline - 5:
                break
            old_val = h[j]
            h[j] = -h[j]
            new_score = _calc_score(h)
            if new_score < current_score:
                current_score = new_score
                improved = True
                if new_score < best_score:
                    best_score = new_score
                    best_h = h.copy()
            else:
                h[j] = old_val
    
    return best_h
