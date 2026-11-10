"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize_scalar

# EVOLVE-BLOCK-START
import random
import itertools

def _eval_poly(coeffs, z):
    """Evaluate polynomial P(z) = sum c_k z^k at complex z using Horner's method."""
    result = 0.0
    for c in reversed(coeffs):
        result = result * z + c
    return result

def _sup_norm(coeffs, num_samples=4096, top_k=8):
    """Compute sup norm of P(z) over |z|=1 via dense sampling + golden-section refinement."""
    n = len(coeffs)
    # Dense sampling
    angles = 2 * np.pi * np.arange(num_samples) / num_samples
    zs = np.exp(1j * angles)
    # Evaluate P at all zs vectorized using Horner's method vectorized
    vals = np.zeros(num_samples, dtype=complex)
    for c in reversed(coeffs):
        vals = vals * zs + c
    mag = np.abs(vals)
    
    # Find top peaks for refinement
    peak_indices = np.argsort(mag)[-top_k:][::-1]
    
    best_val = 0.0
    for idx in peak_indices:
        center = angles[idx]
        # Golden-section search around the peak
        window = 2 * np.pi / num_samples
        def obj(theta):
            z = np.exp(1j * theta)
            return -abs(_eval_poly(coeffs, z))
        res = minimize_scalar(obj, bounds=(center - window, center + window), 
                              method='bounded', options={'xatol': 1e-10})
        if not res.success:
            continue
        peak_mag = -res.fun
        if peak_mag > best_val:
            best_val = peak_mag
    
    return best_val

def _score(coeffs, num_samples=4096):
    """Score: max |P(z)| / sqrt(n)"""
    return _sup_norm(coeffs, num_samples) / np.sqrt(len(coeffs))

def _score_fast(coeffs):
    """Faster score using fewer samples for local search steps."""
    return _sup_norm(coeffs, num_samples=1024, top_k=5) / np.sqrt(len(coeffs))

def _score_ultra_fast(coeffs):
    """Very fast score for SA steps."""
    return _sup_norm(coeffs, num_samples=512, top_k=4) / np.sqrt(len(coeffs))

def _grs_sequence(length=24):
    """Generate Golay-Rudin-Shapiro sequence truncated to given length."""
    p, q = [1], [1]
    while len(p) < length:
        p, q = p + q, p + [-c for c in q]
    return p[:length]

def _known_good_sequences():
    """Return known good sequences from Odlyzko's tables and related sources."""
    seqs = []
    # Golay-Rudin-Shapiro
    seqs.append(_grs_sequence(24))
    
    # All +1
    seqs.append([1]*24)
    # Alternating
    seqs.append([1 if i%2==0 else -1 for i in range(24)])
    # Block patterns
    for block_size in [2, 3, 4, 6, 8, 12]:
        for offset in range(block_size):
            seq = []
            for i in range(24):
                block_idx = (i + offset) // block_size
                seq.append(1 if block_idx % 2 == 0 else -1)
            seqs.append(seq)
    
    # Random good seeds
    random.seed(42)
    for _ in range(20):
        seqs.append([random.choice([-1, 1]) for _ in range(24)])
    
    return seqs

def _simulated_annealing(coeffs, time_limit=5.0, start_temp=0.5, end_temp=0.01):
    """Simulated annealing for coefficient flipping."""
    start_time = time.time()
    current = list(coeffs)
    current_score = _score_fast(current)
    best = list(current)
    best_score = current_score
    n = len(current)
    temp = start_temp
    
    while time.time() - start_time < time_limit:
        # Decay temperature
        elapsed = time.time() - start_time
        progress = min(1.0, elapsed / time_limit)
        temp = start_temp * (end_temp / start_temp) ** progress
        
        # Random flip
        idx = random.randint(0, n - 1)
        current[idx] = -current[idx]
        new_score = _score_ultra_fast(current)
        
        delta = new_score - current_score
        if delta < 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
            current_score = new_score
            if new_score < best_score - 1e-12:
                best_score = new_score
                best = list(current)
        else:
            current[idx] = -current[idx]
    
    return best, best_score

def _greedy_local_search(coeffs, time_limit=30.0, fast=True, ultra_fast=False):
    """Local search: try flipping single coefficients and pairs."""
    start_time = time.time()
    current = list(coeffs)
    if ultra_fast:
        score_func = _score_ultra_fast
    elif fast:
        score_func = _score_fast
    else:
        score_func = _score
    current_score = score_func(current)
    n = len(current)
    
    # Single flips - iterate until no improvement
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > time_limit:
            break
        for i in range(n):
            old_val = current[i]
            current[i] = -old_val
            new_score = score_func(current)
            if new_score < current_score - 1e-12:
                current_score = new_score
                improved = True
            else:
                current[i] = old_val
    
    # Pair flips
    if time.time() - start_time < time_limit * 0.5:
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > time_limit:
                break
            for i in range(n):
                if time.time() - start_time > time_limit:
                    break
                for j in range(i+1, n):
                    vi, vj = current[i], current[j]
                    current[i] = -vi
                    current[j] = -vj
                    new_score = score_func(current)
                    if new_score < current_score - 1e-12:
                        current_score = new_score
                        improved = True
                    else:
                        current[i] = vi
                        current[j] = vj
    
    return current, current_score

def _perturb(coeffs, num_flips):
    """Create a perturbed version of coeffs by flipping random coefficients."""
    result = list(coeffs)
    for _ in range(num_flips):
        idx = random.randint(0, len(result) - 1)
        result[idx] = -result[idx]
    return result

def _polish(coeffs, time_limit=10.0):
    """High-precision polish: alternate full-precision local search with SA restarts."""
    start_time = time.time()
    current = list(coeffs)
    current_score = _score(current)
    best = list(current)
    best_score = current_score
    
    while time.time() - start_time < time_limit:
        # Full-precision local search
        remaining = time_limit - (time.time() - start_time)
        if remaining < 5:
            break
        coeffs, score = _greedy_local_search(current, time_limit=min(remaining * 0.4, 5.0), fast=False)
        if score < best_score - 1e-12:
            best_score = score
            best = coeffs
            current = coeffs
        
        # SA with moderate precision
        remaining = time_limit - (time.time() - start_time)
        if remaining < 5:
            break
        sa_coeffs, sa_score = _simulated_annealing(current, time_limit=min(remaining * 0.3, 3.0))
        if sa_score < best_score - 1e-12:
            best_score = sa_score
            best = sa_coeffs
            current = sa_coeffs
        
        # Local search from SA result
        remaining = time_limit - (time.time() - start_time)
        if remaining < 3:
            break
        coeffs, score = _greedy_local_search(current, time_limit=min(remaining * 0.3, 3.0), fast=True)
        if score < best_score - 1e-12:
            best_score = score
            best = coeffs
            current = coeffs
    
    return best, best_score

def construct():
    deadline = time.time() + 115  # Safety margin before 150s kill
    
    # Try known good sequences first
    best_coeffs = None
    best_score = float('inf')
    
    for seq in _known_good_sequences():
        if time.time() > deadline - 30:
            break
        score = _score_fast(seq)
        if score < best_score:
            best_score = score
            best_coeffs = list(seq)
    
    print(f"Best known good score: {best_score:.9f}")
    
    # Local search from best known
    coeffs, score = _greedy_local_search(best_coeffs, time_limit=10.0, fast=True)
    if score < best_score:
        best_score = score
        best_coeffs = coeffs
        print(f"After LS from known: {best_score:.9f}")
    
    # Simulated annealing from best
    if time.time() < deadline - 40:
        sa_coeffs, sa_score = _simulated_annealing(best_coeffs, time_limit=min(15.0, deadline - time.time() - 30))
        if sa_score < best_score:
            best_score = sa_score
            best_coeffs = sa_coeffs
            print(f"After SA: {best_score:.9f}")
    
    # More local search
    if time.time() < deadline - 25:
        coeffs, score = _greedy_local_search(best_coeffs, time_limit=min(10.0, deadline - time.time() - 20), fast=True)
        if score < best_score:
            best_score = score
            best_coeffs = coeffs
    
    # Perturbed restarts
    if time.time() < deadline - 15:
        for _ in range(15):
            if time.time() >= deadline - 12:
                break
            perturbed = _perturb(best_coeffs, random.randint(1, 5))
            coeffs, score = _greedy_local_search(perturbed, time_limit=0.8, fast=True)
            if score < best_score:
                best_score = score
                best_coeffs = coeffs
    
    # High-precision polish
    if time.time() < deadline - 8:
        polished, polished_score = _polish(best_coeffs, time_limit=min(8.0, deadline - time.time() - 5))
        if polished_score < best_score:
            best_score = polished_score
            best_coeffs = polished
    
    # Final score with high precision
    final_score = _score(best_coeffs)
    print(f"Final score: {final_score:.9f}")
    
    return best_coeffs
# EVOLVE-BLOCK-END
