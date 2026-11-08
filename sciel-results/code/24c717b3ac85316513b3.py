import numpy as np
import time
import random

N = 100

def compute_overlap_fast(h):
    """Compute overlap at all shifts using FFT. h is array of length N."""
    n = len(h)
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    overlaps = n / 2.0 - autocorr
    return overlaps

def score(h):
    """Return the max overlap."""
    return np.max(compute_overlap_fast(h))

def generate_m_sequence(n_bits, primitive_poly):
    """Generate an m-sequence of length 2^n_bits - 1."""
    seq = [1] * n_bits
    for _ in range(2**n_bits - 1 - n_bits):
        feedback = 0
        for i, coeff in enumerate(primitive_poly):
            if coeff:
                feedback ^= seq[i]
        seq.append(feedback)
        seq = seq[1:]
    return [1] * n_bits + seq[n_bits:]

def construct():
    start = time.time()
    deadline = 140.0
    
    h = np.full(N, 0.5)
    best_h = h.copy()
    best_score = score(h)
    
    random.seed(42)
    np.random.seed(42)
    
    def try_binary_pattern(pattern):
        h_bin = np.array(pattern, dtype=float)
        if len(h_bin) != N or abs(np.sum(h_bin) - N/2) > 1e-10:
            return None
        s = score(h_bin)
        return s
    
    # Try alternating pattern
    alt_pattern = [i % 2 for i in range(N)]
    s_alt = try_binary_pattern(alt_pattern)
    if s_alt is not None and s_alt < best_score:
        best_score = s_alt
        best_h = np.array(alt_pattern, dtype=float)
    
    # Try m-sequences truncated/padded to length N
    # m-sequence of length 127 (2^7-1), take first 100
    # We need exactly 50 ones, so we may need to adjust
    
    # Try several structured patterns
    # Pattern based on quadratic residues
    # For prime p, the sequence based on quadratic residues mod p has good autocorrelation
    # Use p = 101 (prime), take first 100 terms
    def quadratic_residue_sequence(p, length):
        seq = []
        for i in range(length):
            x = i % p
            if x == 0:
                seq.append(0)
            else:
                # Check if x is a quadratic residue mod p
                is_residue = False
                for y in range(1, p):
                    if (y * y) % p == x:
                        is_residue = True
                        break
                seq.append(1 if is_residue else 0)
        return seq
    
    # Try quadratic residue sequence mod 101
    qr_seq = quadratic_residue_sequence(101, N)
    # Count ones and adjust if needed
    ones_count = sum(qr_seq)
    if ones_count != N // 2:
        # Adjust by flipping some bits
        indices = list(range(N))
        random.shuffle(indices)
        if ones_count < N // 2:
            for idx in indices:
                if qr_seq[idx] == 0:
                    qr_seq[idx] = 1
                    ones_count += 1
                    if ones_count == N // 2:
                        break
        else:
            for idx in indices:
                if qr_seq[idx] == 1:
                    qr_seq[idx] = 0
                    ones_count -= 1
                    if ones_count == N // 2:
                        break
    
    s_qr = try_binary_pattern(qr_seq)
    if s_qr is not None and s_qr < best_score:
        best_score = s_qr
        best_h = np.array(qr_seq, dtype=float)
    
    # Try m-sequence based approach
    # Use m-sequence of length 127, take a window of 100
    # The m-sequence has exactly 64 ones in 127 positions
    # We need to find a window with exactly 50 ones
    
    # Generate m-sequence using primitive polynomial x^7 + x + 1 (coefficients: 1,0,0,0,0,1,1)
    # Actually, let's use a simpler approach: try many random binary sequences
    
    # More systematic search: try all cyclic shifts of promising patterns
    # Also try random search with more iterations
    
    for attempt in range(2000):
        if time.time() > deadline:
            break
        
        # Create random binary pattern with exactly N/2 ones
        indices = list(range(N))
        random.shuffle(indices)
        ones_positions = set(indices[:N//2])
        h_bin = np.zeros(N)
        for pos in ones_positions:
            h_bin[pos] = 1.0
        
        s = score(h_bin)
        if s < best_score:
            best_score = s
            best_h = h_bin.copy()
    
    # Local search to refine
    current_h = best_h.copy()
    current_score = best_score
    
    for iteration in range(80000):
        if time.time() > deadline:
            break
        
        idx = random.randint(0, N-1)
        delta = random.uniform(-0.1, 0.1)
        new_h = current_h.copy()
        new_h[idx] = np.clip(new_h[idx] + delta, 0, 1)
        
        diff = N/2.0 - np.sum(new_h)
        if abs(diff) > 1e-10:
            candidates = [j for j in range(N) if j != idx]
            random.shuffle(candidates)
            adjusted = False
            for other_idx in candidates:
                adj = new_h[other_idx] - diff
                if 0 <= adj <= 1:
                    new_h[other_idx] = adj
                    adjusted = True
                    break
            
            if not adjusted:
                continue
        
        if abs(np.sum(new_h) - N/2.0) > 1e-8:
            continue
        
        s = score(new_h)
        if s < current_score - 1e-10:
            current_h = new_h
            current_score = s
            if s < best_score:
                best_score = s
                best_h = new_h.copy()
    
    return best_h
