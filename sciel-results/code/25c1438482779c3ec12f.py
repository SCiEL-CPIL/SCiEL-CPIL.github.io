"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autocorr(h):
    """Compute autocorrelation r[k] = sum_i h[i]*h[i+k] for k=0..N-1."""
    n_fft = 1 << (2 * N).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    r = np.fft.irfft(power, n=n_fft)
    return r[:N]


def _score(h):
    """Compute the upper bound on C3 (peak sidelobe metric)."""
    r = _autocorr(h)
    # Peak sidelobe: max |r[k]| for k = 1..N-1
    return np.max(np.abs(r[1:N]))


def _generate_lfsr_sequence(polynomial, length, initial_state):
    """Generate a binary sequence from an LFSR with the given primitive polynomial."""
    # polynomial is an integer where bit i represents the coefficient of x^i
    # For degree d, we need a d-bit register
    degree = polynomial.bit_length()
    
    # Initialize register
    register = initial_state & ((1 << degree) - 1)
    
    # Generate sequence
    sequence = []
    for _ in range(length):
        # Output bit is the least significant bit
        bit = register & 1
        sequence.append(bit)
        
        # Feedback bit
        feedback = register & 1
        
        # Shift right
        register >>= 1
        
        # XOR feedback with polynomial taps
        if feedback:
            register ^= polynomial
    
    return sequence


def _get_primitive_polynomials(max_degree=16):
    """Get a list of known primitive polynomials for LFSRs."""
    # These are x^n + ... + 1 represented as integers
    # For degree n, the polynomial has n+1 bits
    primitive_pols = []
    
    # Known primitive polynomials (from literature)
    # Format: integer representation where bit n is set for x^n term
    known_primitives = [
        0b10011,      # x^4 + x + 1
        0b100011,     # x^5 + x^2 + 1
        0b1000101,    # x^6 + x + 1
        0b10000101,   # x^7 + x + 1
        0b100000111,  # x^8 + x^4 + x^3 + x^2 + 1
        0b1000000011, # x^9 + x^4 + x^3 + x^2 + 1
        0b10000000101,# x^10 + x^3 + x^2 + x + 1
        0b100000000011, # x^11 + x^2 + 1
        0b1000000000011, # x^12 + x^6 + x^4 + x + 1
        0b100000000000011, # x^13 + x^4 + x^3 + x + 1
        0b1000000000000011, # x^14 + x^5 + x^3 + x^2 + 1
        0b100000000000000011, # x^15 + x + 1
        0b1000000000000000011, # x^16 + x^5 + x^3 + x^2 + 1
    ]
    
    for poly in known_primitives:
        degree = poly.bit_length()
        if degree <= max_degree:
            primitive_pols.append(poly)
    
    return primitive_pols


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_h = None
    best_score = np.inf
    
    np.random.seed(42)
    
    # Get primitive polynomials
    primitive_pols = _get_primitive_polynomials(max_degree=16)
    
    # Phase 1: Try LFSR-based sequences
    for poly in primitive_pols:
        if time.time() > deadline - 10.0:
            break
        
        degree = poly.bit_length()
        # Need at least N bits, so use multiple periods if needed
        period = (1 << degree) - 1
        
        # Try different initial states
        for init_state in range(1, min(10, (1 << degree))):
            if time.time() > deadline - 10.0:
                break
            
            # Generate sequence
            bits = _generate_lfsr_sequence(poly, N, init_state)
            h = np.array([2 * b - 1 for b in bits], dtype=np.float64)
            
            s = _score(h)
            if s < best_score:
                best_score = s
                best_h = h.copy()
    
    # Phase 2: Random restarts to find good initial candidates
    while time.time() < deadline - 30.0:
        h = np.random.choice([-1, 1], N).astype(np.float64)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # Phase 3: Local search (flip individual bits) from best candidate
    if best_h is not None and time.time() < deadline - 5.0:
        current_h = best_h.copy()
        current_score = best_score
        n_no_improve = 0
        max_no_improve = 500
        
        while time.time() < deadline - 5.0 and n_no_improve < max_no_improve:
            # Try flipping a random position
            idx = np.random.randint(N)
            old_val = current_h[idx]
            current_h[idx] = -old_val
            
            new_score = _score(current_h)
            if new_score < current_score:
                current_score = new_score
                n_no_improve = 0
                if current_score < best_score:
                    best_score = current_score
                    best_h = current_h.copy()
            else:
                # Revert
                current_h[idx] = old_val
                n_no_improve += 1
        
        # Phase 4: Try multiple restarts of local search from different random starts
        while time.time() < deadline - 5.0:
            h = np.random.choice([-1, 1], N).astype(np.float64)
            s = _score(h)
            if s < best_score:
                best_score = s
                best_h = h.copy()
            
            # Local search from this point
            current_h = h.copy()
            current_score = s
            n_no_improve = 0
            max_no_improve = 200
            
            while time.time() < deadline - 5.0 and n_no_improve < max_no_improve:
                idx = np.random.randint(N)
                old_val = current_h[idx]
                current_h[idx] = -old_val
                
                new_score = _score(current_h)
                if new_score < current_score:
                    current_score = new_score
                    n_no_improve = 0
                    if current_score < best_score:
                        best_score = current_score
                        best_h = current_h.copy()
                else:
                    current_h[idx] = old_val
                    n_no_improve += 1
    
    if best_h is None:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
