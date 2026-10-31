import numpy as np
import time
import sys

N = 400

def generate_m_sequence(length):
    """Generate an m-sequence of length 2^k - 1 where 2^k - 1 >= length."""
    # Find smallest k such that 2^k - 1 >= length
    k = 1
    while (1 << k) - 1 < length:
        k += 1
    
    # Primitive polynomial for GF(2^k)
    # For k=9 (511), x^9 + x^4 + 1 is primitive?
    # Let's use a known one or just use a simple LFSR.
    # We need a primitive polynomial.
    # For k=9, x^9 + x^4 + 1 is primitive.
    # For k=8, x^8 + x^4 + x^3 + x^2 + 1 is primitive.
    # We will pick based on k.
    
    # Simple primitive polynomials
    polys = {
        1: [0], # 2^1-1=1, trivial
        2: [1, 0], # x^2 + x + 1? No, x^2+x+1 is primitive for k=2.
        3: [1, 0, 1], # x^3 + x + 1
        4: [1, 0, 1, 0, 1], # x^4 + x + 1
        5: [1, 1, 0, 1, 0, 1], # x^5 + x^2 + 1
        6: [1, 1, 1, 0, 0, 0, 1], # x^6 + x^5 + x^3 + x^2 + x + 1? No.
        # Let's just use a generic LFSR and check if it's maximal.
        # For simplicity, we'll use a known good one for k=9 (511 > 400).
        # x^9 + x^4 + 1
        9: [1, 0, 0, 0, 1, 0, 0, 0, 0, 1] # Coeffs for x^9...x^0? 
    }
    
    # Let's hardcode a primitive polynomial for k=9
    # x^9 + x^4 + 1
    # Taps at 9 and 4 (0-indexed from LSB? No, standard LFSR)
    # Let's implement a simple LFSR.
    
    # For k=9, length 511.
    # We'll just use a random seed LFSR and hope it's maximal, or use a fixed one.
    # Fixed primitive poly for k=9: 0x183 (binary 1 1000 0011)? 
    # Let's use a standard one.
    
    # Actually, for N=400, k=9 is sufficient.
    # We'll use the polynomial x^9 + x^4 + 1.
    # This corresponds to taps 9 and 4.
    
    lfsr_state = 1
    seq = []
    for i in range((1 << k) - 1):
        seq.append(lfsr_state & 1)
        # Feedback
        # x^9 + x^4 + 1 = 0 -> x^9 = x^4 + 1
        # In LFSR, new bit = bit[8] XOR bit[3] ?
        # Let's just use a simple implementation.
        # For k=9, taps are 9 and 4.
        # feedback = ((lfsr_state >> (k-1)) ^ (lfsr_state >> (k-5))) & 1
        # Wait, x^9 + x^4 + 1.
        # If we shift left, we lose MSB.
        # Let's use right shift LFSR.
        # feedback = (lfsr_state & 1) ^ ((lfsr_state >> 3) & 1) ?
        # This is tricky without a verified poly.
        
        # Alternative: Use a known m-sequence generator.
        # We'll just use np.random with a fixed seed to generate a "pseudo-m-sequence" 
        # or actually, since we have time, we can just use a random sequence 
        # and local search. The m-sequence hint might be a red herring 
        # or just a good initial guess.
        
        # Let's stick to the local search on random init.
        break

    # Fallback: Just return a random sequence for init
    return np.random.choice([-1, 1], size=length)

def compute_score(x):
    # Autoconvolution
    c = np.convolve(x, x, mode='full')
    max_abs = np.max(np.abs(c))
    integral = np.sum(x)
    if integral == 0:
        return 1e10
    return max_abs / (integral * integral)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # Initialize with m-sequence like structure (random +/-1)
    # But we know +/-1 is bad.
    # Let's initialize with all 1s (the seed) and perturb.
    x = np.ones(N)
    best_score = compute_score(x)
    best_x = x.copy()
    
    # Local Search: Flip random indices
    # Since flipping signs reduces Sum, we might want to change magnitudes?
    # The problem allows "signed step heights".
    # Let's try optimizing the values directly using a simple hill climbing.
    
    # We'll try a few random restarts with local search on signs
    # and also a gradient step on values.
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Initialize
        if restart == 0:
            x = np.ones(N)
        else:
            # Random init
            x = np.random.choice([-1, 1], size=N)
            # Or maybe mix of 1 and -1 with some zeros?
            # No, non-zero is better for Sum.
        
        current_score = compute_score(x)
        
        # Local search
        for step in range(1000):
            if time.time() > deadline:
                break
            
            if current_score < best_score:
                best_score = current_score
                best_x = x.copy()
            
            # Pick a random index to flip or adjust
            idx = np.random.randint(N)
            
            # Try flipping sign
            old_val = x[idx]
            x[idx] = -old_val
            new_score = compute_score(x)
            
            if new_score < current_score:
                current_score = new_score
            else:
                x[idx] = old_val
                
            # Also try small perturbation of magnitude?
            # If we allow real values, we can scale.
            # But scaling doesn't change score.
            # Changing relative magnitudes does.
            # Let's try changing value to 0? No, reduces sum.
            # Let's try changing value to 2*old_val?
            # This increases sum and C0.
            
            # For now, stick to sign flips.
            
    return best_x

# EVOLVE-BLOCK-START
import numpy as np
import time

N = 400

def _compute_score(x):
    c = np.convolve(x, x, mode='full')
    max_abs = np.max(np.abs(c))
    integral = np.sum(x)
    if integral == 0:
        return 1e10
    return max_abs / (integral * integral)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_x = np.ones(N)
    best_score = _compute_score(best_x)
    
    # Try a few random initializations and local search
    for restart in range(20):
        if time.time() > deadline:
            break
            
        if restart == 0:
            x = np.ones(N)
        else:
            # Random +/-1
            x = np.random.choice([-1, 1], size=N)
            
        current_score = _compute_score(x)
        
        for step in range(5000):
            if time.time() > deadline:
                break
                
            if current_score < best_score:
                best_score = current_score
                best_x = x.copy()
                
            idx = np.random.randint(N)
            old_val = x[idx]
            x[idx] = -old_val
            new_score = _compute_score(x)
            
            if new_score < current_score:
                current_score = new_score
            else:
                x[idx] = old_val
                
    return best_x
# EVOLVE-BLOCK-END
