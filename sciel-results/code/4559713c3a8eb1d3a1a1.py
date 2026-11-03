"""
Strategy: Simulated Annealing with Block-Shift Moves
The current program uses random swaps and generic local search.
This program uses:
1. A structured move: rotating a contiguous block of indices.
   This preserves the multiset of values (and thus the sum) exactly,
   which is a strong constraint satisfaction property.
2. Simulated Annealing to escape local minima.
3. Fast overlap calculation using FFT for O(N log N) cost per evaluation,
   allowing many more iterations within the time budget.
"""

import numpy as np
import time

N = 100

def _fast_max_overlap(h):
    """
    Compute max overlap using FFT.
    Overlap at shift k is sum_i h[i] * (1 - h[i-k]) = sum_i h[i] - sum_i h[i]h[i-k].
    sum_i h[i] is constant (N/2).
    We need to compute c_k = sum_i h[i]h[i-k] for all k.
    This is the circular cross-correlation of h with h.
    Using FFT: corr = ifft(fft(h) * conj(fft(h))).real
    Note: numpy's ifft of |fft(h)|^2 gives circular autocorrelation.
    Specifically, if H = fft(h), then ifft(H * conj(H)).real[k] = sum_i h[i] h[(i-k) % N].
    Wait, let's verify the index convention.
    np.correlate(h, h, mode='full') gives linear correlation.
    Circular correlation:
    Let H = fft(h).
    P = H * np.conj(H)
    c = np.fft.ifft(P).real
    c[k] = sum_{i=0}^{N-1} h[i] * h[(i-k) % N]
    
    Overlap(k) = sum_i h[i] - c[k]
    Max Overlap = max_k (sum_i h[i] - c[k]) = sum_i h[i] - min_k c[k]
    """
    H = np.fft.fft(h)
    power = H * np.conj(H)
    corr = np.fft.ifft(power).real
    # Due to floating point, corr might have tiny imaginary parts, .real handles that.
    # Also, numerical errors might make values slightly off.
    min_corr = np.min(corr)
    total_sum = np.sum(h)
    return total_sum - min_corr

def _project(h):
    """
    Project h onto the feasible set: 0 <= h_i <= 1, sum(h) = N/2.
    Simple approach: Clip, then adjust sum.
    """
    h = np.clip(h, 0, 1)
    diff = N / 2.0 - np.sum(h)
    if abs(diff) < 1e-12:
        return h
    # Distribute difference
    # To keep it simple and robust, we can add diff/N to all, then clip, then repeat?
    # Or just add diff to elements with room.
    # For speed in inner loop, we rely on the block shift move which preserves sum exactly.
    # This projection is only used for initialization or if we add noise.
    if diff > 0:
        # Need to increase sum. Add to elements < 1.
        mask = h < 1.0
        if np.sum(mask) > 0:
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0, 1)
    else:
        # Need to decrease sum. Subtract from elements > 0.
        mask = h > 0.0
        if np.sum(mask) > 0:
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0, 1)
    return h

def _block_shift_move(h, rng):
    """
    Pick a random start index and a random length (1 to N//2).
    Rotate that block.
    This preserves the sum exactly.
    """
    start = rng.integers(0, N)
    length = rng.integers(1, N // 2 + 1)
    # Create new h
    new_h = h.copy()
    # Indices of the block
    # We want to rotate the segment h[start : start+length]
    # Let's say we shift by 'shift' amount within the block.
    shift = rng.integers(1, length)
    
    # Extract block
    # Need to handle wrap-around for the block itself? No, the block is contiguous in the array.
    # But the array is circular for the objective.
    # The move is on the array indices.
    
    # Indices: start, start+1, ..., start+length-1 (mod N)
    idxs = np.arange(start, start + length) % N
    # We want to permute h[idxs] by shifting.
    # Let's just create a new array for the block
    block = h[idxs]
    new_block = np.roll(block, shift)
    new_h[idxs] = new_block
    return new_h

def construct():
    start_time = time.time()
    deadline = time.time() + 145.0  # 5s safety margin
    
    # Initialize with a structured candidate
    # Start with a "sawtooth" or "square wave" to break symmetry
    # A square wave (half 0, half 1) is a good starting point for combinatorial structures
    # But let's start with a smooth wave to avoid high frequency noise initially
    # Actually, the problem is about "overlap", which is related to correlation.
    # Low frequency waves have high correlation with small shifts.
    # We want to MINIMIZE overlap.
    # Overlap(k) = Sum h - Corr(k).
    # Min Overlap = Max Corr.
    # We want to MAXIMIZE the minimum correlation.
    # Wait.
    # Task: "minimize the maximum overlap"
    # Overlap(k) = sum h[i] (1-h[i-k])
    # We want min_k max Overlap(k).
    # Overlap(k) = S - C(k).
    # Minimize max_k (S - C(k)) = S - min_k C(k).
    # So we want to MAXIMIZE min_k C(k).
    # C(k) is the circular autocorrelation.
    # C(0) = sum h[i]^2. This is fixed if h is binary, but varies for continuous.
    # For a given L2 norm, C(0) is fixed.
    # We want the autocorrelation to be as "flat" as possible (high minimum).
    # This is similar to designing a " Barker code" or "Golay sequence" but for real values in [0,1].
    # Ideally, C(k) is constant for all k != 0.
    # This is an "equi-correlation" problem.
    
    # Let's use Simulated Annealing to maximize min(C(k)).
    # Objective to maximize: f(h) = min_k C(k).
    # This is non-smooth.
    # We can use a soft-min: -logsumexp(-C(k)/T) or just optimize the min directly with SA.
    # SA works well with discrete-like moves.
    
    # Initialize
    # Start with a random permutation of a base signal?
    # Or just random uniform.
    # Let's start with a binary sequence (0s and 1s) with exactly N/2 ones.
    # This is a strong candidate structure.
    h = np.zeros(N)
    ones_indices = np.random.choice(N, N // 2, replace=False)
    h[ones_indices] = 1.0
    
    current_h = h.copy()
    current_score = _fast_max_overlap(current_h) # This is the overlap. We want to minimize this.
    best_h = current_h.copy()
    best_score = current_score
    
    # SA Parameters
    T_start = 1.0
    T_end = 1e-4
    alpha = 0.995 # Cooling rate
    
    # We need to track time
    while time.time() < deadline:
        # Generate move
        new_h = _block_shift_move(current_h, np.random)
        
        # Calculate scores
        # Note: _fast_max_overlap returns the overlap value. Lower is better.
        new_score = _fast_max_overlap(new_h)
        
        delta = new_score - current_score # Positive means worse
        
        if delta < 0 or np.random.random() < np.exp(-delta / T_start):
            current_h = new_h
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_h = current_h.copy()
        
        # Cool down
        T_start *= alpha
        if T_start < T_end:
            # Restart with a new random start? Or just keep going?
            # Let's restart with a slightly perturbed best solution
            T_start = T_start * 10
            # Perturb best
            current_h = best_h.copy()
            # Add small noise and project? 
            # Or just restart SA from best_h
            current_score = best_score
            # Add a few random moves to escape
            for _ in range(10):
                current_h = _block_shift_move(current_h, np.random)
            current_score = _fast_max_overlap(current_h)
            
    return best_h

# Ensure the function name matches
# The task requires: construct() -> sequence of step heights

# One refinement: The block shift move might be too restrictive if we stick to binary.
# The optimal solution might be non-binary.
# Let's allow "value mixing" moves as well?
# Or start with a non-binary candidate?
# The binary constraint is not in the problem statement. h in [0,1].
# However, for linear constraints and convex objectives, extremes are often optimal.
# But min-max is not convex.
# Let's stick to the current SA which can explore the space of permutations of the initial values.
# If we start with binary, we stay binary.
# Let's initialize with a mix of values to allow continuous optimization.
# Actually, let's initialize with a sinusoid, which is continuous.

def construct():
    start_time = time.time()
    deadline = time.time() + 145.0
    
    # Multiple restarts with different initializations
    best_h = None
    best_score = float('inf')
    
    # Strategy: 
    # 1. Try binary initializations (combinatorial)
    # 2. Try sinusoidal initializations (continuous)
    
    init_candidates = []
    
    # Binary
    for _ in range(5):
        h = np.zeros(N)
        ones_indices = np.random.choice(N, N // 2, replace=False)
        h[ones_indices] = 1.0
        init_candidates.append(h)
        
    # Sinusoidal
    for freq in [1, 2, 3]:
        x = np.linspace(0, 2*np.pi*freq, N, endpoint=False)
        h = 0.5 + 0.5 * np.sin(x)
        h = np.clip(h, 0, 1)
        # Adjust sum
        h = _project(h)
        init_candidates.append(h)
        
    # Random
    for _ in range(5):
        h = np.random.uniform(0, 1, N)
        h = _project(h)
        init_candidates.append(h)
        
    T_start = 0.5
    T_end = 1e-4
    alpha = 0.99
    
    for h0 in init_candidates:
        if time.time() > deadline - 5:
            break
            
        current_h = h0.copy()
        current_score = _fast_max_overlap(current_h)
        
        if current_score < best_score:
            best_score = current_score
            best_h = current_h.copy()
            
        T = T_start
        while T > T_end and time.time() < deadline - 5:
            new_h = _block_shift_move(current_h, np.random)
            new_score = _fast_max_overlap(new_h)
            
            delta = new_score - current_score
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current_h = new_h
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_h = current_h.copy()
                    
            T *= alpha
            
    return best_h

# Wait, I defined construct twice. The second one overwrites the first.
# The second one is better.

# Final check on _fast_max_overlap correctness.
# h = [1, 0, 1, 0] (N=4)
# C(0) = 1^2+0+1^2+0 = 2
# C(1) = 1*0 + 0*1 + 1*0 + 0*1 = 0 (shift 1: h[0]h[-1]=1*0=0, h[1]h[0]=0*1=0... wait)
# Correlation definition: c[k] = sum h[i] h[i-k]
# k=1: h[0]h[3] + h[1]h[0] + h[2]h[1] + h[3]h[2]
#      1*0 + 0*1 + 1*0 + 0*1 = 0
# k=2: h[0]h[2] + h[1]h[3] + h[2]h[1] + h[3]h[0]
#      1*1 + 0*0 + 1*0 + 0*1 = 1
# k=3: h[0]h[1] + h[1]h[2] + h[2]h[3] + h[3]h[0]
#      1*0 + 0*1 + 1*0 + 0*1 = 0
# Min C = 0.
# Sum h = 2.
# Max Overlap = 2 - 0 = 2.
# Let's check manually.
# h = [1, 0, 1, 0]
# 1-h = [0, 1, 0, 1]
# Shift 1: [0, 0, 1, 1] (h shifted right? or left?)
# "overlap of h with shifted 1-h"
# Usually shift k means x(t-k).
# If h=[1,0,1,0], 1-h=[0,1,0,1].
# Shift 1 (right): [1, 0, 1, 0] -> 1-h becomes [1, 0, 1, 0]? No.
# Let's stick to the formula: Overlap(k) = sum h[i] (1 - h[i-k]).
# k=1: sum h[i] (1 - h[i-1])
# i=0: h[0](1-h[-1]) = 1(1-0) = 1
# i=1: h[1](1-h[0]) = 0(1-1) = 0
# i=2: h[2](1-h[1]) = 1(1-0) = 1
# i=3: h[3](1-h[2]) = 0(1-1) = 0
# Sum = 2.
# Formula: S - C(1) = 2 - 0 = 2. Matches.
# k=2:
# i=0: 1(1-h[2]) = 1(0) = 0
# i=1: 0
# i=2: 1(1-h[0]) = 1(0) = 0
# i=3: 0
# Sum = 0.
# Formula: S - C(2) = 2 - 1 = 1.
# Discrepancy!
# Let's re-evaluate C(2).
# C(2) = sum h[i] h[i-2]
# i=0: h[0]h[2] = 1*1 = 1
# i=1: h[1]h[3] = 0
# i=2: h[2]h[0] = 1*1 = 1
# i=3: h[3]h[1] = 0
# Sum = 2.
# My previous manual C(2) calculation was wrong.
# So C(2) = 2.
# Formula: S - C(2) = 2 - 2 = 0. Matches.
# So Min C is 0 (at k=1,3). Max Overlap is 2.
# The code should work.

# One detail: np.fft.ifft returns complex. .real is correct.
# The order of correlation from FFT:
# ifft(fft(h) * conj(fft(h))) gives circular autocorrelation where index k corresponds to lag k.
# Specifically, c[k] = sum_{n=0}^{N-1} h[n] h[n-k].
# This matches the definition used.

# The code is ready.

