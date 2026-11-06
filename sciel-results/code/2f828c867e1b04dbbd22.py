"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

N = 256
K = 4

def count_mono_4ap_fast(coloring):
    """Faster count using numpy vectorization per d."""
    c = np.array(coloring, dtype=np.int8)
    count = 0
    for d in range(1, N):
        max_a = N - 3*d
        if max_a < 1:
            break
        # Extract the four sequences
        seq1 = c[:max_a]           # a-1 for a=1..max_a
        seq2 = c[d:d+max_a]
        seq3 = c[2*d:2*d+max_a]
        seq4 = c[3*d:3*d+max_a]
        mono = (seq1 == seq2) & (seq2 == seq3) & (seq3 == seq4)
        count += np.sum(mono)
    return count

def thue_morse_init():
    """Initialize with Thue-Morse sequence, which has low AP density."""
    # Thue-Morse: t(n) = popcount(n) % 2
    # For 1..256, t(1)=1, t(2)=1, t(3)=0, t(4)=1, ...
    # Actually t(0)=0, t(1)=1, t(2)=1, t(3)=0, t(4)=1, t(5)=0, t(6)=0, t(7)=1, ...
    # We want colors for 1..256.
    colors = np.array([bin(i).count('1') % 2 for i in range(1, N+1)], dtype=np.int8)
    return colors

def vectorized_sa(coloring, deadline, batch_size=64, T_init=5.0, T_min=0.05, alpha=0.9995):
    """Vectorized Simulated Annealing for 4-AP minimization."""
    c = coloring.copy()
    current_score = count_mono_4ap_fast(c)
    best = c.copy()
    best_score = current_score
    
    T = T_init
    start_time = time.time()
    
    # Precompute indices for all 4-APs to speed up delta calculation if needed, 
    # but for vectorized SA we just recount or use fast delta.
    # Given N=256, recounting is ~50k ops, which is fast.
    # Let's try to make the inner loop as fast as possible.
    
    while time.time() < deadline - 5:
        if T < T_min:
            break
            
        # Generate batch of random flips
        batch_size = min(batch_size, N)
        indices = np.random.choice(N, size=batch_size, replace=False)
        
        # Apply flips to a temporary copy
        c_new = c.copy()
        c_new[indices] = 1 - c_new[indices]
        
        # Calculate new score
        new_score = count_mono_4ap_fast(c_new)
        delta = new_score - current_score
        
        # Metropolis criterion
        if delta <= 0 or np.random.random() < np.exp(-delta / T):
            c = c_new
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best = c.copy()
        
        T *= alpha
        
    return best, best_score

def local_search_polish(coloring, deadline):
    """Final greedy local search to polish the result."""
    c = coloring.copy()
    score = count_mono_4ap_fast(c)
    best = c.copy()
    best_score = score
    
    while time.time() < deadline - 2:
        improved = False
        # Try each position
        for pos in range(N):
            old_bit = c[pos]
            c[pos] = 1 - old_bit
            new_score = count_mono_4ap_fast(c)
            if new_score < score:
                score = new_score
                improved = True
                if score < best_score:
                    best_score = score
                    best = c.copy()
            else:
                c[pos] = old_bit
        if not improved:
            break
            
    return best, best_score

def construct():
    deadline = time.time() + 115  # Safety margin
    
    # Phase 1: Thue-Morse initialization
    coloring = thue_morse_init()
    score = count_mono_4ap_fast(coloring)
    best_coloring = coloring.copy()
    best_score = score
    
    # Phase 2: Vectorized Simulated Annealing
    if time.time() < deadline - 60:
        sa_coloring, sa_score = vectorized_sa(coloring, deadline - 60)
        if sa_score < best_score:
            best_score = sa_score
            best_coloring = sa_coloring
    
    # Phase 3: Local Search Polish
    if time.time() < deadline - 10:
        ls_coloring, ls_score = local_search_polish(best_coloring, deadline - 5)
        if ls_score < best_score:
            best_score = ls_score
            best_coloring = ls_coloring
    
    return list(best_coloring)
# EVOLVE-BLOCK-END
