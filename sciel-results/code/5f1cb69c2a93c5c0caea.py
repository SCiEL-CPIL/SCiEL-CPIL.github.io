"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Construct the sequence of step heights.
    
    Returns: sequence of N step heights in [0, 1] whose sum is exactly N/2.
    """
    best_cost = float('inf')
    best_x = None
    
    # Try different numbers of domain walls (even M)
    # M is the number of transitions, so number of segments is M
    # M must be even for sum 0 (equal number of +1 and -1 segments? No, just lengths summing to N/2)
    # Actually, if M is even, we have M/2 segments of +1 and M/2 of -1 (if we start with +1 and alternate).
    # Sum of +1 lengths = N/2. Sum of -1 lengths = N/2.
    
    for M in range(2, N + 1, 2):
        # Initialize lengths
        # L_odd sum to N/2, L_even sum to N/2
        # L_i >= 1
        
        half_N = N // 2
        num_odd = M // 2
        num_even = M // 2
        
        # Random initialization
        def init_lengths():
            # Generate random compositions of half_N into num_odd parts
            # Stars and bars: place num_odd-1 bars in half_N-1 gaps
            if num_odd == 1:
                l_odd = [half_N]
            else:
                bars = np.sort(np.random.choice(range(1, half_N), num_odd - 1, replace=False))
                l_odd = []
                prev = 0
                for b in bars:
                    l_odd.append(b - prev)
                    prev = b
                l_odd.append(half_N - prev)
            
            if num_even == 1:
                l_even = [half_N]
            else:
                bars = np.sort(np.random.choice(range(1, half_N), num_even - 1, replace=False))
                l_even = []
                prev = 0
                for b in bars:
                    l_even.append(b - prev)
                    prev = b
                l_even.append(half_N - prev)
            
            # Interleave
            lengths = []
            for i in range(M // 2):
                lengths.append(l_odd[i])
                lengths.append(l_even[i])
            return lengths
        
        lengths = init_lengths()
        
        def compute_cost(lengths):
            # Construct x
            x = np.zeros(N)
            curr = 0
            sign = 1
            for i, L in enumerate(lengths):
                x[curr:curr+L] = sign
                curr += L
                sign = -sign
            
            # Compute max D_k
            # D_k = sum (x_i != x_{i+k})
            # A_k = sum x_i x_{i+k}
            # D_k = N - A_k
            # Minimize max D_k -> Minimize max (N - A_k) -> Maximize min A_k
            
            min_A = N
            for k in range(1, N):
                A_k = np.dot(x, np.roll(x, -k))
                if A_k < min_A:
                    min_A = A_k
            
            cost = N - min_A
            return cost, x
        
        current_cost, current_x = compute_cost(lengths)
        best_M_cost = current_cost
        best_M_x = current_x
        best_M_lengths = lengths[:]
        
        # SA
        T = 1.0
        T_min = 0.001
        steps = 2000
        cooling_rate = (T_min / T) ** (1.0 / steps)
        
        for s in range(steps):
            # Move: transfer length between two segments of same parity
            # Pick two random indices i, j from 0..M-1
            i = np.random.randint(M)
            j = np.random.randint(M)
            if i == j:
                continue
            if (i % 2) != (j % 2):
                continue
            
            # Transfer delta
            # Max delta is min(lengths[i]-1, lengths[j]-1)
            max_delta = min(lengths[i] - 1, lengths[j] - 1)
            if max_delta <= 0:
                continue
            
            delta = np.random.randint(1, max_delta + 1)
            
            # Check direction: i to j or j to i
            if np.random.rand() < 0.5:
                # i loses delta, j gains
                new_lengths = lengths[:]
                new_lengths[i] -= delta
                new_lengths[j] += delta
            else:
                new_lengths = lengths[:]
                new_lengths[i] += delta
                new_lengths[j] -= delta
            
            new_cost, new_x = compute_cost(new_lengths)
            
            if new_cost < current_cost:
                current_cost = new_cost
                current_x = new_x
                lengths = new_lengths
                if current_cost < best_M_cost:
                    best_M_cost = current_cost
                    best_M_x = new_x
                    best_M_lengths = new_lengths[:]
            else:
                if np.random.rand() < np.exp((current_cost - new_cost) / T):
                    current_cost = new_cost
                    current_x = new_x
                    lengths = new_lengths
                # else revert (lengths unchanged)
            
            T *= cooling_rate
        
        if best_M_cost < best_cost:
            best_cost = best_M_cost
            best_x = best_M_x
    
    # Convert best_x to h
    # best_x is in {-1, 1}
    h = (best_x + 1.0) / 2.0
    
    return h
# EVOLVE-BLOCK-END
