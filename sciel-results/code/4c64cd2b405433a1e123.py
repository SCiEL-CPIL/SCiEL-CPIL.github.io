import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin
    
    n = 128
    # A is a subset of [0, 128]
    # Target sums: [128, 256]
    
    def compute_reps(A_bool):
        """Compute representation counts for sums in [128, 256] using convolution."""
        # Convolution of A with itself gives ordered pairs
        # For unordered with repetition: rep[s] = (conv[s] + (s even and s//2 in A ? 1 : 0)) // 2
        # Actually: ordered pairs count = conv[s]
        # Unordered pairs with equal allowed: if a != b, each pair counted twice in conv
        # if a == b, counted once
        # So unordered = (conv[s] + diag[s]) // 2 where diag[s] = 1 if s even and s//2 in A
        conv = np.convolve(A_bool, A_bool)
        # conv[s] = number of ordered pairs (a,b) in A x A with a+b=s
        # For unordered: pairs with a<b counted twice, a=b counted once
        # unordered_count[s] = (conv[s] + (1 if s%2==0 and A_bool[s//2] else 0)) // 2
        reps = np.zeros(257, dtype=np.int32)
        for s in range(128, 257):
            diag = 1 if (s % 2 == 0 and s // 2 <= 128 and A_bool[s // 2]) else 0
            reps[s] = (conv[s] + diag) // 2
        return reps[128:257]  # 129 values
    
    def get_score(A_bool):
        """Return (peak_rep, uncovered_count). Lower is better."""
        reps = compute_reps(A_bool)
        uncovered = np.sum(reps == 0)
        if uncovered > 0:
            return (10000 + uncovered, uncovered)  # heavy penalty for uncovered
        peak = int(np.max(reps))
        return (peak, 0)
    
    # Initial solution: try a structured approach
    # Start with the seed and improve
    A = np.zeros(129, dtype=np.int8)
    # Seed: [64, 128]
    A[64:129] = 1
    
    # Try a few different initial structures
    best_A = A.copy()
    best_score = get_score(A)
    
    # Also try: split into two groups for better coverage
    # e.g., [0, 32] U [96, 128] -> sums cover [96, 256], need [128,256]
    # Let's try [0, 48] U [80, 128]
    A2 = np.zeros(129, dtype=np.int8)
    A2[0:49] = 1
    A2[80:129] = 1
    s2 = get_score(A2)
    if s2 < best_score:
        best_A = A2.copy()
        best_score = s2
    
    # Try [0, 32] U [96, 128]
    A3 = np.zeros(129, dtype=np.int8)
    A3[0:33] = 1
    A3[96:129] = 1
    s3 = get_score(A3)
    if s3 < best_score:
        best_A = A3.copy()
        best_score = s3
    
    # Try [0, 40] U [88, 128]
    A4 = np.zeros(129, dtype=np.int8)
    A4[0:41] = 1
    A4[88:129] = 1
    s4 = get_score(A4)
    if s4 < best_score:
        best_A = A4.copy()
        best_score = s4
    
    # Local search with simulated annealing
    current_A = best_A.copy()
    current_score = best_score
    best_A_sa = best_A.copy()
    best_score_sa = best_score
    
    temperature = 5.0
    cooling = 0.995
    
    import random
    rng = np.random.RandomState(42)
    
    while time.time() < deadline:
        # Pick a random move: add, remove, or swap an element
        move_type = rng.randint(0, 3)
        
        if move_type == 0:  # Add an element
            candidates = np.where(current_A == 0)[0]
            if len(candidates) == 0:
                continue
            idx = candidates[rng.randint(0, len(candidates))]
            new_A = current_A.copy()
            new_A[idx] = 1
        elif move_type == 1:  # Remove an element
            candidates = np.where(current_A == 1)[0]
            if len(candidates) <= 1:  # keep nonempty
                continue
            idx = candidates[rng.randint(0, len(candidates))]
            new_A = current_A.copy()
            new_A[idx] = 0
        else:  # Swap: remove one, add another
            in_candidates = np.where(current_A == 1)[0]
            out_candidates = np.where(current_A == 0)[0]
            if len(in_candidates) == 0 or len(out_candidates) == 0:
                continue
            idx_in = in_candidates[rng.randint(0, len(in_candidates))]
            idx_out = out_candidates[rng.randint(0, len(out_candidates))]
            new_A = current_A.copy()
            new_A[idx_in] = 0
            new_A[idx_out] = 1
        
        new_score = get_score(new_A)
        
        # Accept based on simulated annealing
        if new_score < current_score:
            current_A = new_A
            current_score = new_score
            if current_score < best_score_sa:
                best_A_sa = current_A.copy()
                best_score_sa = current_score
        else:
            delta = new_score[0] - current_score[0]
            if delta == 0:
                # Tie: prefer fewer elements or just accept randomly
                if rng.random() < 0.5:
                    current_A = new_A
            else:
                prob = np.exp(-delta / max(temperature, 0.01))
                if rng.random() < prob:
                    current_A = new_A
                    current_score = new_score
        
        temperature *= cooling
        if temperature < 0.01:
            temperature = 0.01
    
    # Use the best found
    final_A = best_A_sa
    return [i for i in range(129) if final_A[i] == 1]
