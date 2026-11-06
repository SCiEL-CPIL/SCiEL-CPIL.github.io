import time
import itertools
from collections import defaultdict
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    # Fallback: Simple polynomial, score is likely 21
    # (1 + x + ... + x^11)^2 has 21 non-zero terms
    fallback = [1] * 12
    fallback_score = 21

    best_score = fallback_score
    best_candidate = fallback

    def compute_square_nonzeros_fast(exps, coeffs):
        """
        Fast computation using numpy for the square.
        """
        n = len(exps)
        # Build dense representation of P
        max_deg = max(exps)
        p = np.zeros(max_deg + 1, dtype=np.int64)
        for e, c in zip(exps, coeffs):
            p[e] = c
        
        # Compute square via convolution
        square = np.convolve(p, p)
        
        # Count nonzeros
        return int(np.count_nonzero(square))

    def try_candidate(exps, coeffs):
        """
        Check if candidate is valid and update best if better.
        """
        nonlocal best_score, best_candidate
        
        # Validation
        if len(exps) != 12 or len(coeffs) != 12:
            return
        if len(set(exps)) != 12:
            return
        if max(exps) > 40:
            return
        if any(abs(c) > 64 for c in coeffs):
            return
        if any(c == 0 for c in coeffs): # Must be exactly 12 non-zero terms in P
            return
        
        score = compute_square_nonzeros_fast(exps, coeffs)
        
        if score < best_score:
            best_score = score
            # Build dense list
            max_deg = max(exps)
            dense = [0] * (max_deg + 1)
            for e, c in zip(exps, coeffs):
                dense[e] = c
            # Ensure final coeff is non-zero (guaranteed by max(exps))
            if dense[-1] != 0:
                best_candidate = dense

    def optimize_signs(exps, max_iter=200, restarts=8):
        """
        Local search for best sign assignment {-1, 1} for given exponents.
        Uses multi-flip moves to escape local optima.
        """
        if time.time() > deadline:
            return
        
        import random
        random.seed(42 + hash(tuple(exps)) % 1000)
        
        best_local_score = float('inf')
        best_local_signs = [1] * 12

        for r in range(restarts):
            if time.time() > deadline:
                break
            signs = [1 if random.random() > 0.5 else -1 for _ in range(12)]
            current_score = compute_square_nonzeros_fast(exps, signs)
            
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Try single flips
                improved = False
                for i in range(12):
                    old = signs[i]
                    signs[i] = -signs[i]
                    new_score = compute_square_nonzeros_fast(exps, signs)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        signs[i] = old
                
                # Try multi-flips (2-3 flips) to escape local optima
                if not improved and it % 5 == 0:
                    for _ in range(3):
                        idx1 = random.randint(0, 11)
                        idx2 = random.randint(0, 11)
                        if idx1 == idx2:
                            continue
                        old1, old2 = signs[idx1], signs[idx2]
                        signs[idx1] = -signs[idx1]
                        signs[idx2] = -signs[idx2]
                        new_score = compute_square_nonzeros_fast(exps, signs)
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                        else:
                            signs[idx1] = old1
                            signs[idx2] = old2
                
                if not improved:
                    break
            
            if current_score < best_local_score:
                best_local_score = current_score
                best_local_signs = signs[:]
        
        if best_local_score < best_score:
            try_candidate(exps, best_local_signs)

    def optimize_coeffs(exps, max_iter=100, restarts=5):
        """
        Local search for small integer coefficients in [-64, 64].
        """
        if time.time() > deadline:
            return
        
        import random
        random.seed(99 + hash(tuple(exps)) % 1000)
        
        best_local_score = float('inf')
        best_local_coeffs = [1] * 12
        
        for r in range(restarts):
            if time.time() > deadline:
                break
            # Start with random small coefficients
            coeffs = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]
            current_score = compute_square_nonzeros_fast(exps, coeffs)
            
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                improved = False
                i = random.randint(0, 11)
                old_val = coeffs[i]
                # Try a range of values
                new_val = random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
                if new_val == old_val:
                    continue
                coeffs[i] = new_val
                new_score = compute_square_nonzeros_fast(exps, coeffs)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    coeffs[i] = old_val
                
                # Multi-change move
                if not improved and it % 10 == 0:
                    i1 = random.randint(0, 11)
                    i2 = random.randint(0, 11)
                    if i1 != i2:
                        old1, old2 = coeffs[i1], coeffs[i2]
                        coeffs[i1] = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
                        coeffs[i2] = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
                        new_score = compute_square_nonzeros_fast(exps, coeffs)
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                        else:
                            coeffs[i1] = old1
                            coeffs[i2] = old2
            
            if current_score < best_local_score:
                best_local_score = current_score
                best_local_coeffs = coeffs[:]
        
        if best_local_score < best_score:
            try_candidate(exps, best_local_coeffs)

    # --- Generate Candidate Exponent Sets ---
    
    candidate_exps = []
    
    # 1. Arithmetic Progressions
    for start in range(0, 10):
        for step in range(1, 5):
            exps = [start + i * step for i in range(12)]
            if max(exps) <= 40:
                candidate_exps.append(exps)
    
    # 2. Two Interleaved APs
    for s1 in range(0, 8):
        for d1 in range(1, 4):
            for s2 in range(3, 20):
                for d2 in range(1, 4):
                    exps1 = [s1 + i * d1 for i in range(6)]
                    exps2 = [s2 + i * d2 for i in range(6)]
                    exps = sorted(exps1 + exps2)
                    if len(set(exps)) == 12 and max(exps) <= 40:
                        candidate_exps.append(exps)
    
    # 3. Quadratic patterns: e_i = a*i^2 + b*i + c
    for a in range(0, 3):
        for b in range(0, 8):
            for c in range(0, 5):
                exps = [a * i * i + b * i + c for i in range(12)]
                if len(set(exps)) == 12 and max(exps) <= 40:
                    candidate_exps.append(exps)
    
    # 4. Geometric-like: e_i = floor(a * r^i) for small r
    for base in [2, 3]:
        for scale in range(1, 4):
            exps = [scale * (base ** i) for i in range(12)]
            if len(set(exps)) == 12 and max(exps) <= 40:
                candidate_exps.append(exps)
    
    # 5. Mixed step patterns: alternate between two step sizes
    for s in range(0, 5):
        for d1 in range(1, 4):
            for d2 in range(1, 4):
                exps = [s]
                cur = s
                for i in range(11):
                    step = d1 if i % 2 == 0 else d2
                    cur += step
                    exps.append(cur)
                if len(set(exps)) == 12 and max(exps) <= 40:
                    candidate_exps.append(exps)
    
    # 6. Clusters: group terms in pairs with small gaps, larger gap between clusters
    for cluster_gap in range(3, 8):
        for within_gap in range(1, 3):
            exps = []
            for cluster in range(6):
                base = cluster * cluster_gap
                exps.extend([base, base + within_gap])
            exps = sorted(exps)
            if len(set(exps)) == 12 and max(exps) <= 40:
                candidate_exps.append(exps)
    
    # 7. Random symmetric patterns
    import random
    random.seed(12345)
    for _ in range(50):
        if time.time() > deadline - 30:
            break
        # Generate 6 values, then mirror
        half = sorted(random.sample(range(0, 25), 6))
        total = random.randint(30, 40)
        mirrored = [total - h for h in reversed(half)]
        exps = sorted(half + mirrored)
        if len(set(exps)) == 12 and max(exps) <= 40:
            candidate_exps.append(exps)

    # Deduplicate
    seen = set()
    unique_exps = []
    for exps in candidate_exps:
        key = tuple(sorted(exps))
        if key not in seen:
            seen.add(key)
            unique_exps.append(sorted(exps))

    # Sort by span (smaller span = more collisions = potentially better)
    unique_exps.sort(key=lambda e: max(e) - min(e))
    
    # Run optimization on candidates
    for exps in unique_exps:
        if time.time() > deadline:
            break
            
        # Try signs first
        optimize_signs(exps)
        
        # If we found a good score, try coeff optimization
        if best_score <= 20 and time.time() < deadline - 15:
            optimize_coeffs(exps)
            
        # Early termination if we hit near theoretical lower bound
        if best_score <= 14:
            break

    return best_candidate

# EVOLVE-BLOCK-END
