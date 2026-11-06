import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    best_score = 21
    best_candidate = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # fallback

    def compute_square_nonzeros_fast(exps, coeffs):
        """Compute number of nonzero terms in P(x)^2."""
        max_deg = max(exps) * 2
        sq = [0] * (max_deg + 1)
        n = len(exps)
        for i in range(n):
            ei = exps[i]
            ci = coeffs[i]
            sq[2 * ei] += ci * ci
            for j in range(i + 1, n):
                ej = exps[j]
                cj = coeffs[j]
                sq[ei + ej] += 2 * ci * cj
        return sum(1 for v in sq if v != 0)

    def try_candidate(exps, coeffs):
        nonlocal best_score, best_candidate
        if len(exps) != 12 or len(coeffs) != 12:
            return
        if len(set(exps)) != 12:
            return
        if max(exps) > 40 or min(exps) < 0:
            return
        if any(abs(c) > 64 or c == 0 for c in coeffs):
            return
        score = compute_square_nonzeros_fast(exps, coeffs)
        if score < best_score:
            best_score = score
            max_deg = max(exps)
            dense = [0] * (max_deg + 1)
            for e, c in zip(exps, coeffs):
                dense[e] = c
            if dense[-1] != 0:
                best_candidate = dense

    def local_search_aggressive(exps, coeffs, max_iters=50):
        """Aggressive local search: try all small integer values for each coeff."""
        best = list(coeffs)
        best_score_local = compute_square_nonzeros_fast(exps, best)
        small_vals = [-4, -3, -2, -1, 1, 2, 3, 4]
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            for i in range(12):
                old = best[i]
                for new_val in small_vals:
                    if new_val == old:
                        continue
                    best[i] = new_val
                    s = compute_square_nonzeros_fast(exps, best)
                    if s < best_score_local:
                        best_score_local = s
                        improved = True
                    else:
                        best[i] = old
            if not improved:
                break
        return best, best_score_local

    def two_part_search():
        """
        Model P = A + B where A and B each have 6 terms.
        Structure: A has exponents {0, 1, 2, 3, 4, 5} with coefficients a0..a5
                  B has exponents {D, D+1, D+2, D+3, D+4, D+5} with coefficients b0..b5
        This ensures:
        - A^2 has exponents in [0, 10]
        - B^2 has exponents in [2D, 2D+10]
        - AB has exponents in [D, D+10]
        If D >= 11, these three sets are disjoint, so:
        nonzero(P^2) = nonzero(A^2) + nonzero(B^2) + nonzero(2AB)
        We can independently minimize A^2 and B^2, then check cross terms.
        """
        nonlocal best_score, best_candidate
        
        # Search over D (separation) and coefficient patterns
        for D in range(11, 30):
            if time.time() > deadline:
                break
            
            # Generate candidate coefficient patterns for 6-term polys
            # Focus on patterns that minimize self-square nonzeros
            patterns_6 = [
                [1, 1, 1, 1, 1, 1],
                [1, -1, 1, -1, 1, -1],
                [1, 1, -1, -1, 1, 1],
                [1, -1, -1, 1, 1, -1],
                [1, 1, 1, -1, -1, -1],
                [1, -1, 1, 1, -1, 1],
                [1, 1, -1, 1, -1, 1],
                [1, -1, 1, -1, -1, 1],
                [1, 1, 1, 1, -1, -1],
                [1, -1, -1, -1, 1, 1],
                [1, 1, -1, -1, -1, 1],
                [1, -1, 1, 1, 1, -1],
            ]
            
            # Also try with magnitudes
            for mag in [1, 2, 3]:
                patterns_6.append([mag] * 6)
                patterns_6.append([mag if i % 2 == 0 else -mag for i in range(6)])
            
            # Deduplicate patterns
            seen_patterns = set()
            unique_patterns = []
            for p in patterns_6:
                key = tuple(p)
                if key not in seen_patterns:
                    seen_patterns.add(key)
                    unique_patterns.append(p)
            
            # For each pair of patterns (A, B), compute the full polynomial
            for pa in unique_patterns:
                if time.time() > deadline:
                    break
                for pb in unique_patterns:
                    if time.time() > deadline:
                        break
                    
                    # A: exponents 0..5, coeffs pa
                    # B: exponents D..D+5, coeffs pb
                    exps = list(range(6)) + [D + i for i in range(6)]
                    coeffs = pa + pb
                    
                    # Validate
                    if len(set(exps)) != 12 or max(exps) > 40 or min(exps) < 0:
                        continue
                    if any(abs(c) > 64 or c == 0 for c in coeffs):
                        continue
                    
                    try_candidate(exps, coeffs)
            
            # Local search from best found so far for this D
            if time.time() < deadline - 3:
                # Find a good starting point
                best_start = None
                best_start_score = 100
                for pa in unique_patterns[:10]:
                    for pb in unique_patterns[:10]:
                        exps = list(range(6)) + [D + i for i in range(6)]
                        coeffs = pa + pb
                        if len(set(exps)) != 12 or max(exps) > 40:
                            continue
                        if any(abs(c) > 64 or c == 0 for c in coeffs):
                            continue
                        s = compute_square_nonzeros_fast(exps, coeffs)
                        if s < best_start_score:
                            best_start_score = s
                            best_start = coeffs
                if best_start:
                    exps = list(range(6)) + [D + i for i in range(6)]
                    improved, improved_score = local_search_aggressive(exps, best_start, max_iters=30)
                    try_candidate(exps, improved)
    
    def structured_search():
        nonlocal best_score, best_candidate
        
        # Focus on Arithmetic Progressions (APs) which have minimal distinct sum degrees
        ap_sets = []
        for start in range(0, 10):
            for step in range(1, 4):
                exps = [start + i * step for i in range(12)]
                if max(exps) <= 40:
                    ap_sets.append(tuple(exps))
        
        # Symmetric sets where e_i + e_{11-i} = D
        sym_sets = []
        for D in range(10, 41):
            for step in range(1, 3):
                for start in range(0, D//2):
                    left = [start + i * step for i in range(6)]
                    if left[5] >= D // 2:
                        continue
                    right = [D - x for x in reversed(left)]
                    exps = sorted(left + right)
                    if len(set(exps)) == 12 and max(exps) <= 40 and min(exps) >= 0:
                        sym_sets.append(tuple(exps))
        
        # Deduplicate
        seen = set()
        unique_exps = []
        for exps in ap_sets + sym_sets:
            if exps not in seen:
                seen.add(exps)
                unique_exps.append(list(exps))
        
        # Sort by span (smaller span = more collisions)
        unique_exps.sort(key=lambda e: max(e) - min(e))
        
        # Initial patterns
        patterns = [
            [1]*12,
            [1 if i%2==0 else -1 for i in range(12)],
            [1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1],
            [1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1],
        ]
        
        for exps in unique_exps:
            if time.time() > deadline:
                break
            
            # Try initial patterns
            for p in patterns:
                try_candidate(exps, p)
            
            # Run local search from best initial pattern
            if time.time() < deadline - 2:
                best_init = None
                best_init_score = 100
                for p in patterns:
                    s = compute_square_nonzeros_fast(exps, p)
                    if s < best_init_score:
                        best_init_score = s
                        best_init = p
                
                if best_init:
                    improved, improved_score = local_search_aggressive(exps, best_init, max_iters=50)
                    try_candidate(exps, improved)
                    
                    # If we found a very good score, do more search
                    if improved_score < 10:
                        if time.time() < deadline - 2:
                            improved2, improved_score2 = local_search_aggressive(exps, improved, max_iters=50)
                            try_candidate(exps, improved2)

    # Run two-part search first (structural decomposition)
    two_part_search()
    
    # Run structured search
    structured_search()
    
    return best_candidate

# EVOLVE-BLOCK-END
