import time
import random

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

    def local_search_aggressive(exps, coeffs, max_iters=100):
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

    def structured_search():
        nonlocal best_score, best_candidate
        
        # Focus on Arithmetic Progressions (APs) which have minimal distinct sum degrees
        # e_i = start + i * step
        # Distinct sums: 2*start + k*step for k=0..22 (23 values)
        # We want to cancel as many of these 23 values as possible.
        
        # Generate all valid APs
        ap_sets = []
        for start in range(0, 10):
            for step in range(1, 4):
                exps = [start + i * step for i in range(12)]
                if max(exps) <= 40:
                    ap_sets.append(tuple(exps))
        
        # Also try "Symmetric" sets where e_i + e_{11-i} = D
        # This creates pairs that sum to D.
        sym_sets = []
        for D in range(10, 41):
            # Left half: 6 terms, Right half: 6 terms
            # e_0 < e_1 < ... < e_5 < D - e_5 < ... < D - e_0
            # Let left terms be l_0...l_5. Right terms are D-l_5...D-l_0.
            # We need l_5 < D/2.
            # Generate left terms as APs or random
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
            [1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1], # Symmetric antisymmetric?
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
                # Find best initial pattern
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

    # Run structured search
    structured_search()
    
    return best_candidate

# EVOLVE-BLOCK-END
