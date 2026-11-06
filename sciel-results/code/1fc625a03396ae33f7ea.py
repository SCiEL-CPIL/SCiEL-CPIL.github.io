"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50

    def get_nz(poly):
        """Get list of (index, coeff) for non-zero terms."""
        return [(i, poly[i]) for i in range(len(poly)) if poly[i] != 0]

    def square_nonzero_count_fast(poly):
        """Count nonzero terms in exact integer square using symmetry."""
        nz = get_nz(poly)
        n = len(nz)
        if n == 0:
            return 0
        max_deg = 2 * (len(poly) - 1)
        sq = [0] * (max_deg + 1)
        # Use symmetry: for i <= j, add ai*aj to sq[i+j], double if i != j
        for ii in range(n):
            i, ai = nz[ii]
            for jj in range(ii, n):
                j, aj = nz[jj]
                prod = ai * aj
                if ii == jj:
                    sq[i + j] += prod
                else:
                    sq[i + j] += 2 * prod
        count = 0
        for c in sq:
            if c != 0:
                count += 1
        return count

    def simulate_annealing(poly, max_iter=5000, t_start=1.0, t_end=0.01, cooling=0.9995):
        """Simulated annealing with diverse moves."""
        current = poly[:]
        current_score = square_nonzero_count_fast(current)
        best = current[:]
        best_score = current_score
        temp = t_start

        for it in range(max_iter):
            if time.time() > deadline:
                break
            nz = get_nz(current)
            if len(nz) < 12:
                break

            # Choose a move type
            move_type = random.random()
            old_nz = nz[:]
            old_val = None
            old_pos = None

            if move_type < 0.3:
                # Change coefficient of an existing term
                idx, old_val = random.choice(nz)
                new_val = random.randint(1, 64)
                if random.random() < 0.5:
                    new_val = -new_val
                if new_val == old_val:
                    continue
                current[idx] = new_val
                old_pos = idx
            elif move_type < 0.6:
                # Move a term to a new position
                idx, coeff = random.choice(nz)
                new_pos = random.randint(0, len(current) - 1)
                if current[new_pos] != 0:
                    continue
                current[idx] = 0
                current[new_pos] = coeff
                old_pos = idx
                old_val = 0
            elif move_type < 0.8:
                # Add a new term at a zero position
                zero_pos = [i for i in range(len(current)) if current[i] == 0]
                if not zero_pos:
                    continue
                new_pos = random.choice(zero_pos)
                new_val = random.randint(1, 64)
                if random.random() < 0.5:
                    new_val = -new_val
                current[new_pos] = new_val
                # Remove a random existing term to keep 12 nonzero
                if len(get_nz(current)) > 12:
                    rm_idx, _ = random.choice(nz)
                    current[rm_idx] = 0
                    old_pos = rm_idx
                    old_val = 0
                else:
                    old_pos = new_pos
                    old_val = 0
            else:
                # Swap coefficients of two terms
                if len(nz) < 2:
                    continue
                idx1, val1 = random.choice(nz)
                idx2, val2 = random.choice(nz)
                if idx1 == idx2:
                    continue
                current[idx1], current[idx2] = val2, val1
                old_pos = idx1
                old_val = val1

            new_score = square_nonzero_count_fast(current)

            # Accept or reject
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best = current[:]
            else:
                delta = new_score - current_score
                if delta > 0 and random.random() < math.exp(-delta / max(temp, 1e-10)):
                    current_score = new_score
                else:
                    # Revert
                    if move_type < 0.3:
                        current[old_pos] = old_val
                    elif move_type < 0.6:
                        current[old_pos] = old_val if old_val else 0
                        # Need to restore the moved term
                        # This is tricky; let's just revert by recomputing
                        pass
                    # For simplicity, on rejection just recompute from best if needed
                    # Actually, let's just not revert and accept the risk of drift
                    # Better: track state properly. Let's simplify: only accept improvements
                    # or small worsening with annealing, but revert on reject
                    current = best[:]  # Reset to best on reject to avoid drift
                    current_score = best_score

            temp *= cooling

        return best, best_score

    # 1. Try structured candidates
    best_poly = [0] * 41
    best_score = float('inf')

    # Candidate: sparse structure with potential cancellations
    # Try: 1 - x + x^2 - x^3 + ... with gaps
    candidates = [
        # All ones
        [1 if i < 12 else 0 for i in range(41)],
        # Alternating signs
        [(1 if i % 2 == 0 else -1) if i < 12 else 0 for i in range(41)],
        # Sparse: powers of 2
        [1 if i in [0, 1, 2, 4, 8, 16, 0, 1, 2, 4, 8, 16] else 0 for i in range(41)],
        # Try: 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0
        [1 if i % 3 == 0 else (-1 if i % 3 == 1 else 0) for i in range(12)] + [0]*29,
    ]

    for cand in candidates:
        if len(cand) < 41:
            cand = cand + [0]*(41-len(cand))
        nz_count = sum(1 for c in cand if c != 0)
        if nz_count == 12:
            s = square_nonzero_count_fast(cand)
            if s < best_score:
                best_score = s
                best_poly = cand[:]

    # 2. Simulated annealing from random starts
    restarts = 0
    while time.time() < deadline - 5:
        positions = random.sample(range(41), 12)
        poly = [0] * 41
        for pos in positions:
            c = random.randint(1, 64)
            if random.random() < 0.5:
                c = -c
            poly[pos] = c

        # Run simulated annealing
        poly, score = simulate_annealing(poly, max_iter=3000)

        if score < best_score:
            best_score = score
            best_poly = poly[:]
            if best_score <= 12:
                # Found something very good, do more refinement
                poly, score = simulate_annealing(best_poly, max_iter=5000)
                if score < best_score:
                    best_score = score
                    best_poly = poly[:]

        restarts += 1
        if restarts > 500:
            break

    # Final refinement with more iterations
    if time.time() < deadline - 2:
        best_poly, best_score = simulate_annealing(best_poly, max_iter=10000)

    # Trim trailing zeros
    last_nz = -1
    for i in range(len(best_poly) - 1, -1, -1):
        if best_poly[i] != 0:
            last_nz = i
            break

    if last_nz == -1:
        return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

    # Verify exactly 12 nonzero terms
    nz_count = sum(1 for c in best_poly[:last_nz+1] if c != 0)
    if nz_count != 12:
        # Fallback
        return [1] + [0]*10 + [1]

    return best_poly[:last_nz + 1]
# EVOLVE-BLOCK-END
