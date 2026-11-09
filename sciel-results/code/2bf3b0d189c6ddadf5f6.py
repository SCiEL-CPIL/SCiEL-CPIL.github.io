import numpy as np
import time
import random

def _sq_nonzero_count(coeffs):
    """Compute the number of nonzero terms in the square of a polynomial given by coefficient list."""
    sq = np.convolve(coeffs, coeffs)
    return int(np.count_nonzero(sq))

def _valid(coeffs):
    """Check if coeffs is a valid candidate: length<=41, nonzero final coeff, entries in [-64,64], exactly 12 nonzero terms."""
    if len(coeffs) > 41:
        return False
    if len(coeffs) == 0:
        return False
    if coeffs[-1] == 0:
        return False
    if any(abs(c) > 64 for c in coeffs):
        return False
    nz = sum(1 for c in coeffs if c != 0)
    if nz != 12:
        return False
    return True

def construct():
    start_time = time.time()
    # Safety margin: 90s limit, aim to finish by 85s
    deadline = start_time + 85.0
    
    best_score = float('inf')
    best_coeffs = None
    
    np.random.seed(42)
    random.seed(42)
    
    def count_nz_square(poly):
        """Fast score calculation using NumPy"""
        p_np = np.array(poly, dtype=np.int64)
        sq = np.convolve(p_np, p_np)
        return np.count_nonzero(sq)

    def random_valid_poly():
        """Generate a random valid polynomial."""
        p = [0] * 41
        # Choose 11 random indices, force the 12th to be index 40 to ensure validity
        # This guarantees coeffs[-1] != 0 and exactly 12 nonzeros if we don't collide
        indices = random.sample(range(0, 40), 11)
        indices.append(40)
        random.shuffle(indices) # Ensure 40 isn't always last in processing, though position in list doesn't matter for array
        
        for idx in indices:
            # Small coefficients are better for cancellation? 
            # Actually, large coefficients make cancellation harder (sums get larger).
            # But we need integers. Let's stick to small integers first.
            p[idx] = random.choice([-2, -1, 1, 2])
        
        # Check for accidental zeroing (shouldn't happen with choice)
        # Check exactly 12 nonzeros
        nz = sum(1 for c in p if c != 0)
        if nz != 12:
             # Fallback if collision occurred (extremely unlikely with sample)
             p[40] = 1
             return p
        return p

    # Phase 1: Generate high-quality structured seeds
    # These provide better starting points than pure random.
    seeds = []
    
    # 1. Palindromic patterns
    # P(x) = sum c_i x^i, P(x) = x^d P(1/x)
    # Try degrees 12 to 40
    for d in range(12, 41):
        if time.time() > deadline - 5:
            break
        half = d // 2
        # We need 12 terms.
        # If d is even, terms are pairs (i, d-i). 6 pairs = 12 terms.
        # If d is odd, terms are pairs. 6 pairs = 12 terms.
        # We need to choose 6 exponents from 0..d such that if i is chosen, d-i is effectively the partner.
        # Actually, simpler: choose 6 exponents from 0..d//2. 
        # If i == d-i (i=d/2), it's a single term. 
        # To get exactly 12 terms, we must avoid the center if d is even, or ensure we pick 6 distinct pairs.
        
        candidates = list(range(0, d + 1))
        # Filter out center if d is even to avoid single terms reducing count? 
        # If we pick i and d-i, and i != d-i, that's 2 terms.
        # If we pick i=d/2, that's 1 term.
        # We want 12 terms.
        # Case A: d even. Pick 6 indices from 0..d/2 - 1. Pairs (i, d-i). Total 12 terms.
        # Case B: d odd. Pick 6 indices from 0..d//2. Pairs (i, d-i). Total 12 terms.
        
        if d % 2 == 0:
            pool = list(range(0, d // 2))
            if len(pool) < 6: continue
            exponents = random.sample(pool, 6)
        else:
            pool = list(range(0, d // 2 + 1))
            if len(pool) < 6: continue
            exponents = random.sample(pool, 6)
        
        p = [0] * 41
        # Assign coefficients. Symmetric coefficients help?
        # If p[i] = p[d-i], then square is palindromic.
        # Let's try random small coeffs, but enforce symmetry for structure?
        # Or just random?
        for e in exponents:
            partner = d - e
            c = random.choice([-2, -1, 1, 2])
            p[e] = c
            # If partner is different and within bounds, assign same or opposite?
            # Same coeff: (a x^e + a x^f)^2 = a^2 x^{2e} + 2a^2 x^{e+f} + a^2 x^{2f}
            # Opposite coeff: (a x^e - a x^f)^2 = a^2 x^{2e} - 2a^2 x^{e+f} + a^2 x^{2f}
            # Both have 3 terms.
            # Let's try random assignment for partner to explore more.
            if partner != e and 0 <= partner < 41:
                # 50% chance symmetric, 50% random
                if random.random() < 0.5:
                    p[partner] = c
                else:
                    p[partner] = random.choice([-2, -1, 1, 2])
        
        # Ensure exactly 12 nonzeros and last coeff nonzero
        nz_indices = [i for i in range(41) if p[i] != 0]
        if len(nz_indices) != 12:
            continue
        if p[40] == 0:
            # Move a term to 40
            # Pick a random nonzero index k != 40
            movable = [i for i in nz_indices if i != 40]
            if movable:
                k = random.choice(movable)
                p[40] = p[k]
                p[k] = 0
            else:
                # All terms were at 40? Impossible with 12 terms.
                continue
        seeds.append(p)

    # 2. Arithmetic Progressions
    for _ in range(50):
        start = random.randint(0, 29)
        step = random.randint(1, 4)
        exponents = [start + i * step for i in range(12)]
        if exponents[-1] > 40:
            continue
        p = [0] * 41
        for e in exponents:
            p[e] = random.choice([-2, -1, 1, 2])
        
        # Fix validity
        nz_indices = [i for i in range(41) if p[i] != 0]
        if len(nz_indices) != 12: continue
        if p[40] == 0:
            movable = [i for i in nz_indices if i != 40]
            if movable:
                k = random.choice(movable)
                p[40] = p[k]
                p[k] = 0
            else: continue
        seeds.append(p)

    # 3. Random
    for _ in range(50):
        seeds.append(random_valid_poly())

    # Evaluate seeds
    for p in seeds:
        if not _valid(p): continue
        score = count_nz_square(p)
        if score < best_score:
            best_score = score
            best_coeffs = p[:]

    if best_coeffs is None:
        best_coeffs = random_valid_poly()
        best_score = count_nz_square(best_coeffs)

    # Phase 2: Simulated Annealing
    # This is the main search engine.
    current_poly = best_coeffs[:]
    current_score = best_score
    
    T = 1.0
    T_min = 0.001
    cooling = 0.9998
    
    no_improve = 0
    max_no_improve = 2000
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            # Restart
            current_poly = random_valid_poly()
            current_score = count_nz_square(current_poly)
            if current_score < best_score:
                best_score = current_score
                best_coeffs = current_poly[:]
            no_improve = 0
            T = 1.0
            continue

        # Generate neighbor
        neighbor = current_poly[:]
        move_type = random.random()
        
        if move_type < 0.5:
            # Change coefficient value
            nz_indices = [i for i in range(41) if neighbor[i] != 0]
            if not nz_indices: continue
            idx = random.choice(nz_indices)
            old_val = neighbor[idx]
            # Try small changes
            delta = random.choice([-2, -1, 1, 2])
            new_val = old_val + delta
            if new_val == 0:
                # If we zero it out, we break the 12-term constraint.
                # We must move it or keep it.
                # Let's just pick a new non-zero value.
                new_val = random.choice([-2, -1, 1, 2])
            elif abs(new_val) > 64:
                new_val = 64 if new_val > 0 else -64
            else:
                # Occasionally try a random small value
                if random.random() < 0.1:
                    new_val = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
                    if new_val == 0: new_val = 1
            
            neighbor[idx] = new_val
            
        elif move_type < 0.8:
            # Move a term to a new location
            nz_indices = [i for i in range(41) if neighbor[i] != 0]
            z_indices = [i for i in range(41) if neighbor[i] == 0]
            if not nz_indices or not z_indices: continue
            
            src = random.choice(nz_indices)
            # Don't move the term at 40, or if we do, we must ensure 40 stays non-zero.
            # Easiest: never move the term at 40.
            if src == 40:
                # Pick a different src
                movable = [i for i in nz_indices if i != 40]
                if not movable: continue
                src = random.choice(movable)
            
            dst = random.choice(z_indices)
            
            val = neighbor[src]
            neighbor[src] = 0
            neighbor[dst] = val
            
        else:
            # Swap two terms? Or swap values?
            # Let's try swapping values of two nonzero terms.
            nz_indices = [i for i in range(41) if neighbor[i] != 0]
            if len(nz_indices) < 2: continue
            i1, i2 = random.sample(nz_indices, 2)
            neighbor[i1], neighbor[i2] = neighbor[i2], neighbor[i1]

        # Validate neighbor
        # 1. Exactly 12 nonzeros
        nz_count = sum(1 for c in neighbor if c != 0)
        if nz_count != 12:
            no_improve += 1
            T *= cooling
            continue
        
        # 2. Last coeff nonzero
        if neighbor[40] == 0:
            no_improve += 1
            T *= cooling
            continue
            
        # 3. Coefficients in range
        if any(abs(c) > 64 for c in neighbor):
            no_improve += 1
            T *= cooling
            continue
            
        # Calculate score
        new_score = count_nz_square(neighbor)
        
        # Acceptance
        if new_score < current_score:
            current_poly = neighbor
            current_score = new_score
            no_improve = 0
            if current_score < best_score:
                best_score = current_score
                best_coeffs = current_poly[:]
                if best_score <= 2: # Theoretical minimum is likely 2 or 3
                    break
        else:
            delta = new_score - current_score
            if T > T_min:
                # exp(-delta/T)
                # delta is integer >= 1
                # T starts at 1.0
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_poly = neighbor
                    current_score = new_score
            no_improve += 1
            T *= cooling

    # Final check
    if not _valid(best_coeffs):
        # Fallback
        best_coeffs = [0] * 41
        for i in range(12):
            best_coeffs[i] = 1
        best_coeffs[40] = 1
        # Ensure 12 nonzeros: indices 0-10 are 1, index 40 is 1. Total 12.
        # Wait, range(12) is 0..11. If 40 is also 1, and 11 != 40, we have 13 nonzeros?
        # 0,1,2,3,4,5,6,7,8,9,10,11 -> 12 terms.
        # If I set best_coeffs[40]=1, and 40 is not in 0..11, I add a 13th term.
        # So I must remove one.
        best_coeffs = [0] * 41
        indices = list(range(11)) + [40] # 11 terms from 0-10, 1 term at 40. Total 12.
        for i in indices:
            best_coeffs[i] = 1
        best_score = _sq_nonzero_count(best_coeffs)

    return best_coeffs
