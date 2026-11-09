"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 85.0  # Safety margin
    
    # Constants
    MAX_DEG = 40
    SQ_SIZE = 2 * MAX_DEG + 1  # 81
    N_TERMS = 12
    
    def square_nonzero_count(terms):
        """
        terms: list of (exponent, coefficient)
        Returns count of nonzero terms in the square.
        Uses a fixed-size list for speed.
        """
        sq = [0] * SQ_SIZE
        n = len(terms)
        # Pre-sort by exponent? Not strictly necessary for summation, but helps caching?
        # Just O(n^2) loops.
        for i in range(n):
            ei, ci = terms[i]
            for j in range(i, n):
                ej, cj = terms[j]
                idx = ei + ej
                if i == j:
                    sq[idx] += ci * ci
                else:
                    sq[idx] += 2 * ci * cj
        return sum(1 for v in sq if v != 0)

    def is_valid(terms):
        if len(terms) != N_TERMS:
            return False
        exps = [e for e, c in terms]
        if len(set(exps)) != N_TERMS:
            return False
        if any(e < 0 or e > MAX_DEG for e in exps):
            return False
        if any(c == 0 or c < -64 or c > 64 for e, c in terms):
            return False
        return True

    def terms_to_poly(terms):
        poly = [0] * (MAX_DEG + 1)
        for e, c in terms:
            poly[e] = c
        # Trim trailing zeros but keep at least length 1
        # Actually, the interface expects length <= 41.
        # We can just return the list of length max_deg+1.
        max_deg = max(e for e, c in terms)
        return poly[:max_deg+1]

    # Best found
    best_count = 23
    best_terms = [(i, 1) for i in range(12)]
    
    # Phase 1: Structured Search
    # Try patterns like 1, x, x^2, -x^3, etc.
    # Specifically, try "block" structures.
    rng = random.Random(42)
    
    # Try to find a good initial guess using a small heuristic search
    # We will run a quick random search to initialize
    for _ in range(100):
        exps = sorted(rng.sample(range(MAX_DEG + 1), N_TERMS))
        coeffs = [rng.choice([-1, 1]) for _ in range(N_TERMS)]
        # Force first coeff to 1
        coeffs[0] = 1
        terms = list(zip(exps, coeffs))
        cnt = square_nonzero_count(terms)
        if cnt < best_count:
            best_count = cnt
            best_terms = terms[:]

    # Phase 2: Simulated Annealing
    # We will run multiple SA instances in parallel (sequentially) with different temps
    
    def run_sa(start_terms, T_start, steps, seed):
        rng_sa = random.Random(seed)
        current = start_terms[:]
        current_count = square_nonzero_count(current)
        T = T_start
        cooling = 0.995
        best_local = current_count
        best_local_terms = current[:]
        
        for step in range(steps):
            if time.time() > deadline:
                break
            
            # Generate a move
            new_terms = current[:]
            move_type = rng_sa.random()
            
            if move_type < 0.4:
                # Change one exponent
                idx = rng_sa.randint(0, N_TERMS - 1)
                old_e = new_terms[idx][0]
                new_e = rng_sa.randint(0, MAX_DEG)
                if new_e == old_e:
                    continue
                # Check collision
                exps = [e for e, c in new_terms]
                exps.pop(idx)
                if new_e in exps:
                    continue
                new_terms[idx] = (new_e, new_terms[idx][1])
                
            elif move_type < 0.7:
                # Change one coefficient
                idx = rng_sa.randint(0, N_TERMS - 1)
                old_c = new_terms[idx][1]
                # Try small integers
                new_c = rng_sa.choice([-3, -2, -1, 1, 2, 3])
                if new_c == old_c or new_c == 0:
                    continue
                new_terms[idx] = (new_terms[idx][0], new_c)
                
            elif move_type < 0.85:
                # Swap two exponents
                i, j = rng_sa.sample(range(N_TERMS), 2)
                ei, ci = new_terms[i]
                ej, cj = new_terms[j]
                new_terms[i] = (ej, ci)
                new_terms[j] = (ei, cj)
                
            else:
                # Change two exponents (kick)
                i, j = rng_sa.sample(range(N_TERMS), 2)
                old_ei = new_terms[i][0]
                old_ej = new_terms[j][0]
                new_ei = rng_sa.randint(0, MAX_DEG)
                new_ej = rng_sa.randint(0, MAX_DEG)
                exps = [e for e, c in new_terms]
                exps.pop(j)
                exps.pop(i)
                if new_ei in exps or new_ej in exps or new_ei == new_ej:
                    continue
                new_terms[i] = (new_ei, new_terms[i][1])
                new_terms[j] = (new_ej, new_terms[j][1])
            
            if not is_valid(new_terms):
                continue
                
            new_count = square_nonzero_count(new_terms)
            delta = new_count - current_count
            
            if delta < 0 or rng_sa.random() < np.exp(-delta / max(T, 0.01)):
                current = new_terms
                current_count = new_count
                if current_count < best_local:
                    best_local = current_count
                    best_local_terms = current[:]
            
            T *= cooling
            if T < 0.05:
                T = 0.05
                
        return best_local, best_local_terms

    # Run multiple SA instances
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Start from best known or random
        if i % 2 == 0:
            start = best_terms[:]
        else:
            exps = sorted(random.Random(100+i).sample(range(MAX_DEG + 1), N_TERMS))
            coeffs = [1 if random.Random(100+i).random() < 0.5 else -1 for _ in range(N_TERMS)]
            coeffs[0] = 1
            start = list(zip(exps, coeffs))
            
        local_best, local_terms = run_sa(start, 2.0, 2000, 1000 + i)
        
        if local_best < best_count:
            best_count = local_best
            best_terms = local_terms[:]

    # Phase 3: Intensive Local Search on the best candidate
    if time.time() < deadline:
        current = best_terms[:]
        current_count = best_count
        rng_local = random.Random(999)
        
        no_improve = 0
        max_no_improve = 100
        
        while no_improve < max_no_improve:
            if time.time() > deadline:
                break
                
            best_neighbor = None
            best_neighbor_count = current_count
            
            # Try all single moves
            # 1. Exponent changes
            for idx in range(N_TERMS):
                old_e = current[idx][0]
                for new_e in range(MAX_DEG + 1):
                    if new_e == old_e:
                        continue
                    exps = [e for e, c in current]
                    exps.pop(idx)
                    if new_e in exps:
                        continue
                    trial = current[:]
                    trial[idx] = (new_e, current[idx][1])
                    cnt = square_nonzero_count(trial)
                    if cnt < best_neighbor_count:
                        best_neighbor_count = cnt
                        best_neighbor = trial[:]
                        
            # 2. Coefficient changes
            for idx in range(N_TERMS):
                old_c = current[idx][1]
                for new_c in [-4, -3, -2, -1, 1, 2, 3, 4]:
                    if new_c == old_c or new_c == 0:
                        continue
                    trial = current[:]
                    trial[idx] = (current[idx][0], new_c)
                    cnt = square_nonzero_count(trial)
                    if cnt < best_neighbor_count:
                        best_neighbor_count = cnt
                        best_neighbor = trial[:]
                        
            if best_neighbor is not None:
                current = best_neighbor
                current_count = best_neighbor_count
                no_improve = 0
                if current_count < best_count:
                    best_count = current_count
                    best_terms = current[:]
            else:
                # Perturb to escape
                for _ in range(5):
                    if time.time() > deadline:
                        break
                    # Random kick
                    idx = rng_local.randint(0, N_TERMS - 1)
                    old_e = current[idx][0]
                    new_e = rng_local.randint(0, MAX_DEG)
                    if new_e == old_e:
                        continue
                    exps = [e for e, c in current]
                    exps.pop(idx)
                    if new_e in exps:
                        continue
                    trial = current[:]
                    trial[idx] = (new_e, current[idx][1])
                    cnt = square_nonzero_count(trial)
                    if cnt < current_count:
                        current = trial
                        current_count = cnt
                        no_improve = 0
                        if current_count < best_count:
                            best_count = current_count
                            best_terms = current[:]
                        break
                else:
                    no_improve += 1

    # Convert to polynomial
    best_poly = terms_to_poly(best_terms)
    
    # Final checks
    if len(best_poly) > 41:
        best_poly = best_poly[:41]
        
    # Ensure exactly 12 nonzero terms
    nonzero = sum(1 for c in best_poly if c != 0)
    if nonzero != 12:
        # Fallback to seed
        return [1] * 12
        
    # Ensure integers in range
    for i in range(len(best_poly)):
        best_poly[i] = int(best_poly[i])
        if best_poly[i] < -64:
            best_poly[i] = -64
        elif best_poly[i] > 64:
            best_poly[i] = 64
        # Ensure no zero coefficients in the list if we want to be strict? 
        # The list can have zeros. The constraint is "exactly 12 nonzero terms" in the list.
        
    # Ensure final coefficient is nonzero
    if best_poly[-1] == 0:
        # Find last nonzero
        last_nonzero = len(best_poly) - 1
        while last_nonzero >= 0 and best_poly[last_nonzero] == 0:
            last_nonzero -= 1
        if last_nonzero < 0:
            return [1] * 12
        best_poly = best_poly[:last_nonzero + 1]
        
    # Re-verify
    if len(best_poly) > 41 or len(best_poly) < 1 or best_poly[-1] == 0:
        return [1] * 12
    if sum(1 for c in best_poly if c != 0) != 12:
        return [1] * 12
    if any(c < -64 or c > 64 for c in best_poly):
        return [1] * 12
        
    return best_poly

# EVOLVE-BLOCK-END
