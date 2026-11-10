import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit

    def count_nonzero(poly):
        return np.count_nonzero(poly)

    def poly_square(coeffs):
        return np.convolve(coeffs, coeffs)

    def evaluate(coeffs):
        sq = poly_square(coeffs)
        return count_nonzero(sq)

    def validate(coeffs):
        if len(coeffs) > 41:
            return False
        if len(coeffs) == 0:
            return False
        if coeffs[-1] == 0:
            return False
        if not np.all((coeffs >= -64) & (coeffs <= 64)):
            return False
        if count_nonzero(coeffs) != 12:
            return False
        return True

    best_score = float('inf')
    best_poly = None

    # For a 12-term polynomial with support at positions pos[0..11],
    # the square has terms at degrees pos[i]+pos[j] for i<=j.
    # We want to choose positions and coefficients to maximize cancellations.
    
    # Strategy: Enumerate promising position sets (collision-rich), then
    # for each, use branch-and-bound over sign assignments to find optimal signs.
    
    # Key insight: For positions pos, the coefficient at degree d in the square
    # is sum_{i+j=d} a_i * a_j. If we fix |a_i| values, the signs determine
    # which sums cancel.
    
    # Generate candidate position sets with high collision counts
    def generate_position_sets():
        """Generate position sets likely to have many collisions in their sumset."""
        sets = []
        max_pos = 40  # degree <= 40
        
        # Arithmetic progressions
        for start in range(0, 10):
            for step in range(1, 5):
                positions = [start + i*step for i in range(12)]
                if max(positions) <= max_pos:
                    sets.append(tuple(sorted(positions)))
        
        # Interleaved APs (two interleaved sequences)
        for start1 in range(0, 5):
            for step1 in range(1, 4):
                for start2 in range(0, 5):
                    for step2 in range(1, 4):
                        positions = []
                        for i in range(6):
                            positions.append(start1 + i*step1)
                            positions.append(start2 + i*step2)
                        positions = sorted(set(positions))
                        if len(positions) == 12 and max(positions) <= max_pos:
                            sets.append(tuple(positions))
        
        # Clustered positions (dense in some regions)
        random.seed(42)
        for _ in range(200):
            # Generate 12 random positions with bias toward clustering
            center = random.randint(0, 30)
            spread = random.choice([2, 3, 4, 5])
            positions = []
            for _ in range(12):
                p = center + random.randint(-spread, spread)
                p = max(0, min(max_pos, p))
                positions.append(p)
            positions = sorted(set(positions))
            if len(positions) == 12:
                sets.append(tuple(positions))
        
        # Symmetric positions around a center
        for center in range(0, 25):
            for spread in range(3, 8):
                positions = []
                for i in range(6):
                    positions.append(center - i)
                    positions.append(center + i)
                positions = [p for p in positions if 0 <= p <= max_pos]
                if len(positions) >= 12:
                    positions = sorted(positions[:12])
                    if len(positions) == 12:
                        sets.append(tuple(positions))
        
        # Remove duplicates
        sets = list(set(sets))
        return sets

    def collision_count(positions):
        """Count the number of collisions in the sumset pos[i]+pos[j]."""
        sums = {}
        for i in range(len(positions)):
            for j in range(i, len(positions)):
                s = positions[i] + positions[j]
                sums[s] = sums.get(s, 0) + 1
        # Number of sums with multiplicity > 1
        collisions = sum(1 for v in sums.values() if v > 1)
        return collisions

    def optimal_signs_for_positions(positions, abs_vals, time_limit):
        """
        Given fixed positions and absolute values, find sign assignment
        that minimizes nonzero terms in the square.
        
        Uses branch-and-bound over sign assignments.
        """
        n = len(positions)
        if n == 0:
            return 0, []
        
        # Precompute which pairs contribute to each degree
        degree_to_pairs = {}
        for i in range(n):
            for j in range(i, n):
                d = positions[i] + positions[j]
                if d not in degree_to_pairs:
                    degree_to_pairs[d] = []
                degree_to_pairs[d].append((i, j))
        
        # For each degree, the coefficient is sum over pairs (i,j) of
        # sign_i * sign_j * abs_val_i * abs_val_j
        # We want this to be 0 for as many degrees as possible.
        
        # Branch and bound: assign signs one by one
        best_nonzero = float('inf')
        best_signs = [1] * n
        
        def bnb(idx, signs, current_nonzero):
            nonlocal best_nonzero, best_signs
            
            if time.time() > time_limit:
                return
            
            if idx == n:
                if current_nonzero < best_nonzero:
                    best_nonzero = current_nonzero
                    best_signs = signs[:]
                return
            
            # Try sign +1
            signs[idx] = 1
            new_nonzero = current_nonzero
            # Check which degrees are affected by fixing sign[idx]
            # A degree d is "decided" when all pairs contributing to it have their signs fixed
            # For now, just compute the contribution incrementally
            
            # Actually, let's compute the full square for this partial assignment
            # and count how many degrees are "forced" to be nonzero
            # This is expensive, so let's use a smarter approach
            
            # For each degree, if all pairs (i,j) with i<=j and pos[i]+pos[j]=d
            # have both i,j < idx+1 (i.e., signs assigned), we can check if sum is 0
            # Otherwise, the degree is "undecided"
            
            # Simpler: just compute the partial sum for each degree and track
            # whether it's already nonzero or could still be zero
            
            # Let's just do a quick evaluation
            temp_coeffs = np.zeros(max(positions) + 1, dtype=np.int64)
            for i in range(n):
                if i <= idx:
                    temp_coeffs[positions[i]] = signs[i] * abs_vals[i]
                else:
                    temp_coeffs[positions[i]] = 0  # unassigned, treat as 0 for now
            
            # This approach doesn't work well for partial assignments.
            # Let's use a different strategy: enumerate all 2^n sign assignments
            # but prune using bounds.
            
            # For now, let's just do exhaustive search with pruning
            # when n is small enough
            if n <= 16:
                # Continue recursion
                bnb(idx + 1, signs, current_nonzero)
                signs[idx] = -1
                bnb(idx + 1, signs, current_nonzero)
            else:
                # Too many variables, use heuristic
                signs[idx] = 1 if random.random() < 0.5 else -1
                bnb(idx + 1, signs, current_nonzero)
        
        # This naive approach is too slow. Let's use a better method:
        # For each position set, try all 2^n sign combinations if n is small,
        # otherwise use local search.
        
        if n <= 12:
            # Try all sign combinations
            from itertools import product
            for signs_tuple in product([1, -1], repeat=n):
                if time.time() > time_limit:
                    break
                coeffs = np.zeros(max(positions) + 1, dtype=np.int64)
                for i in range(n):
                    coeffs[positions[i]] = signs_tuple[i] * abs_vals[i]
                sq = np.convolve(coeffs, coeffs)
                nz = np.count_nonzero(sq)
                if nz < best_nonzero:
                    best_nonzero = nz
                    best_signs = list(signs_tuple)
        else:
            # Use local search
            signs = [1] * n
            for _ in range(1000):
                if time.time() > time_limit:
                    break
                idx = random.randint(0, n-1)
                signs[idx] = -signs[idx]
                coeffs = np.zeros(max(positions) + 1, dtype=np.int64)
                for i in range(n):
                    coeffs[positions[i]] = signs[i] * abs_vals[i]
                sq = np.convolve(coeffs, coeffs)
                nz = np.count_nonzero(sq)
                if nz < best_nonzero:
                    best_nonzero = nz
                    best_signs = signs[:]
        
        return best_nonzero, best_signs

    # Phase 1: Try structured position sets with unit coefficients
    position_sets = generate_position_sets()
    
    # Sort by collision count (more collisions = more opportunities for cancellation)
    position_sets.sort(key=lambda p: collision_count(p), reverse=True)
    
    random.seed(42)
    
    for positions in position_sets[:500]:  # Limit to top 500
        if time.time() > deadline - 10:
            break
        
        # Try with unit absolute values
        abs_vals = [1] * 12
        nz, signs = optimal_signs_for_positions(positions, abs_vals, deadline - 5)
        
        if nz < best_score:
            coeffs = np.zeros(max(positions) + 1, dtype=np.int64)
            for i in range(12):
                coeffs[positions[i]] = signs[i] * abs_vals[i]
            # Trim trailing zeros
            while len(coeffs) > 0 and coeffs[-1] == 0:
                coeffs = coeffs[:-1]
            if len(coeffs) > 0 and coeffs[-1] != 0 and np.count_nonzero(coeffs) == 12:
                best_score = nz
                best_poly = coeffs.copy()
        
        if best_score <= 14:
            break

    # Phase 2: Local search refinement on best solution
    if best_poly is not None and time.time() < deadline - 5:
        curr_c = best_poly.copy()
        curr_score = best_score
        
        no_improvement = 0
        max_no_improvement = 5000
        temperature = 1.0
        
        while time.time() < deadline - 5 and no_improvement < max_no_improvement:
            # Pick a random index to modify
            if len(curr_c) > 0:
                idx = random.randint(0, len(curr_c) - 1)
                new_val = random.randint(-8, 8)
                if new_val == 0:
                    new_val = 1 if random.random() < 0.5 else -1
                
                if curr_c[idx] == new_val:
                    continue
                
                old_val = curr_c[idx]
                curr_c[idx] = new_val
                
                # Check constraints
                valid = True
                if len(curr_c) > 41:
                    valid = False
                if len(curr_c) == 0:
                    valid = False
                if curr_c[-1] == 0:
                    valid = False
                if not np.all((curr_c >= -64) & (curr_c <= 64)):
                    valid = False
                if np.count_nonzero(curr_c) != 12:
                    valid = False
                
                if valid:
                    new_score = evaluate(curr_c)
                    if new_score < curr_score:
                        curr_score = new_score
                        no_improvement = 0
                        if curr_score < best_score:
                            best_score = curr_score
                            best_poly = curr_c.copy()
                            if best_score <= 14:
                                break
                    else:
                        delta = new_score - curr_score
                        if random.random() < np.exp(-delta / max(temperature, 0.1)):
                            curr_score = new_score
                            no_improvement = 0
                            if curr_score < best_score:
                                best_score = curr_score
                                best_poly = curr_c.copy()
                        else:
                            curr_c[idx] = old_val
                            no_improvement += 1
                else:
                    curr_c[idx] = old_val
                    no_improvement += 1
            
            temperature *= 0.999

    # Phase 3: Try some additional structured candidates
    if time.time() < deadline - 5:
        additional_candidates = [
            [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1],
            [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
            [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
            [1, 2, -1, -2, 1, 2, -1, -2, 1, 2, -1, -2],
            [1, -2, 1, 2, -1, -2, 1, -2, 1, 2, -1, 2],
            [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1],
            [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
            [2, 1, -1, -2, 1, -1, 2, -1, 1, -2, 1, -1],
            [1, 2, 1, -1, -2, -1, 1, 2, 1, -1, -2, -1],
        ]
        
        for c in additional_candidates:
            if time.time() > deadline - 5:
                break
            c_arr = np.array(c, dtype=np.int64)
            score = evaluate(c_arr)
            if score < best_score:
                best_score = score
                best_poly = c_arr.copy()

    # Ensure final coefficient is nonzero
    if best_poly is None:
        best_poly = np.array([1]*12, dtype=np.int64)
    
    while len(best_poly) > 0 and best_poly[-1] == 0:
        best_poly = best_poly[:-1]

    return best_poly.tolist()
