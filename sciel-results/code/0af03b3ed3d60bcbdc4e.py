"""Search for a 12-term integer polynomial (deg<=40, coeffs in [-64,64])
whose exact integer square has minimal number of nonzero terms."""
import random
import time
from itertools import combinations

DEADLINE = 80.0  # seconds; evaluator kills at 90s, keep safety margin
N_TERMS = 12
MAX_DEG = 40
COEFF_RANGE = 64

def _square_nonzero(coeffs):
    """Given a coefficient list (ascending degree), return count of nonzero
    terms in the exact square."""
    n = len(coeffs)
    # convolution
    sq = [0] * (2 * n - 1)
    for i in range(n):
        ci = coeffs[i]
        if ci == 0:
            continue
        for j in range(n):
            cj = coeffs[j]
            if cj != 0:
                sq[i + j] += ci * cj
    return sum(1 for v in sq if v != 0)

def _valid(coeffs):
    """Check constraints: length<=41, nonzero final coeff, ints in [-64,64],
    exactly 12 nonzero terms."""
    if len(coeffs) > 41:
        return False
    if len(coeffs) == 0 or coeffs[-1] == 0:
        return False
    if len([c for c in coeffs if c != 0]) != N_TERMS:
        return False
    for c in coeffs:
        if not isinstance(c, int) or c < -COEFF_RANGE or c > COEFF_RANGE:
            return False
    return True

def _random_coeffs(rng):
    """Generate a random valid coefficient list."""
    # choose 12 distinct exponents within [0, 40]
    exponents = sorted(rng.sample(range(MAX_DEG + 1), N_TERMS))
    # ensure top exponent has nonzero coeff (it will by construction)
    coeffs = [0] * (exponents[-1] + 1)
    for e in exponents:
        coeffs[e] = rng.randint(1, COEFF_RANGE) * (1 if rng.random() < 0.5 else -1)
    return coeffs

def _ap_coeffs(step, rng):
    """Arithmetic progression exponents with random coefficients."""
    start = rng.randint(0, MAX_DEG - step * (N_TERMS - 1))
    exponents = [start + k * step for k in range(N_TERMS)]
    coeffs = [0] * (exponents[-1] + 1)
    for e in exponents:
        coeffs[e] = rng.randint(1, COEFF_RANGE) * (1 if rng.random() < 0.5 else -1)
    return coeffs

def _local_search(coeffs, rng, deadline, best_score, best_coeffs):
    """Local search: try swaps, coefficient changes, exponent shifts."""
    current_score = _square_nonzero(coeffs)
    current = list(coeffs)
    n = len(current)
    
    while time.time() < deadline:
        improved = False
        # generate a set of candidate moves
        moves = []
        
        # 1. coefficient sign flips
        nonzero_idx = [i for i in range(n) if current[i] != 0]
        for i in nonzero_idx:
            moves.append(('flip', i))
        
        # 2. coefficient magnitude changes
        for i in nonzero_idx:
            for delta in [-2, -1, 1, 2, -5, 5, -10, 10]:
                new_val = current[i] + delta
                if 1 <= abs(new_val) <= COEFF_RANGE:
                    moves.append(('coeff', i, new_val))
        
        # 3. swap two nonzero coefficients
        if len(nonzero_idx) >= 2:
            for a, b in combinations(nonzero_idx, 2):
                moves.append(('swap', a, b))
        
        # 4. move a nonzero term to a nearby zero slot
        zero_idx = [i for i in range(n) if current[i] == 0]
        for i in nonzero_idx:
            for j in zero_idx:
                if abs(i - j) <= 5:
                    moves.append(('move', i, j))
        
        # 5. insert a new term and remove another (keep 12 terms)
        for i in nonzero_idx:
            for j in range(n):
                if current[j] == 0 and abs(i - j) <= 3:
                    moves.append(('shift', i, j))
        
        rng.shuffle(moves)
        
        for move in moves:
            if time.time() >= deadline:
                break
            trial = list(current)
            if move[0] == 'flip':
                trial[move[1]] = -trial[move[1]]
            elif move[0] == 'coeff':
                trial[move[1]] = move[2]
            elif move[0] == 'swap':
                trial[move[1]], trial[move[2]] = trial[move[2]], trial[move[1]]
            elif move[0] == 'move':
                val = trial[move[1]]
                trial[move[1]] = 0
                trial[move[2]] = val
            elif move[0] == 'shift':
                val = trial[move[1]]
                trial[move[1]] = 0
                trial[move[2]] = val
            
            if not _valid(trial):
                continue
            
            score = _square_nonzero(trial)
            if score < current_score:
                current = trial
                current_score = score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_coeffs = list(current)
        
        if not improved:
            # try a random restart from current state with perturbation
            if time.time() < deadline - 5:
                # random perturbation: change 1-3 random coefficients
                for _ in range(rng.randint(1, 3)):
                    if nonzero_idx:
                        i = rng.choice(nonzero_idx)
                        trial = list(current)
                        trial[i] = rng.randint(1, COEFF_RANGE) * (1 if rng.random() < 0.5 else -1)
                        if _valid(trial):
                            s = _square_nonzero(trial)
                            if s <= current_score + 3:  # allow small degradation
                                current = trial
                                current_score = s
                                if current_score < best_score:
                                    best_score = current_score
                                    best_coeffs = list(current)
                # recompute nonzero_idx
                nonzero_idx = [i for i in range(len(current)) if current[i] != 0]
                if not nonzero_idx:
                    break
    
    return best_score, best_coeffs

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    rng = random.Random(42)
    
    best_score = float('inf')
    best_coeffs = None
    
    # Phase 1: try structured initial candidates (APs with various steps)
    for step in range(1, 7):
        if time.time() > deadline:
            break
        for _ in range(20):
            coeffs = _ap_coeffs(step, rng)
            if _valid(coeffs):
                score = _square_nonzero(coeffs)
                if score < best_score:
                    best_score = score
                    best_coeffs = list(coeffs)
    
    # Phase 2: random initial candidates
    if time.time() < deadline - 10:
        for _ in range(50):
            coeffs = _random_coeffs(rng)
            if _valid(coeffs):
                score = _square_nonzero(coeffs)
                if score < best_score:
                    best_score = score
                    best_coeffs = list(coeffs)
    
    # Phase 3: local search from best candidate
    if best_coeffs is not None and time.time() < deadline - 5:
        best_score, best_coeffs = _local_search(
            best_coeffs, rng, deadline, best_score, best_coeffs
        )
    
    # Phase 4: additional local search from other good candidates
    if time.time() < deadline - 5:
        # try a few more structured candidates and local search
        for step in [1, 2, 3]:
            if time.time() > deadline - 5:
                break
            coeffs = _ap_coeffs(step, rng)
            if _valid(coeffs):
                score = _square_nonzero(coeffs)
                if score <= best_score:
                    best_score, best_coeffs = _local_search(
                        coeffs, rng, deadline, best_score, best_coeffs
                    )
    
    # Fallback: return a valid candidate if we didn't find one
    if best_coeffs is None:
        best_coeffs = [1] * N_TERMS
    
    # Final validation
    if not _valid(best_coeffs):
        best_coeffs = [1] * N_TERMS
    
    return best_coeffs
